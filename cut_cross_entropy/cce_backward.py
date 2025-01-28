# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import torch
import triton
import triton.language as tl

from cut_cross_entropy.tl_autotune import cce_backward_autotune
from cut_cross_entropy.tl_utils import (
    b_bin_fn,
    tl_and_reduce_fn,
    tl_lock_add,
    tl_lock_kahan_sum,
    tl_softcapping,
    tl_softcapping_grad,
)


@triton.jit
def _mm_backward(
    do,
    da_ptrs,
    dac_ptrs,
    partial_mask_a,
    da_lock_ptr,
    n_locks,
    b_ptrs,
    partial_mask_b,
    stride_ad,
    stride_bd,
    D,
    BLOCK_D: tl.constexpr,
    EVEN_D: tl.constexpr,
    USE_KAHAN: tl.constexpr,
):
    d_inds = tl.arange(0, BLOCK_D)[None, :]

    b_ptrs = b_ptrs + d_inds * stride_bd
    da_ptrs = da_ptrs + d_inds * stride_ad
    if USE_KAHAN:
        dac_ptrs = dac_ptrs + d_inds * stride_ad

    for d in range(0, tl.cdiv(D, BLOCK_D)):
        if EVEN_D:
            mask = partial_mask_b
        else:
            mask = partial_mask_b & (d_inds < (D - d * BLOCK_D))

        b = tl.load(b_ptrs, mask=mask, other=0.0)

        da_i = tl.dot(do, b).to(da_ptrs.dtype.element_ty)

        if EVEN_D:
            mask = partial_mask_a
        else:
            mask = partial_mask_a & (d_inds < (D - d * BLOCK_D))

        lock_offset = d // tl.cdiv(D, BLOCK_D * n_locks)
        this_da_lock_ptr = da_lock_ptr + lock_offset

        if USE_KAHAN:
            tl_lock_kahan_sum(da_ptrs, dac_ptrs, da_i, mask, this_da_lock_ptr)
        else:
            tl_lock_add(da_ptrs, da_i, mask, this_da_lock_ptr)

        b_ptrs += BLOCK_D * stride_bd
        da_ptrs += BLOCK_D * stride_ad
        if USE_KAHAN:
            dac_ptrs += BLOCK_D * stride_ad


@triton.jit
def _block_is_filtered(check_val: tl.tensor, filter_eps: tl.tensor) -> tl.tensor:
    return tl.reduce(check_val < filter_eps, None, tl_and_reduce_fn)


def _cce_backward_kernel(
    E,
    C,
    Bias,
    LSE,
    dOut,
    grad_scale,
    Valids,
    VocabOrdering,
    softcap,
    Targets,
    dE,
    dEC,
    dELocks,
    dC,
    dCC,
    dCLocks,
    dBias,
    B,
    D,
    V,
    BMax,
    n_de_locks_0,
    n_de_locks_1,
    n_dc_locks_0,
    n_dc_locks_1,
    stride_eb,
    stride_ed,
    stride_cv,
    stride_cd,
    stride_biasv,
    stride_vb,
    filter_eps,
    shift,
    B_BIN,
    BLOCK_B: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MM_BACK_BLOCK_D: tl.constexpr,
    GROUP_B: tl.constexpr,
    EVEN_D: tl.constexpr,
    MM_BACK_EVEN_D: tl.constexpr,
    ITEM_DO: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_VALIDS: tl.constexpr,
    HAS_VOCAB_ORDERING: tl.constexpr,
    FILTER_GRAD: tl.constexpr,
    HAS_TARGETS: tl.constexpr,
    HAS_SOFTCAP: tl.constexpr,
    HAS_SHIFT: tl.constexpr,
    USE_KAHAN: tl.constexpr,
    COMPUTE_DC: tl.constexpr,
    COMPUTE_DE: tl.constexpr,
    COMPUTE_DBIAS: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_b_chunks = tl.cdiv(B, BLOCK_B)
    num_v_chunks = tl.cdiv(V, BLOCK_V)
    num_v_in_group = GROUP_B * num_v_chunks
    group_id = pid // num_v_in_group
    first_pid_b = group_id * GROUP_B
    group_size_b = min(num_b_chunks - first_pid_b, GROUP_B)
    pid_b = first_pid_b + ((pid % num_v_in_group) % group_size_b)
    pid_v = (pid % num_v_in_group) // group_size_b

    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    if HAS_VALIDS:
        offs_b = tl.load(Valids + stride_vb * offs_b, mask=offs_b < B, other=BMax)

    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    if HAS_VOCAB_ORDERING:
        offs_v = tl.load(VocabOrdering + offs_v, mask=offs_v < V, other=V)

    offs_d = tl.arange(0, BLOCK_D)
    e_ptrs = E + (offs_b[:, None] * stride_eb + offs_d[None, :] * stride_ed)
    c_ptrs = C + (offs_v[None, :] * stride_cv + offs_d[:, None] * stride_cd)

    accum = tl.zeros((BLOCK_B, BLOCK_V), dtype=tl.float32)
    for d in range(0, tl.cdiv(D, BLOCK_D)):
        e_mask = offs_b[:, None] < BMax
        if not EVEN_D:
            e_mask = e_mask & (offs_d[None, :] < (D - d * BLOCK_D))

        e = tl.load(e_ptrs, mask=e_mask, other=0.0)

        c_mask = offs_v[None, :] < V
        if not EVEN_D:
            c_mask = c_mask & (offs_d[:, None] < (D - d * BLOCK_D))

        c = tl.load(c_ptrs, mask=c_mask, other=0.0)

        accum = tl.dot(e, c, accum)

        e_ptrs += BLOCK_D * stride_ed
        c_ptrs += BLOCK_D * stride_cd

    tl.debug_barrier()

    if HAS_BIAS:
        bias = tl.load(Bias + offs_v * stride_biasv, mask=offs_v < V, other=0.0)
        bias = bias.to(dtype=accum.dtype)
        accum += bias[None, :]

    if HAS_SOFTCAP:
        accum = tl_softcapping(accum, softcap)

    if HAS_VALIDS:
        direct_offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
        lse = tl.load(LSE + direct_offs_b, mask=direct_offs_b < B, other=float("inf"))
    else:
        lse = tl.load(LSE + offs_b, mask=offs_b < B, other=float("inf"))

    d_accum = tl.exp(accum - lse[:, None])
    d_accum = tl.where(offs_v[None, :] < V, d_accum, 0.0)

    if HAS_TARGETS:
        if HAS_SHIFT:
            target_offs_b = offs_b + shift
        else:
            target_offs_b = offs_b

        targets = tl.load(Targets + target_offs_b, mask=target_offs_b < BMax, other=V + 1)
        is_target = targets[:, None] == offs_v[None, :]
        d_accum += tl.where(is_target, -1.0, 0.0)
    else:
        is_target = None

    if FILTER_GRAD:
        if _block_is_filtered(tl.abs(d_accum), filter_eps):
            return

    if HAS_SOFTCAP:
        d_accum = tl_softcapping_grad(d_accum, accum, softcap)

    if ITEM_DO:
        d_out = tl.load(dOut)
    else:
        if HAS_SHIFT:
            d_out_offs_b = offs_b + shift
        else:
            d_out_offs_b = offs_b

        d_out = tl.load(dOut + d_out_offs_b, mask=d_out_offs_b < BMax, other=0.0)[:, None]

    d_out = grad_scale * d_out

    d_accum = d_accum * d_out

    if COMPUTE_DBIAS:
        tl.atomic_add(dBias + offs_v * stride_biasv, tl.sum(d_accum, 0), mask=offs_v < V)

    d_accum = d_accum.to(e_ptrs.dtype.element_ty)

    if COMPUTE_DE:
        lock_offset = (pid_b // tl.cdiv(B, BLOCK_B * n_de_locks_0)) * n_de_locks_1

        _mm_backward(
            d_accum,
            dE + (offs_b[:, None] * stride_eb),
            dEC + (offs_b[:, None] * stride_eb) if USE_KAHAN else None,
            offs_b[:, None] < BMax,
            dELocks + lock_offset,
            n_de_locks_1,
            C + offs_v[:, None] * stride_cv,
            offs_v[:, None] < V,
            stride_ed,
            stride_cd,
            D,
            MM_BACK_BLOCK_D,
            MM_BACK_EVEN_D,
            USE_KAHAN,
        )

    if COMPUTE_DC:
        lock_offset = (pid_v // tl.cdiv(V, BLOCK_V * n_dc_locks_0)) * n_dc_locks_1

        _mm_backward(
            tl.trans(d_accum),
            dC + (offs_v[:, None] * stride_cv),
            dCC + (offs_v[:, None] * stride_cv) if USE_KAHAN else None,
            offs_v[:, None] < V,
            dCLocks + lock_offset,
            n_dc_locks_1,
            E + (offs_b[:, None] * stride_eb),
            offs_b[:, None] < BMax,
            stride_cd,
            stride_ed,
            D,
            MM_BACK_BLOCK_D,
            MM_BACK_EVEN_D,
            USE_KAHAN,
        )


def _cce_back_block_d(args) -> int:
    block_d = args["BLOCK_D"]
    return 2 * block_d


_cce_backward_kernel = triton.jit(_cce_backward_kernel)
_cce_backward_kernel = triton.heuristics(  # type: ignore
    {
        "EVEN_D": lambda args: (args["D"] % args["BLOCK_D"]) == 0,
        "MM_BACK_BLOCK_D": lambda args: _cce_back_block_d(args),
        "MM_BACK_EVEN_D": lambda args: (args["D"] % _cce_back_block_d(args)) == 0,
        "HAS_VALIDS": lambda args: args["Valids"] is not None,
        "HAS_BIAS": lambda args: args["Bias"] is not None,
        "HAS_VOCAB_ORDERING": lambda args: args["VocabOrdering"] is not None,
        "FILTER_GRAD": lambda args: args["filter_eps"] is not None,
        "HAS_TARGETS": lambda args: args["Targets"] is not None,
        "HAS_SOFTCAP": lambda args: args["softcap"] is not None,
        "HAS_SHIFT": lambda args: args["shift"] != 0,
        "ITEM_DO": lambda args: args["dOut"].numel() == 1,
        "GROUP_B": lambda args: 8,
        "COMPUTE_DC": lambda args: args["dC"] is not None,
        "COMPUTE_DE": lambda args: args["dE"] is not None,
        "COMPUTE_DBIAS": lambda args: args["dBias"] is not None,
    }
)(_cce_backward_kernel)
_cce_backward_kernel = cce_backward_autotune()(_cce_backward_kernel)  # type: ignore


def cce_backward_kernel(
    do: torch.Tensor,
    e: torch.Tensor,
    c: torch.Tensor,
    bias: torch.Tensor | None,
    lse: torch.Tensor,
    valids: torch.Tensor | None,
    softcap: float | None,
    filter_eps: float | None,
    targets: torch.Tensor | None = None,
    shift: int = 0,
    vocab_ordering: torch.Tensor | None = None,
    grad_scale: float = 1.0,
    use_kahan: bool = False,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    assert do.numel() in (e.size(0), 1)
    assert c.size(1) == e.size(1)
    assert lse.size(0) == e.size(0) or (valids is not None and lse.size(0) == valids.size(0))
    assert e.dtype in (
        torch.float16,
        torch.bfloat16,
    ), "Backwards requires embeddings to be bf16 or fp16"
    assert c.dtype in (
        torch.float16,
        torch.bfloat16,
    ), "Backwards requires classifier to be bf16 or fp16"

    do = do.contiguous()
    lse = lse.contiguous()

    de = torch.zeros_like(e) if e.requires_grad else None
    dc = torch.zeros_like(c) if c.requires_grad else None

    if bias is not None:
        dbias = torch.zeros_like(bias, dtype=torch.float32) if bias.requires_grad else None
    else:
        dbias = None

    if de is not None:
        assert de.stride() == e.stride()

    if dc is not None:
        assert dc.stride() == c.stride()

    if dbias is not None:
        assert bias is not None
        assert dbias.stride() == bias.stride()

    if use_kahan:
        dec = torch.zeros_like(e) if de is not None else None
        dcc = torch.zeros_like(c) if dc is not None else None
    else:
        dec = None
        dcc = None

    if dec is not None:
        assert dec.stride() == e.stride()

    if dcc is not None:
        assert dcc.stride() == e.stride()

    if valids is not None:
        assert valids.ndim == 1
        B = valids.size(0)
    else:
        B = e.size(0)

    if do.numel() > 1:
        do = do.contiguous()
        lse = lse.contiguous()
        assert do.stride(0) == lse.stride(0), f"{do.stride()=}, {lse.stride()=}"

    def grid(META):
        return (triton.cdiv(B, META["BLOCK_B"]) * triton.cdiv(c.size(0), META["BLOCK_V"]),)

    if vocab_ordering is not None:
        assert vocab_ordering.ndim == 1
        assert vocab_ordering.numel() == c.size(0)
        assert vocab_ordering.stride(0) == 1

    nd_locks = triton.cdiv(c.size(1), 64)
    if de is not None:
        de_locks = e.new_zeros((triton.cdiv(B, 128), nd_locks), dtype=torch.int32)
        de_lock_sizes = de_locks.size()
    else:
        de_locks = None
        de_lock_sizes = (None, None)

    if dc is not None:
        dc_locks = c.new_zeros((triton.cdiv(c.size(0), 128), nd_locks), dtype=torch.int32)
        dc_lock_sizes = dc_locks.size()
    else:
        dc_locks = None
        dc_lock_sizes = (None, None)

    _cce_backward_kernel[grid](
        e,
        c,
        bias,
        lse,
        do,
        grad_scale,
        valids,
        vocab_ordering,
        softcap,
        targets,
        de,
        dec,
        de_locks,
        dc,
        dcc,
        dc_locks,
        dbias,
        B,
        e.size(1),
        c.size(0),
        e.size(0),
        *de_lock_sizes,
        *dc_lock_sizes,
        e.stride(0),
        e.stride(1),
        c.stride(0),
        c.stride(1),
        1 if bias is None else bias.stride(0),
        1 if valids is None else valids.stride(0),
        filter_eps,
        shift=shift,
        B_BIN=b_bin_fn(B),
        USE_KAHAN=use_kahan,
    )

    if dbias is not None:
        assert bias is not None
        dbias = dbias.to(dtype=bias.dtype)

    return de, dc, dbias
