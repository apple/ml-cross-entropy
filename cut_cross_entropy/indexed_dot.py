# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import torch
import triton
import triton.language as tl

from cut_cross_entropy.tl_autotune import indexed_dot_autotune
from cut_cross_entropy.tl_utils import b_bin_fn
from cut_cross_entropy.utils import softcapping


def _indexed_neg_dot_forward_kernel(
    E,
    C,
    Inds,
    Valids,
    Out,
    B,
    D,
    stride_eb,
    stride_ed,
    stride_cv,
    stride_cd,
    stride_ib,
    stride_vb,
    B_BIN,
    BLOCK_B: tl.constexpr,
    BLOCK_D: tl.constexpr,
    GROUP_B: tl.constexpr,
    HAS_VALIDS: tl.constexpr,
    EVEN_D: tl.constexpr,
    SHIFT: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_b_chunks = tl.cdiv(B, BLOCK_B)
    num_d_chunks = tl.cdiv(D, BLOCK_D)
    num_d_in_group = GROUP_B * num_d_chunks
    group_id = pid // num_d_in_group
    first_pid_b = group_id * GROUP_B
    group_size_b = min(num_b_chunks - first_pid_b, GROUP_B)
    pid_b = first_pid_b + ((pid % num_d_in_group) % group_size_b)
    pid_d = (pid % num_d_in_group) // group_size_b

    offs_b = (tl.arange(0, BLOCK_B) + pid_b * BLOCK_B) % B
    if HAS_VALIDS:
        offs_b = tl.load(Valids + stride_vb * offs_b)

    offs_d = tl.arange(0, BLOCK_D) + pid_d * BLOCK_D
    e_ptrs = E + (stride_eb * offs_b[:, None] + stride_ed * offs_d[None, :])
    if EVEN_D:
        e = tl.load(e_ptrs)
    else:
        e = tl.load(e_ptrs, mask=offs_d[None, :] < D, other=0.0)

    inds = tl.load(Inds + stride_ib * ((offs_b + 1) if SHIFT else offs_b))

    c_ptrs = C + (inds[:, None] * stride_cv + offs_d[None, :] * stride_cd)
    if EVEN_D:
        c = tl.load(c_ptrs)
    else:
        c = tl.load(c_ptrs, mask=offs_d[None, :] < D, other=0.0)

    offs_b = tl.arange(0, BLOCK_B) + pid_b * BLOCK_B
    out_ptrs = Out + offs_b
    dot = (e * c).to(tl.float32)
    neg_dot = -tl.sum(dot, 1).to(out_ptrs.dtype.element_ty)
    tl.atomic_add(out_ptrs, neg_dot, mask=offs_b < B)


_indexed_neg_dot_forward_kernel = triton.jit(_indexed_neg_dot_forward_kernel)
_indexed_neg_dot_forward_kernel = triton.heuristics(  # type: ignore
    {
        "EVEN_D": lambda args: args["D"] % args["BLOCK_D"] == 0,
        "HAS_VALIDS": lambda args: args["Valids"] is not None,
        "GROUP_B": lambda args: 8,
    }
)(_indexed_neg_dot_forward_kernel)
_indexed_neg_dot_forward_kernel = indexed_dot_autotune()(_indexed_neg_dot_forward_kernel)  # type: ignore


def indexed_neg_dot_forward_kernel(
    e: torch.Tensor,
    c: torch.Tensor,
    inds: torch.Tensor,
    shift: bool = False,
    valids: torch.Tensor | None = None,
    softcap: float | None = None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    assert inds.ndim == 1
    assert e.ndim == 2
    assert c.ndim == 2
    assert inds.size(0) == e.size(0)
    assert c.size(1) == e.size(1)

    if valids is not None:
        assert valids.ndim == 1
        B = valids.size(0)
    else:
        B = e.size(0)

    out = e.new_zeros((B,), dtype=torch.float32)

    def grid(META) -> tuple[int]:
        return (triton.cdiv(B, META["BLOCK_B"]) * triton.cdiv(e.size(1), META["BLOCK_D"]),)

    _indexed_neg_dot_forward_kernel[grid](
        e,
        c,
        inds,
        valids,
        out,
        B,
        e.size(1),
        e.stride(0),
        e.stride(1),
        c.stride(0),
        c.stride(1),
        inds.stride(0),
        1 if valids is None else valids.stride(0),
        B_BIN=b_bin_fn(B),
        SHIFT=shift,
    )

    if softcap is not None:
        out = softcapping(out, softcap)

    if out_dtype is None:
        out_dtype = e.dtype

    out = out.to(out_dtype)

    return out
