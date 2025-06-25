# Copyright (C) 2024 Apple Inc. All Rights Reserved.
from dataclasses import dataclass
from typing import cast

import torch
import torch.amp

from cut_cross_entropy.cce_backward import cce_backward_kernel
from cut_cross_entropy.cce_lse_forward import cce_lse_forward_kernel
from cut_cross_entropy.constants import IGNORE_INDEX
from cut_cross_entropy.doc import CCE_OPTS_DOC, LINEAR_CROSS_ENTROPY_DOC, add_doc_start
from cut_cross_entropy.indexed_dot import indexed_neg_dot_forward_kernel
from cut_cross_entropy.utils import (
    _build_flat_valids,
    _handle_eps,
    handle_reduction_none,
)
from cut_cross_entropy.vocab_parallel.utils import (
    VocabParallelOptions,
    vp_reduce_correct_logit,
    vp_reduce_lse,
)


@dataclass
class CCEParams:
    targets: torch.Tensor
    valids: torch.Tensor | None
    softcap: float | None
    reduction: str
    filter_eps: float | None
    shift: int
    batch_shape: torch.Size
    accum_e_fp32: bool
    accum_c_fp32: bool
    filter_e_grad: bool
    filter_c_grad: bool
    vocab_parallel_options: VocabParallelOptions | None
    return_lse: bool


@torch.compile(fullgraph=True)
def sort_logit_avg(logit_avg: torch.Tensor) -> torch.Tensor:
    return torch.argsort(logit_avg).to(torch.int32)


class LinearCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        e: torch.Tensor,
        c: torch.Tensor,
        bias: torch.Tensor | None,
        params: CCEParams,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        needs_grad = e.requires_grad or c.requires_grad
        if bias is not None:
            needs_grad = needs_grad or bias.requires_grad

        return_logit_avg = needs_grad and params.filter_eps is not None

        compute_dbias = bias is not None and bias.requires_grad
        compute_de = e.requires_grad
        compute_dc = c.requires_grad

        if torch.is_autocast_enabled():
            e = e.to(dtype=torch.get_autocast_dtype("cuda"))
            c = c.to(dtype=torch.get_autocast_dtype("cuda"))

            if bias is not None:
                bias = bias.to(dtype=torch.get_autocast_dtype("cuda"))

        ret = cce_lse_forward_kernel(
            e=e,
            c=c,
            bias=bias,
            valids=params.valids,
            softcap=params.softcap,
            return_logit_avg=return_logit_avg,
        )
        if return_logit_avg:
            assert isinstance(ret, tuple)
            lse, logit_avg = ret
        else:
            assert isinstance(ret, torch.Tensor)
            lse = ret
            logit_avg = None

        if (vp_opts := params.vocab_parallel_options) is not None:
            lse = vp_reduce_lse(lse, pg=vp_opts.group)

            if params.valids is not None:
                targets = params.targets[params.valids + params.shift]
            else:
                targets = params.targets

            vp_valids = (
                ((targets >= vp_opts.start) & (targets < vp_opts.stop)).nonzero().to(torch.int32)
            )
            assert vp_valids.size(1) == 1
            vp_valids = vp_valids.squeeze(-1)

            if params.valids is not None:
                neg_dot_valids = params.valids[vp_valids]
            else:
                neg_dot_valids = vp_valids

            neg_dot_targets = params.targets - vp_opts.start
        else:
            neg_dot_valids = params.valids
            neg_dot_targets = params.targets
            vp_valids = None

        neg_dot = indexed_neg_dot_forward_kernel(
            e=e,
            c=c,
            inds=neg_dot_targets,
            bias=bias,
            shift=params.shift,
            valids=neg_dot_valids,
            softcap=params.softcap,
            out_dtype=lse.dtype,
        )

        if params.vocab_parallel_options is not None:
            global_neg_dot = neg_dot.new_zeros(lse.size())
            assert vp_valids is not None
            global_neg_dot[vp_valids] = neg_dot

            neg_dot = vp_reduce_correct_logit(
                global_neg_dot, pg=params.vocab_parallel_options.group, dtype=lse.dtype
            )

        nll = neg_dot.add_(lse)

        ctx.save_for_backward(e, c, bias, lse, params.targets, params.valids, logit_avg)
        ctx.params = params
        ctx.compute_dbias = compute_dbias
        ctx.compute_de = compute_de
        ctx.compute_dc = compute_dc

        if not params.return_lse:
            lse = None
        else:
            lse = handle_reduction_none(params.batch_shape, params.valids, params.shift, lse)

        reduction = params.reduction
        if reduction == "mean":
            loss = nll.mean()
        elif reduction == "sum":
            loss = nll.sum()
        elif reduction == "none":
            loss = handle_reduction_none(params.batch_shape, params.valids, params.shift, nll)
        else:
            raise ValueError(f"Unknown reduction {reduction}")

        return loss, lse

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(
        ctx, grad_out: torch.Tensor, grad_lse_out: torch.Tensor | None
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, None]:
        e, c, bias, lse, targets, valids, logit_avg = ctx.saved_tensors

        if logit_avg is not None:
            vocab_ordering = sort_logit_avg(logit_avg)
        else:
            vocab_ordering = None

        params = cast(CCEParams, ctx.params)
        reduction = params.reduction
        if reduction == "mean":
            grad_scale = 1 / lse.numel()
        elif reduction == "sum":
            grad_scale = 1.0
        elif reduction == "none":
            grad_scale = 1.0
            grad_out = grad_out.view(-1)
        else:
            raise ValueError(f"Unknown reduction {reduction}")

        if grad_lse_out is not None:
            grad_lse_out = grad_lse_out.view(-1)

        reduce_e_grad = False
        pg = None
        if (vp_opts := params.vocab_parallel_options) is not None:
            is_my_target = (targets >= vp_opts.start) & (targets < vp_opts.stop)
            targets = torch.where(
                is_my_target,
                targets - vp_opts.start,
                ## NB
                # The backward kernel already uses
                # c.size(0) + 1 as the padding value to ensure that
                # (targets.size(0) % block_size) == 0, so for targets
                # that aren't in this VP rank's range, we can just consider
                # them as padded and all work work as expected.
                targets.new_full((), c.size(0) + 1),
            )

            reduce_e_grad = vp_opts.reduce_e_grad
            pg = vp_opts.group

        de, dc, dbias = cce_backward_kernel(
            do=grad_out,
            dlse=grad_lse_out,
            e=e,
            compute_de=ctx.compute_de,
            c=c,
            compute_dc=ctx.compute_dc,
            bias=bias,
            compute_dbias=ctx.compute_dbias,
            lse=lse,
            valids=valids,
            softcap=params.softcap,
            filter_eps=params.filter_eps,
            targets=targets,
            shift=params.shift,
            vocab_ordering=vocab_ordering,
            grad_scale=grad_scale,
            accum_e_fp32=params.accum_e_fp32,
            accum_c_fp32=params.accum_c_fp32,
            filter_e_grad=params.filter_e_grad,
            filter_c_grad=params.filter_c_grad,
            reduce_e_grad=reduce_e_grad,
            pg=pg,
        )

        return de, dc, dbias, None


def linear_cross_entropy_apply(
    e: torch.Tensor,
    c: torch.Tensor,
    bias: torch.Tensor | None,
    params: CCEParams,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    loss, lse = cast(
        tuple[torch.Tensor, torch.Tensor | None],
        LinearCrossEntropyFunction.apply(e, c, bias, params),
    )

    if params.shift != 0 and params.reduction == "none":
        loss = loss[..., params.shift :]

    if params.return_lse and params.shift != 0:
        assert lse is not None
        lse = lse[..., params.shift :]

    return loss, lse


@add_doc_start(LINEAR_CROSS_ENTROPY_DOC)
@add_doc_start(*(doc_str + "\n" for doc_str in CCE_OPTS_DOC))
def cce_linear_cross_entropy(
    e: torch.Tensor,
    c: torch.Tensor,
    targets: torch.Tensor,
    bias: torch.Tensor | None = None,
    ignore_index: int = IGNORE_INDEX,
    softcap: float | None = None,
    reduction: str = "mean",
    shift: bool | int = 0,
    return_lse: bool = False,
    filter_eps: float | str | None = "auto",
    accum_e_fp32: bool = False,
    accum_c_fp32: bool = False,
    filter_e_grad: bool = True,
    filter_c_grad: bool = True,
    vocab_parallel_options: VocabParallelOptions | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    assert e.size()[0:-1] == targets.size()
    assert e.size(-1) == c.size(1)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "Cut Cross Entropy requires an ampere GPU or newer. "
            "Consider using torch_compile_linear_cross_entropy for scenarios where one is not available."
        )

    batch_shape = targets.size()

    e = e.contiguous()
    targets = targets.contiguous()

    shift = int(shift)
    valids = _build_flat_valids(targets, ignore_index, shift)

    e = e.flatten(0, -2)
    targets = targets.flatten()

    if (targets.data_ptr() % 16) != 0:
        targets = torch.nn.functional.pad(targets, (0, 1))[:-1]

    assert (targets.data_ptr() % 16) == 0
    cce_params = CCEParams(
        targets,
        valids,
        softcap,
        reduction,
        _handle_eps(filter_eps, e.dtype),
        shift,
        batch_shape,
        accum_e_fp32,
        accum_c_fp32,
        filter_e_grad=filter_e_grad and filter_eps is not None,
        filter_c_grad=filter_c_grad and filter_eps is not None,
        vocab_parallel_options=vocab_parallel_options,
        return_lse=return_lse,
    )

    return linear_cross_entropy_apply(e, c, bias, cce_params)
