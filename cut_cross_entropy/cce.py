# Copyright (C) 2024 Apple Inc. All Rights Reserved.
from dataclasses import dataclass
from typing import cast

import torch

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


@dataclass
class CCEParams:
    targets: torch.Tensor
    valids: torch.Tensor | None
    softcap: float | None
    reduction: str
    filter_eps: float | None
    shift: int
    batch_shape: torch.Size
    use_kahan: bool


@torch.compile(fullgraph=True, dynamic=True)
def sort_logit_avg(logit_avg: torch.Tensor) -> torch.Tensor:
    return torch.argsort(logit_avg).to(torch.int32)


class LinearCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        e: torch.Tensor,
        c: torch.Tensor,
        bias: torch.Tensor | None,
        params: CCEParams,
    ) -> torch.Tensor:
        needs_grad = e.requires_grad or c.requires_grad
        return_logit_avg = needs_grad and params.filter_eps is not None

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

        neg_dot = indexed_neg_dot_forward_kernel(
            e=e,
            c=c,
            inds=params.targets,
            bias=bias,
            shift=params.shift,
            valids=params.valids,
            softcap=params.softcap,
            out_dtype=lse.dtype,
        )

        nll = neg_dot.add_(lse)

        reduction = params.reduction
        if reduction == "mean":
            loss = nll.mean()
        elif reduction == "sum":
            loss = nll.sum()
        elif reduction == "none":
            loss = handle_reduction_none(params.batch_shape, params.valids, params.shift, nll)
        else:
            raise ValueError(f"Unknown reduction {reduction}")

        ctx.save_for_backward(e, c, bias, lse, params.targets, params.valids, logit_avg)
        ctx.params = params

        return loss

    @staticmethod
    def backward(
        ctx, grad_out: torch.Tensor
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

        de, dc, dbias = cce_backward_kernel(
            do=grad_out,
            e=e,
            c=c,
            bias=bias,
            lse=lse,
            valids=valids,
            softcap=params.softcap,
            filter_eps=params.filter_eps,
            targets=targets,
            shift=params.shift,
            vocab_ordering=vocab_ordering,
            grad_scale=grad_scale,
            use_kahan=params.use_kahan,
        )

        return de, dc, dbias, None


def linear_cross_entropy_apply(
    e: torch.Tensor,
    c: torch.Tensor,
    bias: torch.Tensor | None,
    params: CCEParams,
) -> torch.Tensor:
    loss = LinearCrossEntropyFunction.apply(e, c, bias, params)
    assert isinstance(loss, torch.Tensor)

    if params.shift != 0 and params.reduction == "none":
        loss = loss[..., params.shift :]

    return loss


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
    filter_eps: float | str | None = "auto",
    use_kahan: bool = False,
) -> torch.Tensor:
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

    return linear_cross_entropy_apply(
        e,
        c,
        bias,
        CCEParams(
            targets,
            valids,
            softcap,
            reduction,
            _handle_eps(filter_eps, e.dtype),
            shift,
            batch_shape,
            use_kahan,
        ),
    )
