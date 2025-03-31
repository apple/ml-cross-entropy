# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import torch
import torch.distributed

from cut_cross_entropy.utils import (
    softcapping,
)
from cut_cross_entropy.vocab_parallel.vocab_parallel_lce import VocabParallelOptions


class _VocabParallelLossFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        correct_logit: torch.Tensor,
        lse: torch.Tensor,
        this_weight: float,
        pg: torch.distributed.ProcessGroup | None,
    ) -> torch.Tensor:
        lse = lse.clone()
        correct_logit = correct_logit.clone()

        lse_max = lse.clone()
        torch.distributed.all_reduce(lse_max, op=torch.distributed.ReduceOp.MAX, group=pg)

        lse = (lse - lse_max).exp() * this_weight
        torch.distributed.all_reduce(lse, group=pg)
        lse = lse_max + lse.log()

        torch.distributed.all_reduce(correct_logit, group=pg)
        return lse - correct_logit.type_as(lse)

    @staticmethod
    def backward(
        ctx, grad_loss: torch.Tensor | None
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None, None]:
        return (-grad_loss if grad_loss is not None else None, grad_loss, None, None)


def _vocab_parallel_loss_fn(
    correct_logit: torch.Tensor,
    lse: torch.Tensor,
    this_weight: float,
    pg: torch.distributed.ProcessGroup | None,
) -> torch.Tensor:
    return _VocabParallelLossFn.apply(correct_logit, lse, this_weight, pg)


@torch.compile(fullgraph=True)
def _vocab_parallel_torch_compile_lce_apply(
    e: torch.Tensor,
    vocab_parallel_c: torch.Tensor,
    targets: torch.Tensor,
    start: int,
    stop: int,
    vocab_parallel_bias: torch.Tensor | None = None,
    softcap: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = e @ vocab_parallel_c.T

    if vocab_parallel_bias is not None:
        logits = logits + vocab_parallel_bias

    if softcap is not None:
        logits = softcapping(logits, softcap)

    lse = torch.logsumexp(logits.float(), -1)

    is_target_in_range = (targets < stop) & (targets >= start)
    arange_indexer = torch.arange(0, len(lse), device=targets.device, dtype=targets.dtype)
    masked_targets = torch.where(is_target_in_range, targets, targets.new_zeros(()))

    correct_logit = torch.where(
        is_target_in_range, logits[arange_indexer, masked_targets], logits.new_zeros(())
    )

    return correct_logit, lse


@torch.compile(fullgraph=True)
def _vocab_parallel_loss(
    correct_logit: torch.Tensor,
    lse: torch.Tensor,
    this_vocab_size: int,
    total_vocab_size: int,
    reduction: str,
    pg: torch.distributed.ProcessGroup | None,
) -> torch.Tensor:
    loss = _vocab_parallel_loss_fn(correct_logit, lse, this_vocab_size / total_vocab_size, pg)

    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError(f"Unknown reduction {reduction!r}")

    return loss


def vocab_parallel_torch_compile_lce_apply(
    vocab_parallel_options: VocabParallelOptions,
    e: torch.Tensor,
    vocab_parallel_c: torch.Tensor,
    targets: torch.Tensor,
    vocab_parallel_bias: torch.Tensor | None,
    softcap: float | None,
    reduction: str,
) -> torch.Tensor:
    pg = vocab_parallel_options.group

    this_vocab_size = vocab_parallel_options.stop - vocab_parallel_options.start
    total_vocab_size = vocab_parallel_options.total_vocab_size
    if total_vocab_size is None:
        total_vocab_size_t = torch.as_tensor(this_vocab_size, dtype=torch.float32, device=e.device)
        torch.distributed.all_reduce(total_vocab_size_t, group=pg)
        total_vocab_size = int(total_vocab_size_t.item())

    correct_logit, lse = _vocab_parallel_torch_compile_lce_apply(
        e,
        vocab_parallel_c,
        targets,
        vocab_parallel_options.start,
        vocab_parallel_options.stop,
        vocab_parallel_bias=vocab_parallel_bias,
        softcap=softcap,
    )

    loss = _vocab_parallel_loss(
        correct_logit, lse, this_vocab_size, total_vocab_size, reduction, pg=pg
    )

    return loss
