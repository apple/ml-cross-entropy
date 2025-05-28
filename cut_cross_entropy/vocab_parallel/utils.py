# Copyright (C) 2024 Apple Inc. All Rights Reserved.
from dataclasses import dataclass

import torch
import torch.distributed
from typing_extensions import Self


def partition_n_into_range(n: int, rank: int, world_size: int) -> tuple[int, int]:
    start = rank * (n // world_size) + min(rank, n % world_size)
    stop = start + n // world_size + (1 if rank < (n % world_size) else 0)

    return start, stop


@dataclass
class VocabParallelOptions:
    """Options to configure vocab parallel loss computation

    :param start: The start index for this rank's range in the vocab.
    :param end: The ending index (non-inclusive)
    :param group: The distributed process group defining the world for this vocab parallel rank.
    :param reduce_e_grad: Whether or not to all_reduce/synchronize the gradient for the embedding
        matrix across all ranks. This typically should be true, but some frameworks may require setting this to false.
    """

    start: int
    stop: int
    group: torch.distributed.ProcessGroup | None = None
    reduce_e_grad: bool = True

    @classmethod
    def from_vocab(
        cls,
        vocab_size: int,
        group: torch.distributed.ProcessGroup | None = None,
        reduce_e_grad: bool = True,
    ) -> Self:
        rank = torch.distributed.get_rank(group)
        world_size = torch.distributed.get_world_size(group)

        start, stop = partition_n_into_range(vocab_size, rank, world_size)

        return cls(start, stop, group, reduce_e_grad)


@torch.compile(fullgraph=True)
def vp_reduce_lse(vp_lse: torch.Tensor, pg: torch.distributed.ProcessGroup | None) -> torch.Tensor:
    lse_max = vp_lse.clone()
    torch.distributed.all_reduce(lse_max, op=torch.distributed.ReduceOp.MAX, group=pg)

    lse = (vp_lse - lse_max).exp()
    torch.distributed.all_reduce(lse, group=pg)
    return lse_max + lse.log()


@torch.compile(fullgraph=True)
def vp_reduce_correct_logit(
    vp_correct_logit: torch.Tensor,
    pg: torch.distributed.ProcessGroup | None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    correct_logit = vp_correct_logit.to(dtype=dtype, copy=True)
    torch.distributed.all_reduce(correct_logit, group=pg)
    return correct_logit


@torch.compile(fullgraph=True)
def vp_reduce_e_grad(
    e_grad: torch.Tensor,
    pg: torch.distributed.ProcessGroup | None,
) -> torch.Tensor:
    reduced_grad = e_grad.to(dtype=torch.float32, copy=True)
    torch.distributed.all_reduce(reduced_grad, group=pg)

    return reduced_grad.type_as(e_grad)


class _VocabParallelReduceEGradHook(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, e: torch.Tensor, pg: torch.distributed.ProcessGroup | None = None
    ) -> torch.Tensor:
        ctx.pg = pg
        return e

    @staticmethod
    def backward(ctx, e_grad: torch.Tensor) -> tuple[torch.Tensor, None]:
        return vp_reduce_e_grad(e_grad, ctx.pg), None


def vp_reduce_e_grad_hook(
    e: torch.Tensor, vocab_parallel_options: VocabParallelOptions
) -> torch.Tensor:
    if vocab_parallel_options.reduce_e_grad:
        return _VocabParallelReduceEGradHook.apply(e, vocab_parallel_options.group)
    else:
        return e
