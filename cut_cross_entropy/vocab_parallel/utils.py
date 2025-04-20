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
    start: int
    stop: int
    group: torch.distributed.ProcessGroup | None = None

    @classmethod
    def from_vocab(
        cls, vocab_size: int, group: torch.distributed.ProcessGroup | None = None
    ) -> Self:
        rank = torch.distributed.get_rank(group)
        world_size = torch.distributed.get_world_size(group)

        start, stop = partition_n_into_range(vocab_size, rank, world_size)

        return cls(start, stop, group)


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
