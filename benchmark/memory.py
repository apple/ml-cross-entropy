# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import gc
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Callable, TypeVar

import torch
from typing_extensions import Self

T = TypeVar("T")


def mem_delta(s1, s0):
    return s1["allocated_bytes.all.peak"] - s0["allocated_bytes.all.peak"]


def calc_tensor_memory_usage(t: torch.Tensor | Iterable[torch.Tensor]) -> float:
    mem = 0.0
    if isinstance(t, torch.Tensor):
        mem += t.numel() * torch.finfo(t.dtype).bits / 8
    elif isinstance(t, Iterable):
        mem += sum(calc_tensor_memory_usage(v) for v in t)
    else:
        raise RuntimeError(f"Unknown return type {type(t)}")

    return mem


def detach_rval(t: torch.Tensor, *_args, **_kwargs) -> torch.Tensor:
    return t.detach()


@dataclass(frozen=True)
class Stats:
    runtime_ms: float  # in seconds
    op_mem_mb: float  # in GB

    @classmethod
    def measure(
        cls,
        f: Callable[..., torch.Tensor],
        *args,
        n_iteration: int = 1,
        forward: bool = True,
        backward: bool = False,
        pre_fn: Callable[..., None] | None = None,
        **kwds,
    ) -> Self:
        if pre_fn is not None:
            pre_fn(*args, **kwds)

        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

        mem_usages = 0
        torch.cuda.reset_peak_memory_stats()
        s0 = torch.cuda.memory_stats()
        t = f(*args, **kwds)

        torch.cuda.synchronize()
        if forward:
            s1 = torch.cuda.memory_stats()

        if backward and not forward:
            torch.cuda.reset_peak_memory_stats()
            s0 = torch.cuda.memory_stats()

        if backward:
            if t.numel() > 1:
                t = t.mean()
            t.backward()

        torch.cuda.synchronize()
        if backward:
            s1 = torch.cuda.memory_stats()

        mem_usages = mem_delta(s1, s0)

        # Run the function to benchmark
        # rval = None
        all_cuda_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        for _ in range(n_iteration):
            if pre_fn is not None:
                pre_fn(*args, **kwds)

            start = torch.cuda.Event(enable_timing=True)

            if forward:
                start.record(torch.cuda.current_stream())

            t = f(*args, **kwds)

            if not forward:
                start.record(torch.cuda.current_stream())

            if backward:
                if t.numel() > 1:
                    t = t.mean()
                t.backward()

            end = torch.cuda.Event(enable_timing=True)
            end.record(torch.cuda.current_stream())

            all_cuda_events.append((start, end))

        torch.cuda.synchronize()
        total_time = 0.0
        for start, end in all_cuda_events:
            start.synchronize()
            end.synchronize()
            total_time += start.elapsed_time(end)

        s1 = torch.cuda.memory_stats()

        return cls(
            total_time / n_iteration,
            mem_usages / 2**20,
        )
