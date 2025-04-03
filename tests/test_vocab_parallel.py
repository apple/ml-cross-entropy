# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import contextlib
import socket

import pytest
import torch
import torch.distributed
from torch.multiprocessing.spawn import spawn as mp_spawn

from cut_cross_entropy import VocabParallelOptions, linear_cross_entropy
from cut_cross_entropy.constants import IGNORE_INDEX
from cut_cross_entropy.vocab_parallel.utils import partition_n_into_range


def find_free_port() -> int:
    """
    Returns a free port on the system.
    Note that this can only be used to find a port for torch.distribted
    if it's called by a process on the node that will have
    world_rank == 0 and then all ranks are created. If you
    just called `find_free_port()` on each rank independently, every
    rank will have a different port!
    """
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("localhost", 0))
        _, port = sock.getsockname()
        return port


def _target_fn_test_vp(
    rank: int, world_size: int, port: int, dtype: torch.dtype, error_tol: float, invalids: bool
):
    device = (
        torch.device("cpu")
        if not torch.cuda.is_available()
        else torch.device("cuda", rank % torch.cuda.device_count())
    )

    if device.type == "cuda":
        torch.cuda.set_device(device)
        backend = "cpu:gloo,cuda:nccl"
    else:
        backend = "gloo"

    store = torch.distributed.TCPStore(
        "localhost", port, world_size=world_size, is_master=rank == 0
    )

    torch.distributed.init_process_group(
        backend=backend, store=store, world_size=world_size, rank=rank
    )

    N, V, D = (252, 507, 123)

    e = torch.randn((N, D), device=device, dtype=dtype) / (D**0.5)
    c = torch.randn((V, D), device=device, dtype=dtype)

    targets = torch.randint(0, V, size=(N,), device=device)
    if invalids:
        inds = torch.randperm(len(targets), device=device)[0 : int(0.2 * len(targets))]
        targets[inds] = IGNORE_INDEX

    e = e.view(4, -1, D)
    targets = targets.view(e.size()[0:-1])

    vocab_parallel_options = VocabParallelOptions.for_vocab(V)

    vp_c = c[vocab_parallel_options.start : vocab_parallel_options.stop].clone()

    vp_c.requires_grad_(True)
    vp_loss = linear_cross_entropy(e, vp_c, targets, vocab_parallel_options=vocab_parallel_options)
    vp_loss.backward()

    c.requires_grad_(True)
    loss = linear_cross_entropy(e, c, targets)
    loss.backward()

    assert c.grad is not None
    assert vp_c.grad is not None
    assert torch.allclose(
        c.grad[vocab_parallel_options.start : vocab_parallel_options.stop],
        vp_c.grad,
        atol=error_tol,
    )


@pytest.mark.parametrize("impl", ["torch_compile", "cce"])
@pytest.mark.parametrize("dtype,error_tol", [(torch.float16, 1e-3), (torch.bfloat16, 1e-2)])
@pytest.mark.parametrize("nprocs", [4])
@pytest.mark.parametrize("invalids", [False, True])
def test_vocab_parallel(
    impl: str, dtype: torch.dtype, error_tol: float, nprocs: int, invalids: bool
):
    if impl == "cce" and not torch.cuda.is_available():
        pytest.skip("Testing vocab parallel CCE requires cuda")

    mp_spawn(
        _target_fn_test_vp,
        args=(nprocs, find_free_port(), dtype, error_tol, invalids),
        nprocs=nprocs,
        join=True,
    )


@pytest.mark.parametrize("n", [1023, 2048])
@pytest.mark.parametrize("world_size", [7, 8])
def test_partition_n_into_range(n: int, world_size: int):
    start = 0
    for rank in range(world_size):
        end = start + n // world_size + (1 if rank < (n % world_size) else 0)

        assert partition_n_into_range(n, rank, world_size) == (start, end)

        start = end

    assert end == n
    assert partition_n_into_range(n, world_size - 1, world_size)[1] == n
