# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import pytest
import torch

from cut_cross_entropy.indexed_dot import indexed_neg_dot_forward_kernel
from cut_cross_entropy.utils import softcapping

skip_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")


@skip_no_cuda
@pytest.mark.parametrize(
    "dtype,error_tol", [(torch.float32, 5e-7), (torch.float16, 1e-3), (torch.bfloat16, 1e-2)]
)
@pytest.mark.parametrize("softcap", [None, 20.0])
@pytest.mark.parametrize("shape", [(256, 512, 128), (255, 507, 128), (255, 507, 123)])
def test_indexed_dot(
    dtype: torch.dtype,
    error_tol: float,
    softcap: float | None,
    shape: tuple[int, int, int],
):
    torch.cuda.manual_seed(0)

    if dtype == torch.bfloat16 and not torch.cuda.is_available():
        pytest.skip(reason="BF16 not avaliable")

    N, V, D = shape
    e = torch.randn((N, D), device="cuda", dtype=dtype) / (D**0.5)
    c = torch.randn((V, D), device="cuda", dtype=dtype)

    c[0 : min(N, V) // 2] = e[0 : min(N, V) // 2]

    inds = torch.randint(0, V, size=(N,), device="cuda")

    gt = -(e.float() * c[inds].float()).sum(-1)
    if softcap is not None:
        gt = softcapping(gt, softcap)

    ref = -(e * c[inds]).sum(-1, dtype=torch.float32)
    if softcap is not None:
        ref = softcapping(ref, softcap)

    cce_neg_dot = indexed_neg_dot_forward_kernel(e, c, inds, softcap=softcap)

    expected_error = (gt - ref).abs()
    cce_error = (gt - cce_neg_dot).abs()

    assert (
        cce_error <= (expected_error + error_tol)
    ).all(), f"{(cce_error - expected_error).relu().max()=}"
