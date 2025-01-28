# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import pytest
import torch

from cut_cross_entropy import linear_cross_entropy
from cut_cross_entropy.constants import IGNORE_INDEX
from cut_cross_entropy.utils import softcapping

skip_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")


def _grads(
    e: torch.Tensor,
    c: torch.Tensor,
    targets: torch.Tensor,
    bias: torch.Tensor | None,
    softcap: float | None,
    shift: bool,
    reduction: str,
    fp32: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    orig_e, orig_c, orig_bias = e, c, bias
    if bias is not None:
        bias.grad = None
    e.grad = c.grad = None

    N, T = targets.size()
    if shift:
        e = e[:, :-1]
        targets = targets[:, 1:]
        T = T - 1

    e = e.flatten(0, -2)
    targets = targets.flatten()

    if fp32:
        e = e.float()
        c = c.float()
        bias = bias.float() if bias is not None else None

    logits = e @ c.T
    if bias is not None:
        logits += bias

    if softcap is not None:
        logits = softcapping(logits, softcap)

    loss = torch.nn.functional.cross_entropy(
        logits.float(), targets, ignore_index=IGNORE_INDEX, reduction=reduction
    )

    if reduction == "sum":
        loss = loss / (targets != IGNORE_INDEX).count_nonzero()

    loss.mean().backward()

    assert orig_e.grad is not None
    assert orig_c.grad is not None

    if bias is not None:
        assert orig_bias is not None
        assert orig_bias.grad is not None
        return (
            orig_e.grad.detach().clone(),
            orig_c.grad.detach().clone(),
            orig_bias.grad.detach().clone(),
        )
    else:
        return orig_e.grad.detach().clone(), orig_c.grad.detach().clone()


@skip_no_cuda
@pytest.mark.parametrize("impl", ["cce", "torch_compile", "cce_exact"])
@pytest.mark.parametrize("dtype,error_tol", [(torch.float16, 1e-3), (torch.bfloat16, 1e-2)])
@pytest.mark.parametrize("softcap", [None, 20.0])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("shift", [False, True])
@pytest.mark.parametrize("invalids", [False, True])
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
@pytest.mark.parametrize("shape", [(256, 512, 128), (252, 507, 128), (252, 507, 123)])
def test_loss_backward(
    impl: str,
    dtype: torch.dtype,
    error_tol: float,
    softcap: float | None,
    has_bias: bool,
    shift: bool,
    invalids: bool,
    reduction: str,
    shape: tuple[int, int, int],
):
    torch.set_float32_matmul_precision("highest")
    torch._dynamo.config.cache_size_limit = 256
    torch.cuda.manual_seed(0)

    if dtype == torch.bfloat16 and not torch.cuda.is_available():
        pytest.skip(reason="BF16 not avaliable")

    N, V, D = shape
    e = torch.randn((N, D), device="cuda", dtype=dtype, requires_grad=False) / (D**0.5)
    c = torch.randn((V, D), device="cuda", dtype=dtype, requires_grad=False)

    c[0 : min(N, V) // 2] = e[0 : min(N, V) // 2]

    targets = torch.randint(0, V, size=(N,), device="cuda")

    if invalids:
        inds = torch.randperm(len(targets), device="cuda")[0 : int(0.2 * len(targets))]
        targets[inds] = IGNORE_INDEX

    e = e.view(4, -1, D)

    targets = targets.view(e.size()[0:-1])

    if has_bias:
        bias = torch.randn(V, device="cuda", dtype=dtype) * 0.02
        bias.requires_grad_(True)
    else:
        bias = None

    e.requires_grad_(True)
    c.requires_grad_(True)

    gt = _grads(e, c, targets, bias, softcap, shift, reduction, fp32=True)

    ref = _grads(e, c, targets, bias, softcap, shift, reduction)

    e.grad = c.grad = None
    if bias is not None:
        bias.grad = None

    loss = linear_cross_entropy(
        e, c, targets, bias=bias, softcap=softcap, shift=shift, reduction=reduction, impl=impl
    )
    if reduction == "sum":
        loss = loss / (targets != IGNORE_INDEX).count_nonzero()
    loss.mean().backward()
    assert e.grad is not None
    assert c.grad is not None

    if bias is not None:
        assert bias.grad is not None
        cce = (e.grad, c.grad, bias.grad)
    else:
        cce = (e.grad, c.grad)

    expected_error = tuple((vgt - vref).abs().flatten() for vgt, vref in zip(gt, ref, strict=True))
    cce_error = tuple((vgt - vcce).abs().flatten() for vgt, vcce in zip(gt, cce, strict=True))

    for i in range(len(expected_error)):
        if not (cce_error[i] <= (expected_error[i] + error_tol)).all():
            errors = (cce_error[i] - expected_error[i]).relu()
            argmax_error = int(errors.argmax())
            raise ValueError(
                f"{i=}, {errors.max()=}, {cce[i].flatten()[argmax_error]=}, "
                f"{gt[i].flatten()[argmax_error]=}, {ref[i].flatten()[argmax_error]=}"
            )


@skip_no_cuda
@pytest.mark.parametrize(
    "compute_de,compute_dc,compute_dbias",
    [(True, False, True), (False, True, False), (False, False, True)],
)
def test_loss_partials(compute_de: bool, compute_dc: bool, compute_dbias: bool):
    torch.cuda.manual_seed(0)
    dtype = torch.bfloat16

    N, V, D = (256, 512, 128)
    e = torch.randn((N, D), device="cuda", dtype=dtype, requires_grad=False) / (D**0.5)
    c = torch.randn((V, D), device="cuda", dtype=dtype, requires_grad=False)
    bias = torch.randn(V, device="cuda", dtype=dtype, requires_grad=False) * 0.01

    c[0 : min(N, V) // 2] = e[0 : min(N, V) // 2]

    targets = torch.randint(0, V, size=(N,), device="cuda")

    e = e.view(4, -1, D)
    targets = targets.view(e.size()[0:-1])

    e.requires_grad_(compute_de)
    c.requires_grad_(compute_dc)
    bias.requires_grad_(compute_dbias)

    e.grad = c.grad = bias.grad = None
    loss = linear_cross_entropy(e, c, targets, bias=bias, reduction="mean")
    loss.backward()

    assert (e.grad is not None) == compute_de
    assert (c.grad is not None) == compute_dc
    assert (bias.grad is not None) == compute_dbias
