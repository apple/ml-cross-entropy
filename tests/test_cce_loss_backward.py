# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import pytest
import torch

from cut_cross_entropy import linear_cross_entropy
from cut_cross_entropy.constants import IGNORE_INDEX
from cut_cross_entropy.utils import compute_z_loss, softcapping

skip_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")


def _grads(
    e: torch.Tensor,
    c: torch.Tensor,
    targets: torch.Tensor,
    bias: torch.Tensor | None,
    softcap: float | None,
    shift: bool,
    reduction: str,
    z_loss: bool,
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

    loss = loss.mean()

    if z_loss:
        lse = torch.logsumexp(logits.float(), dim=-1)
        loss = loss + compute_z_loss(lse, targets)

    loss.backward()

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
@pytest.mark.parametrize(
    "dtype,error_tol", [(torch.float32, 5e-4), (torch.float16, 1e-3), (torch.bfloat16, 1e-2)]
)
@pytest.mark.parametrize("softcap", [None, 20.0])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("shift", [False, True])
@pytest.mark.parametrize("invalids", [False, True])
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
@pytest.mark.parametrize("z_loss", [True, False])
@pytest.mark.parametrize("shape", [(256, 512, 512), (252, 507, 512), (252, 507, 497)])
def test_loss_backward(
    impl: str,
    dtype: torch.dtype,
    error_tol: float,
    softcap: float | None,
    has_bias: bool,
    shift: bool,
    invalids: bool,
    reduction: str,
    z_loss: bool,
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
        bias = torch.randn(V, device="cuda", dtype=dtype) * 0.1
        bias.requires_grad_(True)
    else:
        bias = None

    e.requires_grad_(True)
    c.requires_grad_(True)

    gt = _grads(e, c, targets, bias, softcap, shift, reduction, z_loss, fp32=True)

    torch.set_float32_matmul_precision("high")
    ref = _grads(e, c, targets, bias, softcap, shift, reduction, z_loss)

    e.grad = c.grad = None
    if bias is not None:
        bias.grad = None

    loss_lse = linear_cross_entropy(
        e,
        c,
        targets,
        bias=bias,
        softcap=softcap,
        shift=shift,
        reduction=reduction,
        impl=impl,
        return_lse=z_loss,
    )
    if z_loss:
        assert isinstance(loss_lse, tuple)
        loss, lse = loss_lse
    else:
        assert isinstance(loss_lse, torch.Tensor)
        loss = loss_lse
        lse = None

    if reduction == "sum":
        loss = loss / (targets != IGNORE_INDEX).count_nonzero()

    loss = loss.mean()

    if z_loss:
        assert lse is not None
        loss = loss + compute_z_loss(lse, targets, shift)

    loss.backward()

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


@skip_no_cuda
def test_loss_all_ignored():
    torch.cuda.manual_seed(0)
    dtype = torch.bfloat16

    N, V, D = (256, 512, 128)
    e = torch.randn((N, D), device="cuda", dtype=dtype, requires_grad=False)
    c = torch.randn((V, D), device="cuda", dtype=dtype, requires_grad=False)

    targets = torch.full((N,), IGNORE_INDEX, device="cuda", dtype=torch.int64)

    e = e.view(4, -1, D)
    targets = targets.view(e.size()[0:-1])

    e.requires_grad_(True)
    c.requires_grad_(True)

    loss = linear_cross_entropy(e, c, targets, reduction="mean")
    loss.backward()


@skip_no_cuda
@pytest.mark.manual
def test_gradient_divergence():
    """Compare gradients between PyTorch and CCE at every training step."""
    use_z_loss = True  # Easy toggle for Z-loss
    use_bias = True  # Easy toggle for bias
    z_loss_weight = 0.001  # Z-loss weight
    num_batches = 50  # Number of different e/target pairs to create
    softcap = 20.0

    torch.cuda.manual_seed(41)
    torch.set_float32_matmul_precision("highest")
    dtype = torch.bfloat16
    param_dtype = torch.float32

    N, V, D = (512, 1024 - 21, 1024)
    print(f"{N=} {V=} {D=} {num_batches=}")
    batch_size = 4
    seq_len = N // batch_size
    n_steps = 1000
    lr = 1e-4

    # Create multiple different e/target pairs for mini-batch simulation
    e_batches = []
    targets_batches = []

    for batch_idx in range(num_batches):
        # Create different e and targets for each batch
        e_batch = torch.randn((N, D), device="cuda", dtype=param_dtype)
        e_batch = e_batch * torch.rsqrt(e_batch.pow(2).mean(dim=-1, keepdim=True)) * 10
        targets_batch = torch.randint(0, V, size=(N,), device="cuda")

        e_batch = e_batch.view(batch_size, seq_len, D)
        targets_batch = targets_batch.view(batch_size, seq_len)

        e_batches.append(e_batch)
        targets_batches.append(targets_batch)

    # Create shared C and bias parameters
    c_init = torch.zeros((V, D), device="cuda", dtype=param_dtype)
    if use_bias:
        bias_init = torch.full(
            (V,),
            -torch.log(torch.tensor(V, dtype=torch.float32)).item(),
            device="cuda",
            dtype=param_dtype,
        )
    else:
        bias_init = None
    c_init = torch.nn.init.kaiming_normal_(c_init, 5**0.5)

    # Initialize both models with same parameters (only C and bias are shared)
    c_torch = c_init.clone().requires_grad_(True)
    bias_torch = (
        bias_init.clone().requires_grad_(True) if use_bias and bias_init is not None else None
    )

    c_cce = c_init.clone().requires_grad_(True)
    bias_cce = (
        bias_init.clone().requires_grad_(True) if use_bias and bias_init is not None else None
    )

    params_torch = [c_torch]
    if use_bias and bias_torch is not None:
        params_torch.append(bias_torch)
    params_cce = [c_cce]
    if use_bias and bias_cce is not None:
        params_cce.append(bias_cce)
    optimizer_torch = torch.optim.Adam(params_torch, lr=lr)
    optimizer_cce = torch.optim.Adam(params_cce, lr=lr)

    # Track min/max differences over training
    ce_diff_min = float("inf")
    ce_diff_max = float("-inf")
    z_diff_min = float("inf")
    z_diff_max = float("-inf")

    # Track average gradient differences (only C gradients are meaningful)
    c_grad_diff_sum = 0.0
    c_grad_diff_raw_sum = 0.0
    grad_diff_count = 0

    # Track average loss differences
    ce_diff_sum = 0.0
    z_diff_sum = 0.0
    loss_diff_count = 0

    # Create batch order for shuffling
    batch_order = list(range(num_batches))
    torch.manual_seed(42)  # For reproducible shuffling

    print(
        "Step | PyTorch Loss |  CCE Loss  | PyTorch Z  |  CCE Z   | CE Diff  | Z Diff   | E Grad Diff | C Grad Diff |B Grad Diff| E Param Diff| C Param Diff|B Param Diff"
    )
    print("-" * 160)

    for step in range(n_steps):
        # Shuffle batch order at the start of each epoch (when step is divisible by num_batches)
        if step % num_batches == 0 and step > 0:
            batch_order = torch.randperm(num_batches).tolist()

        # Sample one batch for this step using shuffled order
        batch_idx_in_order = step % num_batches
        batch_idx = batch_order[batch_idx_in_order]
        e_current = e_batches[batch_idx]
        targets_current = targets_batches[batch_idx]

        # PyTorch training step
        optimizer_torch.zero_grad()
        e_flat_torch = e_current.flatten(0, -2).to(dtype=dtype)
        targets_flat = targets_current.flatten()
        logits_torch = e_flat_torch @ c_torch.to(dtype=dtype).T
        if use_bias and bias_torch is not None:
            logits_torch += bias_torch.to(dtype=dtype)

        if softcap is not None:
            logits_torch = softcapping(logits_torch, softcap)

        ce_loss_torch = torch.nn.functional.cross_entropy(
            logits_torch.float(), targets_flat, reduction="mean"
        )

        # Add Z-loss for PyTorch
        if use_z_loss:
            lse_torch = torch.logsumexp(logits_torch.float(), dim=-1)
            z_loss_torch = (lse_torch * lse_torch).mean()
            total_loss_torch = ce_loss_torch + z_loss_weight * z_loss_torch
        else:
            z_loss_torch = torch.tensor(0.0, device="cuda")
            total_loss_torch = ce_loss_torch

        total_loss_torch.backward()

        # Store PyTorch gradients before optimizer step (only for C and bias, not E)
        assert c_torch.grad is not None
        c_grad_torch = c_torch.grad.clone()
        if use_bias and bias_torch is not None:
            assert bias_torch.grad is not None
            bias_grad_torch = bias_torch.grad.clone()
        else:
            bias_grad_torch = None

        # Clip gradients
        # torch.nn.utils.clip_grad_norm_([e_torch, c_torch, bias_torch], max_norm=1.0)

        optimizer_torch.step()

        # CCE training step
        optimizer_cce.zero_grad()

        loss_lse_cce = linear_cross_entropy(
            e_current.to(dtype=dtype),
            c_cce.to(dtype=dtype),
            targets_current,
            bias=bias_cce.to(dtype=dtype) if use_bias and bias_cce is not None else None,
            reduction="none",
            impl="cce_exact",
            return_lse=use_z_loss,
            softcap=softcap,
        )
        if use_z_loss:
            assert isinstance(loss_lse_cce, tuple)
            ce_loss_cce, lse_cce = loss_lse_cce
            ce_loss_cce = ce_loss_cce.mean()
            z_loss_cce = compute_z_loss(lse_cce, targets_current)
            total_loss_cce = ce_loss_cce + z_loss_weight * z_loss_cce
        else:
            assert isinstance(loss_lse_cce, torch.Tensor)
            total_loss_cce = ce_loss_cce = loss_lse_cce.mean()
            z_loss_cce = torch.tensor(0.0, device="cuda")

        total_loss_cce.backward()

        # Store CCE gradients before optimizer step (only for C and bias, not E)
        assert c_cce.grad is not None
        c_grad_cce = c_cce.grad.clone()
        if use_bias and bias_cce is not None:
            assert bias_cce.grad is not None
            bias_grad_cce = bias_cce.grad.clone()
        else:
            bias_grad_cce = None

        # Clip gradients
        # torch.nn.utils.clip_grad_norm_([e_cce, c_cce, bias_cce], max_norm=1.0)

        optimizer_cce.step()

        # Compare gradients (only C and bias, since E batches are different)
        c_grad_diff_raw = c_grad_torch - c_grad_cce
        c_grad_diff = c_grad_diff_raw.abs().max().item()

        # Find the sign of the max absolute difference
        c_grad_max_idx = c_grad_diff_raw.abs().argmax()
        c_grad_sign = "+" if c_grad_diff_raw.flatten()[c_grad_max_idx] >= 0 else "-"

        # E grad diff is not meaningful since we're using different E batches
        e_grad_diff = 0.0
        e_grad_sign = "N"

        if (
            use_bias
            and bias_torch is not None
            and bias_cce is not None
            and bias_grad_torch is not None
            and bias_grad_cce is not None
        ):
            bias_grad_diff_raw = bias_grad_torch - bias_grad_cce
            bias_grad_diff = bias_grad_diff_raw.abs().max().item()
            bias_grad_max_idx = bias_grad_diff_raw.abs().argmax()
            bias_grad_sign = "+" if bias_grad_diff_raw.flatten()[bias_grad_max_idx] >= 0 else "-"
        else:
            bias_grad_diff = 0.0
            bias_grad_sign = "+"

        # Compare parameters (only C and bias, since E are different batches)
        c_param_diff = (c_torch - c_cce).abs().max().item()
        if use_bias and bias_torch is not None and bias_cce is not None:
            bias_param_diff = (bias_torch - bias_cce).abs().max().item()
        else:
            bias_param_diff = 0.0

        # E param diff is not meaningful since we're using different E batches
        e_param_diff = 0.0

        # Compare loss components
        ce_diff = ce_loss_torch.item() - ce_loss_cce.item()
        z_diff = z_loss_torch.item() - z_loss_cce.item()

        # Update min/max tracking
        ce_diff_min = min(ce_diff_min, ce_diff)
        ce_diff_max = max(ce_diff_max, ce_diff)
        z_diff_min = min(z_diff_min, z_diff)
        z_diff_max = max(z_diff_max, z_diff)

        # Update gradient difference averages (only C gradients are meaningful)
        c_grad_diff_sum += c_grad_diff
        c_grad_diff_raw_sum += c_grad_diff_raw.mean().item()
        grad_diff_count += 1

        # Update loss difference averages
        ce_diff_sum += abs(ce_diff)
        z_diff_sum += abs(z_diff)
        loss_diff_count += 1

        if step % 10 == 0:
            print(
                f"{step:4d} |  {total_loss_torch.item():10.6f} | {total_loss_cce.item():10.6f} | "
                f"{z_loss_torch.item():10.6f} | {z_loss_cce.item():8.6f} | "
                f"{ce_diff:8.6f} | {z_diff:8.6f} | "
                f"{e_grad_sign}{e_grad_diff:10.6f} | {c_grad_sign}{c_grad_diff:10.6f} |{bias_grad_sign}{bias_grad_diff:10.6f}| {e_param_diff:11.6f} | {c_param_diff:11.6f} |{bias_param_diff:11.6f}"
            )

        # Break if differences get too large
        if c_grad_diff > 0.1:
            print(f"Large gradient differences detected at step {step}, stopping early")
            break

    final_c_diff = (c_torch - c_cce).abs().max().item()
    if use_bias and bias_torch is not None and bias_cce is not None:
        final_bias_diff = (bias_torch - bias_cce).abs().max().item()
    else:
        final_bias_diff = 0.0

    print("\nFinal parameter differences:")
    print("E: N/A (different batches)")
    print(f"C: {final_c_diff:.8f}")
    if use_bias:
        print(f"Bias: {final_bias_diff:.8f}")
    else:
        print("Bias: N/A (disabled)")

    print("\nLoss difference ranges over training:")
    print(
        f"CE Diff - Min: {ce_diff_min:.8f}, Max: {ce_diff_max:.8f}, Range: {ce_diff_max - ce_diff_min:.8f}"
    )
    print(
        f"Z Diff  - Min: {z_diff_min:.8f}, Max: {z_diff_max:.8f}, Range: {z_diff_max - z_diff_min:.8f}"
    )

    print("\nAverage gradient differences over training:")
    c_grad_diff_avg = c_grad_diff_sum / grad_diff_count if grad_diff_count > 0 else 0.0
    c_grad_diff_raw_avg = c_grad_diff_raw_sum / grad_diff_count if grad_diff_count > 0 else 0.0
    print("E Grad Diff Avg (abs max): N/A (different batches)")
    print(f"C Grad Diff Avg (abs max): {c_grad_diff_avg:.8f}")
    print("E Grad Diff Avg (raw):     N/A (different batches)")
    print(f"C Grad Diff Avg (raw):     {c_grad_diff_raw_avg:.8f}")

    print("\nAverage loss differences over training:")
    ce_diff_avg = ce_diff_sum / loss_diff_count if loss_diff_count > 0 else 0.0
    z_diff_avg = z_diff_sum / loss_diff_count if loss_diff_count > 0 else 0.0
    print(f"CE Diff Avg (abs): {ce_diff_avg:.8f}")
    print(f"Z Diff Avg (abs):  {z_diff_avg:.8f}")
