import pytest
import torch

import cut_cross_entropy.constants
from cut_cross_entropy.utils import compute_z_loss


@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
@pytest.mark.parametrize("shift", [False, True])
@pytest.mark.parametrize("invalids", [False, True])
def test_compute_z_loss(reduction: str, invalids: bool, shift: bool):
    torch._dynamo.config.cache_size_limit = 256
    lse = torch.arange(1, 11, dtype=torch.float64)

    if invalids:
        targets_shape = lse.size(0) + int(shift)
        targets = torch.full(
            (targets_shape,), cut_cross_entropy.constants.IGNORE_INDEX, dtype=torch.int64
        )
        targets[3] = 1
        targets[5] = 1
    else:
        targets = None

    z_loss = compute_z_loss(lse, shift=shift, targets=targets, reduction=reduction)

    if reduction == "none":
        assert z_loss.shape == lse.shape
    elif reduction == "mean":
        assert z_loss.shape == torch.Size([])
    elif reduction == "sum":
        assert z_loss.shape == torch.Size([])

    if reduction == "mean":
        z_loss *= lse.numel() if targets is None else 2

    z_loss = z_loss.sum()

    if targets is None:
        expected_z_loss = sum(i**2 for i in range(1, 11))
    else:
        expected_z_loss = (4 - int(shift)) ** 2 + (6 - int(shift)) ** 2

    assert torch.isclose(z_loss.sum(), torch.as_tensor(expected_z_loss, dtype=torch.float64))
