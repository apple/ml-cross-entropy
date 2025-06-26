# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import functools
import importlib.metadata
from dataclasses import dataclass

import packaging.version
import torch

from cut_cross_entropy.constants import IGNORE_INDEX


@torch.compile(fullgraph=True)
def softcapping(logits: torch.Tensor, softcap: float) -> torch.Tensor:
    return torch.tanh(logits / softcap) * softcap


def _handle_eps(filter_eps: float | str | None, dtype: torch.dtype) -> float | None:
    if filter_eps is None:
        return None
    elif isinstance(filter_eps, float):
        return filter_eps
    elif filter_eps == "auto":
        return torch.finfo(dtype).eps / 32
    else:
        raise RuntimeError(f"Unknown eps {filter_eps=}")


def _build_flat_valids(
    targets: torch.Tensor,
    ignore_index: int,
    shift: int,
) -> torch.Tensor | None:
    if shift != 0:
        targets = targets[..., shift:]
    else:
        targets = targets.flatten()

    valids = (targets != ignore_index).nonzero().to(torch.int32)

    if shift == 0:
        assert valids.size(1) == 1
        return valids.squeeze(1) if valids.numel() != targets.numel() else None

    for i in range(targets.ndim - 1):
        valids[:, i] *= targets.stride(i)

    assert targets.stride(-1) == 1

    return valids.sum(1)


def handle_reduction_none(
    batch_shape: torch.Size, valids: torch.Tensor | None, shift: int, loss: torch.Tensor
) -> torch.Tensor:
    if valids is None:
        return loss.view(batch_shape)

    full_loss = loss.new_zeros((batch_shape.numel(),))
    full_loss[(valids + shift) if shift != 0 else valids] = loss

    return full_loss.view(batch_shape)


@torch.compile(fullgraph=True)
def compute_z_loss(
    lse: torch.Tensor,
    targets: torch.Tensor | None = None,
    shift: bool | int = False,
    ignore_index: int = IGNORE_INDEX,
    reduction: str = "mean",
) -> torch.Tensor:
    """Computes Z Loss.

    Specifically it computes z_loss = mean(||lse||_2^2).

    Providing the targets/shift/ignore index is used to mask out the loss for ignored tokens.
    """

    z_loss = lse.pow(2)

    if targets is not None:
        shift = int(shift)
        if shift != 0:
            targets = targets[..., shift:]

        is_not_ignore_index = targets != ignore_index

        z_loss = torch.where(is_not_ignore_index, z_loss, 0.0)

        if reduction == "mean":
            z_loss *= z_loss.numel() / is_not_ignore_index.count_nonzero().type_as(z_loss)

    if reduction == "mean":
        z_loss = z_loss.mean()
    elif reduction == "sum":
        z_loss = z_loss.sum()
    elif reduction == "none":
        pass
    else:
        raise ValueError(f"Invalid reduction: {reduction}")

    return z_loss


@functools.cache
def is_package_greater_or_equal(package: str, version: str) -> bool:
    return packaging.version.parse(importlib.metadata.version(package)) >= packaging.version.parse(
        version
    )


@functools.cache
def is_torch_greater_or_equal_2_5() -> bool:
    return is_package_greater_or_equal("torch", "2.5")


@dataclass
class TensorInfo:
    dtype: torch.dtype
    requires_grad: bool
