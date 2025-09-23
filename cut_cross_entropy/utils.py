# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import functools
import importlib.metadata
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast, overload

import packaging.version
import torch
from torch.autograd import Function

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
    batch_shape: torch.Size, valids: torch.Tensor | None, shift: int, value: torch.Tensor
) -> torch.Tensor:
    if valids is None:
        return value.view(batch_shape)

    full_value = value.new_zeros((batch_shape.numel(),))
    full_value[(valids + shift) if shift != 0 else valids] = value

    return full_value.view(batch_shape)


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


@functools.cache
def is_triton_3_2() -> bool:
    return packaging.version.parse(
        packaging.version.parse(importlib.metadata.version("triton")).base_version
    ) == packaging.version.parse("3.2")


@dataclass(slots=True)
class TensorInfo:
    dtype: torch.dtype
    requires_grad: bool


if TYPE_CHECKING or is_torch_greater_or_equal_2_5():
    import torch.distributed.tensor
    from torch.distributed.device_mesh import DeviceMesh


class _ToFullTensorAllReduceHook(Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, device_mesh: DeviceMesh) -> torch.Tensor:
        ctx.device_mesh = device_mesh
        return tensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:  # type: ignore
        device_mesh = cast("DeviceMesh", ctx.device_mesh)
        grad_output = grad_output.clone()
        grad_output.div_(device_mesh.size())

        for pg in device_mesh.get_all_groups():
            torch.distributed.all_reduce(grad_output, group=pg)

        return grad_output, None


@overload
def to_full_tensor(t: torch.Tensor) -> torch.Tensor: ...


@overload
def to_full_tensor(t: None) -> None: ...


def to_full_tensor(t: torch.Tensor | None) -> torch.Tensor | None:
    if isinstance(t, torch.distributed.tensor.DTensor):
        return _ToFullTensorAllReduceHook.apply(t.full_tensor(), t.device_mesh)
    else:
        return t


@overload
def maybe_type_as(t: torch.Tensor, other: torch.Tensor) -> torch.Tensor: ...


@overload
def maybe_type_as(t: None, other: torch.Tensor) -> None: ...


def maybe_type_as(t: torch.Tensor | None, other: torch.Tensor) -> torch.Tensor | None:
    if isinstance(t, torch.Tensor):
        return t.type_as(other)
    else:
        return None


class CCEWarning(Warning):
    pass
