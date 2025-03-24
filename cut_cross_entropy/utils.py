# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import torch


@torch.compile(fullgraph=True, dynamic=True)
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
