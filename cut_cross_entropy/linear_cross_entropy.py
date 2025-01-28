# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import enum
import platform
from enum import auto
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from cut_cross_entropy.constants import IGNORE_INDEX
from cut_cross_entropy.doc import CCE_OPTS_DOC, LINEAR_CROSS_ENTROPY_DOC, add_doc_start
from cut_cross_entropy.torch_compile import torch_compile_linear_cross_entropy


class LinearCrossEntropyImpl(enum.IntEnum):
    CCE = auto()
    TORCH_COMPILE = auto()
    CCE_EXACT = auto()


PLATFORM_SYSTEM = platform.system()

if TYPE_CHECKING or PLATFORM_SYSTEM != "Darwin":
    from cut_cross_entropy.cce import cce_linear_cross_entropy

    LCE_IMPL_DEFAULT = LinearCrossEntropyImpl.CCE
else:
    cce_linear_cross_entropy = None
    LCE_IMPL_DEFAULT = LinearCrossEntropyImpl.TORCH_COMPILE


@add_doc_start(LINEAR_CROSS_ENTROPY_DOC)
@add_doc_start(*(doc_str + " Only valid for the cce implementation.\n" for doc_str in CCE_OPTS_DOC))
def linear_cross_entropy(
    e: torch.Tensor,
    c: torch.Tensor,
    targets: torch.Tensor,
    bias: torch.Tensor | None = None,
    ignore_index: int = IGNORE_INDEX,
    softcap: float | None = None,
    reduction: str = "mean",
    shift: bool = False,
    filter_eps: float | str | None = "auto",
    use_kahan: bool = False,
    impl: str | LinearCrossEntropyImpl = LCE_IMPL_DEFAULT,
) -> torch.Tensor:
    """
    :param impl: The linear cross entropy implementation to use. Currently supports cce, torch_compile, and cce_exact.
    """

    if isinstance(impl, LinearCrossEntropyImpl):
        impl = impl.name.lower()

    match impl:
        case "cce" | "cce_exact":
            if platform.system() == "Darwin":
                raise RuntimeError(
                    "CCE does not support MacOS. Please use torch_compile when running on MacOS instead."
                )

            if impl == "cce_exact":
                filter_eps = None
                use_kahan = True

            assert cce_linear_cross_entropy is not None
            return cce_linear_cross_entropy(
                e, c, targets, bias, ignore_index, softcap, reduction, shift, filter_eps, use_kahan
            )
        case "torch_compile":
            return torch_compile_linear_cross_entropy(
                e, c, targets, bias, ignore_index, softcap, reduction, shift
            )
        case _:
            raise NotImplementedError(f"{impl} is not implemented.")


class LinearCrossEntropy(nn.Module):
    def __init__(
        self,
        ignore_index: int = IGNORE_INDEX,
        softcap: float | None = None,
        reduction: str = "mean",
        shift: bool = False,
        filter_eps: float | str | None = "auto",
        use_kahan: bool = False,
        impl: str | LinearCrossEntropyImpl = LCE_IMPL_DEFAULT,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.softcap = softcap
        self.reduction = reduction
        self.filter_eps = filter_eps
        self.shift = shift
        self.use_kahan = use_kahan

        self.impl = impl

    def forward(self, e: torch.Tensor, c: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return linear_cross_entropy(
            e,
            c,
            targets,
            self.ignore_index,
            self.softcap,
            reduction=self.reduction,
            shift=self.shift,
            filter_eps=self.filter_eps,
            use_kahan=self.use_kahan,
            impl=self.impl,
        )
