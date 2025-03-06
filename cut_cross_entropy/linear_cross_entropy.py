# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import platform
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from cut_cross_entropy.cce_utils import CCEPreset, CCEPresets, LinearCrossEntropyImpl
from cut_cross_entropy.constants import IGNORE_INDEX
from cut_cross_entropy.doc import CCE_OPTS_DOC, IMPL_DOC, LINEAR_CROSS_ENTROPY_DOC, add_doc_start
from cut_cross_entropy.torch_compile import torch_compile_linear_cross_entropy

PLATFORM_SYSTEM = platform.system()

if TYPE_CHECKING or PLATFORM_SYSTEM != "Darwin":
    from cut_cross_entropy.cce import cce_linear_cross_entropy

    LCE_IMPL_DEFAULT = LinearCrossEntropyImpl.CCE
else:
    cce_linear_cross_entropy = None
    LCE_IMPL_DEFAULT = LinearCrossEntropyImpl.TORCH_COMPILE


@add_doc_start(LINEAR_CROSS_ENTROPY_DOC)
@add_doc_start(*(doc_str + " Only valid for the cce implementation.\n" for doc_str in CCE_OPTS_DOC))
@add_doc_start(IMPL_DOC)
def linear_cross_entropy(
    e: torch.Tensor,
    c: torch.Tensor,
    targets: torch.Tensor,
    bias: torch.Tensor | None = None,
    ignore_index: int = IGNORE_INDEX,
    softcap: float | None = None,
    reduction: str = "mean",
    shift: bool | int = 0,
    filter_eps: float | str | None = "auto",
    accum_e_fp32: bool = False,
    accum_c_fp32: bool = False,
    filter_e_grad: bool = True,
    filter_c_grad: bool = True,
    impl: str | LinearCrossEntropyImpl = LCE_IMPL_DEFAULT,
) -> torch.Tensor:
    """
    :param impl: The linear cross entropy implementation to use. Currently supports cce, torch_compile, and cce_exact.
    """

    if isinstance(impl, LinearCrossEntropyImpl):
        impl = impl.name.lower()

    if isinstance(shift, int) and (shift < 0 or shift >= targets.size(-1)):
        raise ValueError(f"Shift must be in the range [0, {targets.size(-1)}). Got {shift}.")

    if impl in CCEPresets.names:
        if platform.system() == "Darwin":
            raise RuntimeError(
                "CCE does not support MacOS. Please use torch_compile when running on MacOS instead."
            )

        cce_opts = CCEPresets.handle(
            impl,
            CCEPreset(
                filter_eps=filter_eps,
                accum_e_fp32=accum_e_fp32,
                accum_c_fp32=accum_c_fp32,
                filter_e_grad=filter_e_grad,
                filter_c_grad=filter_c_grad,
            ),
        )

        assert cce_linear_cross_entropy is not None
        return cce_linear_cross_entropy(
            e,
            c,
            targets,
            bias,
            ignore_index,
            softcap,
            reduction,
            shift,
            **cce_opts,
        )
    elif impl == "torch_compile":
        return torch_compile_linear_cross_entropy(
            e, c, targets, bias, ignore_index, softcap, reduction, shift
        )
    else:
        raise NotImplementedError(f"{impl} is not implemented.")


class LinearCrossEntropy(nn.Module):
    def __init__(
        self,
        ignore_index: int = IGNORE_INDEX,
        softcap: float | None = None,
        reduction: str = "mean",
        shift: bool | int = 0,
        filter_eps: float | str | None = "auto",
        accum_e_fp32: bool = False,
        accum_c_fp32: bool = False,
        filter_e_grad: bool = True,
        filter_c_grad: bool = True,
        impl: str | LinearCrossEntropyImpl = LCE_IMPL_DEFAULT,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.softcap = softcap
        self.reduction = reduction
        self.filter_eps = filter_eps
        self.shift = shift

        self.accum_e_fp32 = accum_e_fp32
        self.accum_c_fp32 = accum_c_fp32

        self.filter_e_grad = filter_e_grad
        self.filter_c_grad = filter_c_grad

        self.impl = impl

    def forward(
        self,
        e: torch.Tensor,
        c: torch.Tensor,
        targets: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return linear_cross_entropy(
            e,
            c,
            targets,
            bias=bias,
            ignore_index=self.ignore_index,
            softcap=self.softcap,
            reduction=self.reduction,
            shift=self.shift,
            filter_eps=self.filter_eps,
            accum_e_fp32=self.accum_e_fp32,
            accum_c_fp32=self.accum_c_fp32,
            filter_e_grad=self.filter_e_grad,
            filter_c_grad=self.filter_c_grad,
            impl=self.impl,
        )
