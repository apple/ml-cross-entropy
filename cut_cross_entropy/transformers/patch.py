# Copyright (C) 2024 Apple Inc. All Rights Reserved.
from typing import overload

import transformers

from cut_cross_entropy.cce_utils import LinearCrossEntropyImpl
from cut_cross_entropy.linear_cross_entropy import LCE_IMPL_DEFAULT

from .gemma2 import patch_gemma2
from .llama import patch_llama
from .mistral import patch_mistral
from .phi3 import patch_phi3
from .qwen2 import patch_qwen2
from .utils import PatchOptions, TransformersModelT

PATCH_FNS = {
    "llama": patch_llama,
    "phi3": patch_phi3,
    "gemma2": patch_gemma2,
    "mistral": patch_mistral,
    "qwen2": patch_qwen2,
}


@overload
def cce_patch(
    model_type_or_model: str | transformers.PretrainedConfig,
    impl: str | LinearCrossEntropyImpl = LCE_IMPL_DEFAULT,
    reduction: str = "mean",
    filter_eps: float | str | None = "auto",
    accum_e_fp32: bool = False,
    accum_c_fp32: bool = False,
    filter_e_grad: bool = True,
    filter_c_grad: bool = True,
    train_only: bool = False,
) -> None: ...


@overload
def cce_patch(
    model_type_or_model: TransformersModelT,
    impl: str | LinearCrossEntropyImpl = LCE_IMPL_DEFAULT,
    reduction: str = "mean",
    filter_eps: float | str | None = "auto",
    accum_e_fp32: bool = False,
    accum_c_fp32: bool = False,
    filter_e_grad: bool = True,
    filter_c_grad: bool = True,
    train_only: bool = False,
) -> TransformersModelT: ...


def cce_patch(
    model_type_or_model: str | TransformersModelT | transformers.PretrainedConfig,
    impl: str | LinearCrossEntropyImpl = LCE_IMPL_DEFAULT,
    reduction: str = "mean",
    filter_eps: float | str | None = "auto",
    accum_e_fp32: bool = False,
    accum_c_fp32: bool = False,
    filter_e_grad: bool = True,
    filter_c_grad: bool = True,
    train_only: bool = False,
) -> TransformersModelT | None:
    if isinstance(impl, LinearCrossEntropyImpl):
        impl = impl.name.lower()

    if impl not in (v.name.lower() for v in LinearCrossEntropyImpl):
        raise ValueError(f"Unknown {impl=}")

    if isinstance(model_type_or_model, transformers.PreTrainedModel):
        model_type = model_type_or_model.config.model_type
    elif isinstance(model_type_or_model, transformers.PretrainedConfig):
        model_type = model_type_or_model.model_type
    else:
        model_type = model_type_or_model

    patch_options = PatchOptions(
        impl=impl,
        reduction=reduction,
        filter_eps=filter_eps,
        accum_e_fp32=accum_e_fp32,
        accum_c_fp32=accum_c_fp32,
        filter_e_grad=filter_e_grad,
        filter_c_grad=filter_c_grad,
        train_only=train_only,
    )

    if model_type in PATCH_FNS:
        return PATCH_FNS[model_type](model_type_or_model, patch_options)
    else:
        raise RuntimeError(f"Unknown model type {model_type}")
