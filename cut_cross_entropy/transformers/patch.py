# Copyright (C) 2024 Apple Inc. All Rights Reserved.
from typing import overload

import transformers

from cut_cross_entropy.linear_cross_entropy import LCE_IMPL_DEFAULT, LinearCrossEntropyImpl

from .gemma2 import patch_gemma2
from .llama import patch_llama
from .mistral import patch_mistral
from .phi3 import patch_phi3
from .utils import PatchOptions, TransformersModelT


@overload
def cce_patch(
    model_type_or_model: str | transformers.PretrainedConfig,
    impl: str | LinearCrossEntropyImpl = LCE_IMPL_DEFAULT,
    reduction: str = "mean",
    use_kahan: bool = False,
) -> None: ...


@overload
def cce_patch(
    model_type_or_model: TransformersModelT,
    impl: str | LinearCrossEntropyImpl = LCE_IMPL_DEFAULT,
    reduction: str = "mean",
    use_kahan: bool = False,
) -> TransformersModelT: ...


def cce_patch(
    model_type_or_model: str | TransformersModelT | transformers.PretrainedConfig,
    impl: str | LinearCrossEntropyImpl = LCE_IMPL_DEFAULT,
    reduction: str = "mean",
    use_kahan: bool = False,
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

    patch_options = PatchOptions(impl, reduction, use_kahan)

    match model_type:
        case "llama":
            return patch_llama(model_type_or_model, patch_options)
        case "phi3":
            return patch_phi3(model_type_or_model, patch_options)
        case "gemma2":
            return patch_gemma2(model_type_or_model, patch_options)
        case "mistral":
            return patch_mistral(model_type_or_model, patch_options)
        case _:
            raise RuntimeError(f"Unknown model type {model_type}")
