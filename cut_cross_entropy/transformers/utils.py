# Copyright (C) 2024 Apple Inc. All Rights Reserved.
from dataclasses import dataclass
from typing import Any, TypeVar

import transformers

TransformersModelT = TypeVar("TransformersModelT", bound=transformers.PreTrainedModel)


@dataclass
class PatchOptions:
    impl: str
    reduction: str
    use_kahan: bool

    def to_kwargs(self) -> dict[str, Any]:
        return dict(impl=self.impl, reduction=self.reduction, use_kahan=self.use_kahan)
