# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import enum
from enum import auto
from typing import TypedDict


class LinearCrossEntropyImpl(enum.IntEnum):
    CCE = auto()
    TORCH_COMPILE = auto()
    CCE_KAHAN_FULL_C = auto()
    CCE_KAHAN_FULL_E = auto()

    CCE_EXACT = auto()
    CCE_KAHAN_FULL_C_FULL_E = auto()
    CCE_KAHAN_FULL = auto()


class CCEPreset(TypedDict):
    filter_eps: float | str | None
    accum_e_fp32: bool
    accum_c_fp32: bool
    filter_e_grad: bool
    filter_c_grad: bool


class CCEPresets:
    names: set[str] = set(
        v.name.lower() for v in LinearCrossEntropyImpl if v.name.lower() != "torch_compile"
    )

    @classmethod
    def handle(cls, impl: str, opts: CCEPreset) -> CCEPreset:
        if impl not in cls.names:
            raise ValueError(f"{impl!r} not in {cls.names}")

        if impl == "cce":
            return opts

        opts = opts.copy()
        if impl in ("cce_exact", "cce_kahan_full", "cce_kahan_full_c_full_e"):
            opts["filter_eps"] = None
            opts["accum_e_fp32"] = True
            opts["accum_c_fp32"] = True

            return opts

        if impl == "cce_kahan_full_c":
            opts["filter_eps"] = "auto"

            opts["accum_c_fp32"] = True
            opts["filter_c_grad"] = False

            opts["accum_e_fp32"] = True
            opts["filter_e_grad"] = True

            return opts

        if impl == "cce_kahan_full_e":
            opts["filter_eps"] = "auto"

            opts["accum_c_fp32"] = True
            opts["filter_c_grad"] = True

            opts["accum_e_fp32"] = True
            opts["filter_e_grad"] = False

            return opts

        raise NotImplementedError(f"{impl=}")
