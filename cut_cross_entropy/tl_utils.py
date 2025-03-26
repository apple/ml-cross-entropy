# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import functools

import triton
import triton.language as tl
from triton.language.extra import libdevice as tl_libdevice

from cut_cross_entropy.utils import is_package_greater_or_equal


@triton.jit
def tl_and_reduce_fn(a, b):
    return a & b


@triton.jit
def tl_tanh(a: tl.tensor) -> tl.tensor:
    return tl_libdevice.tanh(a)


@triton.jit
def tl_log1p(a: tl.tensor) -> tl.tensor:
    return tl_libdevice.log1p(a)


@triton.jit
def tl_softcapping(v: tl.tensor, softcap: float) -> tl.tensor:
    return tl_tanh(v / softcap) * softcap


@triton.jit
def tl_softcapping_grad(dv: tl.tensor, v: tl.tensor, softcap: float) -> tl.tensor:
    v = v / softcap
    return dv * (1 - v * v)


@triton.jit
def tl_logaddexp(a, b) -> tl.tensor:
    minx = tl.minimum(a, b)
    mx = tl.maximum(a, b)
    return tl_log1p(tl.exp(minx - mx)) + mx


@triton.jit
def tl_2sum(a: tl.tensor, b: tl.tensor) -> tuple[tl.tensor, tl.tensor]:
    s = a + b

    a_prime = s - b
    b_prime = s - a_prime

    delta_a = a - a_prime
    delta_b = b - b_prime

    t = delta_a + delta_b
    return s, t


@triton.jit
def tl_lock_kahan_sum(ptrs, c_ptrs, v, mask, lock_ptr):
    while tl.atomic_cas(lock_ptr, 0, 1) == 1:
        pass

    s = tl.load(ptrs, mask=mask, other=0.0, eviction_policy="evict_last")
    c = tl.load(c_ptrs, mask=mask, other=0.0, eviction_policy="evict_last")

    s, c = tl_2sum(s, c + v)

    tl.store(ptrs, s, mask=mask, eviction_policy="evict_last")
    tl.store(c_ptrs, c, mask=mask, eviction_policy="evict_last")

    tl.debug_barrier()
    tl.atomic_xchg(lock_ptr, 0)


@triton.jit
def tl_lock_add(ptrs, v, mask, lock_ptr):
    while tl.atomic_cas(lock_ptr, 0, 1) == 1:
        pass

    cur_v = tl.load(ptrs, mask=mask, other=0.0, eviction_policy="evict_last")
    new_v = v + cur_v
    tl.store(ptrs, new_v, mask=mask, eviction_policy="evict_last")

    tl.debug_barrier()
    tl.atomic_xchg(lock_ptr, 0)


def b_bin_fn(b: int) -> int:
    if b >= 1024:
        return 1024
    elif b <= 128:
        return 128
    else:
        return 512


@functools.cache
def is_triton_greater_or_equal_3_2_0() -> bool:
    return is_package_greater_or_equal("triton", "3.2.0")
