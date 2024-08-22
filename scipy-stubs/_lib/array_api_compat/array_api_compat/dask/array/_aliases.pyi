import numpy as np
from numpy import (
    bool_ as bool,
    can_cast as can_cast,
    complex64 as complex64,
    complex128 as complex128,
    e as e,
    finfo as finfo,
    float32 as float32,
    float64 as float64,
    iinfo as iinfo,
    inf as inf,
    int8 as int8,
    int16 as int16,
    int32 as int32,
    int64 as int64,
    nan as nan,
    newaxis as newaxis,
    pi as pi,
    result_type as result_type,
    uint8 as uint8,
    uint16 as uint16,
    uint32 as uint32,
    uint64 as uint64,
)

from ..._internal import get_xp as get_xp
from ...common._typing import (
    Array as Array,
    Device as Device,
    Dtype as Dtype,
    NestedSequence as NestedSequence,
    SupportsBufferProtocol as SupportsBufferProtocol,
)
from scipy._typing import Untyped

isdtype: Untyped
astype: Untyped
arange: Untyped
eye: Untyped
linspace: Untyped
UniqueAllResult: Untyped
UniqueCountsResult: Untyped
UniqueInverseResult: Untyped
unique_all: Untyped
unique_counts: Untyped
unique_inverse: Untyped
unique_values: Untyped
permute_dims: Untyped
std: Untyped
var: Untyped
clip: Untyped
empty: Untyped
empty_like: Untyped
full: Untyped
full_like: Untyped
ones: Untyped
ones_like: Untyped
zeros: Untyped
zeros_like: Untyped
reshape: Untyped
matrix_transpose: Untyped
vecdot: Untyped
nonzero: Untyped
sum: Untyped
prod: Untyped
ceil: Untyped
floor: Untyped
trunc: Untyped
matmul: Untyped
tensordot: Untyped

def asarray(
    obj: Array | bool | int | float | NestedSequence[bool | int | float] | SupportsBufferProtocol,
    /,
    *,
    dtype: Dtype | None = None,
    device: Device | None = None,
    copy: bool | np._CopyMode | None = None,
    **kwargs,
) -> Array: ...

common_aliases: Untyped
