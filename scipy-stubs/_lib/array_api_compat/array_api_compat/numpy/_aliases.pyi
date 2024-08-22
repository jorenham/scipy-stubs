import numpy as np

from .._internal import get_xp as get_xp
from ._typing import (
    Device as Device,
    Dtype as Dtype,
    NestedSequence as NestedSequence,
    SupportsBufferProtocol as SupportsBufferProtocol,
    ndarray as ndarray,
)
from scipy._typing import Untyped

bool: Untyped
acos: Untyped
acosh: Untyped
asin: Untyped
asinh: Untyped
atan: Untyped
atan2: Untyped
atanh: Untyped
bitwise_left_shift: Untyped
bitwise_invert: Untyped
bitwise_right_shift: Untyped
concat: Untyped
pow: Untyped
arange: Untyped
empty: Untyped
empty_like: Untyped
eye: Untyped
full: Untyped
full_like: Untyped
linspace: Untyped
ones: Untyped
ones_like: Untyped
zeros: Untyped
zeros_like: Untyped
UniqueAllResult: Untyped
UniqueCountsResult: Untyped
UniqueInverseResult: Untyped
unique_all: Untyped
unique_counts: Untyped
unique_inverse: Untyped
unique_values: Untyped
astype: Untyped
std: Untyped
var: Untyped
clip: Untyped
permute_dims: Untyped
reshape: Untyped
argsort: Untyped
sort: Untyped
nonzero: Untyped
sum: Untyped
prod: Untyped
ceil: Untyped
floor: Untyped
trunc: Untyped
matmul: Untyped
matrix_transpose: Untyped
tensordot: Untyped

def asarray(
    obj: ndarray | bool | int | float | NestedSequence[bool | int | float] | SupportsBufferProtocol,
    /,
    *,
    dtype: Dtype | None = None,
    device: Device | None = None,
    copy: bool | np._CopyMode | None = None,
    **kwargs,
) -> ndarray: ...

vecdot: Untyped
isdtype: Untyped
