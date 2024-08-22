import typing

from numpy import dtype, float32, float64, int8, int16, int32, int64, ndarray as ndarray, uint8, uint16, uint32, uint64

from scipy._typing import Untyped

Device: Untyped
Dtype: typing.TypeAlias = dtype[int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64 | float32 | float64]
Dtype = dtype
