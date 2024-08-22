from collections.abc import Sequence
from typing import NamedTuple

from ._helpers import array_namespace as array_namespace
from ._typing import Device as Device, Dtype as Dtype, ndarray as ndarray

def arange(
    start: int | float,
    /,
    stop: int | float | None = None,
    step: int | float = 1,
    *,
    xp,
    dtype: Dtype | None = None,
    device: Device | None = None,
    **kwargs,
) -> ndarray: ...
def empty(shape: int | tuple[int, ...], xp, *, dtype: Dtype | None = None, device: Device | None = None, **kwargs) -> ndarray: ...
def empty_like(x: ndarray, /, xp, *, dtype: Dtype | None = None, device: Device | None = None, **kwargs) -> ndarray: ...
def eye(
    n_rows: int,
    n_cols: int | None = None,
    /,
    *,
    xp,
    k: int = 0,
    dtype: Dtype | None = None,
    device: Device | None = None,
    **kwargs,
) -> ndarray: ...
def full(
    shape: int | tuple[int, ...],
    fill_value: int | float,
    xp,
    *,
    dtype: Dtype | None = None,
    device: Device | None = None,
    **kwargs,
) -> ndarray: ...
def full_like(
    x: ndarray, /, fill_value: int | float, *, xp, dtype: Dtype | None = None, device: Device | None = None, **kwargs
) -> ndarray: ...
def linspace(
    start: int | float,
    stop: int | float,
    /,
    num: int,
    *,
    xp,
    dtype: Dtype | None = None,
    device: Device | None = None,
    endpoint: bool = True,
    **kwargs,
) -> ndarray: ...
def ones(shape: int | tuple[int, ...], xp, *, dtype: Dtype | None = None, device: Device | None = None, **kwargs) -> ndarray: ...
def ones_like(x: ndarray, /, xp, *, dtype: Dtype | None = None, device: Device | None = None, **kwargs) -> ndarray: ...
def zeros(shape: int | tuple[int, ...], xp, *, dtype: Dtype | None = None, device: Device | None = None, **kwargs) -> ndarray: ...
def zeros_like(x: ndarray, /, xp, *, dtype: Dtype | None = None, device: Device | None = None, **kwargs) -> ndarray: ...

class UniqueAllResult(NamedTuple):
    values: ndarray
    indices: ndarray
    inverse_indices: ndarray
    counts: ndarray

class UniqueCountsResult(NamedTuple):
    values: ndarray
    counts: ndarray

class UniqueInverseResult(NamedTuple):
    values: ndarray
    inverse_indices: ndarray

def unique_all(x: ndarray, /, xp) -> UniqueAllResult: ...
def unique_counts(x: ndarray, /, xp) -> UniqueCountsResult: ...
def unique_inverse(x: ndarray, /, xp) -> UniqueInverseResult: ...
def unique_values(x: ndarray, /, xp) -> ndarray: ...
def astype(x: ndarray, dtype: Dtype, /, *, copy: bool = True) -> ndarray: ...
def std(
    x: ndarray,
    /,
    xp,
    *,
    axis: int | tuple[int, ...] | None = None,
    correction: int | float = 0.0,
    keepdims: bool = False,
    **kwargs,
) -> ndarray: ...
def var(
    x: ndarray,
    /,
    xp,
    *,
    axis: int | tuple[int, ...] | None = None,
    correction: int | float = 0.0,
    keepdims: bool = False,
    **kwargs,
) -> ndarray: ...
def clip(
    x: ndarray,
    /,
    min: int | float | ndarray | None = None,
    max: int | float | ndarray | None = None,
    *,
    xp,
    out: ndarray | None = None,
) -> ndarray: ...
def permute_dims(x: ndarray, /, axes: tuple[int, ...], xp) -> ndarray: ...
def reshape(x: ndarray, /, shape: tuple[int, ...], xp, copy: bool | None = None, **kwargs) -> ndarray: ...
def argsort(x: ndarray, /, xp, *, axis: int = -1, descending: bool = False, stable: bool = True, **kwargs) -> ndarray: ...
def sort(x: ndarray, /, xp, *, axis: int = -1, descending: bool = False, stable: bool = True, **kwargs) -> ndarray: ...
def nonzero(x: ndarray, /, xp, **kwargs) -> tuple[ndarray, ...]: ...
def sum(
    x: ndarray, /, xp, *, axis: int | tuple[int, ...] | None = None, dtype: Dtype | None = None, keepdims: bool = False, **kwargs
) -> ndarray: ...
def prod(
    x: ndarray, /, xp, *, axis: int | tuple[int, ...] | None = None, dtype: Dtype | None = None, keepdims: bool = False, **kwargs
) -> ndarray: ...
def ceil(x: ndarray, /, xp, **kwargs) -> ndarray: ...
def floor(x: ndarray, /, xp, **kwargs) -> ndarray: ...
def trunc(x: ndarray, /, xp, **kwargs) -> ndarray: ...
def matmul(x1: ndarray, x2: ndarray, /, xp, **kwargs) -> ndarray: ...
def matrix_transpose(x: ndarray, /, xp) -> ndarray: ...
def tensordot(x1: ndarray, x2: ndarray, /, xp, *, axes: int | tuple[Sequence[int], Sequence[int]] = 2, **kwargs) -> ndarray: ...
def vecdot(x1: ndarray, x2: ndarray, /, xp, *, axis: int = -1) -> ndarray: ...
def isdtype(dtype: Dtype, kind: Dtype | str | tuple[Dtype | str, ...], xp, *, _tuple: bool = True) -> bool: ...
