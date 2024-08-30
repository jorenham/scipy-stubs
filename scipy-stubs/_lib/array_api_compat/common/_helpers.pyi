from types import ModuleType
from typing_extensions import TypeIs

import numpy as np
from ._typing import Array, Device

__all__ = [
    "array_namespace",
    "device",
    "get_namespace",
    "is_array_api_obj",
    "is_cupy_array",
    "is_dask_array",
    "is_jax_array",
    "is_numpy_array",
    "is_torch_array",
    "size",
    "to_device",
]

def is_numpy_array(x: object) -> TypeIs[np.ndarray[tuple[int, ...], np.dtype[np.generic]] | np.generic]: ...
def is_cupy_array(x: object) -> bool: ...
def is_torch_array(x: object) -> bool: ...
def is_dask_array(x: object) -> bool: ...
def is_jax_array(x: object) -> bool: ...
def is_array_api_obj(x: object) -> bool: ...
def array_namespace(*xs: object, api_version: str | None = None) -> ModuleType: ...
def device(x: Array, /) -> Device: ...
def to_device(x: Array, device: Device, /, *, stream: int | object | None = None) -> Array: ...
def size(x: Array) -> int: ...

get_namespace = array_namespace
