# This module is not meant for public use and will be removed in SciPy v2.0.0.
from typing_extensions import deprecated

__all__ = ["dct", "dctn", "dst", "dstn", "idct", "idctn", "idst", "idstn"]

@deprecated("will be removed in SciPy v2.0.0")
def dctn(
    x: object,
    type: object = ...,
    shape: object = ...,
    axes: object = ...,
    norm: object = ...,
    overwrite_x: object = ...,
) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def idctn(
    x: object,
    type: object = ...,
    shape: object = ...,
    axes: object = ...,
    norm: object = ...,
    overwrite_x: object = ...,
) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def dstn(
    x: object,
    type: object = ...,
    shape: object = ...,
    axes: object = ...,
    norm: object = ...,
    overwrite_x: object = ...,
) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def idstn(
    x: object,
    type: object = ...,
    shape: object = ...,
    axes: object = ...,
    norm: object = ...,
    overwrite_x: object = ...,
) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def dct(
    x: object, type: object = ..., n: object = ..., axis: object = ..., norm: object = ..., overwrite_x: object = ...
) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def idct(
    x: object, type: object = ..., n: object = ..., axis: object = ..., norm: object = ..., overwrite_x: object = ...
) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def dst(
    x: object, type: object = ..., n: object = ..., axis: object = ..., norm: object = ..., overwrite_x: object = ...
) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def idst(
    x: object, type: object = ..., n: object = ..., axis: object = ..., norm: object = ..., overwrite_x: object = ...
) -> object: ...
