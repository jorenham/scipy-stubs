from typing import overload

import numpy as np
import optype.numpy as onp
from scipy._typing import AnyShape, Falsy, Truthy

__all__ = ["log_softmax", "logsumexp", "softmax"]

@overload
def logsumexp(
    a: onp.ToFloat,
    axis: AnyShape | None = None,
    b: onp.ToFloat | None = None,
    keepdims: bool = False,
    return_sign: Falsy = False,
) -> np.float64: ...
@overload
def logsumexp(
    a: onp.ToComplex,
    axis: AnyShape | None = None,
    b: onp.ToFloat | None = None,
    keepdims: bool = False,
    return_sign: Falsy = False,
) -> np.float64 | np.complex128: ...
@overload
def logsumexp(
    a: onp.ToFloatND,
    axis: AnyShape,
    b: onp.ToFloat | onp.ToFloatND | None = None,
    keepdims: bool = False,
    return_sign: Falsy = False,
) -> np.float64 | onp.ArrayND[np.float64]: ...
@overload
def logsumexp(
    a: onp.ToComplexND,
    axis: AnyShape,
    b: onp.ToFloat | onp.ToFloatND | None = None,
    keepdims: bool = False,
    return_sign: Falsy = False,
) -> np.float64 | np.complex128 | onp.ArrayND[np.float64 | np.complex128]: ...
@overload
def logsumexp(
    a: onp.ToFloat,
    axis: AnyShape | None = None,
    b: onp.ToFloat | None = None,
    keepdims: bool = False,
    *,
    return_sign: Truthy,
) -> tuple[np.float64, bool | np.bool_]: ...
@overload
def logsumexp(
    a: onp.ToComplex,
    axis: AnyShape | None = None,
    b: onp.ToFloat | None = None,
    keepdims: bool = False,
    *,
    return_sign: Truthy,
) -> tuple[np.float64 | np.complex128, bool | np.bool_]: ...
@overload
def logsumexp(
    a: onp.ToFloatND,
    axis: AnyShape,
    b: onp.ToFloat | onp.ToFloatND | None = None,
    keepdims: bool = False,
    *,
    return_sign: Truthy,
) -> tuple[np.float64, bool | np.bool_] | tuple[onp.ArrayND[np.float64], onp.ArrayND[np.bool_]]: ...
@overload
def logsumexp(
    a: onp.ToComplexND,
    axis: AnyShape,
    b: onp.ToFloat | onp.ToFloatND | None = None,
    keepdims: bool = False,
    *,
    return_sign: Truthy,
) -> (
    tuple[np.float64 | np.complex128, bool | np.bool_] | tuple[onp.ArrayND[np.float64 | np.complex128], onp.ArrayND[np.bool_]]
): ...

#
@overload
def softmax(x: onp.ToFloat, axis: AnyShape | None = None) -> np.float64: ...
@overload
def softmax(x: onp.ToFloatND, axis: AnyShape | None = None) -> onp.ArrayND[np.float64]: ...
@overload
def softmax(x: onp.ToComplex, axis: AnyShape | None = None) -> np.float64 | np.complex128: ...
@overload
def softmax(x: onp.ToComplexND, axis: AnyShape | None = None) -> onp.ArrayND[np.float64 | np.complex128]: ...

#
@overload
def log_softmax(x: onp.ToFloat, axis: AnyShape | None = None) -> np.float64: ...
@overload
def log_softmax(x: onp.ToFloatND, axis: AnyShape | None = None) -> onp.ArrayND[np.float64]: ...
@overload
def log_softmax(x: onp.ToComplex, axis: AnyShape | None = None) -> np.float64 | np.complex128: ...
@overload
def log_softmax(x: onp.ToComplexND, axis: AnyShape | None = None) -> onp.ArrayND[np.float64 | np.complex128]: ...
