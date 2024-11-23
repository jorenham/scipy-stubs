from typing import Literal, overload

import numpy as np
import optype.numpy as onp

__all__ = ["log_softmax", "logsumexp", "softmax"]

# TODO: Support `return_sign=True`
@overload
def logsumexp(
    a: onp.ToFloat,
    axis: int | tuple[int, ...] | None = None,
    b: onp.ToFloat | None = None,
    keepdims: bool = False,
    return_sign: Literal[False, 0] = False,
) -> np.float64: ...
@overload
def logsumexp(
    a: onp.ToComplex,
    axis: int | tuple[int, ...] | None = None,
    b: onp.ToFloat | None = None,
    keepdims: bool = False,
    return_sign: Literal[False, 0] = False,
) -> np.float64 | np.complex128: ...
@overload
def logsumexp(
    a: onp.ToFloatND,
    axis: int | tuple[int, ...],
    b: onp.ToFloat | onp.ToFloatND | None = None,
    keepdims: bool = False,
    return_sign: Literal[False, 0] = False,
) -> np.float64 | onp.ArrayND[np.float64]: ...
@overload
def logsumexp(
    a: onp.ToComplexND,
    axis: int | tuple[int, ...],
    b: onp.ToFloat | onp.ToFloatND | None = None,
    keepdims: bool = False,
    return_sign: Literal[False, 0] = False,
) -> np.float64 | np.complex128 | onp.ArrayND[np.float64 | np.complex128]: ...

# TODO: Overload real/complex and scalar/array
def softmax(
    x: onp.ToComplex | onp.ToComplexND,
    axis: int | tuple[int, ...] | None = None,
) -> np.float64 | np.complex128 | onp.ArrayND[np.float64 | np.complex128]: ...
def log_softmax(
    x: onp.ToComplex | onp.ToComplexND,
    axis: int | tuple[int, ...] | None = None,
) -> np.float64 | np.complex128 | onp.ArrayND[np.float64 | np.complex128]: ...
