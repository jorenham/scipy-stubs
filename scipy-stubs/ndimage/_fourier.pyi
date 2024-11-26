from typing import TypeVar, overload

import optype.numpy as onp
from ._typing import _ComplexArrayOut, _FloatArrayOut

__all__ = ["fourier_ellipsoid", "fourier_gaussian", "fourier_shift", "fourier_uniform"]

_FloatArrayOutT = TypeVar("_FloatArrayOutT", bound=_FloatArrayOut)
_ComplexArrayOutT = TypeVar("_ComplexArrayOutT", bound=_ComplexArrayOut)

#
@overload
def fourier_gaussian(
    input: _FloatArrayOutT | onp.ToFloat | onp.ToFloatND,
    sigma: onp.ToFloat | onp.ToFloatND,
    n: onp.ToInt = -1,
    axis: onp.ToInt = -1,
    output: _FloatArrayOutT | None = None,
) -> _FloatArrayOutT: ...
@overload
def fourier_gaussian(
    input: _ComplexArrayOutT | onp.ToComplex | onp.ToComplexND,
    sigma: onp.ToFloat | onp.ToFloatND,
    n: onp.ToInt = -1,
    axis: onp.ToInt = -1,
    output: _ComplexArrayOutT | None = None,
) -> _ComplexArrayOutT: ...

#
@overload
def fourier_uniform(
    input: _FloatArrayOutT | onp.ToFloat | onp.ToFloatND,
    size: onp.ToFloat | onp.ToFloatND,
    n: onp.ToInt = -1,
    axis: onp.ToInt = -1,
    output: _FloatArrayOutT | None = None,
) -> _FloatArrayOutT: ...
@overload
def fourier_uniform(
    input: _ComplexArrayOutT | onp.ToComplex | onp.ToComplexND,
    size: onp.ToFloat | onp.ToFloatND,
    n: onp.ToInt = -1,
    axis: onp.ToInt = -1,
    output: _ComplexArrayOutT | None = None,
) -> _ComplexArrayOutT: ...

#
@overload
def fourier_ellipsoid(
    input: _FloatArrayOutT | onp.ToFloat | onp.ToFloatND,
    size: onp.ToFloat | onp.ToFloatND,
    n: onp.ToInt = -1,
    axis: onp.ToInt = -1,
    output: _FloatArrayOutT | None = None,
) -> _FloatArrayOutT: ...
@overload
def fourier_ellipsoid(
    input: _ComplexArrayOutT | onp.ToComplex | onp.ToComplexND,
    size: onp.ToFloat | onp.ToFloatND,
    n: onp.ToInt = -1,
    axis: onp.ToInt = -1,
    output: _ComplexArrayOutT | None = None,
) -> _ComplexArrayOutT: ...

#
@overload
def fourier_shift(
    input: _FloatArrayOutT | onp.ToFloat | onp.ToFloatND,
    shift: onp.ToFloat | onp.ToFloatND,
    n: onp.ToInt = -1,
    axis: onp.ToInt = -1,
    output: _FloatArrayOutT | None = None,
) -> _FloatArrayOutT: ...
@overload
def fourier_shift(
    input: _ComplexArrayOutT | onp.ToComplex | onp.ToComplexND,
    shift: onp.ToFloat | onp.ToFloatND,
    n: onp.ToInt = -1,
    axis: onp.ToInt = -1,
    output: _ComplexArrayOutT | None = None,
) -> _ComplexArrayOutT: ...
