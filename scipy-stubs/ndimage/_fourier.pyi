from typing import overload

from ._typing import _ComplexArrayOutT, _FloatArrayIn, _FloatArrayOutT, _IntValueIn, _ScalarArrayIn

__all__ = ["fourier_ellipsoid", "fourier_gaussian", "fourier_shift", "fourier_uniform"]

#
@overload
def fourier_gaussian(
    input: _FloatArrayOutT | _FloatArrayIn,
    sigma: _FloatArrayIn,
    n: _IntValueIn = -1,
    axis: _IntValueIn = -1,
    output: _FloatArrayOutT | None = None,
) -> _FloatArrayOutT: ...
@overload
def fourier_gaussian(
    input: _ComplexArrayOutT | _ScalarArrayIn,
    sigma: _FloatArrayIn,
    n: _IntValueIn = -1,
    axis: _IntValueIn = -1,
    output: _ComplexArrayOutT | None = None,
) -> _ComplexArrayOutT: ...

#
@overload
def fourier_uniform(
    input: _FloatArrayOutT | _FloatArrayIn,
    size: _FloatArrayIn,
    n: _IntValueIn = -1,
    axis: _IntValueIn = -1,
    output: _FloatArrayOutT | None = None,
) -> _FloatArrayOutT: ...
@overload
def fourier_uniform(
    input: _ComplexArrayOutT | _ScalarArrayIn,
    size: _FloatArrayIn,
    n: _IntValueIn = -1,
    axis: _IntValueIn = -1,
    output: _ComplexArrayOutT | None = None,
) -> _ComplexArrayOutT: ...

#
@overload
def fourier_ellipsoid(
    input: _FloatArrayOutT | _FloatArrayIn,
    size: _FloatArrayIn,
    n: _IntValueIn = -1,
    axis: _IntValueIn = -1,
    output: _FloatArrayOutT | None = None,
) -> _FloatArrayOutT: ...
@overload
def fourier_ellipsoid(
    input: _ComplexArrayOutT | _ScalarArrayIn,
    size: _FloatArrayIn,
    n: _IntValueIn = -1,
    axis: _IntValueIn = -1,
    output: _ComplexArrayOutT | None = None,
) -> _ComplexArrayOutT: ...

#
@overload
def fourier_shift(
    input: _FloatArrayOutT | _FloatArrayIn,
    shift: _FloatArrayIn,
    n: _IntValueIn = -1,
    axis: _IntValueIn = -1,
    output: _FloatArrayOutT | None = None,
) -> _FloatArrayOutT: ...
@overload
def fourier_shift(
    input: _ComplexArrayOutT | _ScalarArrayIn,
    shift: _FloatArrayIn,
    n: _IntValueIn = -1,
    axis: _IntValueIn = -1,
    output: _ComplexArrayOutT | None = None,
) -> _ComplexArrayOutT: ...
