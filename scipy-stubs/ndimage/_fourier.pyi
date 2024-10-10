from collections.abc import Sequence
from typing import Any, TypeAlias, overload
from typing_extensions import TypeVar

import numpy as np
import numpy.typing as npt
from numpy._typing import _ArrayLikeComplex_co, _SupportsArray

__all__ = ["fourier_ellipsoid", "fourier_gaussian", "fourier_shift", "fourier_uniform"]

#
_IntValueIn: TypeAlias = int | np.integer[Any]
_FloatArrayIn: TypeAlias = float | _SupportsArray[np.dtype[np.floating[Any] | np.integer[Any]]] | Sequence[_FloatArrayIn]

_FloatValueOut: TypeAlias = np.float64 | np.float32 | np.double | np.single
_FloatArrayOut: TypeAlias = npt.NDArray[_FloatValueOut]
_FloatArrayOutT = TypeVar("_FloatArrayOutT", bound=_FloatArrayOut, default=_FloatArrayOut)

_ComplexValueOut: TypeAlias = _FloatValueOut | np.complex128 | np.complex64 | np.cdouble | np.csingle
_ComplexArrayOut: TypeAlias = npt.NDArray[_ComplexValueOut]
_ComplexArrayOutT = TypeVar("_ComplexArrayOutT", bound=_ComplexArrayOut, default=_ComplexArrayOut)

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
    input: _ComplexArrayOutT | _ArrayLikeComplex_co,
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
    input: _ComplexArrayOutT | _ArrayLikeComplex_co,
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
    input: _ComplexArrayOutT | _ArrayLikeComplex_co,
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
    input: _ComplexArrayOutT | _ArrayLikeComplex_co,
    shift: _FloatArrayIn,
    n: _IntValueIn = -1,
    axis: _IntValueIn = -1,
    output: _ComplexArrayOutT | None = None,
) -> _ComplexArrayOutT: ...
