from typing import Final, Literal, TypeAlias, overload

import numpy as np
import optype as op
import optype.numpy as onp

__all__ = [
    "solve_continuous_are",
    "solve_continuous_lyapunov",
    "solve_discrete_are",
    "solve_discrete_lyapunov",
    "solve_lyapunov",
    "solve_sylvester",
]

_Float: TypeAlias = np.float32 | np.float64
_Complex: TypeAlias = np.complex64 | np.complex128

_DiscreteMethod: TypeAlias = Literal["direct", "bilinear"]

###

@overload  # real
def solve_sylvester(a: onp.ToFloat2D, b: onp.ToFloat2D, q: onp.ToFloat2D) -> onp.Array2D[_Float]: ...
@overload  # complex
def solve_sylvester(a: onp.ToComplex2D, b: onp.ToComplex2D, q: onp.ToComplex2D) -> onp.Array2D[_Float | _Complex]: ...

#
@overload  # real
def solve_continuous_lyapunov(a: onp.ToFloat2D, q: onp.ToFloat2D) -> onp.Array2D[_Float]: ...
@overload  # complex
def solve_continuous_lyapunov(a: onp.ToComplex2D, q: onp.ToComplex2D) -> onp.Array2D[_Float | _Complex]: ...

#
solve_lyapunov: Final = solve_continuous_lyapunov

#
@overload  # real
def solve_discrete_lyapunov(
    a: onp.ToFloat2D,
    q: onp.ToFloat2D,
    method: _DiscreteMethod | None = None,
) -> onp.Array2D[_Float]: ...
@overload  # complex
def solve_discrete_lyapunov(
    a: onp.ToComplex2D,
    q: onp.ToComplex2D,
    method: _DiscreteMethod | None = None,
) -> onp.Array2D[_Float | _Complex]: ...

#
@overload  # real
def solve_continuous_are(
    a: onp.ToFloat2D,
    b: onp.ToFloat2D,
    q: onp.ToFloat2D,
    r: onp.ToFloat2D,
    e: onp.ToFloat2D | None = None,
    s: onp.ToFloat2D | None = None,
    balanced: op.CanBool = True,
) -> onp.Array2D[_Float]: ...
@overload  # complex
def solve_continuous_are(
    a: onp.ToComplex2D,
    b: onp.ToComplex2D,
    q: onp.ToComplex2D,
    r: onp.ToComplex2D,
    e: onp.ToComplex2D | None = None,
    s: onp.ToComplex2D | None = None,
    balanced: op.CanBool = True,
) -> onp.Array2D[_Float | _Complex]: ...

#
@overload  # real
def solve_discrete_are(
    a: onp.ToFloat2D,
    b: onp.ToFloat2D,
    q: onp.ToFloat2D,
    r: onp.ToFloat2D,
    e: onp.ToFloat2D | None = None,
    s: onp.ToFloat2D | None = None,
    balanced: op.CanBool = True,
) -> onp.Array2D[_Float]: ...
@overload  # complex
def solve_discrete_are(
    a: onp.ToComplex2D,
    b: onp.ToComplex2D,
    q: onp.ToComplex2D,
    r: onp.ToComplex2D,
    e: onp.ToComplex2D | None = None,
    s: onp.ToComplex2D | None = None,
    balanced: op.CanBool = True,
) -> onp.Array2D[_Float | _Complex]: ...
