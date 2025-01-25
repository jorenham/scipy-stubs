from typing import Final, Literal, TypeAlias, TypeVar

import numpy as np
import optype.numpy as onp
from _typeshed import IdentityFunction
from scipy.sparse._base import _spbase
from scipy.sparse._typing import Numeric
from scipy.sparse.linalg import LinearOperator

__all__: list[str] = []

_T = TypeVar("_T")
_Char: TypeAlias = Literal["f", "d", "F", "D"]
_ToLinearOperator: TypeAlias = onp.CanArrayND[Numeric] | _spbase[Numeric] | LinearOperator[Numeric]
_Inexact: TypeAlias = np.float32 | np.float64 | np.complex64 | np.complex128

###

__docformat__: Final = "restructuredtext en"
_coerce_rules: Final[dict[tuple[_Char, _Char], _Char]]

def id(x: _T) -> _T: ...
def coerce(x: str, y: str) -> _Char: ...
def make_system(
    A: _ToLinearOperator,
    M: _ToLinearOperator | None,
    x0: onp.ToComplex1D | Literal["Mb"] | None,
    b: onp.ToComplex1D,
) -> tuple[
    LinearOperator,  # A
    LinearOperator,  # M
    onp.Array1D[_Inexact],  # x
    onp.Array1D[_Inexact],  # b
    IdentityFunction,  # postprocess
]: ...
