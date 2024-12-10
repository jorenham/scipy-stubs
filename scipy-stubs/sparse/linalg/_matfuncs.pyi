from typing import Final

from scipy._typing import Untyped
from scipy.sparse.linalg._interface import LinearOperator

__all__ = ["expm", "inv", "matrix_power"]

UPPER_TRIANGULAR: Final = "upper_triangular"

class MatrixPowerOperator(LinearOperator):
    def __init__(self, /, A: Untyped, p: Untyped, structure: Untyped | None = None) -> None: ...

class ProductOperator(LinearOperator):
    def __init__(self, /, *args: Untyped, **kwargs: Untyped) -> None: ...

def inv(A: Untyped) -> Untyped: ...
def expm(A: Untyped) -> Untyped: ...
def matrix_power(A: Untyped, power: Untyped) -> Untyped: ...
