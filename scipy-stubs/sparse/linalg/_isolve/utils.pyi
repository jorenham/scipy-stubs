from scipy._typing import Untyped
from scipy.sparse.linalg._interface import (
    IdentityOperator as IdentityOperator,
    LinearOperator as LinearOperator,
    aslinearoperator as aslinearoperator,
)

__docformat__: str

def coerce(x, y) -> Untyped: ...
def id(x) -> Untyped: ...
def make_system(A, M, x0, b) -> Untyped: ...
