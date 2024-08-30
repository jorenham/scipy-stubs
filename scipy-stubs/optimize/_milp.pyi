from scipy._lib._util import VisibleDeprecationWarning as VisibleDeprecationWarning
from scipy._typing import Untyped
from scipy.sparse import csc_array as csc_array, issparse as issparse, vstack as vstack
from ._constraints import Bounds as Bounds, LinearConstraint as LinearConstraint
from ._optimize import OptimizeResult as OptimizeResult

def milp(
    c,
    *,
    integrality: Untyped | None = None,
    bounds: Untyped | None = None,
    constraints: Untyped | None = None,
    options: Untyped | None = None,
) -> Untyped: ...
