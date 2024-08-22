from ..utils import get_arrays_tol as get_arrays_tol
from scipy._typing import Untyped

TINY: Untyped

def cauchy_geometry(const, grad, curv, xl, xu, delta, debug) -> Untyped: ...
def spider_geometry(const, grad, curv, xpt, xl, xu, delta, debug) -> Untyped: ...
