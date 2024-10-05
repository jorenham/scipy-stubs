# This module is not meant for public use and will be removed in SciPy v2.0.0.
# This stub simply re-exports the imported functions.
# TODO: Add type annotated dummy functions marked deprecated.
from ._dsolve import spsolve
from ._interface import *
from ._matfuncs import *

__all__ = ["LinearOperator", "expm", "inv", "spsolve"]
