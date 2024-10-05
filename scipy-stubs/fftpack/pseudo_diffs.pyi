# This module is not meant for public use and will be removed in SciPy v2.0.0.
# This stub simply re-exports the imported functions.
# TODO: Add type annotated dummy functions marked deprecated.
from . import convolve
from ._pseudo_diffs import *

__all__ = ["cc_diff", "convolve", "cs_diff", "diff", "hilbert", "ihilbert", "itilbert", "sc_diff", "shift", "ss_diff", "tilbert"]
