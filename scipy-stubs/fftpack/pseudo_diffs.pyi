# This module is not meant for public use and will be removed in SciPy v2.0.0.
# This stub simply re-exports the imported functions.
# TODO: Add type annotated dummy functions marked deprecated.
from typing_extensions import deprecated

from ._pseudo_diffs import *

@deprecated("will be removed in SciPy v2.0.0")
def convolve(*args: object, **kwargs: object) -> object: ...

__all__ = ["cc_diff", "convolve", "cs_diff", "diff", "hilbert", "ihilbert", "itilbert", "sc_diff", "shift", "ss_diff", "tilbert"]
