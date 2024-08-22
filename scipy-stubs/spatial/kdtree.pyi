# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.spatial` namespace for importing the functions
# included below.
from . import _ckdtree, _kdtree
from ._ckdtree import *
from ._kdtree import *

__all__: list[str] = []
__all__ += _ckdtree.__all__
__all__ += _kdtree.__all__
