# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.spatial` namespace for importing the functions
# included below.
from ._ckdtree import *
from ._kdtree import *

__all__ = ["KDTree", "Rectangle", "cKDTree", "distance_matrix", "minkowski_distance", "minkowski_distance_p"]
