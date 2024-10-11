# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.spatial` namespace for importing the functions
# included below.
from ._qhull import *

__all__ = ["ConvexHull", "Delaunay", "HalfspaceIntersection", "QhullError", "Voronoi", "tsearch"]
