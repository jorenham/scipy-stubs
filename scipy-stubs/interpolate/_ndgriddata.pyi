from .interpnd import (
    CloughTocher2DInterpolator as CloughTocher2DInterpolator,
    LinearNDInterpolator as LinearNDInterpolator,
    NDInterpolatorBase as NDInterpolatorBase,
)
from scipy._typing import Untyped
from scipy.spatial import cKDTree as cKDTree

class NearestNDInterpolator(NDInterpolatorBase):
    tree: Untyped
    values: Untyped
    def __init__(self, x, y, rescale: bool = False, tree_options: Untyped | None = None): ...
    def __call__(self, *args, **query_options) -> Untyped: ...

def griddata(points, values, xi, method: str = "linear", fill_value=..., rescale: bool = False) -> Untyped: ...
