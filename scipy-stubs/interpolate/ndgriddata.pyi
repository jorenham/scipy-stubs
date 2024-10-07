# This module is not meant for public use and will be removed in SciPy v2.0.0.
from typing_extensions import deprecated

__all__ = ["CloughTocher2DInterpolator", "LinearNDInterpolator", "NearestNDInterpolator", "griddata"]

@deprecated("will be removed in SciPy v2.0.0")
class NearestNDInterpolator:
    def __init__(
        self,
        /,
        x: object,
        y: object,
        rescale: object = ...,
        tree_options: object = ...,
    ) -> None: ...
    def __call__(self, /, *args: object, **query_options: object) -> object: ...

@deprecated("will be removed in SciPy v2.0.0")
def griddata(
    points: object,
    values: object,
    xi: object,
    method: object = ...,
    fill_value: float = ...,
    rescale: object = ...,
) -> object: ...

# interpnd
@deprecated("will be removed in SciPy v2.0.0")
class LinearNDInterpolator:
    def __init__(
        self,
        /,
        points: object,
        values: object,
        fill_value: object = ...,
        rescale: object = ...,
    ) -> None: ...

@deprecated("will be removed in SciPy v2.0.0")
class CloughTocher2DInterpolator:
    def __init__(
        self,
        /,
        points: object,
        values: object,
        fill_value: object = ...,
        tol: object = ...,
        maxiter: object = ...,
        rescale: object = ...,
    ) -> None: ...
