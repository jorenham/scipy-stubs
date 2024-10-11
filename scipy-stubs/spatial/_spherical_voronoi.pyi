import numpy as np
import numpy.typing as npt
from numpy._typing import _ArrayLikeComplex_co, _ArrayLikeFloat_co

__all__ = ["SphericalVoronoi"]

def calculate_solid_angles(R: _ArrayLikeComplex_co) -> npt.NDArray[np.float64]: ...

class SphericalVoronoi:
    points: np.ndarray[tuple[int, int], np.dtype[np.float64]]
    center: np.ndarray[tuple[int], np.dtype[np.float64]]
    radius: float

    def __init__(
        self,
        /,
        points: _ArrayLikeFloat_co,
        radius: float = 1,
        center: _ArrayLikeFloat_co | None = None,
        threshold: float = 1e-06,
    ) -> None: ...
    def sort_vertices_of_regions(self, /) -> None: ...
    def calculate_areas(self, /) -> np.ndarray[tuple[int], np.dtype[np.float64]]: ...
