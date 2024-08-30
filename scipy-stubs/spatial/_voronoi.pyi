import numpy as np
import numpy.typing as npt

__all__ = ["sort_vertices_of_regions"]

def sort_vertices_of_regions(simplices: npt.NDArray[np.generic], regions: list[list[int]]) -> None: ...
