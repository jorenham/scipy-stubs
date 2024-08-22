import numpy as np
import numpy.typing as npt

from scipy.spatial.distance import euclidean as euclidean

def geometric_slerp(start: npt.ArrayLike, end: npt.ArrayLike, t: npt.ArrayLike, tol: float = 1e-07) -> np.ndarray: ...
