import numpy as np
import numpy.typing as npt
import scipy._typing as spt
from scipy.sparse import csc_matrix

__all__ = ["clarkson_woodruff_transform"]

def cwt_matrix(n_rows: int, n_columns: int, seed: spt.Seed | None = None) -> csc_matrix: ...
def clarkson_woodruff_transform(
    input_matrix: npt.ArrayLike,
    sketch_size: int,
    seed: spt.Seed | None = None,
) -> npt.NDArray[np.float64]: ...
