from typing import overload

import numpy as np
import optype.numpy as onp
from scipy._typing import ToRNG
from scipy.sparse import csc_matrix

__all__ = ["clarkson_woodruff_transform"]

###

def cwt_matrix(n_rows: onp.ToInt, n_columns: onp.ToInt, rng: ToRNG = None) -> csc_matrix: ...

#
@overload
def clarkson_woodruff_transform(input_matrix: onp.ToInt2D, sketch_size: onp.ToInt, rng: ToRNG = None) -> onp.Array2D[np.int_]: ...
@overload
def clarkson_woodruff_transform(
    input_matrix: onp.ToJustFloat2D,
    sketch_size: onp.ToInt,
    rng: ToRNG = None,
) -> onp.Array2D[np.float64 | np.longdouble]: ...
@overload
def clarkson_woodruff_transform(
    input_matrix: onp.ToJustComplex2D,
    sketch_size: onp.ToInt,
    rng: ToRNG = None,
) -> onp.Array2D[np.complex64 | np.clongdouble]: ...
