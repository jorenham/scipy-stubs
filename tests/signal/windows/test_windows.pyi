from typing_extensions import assert_type

import numpy as np
import optype.numpy as onp
from scipy.signal.windows import dpss

# test dpss function overloads
assert_type(dpss(64, 3), onp.Array1D[np.float64])
assert_type(dpss(64, 3, 2), onp.Array2D[np.float64])
assert_type(dpss(64, 3, return_ratios=True), tuple[onp.Array1D[np.float64], np.float64])
assert_type(dpss(64, 3, 2, return_ratios=True), tuple[onp.Array2D[np.float64], onp.Array1D[np.float64]])
