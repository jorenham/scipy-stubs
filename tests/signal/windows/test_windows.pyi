from typing import TypeAlias
from typing_extensions import assert_type

import numpy as np
from scipy.signal.windows import dpss

_Array_f8_1d: TypeAlias = np.ndarray[tuple[int], np.dtype[np.float64]]
_Array_f8_2d: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.float64]]

# test dpss function overloads
assert_type(dpss(64, 3), _Array_f8_1d)
assert_type(dpss(64, 3, 2), _Array_f8_2d)
assert_type(dpss(64, 3, return_ratios=True), tuple[_Array_f8_1d, np.float64])
assert_type(dpss(64, 3, 2, return_ratios=True), tuple[_Array_f8_2d, _Array_f8_1d])
