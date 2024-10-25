from collections.abc import Callable
from typing import Literal
from typing_extensions import assert_type

import numpy as np
from scipy.integrate import quad
from scipy.integrate._quadpack_py import _QuadExplain
from scipy.integrate._typing import QuadInfoDict

TRUE: Literal[True] = True

# ufunc
_ = assert_type(quad(np.exp, 0, 1), tuple[float, float])

# (float) -> float
f0_float_float: Callable[[float], float]
_ = assert_type(quad(f0_float_float, 0, 1), tuple[float, float])

# (float) -> np.float64
f0_float_f8: Callable[[float], np.float64]
_ = assert_type(quad(f0_float_f8, 0, 1), tuple[float, float])

# (np.float64) -> float
f0_f8_float: Callable[[np.float64], float]
_ = assert_type(quad(f0_f8_float, 0, 1), tuple[float, float])

# (float, str) -> float
f1_float_float: Callable[[float, str], float]
_ = assert_type(quad(f1_float_float, 0, 1, args=("",)), tuple[float, float])

# (float, str, str) -> float
f2_float_float: Callable[[float, str, str], float]
_ = assert_type(quad(f2_float_float, 0, 1, args=("", "")), tuple[float, float])

# (float) -> float, full output
# NOTE: this test fails (only) in mypy due to some mypy bug
_ = assert_type(  # type: ignore[assignment]
    quad(f0_float_float, 0, 1, full_output=TRUE),
    tuple[float, float, QuadInfoDict]
    | tuple[float, float, QuadInfoDict, str]
    | tuple[float, float, QuadInfoDict, str, _QuadExplain],
)

# (float) -> complex
# NOTE: this test fails (only) in mypy due to some mypy bug
z0_float_complex: Callable[[float], complex]
_ = assert_type(quad(z0_float_complex, 0, 1, complex_func=TRUE), tuple[complex, complex])  # type: ignore[assignment]
