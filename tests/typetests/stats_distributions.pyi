from typing import Any, TypeAlias
from typing_extensions import assert_type

import numpy as np
from scipy.stats import distributions as d

_F8_0d: TypeAlias = float | np.float64
_F8_nd: TypeAlias = _F8_0d | np.ndarray[tuple[int, ...], np.dtype[np.float64]]

arg_0d: float
arg_nd: list[float]

rv_c = d.uniform()
rv_c_0d = d.uniform(arg_0d)
rv_c_nd = d.uniform(arg_nd)

rv_d = d.bernoulli()
rv_d_0d = d.bernoulli(arg_0d)
rv_d_nd = d.bernoulli(arg_nd)

_: Any

_ = assert_type(rv_c.mean(), _F8_0d)
_ = assert_type(rv_c_0d.mean(), _F8_0d)
_ = assert_type(rv_c_nd.mean(), _F8_nd)

_ = assert_type(rv_d.mean(), _F8_0d)
_ = assert_type(rv_d_0d.mean(), _F8_0d)
_ = assert_type(rv_d_nd.mean(), _F8_nd)

# TODO: more tests
