from typing import TypeAlias
from typing_extensions import assert_type

import numpy as np
import optype.numpy as onp
from scipy.stats import distributions as d

_Value_f8: TypeAlias = float | np.float64
_Array_f8: TypeAlias = _Value_f8 | onp.ArrayND[np.float64]

# test `rv_continuous_frozen`
assert_type(d.uniform().mean(), _Value_f8)
assert_type(d.uniform(0).mean(), _Value_f8)
assert_type(d.uniform(0.5, 2).mean(), _Value_f8)
assert_type(d.uniform([0, -1]).mean(), _Array_f8)
assert_type(d.uniform([0, 0.5], 2).mean(), _Array_f8)
assert_type(d.uniform(0, [0.5, 2]).mean(), _Array_f8)

# test `rv_discrete_frozen`
assert_type(d.bernoulli().mean(), _Value_f8)
assert_type(d.bernoulli(0).mean(), _Value_f8)
assert_type(d.bernoulli(0.5).mean(), _Value_f8)
assert_type(d.bernoulli([0, -1]).mean(), _Array_f8)
assert_type(d.bernoulli([0, 0.5]).mean(), _Array_f8)

# TODO: more tests
