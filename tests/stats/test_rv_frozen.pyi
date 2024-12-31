from typing import TypeAlias
from typing_extensions import assert_type

import numpy as np
import optype.numpy as onp
from scipy.stats import distributions as d

_Float: TypeAlias = float | np.float64
_FloatND: TypeAlias = _Float | onp.ArrayND[np.float64]

###
# `rv_continuous_frozen`
# .mean()
assert_type(d.uniform().mean(), _Float)
assert_type(d.uniform(0).mean(), _Float)
assert_type(d.uniform(0.5, 2).mean(), _Float)
assert_type(d.uniform([0, -1]).mean(), _FloatND)
assert_type(d.uniform([0, 0.5], 2).mean(), _FloatND)
assert_type(d.uniform(0, [0.5, 2]).mean(), _FloatND)
# .expect()
assert_type(d.uniform().expect(), _Float)
assert_type(d.uniform(0).expect(), _Float)
assert_type(d.uniform(0.5, 2).expect(), _Float)
d.uniform([0, -1]).expect()  # type: ignore[misc]  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
d.uniform([0, 0.5], 2).expect()  # type: ignore[misc]  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
d.uniform(0, [0.5, 2]).expect()  # type: ignore[misc]  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
###

###
# `rv_discrete_frozen`
# .mean()
assert_type(d.bernoulli().mean(), _Float)  # TODO: Reject empty constructor (for all `rv_discrete`?)
assert_type(d.bernoulli(0.5).mean(), _Float)
assert_type(d.bernoulli(0.5, 1).mean(), _Float)
assert_type(d.bernoulli(0.5, loc=1).mean(), _Float)
assert_type(d.bernoulli([0, 0.5]).mean(), _FloatND)
# .expect()
assert_type(d.bernoulli(0.5).mean(), _Float)
assert_type(d.bernoulli(0.5, 1).mean(), _Float)
assert_type(d.bernoulli(0.5, loc=1).mean(), _Float)
d.bernoulli([0, 0.5]).expect()  # type: ignore[misc]  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
###

# TODO: more tests
