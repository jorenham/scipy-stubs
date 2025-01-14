from typing_extensions import assert_type

import numpy as np
import optype.numpy as onp
from scipy.stats._distn_infrastructure import rv_discrete, rv_sample

xk: onp.Array1D[np.int_]
pk: tuple[float, ...]

# mypy fails because it (still) doesn't support __new__ returning something that isn't `Self`
assert_type(rv_discrete(values=(xk, pk)), rv_sample)  # type: ignore[assert-type]
