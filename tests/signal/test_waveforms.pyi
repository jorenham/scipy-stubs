from typing import Literal, TypeAlias
from typing_extensions import assert_type

import numpy as np
import numpy.typing as npt
import optype.numpy as onp
from scipy.signal import gausspulse

_Array_f8: TypeAlias = npt.NDArray[np.float64]
_Falsy: TypeAlias = Literal[0, False]
_Truthy: TypeAlias = Literal[1, True]

_time: onp.ToFloatND
_int: onp.ToInt
_float: onp.ToFloat
_truthy: _Truthy
_falsy: _Falsy

# test gausspulse function overloads
assert_type(gausspulse(_time), _Array_f8)
# Full positional arguments
assert_type(gausspulse(_time, _int, _float, _int, _int, _falsy, _falsy), _Array_f8)
assert_type(
    gausspulse(_time, _int, _float, _int, _int, _truthy, _falsy),
    tuple[_Array_f8, _Array_f8],
)
assert_type(
    gausspulse(_time, _int, _float, _int, _int, _falsy, _truthy),
    tuple[_Array_f8, _Array_f8],
)
assert_type(
    gausspulse(_time, _int, _float, _int, _int, _truthy, _truthy),
    tuple[_Array_f8, _Array_f8, _Array_f8],
)
# Full keyword arguments
assert_type(gausspulse(t=_time), _Array_f8)
assert_type(
    gausspulse(
        t=_time,
        retquad=_falsy,
        retenv=_falsy,
    ),
    _Array_f8,
)
assert_type(
    gausspulse(
        t=_time,
        retquad=_truthy,
        retenv=_falsy,
    ),
    tuple[_Array_f8, _Array_f8],
)
assert_type(
    gausspulse(
        t=_time,
        retquad=_falsy,
        retenv=_truthy,
    ),
    tuple[_Array_f8, _Array_f8],
)
assert_type(
    gausspulse(
        t=_time,
        retquad=_truthy,
        retenv=_truthy,
    ),
    tuple[_Array_f8, _Array_f8, _Array_f8],
)

# Mixed positional and keyword arguments
assert_type(
    gausspulse(_time, _int, _float, _int, _int, retquad=_falsy, retenv=_falsy),
    _Array_f8,
)
assert_type(
    gausspulse(_time, _int, _float, _int, _int, retquad=_truthy, retenv=_falsy),
    tuple[_Array_f8, _Array_f8],
)
assert_type(
    gausspulse(_time, _int, _float, _int, _int, retquad=_falsy, retenv=_truthy),
    tuple[_Array_f8, _Array_f8],
)
assert_type(
    gausspulse(_time, _int, _float, _int, _int, retquad=_truthy, retenv=_truthy),
    tuple[_Array_f8, _Array_f8, _Array_f8],
)
