from typing import Literal, TypeAlias
from typing_extensions import assert_type

import numpy as np
import optype.numpy as onp
from scipy.signal import gausspulse

_Array_f8: TypeAlias = onp.ArrayND[np.float64]
_Scalar_f8: TypeAlias = np.float64
_Falsy: TypeAlias = Literal[0, False]
_Truthy: TypeAlias = Literal[1, True]

_time_scalar: onp.ToFloat
_time_array: onp.ToFloatND
_time_cutoff: Literal["cutoff"]
_float: onp.ToFloat
_truthy: _Truthy
_falsy: _Falsy

# test gausspulse function overloads
# `time` as an array
assert_type(gausspulse(_time_array), _Array_f8)
# Full positional arguments
assert_type(gausspulse(_time_array, _float, _float, _float, _float, _falsy, _falsy), _Array_f8)
assert_type(
    gausspulse(_time_array, _float, _float, _float, _float, _truthy, _falsy),
    tuple[_Array_f8, _Array_f8],
)
assert_type(
    gausspulse(_time_array, _float, _float, _float, _float, _falsy, _truthy),
    tuple[_Array_f8, _Array_f8],
)
assert_type(
    gausspulse(_time_array, _float, _float, _float, _float, _truthy, _truthy),
    tuple[_Array_f8, _Array_f8, _Array_f8],
)
# Full keyword arguments
assert_type(gausspulse(t=_time_array), _Array_f8)
assert_type(
    gausspulse(
        t=_time_array,
        retquad=_falsy,
        retenv=_falsy,
    ),
    _Array_f8,
)
assert_type(
    gausspulse(
        t=_time_array,
        retquad=_truthy,
        retenv=_falsy,
    ),
    tuple[_Array_f8, _Array_f8],
)
assert_type(
    gausspulse(
        t=_time_array,
        retquad=_falsy,
        retenv=_truthy,
    ),
    tuple[_Array_f8, _Array_f8],
)
assert_type(
    gausspulse(
        t=_time_array,
        retquad=_truthy,
        retenv=_truthy,
    ),
    tuple[_Array_f8, _Array_f8, _Array_f8],
)

# Mixed positional and keyword arguments
assert_type(
    gausspulse(_time_array, _float, _float, _float, _float, retquad=_falsy, retenv=_falsy),
    _Array_f8,
)
assert_type(
    gausspulse(_time_array, _float, _float, _float, _float, retquad=_truthy, retenv=_falsy),
    tuple[_Array_f8, _Array_f8],
)
assert_type(
    gausspulse(_time_array, _float, _float, _float, _float, retquad=_falsy, retenv=_truthy),
    tuple[_Array_f8, _Array_f8],
)
assert_type(
    gausspulse(_time_array, _float, _float, _float, _float, retquad=_truthy, retenv=_truthy),
    tuple[_Array_f8, _Array_f8, _Array_f8],
)

# `time` as a scalar
assert_type(gausspulse(_time_scalar), _Scalar_f8)
# Full positional arguments
assert_type(gausspulse(_time_scalar, _float, _float, _float, _float, _falsy, _falsy), _Scalar_f8)
assert_type(
    gausspulse(_time_scalar, _float, _float, _float, _float, _truthy, _falsy),
    tuple[_Scalar_f8, _Scalar_f8],
)
assert_type(
    gausspulse(_time_scalar, _float, _float, _float, _float, _falsy, _truthy),
    tuple[_Scalar_f8, _Scalar_f8],
)
assert_type(
    gausspulse(_time_scalar, _float, _float, _float, _float, _truthy, _truthy),
    tuple[_Scalar_f8, _Scalar_f8, _Scalar_f8],
)
# Full keyword arguments
assert_type(gausspulse(t=_time_scalar), _Scalar_f8)
assert_type(
    gausspulse(
        t=_time_scalar,
        retquad=_falsy,
        retenv=_falsy,
    ),
    _Scalar_f8,
)
assert_type(
    gausspulse(
        t=_time_scalar,
        retquad=_truthy,
        retenv=_falsy,
    ),
    tuple[_Scalar_f8, _Scalar_f8],
)
assert_type(
    gausspulse(
        t=_time_scalar,
        retquad=_falsy,
        retenv=_truthy,
    ),
    tuple[_Scalar_f8, _Scalar_f8],
)
assert_type(
    gausspulse(
        t=_time_scalar,
        retquad=_truthy,
        retenv=_truthy,
    ),
    tuple[_Scalar_f8, _Scalar_f8, _Scalar_f8],
)

# Mixed positional and keyword arguments
assert_type(
    gausspulse(_time_scalar, _float, _float, _float, _float, retquad=_falsy, retenv=_falsy),
    _Scalar_f8,
)
assert_type(
    gausspulse(_time_scalar, _float, _float, _float, _float, retquad=_truthy, retenv=_falsy),
    tuple[_Scalar_f8, _Scalar_f8],
)
assert_type(
    gausspulse(_time_scalar, _float, _float, _float, _float, retquad=_falsy, retenv=_truthy),
    tuple[_Scalar_f8, _Scalar_f8],
)
assert_type(
    gausspulse(_time_scalar, _float, _float, _float, _float, retquad=_truthy, retenv=_truthy),
    tuple[_Scalar_f8, _Scalar_f8, _Scalar_f8],
)

# `time` as the literal `"cutoff"`
assert_type(gausspulse(_time_cutoff), _Scalar_f8)
# Full positional arguments
assert_type(gausspulse(_time_cutoff, _float, _float, _float, _float, _falsy, _falsy), _Scalar_f8)
assert_type(gausspulse(_time_cutoff, _float, _float, _float, _float, _truthy, _falsy), _Scalar_f8)
assert_type(gausspulse(_time_cutoff, _float, _float, _float, _float, _falsy, _truthy), _Scalar_f8)
assert_type(gausspulse(_time_cutoff, _float, _float, _float, _float, _truthy, _truthy), _Scalar_f8)
# Full keyword arguments
assert_type(gausspulse(t=_time_cutoff), _Scalar_f8)
assert_type(
    gausspulse(
        t=_time_cutoff,
        retquad=_falsy,
        retenv=_falsy,
    ),
    _Scalar_f8,
)
assert_type(
    gausspulse(
        t=_time_cutoff,
        retquad=_truthy,
        retenv=_falsy,
    ),
    _Scalar_f8,
)
assert_type(
    gausspulse(
        t=_time_cutoff,
        retquad=_falsy,
        retenv=_truthy,
    ),
    _Scalar_f8,
)
assert_type(
    gausspulse(
        t=_time_cutoff,
        retquad=_truthy,
        retenv=_truthy,
    ),
    _Scalar_f8,
)

# Mixed positional and keyword arguments
assert_type(
    gausspulse(_time_cutoff, _float, _float, _float, _float, retquad=_falsy, retenv=_falsy),
    _Scalar_f8,
)
assert_type(gausspulse(_time_cutoff, _float, _float, _float, _float, retquad=_truthy, retenv=_falsy), _Scalar_f8)
assert_type(gausspulse(_time_cutoff, _float, _float, _float, _float, retquad=_falsy, retenv=_truthy), _Scalar_f8)
assert_type(gausspulse(_time_cutoff, _float, _float, _float, _float, retquad=_truthy, retenv=_truthy), _Scalar_f8)
