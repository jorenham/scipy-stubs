from typing import Any, TypeAlias
from typing_extensions import assert_type

import numpy as np
import numpy.typing as npt
from scipy.signal import spectrogram

_Array_f8: TypeAlias = npt.NDArray[np.float64]
_ArrayReal: TypeAlias = npt.NDArray[np.floating[Any]]
_ArrayComplex: TypeAlias = npt.NDArray[np.complexfloating[Any, Any]]

# test spectrogram function overloads
assert_type(spectrogram(np.linspace(200, 300, 256)), tuple[_Array_f8, _Array_f8, _ArrayReal])
assert_type(spectrogram(np.linspace(200, 300, 256), mode="psd"), tuple[_Array_f8, _Array_f8, _ArrayReal])
assert_type(spectrogram(np.linspace(200, 300, 256), mode="magnitude"), tuple[_Array_f8, _Array_f8, _ArrayReal])
assert_type(spectrogram(np.linspace(200, 300, 256), mode="angle"), tuple[_Array_f8, _Array_f8, _ArrayReal])
assert_type(spectrogram(np.linspace(200, 300, 256), mode="phase"), tuple[_Array_f8, _Array_f8, _ArrayReal])
assert_type(spectrogram(np.linspace(200, 300, 256), mode="complex"), tuple[_Array_f8, _Array_f8, _ArrayComplex])
assert_type(
    spectrogram(np.linspace(200, 300, 256), 1.0, ("tukey", 2.5), None, None, None, "constant", True, "density", -1, "complex"),
    tuple[_Array_f8, _Array_f8, _ArrayComplex],
)
