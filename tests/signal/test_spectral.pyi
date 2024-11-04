from typing import Any, TypeAlias
from typing_extensions import assert_type

import numpy as np
import numpy.typing as npt
from scipy.signal import istft, spectrogram

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

# test isft function overloads
assert_type(istft(np.ones((129, 100), dtype=np.complex128)), tuple[_Array_f8, _ArrayReal])
assert_type(istft(np.ones((129, 100), dtype=np.complex128), input_onesided=True), tuple[_Array_f8, _ArrayReal])
assert_type(istft(np.ones((129, 100), dtype=np.complex128), 1.0, "hann", 256, 128, 256, False), tuple[_Array_f8, _ArrayComplex])
assert_type(
    istft(
        np.ones((129, 100), dtype=np.complex128), input_onesided=False, fs=1.0, window="hann", nperseg=256, noverlap=128, nfft=256
    ),
    tuple[_Array_f8, _ArrayComplex],
)
assert_type(
    istft(
        np.ones((129, 100), dtype=np.complex128),
        fs=2.0,
        window=("tukey", 0.25),
        nperseg=256,
        noverlap=128,
        nfft=256,
        input_onesided=True,
        boundary=False,
        time_axis=-1,
        freq_axis=0,
        scaling="spectrum",
    ),
    tuple[_Array_f8, _ArrayReal],
)
assert_type(
    istft(
        np.ones((129, 100), dtype=np.complex128),
        fs=2.0,
        window=("tukey", 0.25),
        nperseg=256,
        noverlap=128,
        nfft=256,
        input_onesided=False,
        boundary=False,
        time_axis=0,
        freq_axis=1,
        scaling="spectrum",
    ),
    tuple[_Array_f8, _ArrayComplex],
)
