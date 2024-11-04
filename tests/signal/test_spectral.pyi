from typing import Literal, TypeAlias
from typing_extensions import assert_type

import numpy as np
import numpy.typing as npt
import optype.numpy as onpt
from scipy.signal import istft, spectrogram

_Array_f8: TypeAlias = npt.NDArray[np.float64]
_ArrayFloat: TypeAlias = npt.NDArray[np.float32 | np.float64 | np.float128]
_ArrayComplex: TypeAlias = npt.NDArray[np.complex64 | np.complex128 | np.complex256]

array_f8_1d: onpt.Array[tuple[Literal[256]], np.float64]
array_c16_1d: onpt.Array[tuple[Literal[256]], np.complex128]
spectrogram_mode_real: Literal["psd", "magnitude", "angle", "phase"]

# test spectrogram function overloads
assert_type(spectrogram(array_f8_1d), tuple[_Array_f8, _Array_f8, _ArrayFloat])
assert_type(spectrogram(array_f8_1d, mode=spectrogram_mode_real), tuple[_Array_f8, _Array_f8, _ArrayFloat])
assert_type(spectrogram(array_f8_1d, mode="complex"), tuple[_Array_f8, _Array_f8, _ArrayComplex])
assert_type(
    spectrogram(array_f8_1d, 1.0, ("tukey", 2.5), None, None, None, "constant", True, "density", -1, "complex"),
    tuple[_Array_f8, _Array_f8, _ArrayComplex],
)

# test isft function overloads
assert_type(istft(array_c16_1d), tuple[_Array_f8, _ArrayFloat])
assert_type(istft(array_c16_1d, input_onesided=True), tuple[_Array_f8, _ArrayFloat])
assert_type(istft(array_c16_1d, 1.0, "hann", 256, 128, 256, False), tuple[_Array_f8, _ArrayComplex])
assert_type(
    istft(array_c16_1d, input_onesided=False, fs=1.0, window="hann", nperseg=256, noverlap=128, nfft=256),
    tuple[_Array_f8, _ArrayComplex],
)
assert_type(
    istft(
        array_c16_1d,
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
    tuple[_Array_f8, _ArrayFloat],
)
assert_type(
    istft(
        array_c16_1d,
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
