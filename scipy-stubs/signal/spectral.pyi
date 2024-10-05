# This module is not meant for public use and will be removed in SciPy v2.0.0.
# This stub simply re-exports the imported functions.
# TODO: Add type annotated dummy functions marked deprecated.
from ._spectral_py import *

__all__ = [
    "check_COLA",
    "check_NOLA",
    "coherence",
    "csd",
    "get_window",
    "istft",
    "lombscargle",
    "periodogram",
    "spectrogram",
    "stft",
    "welch",
]
