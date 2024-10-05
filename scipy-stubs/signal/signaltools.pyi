# This module is not meant for public use and will be removed in SciPy v2.0.0.
# This stub simply re-exports the imported functions.
# TODO: Add type annotated dummy functions marked deprecated.
from ._filter_design import *
from ._fir_filter_design import *
from ._ltisys import *
from ._signaltools import *
from ._upfirdn import *
from .windows import *

__all__ = [
    "cheby1",
    "choose_conv_method",
    "cmplx_sort",
    "convolve",
    "convolve2d",
    "correlate",
    "correlate2d",
    "correlation_lags",
    "decimate",
    "deconvolve",
    "detrend",
    "dlti",
    "fftconvolve",
    "filtfilt",
    "firwin",
    "get_window",
    "hilbert",
    "hilbert2",
    "invres",
    "invresz",
    "lfilter",
    "lfilter_zi",
    "lfiltic",
    "medfilt",
    "medfilt2d",
    "oaconvolve",
    "order_filter",
    "resample",
    "resample_poly",
    "residue",
    "residuez",
    "sosfilt",
    "sosfilt_zi",
    "sosfiltfilt",
    "unique_roots",
    "upfirdn",
    "vectorstrength",
    "wiener",
]
