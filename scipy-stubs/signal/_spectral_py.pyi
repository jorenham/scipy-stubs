from scipy._typing import Untyped
from ._arraytools import const_ext as const_ext, even_ext as even_ext, odd_ext as odd_ext, zero_ext as zero_ext
from .windows import get_window as get_window

def lombscargle(x, y, freqs, precenter: bool = False, normalize: bool = False) -> Untyped: ...
def periodogram(
    x,
    fs: float = 1.0,
    window: str = "boxcar",
    nfft: Untyped | None = None,
    detrend: str = "constant",
    return_onesided: bool = True,
    scaling: str = "density",
    axis: int = -1,
) -> Untyped: ...
def welch(
    x,
    fs: float = 1.0,
    window: str = "hann",
    nperseg: Untyped | None = None,
    noverlap: Untyped | None = None,
    nfft: Untyped | None = None,
    detrend: str = "constant",
    return_onesided: bool = True,
    scaling: str = "density",
    axis: int = -1,
    average: str = "mean",
) -> Untyped: ...
def csd(
    x,
    y,
    fs: float = 1.0,
    window: str = "hann",
    nperseg: Untyped | None = None,
    noverlap: Untyped | None = None,
    nfft: Untyped | None = None,
    detrend: str = "constant",
    return_onesided: bool = True,
    scaling: str = "density",
    axis: int = -1,
    average: str = "mean",
) -> Untyped: ...
def spectrogram(
    x,
    fs: float = 1.0,
    window=("tukey", 0.25),
    nperseg: Untyped | None = None,
    noverlap: Untyped | None = None,
    nfft: Untyped | None = None,
    detrend: str = "constant",
    return_onesided: bool = True,
    scaling: str = "density",
    axis: int = -1,
    mode: str = "psd",
) -> Untyped: ...
def check_COLA(window, nperseg, noverlap, tol: float = 1e-10) -> Untyped: ...
def check_NOLA(window, nperseg, noverlap, tol: float = 1e-10) -> Untyped: ...
def stft(
    x,
    fs: float = 1.0,
    window: str = "hann",
    nperseg: int = 256,
    noverlap: Untyped | None = None,
    nfft: Untyped | None = None,
    detrend: bool = False,
    return_onesided: bool = True,
    boundary: str = "zeros",
    padded: bool = True,
    axis: int = -1,
    scaling: str = "spectrum",
) -> Untyped: ...
def istft(
    Zxx,
    fs: float = 1.0,
    window: str = "hann",
    nperseg: Untyped | None = None,
    noverlap: Untyped | None = None,
    nfft: Untyped | None = None,
    input_onesided: bool = True,
    boundary: bool = True,
    time_axis: int = -1,
    freq_axis: int = -2,
    scaling: str = "spectrum",
) -> Untyped: ...
def coherence(
    x,
    y,
    fs: float = 1.0,
    window: str = "hann",
    nperseg: Untyped | None = None,
    noverlap: Untyped | None = None,
    nfft: Untyped | None = None,
    detrend: str = "constant",
    axis: int = -1,
) -> Untyped: ...
