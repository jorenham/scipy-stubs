from typing import Literal

from scipy._typing import Untyped, UntypedCallable

__all__ = ["check_COLA", "check_NOLA", "coherence", "csd", "istft", "lombscargle", "periodogram", "spectrogram", "stft", "welch"]

def lombscargle(x, y, freqs, precenter: bool = False, normalize: bool = False) -> Untyped: ...
def periodogram(
    x,
    fs: float = 1.0,
    window: str = "boxcar",
    nfft: int | None = None,
    detrend: str | Literal[False] | UntypedCallable = "constant",
    return_onesided: bool = True,
    scaling: str = "density",
    axis: int = -1,
) -> Untyped: ...
def welch(
    x,
    fs: float = 1.0,
    window: str = "hann",
    nperseg: int | None = None,
    noverlap: int | None = None,
    nfft: int | None = None,
    detrend: str | Literal[False] | UntypedCallable = "constant",
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
    nperseg: int | None = None,
    noverlap: int | None = None,
    nfft: int | None = None,
    detrend: str | Literal[False] | UntypedCallable = "constant",
    return_onesided: bool = True,
    scaling: str = "density",
    axis: int = -1,
    average: str = "mean",
) -> Untyped: ...
def spectrogram(
    x,
    fs: float = 1.0,
    window=("tukey", 0.25),
    nperseg: int | None = None,
    noverlap: int | None = None,
    nfft: int | None = None,
    detrend: str | Literal[False] | UntypedCallable = "constant",
    return_onesided: bool = True,
    scaling: str = "density",
    axis: int = -1,
    mode: str = "psd",
) -> Untyped: ...
def check_COLA(window, nperseg: int, noverlap: int, tol: float = 1e-10) -> Untyped: ...
def check_NOLA(window, nperseg: int, noverlap: int, tol: float = 1e-10) -> Untyped: ...
def stft(
    x,
    fs: float = 1.0,
    window: str = "hann",
    nperseg: int = 256,
    noverlap: int | None = None,
    nfft: int | None = None,
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
    nperseg: int | None = None,
    noverlap: int | None = None,
    nfft: int | None = None,
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
    nperseg: int | None = None,
    noverlap: int | None = None,
    nfft: int | None = None,
    detrend: str | Literal[False] | UntypedCallable = "constant",
    axis: int = -1,
) -> Untyped: ...
