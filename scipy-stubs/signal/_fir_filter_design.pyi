from typing import Literal

from scipy._typing import Untyped, UntypedArray

__all__ = ["firls", "firwin", "firwin2", "kaiser_atten", "kaiser_beta", "kaiserord", "minimum_phase", "remez"]

def kaiser_beta(a: Untyped) -> Untyped: ...
def kaiser_atten(numtaps: int, width: float | None) -> Untyped: ...
def kaiserord(ripple: Untyped, width: float | None) -> Untyped: ...
def firwin(
    numtaps: int,
    cutoff: Untyped,
    *,
    width: Untyped | None = None,
    window: str = "hamming",
    pass_zero: Literal["bandpass", "lowpass", "highpass", "bandstop"] | bool = True,
    scale: bool = True,
    fs: float | None = None,
) -> Untyped: ...
def firwin2(
    numtaps: int,
    freq: Untyped,
    gain: Untyped,
    *,
    nfreqs: Untyped | None = None,
    window: str = "hamming",
    antisymmetric: bool = False,
    fs: float | None = None,
) -> Untyped: ...
def remez(
    numtaps: int,
    bands: Untyped,
    desired: Untyped,
    *,
    weight: Untyped | None = None,
    type: str = "bandpass",
    maxiter: int = 25,
    grid_density: int = 16,
    fs: float | None = None,
) -> Untyped: ...
def firls(
    numtaps: int,
    bands: Untyped,
    desired: Untyped,
    *,
    weight: Untyped | None = None,
    fs: float | None = None,
) -> Untyped: ...
def minimum_phase(
    h: UntypedArray,
    method: Literal["homomorphic", "hilbert"] = "homomorphic",
    n_fft: int | None = None,
    *,
    half: bool = True,
) -> UntypedArray: ...
