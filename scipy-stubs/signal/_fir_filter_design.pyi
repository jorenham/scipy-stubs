from typing import Literal

from scipy._typing import Untyped, UntypedArray

__all__ = ["firls", "firwin", "firwin2", "kaiser_atten", "kaiser_beta", "kaiserord", "minimum_phase", "remez"]

def kaiser_beta(a) -> Untyped: ...
def kaiser_atten(numtaps: int, width: float | None) -> Untyped: ...
def kaiserord(ripple, width: float | None) -> Untyped: ...
def firwin(
    numtaps: int,
    cutoff,
    *,
    width: Untyped | None = None,
    window: str = "hamming",
    pass_zero: Literal[True, False, "bandpass", "lowpass", "highpass", "bandstop"] = True,
    scale: bool = True,
    fs: float | None = None,
) -> Untyped: ...
def firwin2(
    numtaps: int,
    freq,
    gain,
    *,
    nfreqs: Untyped | None = None,
    window: str = "hamming",
    antisymmetric: bool = False,
    fs: float | None = None,
) -> Untyped: ...
def remez(
    numtaps: int,
    bands,
    desired,
    *,
    weight: Untyped | None = None,
    type: str = "bandpass",
    maxiter: int = 25,
    grid_density: int = 16,
    fs: float | None = None,
) -> Untyped: ...
def firls(
    numtaps: int,
    bands,
    desired,
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
