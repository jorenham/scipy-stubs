from typing import Literal

import numpy as np
from scipy._typing import Untyped, UntypedArray
from scipy.linalg import (
    LinAlgError as LinAlgError,
    LinAlgWarning as LinAlgWarning,
    hankel as hankel,
    lstsq as lstsq,
    solve as solve,
    toeplitz as toeplitz,
)
from scipy.special import sinc as sinc

def kaiser_beta(a) -> Untyped: ...
def kaiser_atten(numtaps, width) -> Untyped: ...
def kaiserord(ripple, width) -> Untyped: ...
def firwin(
    numtaps,
    cutoff,
    *,
    width: Untyped | None = None,
    window: str = "hamming",
    pass_zero: bool = True,
    scale: bool = True,
    fs: Untyped | None = None,
) -> Untyped: ...
def firwin2(
    numtaps,
    freq,
    gain,
    *,
    nfreqs: Untyped | None = None,
    window: str = "hamming",
    antisymmetric: bool = False,
    fs: Untyped | None = None,
) -> Untyped: ...
def remez(
    numtaps,
    bands,
    desired,
    *,
    weight: Untyped | None = None,
    type: str = "bandpass",
    maxiter: int = 25,
    grid_density: int = 16,
    fs: Untyped | None = None,
) -> Untyped: ...
def firls(numtaps, bands, desired, *, weight: Untyped | None = None, fs: Untyped | None = None) -> Untyped: ...
def minimum_phase(
    h: UntypedArray,
    method: Literal["homomorphic", "hilbert"] = "homomorphic",
    n_fft: int | None = None,
    *,
    half: bool = True,
) -> UntypedArray: ...
