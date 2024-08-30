from scipy._typing import Untyped

__all__ = ["argrelextrema", "argrelmax", "argrelmin", "find_peaks", "find_peaks_cwt", "peak_prominences", "peak_widths"]

def argrelmin(data: Untyped, axis: int = 0, order: int = 1, mode: str = "clip") -> Untyped: ...
def argrelmax(data: Untyped, axis: int = 0, order: int = 1, mode: str = "clip") -> Untyped: ...
def argrelextrema(data: Untyped, comparator: Untyped, axis: int = 0, order: int = 1, mode: str = "clip") -> Untyped: ...
def peak_prominences(x: Untyped, peaks: Untyped, wlen: Untyped | None = None) -> Untyped: ...
def peak_widths(
    x: Untyped,
    peaks: Untyped,
    rel_height: float = 0.5,
    prominence_data: Untyped | None = None,
    wlen: Untyped | None = None,
) -> Untyped: ...
def find_peaks(
    x: Untyped,
    height: Untyped | None = None,
    threshold: Untyped | None = None,
    distance: Untyped | None = None,
    prominence: Untyped | None = None,
    width: Untyped | None = None,
    wlen: Untyped | None = None,
    rel_height: float = 0.5,
    plateau_size: Untyped | None = None,
) -> Untyped: ...
def find_peaks_cwt(
    vector: Untyped,
    widths: Untyped,
    wavelet: Untyped | None = None,
    max_distances: Untyped | None = None,
    gap_thresh: Untyped | None = None,
    min_length: Untyped | None = None,
    min_snr: int = 1,
    noise_perc: int = 10,
    window_size: Untyped | None = None,
) -> Untyped: ...
