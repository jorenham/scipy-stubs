from scipy._typing import Untyped
from scipy.stats import scoreatpercentile as scoreatpercentile

def argrelmin(data, axis: int = 0, order: int = 1, mode: str = "clip") -> Untyped: ...
def argrelmax(data, axis: int = 0, order: int = 1, mode: str = "clip") -> Untyped: ...
def argrelextrema(data, comparator, axis: int = 0, order: int = 1, mode: str = "clip") -> Untyped: ...
def peak_prominences(x, peaks, wlen: Untyped | None = None) -> Untyped: ...
def peak_widths(
    x, peaks, rel_height: float = 0.5, prominence_data: Untyped | None = None, wlen: Untyped | None = None
) -> Untyped: ...
def find_peaks(
    x,
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
    vector,
    widths,
    wavelet: Untyped | None = None,
    max_distances: Untyped | None = None,
    gap_thresh: Untyped | None = None,
    min_length: Untyped | None = None,
    min_snr: int = 1,
    noise_perc: int = 10,
    window_size: Untyped | None = None,
) -> Untyped: ...
