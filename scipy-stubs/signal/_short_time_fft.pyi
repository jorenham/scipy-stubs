from collections.abc import Callable
from typing import Literal

import numpy as np

from scipy._typing import Untyped
from scipy.signal import detrend as detrend
from scipy.signal.windows import get_window as get_window

PAD_TYPE: Untyped
FFT_MODE_TYPE: Untyped

class ShortTimeFFT:
    def __init__(
        self,
        win: np.ndarray,
        hop: int,
        fs: float,
        *,
        fft_mode: FFT_MODE_TYPE = "onesided",
        mfft: int | None = None,
        dual_win: np.ndarray | None = None,
        scale_to: Literal["magnitude", "psd"] | None = None,
        phase_shift: int | None = 0,
    ): ...
    @classmethod
    def from_dual(
        cls,
        dual_win: np.ndarray,
        hop: int,
        fs: float,
        *,
        fft_mode: FFT_MODE_TYPE = "onesided",
        mfft: int | None = None,
        scale_to: Literal["magnitude", "psd"] | None = None,
        phase_shift: int | None = 0,
    ): ...
    @classmethod
    def from_window(
        cls,
        win_param: str | tuple | float,
        fs: float,
        nperseg: int,
        noverlap: int,
        *,
        symmetric_win: bool = False,
        fft_mode: FFT_MODE_TYPE = "onesided",
        mfft: int | None = None,
        scale_to: Literal["magnitude", "psd"] | None = None,
        phase_shift: int | None = 0,
    ): ...
    @property
    def win(self) -> np.ndarray: ...
    @property
    def hop(self) -> int: ...
    @property
    def T(self) -> float: ...
    @T.setter
    def T(self, v: float): ...
    @property
    def fs(self) -> float: ...
    @fs.setter
    def fs(self, v: float): ...
    @property
    def fft_mode(self) -> FFT_MODE_TYPE: ...
    @fft_mode.setter
    def fft_mode(self, t: FFT_MODE_TYPE): ...
    @property
    def mfft(self) -> int: ...
    @mfft.setter
    def mfft(self, n_: int): ...
    @property
    def scaling(self) -> Literal["magnitude", "psd"] | None: ...
    def scale_to(self, scaling: Literal["magnitude", "psd"]): ...
    @property
    def phase_shift(self) -> int | None: ...
    @phase_shift.setter
    def phase_shift(self, v: int | None): ...
    def stft(
        self,
        x: np.ndarray,
        p0: int | None = None,
        p1: int | None = None,
        *,
        k_offset: int = 0,
        padding: PAD_TYPE = "zeros",
        axis: int = -1,
    ) -> np.ndarray: ...
    def stft_detrend(
        self,
        x: np.ndarray,
        detr: Callable[[np.ndarray], np.ndarray] | Literal["linear", "constant"] | None,
        p0: int | None = None,
        p1: int | None = None,
        *,
        k_offset: int = 0,
        padding: PAD_TYPE = "zeros",
        axis: int = -1,
    ) -> np.ndarray: ...
    def spectrogram(
        self,
        x: np.ndarray,
        y: np.ndarray | None = None,
        detr: Callable[[np.ndarray], np.ndarray] | Literal["linear", "constant"] | None = None,
        *,
        p0: int | None = None,
        p1: int | None = None,
        k_offset: int = 0,
        padding: PAD_TYPE = "zeros",
        axis: int = -1,
    ) -> np.ndarray: ...
    @property
    def dual_win(self) -> np.ndarray: ...
    @property
    def invertible(self) -> bool: ...
    def istft(self, S: np.ndarray, k0: int = 0, k1: int | None = None, *, f_axis: int = -2, t_axis: int = -1) -> np.ndarray: ...
    @property
    def fac_magnitude(self) -> float: ...
    @property
    def fac_psd(self) -> float: ...
    @property
    def m_num(self) -> int: ...
    @property
    def m_num_mid(self) -> int: ...
    @property
    def k_min(self) -> int: ...
    @property
    def p_min(self) -> int: ...
    def k_max(self, n: int) -> int: ...
    def p_max(self, n: int) -> int: ...
    def p_num(self, n: int) -> int: ...
    @property
    def lower_border_end(self) -> tuple[int, int]: ...
    def upper_border_begin(self, n: int) -> tuple[int, int]: ...
    @property
    def delta_t(self) -> float: ...
    def p_range(self, n: int, p0: int | None = None, p1: int | None = None) -> tuple[int, int]: ...
    def t(self, n: int, p0: int | None = None, p1: int | None = None, k_offset: int = 0) -> np.ndarray: ...
    def nearest_k_p(self, k: int, left: bool = True) -> int: ...
    @property
    def delta_f(self) -> float: ...
    @property
    def f_pts(self) -> int: ...
    @property
    def onesided_fft(self) -> bool: ...
    @property
    def f(self) -> np.ndarray: ...
    def extent(
        self, n: int, axes_seq: Literal["tf", "ft"] = "tf", center_bins: bool = False
    ) -> tuple[float, float, float, float]: ...
