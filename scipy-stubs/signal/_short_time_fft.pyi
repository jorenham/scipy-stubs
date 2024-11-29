from collections.abc import Callable
from typing import Any, Literal, TypeAlias
from typing_extensions import Self, Unpack

import numpy as np
import optype.numpy as onp
from .windows._windows import _Window, _WindowNeedsParams

__all__ = ["ShortTimeFFT"]

# awkward naming, but this matches the "attempts at type-aliases" in the implementation
PAD_TYPE: TypeAlias = Literal["zeros", "edge", "even", "odd"]
FFT_MODE_TYPE: TypeAlias = Literal["twosided", "centered", "onesided", "onesided2X"]

class ShortTimeFFT:
    def __init__(
        self,
        /,
        win: onp.ArrayND[np.inexact[Any]],
        hop: int,
        fs: float,
        *,
        fft_mode: FFT_MODE_TYPE = "onesided",
        mfft: int | None = None,
        dual_win: onp.ArrayND[np.inexact[Any]] | None = None,
        scale_to: Literal["magnitude", "psd"] | None = None,
        phase_shift: int | None = 0,
    ) -> None: ...
    @classmethod
    def from_dual(
        cls,
        dual_win: onp.ArrayND[np.inexact[Any]],
        hop: int,
        fs: float,
        *,
        fft_mode: FFT_MODE_TYPE = "onesided",
        mfft: int | None = None,
        scale_to: Literal["magnitude", "psd"] | None = None,
        phase_shift: int | None = 0,
    ) -> Self: ...
    @classmethod
    def from_window(
        cls,
        win_param: _Window | onp.ToFloat | tuple[_Window | _WindowNeedsParams, Unpack[tuple[object, ...]]],
        fs: float,
        nperseg: int,
        noverlap: int,
        *,
        symmetric_win: bool = False,
        fft_mode: FFT_MODE_TYPE = "onesided",
        mfft: int | None = None,
        scale_to: Literal["magnitude", "psd"] | None = None,
        phase_shift: int | None = 0,
    ) -> Self: ...
    @property
    def win(self, /) -> onp.ArrayND[np.inexact[Any]]: ...
    @property
    def hop(self, /) -> int: ...
    @property
    def T(self, /) -> float: ...
    @T.setter
    def T(self, /, v: float) -> None: ...
    @property
    def fs(self, /) -> float: ...
    @fs.setter
    def fs(self, /, v: float) -> None: ...
    @property
    def fft_mode(self, /) -> FFT_MODE_TYPE: ...
    @fft_mode.setter
    def fft_mode(self, /, t: FFT_MODE_TYPE) -> None: ...
    @property
    def mfft(self, /) -> int: ...
    @mfft.setter
    def mfft(self, /, n_: int) -> None: ...
    @property
    def scaling(self, /) -> Literal["magnitude", "psd"] | None: ...
    def scale_to(self, /, scaling: Literal["magnitude", "psd"]) -> None: ...
    @property
    def phase_shift(self, /) -> int | None: ...
    @phase_shift.setter
    def phase_shift(self, /, v: int | None) -> None: ...
    def stft(
        self,
        /,
        x: onp.ArrayND[np.inexact[Any]],
        p0: int | None = None,
        p1: int | None = None,
        *,
        k_offset: int = 0,
        padding: PAD_TYPE = "zeros",
        axis: int = -1,
    ) -> onp.ArrayND[np.inexact[Any]]: ...
    def stft_detrend(
        self,
        /,
        x: onp.ArrayND[np.inexact[Any]],
        detr: Callable[[onp.ArrayND[np.inexact[Any]]], onp.ArrayND[np.inexact[Any]]] | Literal["linear", "constant"] | None,
        p0: int | None = None,
        p1: int | None = None,
        *,
        k_offset: int = 0,
        padding: PAD_TYPE = "zeros",
        axis: int = -1,
    ) -> onp.ArrayND[np.inexact[Any]]: ...
    def spectrogram(
        self,
        /,
        x: onp.ArrayND[np.inexact[Any]],
        y: onp.ArrayND[np.inexact[Any]] | None = None,
        detr: Callable[[onp.ArrayND[np.inexact[Any]]], onp.ArrayND[np.inexact[Any]]]
        | Literal["linear", "constant"]
        | None = None,
        *,
        p0: int | None = None,
        p1: int | None = None,
        k_offset: int = 0,
        padding: PAD_TYPE = "zeros",
        axis: int = -1,
    ) -> onp.ArrayND[np.inexact[Any]]: ...
    @property
    def dual_win(self, /) -> onp.ArrayND[np.inexact[Any]]: ...
    @property
    def invertible(self, /) -> bool: ...
    def istft(
        self,
        /,
        S: onp.ArrayND[np.inexact[Any]],
        k0: int = 0,
        k1: int | None = None,
        *,
        f_axis: int = -2,
        t_axis: int = -1,
    ) -> onp.ArrayND[np.inexact[Any]]: ...
    @property
    def fac_magnitude(self, /) -> float: ...
    @property
    def fac_psd(self, /) -> float: ...
    @property
    def m_num(self, /) -> int: ...
    @property
    def m_num_mid(self, /) -> int: ...
    @property
    def k_min(self, /) -> int: ...
    @property
    def p_min(self, /) -> int: ...
    def k_max(self, /, n: int) -> int: ...
    def p_max(self, /, n: int) -> int: ...
    def p_num(self, /, n: int) -> int: ...
    @property
    def lower_border_end(self, /) -> tuple[int, int]: ...
    def upper_border_begin(self, /, n: int) -> tuple[int, int]: ...
    @property
    def delta_t(self, /) -> float: ...
    def p_range(self, /, n: int, p0: int | None = None, p1: int | None = None) -> tuple[int, int]: ...
    def t(self, /, n: int, p0: int | None = None, p1: int | None = None, k_offset: int = 0) -> onp.ArrayND[np.inexact[Any]]: ...
    def nearest_k_p(self, /, k: int, left: bool = True) -> int: ...
    @property
    def delta_f(self, /) -> float: ...
    @property
    def f_pts(self, /) -> int: ...
    @property
    def onesided_fft(self, /) -> bool: ...
    @property
    def f(self, /) -> onp.ArrayND[np.inexact[Any]]: ...
    def extent(
        self,
        /,
        n: int,
        axes_seq: Literal["tf", "ft"] = "tf",
        center_bins: bool = False,
    ) -> tuple[float, float, float, float]: ...
