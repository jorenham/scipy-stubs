from collections.abc import Callable
from typing import Any, Literal, TypeAlias
from typing_extensions import Self

import numpy as np
import optype.numpy as onp
from .windows._windows import _ToWindow

__all__ = ["ShortTimeFFT"]

###

_InexactND: TypeAlias = onp.ArrayND[np.inexact[Any]]

_PadType: TypeAlias = Literal["zeros", "edge", "even", "odd"]
_FFTModeType: TypeAlias = Literal["twosided", "centered", "onesided", "onesided2X"]
_ScaleTo: TypeAlias = Literal["magnitude", "psd"]
_Detr: TypeAlias = (
    Literal["linear", "constant"]
    | Callable[[onp.ArrayND[np.float64]], onp.ToComplexND]
    | Callable[[onp.ArrayND[np.complex128]], onp.ToComplexND]
)

###

class ShortTimeFFT:
    @property
    def win(self, /) -> _InexactND: ...
    @property
    def dual_win(self, /) -> _InexactND: ...
    @property
    def hop(self, /) -> int: ...
    @property
    def invertible(self, /) -> bool: ...
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
    @property
    def lower_border_end(self, /) -> tuple[int, int]: ...
    @property
    def delta_t(self, /) -> float: ...
    @property
    def delta_f(self, /) -> float: ...
    @property
    def f_pts(self, /) -> int: ...
    @property
    def f(self, /) -> _InexactND: ...
    @property
    def onesided_fft(self, /) -> bool: ...

    #
    @property
    def T(self, /) -> float: ...
    @T.setter
    def T(self, /, v: float) -> None: ...

    #
    @property
    def fs(self, /) -> float: ...
    @fs.setter
    def fs(self, /, v: float) -> None: ...

    #
    @property
    def fft_mode(self, /) -> _FFTModeType: ...
    @fft_mode.setter
    def fft_mode(self, /, t: _FFTModeType) -> None: ...

    #
    @property
    def mfft(self, /) -> int: ...
    @mfft.setter
    def mfft(self, /, n_: int) -> None: ...

    #
    @property
    def phase_shift(self, /) -> int | None: ...
    @phase_shift.setter
    def phase_shift(self, /, v: int | None) -> None: ...

    #
    @property
    def scaling(self, /) -> _ScaleTo | None: ...

    #
    def __init__(
        self,
        /,
        win: _InexactND,
        hop: int,
        fs: float,
        *,
        fft_mode: _FFTModeType = "onesided",
        mfft: int | None = None,
        dual_win: _InexactND | None = None,
        scale_to: _ScaleTo | None = None,
        phase_shift: int | None = 0,
    ) -> None: ...

    #
    def k_max(self, /, n: int) -> int: ...
    def p_max(self, /, n: int) -> int: ...
    def p_num(self, /, n: int) -> int: ...
    def nearest_k_p(self, /, k: int, left: bool = True) -> int: ...
    def upper_border_begin(self, /, n: int) -> tuple[int, int]: ...
    def p_range(self, /, n: int, p0: int | None = None, p1: int | None = None) -> tuple[int, int]: ...
    def t(self, /, n: int, p0: int | None = None, p1: int | None = None, k_offset: int = 0) -> _InexactND: ...
    def scale_to(self, /, scaling: _ScaleTo) -> None: ...

    #
    def stft(
        self,
        /,
        x: _InexactND,
        p0: int | None = None,
        p1: int | None = None,
        *,
        k_offset: int = 0,
        padding: _PadType = "zeros",
        axis: int = -1,
    ) -> _InexactND: ...
    def istft(
        self,
        /,
        S: _InexactND,
        k0: int = 0,
        k1: int | None = None,
        *,
        f_axis: int = -2,
        t_axis: int = -1,
    ) -> _InexactND: ...
    def stft_detrend(
        self,
        /,
        x: _InexactND,
        detr: _Detr | None,
        p0: int | None = None,
        p1: int | None = None,
        *,
        k_offset: int = 0,
        padding: _PadType = "zeros",
        axis: int = -1,
    ) -> _InexactND: ...
    def spectrogram(
        self,
        /,
        x: _InexactND,
        y: _InexactND | None = None,
        detr: _Detr | None = None,
        *,
        p0: int | None = None,
        p1: int | None = None,
        k_offset: int = 0,
        padding: _PadType = "zeros",
        axis: int = -1,
    ) -> _InexactND: ...
    def extent(
        self,
        /,
        n: int,
        axes_seq: Literal["tf", "ft"] = "tf",
        center_bins: bool = False,
    ) -> tuple[float, float, float, float]: ...

    #
    @classmethod
    def from_dual(
        cls,
        dual_win: _InexactND,
        hop: int,
        fs: float,
        *,
        fft_mode: _FFTModeType = "onesided",
        mfft: int | None = None,
        scale_to: _ScaleTo | None = None,
        phase_shift: int | None = 0,
    ) -> Self: ...
    @classmethod
    def from_window(
        cls,
        win_param: _ToWindow,
        fs: float,
        nperseg: int,
        noverlap: int,
        *,
        symmetric_win: bool = False,
        fft_mode: _FFTModeType = "onesided",
        mfft: int | None = None,
        scale_to: _ScaleTo | None = None,
        phase_shift: int | None = 0,
    ) -> Self: ...
