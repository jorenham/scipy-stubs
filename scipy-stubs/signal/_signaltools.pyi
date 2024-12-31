from collections.abc import Callable, Sequence
from typing import Any, Literal as L, TypeAlias, TypedDict, TypeVar, overload, type_check_only

import numpy as np
import optype as op
import optype.numpy as onp
from scipy._typing import ConvMode, Falsy, Truthy
from ._ltisys import dlti
from .windows._windows import _ToWindow

__all__ = [
    "choose_conv_method",
    "convolve",
    "convolve2d",
    "correlate",
    "correlate2d",
    "correlation_lags",
    "decimate",
    "deconvolve",
    "detrend",
    "envelope",
    "fftconvolve",
    "filtfilt",
    "hilbert",
    "hilbert2",
    "invres",
    "invresz",
    "lfilter",
    "lfilter_zi",
    "lfiltic",
    "medfilt",
    "medfilt2d",
    "oaconvolve",
    "order_filter",
    "resample",
    "resample_poly",
    "residue",
    "residuez",
    "sosfilt",
    "sosfilt_zi",
    "sosfiltfilt",
    "unique_roots",
    "vectorstrength",
    "wiener",
]

###

_T = TypeVar("_T")
_InexactT = TypeVar("_InexactT", bound=np.inexact[Any])
_EnvelopeSCT = TypeVar("_EnvelopeSCT", bound=_OutFloat | np.longdouble | _Complex | np.clongdouble)
_FilterSCT = TypeVar("_FilterSCT", bound=_Int | _Float)
_ShapeT = TypeVar("_ShapeT", bound=tuple[int, ...])
_AnyShapeT = TypeVar("_AnyShapeT", tuple[int], tuple[int, int], tuple[int, int, int], tuple[int, ...])

_Tuple2: TypeAlias = tuple[_T, _T]

_ConvMethod: TypeAlias = L["direct", "fft"]
_ToConvMethod: TypeAlias = L["auto", _ConvMethod]
_BoundaryConditions: TypeAlias = L["fill", "wrap", "symm"]
_ResidueType: TypeAlias = L["avg", "min", "max"]
_RootType: TypeAlias = L[_ResidueType, "maximum", "avg", "mean"]
_Domain: TypeAlias = L["time", "freq"]
_TrendType: TypeAlias = L["linear", "constant"]
_PadType: TypeAlias = L["constant", "line", "mean", "median", "maximum", "minimum", "symmetric", "reflect", "edge", "wrap"]
_FiltFiltPadType: TypeAlias = L["odd", "even", "constant"] | None
_FiltFiltMethod: TypeAlias = L["pad", "gust"]
_ResidualKind: TypeAlias = L["lowpass", "all"]
_FilterType: TypeAlias = L["iir", "fir"] | dlti

_Bool: TypeAlias = np.bool_
_CoBool: TypeAlias = _Bool
_ToBool: TypeAlias = bool | _CoBool
_ToBoolND: TypeAlias = onp.CanArrayND[_CoBool] | onp.SequenceND[_ToBool] | onp.SequenceND[onp.CanArrayND[_CoBool]]

_Int: TypeAlias = np.integer[Any]
_CoInt: TypeAlias = _Bool | _Int
_ToInt: TypeAlias = int | _CoInt
_ToIntND: TypeAlias = onp.CanArrayND[_CoInt] | onp.SequenceND[_ToInt] | onp.SequenceND[onp.CanArrayND[_CoInt]]

_Float: TypeAlias = np.float16 | np.float32 | np.float64
_LFloat: TypeAlias = np.float64 | np.longdouble
_OutFloat: TypeAlias = np.float32 | np.float64
_CoFloat: TypeAlias = _CoInt | _Float
_ToFloat: TypeAlias = float | _CoFloat
_ToFloat1D: TypeAlias = onp.CanArrayND[_CoFloat] | Sequence[_ToFloat | onp.CanArray0D[_CoFloat]]
_ToFloat2D: TypeAlias = onp.CanArrayND[_CoFloat] | Sequence[_ToFloat1D] | Sequence[Sequence[_ToFloat | onp.CanArray0D[_CoFloat]]]
_ToFloatND: TypeAlias = onp.CanArrayND[_CoFloat] | onp.SequenceND[_ToFloat] | onp.SequenceND[onp.CanArrayND[_CoFloat]]

_Complex: TypeAlias = np.complex64 | np.complex128
_LComplex: TypeAlias = np.complex128 | np.clongdouble
_CoComplex: TypeAlias = _CoFloat | _Complex
_ToComplex: TypeAlias = complex | _CoComplex
_ToComplex1D: TypeAlias = onp.CanArrayND[_CoComplex] | Sequence[_ToComplex | onp.CanArray0D[_CoComplex]]
_ToComplex2D: TypeAlias = (
    onp.CanArrayND[_CoComplex] | Sequence[_ToComplex1D] | Sequence[Sequence[_ToComplex | onp.CanArray0D[_CoComplex]]]
)
_ToComplexND: TypeAlias = onp.CanArrayND[_CoComplex] | onp.SequenceND[_ToComplex] | onp.SequenceND[onp.CanArrayND[_CoFloat]]

_WindowFuncFloat: TypeAlias = Callable[[onp.Array1D[np.float64]], onp.ToFloat1D]
_WindowFuncComplex: TypeAlias = Callable[[onp.Array1D[np.complex128]], onp.ToFloat1D]

@type_check_only
class _ConvMeasureDict(TypedDict):
    direct: float
    fft: float

###

@overload
def choose_conv_method(
    in1: _ToIntND,
    in2: _ToIntND,
    mode: ConvMode = "full",
    measure: Falsy = False,
) -> L["direct"]: ...
@overload
def choose_conv_method(
    in1: _ToComplexND,
    in2: _ToComplexND,
    mode: ConvMode = "full",
    measure: Falsy = False,
) -> _ConvMethod: ...
@overload
def choose_conv_method(
    in1: _ToComplexND,
    in2: _ToComplexND,
    mode: ConvMode,
    measure: Truthy,
) -> tuple[_ConvMethod, _ConvMeasureDict]: ...
@overload
def choose_conv_method(
    in1: _ToComplexND,
    in2: _ToComplexND,
    mode: ConvMode = "full",
    *,
    measure: Truthy,
) -> tuple[_ConvMethod, _ConvMeasureDict]: ...

#
@overload
def convolve(
    in1: _ToBoolND,
    in2: _ToBoolND,
    mode: ConvMode = "full",
    method: _ToConvMethod = "auto",
) -> onp.ArrayND[_CoBool]: ...
@overload
def convolve(
    in1: _ToIntND,
    in2: _ToIntND,
    mode: ConvMode = "full",
    method: _ToConvMethod = "auto",
) -> onp.ArrayND[_CoInt]: ...
@overload
def convolve(
    in1: _ToFloatND,
    in2: _ToFloatND,
    mode: ConvMode = "full",
    method: _ToConvMethod = "auto",
) -> onp.ArrayND[_CoFloat]: ...
@overload
def convolve(
    in1: _ToComplexND,
    in2: _ToComplexND,
    mode: ConvMode = "full",
    method: _ToConvMethod = "auto",
) -> onp.ArrayND[_CoComplex]: ...

#
@overload
def convolve2d(
    in1: onp.ToInt2D,
    in2: onp.ToInt2D,
    mode: ConvMode = "full",
    boundary: _BoundaryConditions = "fill",
    fillvalue: onp.ToInt = 0,
) -> onp.Array2D[_Int]: ...
@overload
def convolve2d(
    in1: onp.ToFloat2D,
    in2: onp.ToFloat2D,
    mode: ConvMode = "full",
    boundary: _BoundaryConditions = "fill",
    fillvalue: onp.ToFloat = 0,
) -> onp.Array2D[_OutFloat | _Int]: ...
@overload
def convolve2d(
    in1: onp.ToComplex2D,
    in2: onp.ToComplex2D,
    mode: ConvMode = "full",
    boundary: _BoundaryConditions = "fill",
    fillvalue: onp.ToComplex = 0,
) -> onp.Array2D[_Complex | _OutFloat | _Int]: ...

#
@overload
def fftconvolve(  # type: ignore[overload-overlap]
    in1: onp.ArrayND[np.float16, _AnyShapeT],
    in2: onp.ArrayND[np.float16 | np.float32, _AnyShapeT],
    mode: ConvMode = "full",
    axes: None = None,
) -> onp.ArrayND[np.float32, _AnyShapeT]: ...
@overload
def fftconvolve(
    in1: onp.ArrayND[_EnvelopeSCT, _AnyShapeT],
    in2: onp.ArrayND[_EnvelopeSCT, _AnyShapeT],
    mode: ConvMode = "full",
    axes: None = None,
) -> onp.ArrayND[_EnvelopeSCT, _AnyShapeT]: ...
@overload
def fftconvolve(
    in1: onp.ToFloatND,
    in2: onp.ToFloatND,
    mode: ConvMode = "full",
    axes: op.CanIndex | Sequence[op.CanIndex] | None = None,
) -> onp.ArrayND[_OutFloat | np.longdouble]: ...
@overload
def fftconvolve(
    in1: onp.ToComplexND,
    in2: onp.ToComplexND,
    mode: ConvMode = "full",
    axes: op.CanIndex | Sequence[op.CanIndex] | None = None,
) -> onp.ArrayND[_OutFloat | np.longdouble | _Complex | np.clongdouble]: ...

#
@overload
def oaconvolve(  # type: ignore[overload-overlap]
    in1: onp.ArrayND[np.float16, _AnyShapeT],
    in2: onp.ArrayND[np.float16 | np.float32, _AnyShapeT],
    mode: ConvMode = "full",
    axes: None = None,
) -> onp.ArrayND[np.float32, _AnyShapeT]: ...
@overload
def oaconvolve(
    in1: onp.ArrayND[_EnvelopeSCT, _AnyShapeT],
    in2: onp.ArrayND[_EnvelopeSCT, _AnyShapeT],
    mode: ConvMode = "full",
    axes: None = None,
) -> onp.ArrayND[_EnvelopeSCT, _AnyShapeT]: ...
@overload
def oaconvolve(
    in1: onp.ToFloatND,
    in2: onp.ToFloatND,
    mode: ConvMode = "full",
    axes: op.CanIndex | Sequence[op.CanIndex] | None = None,
) -> onp.ArrayND[_OutFloat | np.longdouble]: ...
@overload
def oaconvolve(
    in1: onp.ToComplexND,
    in2: onp.ToComplexND,
    mode: ConvMode = "full",
    axes: op.CanIndex | Sequence[op.CanIndex] | None = None,
) -> onp.ArrayND[_OutFloat | np.longdouble | _Complex | np.clongdouble]: ...

#
@overload
def deconvolve(signal: onp.ToFloat1D, divisor: onp.ToFloat1D) -> _Tuple2[onp.Array1D[_LFloat]]: ...
@overload
def deconvolve(signal: onp.ToComplex1D, divisor: onp.ToComplex1D) -> _Tuple2[onp.Array1D[_LFloat | _LComplex]]: ...

#
@overload
def correlate(
    in1: _ToBoolND,
    in2: _ToBoolND,
    mode: ConvMode = "full",
    method: _ToConvMethod = "auto",
) -> onp.ArrayND[_CoBool]: ...
@overload
def correlate(
    in1: _ToIntND,
    in2: _ToIntND,
    mode: ConvMode = "full",
    method: _ToConvMethod = "auto",
) -> onp.ArrayND[_CoInt]: ...
@overload
def correlate(
    in1: _ToFloatND,
    in2: _ToFloatND,
    mode: ConvMode = "full",
    method: _ToConvMethod = "auto",
) -> onp.ArrayND[_CoFloat]: ...
@overload
def correlate(
    in1: _ToComplexND,
    in2: _ToComplexND,
    mode: ConvMode = "full",
    method: _ToConvMethod = "auto",
) -> onp.ArrayND[_CoComplex]: ...

#
@overload
def correlate2d(
    in1: onp.ToInt2D,
    in2: onp.ToInt2D,
    mode: ConvMode = "full",
    boundary: _BoundaryConditions = "fill",
    fillvalue: onp.ToInt = 0,
) -> onp.Array2D[_Int]: ...
@overload
def correlate2d(
    in1: onp.ToFloat2D,
    in2: onp.ToFloat2D,
    mode: ConvMode = "full",
    boundary: _BoundaryConditions = "fill",
    fillvalue: onp.ToFloat = 0,
) -> onp.Array2D[_OutFloat | _Int]: ...
@overload
def correlate2d(
    in1: onp.ToComplex2D,
    in2: onp.ToComplex2D,
    mode: ConvMode = "full",
    boundary: _BoundaryConditions = "fill",
    fillvalue: onp.ToComplex = 0,
) -> onp.Array2D[_Complex | _OutFloat | _Int]: ...

#
def correlation_lags(in1_len: onp.ToInt, in2_len: onp.ToInt, mode: ConvMode = "full") -> onp.Array1D[np.int_]: ...

#
@overload
def lfilter_zi(b: _ToFloat1D, a: _ToFloat1D) -> onp.Array1D[_Float]: ...
@overload
def lfilter_zi(b: _ToComplex1D, a: _ToComplex1D) -> onp.Array1D[_Float | _Complex]: ...

#
@overload
def lfiltic(
    b: onp.ToFloat1D,
    a: onp.ToFloat1D,
    y: onp.ToFloat1D,
    x: onp.ToFloat1D | None = None,
) -> onp.Array1D[np.floating[Any]]: ...
@overload
def lfiltic(
    b: onp.ToComplex1D,
    a: onp.ToComplex1D,
    y: onp.ToComplex1D,
    x: onp.ToComplex1D | None = None,
) -> onp.Array1D[np.inexact[Any]]: ...

#
@overload
def lfilter(
    b: _ToFloat1D,
    a: _ToFloat1D,
    x: _ToFloatND,
    axis: op.CanIndex = -1,
    zi: None = None,
) -> onp.ArrayND[_Float]: ...
@overload
def lfilter(
    b: _ToFloat1D,
    a: _ToFloat1D,
    x: _ToFloatND,
    axis: op.CanIndex,
    zi: _ToFloatND,
) -> _Tuple2[onp.ArrayND[_Float]]: ...
@overload
def lfilter(
    b: _ToFloat1D,
    a: _ToFloat1D,
    x: _ToFloatND,
    axis: op.CanIndex = -1,
    *,
    zi: _ToFloatND,
) -> _Tuple2[onp.ArrayND[_Float]]: ...
@overload
def lfilter(
    b: _ToComplex1D,
    a: _ToComplex1D,
    x: _ToComplexND,
    axis: op.CanIndex = -1,
    zi: None = None,
) -> onp.ArrayND[_Complex | _Float]: ...
@overload
def lfilter(
    b: _ToComplex1D,
    a: _ToComplex1D,
    x: _ToComplexND,
    axis: op.CanIndex,
    zi: _ToComplexND,
) -> _Tuple2[onp.ArrayND[_Complex | _Float]]: ...
@overload
def lfilter(
    b: _ToComplex1D,
    a: _ToComplex1D,
    x: _ToComplexND,
    axis: op.CanIndex = -1,
    *,
    zi: _ToComplexND,
) -> _Tuple2[onp.ArrayND[_Complex | _Float]]: ...

#
@overload
def filtfilt(
    b: _ToFloat1D,
    a: _ToFloat1D,
    x: _ToFloatND,
    axis: op.CanIndex = -1,
    padtype: _FiltFiltPadType = "odd",
    padlen: onp.ToInt | None = None,
    method: _FiltFiltMethod = "pad",
    irlen: onp.ToInt | None = None,
) -> onp.ArrayND[_Float]: ...
@overload
def filtfilt(
    b: _ToComplex1D,
    a: _ToComplex1D,
    x: _ToComplexND,
    axis: op.CanIndex = -1,
    padtype: _FiltFiltPadType = "odd",
    padlen: onp.ToInt | None = None,
    method: _FiltFiltMethod = "pad",
    irlen: onp.ToInt | None = None,
) -> onp.ArrayND[_Complex | _Float]: ...

#
@overload
def sosfilt_zi(sos: onp.ArrayND[_InexactT]) -> onp.Array2D[_InexactT]: ...
@overload
def sosfilt_zi(sos: onp.ToFloat2D) -> onp.Array2D[np.floating[Any]]: ...
@overload
def sosfilt_zi(sos: onp.ToComplex2D) -> onp.Array2D[np.inexact[Any]]: ...

#
@overload
def sosfilt(sos: _ToFloat2D, x: _ToFloatND, axis: op.CanIndex = -1, zi: None = None) -> onp.ArrayND[_Float]: ...
@overload
def sosfilt(sos: _ToFloat2D, x: _ToFloatND, axis: op.CanIndex, zi: _ToFloatND) -> _Tuple2[onp.ArrayND[_Float]]: ...
@overload
def sosfilt(sos: _ToFloat2D, x: _ToFloatND, axis: op.CanIndex = -1, *, zi: _ToFloatND) -> _Tuple2[onp.ArrayND[_Float]]: ...
@overload
def sosfilt(sos: _ToComplex2D, x: _ToComplexND, axis: op.CanIndex = -1, zi: None = None) -> onp.ArrayND[_Float | _Complex]: ...
@overload
def sosfilt(
    sos: _ToComplex2D,
    x: _ToComplexND,
    axis: op.CanIndex,
    zi: _ToFloatND,
) -> _Tuple2[onp.ArrayND[_Float | _Complex]]: ...
@overload
def sosfilt(
    sos: _ToComplex2D,
    x: _ToComplexND,
    axis: op.CanIndex = -1,
    *,
    zi: _ToFloatND,
) -> _Tuple2[onp.ArrayND[_Float | _Complex]]: ...

#
@overload
def sosfiltfilt(
    sos: _ToFloat2D,
    x: _ToFloatND,
    axis: op.CanIndex = -1,
    padtype: _FiltFiltPadType = "odd",
    padlen: onp.ToInt | None = None,
) -> onp.ArrayND[_Float]: ...
@overload
def sosfiltfilt(
    sos: _ToComplex2D,
    x: _ToComplexND,
    axis: op.CanIndex = -1,
    padtype: _FiltFiltPadType = "odd",
    padlen: onp.ToInt | None = None,
) -> onp.ArrayND[_Float | _Complex]: ...

#
@overload
def order_filter(
    a: onp.ArrayND[_FilterSCT, _ShapeT],
    domain: onp.ToArrayND,
    rank: onp.ToJustInt,
) -> onp.ArrayND[_FilterSCT, _ShapeT]: ...
@overload
def order_filter(a: _ToIntND, domain: onp.ToArrayND, rank: onp.ToJustInt) -> onp.ArrayND[_Int]: ...
@overload
def order_filter(a: _ToFloatND, domain: onp.ToArrayND, rank: onp.ToJustInt) -> onp.ArrayND[_Float | _Int]: ...

#
@overload
def medfilt(
    volume: onp.ArrayND[_FilterSCT, _ShapeT],
    kernel_size: onp.ToInt | onp.ToInt1D | None = None,
) -> onp.ArrayND[_FilterSCT, _ShapeT]: ...
@overload
def medfilt(volume: _ToIntND, kernel_size: onp.ToInt | onp.ToInt1D | None = None) -> onp.ArrayND[_Int]: ...
@overload
def medfilt(volume: _ToFloatND, kernel_size: onp.ToInt | onp.ToInt1D | None = None) -> onp.ArrayND[_Float | _Int]: ...

#
def medfilt2d(input: _ToFloat2D, kernel_size: onp.ToInt | onp.ToInt1D = 3) -> onp.Array2D[_Float]: ...

#
@overload
def wiener(
    im: onp.ToFloatND,
    mysize: onp.ToInt | onp.ToInt1D | None = None,
    noise: onp.ToFloat | None = None,
) -> onp.ArrayND[_LFloat]: ...
@overload
def wiener(
    im: onp.ToComplexND,
    mysize: onp.ToInt | onp.ToInt1D | None = None,
    noise: onp.ToFloat | None = None,
) -> onp.ArrayND[_LFloat | _LComplex]: ...

#
def hilbert(
    x: onp.ToFloatND,
    N: onp.ToInt | None = None,
    axis: op.CanIndex = -1,
) -> onp.ArrayND[np.complexfloating[Any, Any]]: ...

#
def hilbert2(
    x: onp.ToFloat2D,
    N: onp.ToInt | tuple[onp.ToInt, onp.ToInt] | None = None,
) -> onp.Array2D[np.complexfloating[Any, Any]]: ...

#
@overload
def unique_roots(
    p: onp.ToFloat1D,
    tol: onp.ToFloat = 0.001,
    rtype: _RootType = "min",
) -> tuple[onp.Array1D[np.floating[Any]], onp.Array1D[np.int_]]: ...
@overload
def unique_roots(
    p: onp.ToComplex1D,
    tol: onp.ToFloat = 0.001,
    rtype: _RootType = "min",
) -> tuple[onp.Array1D[np.inexact[Any]], onp.Array1D[np.int_]]: ...

#
def residue(
    b: onp.ToComplex1D,
    a: onp.ToComplex1D,
    tol: onp.ToFloat = 0.001,
    rtype: _ResidueType = "avg",
) -> tuple[onp.Array1D[np.complex128], onp.Array1D[np.complex128], onp.Array1D[np.float64]]: ...

#
def residuez(
    b: onp.ToComplex1D,
    a: onp.ToComplex1D,
    tol: onp.ToFloat = 0.001,
    rtype: _ResidueType = "avg",
) -> tuple[onp.Array1D[np.complex128], onp.Array1D[np.complex128], onp.Array1D[np.float64]]: ...

#
def invres(
    r: onp.ToComplex1D,
    p: onp.ToComplex1D,
    k: onp.ToFloat1D,
    tol: onp.ToFloat = 0.001,
    rtype: _ResidueType = "avg",
) -> tuple[onp.Array1D[np.complex128], onp.Array1D[np.complex128]]: ...
def invresz(
    r: onp.ToComplex1D,
    p: onp.ToComplex1D,
    k: onp.ToFloat1D,
    tol: onp.ToFloat = 0.001,
    rtype: _ResidueType = "avg",
) -> tuple[onp.Array1D[np.complex128], onp.Array1D[np.complex128]]: ...

#
@overload
def resample(
    x: onp.ArrayND[_EnvelopeSCT, _AnyShapeT],
    num: onp.ToJustInt,
    t: None = None,
    axis: op.CanIndex = 0,
    window: _WindowFuncFloat | _WindowFuncComplex | onp.ToFloat1D | _ToWindow | None = None,
    domain: _Domain = "time",
) -> onp.ArrayND[_EnvelopeSCT, _AnyShapeT]: ...
@overload
def resample(
    x: onp.ToFloatND,
    num: onp.ToJustInt,
    t: None = None,
    axis: op.CanIndex = 0,
    window: _WindowFuncFloat | onp.ToFloat1D | _ToWindow | None = None,
    domain: _Domain = "time",
) -> onp.ArrayND[np.floating[Any]]: ...
@overload
def resample(
    x: onp.ToComplexND,
    num: onp.ToJustInt,
    t: None = None,
    axis: op.CanIndex = 0,
    window: _WindowFuncComplex | onp.ToFloat1D | _ToWindow | None = None,
    domain: _Domain = "time",
) -> onp.ArrayND[np.inexact[Any]]: ...
@overload
def resample(
    x: onp.ArrayND[_EnvelopeSCT, _AnyShapeT],
    num: onp.ToJustInt,
    t: onp.ToFloat1D,
    axis: op.CanIndex = 0,
    window: _WindowFuncFloat | _WindowFuncComplex | onp.ToFloat1D | _ToWindow | None = None,
    domain: _Domain = "time",
) -> tuple[onp.ArrayND[_EnvelopeSCT, _AnyShapeT], onp.Array1D[np.floating[Any]]]: ...
@overload
def resample(
    x: onp.ToFloatND,
    num: onp.ToJustInt,
    t: onp.ToFloat1D,
    axis: op.CanIndex = 0,
    window: _WindowFuncFloat | onp.ToFloat1D | _ToWindow | None = None,
    domain: _Domain = "time",
) -> tuple[onp.ArrayND[np.floating[Any]], onp.Array1D[np.floating[Any]]]: ...
@overload
def resample(
    x: onp.ToComplexND,
    num: onp.ToJustInt,
    t: onp.ToFloat1D,
    axis: op.CanIndex = 0,
    window: _WindowFuncComplex | onp.ToFloat1D | _ToWindow | None = None,
    domain: _Domain = "time",
) -> tuple[onp.ArrayND[np.inexact[Any]], onp.Array1D[np.floating[Any]]]: ...

#
@overload
def resample_poly(
    x: _ToFloatND,
    up: onp.ToInt,
    down: onp.ToInt,
    axis: op.CanIndex = 0,
    window: _ToWindow = ("kaiser", 5.0),
    padtype: _PadType = "constant",
    cval: onp.ToFloat | None = None,
) -> onp.ArrayND[_Float]: ...
@overload
def resample_poly(
    x: _ToComplexND,
    up: onp.ToInt,
    down: onp.ToInt,
    axis: op.CanIndex = 0,
    window: _ToWindow = ("kaiser", 5.0),
    padtype: _PadType = "constant",
    cval: onp.ToFloat | None = None,
) -> onp.ArrayND[_Complex | _Float]: ...

#
@overload
def vectorstrength(events: onp.ToFloat1D, period: onp.ToFloat) -> _Tuple2[_Float | np.longdouble]: ...
@overload
def vectorstrength(events: onp.ToFloat1D, period: onp.ToFloat1D) -> _Tuple2[onp.Array1D[_Float | np.longdouble]]: ...
@overload
def vectorstrength(
    events: onp.ToComplex1D,
    period: onp.ToComplex,
) -> _Tuple2[_Float | np.longdouble | _Complex | np.clongdouble]: ...
@overload
def vectorstrength(
    events: onp.ToComplex1D,
    period: onp.ToComplex1D,
) -> _Tuple2[onp.Array1D[_Float | np.longdouble | _Complex | np.clongdouble]]: ...

#
@overload
def detrend(
    data: onp.ToFloatND,
    axis: op.CanIndex = -1,
    type: _TrendType = "linear",
    bp: onp.ToJustInt | onp.ToJustIntND = 0,
    overwrite_data: op.CanBool = False,
) -> onp.ArrayND[_Float]: ...
@overload
def detrend(
    data: onp.ToComplexND,
    axis: op.CanIndex = -1,
    type: _TrendType = "linear",
    bp: onp.ToJustInt | onp.ToJustIntND = 0,
    overwrite_data: op.CanBool = False,
) -> onp.ArrayND[_Complex | _Float]: ...

#
@overload
def decimate(
    x: onp.ToFloatND,
    q: onp.ToInt,
    n: onp.ToInt | None = None,
    ftype: _FilterType = "iir",
    axis: op.CanIndex = -1,
    zero_phase: op.CanBool = True,
) -> onp.ArrayND[_Float | np.longdouble]: ...
@overload
def decimate(
    x: onp.ToComplexND,
    q: onp.ToInt,
    n: onp.ToInt | None = None,
    ftype: _FilterType = "iir",
    axis: op.CanIndex = -1,
    zero_phase: op.CanBool = True,
) -> onp.ArrayND[_Float | np.longdouble | _Complex | np.clongdouble]: ...

#
@overload
def envelope(
    z: onp.Array1D[np.float16],
    bp_in: tuple[_ToInt | None, _ToInt | None] = (1, None),
    *,
    n_out: onp.ToInt | None = None,
    squared: op.CanBool = False,
    residual: _ResidualKind | None = "lowpass",
    axis: op.CanIndex = -1,
) -> onp.Array2D[np.float32]: ...
@overload
def envelope(
    z: onp.Array2D[np.float16],
    bp_in: tuple[_ToInt | None, _ToInt | None] = (1, None),
    *,
    n_out: onp.ToInt | None = None,
    squared: op.CanBool = False,
    residual: _ResidualKind | None = "lowpass",
    axis: op.CanIndex = -1,
) -> onp.Array3D[np.float32]: ...
@overload
def envelope(  # type: ignore[overload-overlap]
    z: onp.ArrayND[np.float16],
    bp_in: tuple[_ToInt | None, _ToInt | None] = (1, None),
    *,
    n_out: onp.ToInt | None = None,
    squared: op.CanBool = False,
    residual: _ResidualKind | None = "lowpass",
    axis: op.CanIndex = -1,
) -> onp.ArrayND[np.float32]: ...
@overload
def envelope(
    z: onp.Array1D[_EnvelopeSCT],
    bp_in: tuple[_ToInt | None, _ToInt | None] = (1, None),
    *,
    n_out: onp.ToInt | None = None,
    squared: op.CanBool = False,
    residual: _ResidualKind | None = "lowpass",
    axis: op.CanIndex = -1,
) -> onp.Array2D[_EnvelopeSCT]: ...
@overload
def envelope(
    z: onp.Array2D[_EnvelopeSCT],
    bp_in: tuple[_ToInt | None, _ToInt | None] = (1, None),
    *,
    n_out: onp.ToInt | None = None,
    squared: op.CanBool = False,
    residual: _ResidualKind | None = "lowpass",
    axis: op.CanIndex = -1,
) -> onp.Array3D[_EnvelopeSCT]: ...
@overload
def envelope(
    z: onp.ArrayND[_EnvelopeSCT],
    bp_in: tuple[_ToInt | None, _ToInt | None] = (1, None),
    *,
    n_out: onp.ToInt | None = None,
    squared: op.CanBool = False,
    residual: _ResidualKind | None = "lowpass",
    axis: op.CanIndex = -1,
) -> onp.ArrayND[_EnvelopeSCT]: ...
