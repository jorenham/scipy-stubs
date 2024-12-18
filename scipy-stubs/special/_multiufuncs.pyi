from collections.abc import Callable, Iterable
from typing import Any, Final, Generic, Literal as L, ParamSpec, Protocol, TypeAlias, overload, type_check_only
from typing_extensions import TypeVar, override

import numpy as np
import optype.numpy as onp
import optype.typing as opt

__all__ = [
    "assoc_legendre_p",
    "assoc_legendre_p_all",
    "legendre_p",
    "legendre_p_all",
    "sph_harm_y",
    "sph_harm_y_all",
    "sph_legendre_p",
    "sph_legendre_p_all",
]

_Tss = ParamSpec("_Tss")
_RT = TypeVar("_RT")
_UFuncT_co = TypeVar("_UFuncT_co", bound=Callable[..., object], default=Callable[..., Any], covariant=True)

_Complex: TypeAlias = np.complex64 | np.complex128  # `clongdouble` isn't supported
_ToJustComplex: TypeAlias = opt.Just[complex] | _Complex
_ToJustComplexND: TypeAlias = onp.CanArrayND[_Complex] | onp.SequenceND[onp.CanArrayND[_Complex]] | onp.SequenceND[_ToJustComplex]

_BranchCut: TypeAlias = L[2, 3]

###

@type_check_only
class _LegendreP(Protocol):
    @overload
    def __call__(self, /, n: onp.ToJustInt, z: onp.ToFloat, *, diff_n: opt.JustInt = 0) -> onp.Array1D[np.float64]: ...
    @overload
    def __call__(
        self,
        /,
        n: onp.ToJustInt,
        z: onp.ToFloatND,
        *,
        diff_n: opt.JustInt = 0,
    ) -> onp.Array[onp.AtLeast2D, np.float64]: ...
    @overload
    def __call__(
        self,
        /,
        n: onp.ToJustIntND,
        z: onp.ToFloat | onp.ToFloatND,
        *,
        diff_n: opt.JustInt = 0,
    ) -> onp.Array[onp.AtLeast2D, np.float64]: ...

@type_check_only
class _LegendrePAll(Protocol):
    @overload
    def __call__(self, /, n: onp.ToJustInt, z: onp.ToFloat, *, diff_n: opt.JustInt = 0) -> onp.Array3D[np.float64]: ...
    @overload
    def __call__(
        self,
        /,
        n: onp.ToJustInt,
        z: onp.ToFloat | onp.ToFloatND,
        *,
        diff_n: opt.JustInt = 0,
    ) -> onp.Array[onp.AtLeast3D, np.float64]: ...

@type_check_only
class _AssocLegendreP(Protocol):
    @overload  # real scalar-like
    def __call__(
        self,
        /,
        n: onp.ToJustInt,
        m: onp.ToJustInt,
        z: onp.ToFloat,
        *,
        branch_cut: _BranchCut = 2,
        norm: onp.ToBool = False,
        diff_n: opt.JustInt = 0,
    ) -> onp.Array1D[np.float64]: ...
    @overload  # complex scalar-like
    def __call__(
        self,
        /,
        n: onp.ToJustInt,
        m: onp.ToJustInt,
        z: _ToJustComplex,
        *,
        branch_cut: _BranchCut = 2,
        norm: onp.ToBool = False,
        diff_n: opt.JustInt = 0,
    ) -> onp.Array1D[np.complex128]: ...
    @overload  # real array-like
    def __call__(
        self,
        /,
        n: onp.ToJustInt | onp.ToJustIntND,
        m: onp.ToJustInt | onp.ToJustIntND,
        z: onp.ToFloat | onp.ToFloatND,
        *,
        branch_cut: _BranchCut | onp.ToJustIntND = 2,
        norm: onp.ToBool = False,
        diff_n: opt.JustInt = 0,
    ) -> onp.Array[onp.AtLeast1D, np.float64]: ...
    @overload  # complex array-like
    def __call__(
        self,
        /,
        n: onp.ToJustInt | onp.ToJustIntND,
        m: onp.ToJustInt | onp.ToJustIntND,
        z: _ToJustComplexND,
        *,
        branch_cut: _BranchCut | onp.ToJustIntND = 2,
        norm: onp.ToBool = False,
        diff_n: opt.JustInt = 0,
    ) -> onp.Array[onp.AtLeast1D, np.complex128]: ...

@type_check_only
class _AssocLegendrePAll(Protocol):
    @overload  # real scalar-like
    def __call__(
        self,
        /,
        n: onp.ToJustInt,
        m: onp.ToJustInt,
        z: onp.ToFloat,
        *,
        branch_cut: _BranchCut = 2,
        norm: onp.ToBool = False,
        diff_n: opt.JustInt = 0,
    ) -> onp.Array3D[np.float64]: ...
    @overload  # complex scalar-like
    def __call__(
        self,
        /,
        n: onp.ToJustInt,
        m: onp.ToJustInt,
        z: _ToJustComplex,
        *,
        branch_cut: _BranchCut = 2,
        norm: onp.ToBool = False,
        diff_n: opt.JustInt = 0,
    ) -> onp.Array3D[np.complex128]: ...
    @overload  # real array-like
    def __call__(
        self,
        /,
        n: onp.ToJustInt,
        m: onp.ToJustInt,
        z: onp.ToFloat | onp.ToFloatND,
        *,
        branch_cut: _BranchCut | onp.ToJustIntND = 2,
        norm: onp.ToBool = False,
        diff_n: opt.JustInt = 0,
    ) -> onp.Array[onp.AtLeast3D, np.float64]: ...
    @overload  # complex array-like
    def __call__(
        self,
        /,
        n: onp.ToJustInt,
        m: onp.ToJustInt,
        z: _ToJustComplexND,
        *,
        branch_cut: _BranchCut | onp.ToJustIntND = 2,
        norm: onp.ToBool = False,
        diff_n: opt.JustInt = 0,
    ) -> onp.Array[onp.AtLeast3D, np.complex128]: ...

@type_check_only
class _SphLegendreP(Protocol):
    @overload
    def __call__(
        self,
        /,
        n: onp.ToJustInt,
        m: onp.ToJustInt,
        theta: onp.ToFloat,
        *,
        diff_n: opt.JustInt = 0,
    ) -> onp.Array1D[np.float64]: ...
    @overload
    def __call__(
        self,
        /,
        n: onp.ToJustInt | onp.ToJustIntND,
        m: onp.ToJustInt | onp.ToJustIntND,
        theta: onp.ToFloatND,
        *,
        diff_n: opt.JustInt = 0,
    ) -> onp.Array[onp.AtLeast2D, np.float64]: ...
    @overload
    def __call__(
        self,
        /,
        n: onp.ToJustInt | onp.ToJustIntND,
        m: onp.ToJustIntND,
        theta: onp.ToFloat | onp.ToFloatND,
        *,
        diff_n: opt.JustInt = 0,
    ) -> onp.Array[onp.AtLeast2D, np.float64]: ...
    @overload
    def __call__(
        self,
        /,
        n: onp.ToJustIntND,
        m: onp.ToJustInt | onp.ToJustIntND,
        theta: onp.ToFloat | onp.ToFloatND,
        *,
        diff_n: opt.JustInt = 0,
    ) -> onp.Array[onp.AtLeast2D, np.float64]: ...

@type_check_only
class _SphLegendrePAll(Protocol):
    @overload
    def __call__(
        self,
        /,
        n: onp.ToJustInt,
        m: onp.ToJustInt,
        theta: onp.ToFloat,
        *,
        diff_n: opt.JustInt = 0,
    ) -> onp.Array3D[np.float64]: ...
    @overload
    def __call__(
        self,
        /,
        n: onp.ToJustInt,
        m: onp.ToJustInt,
        theta: onp.ToFloat | onp.ToFloatND,
        *,
        diff_n: opt.JustInt = 0,
    ) -> onp.Array[onp.AtLeast3D, np.float64]: ...

@type_check_only
class _SphHarmY(Protocol):
    @overload
    def __call__(
        self,
        /,
        n: onp.ToJustInt,
        m: onp.ToJustInt,
        theta: onp.ToFloat,
        phi: onp.ToFloat,
        *,
        diff_n: opt.JustInt = 0,
    ) -> onp.Array0D[np.float64]: ...
    @overload
    def __call__(
        self,
        /,
        n: onp.ToJustInt | onp.ToJustIntND,
        m: onp.ToJustInt | onp.ToJustIntND,
        theta: onp.ToFloat | onp.ToFloatND,
        phi: onp.ToFloatND,
        *,
        diff_n: opt.JustInt = 0,
    ) -> onp.Array[onp.AtLeast1D, np.float64]: ...
    @overload
    def __call__(
        self,
        /,
        n: onp.ToJustInt | onp.ToJustIntND,
        m: onp.ToJustInt | onp.ToJustIntND,
        theta: onp.ToFloatND,
        phi: onp.ToFloat | onp.ToFloatND,
        *,
        diff_n: opt.JustInt = 0,
    ) -> onp.Array[onp.AtLeast1D, np.float64]: ...
    @overload
    def __call__(
        self,
        /,
        n: onp.ToJustInt | onp.ToJustIntND,
        m: onp.ToJustIntND,
        theta: onp.ToFloat | onp.ToFloatND,
        phi: onp.ToFloat | onp.ToFloatND,
        *,
        diff_n: opt.JustInt = 0,
    ) -> onp.Array[onp.AtLeast1D, np.float64]: ...
    @overload
    def __call__(
        self,
        /,
        n: onp.ToJustIntND,
        m: onp.ToJustInt | onp.ToJustIntND,
        theta: onp.ToFloat | onp.ToFloatND,
        phi: onp.ToFloat | onp.ToFloatND,
        *,
        diff_n: opt.JustInt = 0,
    ) -> onp.Array[onp.AtLeast1D, np.float64]: ...

@type_check_only
class _SphHarmYAll(Protocol):
    @overload
    def __call__(
        self,
        /,
        n: onp.ToJustInt,
        m: onp.ToJustInt,
        theta: onp.ToFloat,
        phi: onp.ToFloat,
        *,
        diff_n: opt.JustInt = 0,
    ) -> onp.Array3D[np.complex128]: ...
    @overload
    def __call__(
        self,
        /,
        n: onp.ToJustInt,
        m: onp.ToJustInt,
        theta: onp.ToFloat | onp.ToFloatND,
        *,
        diff_n: opt.JustInt = 0,
    ) -> onp.Array[onp.AtLeast3D, np.complex128]: ...

###

class MultiUFunc(Generic[_UFuncT_co]):
    @property
    @override
    def __doc__(self, /) -> str | None: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleVariableOverride]

    #
    def __init__(
        self,
        /,
        ufunc_or_ufuncs: _UFuncT_co | Iterable[_UFuncT_co],
        doc: str | None = None,
        *,
        force_complex_output: bool = False,
        **default_kwargs: object,
    ) -> None: ...
    def __call__(self: MultiUFunc[Callable[_Tss, _RT]], /, *args: _Tss.args, **kwargs: _Tss.kwargs) -> _RT: ...

###

legendre_p: Final[MultiUFunc[_LegendreP]] = ...
legendre_p_all: Final[MultiUFunc[_LegendrePAll]] = ...

assoc_legendre_p: Final[MultiUFunc[_AssocLegendreP]] = ...
assoc_legendre_p_all: Final[MultiUFunc[_AssocLegendrePAll]] = ...

sph_legendre_p: Final[MultiUFunc[_SphLegendreP]] = ...
sph_legendre_p_all: Final[MultiUFunc[_SphLegendrePAll]] = ...

sph_harm_y: Final[MultiUFunc[_SphHarmY]] = ...
sph_harm_y_all: Final[MultiUFunc[_SphHarmYAll]] = ...
