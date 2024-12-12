# mypy: disable-error-code="explicit-override, override, misc"
# pyright: reportIncompatibleMethodOverride=false, reportIncompatibleVariableOverride=false, reportImplicitOverride=false

from types import EllipsisType
from typing import Any, Generic, Literal as L, TypeAlias, TypedDict, final, overload, type_check_only
from typing_extensions import LiteralString, Never, TypeVar, Unpack, override

import numpy as np
import optype as op
import optype.numpy as onp
import optype.typing as opt
from scipy._typing import AnyShape, Casting, EnterNoneMixin, OrderKACF

__all__ = [
    "agm",
    "airy",
    "airye",
    "bdtr",
    "bdtrc",
    "bdtri",
    "bdtrik",
    "bdtrin",
    "bei",
    "beip",
    "ber",
    "berp",
    "besselpoly",
    "beta",
    "betainc",
    "betaincc",
    "betainccinv",
    "betaincinv",
    "betaln",
    "binom",
    "boxcox",
    "boxcox1p",
    "btdtr",
    "btdtri",
    "btdtria",
    "btdtrib",
    "cbrt",
    "chdtr",
    "chdtrc",
    "chdtri",
    "chdtriv",
    "chndtr",
    "chndtridf",
    "chndtrinc",
    "chndtrix",
    "cosdg",
    "cosm1",
    "cotdg",
    "dawsn",
    "ellipe",
    "ellipeinc",
    "ellipj",
    "ellipk",
    "ellipkinc",
    "ellipkm1",
    "elliprc",
    "elliprd",
    "elliprf",
    "elliprg",
    "elliprj",
    "entr",
    "erf",
    "erfc",
    "erfcinv",
    "erfcx",
    "erfi",
    "erfinv",
    "errstate",
    "eval_chebyc",
    "eval_chebys",
    "eval_chebyt",
    "eval_chebyu",
    "eval_gegenbauer",
    "eval_genlaguerre",
    "eval_hermite",
    "eval_hermitenorm",
    "eval_jacobi",
    "eval_laguerre",
    "eval_legendre",
    "eval_sh_chebyt",
    "eval_sh_chebyu",
    "eval_sh_jacobi",
    "eval_sh_legendre",
    "exp1",
    "exp2",
    "exp10",
    "expi",
    "expit",
    "expm1",
    "expn",
    "exprel",
    "fdtr",
    "fdtrc",
    "fdtri",
    "fdtridfd",
    "fresnel",
    "gamma",
    "gammainc",
    "gammaincc",
    "gammainccinv",
    "gammaincinv",
    "gammaln",
    "gammasgn",
    "gdtr",
    "gdtrc",
    "gdtria",
    "gdtrib",
    "gdtrix",
    "geterr",
    "hankel1",
    "hankel1e",
    "hankel2",
    "hankel2e",
    "huber",
    "hyp0f1",
    "hyp1f1",
    "hyp2f1",
    "hyperu",
    "i0",
    "i0e",
    "i1",
    "i1e",
    "inv_boxcox",
    "inv_boxcox1p",
    "it2i0k0",
    "it2j0y0",
    "it2struve0",
    "itairy",
    "iti0k0",
    "itj0y0",
    "itmodstruve0",
    "itstruve0",
    "iv",
    "ive",
    "j0",
    "j1",
    "jn",
    "jv",
    "jve",
    "k0",
    "k0e",
    "k1",
    "k1e",
    "kei",
    "keip",
    "kelvin",
    "ker",
    "kerp",
    "kl_div",
    "kn",
    "kolmogi",
    "kolmogorov",
    "kv",
    "kve",
    "log1p",
    "log_expit",
    "log_ndtr",
    "log_wright_bessel",
    "loggamma",
    "logit",
    "lpmv",
    "mathieu_a",
    "mathieu_b",
    "mathieu_cem",
    "mathieu_modcem1",
    "mathieu_modcem2",
    "mathieu_modsem1",
    "mathieu_modsem2",
    "mathieu_sem",
    "modfresnelm",
    "modfresnelp",
    "modstruve",
    "nbdtr",
    "nbdtrc",
    "nbdtri",
    "nbdtrik",
    "nbdtrin",
    "ncfdtr",
    "ncfdtri",
    "ncfdtridfd",
    "ncfdtridfn",
    "ncfdtrinc",
    "nctdtr",
    "nctdtridf",
    "nctdtrinc",
    "nctdtrit",
    "ndtr",
    "ndtri",
    "ndtri_exp",
    "nrdtrimn",
    "nrdtrisd",
    "obl_ang1",
    "obl_ang1_cv",
    "obl_cv",
    "obl_rad1",
    "obl_rad1_cv",
    "obl_rad2",
    "obl_rad2_cv",
    "owens_t",
    "pbdv",
    "pbvv",
    "pbwa",
    "pdtr",
    "pdtrc",
    "pdtri",
    "pdtrik",
    "poch",
    "powm1",
    "pro_ang1",
    "pro_ang1_cv",
    "pro_cv",
    "pro_rad1",
    "pro_rad1_cv",
    "pro_rad2",
    "pro_rad2_cv",
    "pseudo_huber",
    "psi",
    "radian",
    "rel_entr",
    "rgamma",
    "round",
    "seterr",
    "shichi",
    "sici",
    "sindg",
    "smirnov",
    "smirnovi",
    "spence",
    "sph_harm",
    "stdtr",
    "stdtridf",
    "stdtrit",
    "struve",
    "tandg",
    "tklmbda",
    "voigt_profile",
    "wofz",
    "wright_bessel",
    "wrightomega",
    "xlog1py",
    "xlogy",
    "y0",
    "y1",
    "yn",
    "yv",
    "yve",
    "zetac",
]

_T = TypeVar("_T")
_NameT_co = TypeVar("_NameT_co", bound=LiteralString, covariant=True)
_IdentityT_co = TypeVar("_IdentityT_co", bound=L[0] | None, default=None, covariant=True)
_OutT = TypeVar("_OutT", bound=onp.ArrayND[np.number[Any]])

_Tuple2: TypeAlias = tuple[_T, _T]
_Tuple3: TypeAlias = tuple[_T, _T, _T]
_Tuple4: TypeAlias = tuple[_T, _T, _T, _T]
_Tuple5: TypeAlias = tuple[_T, _T, _T, _T, _T]

_Float: TypeAlias = np.float32 | np.float64
_LFloat: TypeAlias = _Float | np.longdouble
_Complex: TypeAlias = np.complex64 | np.complex128
_Inexact: TypeAlias = _Float | _Complex

_FloatNDT = TypeVar("_FloatNDT", bound=_Float | onp.ArrayND[_Float])
_LFloatNDT = TypeVar("_LFloatNDT", bound=_LFloat | onp.ArrayND[_LFloat])
_ComplexNDT = TypeVar("_ComplexNDT", bound=_Complex | onp.ArrayND[_Complex])
_InexactNDT = TypeVar("_InexactNDT", bound=_Inexact | onp.ArrayND[_Inexact])

_SubFloat: TypeAlias = np.float16 | np.integer[Any] | np.bool_
_ToSubFloat: TypeAlias = float | _SubFloat

_ToDType_l: TypeAlias = onp.AnyLongDType
_ToDType_f: TypeAlias = onp.AnyFloat32DType
_ToDType_d: TypeAlias = onp.AnyFloat64DType
_ToDType_g: TypeAlias = onp.AnyLongDoubleDType
_ToDType_F: TypeAlias = onp.AnyComplex64DType
_ToDType_D: TypeAlias = onp.AnyComplex128DType
_ToDType_FD: TypeAlias = onp.AnyComplex64DType | onp.AnyComplex128DType
_ToDType_fd: TypeAlias = _ToDType_f | _ToDType_d
_ToDType_fdg: TypeAlias = _ToDType_fd | _ToDType_g

_Indices: TypeAlias = op.CanIndex | slice | EllipsisType | tuple[op.CanIndex | slice | EllipsisType, ...] | onp.ToIntND

@type_check_only
class _KwBase(TypedDict, total=False):
    order: OrderKACF
    casting: Casting
    subok: onp.ToBool
    where: onp.ToBool | onp.ToBoolND

@type_check_only
class _UFunc(np.ufunc, Generic[_NameT_co, _IdentityT_co]):  # pyright: ignore[reportGeneralTypeIssues]
    @property
    @override
    def __class__(self, /) -> type[np.ufunc]: ...
    @__class__.setter
    def __class__(self, t: type[np.ufunc], /) -> None: ...
    @property
    @override
    def __name__(self, /) -> _NameT_co: ...
    @property
    @override
    def identity(self, /) -> _IdentityT_co: ...
    @property
    @override
    def signature(self, /) -> None: ...

@type_check_only
class _WithoutAt:
    def at(self, /, *args: object, **kwargs: object) -> Never: ...

@type_check_only
class _WithoutBinOps:
    # The following methods will always raise a `ValueError`
    def accumulate(self, /, *args: object, **kwargs: object) -> Never: ...
    def reduce(self, /, *args: object, **kwargs: object) -> Never: ...
    def reduceat(self, /, *args: object, **kwargs: object) -> Never: ...
    def outer(self, /, *args: object, **kwargs: object) -> Never: ...

@type_check_only
class _UFunc11(_WithoutBinOps, _UFunc[_NameT_co, _IdentityT_co], Generic[_NameT_co, _IdentityT_co]):
    @property
    @override
    def nin(self, /) -> L[1]: ...
    @property
    @override
    def nout(self, /) -> L[1]: ...
    @property
    @override
    def nargs(self, /) -> L[2]: ...

@type_check_only
class _UFunc21(_UFunc[_NameT_co, _IdentityT_co], Generic[_NameT_co, _IdentityT_co]):
    @property
    @override
    def nin(self, /) -> L[2]: ...
    @property
    @override
    def nout(self, /) -> L[1]: ...
    @property
    @override
    def nargs(self, /) -> L[3]: ...

@type_check_only
class _UFunc31(_WithoutAt, _WithoutBinOps, _UFunc[_NameT_co, _IdentityT_co], Generic[_NameT_co, _IdentityT_co]):
    @property
    @override
    def nin(self, /) -> L[3]: ...
    @property
    @override
    def nout(self, /) -> L[1]: ...
    @property
    @override
    def nargs(self, /) -> L[4]: ...

@type_check_only
class _UFunc41(_WithoutAt, _WithoutBinOps, _UFunc[_NameT_co, _IdentityT_co], Generic[_NameT_co, _IdentityT_co]):
    @property
    @override
    def nin(self, /) -> L[4]: ...
    @property
    @override
    def nout(self, /) -> L[1]: ...
    @property
    @override
    def nargs(self, /) -> L[5]: ...

_ToSignature1_fd: TypeAlias = _Tuple2[_ToDType_f] | _Tuple2[_ToDType_d]
_ToSignature1_fdg: TypeAlias = _ToSignature1_fd | _Tuple2[_ToDType_g]
_ToSignature1_FD: TypeAlias = _Tuple2[_ToDType_F] | _Tuple2[_ToDType_D]

@type_check_only
class _Kw11f(_KwBase, TypedDict, total=False):
    dtype: _ToDType_fd | None
    signature: L["f->f", "d->d"] | _ToSignature1_fd

@type_check_only
class _Kw11g(_KwBase, TypedDict, total=False):
    dtype: _ToDType_fdg | None
    signature: L["f->f", "d->d", "g->g"] | _ToSignature1_fdg

@type_check_only
class _Kw11c(_KwBase, TypedDict, total=False):
    dtype: _ToDType_FD | None
    signature: L["F->F", "D->D"] | _ToSignature1_FD

@type_check_only
class _Kw11fc(_KwBase, TypedDict, total=False):
    dtype: _ToDType_fd | _ToDType_FD | None
    signature: L["f->f", "d->d", "F->F", "D->D"] | _ToSignature1_fd | _ToSignature1_FD

_ToSignature2_ld: TypeAlias = tuple[_ToDType_l, _ToDType_d, _ToDType_d]
_ToSignature2_fd2: TypeAlias = _Tuple3[_ToDType_f] | _Tuple3[_ToDType_d]
_ToSignature2_fFdD: TypeAlias = tuple[_ToDType_f, _ToDType_F, _ToDType_F] | tuple[_ToDType_d, _ToDType_D, _ToDType_D]
_ToSignature2_FD2: TypeAlias = _Tuple3[_ToDType_F] | _Tuple3[_ToDType_D]

@type_check_only
class _Kw21ld(_KwBase, TypedDict, total=False):
    dtype: _ToDType_d | None
    signature: L["ld->d"] | _ToSignature2_ld

@type_check_only
class _Kw21f(_KwBase, TypedDict, total=False):
    dtype: _ToDType_fd | None
    signature: L["ff->f", "dd->d"] | _ToSignature2_fd2

@type_check_only
class _Kw21fc1(_KwBase, TypedDict, total=False):
    dtype: _ToDType_fd | _ToDType_FD | None
    signature: L["ff->f", "ld->d", "dd->d", "fF->F", "dD->D"] | _ToSignature2_ld | _ToSignature2_fd2 | _ToSignature2_fFdD

@type_check_only
class _Kw21fc2(_KwBase, TypedDict, total=False):
    dtype: _ToDType_fd | _ToDType_FD | None
    signature: L["ff->f", "dd->d", "FF->F", "DD->D"] | _ToSignature2_fd2 | _ToSignature2_FD2

_ToSignature3_lld: TypeAlias = tuple[_ToDType_l, _ToDType_l, _ToDType_d, _ToDType_d]
_ToSignature3_dld: TypeAlias = tuple[_ToDType_d, _ToDType_l, _ToDType_d, _ToDType_d]
_ToSignature3_fd3: TypeAlias = _Tuple4[_ToDType_f] | _Tuple4[_ToDType_d]

@type_check_only
class _Kw31f(_KwBase, TypedDict, total=False):
    dtype: _ToDType_fd | None
    signature: L["fff->f", "ddd->d", "lld->d", "dld->d"] | _ToSignature3_fd3 | _ToSignature3_lld | _ToSignature3_dld

_ToSignature3_fd4: TypeAlias = _Tuple5[_ToDType_f] | _Tuple5[_ToDType_d]

@type_check_only
class _Kw41f(_KwBase, TypedDict, total=False):
    dtype: _ToDType_fd | None
    signature: L["ffff->f", "dddd->d"] | _ToSignature3_fd4

@final
@type_check_only
class _UFunc11f(_UFunc11[_NameT_co, _IdentityT_co], Generic[_NameT_co, _IdentityT_co]):
    @property
    @override
    def ntypes(self, /) -> L[2]: ...
    @property
    @override
    def types(self, /) -> list[L["f->f", "d->d"]]: ...
    #
    @overload
    def __call__(self, x: _ToSubFloat, /, out: tuple[None] | None = None, **kw: Unpack[_Kw11f]) -> _Float: ...
    @overload
    def __call__(self, x: _FloatNDT, /, out: tuple[None] | None = None, **kw: Unpack[_Kw11f]) -> _FloatNDT: ...
    @overload
    def __call__(self, x: onp.ToFloatND, /, out: tuple[None] | None = None, **kw: Unpack[_Kw11f]) -> onp.ArrayND[_Float]: ...
    @overload
    def __call__(self, x: onp.ToFloat | onp.ToFloatND, /, out: tuple[_OutT] | _OutT, **kw: Unpack[_Kw11f]) -> _OutT: ...
    #
    @override
    def at(self, a: onp.ArrayND[_Float | _SubFloat], indices: _Indices, /) -> None: ...

@final
@type_check_only
class _UFunc11g(_UFunc11[_NameT_co, _IdentityT_co], Generic[_NameT_co, _IdentityT_co]):
    @property
    @override
    def ntypes(self, /) -> L[3]: ...
    @property
    @override
    def types(self, /) -> list[L["f->f", "d->d", "g->g"]]: ...
    #
    @overload
    def __call__(self, x: _ToSubFloat, /, out: tuple[None] | None = None, **kw: Unpack[_Kw11g]) -> _LFloat: ...
    @overload
    def __call__(self, x: _LFloatNDT, /, out: tuple[None] | None = None, **kw: Unpack[_Kw11g]) -> _LFloatNDT: ...
    @overload
    def __call__(self, x: onp.ToFloatND, /, out: tuple[None] | None = None, **kw: Unpack[_Kw11g]) -> onp.ArrayND[_Float]: ...
    @overload
    def __call__(self, x: onp.ToFloat | onp.ToFloatND, /, out: tuple[_OutT] | _OutT, **kw: Unpack[_Kw11g]) -> _OutT: ...
    #
    @override
    def at(self, a: onp.ArrayND[_LFloat | _SubFloat], indices: _Indices, /) -> None: ...

@final
@type_check_only
class _UFunc11c(_UFunc11[_NameT_co, _IdentityT_co], Generic[_NameT_co, _IdentityT_co]):
    @property
    @override
    def ntypes(self, /) -> L[2]: ...
    @property
    @override
    def types(self, /) -> list[L["F->F", "D->D"]]: ...
    #
    @overload
    def __call__(self, x: onp.ToFloat, /, out: tuple[None] | None = None, **kw: Unpack[_Kw11c]) -> _Complex: ...
    @overload
    def __call__(self, x: _ComplexNDT, /, out: tuple[None] | None = None, **kw: Unpack[_Kw11c]) -> _ComplexNDT: ...
    @overload
    def __call__(self, x: onp.ToComplexND, /, out: tuple[None] | None = None, **kw: Unpack[_Kw11c]) -> onp.ArrayND[_Complex]: ...
    @overload
    def __call__(self, x: onp.ToComplex | onp.ToComplexND, /, out: tuple[_OutT] | _OutT, **kw: Unpack[_Kw11c]) -> _OutT: ...
    #
    @override
    def at(self, a: onp.ArrayND[_Inexact | _SubFloat], indices: _Indices, /) -> None: ...

@final
@type_check_only
class _UFunc11fc(_UFunc11[_NameT_co, _IdentityT_co], Generic[_NameT_co, _IdentityT_co]):
    @property
    @override
    def ntypes(self, /) -> L[4]: ...
    @property
    @override
    def types(self, /) -> list[L["f->f", "d->d", "F->F", "D->D"]]: ...
    #
    @overload
    def __call__(self, x: _ToSubFloat, /, out: tuple[None] | None = None, **kw: Unpack[_Kw11fc]) -> _Float: ...
    @overload
    def __call__(self, x: complex | _ToSubFloat, /, out: tuple[None] | None = None, **kw: Unpack[_Kw11fc]) -> _Inexact: ...
    @overload
    def __call__(self, x: _InexactNDT, /, out: tuple[None] | None = None, **kw: Unpack[_Kw11fc]) -> _InexactNDT: ...
    @overload
    def __call__(self, x: onp.ToFloatND, /, out: tuple[None] | None = None, **kw: Unpack[_Kw11fc]) -> onp.ArrayND[_Float]: ...
    @overload
    def __call__(self, x: onp.ToComplexND, /, out: tuple[None] | None = None, **kw: Unpack[_Kw11fc]) -> onp.ArrayND[_Inexact]: ...
    @overload
    def __call__(self, x: onp.ToComplex | onp.ToComplexND, /, out: tuple[_OutT] | _OutT, **kw: Unpack[_Kw11fc]) -> _OutT: ...
    #
    @override
    def at(self, a: onp.ArrayND[_Inexact | _SubFloat], indices: _Indices, /) -> None: ...

@final
@type_check_only
class _UFunc21ld(_UFunc21[_NameT_co, _IdentityT_co], Generic[_NameT_co, _IdentityT_co]):
    @property
    @override
    def ntypes(self, /) -> L[1]: ...
    @property
    @override
    def types(self, /) -> list[L["ld->d"]]: ...
    #
    @overload
    def __call__(
        self,
        n: onp.ToInt,
        x: onp.ToFloat,
        /,
        out: tuple[None] | None = None,
        **kwargs: Unpack[_Kw21ld],
    ) -> np.float64: ...
    @overload
    def __call__(
        self,
        n: onp.ToInt | onp.ToIntND,
        x: onp.ToFloatND,
        /,
        out: tuple[None] | None = None,
        **kwargs: Unpack[_Kw21ld],
    ) -> onp.ArrayND[np.float64]: ...
    @overload
    def __call__(
        self,
        n: onp.ToIntND,
        x: onp.ToFloat | onp.ToFloatND,
        /,
        out: tuple[None] | None = None,
        **kwargs: Unpack[_Kw21ld],
    ) -> onp.ArrayND[np.float64]: ...
    @overload
    def __call__(
        self,
        n: onp.ToInt | onp.ToIntND,
        x: onp.ToFloat | onp.ToFloatND,
        /,
        out: tuple[_OutT] | _OutT,
        **kwargs: Unpack[_Kw21ld],
    ) -> _OutT: ...
    #
    @override
    def at(self, a: onp.ArrayND[np.integer[Any] | np.bool_], indices: _Indices, b: onp.ToFloatND, /) -> None: ...
    #
    @overload
    def outer(self, n: onp.ToInt, x: onp.ToFloat, /, **kwargs: Unpack[_Kw21ld]) -> np.float64: ...
    @overload
    def outer(self, n: onp.ToInt | onp.ToFloatND, x: onp.ToFloatND, /, **kwargs: Unpack[_Kw21ld]) -> onp.ArrayND[np.float64]: ...
    @overload
    def outer(self, n: onp.ToIntND, x: onp.ToFloat | onp.ToFloatND, /, **kwargs: Unpack[_Kw21ld]) -> onp.ArrayND[np.float64]: ...
    @overload
    def outer(
        self,
        n: onp.ToInt | onp.ToIntND,
        x: onp.ToFloat | onp.ToFloatND,
        /,
        *,
        out: tuple[_OutT] | _OutT,
        **kwargs: Unpack[_Kw21ld],
    ) -> _OutT: ...
    #
    @override
    def accumulate(self, /, *args: Never, **kwargs: Never) -> Never: ...
    @override
    def reduce(self, /, *args: Never, **kwargs: Never) -> Never: ...
    def reduceat(self, /, *args: Never, **kwargs: Never) -> Never: ...

@final
@type_check_only
class _UFunc21f(_UFunc21[_NameT_co, _IdentityT_co], Generic[_NameT_co, _IdentityT_co]):
    @property
    @override
    def ntypes(self, /) -> L[2, 3]: ...
    @property
    @override
    def types(self, /) -> list[L["ff->f", "dd->d", "ld->d"]]: ...
    #
    @overload
    def __call__(
        self,
        a: _ToSubFloat,
        b: _ToSubFloat,
        /,
        out: tuple[None] | None = None,
        **kwargs: Unpack[_Kw21f],
    ) -> _Float: ...
    @overload
    def __call__(
        self,
        a: _FloatNDT | _ToSubFloat,
        b: _FloatNDT | _ToSubFloat,
        /,
        out: tuple[None] | None = None,
        **kwargs: Unpack[_Kw21f],
    ) -> _FloatNDT: ...
    @overload
    def __call__(
        self,
        a: onp.ToFloat | onp.ToFloatND,
        b: onp.ToFloatND,
        /,
        out: tuple[None] | None = None,
        **kwargs: Unpack[_Kw21f],
    ) -> onp.ArrayND[_Float]: ...
    @overload
    def __call__(
        self,
        a: onp.ToFloatND,
        b: onp.ToFloat | onp.ToFloatND,
        /,
        out: tuple[None] | None = None,
        **kwargs: Unpack[_Kw21f],
    ) -> onp.ArrayND[_Float]: ...
    @overload
    def __call__(
        self,
        a: onp.ToFloat | onp.ToFloatND,
        b: onp.ToFloat | onp.ToFloatND,
        /,
        out: tuple[_OutT] | _OutT,
        **kwargs: Unpack[_Kw21f],
    ) -> _OutT: ...
    #
    @override
    def at(self, a: onp.ArrayND[_Float | _SubFloat], indices: _Indices, b: onp.ToFloatND, /) -> None: ...
    #
    @override
    def accumulate(
        self,
        /,
        array: onp.ToFloatND,
        axis: op.CanIndex = 0,
        dtype: _ToDType_fd | None = None,
        out: tuple[onp.ArrayND[_Float] | None] | onp.ArrayND[_Float] | None = None,
    ) -> onp.ArrayND[_Float]: ...
    #
    @overload
    def reduce(
        self,
        /,
        array: onp.ToFloatND,
        axis: None,
        dtype: onp.AnyFloat32DType | onp.AnyFloat64DType | None = None,
        out: tuple[None] | None = None,
        keepdims: L[False] = False,
        initial: onp.ToFloat = ...,
        where: onp.ToBool | onp.ToBoolND = True,
    ) -> np.float32 | np.float64: ...
    @overload
    def reduce(
        self,
        /,
        array: onp.ToFloatND,
        axis: AnyShape | None = 0,
        dtype: onp.AnyFloat32DType | onp.AnyFloat64DType | None = None,
        out: tuple[None] | None = None,
        keepdims: L[False] = False,
        initial: onp.ToFloat = ...,
        where: onp.ToBool | onp.ToBoolND = True,
    ) -> np.float32 | np.float64 | onp.ArrayND[np.float32 | np.float64]: ...
    @overload
    def reduce(
        self,
        /,
        array: onp.ToFloatND,
        axis: AnyShape | None = 0,
        dtype: onp.AnyFloat32DType | onp.AnyFloat64DType | None = None,
        out: tuple[None] | None = None,
        *,
        keepdims: L[True],
        initial: onp.ToFloat = ...,
        where: onp.ToBool | onp.ToBoolND = True,
    ) -> onp.ArrayND[np.float32 | np.float64]: ...
    @overload
    def reduce(
        self,
        /,
        array: onp.ToFloatND,
        axis: AnyShape | None,
        dtype: onp.AnyFloat32DType | onp.AnyFloat64DType | None,
        out: tuple[_OutT] | _OutT,
        keepdims: bool = False,
        initial: onp.ToFloat = ...,
        where: onp.ToBool | onp.ToBoolND = True,
    ) -> _OutT: ...
    @overload
    def reduce(
        self,
        /,
        array: onp.ToFloatND,
        axis: AnyShape | None = 0,
        dtype: onp.AnyFloat32DType | onp.AnyFloat64DType | None = None,
        *,
        out: tuple[_OutT] | _OutT,
        keepdims: bool = False,
        initial: onp.ToFloat = ...,
        where: onp.ToBool | onp.ToBoolND = True,
    ) -> _OutT: ...
    #
    def reduceat(
        self,
        /,
        array: onp.ToFloatND,
        indices: _Indices,
        axis: op.CanIndex = 0,
        dtype: _ToDType_fd | None = None,
        out: tuple[onp.ArrayND[_Float] | None] | onp.ArrayND[_Float] | None = None,
    ) -> onp.ArrayND[_Float]: ...
    #
    @overload
    def outer(self, a: onp.ToFloat, b: onp.ToFloat, /, **kwargs: Unpack[_Kw21f]) -> _Float: ...
    @overload
    def outer(self, a: onp.ToFloat | onp.ToFloatND, b: onp.ToFloatND, /, **kwargs: Unpack[_Kw21f]) -> onp.ArrayND[_Float]: ...
    @overload
    def outer(self, a: onp.ToFloatND, b: onp.ToFloat | onp.ToFloatND, /, **kwargs: Unpack[_Kw21f]) -> onp.ArrayND[_Float]: ...
    @overload
    def outer(
        self,
        a: onp.ToFloat | onp.ToFloatND,
        b: onp.ToFloat | onp.ToFloatND,
        /,
        *,
        out: tuple[_OutT] | _OutT,
        **kwargs: Unpack[_Kw21f],
    ) -> _OutT: ...

@final
@type_check_only
class _UFunc21fc1(_UFunc21[_NameT_co, _IdentityT_co], Generic[_NameT_co, _IdentityT_co]):
    @property
    @override
    def ntypes(self, /) -> L[4, 5]: ...
    @property
    @override
    def types(self, /) -> list[L["ff->f", "dd->d", "fF->F", "dD->D", "ld->d"]]: ...
    #
    @overload
    def __call__(
        self,
        a: onp.ToFloat,
        b: _ToSubFloat,
        /,
        out: tuple[None] | None = None,
        **kwargs: Unpack[_Kw21fc1],
    ) -> _Float: ...
    @overload
    def __call__(
        self,
        a: onp.ToFloat,
        b: _InexactNDT,
        /,
        out: tuple[None] | None = None,
        **kwargs: Unpack[_Kw21fc1],
    ) -> _InexactNDT: ...
    @overload
    def __call__(
        self,
        a: onp.ToFloat | onp.ToFloatND,
        b: onp.ToFloatND,
        /,
        out: tuple[None] | None = None,
        **kwargs: Unpack[_Kw21fc1],
    ) -> onp.ArrayND[_Float]: ...
    @overload
    def __call__(
        self,
        a: onp.ToFloatND,
        b: onp.ToFloat | onp.ToFloatND,
        /,
        out: tuple[None] | None = None,
        **kwargs: Unpack[_Kw21fc1],
    ) -> onp.ArrayND[_Float]: ...
    @overload
    def __call__(
        self,
        a: onp.ToFloat | onp.ToFloatND,
        b: onp.ToComplexND,
        /,
        out: tuple[None] | None = None,
        **kwargs: Unpack[_Kw21fc1],
    ) -> onp.ArrayND[_Inexact]: ...
    @overload
    def __call__(
        self,
        a: onp.ToFloatND,
        b: onp.ToComplex | onp.ToComplexND,
        /,
        out: tuple[None] | None = None,
        **kwargs: Unpack[_Kw21fc1],
    ) -> onp.ArrayND[_Inexact]: ...
    @overload
    def __call__(
        self,
        a: onp.ToFloat | onp.ToFloatND,
        b: onp.ToComplex | onp.ToComplexND,
        /,
        out: tuple[_OutT] | _OutT,
        **kwargs: Unpack[_Kw21fc1],
    ) -> _OutT: ...
    #
    @override  # only works if real
    def at(self, a: onp.ArrayND[_Float | _SubFloat], indices: _Indices, b: onp.ToFloatND, /) -> None: ...
    #
    @override
    def accumulate(
        self,
        /,
        array: onp.ToFloatND,
        axis: op.CanIndex = 0,
        dtype: _ToDType_fd | None = None,
        out: tuple[onp.ArrayND[_Float] | None] | onp.ArrayND[_Float] | None = None,
    ) -> onp.ArrayND[_Float]: ...
    #
    @overload
    def reduce(
        self,
        /,
        array: onp.ToFloatND,
        axis: None,
        dtype: onp.AnyFloat32DType | onp.AnyFloat64DType | None = None,
        out: tuple[None] | None = None,
        keepdims: L[False] = False,
        initial: onp.ToFloat = ...,
        where: onp.ToBool | onp.ToBoolND = True,
    ) -> _Float: ...
    @overload
    def reduce(
        self,
        /,
        array: onp.ToFloatND,
        axis: AnyShape | None = 0,
        dtype: onp.AnyFloat32DType | onp.AnyFloat64DType | None = None,
        out: tuple[None] | None = None,
        keepdims: L[False] = False,
        initial: onp.ToFloat = ...,
        where: onp.ToBool | onp.ToBoolND = True,
    ) -> _Float | onp.ArrayND[_Float]: ...
    @overload
    def reduce(
        self,
        /,
        array: onp.ToFloatND,
        axis: AnyShape | None = 0,
        dtype: onp.AnyFloat32DType | onp.AnyFloat64DType | None = None,
        out: tuple[None] | None = None,
        *,
        keepdims: L[True],
        initial: onp.ToFloat = ...,
        where: onp.ToBool | onp.ToBoolND = True,
    ) -> onp.ArrayND[_Float]: ...
    @overload
    def reduce(
        self,
        /,
        array: onp.ToFloatND,
        axis: AnyShape | None,
        dtype: onp.AnyFloat32DType | onp.AnyFloat64DType | None,
        out: tuple[_OutT] | _OutT,
        keepdims: bool = False,
        initial: onp.ToFloat = ...,
        where: onp.ToBool | onp.ToBoolND = True,
    ) -> _OutT: ...
    @overload
    def reduce(
        self,
        /,
        array: onp.ToFloatND,
        axis: AnyShape | None = 0,
        dtype: onp.AnyFloat32DType | onp.AnyFloat64DType | None = None,
        *,
        out: tuple[_OutT] | _OutT,
        keepdims: bool = False,
        initial: onp.ToFloat = ...,
        where: onp.ToBool | onp.ToBoolND = True,
    ) -> _OutT: ...
    #
    def reduceat(
        self,
        /,
        array: onp.ToFloatND,
        indices: _Indices,
        axis: op.CanIndex = 0,
        dtype: _ToDType_fd | None = None,
        out: tuple[onp.ArrayND[_Float] | None] | onp.ArrayND[_Float] | None = None,
    ) -> onp.ArrayND[_Float]: ...
    #
    @overload
    def outer(self, a: onp.ToFloat, b: onp.ToFloat, /, **kw: Unpack[_Kw21fc1]) -> _Float: ...
    @overload
    def outer(self, a: onp.ToFloat, b: onp.ToComplex, /, **kw: Unpack[_Kw21fc1]) -> _Inexact: ...
    @overload
    def outer(self, a: onp.ToFloat | onp.ToFloatND, b: onp.ToFloatND, /, **kw: Unpack[_Kw21fc1]) -> onp.ArrayND[_Float]: ...
    @overload
    def outer(self, a: onp.ToFloatND, b: onp.ToFloat | onp.ToFloatND, /, **kw: Unpack[_Kw21fc1]) -> onp.ArrayND[_Float]: ...
    @overload
    def outer(self, a: onp.ToFloat | onp.ToFloatND, b: onp.ToComplexND, /, **kw: Unpack[_Kw21fc1]) -> onp.ArrayND[_Inexact]: ...
    @overload
    def outer(self, a: onp.ToFloatND, b: onp.ToComplex | onp.ToComplexND, /, **kw: Unpack[_Kw21fc1]) -> onp.ArrayND[_Inexact]: ...
    @overload
    def outer(
        self,
        a: onp.ToFloat | onp.ToFloatND,
        b: onp.ToComplex | onp.ToComplexND,
        /,
        *,
        out: tuple[_OutT] | _OutT,
        **kwargs: Unpack[_Kw21fc1],
    ) -> _OutT: ...

@final
@type_check_only
class _UFunc21fc2(_UFunc21[_NameT_co, _IdentityT_co], Generic[_NameT_co, _IdentityT_co]):
    @property
    @override
    def ntypes(self, /) -> L[4]: ...
    @property
    @override
    def types(self, /) -> list[L["ff->f", "dd->d", "FF->F", "DD->D"]]: ...
    #
    @overload
    def __call__(
        self,
        x: _ToSubFloat,
        y: _ToSubFloat,
        /,
        out: tuple[None] | None = None,
        **kwargs: Unpack[_Kw21fc2],
    ) -> _Float: ...
    @overload
    def __call__(
        self,
        x: _ToSubFloat,
        y: opt.Just[complex],
        /,
        out: tuple[None] | None = None,
        **kwargs: Unpack[_Kw21fc2],
    ) -> _Complex: ...
    @overload
    def __call__(
        self,
        x: opt.Just[complex],
        y: opt.Just[complex] | _ToSubFloat,
        /,
        out: tuple[None] | None = None,
        **kwargs: Unpack[_Kw21fc2],
    ) -> _Complex: ...
    @overload
    def __call__(
        self,
        x: _InexactNDT | _ToSubFloat,
        y: _InexactNDT | _ToSubFloat,
        /,
        out: tuple[None] | None = None,
        **kwargs: Unpack[_Kw21fc2],
    ) -> _InexactNDT: ...
    @overload
    def __call__(
        self,
        x: onp.ToFloat | onp.ToFloatND,
        y: onp.ToFloatND,
        /,
        out: tuple[None] | None = None,
        **kwargs: Unpack[_Kw21fc2],
    ) -> onp.ArrayND[_Float]: ...
    @overload
    def __call__(
        self,
        x: onp.ToFloatND,
        y: onp.ToFloat | onp.ToFloatND,
        /,
        out: tuple[None] | None = None,
        **kwargs: Unpack[_Kw21fc2],
    ) -> onp.ArrayND[_Float]: ...
    @overload
    def __call__(
        self,
        x: onp.ToComplex | onp.ToComplexND,
        y: onp.ToComplexND,
        /,
        out: tuple[None] | None = None,
        **kwargs: Unpack[_Kw21fc2],
    ) -> onp.ArrayND[_Inexact]: ...
    @overload
    def __call__(
        self,
        x: onp.ToComplexND,
        y: onp.ToComplex | onp.ToComplexND,
        /,
        out: tuple[None] | None = None,
        **kwargs: Unpack[_Kw21fc2],
    ) -> onp.ArrayND[_Inexact]: ...
    @overload
    def __call__(
        self,
        x: onp.ToComplex | onp.ToComplexND,
        y: onp.ToComplex | onp.ToComplexND,
        /,
        out: tuple[_OutT] | _OutT,
        **kwargs: Unpack[_Kw21fc2],
    ) -> _OutT: ...
    #
    @overload
    def at(self, x: onp.ArrayND[_Float | _SubFloat], indices: _Indices, y: onp.ToFloatND, /) -> None: ...
    @overload
    def at(self, x: onp.ArrayND[np.complexfloating[Any, Any]], indices: _Indices, y: onp.ToComplexND, /) -> None: ...
    #
    @overload
    def accumulate(
        self,
        /,
        array: onp.ToFloatND,
        axis: op.CanIndex = 0,
        dtype: _ToDType_fd | None = None,
        out: tuple[onp.ArrayND[_Float] | None] | onp.ArrayND[_Float] | None = None,
    ) -> onp.ArrayND[_Float]: ...
    @overload
    def accumulate(
        self,
        /,
        array: onp.ToComplexND,
        axis: op.CanIndex = 0,
        dtype: _ToDType_fd | _ToDType_FD | None = None,
        out: tuple[onp.ArrayND[_Inexact] | None] | onp.ArrayND[_Inexact] | None = None,
    ) -> onp.ArrayND[_Inexact]: ...
    #
    @overload
    def reduce(
        self,
        /,
        array: onp.ToFloatND,
        axis: None,
        dtype: _ToDType_fd | None = None,
        out: tuple[None] | None = None,
        keepdims: L[False] = False,
        initial: onp.ToFloat = ...,
        where: onp.ToBool | onp.ToBoolND = True,
    ) -> _Float: ...
    @overload
    def reduce(
        self,
        /,
        array: onp.ToFloatND,
        axis: AnyShape | None = 0,
        dtype: _ToDType_fd | None = None,
        out: tuple[None] | None = None,
        *,
        keepdims: L[True],
        initial: onp.ToFloat = ...,
        where: onp.ToBool | onp.ToBoolND = True,
    ) -> onp.ArrayND[_Float]: ...
    @overload
    def reduce(
        self,
        /,
        array: onp.ToComplexND,
        axis: None,
        dtype: _ToDType_fd | _ToDType_FD | None = None,
        out: tuple[None] | None = None,
        keepdims: L[False] = False,
        initial: onp.ToComplex = ...,
        where: onp.ToBool | onp.ToBoolND = True,
    ) -> _Inexact: ...
    @overload
    def reduce(
        self,
        /,
        array: onp.ToComplexND,
        axis: AnyShape | None = 0,
        dtype: _ToDType_fd | _ToDType_FD | None = None,
        out: tuple[None] | None = None,
        *,
        keepdims: L[True],
        initial: onp.ToComplex = ...,
        where: onp.ToBool | onp.ToBoolND = True,
    ) -> onp.ArrayND[_Inexact]: ...
    @overload
    def reduce(
        self,
        /,
        array: onp.ToComplexND,
        axis: AnyShape | None = 0,
        dtype: _ToDType_fd | _ToDType_FD | None = None,
        out: tuple[None] | None = None,
        keepdims: L[False] = False,
        initial: onp.ToComplex = ...,
        where: onp.ToBool | onp.ToBoolND = True,
    ) -> _Inexact | onp.ArrayND[_Inexact]: ...
    @overload
    def reduce(
        self,
        /,
        array: onp.ToComplexND,
        axis: AnyShape | None,
        dtype: _ToDType_fd | _ToDType_FD | None,
        out: tuple[_OutT] | _OutT,
        keepdims: bool = False,
        initial: onp.ToComplex = ...,
        where: onp.ToBool | onp.ToBoolND = True,
    ) -> _OutT: ...
    @overload
    def reduce(
        self,
        /,
        array: onp.ToComplexND,
        axis: AnyShape | None = 0,
        dtype: _ToDType_fd | _ToDType_FD | None = None,
        *,
        out: tuple[_OutT] | _OutT,
        keepdims: bool = False,
        initial: onp.ToComplex = ...,
        where: onp.ToBool | onp.ToBoolND = True,
    ) -> _OutT: ...
    #
    @overload
    def reduceat(
        self,
        /,
        array: onp.ToFloatND,
        indices: _Indices,
        axis: op.CanIndex = 0,
        dtype: _ToDType_fd | None = None,
        out: tuple[onp.ArrayND[_Float] | None] | onp.ArrayND[_Float] | None = None,
    ) -> onp.ArrayND[_Float]: ...
    @overload
    def reduceat(
        self,
        /,
        array: onp.ToComplexND,
        indices: _Indices,
        axis: op.CanIndex = 0,
        dtype: _ToDType_fd | _ToDType_FD | None = None,
        out: tuple[onp.ArrayND[_Inexact] | None] | onp.ArrayND[_Inexact] | None = None,
    ) -> onp.ArrayND[_Inexact]: ...
    #
    @overload
    def outer(self, x: onp.ToFloat, y: onp.ToFloat, /, **kw: Unpack[_Kw21fc2]) -> _Float: ...
    @overload
    def outer(self, x: onp.ToComplex, y: onp.ToComplex, /, **kw: Unpack[_Kw21fc2]) -> _Inexact: ...
    @overload
    def outer(self, x: onp.ToFloat | onp.ToFloatND, y: onp.ToFloatND, /, **kw: Unpack[_Kw21fc2]) -> onp.ArrayND[_Float]: ...
    @overload
    def outer(self, x: onp.ToFloatND, y: onp.ToFloat | onp.ToFloatND, /, **kw: Unpack[_Kw21fc2]) -> onp.ArrayND[_Float]: ...
    @overload
    def outer(
        self,
        x: onp.ToComplex | onp.ToComplexND,
        y: onp.ToComplexND,
        /,
        **kw: Unpack[_Kw21fc2],
    ) -> onp.ArrayND[_Inexact]: ...
    @overload
    def outer(
        self,
        x: onp.ToComplexND,
        y: onp.ToComplex | onp.ToComplexND,
        /,
        **kw: Unpack[_Kw21fc2],
    ) -> onp.ArrayND[_Inexact]: ...
    @overload
    def outer(
        self,
        x: onp.ToComplex | onp.ToComplexND,
        y: onp.ToComplex | onp.ToComplexND,
        /,
        *,
        out: tuple[_OutT] | _OutT,
        **kwargs: Unpack[_Kw21fc2],
    ) -> _OutT: ...

@final
@type_check_only
class _UFunc31f(_UFunc31[_NameT_co, _IdentityT_co], Generic[_NameT_co, _IdentityT_co]):
    @property
    @override
    def ntypes(self, /) -> L[2, 3]: ...
    @property
    @override
    def types(self, /) -> list[L["fff->f", "ddd->d", "lld->d"]] | list[L["fff->f", "ddd->d", "dld->d"]]: ...
    #
    @overload
    def __call__(
        self,
        a: _ToSubFloat,
        b: _ToSubFloat,
        x: _ToSubFloat,
        /,
        out: tuple[None] | None = None,
        **kwargs: Unpack[_Kw31f],
    ) -> _Float: ...
    @overload
    def __call__(
        self,
        a: _FloatNDT | _ToSubFloat,
        b: _FloatNDT | _ToSubFloat,
        x: _FloatNDT | _ToSubFloat,
        /,
        out: tuple[None] | None = None,
        **kwargs: Unpack[_Kw31f],
    ) -> _FloatNDT: ...
    @overload
    def __call__(
        self,
        a: onp.ToFloat | onp.ToFloatND,
        b: onp.ToFloat | onp.ToFloatND,
        x: onp.ToFloatND,
        /,
        out: tuple[None] | None = None,
        **kwargs: Unpack[_Kw31f],
    ) -> onp.ArrayND[_Float]: ...
    @overload
    def __call__(
        self,
        a: onp.ToFloat | onp.ToFloatND,
        b: onp.ToFloatND,
        x: onp.ToFloat | onp.ToFloatND,
        /,
        out: tuple[None] | None = None,
        **kwargs: Unpack[_Kw31f],
    ) -> onp.ArrayND[_Float]: ...
    @overload
    def __call__(
        self,
        a: onp.ToFloatND,
        b: onp.ToFloat | onp.ToFloatND,
        x: onp.ToFloat | onp.ToFloatND,
        /,
        out: tuple[None] | None = None,
        **kwargs: Unpack[_Kw31f],
    ) -> onp.ArrayND[_Float]: ...
    @overload
    def __call__(
        self,
        a: onp.ToFloat | onp.ToFloatND,
        b: onp.ToFloat | onp.ToFloatND,
        x: onp.ToFloat | onp.ToFloatND,
        /,
        out: tuple[_OutT] | _OutT,
        **kwargs: Unpack[_Kw31f],
    ) -> _OutT: ...

@final
@type_check_only
class _UFunc41f(_UFunc41[_NameT_co, _IdentityT_co], Generic[_NameT_co, _IdentityT_co]):
    @property
    @override
    def ntypes(self, /) -> L[2]: ...
    @property
    @override
    def types(self, /) -> list[L["ffff->f", "dddd->d"]]: ...
    #
    @overload
    def __call__(
        self,
        dfn: _ToSubFloat,
        dfd: _ToSubFloat,
        nc: _ToSubFloat,
        f: _ToSubFloat,
        /,
        out: tuple[None] | None = None,
        **kwargs: Unpack[_Kw41f],
    ) -> _Float: ...
    @overload
    def __call__(
        self,
        dfn: _FloatNDT | _ToSubFloat,
        dfd: _FloatNDT | _ToSubFloat,
        nc: _FloatNDT | _ToSubFloat,
        f: _FloatNDT | _ToSubFloat,
        /,
        out: tuple[None] | None = None,
        **kwargs: Unpack[_Kw41f],
    ) -> _FloatNDT: ...
    @overload
    def __call__(
        self,
        dfn: onp.ToFloat | onp.ToFloatND,
        dfd: onp.ToFloat | onp.ToFloatND,
        nc: onp.ToFloat | onp.ToFloatND,
        f: onp.ToFloatND,
        /,
        out: tuple[None] | None = None,
        **kwargs: Unpack[_Kw41f],
    ) -> onp.ArrayND[_Float]: ...
    @overload
    def __call__(
        self,
        dfn: onp.ToFloat | onp.ToFloatND,
        dfd: onp.ToFloat | onp.ToFloatND,
        nc: onp.ToFloatND,
        f: onp.ToFloat | onp.ToFloatND,
        /,
        out: tuple[None] | None = None,
        **kwargs: Unpack[_Kw41f],
    ) -> onp.ArrayND[_Float]: ...
    @overload
    def __call__(
        self,
        dfn: onp.ToFloat | onp.ToFloatND,
        dfd: onp.ToFloatND,
        nc: onp.ToFloat | onp.ToFloatND,
        f: onp.ToFloat | onp.ToFloatND,
        /,
        out: tuple[None] | None = None,
        **kwargs: Unpack[_Kw41f],
    ) -> onp.ArrayND[_Float]: ...
    @overload
    def __call__(
        self,
        dfn: onp.ToFloatND,
        dfd: onp.ToFloat | onp.ToFloatND,
        nc: onp.ToFloat | onp.ToFloatND,
        f: onp.ToFloat | onp.ToFloatND,
        /,
        out: tuple[None] | None = None,
        **kwargs: Unpack[_Kw41f],
    ) -> onp.ArrayND[_Float]: ...
    @overload
    def __call__(
        self,
        dfn: onp.ToFloat | onp.ToFloatND,
        dfd: onp.ToFloat | onp.ToFloatND,
        nc: onp.ToFloat | onp.ToFloatND,
        f: onp.ToFloat | onp.ToFloatND,
        /,
        out: tuple[_OutT] | _OutT,
        **kwargs: Unpack[_Kw41f],
    ) -> _OutT: ...

###

def geterr() -> dict[str, str]: ...
def seterr(**kwargs: str) -> dict[str, str]: ...

class errstate(EnterNoneMixin):
    def __init__(self, /, **kwargs: str) -> None: ...

# l->l
_sf_error_test_function: np.ufunc

# f->f; d->d
_cosine_cdf: _UFunc11f[L["_cosine_cdf"], L[0]]
_cosine_invcdf: _UFunc11f[L["_cosine_invcdf"], L[0]]
_factorial: _UFunc11f[L["_factorial"], L[0]]
_kolmogc: _UFunc11f[L["_kolmogc"], L[0]]
_kolmogci: _UFunc11f[L["_kolmogci"], L[0]]
_kolmogp: _UFunc11f[L["_kolmogp"], L[0]]
_lanczos_sum_expg_scaled: _UFunc11f[L["_lanczos_sum_expg_scaled"], L[0]]
_lgam1p: _UFunc11f[L["_lgam1p"], L[0]]
_log1pmx: _UFunc11f[L["_log1pmx"], L[0]]
_riemann_zeta: _UFunc11f[L["_riemann_zeta"], L[0]]
_scaled_exp1: _UFunc11f[L["_scaled_exp1"]]
bei: _UFunc11f[L["bei"]]
beip: _UFunc11f[L["beip"]]
ber: _UFunc11f[L["ber"]]
berp: _UFunc11f[L["berp"]]
cbrt: _UFunc11f[L["cbrt"], L[0]]
cosdg: _UFunc11f[L["cosdg"], L[0]]
cosm1: _UFunc11f[L["cosm1"], L[0]]
cotdg: _UFunc11f[L["cotdg"], L[0]]
ellipe: _UFunc11f[L["ellipe"], L[0]]
ellipk: _UFunc11f[L["ellipk"], L[0]]
ellipkm1: _UFunc11f[L["ellipkm1"], L[0]]
entr: _UFunc11f[L["entr"], L[0]]
erfcinv: _UFunc11f[L["erfcinv"], L[0]]
erfinv: _UFunc11f[L["erfinv"], L[0]]
exp10: _UFunc11f[L["exp10"], L[0]]
exp2: _UFunc11f[L["exp2"], L[0]]
exprel: _UFunc11f[L["exprel"]]
gammaln: _UFunc11f[L["gammaln"]]
gammasgn: _UFunc11f[L["gammasgn"], L[0]]
i0: _UFunc11f[L["i0"], L[0]]
i0e: _UFunc11f[L["i0e"], L[0]]
i1: _UFunc11f[L["i1"], L[0]]
i1e: _UFunc11f[L["i1e"], L[0]]
it2struve0: _UFunc11f[L["it2struve0"]]
itmodstruve0: _UFunc11f[L["itmodstruve0"]]
itstruve0: _UFunc11f[L["itstruve0"]]
j0: _UFunc11f[L["j0"], L[0]]
j1: _UFunc11f[L["j1"], L[0]]
k0: _UFunc11f[L["k0"], L[0]]
k0e: _UFunc11f[L["k0e"], L[0]]
k1: _UFunc11f[L["k1"], L[0]]
k1e: _UFunc11f[L["k1e"], L[0]]
kei: _UFunc11f[L["kei"]]
keip: _UFunc11f[L["keip"]]
ker: _UFunc11f[L["ker"]]
kerp: _UFunc11f[L["kerp"]]
kolmogi: _UFunc11f[L["kolmogi"], L[0]]
kolmogorov: _UFunc11f[L["kolmogorov"], L[0]]
ndtri: _UFunc11f[L["ndtri"], L[0]]
ndtri_exp: _UFunc11f[L["ndtri_exp"], L[0]]
round: _UFunc11f[L["round"], L[0]]
sindg: _UFunc11f[L["sindg"], L[0]]
tandg: _UFunc11f[L["tandg"], L[0]]
y0: _UFunc11f[L["y0"], L[0]]
y1: _UFunc11f[L["y1"], L[0]]
zetac: _UFunc11f[L["zetac"], L[0]]

# f->f; d->d; g->g
expit: _UFunc11g[L["expit"]]
log_expit: _UFunc11g[L["log_expit"]]
logit: _UFunc11g[L["logit"]]

# F->F; D->D
wofz: _UFunc11c[L["wofz"], L[0]]

# f->f; d->d; F->F; D->D
_cospi: _UFunc11fc[L["_cospi"]]
_sinpi: _UFunc11fc[L["_sinpi"]]
dawsn: _UFunc11fc[L["dawsn"], L[0]]
erf: _UFunc11fc[L["erf"], L[0]]
erfc: _UFunc11fc[L["erfc"], L[0]]
erfcx: _UFunc11fc[L["erfcx"], L[0]]
erfi: _UFunc11fc[L["erfi"], L[0]]
exp1: _UFunc11fc[L["exp1"]]
expi: _UFunc11fc[L["expi"]]
expm1: _UFunc11fc[L["expm1"], L[0]]
gamma: _UFunc11fc[L["gamma"]]
log1p: _UFunc11fc[L["log1p"], L[0]]
log_ndtr: _UFunc11fc[L["log_ndtr"], L[0]]
loggamma: _UFunc11fc[L["loggamma"]]
ndtr: _UFunc11fc[L["ndtr"], L[0]]
psi: _UFunc11fc[L["psi"]]
rgamma: _UFunc11fc[L["rgamma"]]
spence: _UFunc11fc[L["spence"], L[0]]
wrightomega: _UFunc11fc[L["wrightomega"], L[0]]

# f->ff; d->dd
# TODO
it2i0k0: np.ufunc
it2j0y0: np.ufunc
iti0k0: np.ufunc
itj0y0: np.ufunc

# f->FF; d->DD
# TODO
modfresnelm: np.ufunc
modfresnelp: np.ufunc

# f->ff; d->dd; f->FF; D->DD
# TODO
fresnel: np.ufunc
shichi: np.ufunc
sici: np.ufunc

# f->ffff; d->dddd
# TODO
itairy: np.ufunc

# f->FFFF; d->DDDD
# TODO
kelvin: np.ufunc

# f->ffff; d->dddd; F->FFFF; D->DDDD
# TODO
airy: np.ufunc
airye: np.ufunc

# ld->d
eval_hermite: _UFunc21ld[L["eval_hermite"], L[0]]
eval_hermitenorm: _UFunc21ld[L["eval_hermitenorm"], L[0]]

# ff->f; (l|d)d->d
_igam_fac: _UFunc21f[L["_igam_fac"], L[0]]
_iv_ratio: _UFunc21f[L["_iv_ratio"], L[0]]
_nbinom_mean: _UFunc21f[L["_nbinom_mean"], L[0]]
_nbinom_variance: _UFunc21f[L["_nbinom_variance"], L[0]]
_nbinom_skewness: _UFunc21f[L["_nbinom_skewness"], L[0]]
_nbinom_kurtosis_excess: _UFunc21f[L["_nbinom_kurtosis_excess"], L[0]]
_nct_mean: _UFunc21f[L["_nct_mean"], L[0]]
_nct_variance: _UFunc21f[L["_nct_variance"], L[0]]
_nct_skewness: _UFunc21f[L["_nct_skewness"], L[0]]
_nct_kurtosis_excess: _UFunc21f[L["_nct_kurtosis_excess"], L[0]]
_smirnovc: _UFunc21f[L["_smirnovc"], L[0]]
_smirnovci: _UFunc21f[L["_smirnovci"], L[0]]
_smirnovp: _UFunc21f[L["_smirnovp"], L[0]]
_stirling2_inexact: _UFunc21f[L["_stirling2_inexact"]]
_zeta: _UFunc21f[L["_zeta"], L[0]]
agm: _UFunc21f[L["agm"], L[0]]
beta: _UFunc21f[L["beta"], L[0]]
betaln: _UFunc21f[L["betaln"], L[0]]
binom: _UFunc21f[L["binom"]]
boxcox: _UFunc21f[L["boxcox"], L[0]]
boxcox1p: _UFunc21f[L["boxcox1p"], L[0]]
chdtr: _UFunc21f[L["chdtr"], L[0]]
chdtrc: _UFunc21f[L["chdtrc"], L[0]]
chdtri: _UFunc21f[L["chdtri"], L[0]]
chdtriv: _UFunc21f[L["chdtriv"], L[0]]
ellipeinc: _UFunc21f[L["ellipeinc"], L[0]]
ellipkinc: _UFunc21f[L["ellipkinc"], L[0]]
expn: _UFunc21f[L["expn"], L[0]]
gammainc: _UFunc21f[L["gammainc"], L[0]]
gammaincc: _UFunc21f[L["gammaincc"], L[0]]
gammainccinv: _UFunc21f[L["gammainccinv"], L[0]]
gammaincinv: _UFunc21f[L["gammaincinv"], L[0]]
huber: _UFunc21f[L["huber"], L[0]]
inv_boxcox1p: _UFunc21f[L["inv_boxcox1p"], L[0]]
inv_boxcox: _UFunc21f[L["inv_boxcox"], L[0]]
kl_div: _UFunc21f[L["kl_div"], L[0]]
kn: _UFunc21f[L["kn"], L[0]]
mathieu_a: _UFunc21f[L["mathieu_a"]]
mathieu_b: _UFunc21f[L["mathieu_b"]]
modstruve: _UFunc21f[L["modstruve"], L[0]]
owens_t: _UFunc21f[L["owens_t"], L[0]]
pdtr: _UFunc21f[L["pdtr"], L[0]]
pdtrc: _UFunc21f[L["pdtrc"], L[0]]
pdtri: _UFunc21f[L["pdtri"], L[0]]
pdtrik: _UFunc21f[L["pdtrik"], L[0]]
poch: _UFunc21f[L["poch"], L[0]]
powm1: _UFunc21f[L["powm1"], L[0]]
pseudo_huber: _UFunc21f[L["pseudo_huber"], L[0]]
rel_entr: _UFunc21f[L["rel_entr"], L[0]]
smirnov: _UFunc21f[L["smirnov"], L[0]]
smirnovi: _UFunc21f[L["smirnovi"], L[0]]
stdtr: _UFunc21f[L["stdtr"], L[0]]
stdtridf: _UFunc21f[L["stdtridf"], L[0]]
stdtrit: _UFunc21f[L["stdtrit"], L[0]]
struve: _UFunc21f[L["struve"], L[0]]
tklmbda: _UFunc21f[L["tklmbda"], L[0]]
yn: _UFunc21f[L["yn"], L[0]]

# fF->F; dD->D
# TODO
hankel1: np.ufunc
hankel1e: np.ufunc
hankel2: np.ufunc
hankel2e: np.ufunc

# ff->f; (l|d)d->d; fF->F; dD->D
eval_chebyc: _UFunc21fc1[L["eval_chebyc"], L[0]]
eval_chebys: _UFunc21fc1[L["eval_chebys"], L[0]]
eval_chebyt: _UFunc21fc1[L["eval_chebyt"], L[0]]
eval_chebyu: _UFunc21fc1[L["eval_chebyu"], L[0]]
eval_laguerre: _UFunc21fc1[L["eval_laguerre"], L[0]]
eval_legendre: _UFunc21fc1[L["eval_legendre"], L[0]]
eval_sh_chebyt: _UFunc21fc1[L["eval_sh_chebyt"], L[0]]
eval_sh_chebyu: _UFunc21fc1[L["eval_sh_chebyu"], L[0]]
eval_sh_legendre: _UFunc21fc1[L["eval_sh_legendre"], L[0]]
hyp0f1: _UFunc21fc1[L["hyp0f1"], L[0]]
iv: _UFunc21fc1[L["iv"]]
ive: _UFunc21fc1[L["ive"]]
jn: _UFunc21fc1[L["jn"]]
jv: _UFunc21fc1[L["jv"]]
jve: _UFunc21fc1[L["jve"]]
kv: _UFunc21fc1[L["kv"]]
kve: _UFunc21fc1[L["kve"]]
yv: _UFunc21fc1[L["yv"]]
yve: _UFunc21fc1[L["yve"]]

# lf->f; ld->d; lF->F; lD->D
_spherical_in: np.ufunc
_spherical_in_d: np.ufunc
_spherical_jn: np.ufunc
_spherical_jn_d: np.ufunc
_spherical_kn: np.ufunc
_spherical_kn_d: np.ufunc
_spherical_yn: np.ufunc
_spherical_yn_d: np.ufunc

# ff->f; dd->d; FF->F; DD->D
elliprc: _UFunc21fc2[L["elliprc"], L[0]]
xlog1py: _UFunc21fc2[L["xlog1py"], L[0]]
xlogy: _UFunc21fc2[L["xlogy"], L[0]]

# ff->ff; dd->dd
# TODO
pbdv: np.ufunc
pbvv: np.ufunc
pbwa: np.ufunc

# ff->ffff; dd->dddd
# TODO
ellipj: np.ufunc

# fff->f; (ll|dl|dd)d->d
_beta_pdf: _UFunc31f[L["_beta_pdf"], L[0]]
_beta_ppf: _UFunc31f[L["_beta_ppf"], L[0]]
_binom_cdf: _UFunc31f[L["_binom_cdf"], L[0]]
_binom_isf: _UFunc31f[L["_binom_isf"], L[0]]
_binom_pmf: _UFunc31f[L["_binom_pmf"], L[0]]
_binom_ppf: _UFunc31f[L["_binom_ppf"], L[0]]
_binom_sf: _UFunc31f[L["_binom_sf"], L[0]]
_hypergeom_mean: _UFunc31f[L["_hypergeom_mean"], L[0]]
_hypergeom_skewness: _UFunc31f[L["_hypergeom_skewness"], L[0]]
_hypergeom_variance: _UFunc31f[L["_hypergeom_variance"], L[0]]
_invgauss_isf: _UFunc31f[L["_invgauss_isf"], L[0]]
_invgauss_ppf: _UFunc31f[L["_invgauss_ppf"], L[0]]
_nbinom_cdf: _UFunc31f[L["_nbinom_cdf"], L[0]]
_nbinom_isf: _UFunc31f[L["_nbinom_isf"], L[0]]
_nbinom_pmf: _UFunc31f[L["_nbinom_pmf"], L[0]]
_nbinom_ppf: _UFunc31f[L["_nbinom_ppf"], L[0]]
_nbinom_sf: _UFunc31f[L["_nbinom_sf"], L[0]]
_ncf_kurtosis_excess: _UFunc31f[L["_ncf_kurtosis_excess"], L[0]]
_ncf_mean: _UFunc31f[L["_ncf_mean"], L[0]]
_ncf_skewness: _UFunc31f[L["_ncf_skewness"], L[0]]
_ncf_variance: _UFunc31f[L["_ncf_variance"], L[0]]
_nct_cdf: _UFunc31f[L["_nct_cdf"], L[0]]
_nct_isf: _UFunc31f[L["_nct_isf"], L[0]]
_nct_ppf: _UFunc31f[L["_nct_ppf"], L[0]]
_nct_sf: _UFunc31f[L["_nct_sf"], L[0]]
_ncx2_cdf: _UFunc31f[L["_ncx2_cdf"], L[0]]
_ncx2_isf: _UFunc31f[L["_ncx2_isf"], L[0]]
_ncx2_pdf: _UFunc31f[L["_ncx2_pdf"], L[0]]
_ncx2_ppf: _UFunc31f[L["_ncx2_ppf"], L[0]]
_ncx2_sf: _UFunc31f[L["_ncx2_sf"], L[0]]
bdtr: _UFunc31f[L["bdtr"], L[0]]
bdtrc: _UFunc31f[L["bdtrc"], L[0]]
bdtri: _UFunc31f[L["bdtri"], L[0]]
bdtrik: _UFunc31f[L["bdtrik"], L[0]]
bdtrin: _UFunc31f[L["bdtrin"], L[0]]
besselpoly: _UFunc31f[L["besselpoly"], L[0]]
betainc: _UFunc31f[L["betainc"], L[0]]
betaincc: _UFunc31f[L["betaincc"], L[0]]
betainccinv: _UFunc31f[L["betainccinv"], L[0]]
betaincinv: _UFunc31f[L["betaincinv"], L[0]]
btdtr: _UFunc31f[L["btdtr"], L[0]]
btdtri: _UFunc31f[L["btdtri"], L[0]]
btdtria: _UFunc31f[L["btdtria"], L[0]]
btdtrib: _UFunc31f[L["btdtrib"], L[0]]
chndtr: _UFunc31f[L["chndtr"], L[0]]
chndtridf: _UFunc31f[L["chndtridf"], L[0]]
chndtrinc: _UFunc31f[L["chndtrinc"], L[0]]
chndtrix: _UFunc31f[L["chndtrix"], L[0]]
fdtr: _UFunc31f[L["fdtr"], L[0]]
fdtrc: _UFunc31f[L["fdtrc"], L[0]]
fdtri: _UFunc31f[L["fdtri"], L[0]]
fdtridfd: _UFunc31f[L["fdtridfd"], L[0]]
gdtr: _UFunc31f[L["gdtr"], L[0]]
gdtrc: _UFunc31f[L["gdtrc"], L[0]]
gdtria: _UFunc31f[L["gdtria"], L[0]]
gdtrib: _UFunc31f[L["gdtrib"], L[0]]
gdtrix: _UFunc31f[L["gdtrix"], L[0]]
hyperu: _UFunc31f[L["hyperu"], L[0]]
log_wright_bessel: _UFunc31f[L["log_wright_bessel"]]
lpmv: _UFunc31f[L["lpmv"], L[0]]
nbdtr: _UFunc31f[L["nbdtr"], L[0]]
nbdtrc: _UFunc31f[L["nbdtrc"], L[0]]
nbdtri: _UFunc31f[L["nbdtri"], L[0]]
nbdtrik: _UFunc31f[L["nbdtrik"], L[0]]
nbdtrin: _UFunc31f[L["nbdtrin"], L[0]]
nctdtr: _UFunc31f[L["nctdtr"], L[0]]
nctdtridf: _UFunc31f[L["nctdtridf"], L[0]]
nctdtrinc: _UFunc31f[L["nctdtrinc"], L[0]]
nctdtrit: _UFunc31f[L["nctdtrit"], L[0]]
nrdtrimn: _UFunc31f[L["nrdtrimn"], L[0]]
nrdtrisd: _UFunc31f[L["nrdtrisd"], L[0]]
obl_cv: _UFunc31f[L["obl_cv"]]
pro_cv: _UFunc31f[L["pro_cv"]]
radian: _UFunc31f[L["radian"], L[0]]
voigt_profile: _UFunc31f[L["voigt_profile"], L[0]]
wright_bessel: _UFunc31f[L["wright_bessel"]]

# Flf->F; Dld->D
_lambertw: np.ufunc

# fff->f; ddd->d; FFF->F; DDD->D
# TODO
elliprd: np.ufunc
elliprf: np.ufunc
elliprg: np.ufunc

# fff->f; ddd->d; ffF->F; (l|d)dD->D
# TODO
eval_gegenbauer: np.ufunc
eval_genlaguerre: np.ufunc
hyp1f1: np.ufunc

# fff->ff; ddd->dd
# TODO
mathieu_cem: np.ufunc
mathieu_modcem1: np.ufunc
mathieu_modcem2: np.ufunc
mathieu_modsem1: np.ufunc
mathieu_modsem2: np.ufunc
mathieu_sem: np.ufunc

# ddl->dd
_struve_asymp_large_z: np.ufunc
_struve_bessel_series: np.ufunc
_struve_power_series: np.ufunc

# ffff->f; dddd->d
_hypergeom_cdf: _UFunc41f[L["_hypergeom_cdf"], L[0]]
_hypergeom_pmf: _UFunc41f[L["_hypergeom_pmf"], L[0]]
_hypergeom_sf: _UFunc41f[L["_hypergeom_sf"], L[0]]
_ncf_cdf: _UFunc41f[L["_ncf_cdf"], L[0]]
_ncf_isf: _UFunc41f[L["_ncf_isf"], L[0]]
_ncf_pdf: _UFunc41f[L["_ncf_pdf"], L[0]]
_ncf_ppf: _UFunc41f[L["_ncf_ppf"], L[0]]
_ncf_sf: _UFunc41f[L["_ncf_sf"], L[0]]
_skewnorm_cdf: _UFunc41f[L["_skewnorm_cdf"], L[0]]
_skewnorm_isf: _UFunc41f[L["_skewnorm_isf"], L[0]]
_skewnorm_ppf: _UFunc41f[L["_skewnorm_ppf"], L[0]]
ncfdtr: _UFunc41f[L["ncfdtr"], L[0]]
ncfdtri: _UFunc41f[L["ncfdtri"], L[0]]
ncfdtridfd: _UFunc41f[L["ncfdtridfd"], L[0]]
ncfdtridfn: _UFunc41f[L["ncfdtridfn"], L[0]]
ncfdtrinc: _UFunc41f[L["ncfdtrinc"], L[0]]

# ffff->f; (l|d)ddd->d; fffF->F; dddD->D
# TODO
eval_jacobi: np.ufunc
eval_sh_jacobi: np.ufunc
hyp2f1: np.ufunc

# ffff->f; dddd->d; FFFF->F; DDDD->D
# TODO
elliprj: np.ufunc

# (ll|ff)ff->F; (ll|dd)dd->D
# TODO
sph_harm: np.ufunc

# ffff->ff; dddd->dd
# TODO
obl_ang1: np.ufunc
obl_rad1: np.ufunc
obl_rad2: np.ufunc
pro_ang1: np.ufunc
pro_rad1: np.ufunc
pro_rad2: np.ufunc

# fffff->ff; ddddd->dd
# TODO
obl_ang1_cv: np.ufunc
obl_rad1_cv: np.ufunc
obl_rad2_cv: np.ufunc
pro_ang1_cv: np.ufunc
pro_rad1_cv: np.ufunc
pro_rad2_cv: np.ufunc

# fffffff->f; dd(ll|dd)ddd->d
_ellip_harm: np.ufunc
