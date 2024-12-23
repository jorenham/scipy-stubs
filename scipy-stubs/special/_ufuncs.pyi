# mypy: disable-error-code="explicit-override, override, misc"
# pyright: reportIncompatibleMethodOverride=false, reportIncompatibleVariableOverride=false, reportImplicitOverride=false

from types import EllipsisType
from typing import Any, Generic, Literal as L, TypeAlias, TypedDict, final, overload, type_check_only
from typing_extensions import LiteralString, Never, TypeVar, Unpack, deprecated, override

import numpy as np
import optype as op
import optype.numpy as onp
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
_OutT1 = TypeVar("_OutT1", bound=onp.ArrayND[np.number[Any]])
_OutT2 = TypeVar("_OutT2", bound=onp.ArrayND[np.number[Any]])
_MaybeOutT = TypeVar("_MaybeOutT", bound=onp.ArrayND[np.number[Any]] | None, default=None)
_MaybeOutT1 = TypeVar("_MaybeOutT1", bound=onp.ArrayND[np.number[Any]] | None, default=None)
_MaybeOutT2 = TypeVar("_MaybeOutT2", bound=onp.ArrayND[np.number[Any]] | None, default=_MaybeOutT1)

_Falsy: TypeAlias = L[False, 0]
_Truthy: TypeAlias = L[True, 1]

_Out1: TypeAlias = tuple[_MaybeOutT] | _MaybeOutT
_Out2: TypeAlias = tuple[_MaybeOutT1, _MaybeOutT2]

_Tuple2: TypeAlias = tuple[_T, _T]
_Tuple3: TypeAlias = tuple[_T, _T, _T]
_Tuple4: TypeAlias = tuple[_T, _T, _T, _T]
_Tuple5: TypeAlias = tuple[_T, _T, _T, _T, _T]

_ToBool_D: TypeAlias = onp.ToBool | onp.ToBoolND
_ToInt_D: TypeAlias = onp.ToInt | onp.ToIntND

_Float64ND: TypeAlias = onp.ArrayND[np.float64]

_Float: TypeAlias = np.float32 | np.float64
_FloatND: TypeAlias = onp.ArrayND[_Float]
_Float_D: TypeAlias = _Float | onp.ArrayND[_Float]
_Float_DT = TypeVar("_Float_DT", bound=_Float_D)

_LFloat: TypeAlias = _Float | np.longdouble
_LFloatND: TypeAlias = onp.ArrayND[_LFloat]
_LFloat_D: TypeAlias = _LFloat | _LFloatND
_LFloat_DT = TypeVar("_LFloat_DT", bound=_LFloat_D)

_Complex: TypeAlias = np.complex64 | np.complex128
_ComplexND: TypeAlias = onp.ArrayND[_Complex]
_Complex_D: TypeAlias = _Complex | _ComplexND
_Complex_DT = TypeVar("_Complex_DT", bound=_Complex_D)

_Inexact: TypeAlias = _Float | _Complex
_InexactND: TypeAlias = onp.ArrayND[_Inexact]
_Inexact_D: TypeAlias = _Inexact | _InexactND
_Inexact_DT = TypeVar("_Inexact_DT", bound=_Inexact_D)

_CoInt: TypeAlias = np.integer[Any] | np.bool_  # coercible to integer
_CoFloat: TypeAlias = np.floating[Any] | _CoInt  # coercible to floating
_CoFloat64: TypeAlias = np.float64 | np.float32 | np.float16 | _CoInt  # coercible to float64
_CoComplex128: TypeAlias = np.complex128 | np.complex64 | _CoFloat64  # coercible to complex128

_CoIntND: TypeAlias = onp.ArrayND[_CoInt]
_CoFloatND: TypeAlias = onp.ArrayND[_CoFloat]
_CoFloat64ND: TypeAlias = onp.ArrayND[_CoFloat64]
_CoComplex128ND: TypeAlias = onp.ArrayND[_CoComplex128]

_SubFloat: TypeAlias = np.float16 | _CoInt  # anything "below" float32 | float64 that isn't float32 | float64
_ToSubFloat: TypeAlias = float | _SubFloat  # does not overlap with float32 | float64
_ToSubComplex: TypeAlias = complex | _SubFloat  # does not overlap with complex64 | complex128

_CoT = TypeVar("_CoT", bound=np.generic)
_ToT = TypeVar("_ToT")
_ToND: TypeAlias = onp.CanArrayND[_CoT] | onp.SequenceND[onp.CanArrayND[_CoT]] | onp.SequenceND[_ToT]

_ToFloat64: TypeAlias = float | _CoFloat64
_ToFloat64ND: TypeAlias = _ToND[_CoFloat64, _ToFloat64]
_ToFloat64_D: TypeAlias = _ToFloat64 | _ToFloat64ND

_ToComplex128: TypeAlias = complex | _CoComplex128
_ToComplex128ND: TypeAlias = _ToND[_CoComplex128, _ToComplex128]
_ToComplex128_D: TypeAlias = _ToComplex128 | _ToComplex128ND

_ToDType_l: TypeAlias = onp.AnyLongDType
_ToDType_q: TypeAlias = onp.AnyLongLongDType
_ToDType_f: TypeAlias = onp.AnyFloat32DType
_ToDType_d: TypeAlias = onp.AnyFloat64DType
_ToDType_g: TypeAlias = onp.AnyLongDoubleDType
_ToDType_F: TypeAlias = onp.AnyComplex64DType
_ToDType_D: TypeAlias = onp.AnyComplex128DType
_ToDType_fd: TypeAlias = _ToDType_f | _ToDType_d
_ToDType_fdg: TypeAlias = _ToDType_fd | _ToDType_g
_ToDType_FD: TypeAlias = _ToDType_F | _ToDType_D
_ToDType_fdFD: TypeAlias = _ToDType_fd | _ToDType_FD

_Axis: TypeAlias = AnyShape | None
_Indices: TypeAlias = op.CanIndex | slice | EllipsisType | tuple[op.CanIndex | slice | EllipsisType, ...] | onp.ToIntND

@type_check_only
class _KwBase(TypedDict, total=False):
    order: OrderKACF
    casting: Casting
    subok: onp.ToBool
    where: _ToBool_D

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
class _WithoutIdentity:
    # in binops, these will raise `TypeError`, otherwise a `ValueError`
    def accumulate(self, /, *args: object, **kwargs: object) -> Never: ...
    def reduce(self, /, *args: object, **kwargs: object) -> Never: ...
    def reduceat(self, /, *args: object, **kwargs: object) -> Never: ...

@type_check_only
class _WithoutBinOps(_WithoutIdentity):
    # Will raise `ValueError`
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
class _UFunc12(_WithoutAt, _WithoutBinOps, _UFunc[_NameT_co, _IdentityT_co], Generic[_NameT_co, _IdentityT_co]):
    @property
    @override
    def nin(self, /) -> L[1]: ...
    @property
    @override
    def nout(self, /) -> L[2]: ...
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
    dtype: _ToDType_fd
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
    dtype: _ToDType_fdFD | None
    signature: L["f->f", "d->d", "F->F", "D->D"] | _ToSignature1_fd | _ToSignature1_FD

_ToSignature2_fd2: TypeAlias = _Tuple3[_ToDType_f] | _Tuple3[_ToDType_d]
_ToSignature2_FD2: TypeAlias = _Tuple3[_ToDType_F] | _Tuple3[_ToDType_D]
_ToSignature2_fFdD: TypeAlias = tuple[_ToDType_f, _ToDType_F, _ToDType_F] | tuple[_ToDType_d, _ToDType_D, _ToDType_D]

@type_check_only
class _Kw12f(_KwBase, TypedDict, total=False):
    dtype: _ToDType_fd
    signature: L["f->ff", "d->dd"] | _ToSignature2_fd2

@type_check_only
class _Kw12c(_KwBase, TypedDict, total=False):
    dtype: _ToDType_FD
    signature: L["f->FF", "d->DD"] | _ToSignature2_fFdD

@type_check_only
class _Kw12fc(_KwBase, TypedDict, total=False):
    dtype: _ToDType_fdFD
    signature: L["f->ff", "d->dd", "F->FF", "D->DD"] | _ToSignature2_fd2 | _ToSignature2_FD2

_ToSignature2_ld: TypeAlias = tuple[_ToDType_l, _ToDType_d, _ToDType_d]

@type_check_only
class _Kw21ld(_KwBase, TypedDict, total=False):
    dtype: _ToDType_d | None
    signature: L["ld->d"] | _ToSignature2_ld

@type_check_only
class _Kw21f(_KwBase, TypedDict, total=False):
    dtype: _ToDType_fd
    signature: L["ff->f", "dd->d"] | _ToSignature2_fd2

@type_check_only
class _Kw21c1(_KwBase, TypedDict, total=False):
    dtype: _ToDType_FD | None
    signature: L["fF->F", "dD->D"] | _ToSignature2_fFdD

@type_check_only
class _Kw21fc1(_KwBase, TypedDict, total=False):
    dtype: _ToDType_fdFD | None
    signature: L["ff->f", "ld->d", "dd->d", "fF->F", "dD->D"] | _ToSignature2_ld | _ToSignature2_fd2 | _ToSignature2_fFdD

@type_check_only
class _Kw21fc2(_KwBase, TypedDict, total=False):
    dtype: _ToDType_fdFD | None
    signature: L["ff->f", "dd->d", "FF->F", "DD->D"] | _ToSignature2_fd2 | _ToSignature2_FD2

_ToSignature3_lld: TypeAlias = tuple[_ToDType_l, _ToDType_l, _ToDType_d, _ToDType_d]
_ToSignature3_dld: TypeAlias = tuple[_ToDType_d, _ToDType_l, _ToDType_d, _ToDType_d]
_ToSignature3_fd3: TypeAlias = _Tuple4[_ToDType_f] | _Tuple4[_ToDType_d]

@type_check_only
class _Kw31f(_KwBase, TypedDict, total=False):
    dtype: _ToDType_fd
    signature: L["fff->f", "ddd->d", "lld->d", "dld->d"] | _ToSignature3_fd3 | _ToSignature3_lld | _ToSignature3_dld

_ToSignature3_fd4: TypeAlias = _Tuple5[_ToDType_f] | _Tuple5[_ToDType_d]

@type_check_only
class _Kw41f(_KwBase, TypedDict, total=False):
    dtype: _ToDType_fd
    signature: L["ffff->f", "dddd->d"] | _ToSignature3_fd4

@type_check_only
class _Kw41fc1(_KwBase, TypedDict, total=False):
    # (qq|ff)ff->F; (qq|dd)dd->D
    dtype: _ToDType_FD
    signature: (
        L["ffff->f", "lddd->d", "dddd->d", "fffF->F", "dddD->D"]
        | tuple[_ToDType_l | _ToDType_fd, _ToDType_fd, _ToDType_fd, _ToDType_fdFD, _ToDType_fdFD]
    )

@type_check_only
class _KwSphHarm(_KwBase, TypedDict, total=False):
    # (qq|ff)ff->F; (qq|dd)dd->D
    dtype: _ToDType_FD
    signature: (
        L["qqff->F", "ffff->F", "qqdd->D", "dddd->D"]
        | tuple[_ToDType_q | _ToDType_fd, _ToDType_q | _ToDType_fd, _ToDType_fd, _ToDType_fd, _ToDType_FD]
    )

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
    def __call__(self, x: _ToSubFloat, /, out: _Out1 = None, **kw: Unpack[_Kw11f]) -> _Float: ...
    @overload
    def __call__(self, x: _Float_DT, /, out: _Out1 = None, **kw: Unpack[_Kw11f]) -> _Float_DT: ...
    @overload
    def __call__(self, x: _ToFloat64ND, /, out: _Out1 = None, **kw: Unpack[_Kw11f]) -> _FloatND: ...
    @overload
    def __call__(self, x: _ToFloat64_D, /, out: _Out1[_OutT], **kw: Unpack[_Kw11f]) -> _OutT: ...
    #
    @override
    def at(self, a: _CoFloat64ND, indices: _Indices, /) -> None: ...

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
    def __call__(self, x: _ToSubFloat, /, out: _Out1 = None, **kw: Unpack[_Kw11g]) -> _LFloat: ...
    @overload
    def __call__(self, x: _LFloat_DT, /, out: _Out1 = None, **kw: Unpack[_Kw11g]) -> _LFloat_DT: ...
    @overload
    def __call__(self, x: onp.ToFloatND, /, out: _Out1 = None, **kw: Unpack[_Kw11g]) -> _LFloatND: ...
    @overload
    def __call__(self, x: onp.ToFloat | onp.ToFloatND, /, out: _Out1[_OutT], **kw: Unpack[_Kw11g]) -> _OutT: ...
    #
    @override
    def at(self, a: _CoFloatND, indices: _Indices, /) -> None: ...

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
    def __call__(self, x: _ToSubComplex, /, out: _Out1 = None, **kw: Unpack[_Kw11c]) -> _Complex: ...
    @overload
    def __call__(self, x: _Complex_DT, /, out: _Out1 = None, **kw: Unpack[_Kw11c]) -> _Complex_DT: ...
    @overload
    def __call__(self, x: _ToComplex128ND, /, out: _Out1 = None, **kw: Unpack[_Kw11c]) -> _ComplexND: ...
    @overload
    def __call__(self, x: _ToComplex128_D, /, out: _Out1[_OutT], **kw: Unpack[_Kw11c]) -> _OutT: ...
    #
    @override
    def at(self, a: _CoComplex128ND, indices: _Indices, /) -> None: ...

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
    def __call__(self, x: _ToSubFloat, /, out: _Out1 = None, **kw: Unpack[_Kw11fc]) -> _Float: ...
    @overload
    def __call__(self, x: _ToSubComplex, /, out: _Out1 = None, **kw: Unpack[_Kw11fc]) -> _Inexact: ...
    @overload
    def __call__(self, x: _Inexact_DT, /, out: _Out1 = None, **kw: Unpack[_Kw11fc]) -> _Inexact_DT: ...
    @overload
    def __call__(self, x: _ToFloat64ND, /, out: _Out1 = None, **kw: Unpack[_Kw11fc]) -> _FloatND: ...
    @overload
    def __call__(self, x: _ToComplex128ND, /, out: _Out1 = None, **kw: Unpack[_Kw11fc]) -> _InexactND: ...
    @overload
    def __call__(self, x: _ToComplex128_D, /, out: _Out1[_OutT], **kw: Unpack[_Kw11fc]) -> _OutT: ...
    #
    @override
    def at(self, a: _CoComplex128ND, indices: _Indices, /) -> None: ...

@final
@type_check_only
class _UFunc12f(_UFunc12[_NameT_co, _IdentityT_co], Generic[_NameT_co, _IdentityT_co]):
    # for `it[2]i0k0` and `it[2]j0y0`
    @property
    @override
    def ntypes(self, /) -> L[2]: ...
    @property
    @override
    def types(self, /) -> list[L["f->ff", "d->dd"]]: ...
    #
    @overload
    def __call__(self, x: _ToSubFloat, /, out: _Out2 = ..., **kw: Unpack[_Kw12fc]) -> _Tuple2[_Float]: ...
    @overload
    def __call__(self, x: _ToSubComplex, /, out: _Out2 = ..., **kw: Unpack[_Kw12fc]) -> _Tuple2[_Inexact]: ...
    @overload
    def __call__(self, x: _Inexact_DT, /, out: _Out2 = ..., **kw: Unpack[_Kw12fc]) -> _Tuple2[_Inexact_DT]: ...
    @overload
    def __call__(self, x: _ToFloat64ND, /, out: _Out2 = ..., **kw: Unpack[_Kw12fc]) -> _Tuple2[_FloatND]: ...
    @overload
    def __call__(self, x: _ToComplex128ND, /, out: _Out2 = ..., **kw: Unpack[_Kw12fc]) -> _Tuple2[_InexactND]: ...
    @overload
    def __call__(self, x: _ToComplex128_D, /, out: tuple[_OutT1, _OutT2], **kw: Unpack[_Kw12fc]) -> tuple[_OutT1, _OutT2]: ...
    @overload
    def __call__(self, x: _ToComplex128_D, out1: _OutT1, out2: _OutT2, /, **kw: Unpack[_Kw12fc]) -> tuple[_OutT1, _OutT2]: ...

@final
@type_check_only
class _UFunc12c(_UFunc12[_NameT_co, _IdentityT_co], Generic[_NameT_co, _IdentityT_co]):
    # for the `modfresnel{m,p}`
    @property
    @override
    def ntypes(self, /) -> L[2]: ...
    @property
    @override
    def types(self, /) -> list[L["f->FF", "d->DD"]]: ...
    #
    @overload
    def __call__(self, x: _ToFloat64, /, out: _Out2 = ..., **kw: Unpack[_Kw12c]) -> _Tuple2[_Complex]: ...
    @overload
    def __call__(self, x: _ToFloat64ND, /, out: _Out2 = ..., **kw: Unpack[_Kw12c]) -> _Tuple2[_ComplexND]: ...
    @overload
    def __call__(self, x: _ToFloat64_D, /, out: tuple[_OutT1, _OutT2], **kw: Unpack[_Kw12c]) -> tuple[_OutT1, _OutT2]: ...
    @overload
    def __call__(self, x: _ToFloat64_D, out1: _OutT1, out2: _OutT2, /, **kw: Unpack[_Kw12c]) -> tuple[_OutT1, _OutT2]: ...

@final
@type_check_only
class _UFunc12fc(_UFunc12[_NameT_co, _IdentityT_co], Generic[_NameT_co, _IdentityT_co]):
    # for `fresnel`, `sici`, and `shichi`
    @property
    @override
    def ntypes(self, /) -> L[4]: ...
    @property
    @override
    def types(self, /) -> list[L["f->ff", "d->dd", "F->FF", "D->DD"]]: ...
    #
    @overload
    def __call__(self, x: _ToSubFloat, /, out: _Out2 = ..., **kw: Unpack[_Kw12f]) -> _Tuple2[_Float]: ...
    @overload
    def __call__(self, x: _Float_DT, /, out: _Out2 = ..., **kw: Unpack[_Kw12f]) -> _Tuple2[_Float_DT]: ...
    @overload
    def __call__(self, x: _ToFloat64ND, /, out: _Out2 = ..., **kw: Unpack[_Kw12f]) -> _Tuple2[_FloatND]: ...
    @overload
    def __call__(self, x: _ToFloat64_D, /, out: tuple[_OutT1, _OutT2], **kw: Unpack[_Kw12f]) -> tuple[_OutT1, _OutT2]: ...
    @overload
    def __call__(self, x: _ToFloat64_D, out1: _OutT1, out2: _OutT2, /, **kw: Unpack[_Kw12f]) -> tuple[_OutT1, _OutT2]: ...

@final
@type_check_only
class _UFunc21ld(_WithoutIdentity, _UFunc21[_NameT_co, _IdentityT_co], Generic[_NameT_co, _IdentityT_co]):
    @property
    @override
    def ntypes(self, /) -> L[1]: ...
    @property
    @override
    def types(self, /) -> list[L["ld->d"]]: ...
    #
    @overload
    def __call__(self, n: onp.ToInt, x: _ToFloat64, /, out: _Out1 = None, **kw: Unpack[_Kw21ld]) -> np.float64: ...
    @overload
    def __call__(self, n: onp.ToIntND, x: _ToFloat64_D, /, out: _Out1 = None, **kw: Unpack[_Kw21ld]) -> _Float64ND: ...
    @overload
    def __call__(self, n: _ToInt_D, x: _ToFloat64ND, /, out: _Out1 = None, **kw: Unpack[_Kw21ld]) -> _Float64ND: ...
    @overload
    def __call__(self, n: _ToInt_D, x: _ToFloat64_D, /, out: _Out1[_OutT], **kw: Unpack[_Kw21ld]) -> _OutT: ...
    #
    @override
    def at(self, a: _CoIntND, indices: _Indices, b: _ToFloat64_D, /) -> None: ...
    #
    @overload
    def outer(self, n: onp.ToInt, x: _ToFloat64, /, **kw: Unpack[_Kw21ld]) -> np.float64: ...
    @overload
    def outer(self, n: onp.ToIntND, x: _ToFloat64_D, /, **kw: Unpack[_Kw21ld]) -> _Float64ND: ...
    @overload
    def outer(self, n: _ToInt_D, x: _ToFloat64ND, /, **kw: Unpack[_Kw21ld]) -> _Float64ND: ...
    @overload
    def outer(self, n: _ToInt_D, x: _ToFloat64_D, /, *, out: _Out1[_OutT], **kw: Unpack[_Kw21ld]) -> _OutT: ...

@final
@type_check_only
class _UFunc21f(_UFunc21[_NameT_co, _IdentityT_co], Generic[_NameT_co, _IdentityT_co]):
    @property
    @override
    def ntypes(self, /) -> L[2, 3]: ...
    @property
    @override
    def types(self, /) -> list[L["ff->f", "dd->d"]] | list[L["ff->f", "ld->d", "dd->d"]]: ...
    #
    @overload
    def __call__(self, a: _ToSubFloat, b: _ToSubFloat, /, out: _Out1 = None, **kw: Unpack[_Kw21f]) -> _Float: ...
    @overload
    def __call__(self, a: _Float_DT, b: _ToFloat64, /, out: _Out1 = None, **kw: Unpack[_Kw21f]) -> _Float_DT: ...
    @overload
    def __call__(self, a: _ToFloat64, b: _Float_DT, /, out: _Out1 = None, **kw: Unpack[_Kw21f]) -> _Float_DT: ...
    @overload
    def __call__(self, a: _ToFloat64ND, b: _ToFloat64_D, /, out: _Out1 = None, **kw: Unpack[_Kw21f]) -> _FloatND: ...
    @overload
    def __call__(self, a: _ToFloat64_D, b: _ToFloat64ND, /, out: _Out1 = None, **kw: Unpack[_Kw21f]) -> _FloatND: ...
    @overload
    def __call__(self, a: _ToFloat64_D, b: _ToFloat64_D, /, out: _Out1[_OutT], **kw: Unpack[_Kw21f]) -> _OutT: ...
    #
    @override
    def at(self, a: _CoFloat64ND, indices: _Indices, b: _ToFloat64_D, /) -> None: ...
    #
    @override
    def accumulate(
        self,
        /,
        array: _ToFloat64ND,
        axis: op.CanIndex = 0,
        dtype: _ToDType_fd = None,
        out: _Out1[_FloatND | None] = None,
    ) -> _FloatND: ...
    #
    @overload
    def reduce(
        self,
        /,
        array: _ToFloat64ND,
        axis: None,
        dtype: _ToDType_fd = None,
        out: _Out1 = None,
        keepdims: _Falsy = False,
        initial: _ToFloat64 = ...,
        where: _ToBool_D = True,
    ) -> _Float: ...
    @overload
    def reduce(
        self,
        /,
        array: _ToFloat64ND,
        axis: _Axis = 0,
        dtype: _ToDType_fd = None,
        out: _Out1 = None,
        keepdims: _Falsy = False,
        initial: _ToFloat64 = ...,
        where: _ToBool_D = True,
    ) -> _Float_D: ...
    @overload
    def reduce(
        self,
        /,
        array: _ToFloat64ND,
        axis: _Axis = 0,
        dtype: _ToDType_fd = None,
        out: _Out1 = None,
        *,
        keepdims: _Truthy,
        initial: _ToFloat64 = ...,
        where: _ToBool_D = True,
    ) -> _FloatND: ...
    @overload
    def reduce(
        self,
        /,
        array: _ToFloat64ND,
        axis: _Axis,
        dtype: _ToDType_fd,
        out: _Out1[_OutT],
        keepdims: bool = False,
        initial: _ToFloat64 = ...,
        where: _ToBool_D = True,
    ) -> _OutT: ...
    @overload
    def reduce(
        self,
        /,
        array: _ToFloat64ND,
        axis: _Axis = 0,
        dtype: _ToDType_fd = None,
        *,
        out: _Out1[_OutT],
        keepdims: bool = False,
        initial: _ToFloat64 = ...,
        where: _ToBool_D = True,
    ) -> _OutT: ...
    #
    def reduceat(
        self,
        /,
        array: _ToFloat64ND,
        indices: _Indices,
        axis: op.CanIndex = 0,
        dtype: _ToDType_fd = None,
        out: _Out1[_FloatND | None] = None,
    ) -> _FloatND: ...
    #
    @overload
    def outer(self, a: _ToFloat64, b: _ToFloat64, /, *, out: _Out1 = None, **kw: Unpack[_Kw21f]) -> _Float: ...
    @overload
    def outer(self, a: _ToFloat64ND, b: _ToFloat64_D, /, *, out: _Out1 = None, **kw: Unpack[_Kw21f]) -> _FloatND: ...
    @overload
    def outer(self, a: _ToFloat64_D, b: _ToFloat64ND, /, *, out: _Out1 = None, **kw: Unpack[_Kw21f]) -> _FloatND: ...
    @overload
    def outer(self, a: _ToFloat64_D, b: _ToFloat64_D, /, *, out: _Out1[_OutT], **kw: Unpack[_Kw21f]) -> _OutT: ...

@final
@type_check_only
class _UFunc21c1(_WithoutIdentity, _UFunc21[_NameT_co, _IdentityT_co], Generic[_NameT_co, _IdentityT_co]):
    # for the four `hankel(1|2)e?` functions
    @property
    @override
    def ntypes(self, /) -> L[2]: ...
    @property
    @override
    def types(self, /) -> list[L["fF->F", "dD->D"]]: ...
    #
    @overload
    def __call__(self, a: _ToFloat64, b: _ToSubComplex, /, out: _Out1 = None, **kw: Unpack[_Kw21c1]) -> _Complex: ...
    @overload
    def __call__(self, a: _ToSubFloat | np.float32, b: _Complex_DT, /, out: _Out1 = None) -> _Complex_DT: ...
    @overload
    def __call__(self, a: _ToFloat64ND, b: _ToComplex128_D, /, out: _Out1 = None, **kw: Unpack[_Kw21c1]) -> _ComplexND: ...
    @overload
    def __call__(self, a: _ToFloat64_D, b: _ToComplex128ND, /, out: _Out1 = None, **kw: Unpack[_Kw21c1]) -> _ComplexND: ...
    @overload
    def __call__(self, a: _ToFloat64_D, b: _ToComplex128_D, /, out: _Out1[_OutT], **kw: Unpack[_Kw21c1]) -> _OutT: ...
    #
    @override
    @deprecated("Casting complex values to real discards the imaginary part.", category=np.exceptions.ComplexWarning)
    def at(self, a: _CoFloat64ND, indices: _Indices, b: _ToFloat64ND, /) -> None: ...

    #
    @overload
    def outer(self, a: _ToFloat64, b: _ToComplex128, /, **kw: Unpack[_Kw21c1]) -> _Complex: ...
    @overload
    def outer(self, a: _ToFloat64ND, b: _ToComplex128_D, /, **kw: Unpack[_Kw21c1]) -> _ComplexND: ...
    @overload
    def outer(self, a: _ToFloat64_D, b: _ToComplex128ND, /, **kw: Unpack[_Kw21c1]) -> _ComplexND: ...
    @overload
    def outer(self, a: _ToFloat64_D, b: _ToComplex128_D, /, *, out: _Out1[_OutT], **kw: Unpack[_Kw21c1]) -> _OutT: ...

@final
@type_check_only
class _UFunc21fc1(_UFunc21[_NameT_co, _IdentityT_co], Generic[_NameT_co, _IdentityT_co]):
    @property
    @override
    def ntypes(self, /) -> L[4, 5]: ...
    @property
    @override
    def types(self, /) -> list[L["ff->f", "dd->d", "fF->F", "dD->D"]] | list[L["ff->f", "ld->d", "dd->d", "fF->F", "dD->D"]]: ...
    #
    @overload
    def __call__(self, a: _ToFloat64, b: _ToSubFloat, /, out: _Out1 = None, **kw: Unpack[_Kw21fc1]) -> _Float: ...
    @overload
    def __call__(self, a: _ToSubFloat | np.float32, b: _Inexact_DT, /, out: _Out1 = None) -> _Inexact_DT: ...
    @overload
    def __call__(self, a: _ToFloat64_D, b: _ToFloat64ND, /, out: _Out1 = None, **kw: Unpack[_Kw21fc1]) -> _FloatND: ...
    @overload
    def __call__(self, a: _ToFloat64ND, b: _ToFloat64_D, /, out: _Out1 = None, **kw: Unpack[_Kw21fc1]) -> _FloatND: ...
    @overload
    def __call__(self, a: _ToFloat64ND, b: _ToComplex128_D, /, out: _Out1 = None, **kw: Unpack[_Kw21fc1]) -> _InexactND: ...
    @overload
    def __call__(self, a: _ToFloat64_D, b: _ToComplex128ND, /, out: _Out1 = None, **kw: Unpack[_Kw21fc1]) -> _InexactND: ...
    @overload
    def __call__(self, a: _ToFloat64_D, b: _ToComplex128_D, /, out: _Out1[_OutT], **kw: Unpack[_Kw21fc1]) -> _OutT: ...
    #
    @override  # only works if real
    def at(self, a: _CoFloat64ND, indices: _Indices, b: _ToFloat64_D, /) -> None: ...
    #
    @override
    def accumulate(
        self,
        /,
        array: _ToFloat64ND,
        axis: op.CanIndex = 0,
        dtype: _ToDType_fd = None,
        out: _Out1[_FloatND | None] = None,
    ) -> _FloatND: ...
    #
    @overload
    def reduce(
        self,
        /,
        array: _ToFloat64ND,
        axis: None,
        dtype: _ToDType_fd = None,
        out: _Out1 = None,
        keepdims: _Falsy = False,
        initial: _ToFloat64 = ...,
        where: _ToBool_D = True,
    ) -> _Float: ...
    @overload
    def reduce(
        self,
        /,
        array: _ToFloat64ND,
        axis: _Axis = 0,
        dtype: _ToDType_fd = None,
        out: _Out1 = None,
        keepdims: _Falsy = False,
        initial: _ToFloat64 = ...,
        where: _ToBool_D = True,
    ) -> _Float_D: ...
    @overload
    def reduce(
        self,
        /,
        array: _ToFloat64ND,
        axis: _Axis = 0,
        dtype: _ToDType_fd = None,
        out: _Out1 = None,
        *,
        keepdims: _Truthy,
        initial: _ToFloat64 = ...,
        where: _ToBool_D = True,
    ) -> _FloatND: ...
    @overload
    def reduce(
        self,
        /,
        array: _ToFloat64ND,
        axis: _Axis,
        dtype: _ToDType_fd,
        out: _Out1[_OutT],
        keepdims: bool = False,
        initial: _ToFloat64 = ...,
        where: _ToBool_D = True,
    ) -> _OutT: ...
    @overload
    def reduce(
        self,
        /,
        array: _ToFloat64ND,
        axis: _Axis = 0,
        dtype: _ToDType_fd = None,
        *,
        out: _Out1[_OutT],
        keepdims: bool = False,
        initial: _ToFloat64 = ...,
        where: _ToBool_D = True,
    ) -> _OutT: ...
    #
    def reduceat(
        self,
        /,
        array: _ToFloat64ND,
        indices: _Indices,
        axis: op.CanIndex = 0,
        dtype: _ToDType_fd = None,
        out: _Out1[_FloatND | None] = None,
    ) -> _FloatND: ...
    #
    @overload
    def outer(self, a: _ToFloat64, b: _ToFloat64, /, **kw: Unpack[_Kw21fc1]) -> _Float: ...
    @overload
    def outer(self, a: _ToFloat64, b: _ToComplex128, /, **kw: Unpack[_Kw21fc1]) -> _Inexact: ...
    @overload
    def outer(self, a: _ToFloat64_D, b: _ToFloat64ND, /, **kw: Unpack[_Kw21fc1]) -> _FloatND: ...
    @overload
    def outer(self, a: _ToFloat64ND, b: _ToFloat64_D, /, **kw: Unpack[_Kw21fc1]) -> _FloatND: ...
    @overload
    def outer(self, a: _ToFloat64ND, b: _ToComplex128_D, /, **kw: Unpack[_Kw21fc1]) -> _InexactND: ...
    @overload
    def outer(self, a: _ToFloat64_D, b: _ToComplex128ND, /, **kw: Unpack[_Kw21fc1]) -> _InexactND: ...
    @overload
    def outer(self, a: _ToFloat64_D, b: _ToComplex128_D, /, *, out: _Out1[_OutT], **kw: Unpack[_Kw21fc1]) -> _OutT: ...

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
    def __call__(self, a: _ToSubFloat, b: _ToSubFloat, /, out: _Out1 = None, **kw: Unpack[_Kw21fc2]) -> _Float: ...
    @overload
    def __call__(self, a: _ToSubComplex, b: _ToSubComplex, /, out: _Out1 = None, **kw: Unpack[_Kw21fc2]) -> _Inexact: ...
    @overload
    def __call__(self, a: _ToSubFloat | np.float32, b: _Inexact_DT, /, out: _Out1 = None) -> _Inexact_DT: ...
    @overload
    def __call__(self, a: _Inexact_DT, b: _ToSubFloat | np.float32, /, out: _Out1 = None) -> _Inexact_DT: ...
    @overload
    def __call__(self, a: _ToFloat64_D, b: _ToFloat64ND, /, out: _Out1 = None, **kw: Unpack[_Kw21fc2]) -> _FloatND: ...
    @overload
    def __call__(self, a: _ToFloat64ND, b: _ToFloat64_D, /, out: _Out1 = None, **kw: Unpack[_Kw21fc2]) -> _FloatND: ...
    @overload
    def __call__(self, a: _ToComplex128ND, b: _ToComplex128_D, /, out: _Out1 = None, **kw: Unpack[_Kw21fc2]) -> _InexactND: ...
    @overload
    def __call__(self, a: _ToComplex128_D, b: _ToComplex128ND, /, out: _Out1 = None, **kw: Unpack[_Kw21fc2]) -> _InexactND: ...
    @overload
    def __call__(self, a: _ToComplex128_D, b: _ToComplex128_D, /, out: _Out1[_OutT], **kw: Unpack[_Kw21fc2]) -> _OutT: ...
    #
    @overload
    def at(self, x: _CoFloat64ND, indices: _Indices, y: _ToFloat64ND, /) -> None: ...
    @overload
    def at(self, x: _CoComplex128ND, indices: _Indices, y: _ToComplex128ND, /) -> None: ...
    #
    @overload
    def accumulate(
        self,
        /,
        array: _ToFloat64ND,
        axis: op.CanIndex = 0,
        dtype: _ToDType_fd = None,
        out: _Out1[_FloatND | None] = None,
    ) -> _FloatND: ...
    @overload
    def accumulate(
        self,
        /,
        array: _ToComplex128ND,
        axis: op.CanIndex = 0,
        dtype: _ToDType_fdFD | None = None,
        out: _Out1[_InexactND | None] = None,
    ) -> _InexactND: ...
    #
    @overload
    def reduce(
        self,
        /,
        array: _ToFloat64ND,
        axis: None,
        dtype: _ToDType_fd = None,
        out: _Out1 = None,
        keepdims: _Falsy = False,
        initial: onp.ToFloat = ...,
        where: _ToBool_D = True,
    ) -> _Float: ...
    @overload
    def reduce(
        self,
        /,
        array: _ToFloat64ND,
        axis: _Axis = 0,
        dtype: _ToDType_fd = None,
        out: _Out1 = None,
        *,
        keepdims: _Truthy,
        initial: onp.ToFloat = ...,
        where: _ToBool_D = True,
    ) -> _FloatND: ...
    @overload
    def reduce(
        self,
        /,
        array: _ToComplex128ND,
        axis: None,
        dtype: _ToDType_fdFD | None = None,
        out: _Out1 = None,
        keepdims: _Falsy = False,
        initial: _ToComplex128 = ...,
        where: _ToBool_D = True,
    ) -> _Inexact: ...
    @overload
    def reduce(
        self,
        /,
        array: _ToComplex128ND,
        axis: _Axis = 0,
        dtype: _ToDType_fdFD | None = None,
        out: _Out1 = None,
        *,
        keepdims: _Truthy,
        initial: _ToComplex128 = ...,
        where: _ToBool_D = True,
    ) -> _InexactND: ...
    @overload
    def reduce(
        self,
        /,
        array: _ToComplex128ND,
        axis: _Axis = 0,
        dtype: _ToDType_fdFD | None = None,
        out: _Out1 = None,
        keepdims: _Falsy = False,
        initial: _ToComplex128 = ...,
        where: _ToBool_D = True,
    ) -> _Inexact_D: ...
    @overload
    def reduce(
        self,
        /,
        array: _ToComplex128ND,
        axis: _Axis,
        dtype: _ToDType_fdFD | None,
        out: _Out1[_OutT],
        keepdims: bool = False,
        initial: _ToComplex128 = ...,
        where: _ToBool_D = True,
    ) -> _OutT: ...
    @overload
    def reduce(
        self,
        /,
        array: _ToComplex128ND,
        axis: _Axis = 0,
        dtype: _ToDType_fdFD | None = None,
        *,
        out: _Out1[_OutT],
        keepdims: bool = False,
        initial: _ToComplex128 = ...,
        where: _ToBool_D = True,
    ) -> _OutT: ...
    #
    @overload
    def reduceat(
        self,
        /,
        array: _ToFloat64ND,
        indices: _Indices,
        axis: op.CanIndex = 0,
        dtype: _ToDType_fd = None,
        out: _Out1[_FloatND | None] = None,
    ) -> _FloatND: ...
    @overload
    def reduceat(
        self,
        /,
        array: _ToComplex128ND,
        indices: _Indices,
        axis: op.CanIndex = 0,
        dtype: _ToDType_fdFD | None = None,
        out: _Out1[_InexactND | None] = None,
    ) -> _InexactND: ...
    #
    @overload
    def outer(self, a: _ToFloat64, b: _ToFloat64, /, **kw: Unpack[_Kw21fc2]) -> _Float: ...
    @overload
    def outer(self, a: _ToComplex128, b: _ToComplex128, /, **kw: Unpack[_Kw21fc2]) -> _Inexact: ...
    @overload
    def outer(self, a: _ToFloat64_D, b: _ToFloat64ND, /, **kw: Unpack[_Kw21fc2]) -> _FloatND: ...
    @overload
    def outer(self, a: _ToFloat64ND, b: _ToFloat64_D, /, **kw: Unpack[_Kw21fc2]) -> _FloatND: ...
    @overload
    def outer(self, a: _ToComplex128ND, b: _ToComplex128_D, /, **kw: Unpack[_Kw21fc2]) -> _InexactND: ...
    @overload
    def outer(self, a: _ToComplex128_D, b: _ToComplex128ND, /, **kw: Unpack[_Kw21fc2]) -> _InexactND: ...
    @overload
    def outer(self, a: _ToComplex128_D, b: _ToComplex128_D, /, *, out: _Out1[_OutT], **kw: Unpack[_Kw21fc2]) -> _OutT: ...

@final
@type_check_only
class _UFunc31f(_UFunc31[_NameT_co, _IdentityT_co], Generic[_NameT_co, _IdentityT_co]):
    @property
    @override
    def ntypes(self, /) -> L[2, 3]: ...
    @property
    @override
    def types(self, /) -> list[L["fff->f", "lld->d", "ddd->d"]] | list[L["fff->f", "dld->d", "ddd->d"]]: ...
    #
    @overload
    def __call__(self, a: _ToSubFloat, b: _ToSubFloat, x: _ToSubFloat, /, out: _Out1 = None, **kw: Unpack[_Kw31f]) -> _Float: ...
    @overload
    def __call__(self, a: _ToFloat64, b: _ToFloat64, x: _Float_DT, /, out: _Out1 = None, **kw: Unpack[_Kw31f]) -> _Float_DT: ...
    @overload
    def __call__(self, a: _ToFloat64, b: _Float_DT, x: _ToFloat64, /, out: _Out1 = None, **kw: Unpack[_Kw31f]) -> _Float_DT: ...
    @overload
    def __call__(self, a: _Float_DT, b: _ToFloat64, x: _ToFloat64, /, out: _Out1 = None, **kw: Unpack[_Kw31f]) -> _Float_DT: ...
    @overload
    def __call__(
        self,
        a: _ToFloat64_D,
        b: _ToFloat64_D,
        x: _ToFloat64ND,
        /,
        out: _Out1 = None,
        **kw: Unpack[_Kw31f],
    ) -> _FloatND: ...
    @overload
    def __call__(
        self,
        a: _ToFloat64_D,
        b: _ToFloat64ND,
        x: _ToFloat64_D,
        /,
        out: _Out1 = None,
        **kw: Unpack[_Kw31f],
    ) -> _FloatND: ...
    @overload
    def __call__(
        self,
        a: _ToFloat64ND,
        b: _ToFloat64_D,
        x: _ToFloat64_D,
        /,
        out: _Out1 = None,
        **kw: Unpack[_Kw31f],
    ) -> _FloatND: ...
    @overload
    def __call__(
        self,
        a: _ToFloat64_D,
        b: _ToFloat64_D,
        x: _ToFloat64_D,
        /,
        out: _Out1[_OutT],
        **kw: Unpack[_Kw31f],
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
        out: _Out1 = None,
        **kw: Unpack[_Kw41f],
    ) -> _Float: ...
    @overload
    def __call__(
        self,
        dfn: _ToFloat64,
        dfd: _ToFloat64,
        nc: _ToFloat64,
        f: _Float_DT,
        /,
        out: _Out1 = None,
        **kw: Unpack[_Kw41f],
    ) -> _Float_DT: ...
    @overload
    def __call__(
        self,
        dfn: _ToFloat64,
        dfd: _ToFloat64,
        nc: _Float_DT,
        f: _ToFloat64,
        /,
        out: _Out1 = None,
        **kw: Unpack[_Kw41f],
    ) -> _Float_DT: ...
    @overload
    def __call__(
        self,
        dfn: _ToFloat64,
        dfd: _Float_DT,
        nc: _ToFloat64,
        f: _ToFloat64,
        /,
        out: _Out1 = None,
        **kw: Unpack[_Kw41f],
    ) -> _Float_DT: ...
    @overload
    def __call__(
        self,
        dfn: _Float_DT,
        dfd: _ToFloat64,
        nc: _ToFloat64,
        f: _ToFloat64,
        /,
        out: _Out1 = None,
        **kw: Unpack[_Kw41f],
    ) -> _Float_DT: ...
    @overload
    def __call__(
        self,
        dfn: _ToFloat64_D,
        dfd: _ToFloat64_D,
        nc: _ToFloat64_D,
        f: _ToFloat64ND,
        /,
        out: _Out1 = None,
        **kw: Unpack[_Kw41f],
    ) -> _FloatND: ...
    @overload
    def __call__(
        self,
        dfn: _ToFloat64_D,
        dfd: _ToFloat64ND,
        nc: _ToFloat64_D,
        f: _ToFloat64_D,
        /,
        out: _Out1 = None,
        **kw: Unpack[_Kw41f],
    ) -> _FloatND: ...
    @overload
    def __call__(
        self,
        dfn: _ToFloat64_D,
        dfd: _ToFloat64_D,
        nc: _ToFloat64ND,
        f: _ToFloat64_D,
        /,
        out: _Out1 = None,
        **kw: Unpack[_Kw41f],
    ) -> _FloatND: ...
    @overload
    def __call__(
        self,
        dfn: _ToFloat64ND,
        dfd: _ToFloat64_D,
        nc: _ToFloat64_D,
        f: _ToFloat64_D,
        /,
        out: _Out1 = None,
        **kw: Unpack[_Kw41f],
    ) -> _FloatND: ...
    @overload
    def __call__(
        self,
        dfn: _ToFloat64_D,
        dfd: _ToFloat64_D,
        nc: _ToFloat64_D,
        f: _ToFloat64_D,
        /,
        out: _Out1[_OutT],
        **kw: Unpack[_Kw41f],
    ) -> _OutT: ...

@final
@type_check_only
class _UFunc41fc1(_UFunc41[_NameT_co, _IdentityT_co], Generic[_NameT_co, _IdentityT_co]):
    # `eval_[sh_]jacobi` and `hyp2f1`
    @property
    @override
    def ntypes(self, /) -> L[4, 5]: ...
    @property
    @override
    def types(self, /) -> list[L["ffff->f", "lddd->d", "dddd->d", "fffF->F", "dddD->D"]]: ...
    #
    @overload
    def __call__(
        self,
        n: _ToSubFloat,
        a: _ToSubFloat,
        b: _ToSubFloat,
        x: _ToSubFloat,
        /,
        out: _Out1 = None,
        **kw: Unpack[_Kw41fc1],
    ) -> _Float: ...
    @overload
    def __call__(
        self,
        n: _ToSubFloat,
        a: _ToSubFloat,
        b: _ToSubFloat,
        x: _ToSubComplex,
        /,
        out: _Out1 = None,
        **kw: Unpack[_Kw41fc1],
    ) -> _Inexact: ...
    @overload
    def __call__(
        self,
        n: _ToFloat64,
        a: _ToFloat64,
        b: _ToFloat64,
        x: _Inexact_DT,
        /,
        out: _Out1 = None,
        **kw: Unpack[_Kw41fc1],
    ) -> _Inexact_DT: ...
    @overload
    def __call__(
        self,
        n: _ToFloat64,
        a: _ToFloat64,
        b: _Float_DT,
        x: _ToFloat64,
        /,
        out: _Out1 = None,
        **kw: Unpack[_Kw41fc1],
    ) -> _Float_DT: ...
    @overload
    def __call__(
        self,
        n: _ToFloat64,
        a: _Float_DT,
        b: _ToFloat64,
        x: _ToFloat64,
        /,
        out: _Out1 = None,
        **kw: Unpack[_Kw41fc1],
    ) -> _Float_DT: ...
    @overload
    def __call__(
        self,
        n: _Float_DT,
        a: _ToFloat64,
        b: _ToFloat64,
        x: _ToFloat64,
        /,
        out: _Out1 = None,
        **kw: Unpack[_Kw41fc1],
    ) -> _Float_DT: ...
    @overload
    def __call__(
        self,
        n: _ToFloat64_D,
        a: _ToFloat64_D,
        b: _ToFloat64_D,
        x: _ToFloat64ND,
        /,
        out: _Out1 = None,
        **kw: Unpack[_Kw41fc1],
    ) -> _FloatND: ...
    @overload
    def __call__(
        self,
        n: _ToFloat64_D,
        a: _ToFloat64_D,
        b: _ToFloat64ND,
        x: _ToFloat64_D,
        /,
        out: _Out1 = None,
        **kw: Unpack[_Kw41fc1],
    ) -> _FloatND: ...
    @overload
    def __call__(
        self,
        n: _ToFloat64_D,
        a: _ToFloat64ND,
        b: _ToFloat64_D,
        x: _ToFloat64_D,
        /,
        out: _Out1 = None,
        **kw: Unpack[_Kw41fc1],
    ) -> _FloatND: ...
    @overload
    def __call__(
        self,
        n: _ToFloat64ND,
        a: _ToFloat64_D,
        b: _ToFloat64_D,
        x: _ToFloat64_D,
        /,
        out: _Out1 = None,
        **kw: Unpack[_Kw41fc1],
    ) -> _FloatND: ...
    @overload
    def __call__(
        self,
        n: _ToFloat64_D,
        a: _ToFloat64_D,
        b: _ToFloat64_D,
        x: _ToComplex128ND,
        /,
        out: _Out1 = None,
        **kw: Unpack[_Kw41fc1],
    ) -> _InexactND: ...
    @overload
    def __call__(
        self,
        n: _ToFloat64_D,
        a: _ToFloat64_D,
        b: _ToFloat64ND,
        x: _ToComplex128_D,
        /,
        out: _Out1 = None,
        **kw: Unpack[_Kw41fc1],
    ) -> _InexactND: ...
    @overload
    def __call__(
        self,
        n: _ToFloat64_D,
        a: _ToFloat64ND,
        b: _ToFloat64_D,
        x: _ToComplex128_D,
        /,
        out: _Out1 = None,
        **kw: Unpack[_Kw41fc1],
    ) -> _InexactND: ...
    @overload
    def __call__(
        self,
        n: _ToFloat64ND,
        a: _ToFloat64_D,
        b: _ToFloat64_D,
        x: _ToComplex128_D,
        /,
        out: _Out1 = None,
        **kw: Unpack[_Kw41fc1],
    ) -> _InexactND: ...
    @overload
    def __call__(
        self,
        n: _ToFloat64_D,
        a: _ToFloat64_D,
        b: _ToFloat64_D,
        x: _ToComplex128_D,
        /,
        out: _Out1[_OutT],
        **kw: Unpack[_Kw41fc1],
    ) -> _OutT: ...

@final
@type_check_only
class _UFuncSphHarm(_UFunc41[L["sph_harm"], None]):
    @property
    @override
    def ntypes(self, /) -> L[4]: ...
    @property
    @override
    def types(self, /) -> list[L["qqff->F", "ffff->F", "qqdd->D", "dddd->D"]]: ...
    #
    @overload
    @deprecated("This function is deprecated and will be removed in SciPy 1.17.0. Use `scipy.special.sph_harm_y` instead.")
    def __call__(
        self,
        m: _ToFloat64,
        n: _ToFloat64,
        theta: _ToFloat64,
        phi: _ToFloat64,
        /,
        out: _Out1 = None,
        **kw: Unpack[_KwSphHarm],
    ) -> _Complex: ...
    @overload
    @deprecated("This function is deprecated and will be removed in SciPy 1.17.0. Use `scipy.special.sph_harm_y` instead.")
    def __call__(
        self,
        m: _ToFloat64_D,
        n: _ToFloat64_D,
        theta: _ToFloat64_D,
        phi: _ToFloat64ND,
        /,
        out: _Out1 = None,
        **kw: Unpack[_KwSphHarm],
    ) -> _ComplexND: ...
    @overload
    @deprecated("This function is deprecated and will be removed in SciPy 1.17.0. Use `scipy.special.sph_harm_y` instead.")
    def __call__(
        self,
        m: _ToFloat64_D,
        n: _ToFloat64_D,
        theta: _ToFloat64ND,
        phi: _ToFloat64_D,
        /,
        out: _Out1 = None,
        **kw: Unpack[_KwSphHarm],
    ) -> _ComplexND: ...
    @overload
    @deprecated("This function is deprecated and will be removed in SciPy 1.17.0. Use `scipy.special.sph_harm_y` instead.")
    def __call__(
        self,
        m: _ToFloat64_D,
        n: _ToFloat64ND,
        theta: _ToFloat64_D,
        phi: _ToFloat64_D,
        /,
        out: _Out1 = None,
        **kw: Unpack[_KwSphHarm],
    ) -> _ComplexND: ...
    @overload
    @deprecated("This function is deprecated and will be removed in SciPy 1.17.0. Use `scipy.special.sph_harm_y` instead.")
    def __call__(
        self,
        m: _ToFloat64_D,
        n: _ToFloat64_D,
        theta: _ToFloat64_D,
        phi: _ToFloat64_D,
        /,
        out: _Out1[_OutT],
        **kw: Unpack[_KwSphHarm],
    ) -> _OutT: ...

###

def geterr() -> dict[str, str]: ...
def seterr(**kwargs: str) -> dict[str, str]: ...

class errstate(EnterNoneMixin):
    def __init__(self, /, **kwargs: str) -> None: ...

# l->l
_sf_error_test_function: np.ufunc

# f->f; d->d
_cosine_cdf: _UFunc11f[L["_cosine_cdf"], L[0]] = ...
_cosine_invcdf: _UFunc11f[L["_cosine_invcdf"], L[0]] = ...
_factorial: _UFunc11f[L["_factorial"], L[0]] = ...
_kolmogc: _UFunc11f[L["_kolmogc"], L[0]] = ...
_kolmogci: _UFunc11f[L["_kolmogci"], L[0]] = ...
_kolmogp: _UFunc11f[L["_kolmogp"], L[0]] = ...
_lanczos_sum_expg_scaled: _UFunc11f[L["_lanczos_sum_expg_scaled"], L[0]] = ...
_lgam1p: _UFunc11f[L["_lgam1p"], L[0]] = ...
_log1pmx: _UFunc11f[L["_log1pmx"], L[0]] = ...
_scaled_exp1: _UFunc11f[L["_scaled_exp1"]] = ...
bei: _UFunc11f[L["bei"]] = ...
beip: _UFunc11f[L["beip"]] = ...
ber: _UFunc11f[L["ber"]] = ...
berp: _UFunc11f[L["berp"]] = ...
cbrt: _UFunc11f[L["cbrt"], L[0]] = ...
cosdg: _UFunc11f[L["cosdg"], L[0]] = ...
cosm1: _UFunc11f[L["cosm1"], L[0]] = ...
cotdg: _UFunc11f[L["cotdg"], L[0]] = ...
ellipe: _UFunc11f[L["ellipe"], L[0]] = ...
ellipk: _UFunc11f[L["ellipk"], L[0]] = ...
ellipkm1: _UFunc11f[L["ellipkm1"], L[0]] = ...
entr: _UFunc11f[L["entr"], L[0]] = ...
erfcinv: _UFunc11f[L["erfcinv"], L[0]] = ...
erfinv: _UFunc11f[L["erfinv"], L[0]] = ...
exp10: _UFunc11f[L["exp10"], L[0]] = ...
exp2: _UFunc11f[L["exp2"], L[0]] = ...
exprel: _UFunc11f[L["exprel"]] = ...
gammaln: _UFunc11f[L["gammaln"]] = ...
gammasgn: _UFunc11f[L["gammasgn"], L[0]] = ...
i0: _UFunc11f[L["i0"], L[0]] = ...
i0e: _UFunc11f[L["i0e"], L[0]] = ...
i1: _UFunc11f[L["i1"], L[0]] = ...
i1e: _UFunc11f[L["i1e"], L[0]] = ...
it2struve0: _UFunc11f[L["it2struve0"]] = ...
itmodstruve0: _UFunc11f[L["itmodstruve0"]] = ...
itstruve0: _UFunc11f[L["itstruve0"]] = ...
j0: _UFunc11f[L["j0"], L[0]] = ...
j1: _UFunc11f[L["j1"], L[0]] = ...
k0: _UFunc11f[L["k0"], L[0]] = ...
k0e: _UFunc11f[L["k0e"], L[0]] = ...
k1: _UFunc11f[L["k1"], L[0]] = ...
k1e: _UFunc11f[L["k1e"], L[0]] = ...
kei: _UFunc11f[L["kei"]] = ...
keip: _UFunc11f[L["keip"]] = ...
ker: _UFunc11f[L["ker"]] = ...
kerp: _UFunc11f[L["kerp"]] = ...
kolmogi: _UFunc11f[L["kolmogi"], L[0]] = ...
kolmogorov: _UFunc11f[L["kolmogorov"], L[0]] = ...
ndtri: _UFunc11f[L["ndtri"], L[0]] = ...
ndtri_exp: _UFunc11f[L["ndtri_exp"], L[0]] = ...
round: _UFunc11f[L["round"], L[0]] = ...
sindg: _UFunc11f[L["sindg"], L[0]] = ...
tandg: _UFunc11f[L["tandg"], L[0]] = ...
y0: _UFunc11f[L["y0"], L[0]] = ...
y1: _UFunc11f[L["y1"], L[0]] = ...
zetac: _UFunc11f[L["zetac"], L[0]] = ...

# f->f; d->d; g->g
expit: _UFunc11g[L["expit"]] = ...
log_expit: _UFunc11g[L["log_expit"]] = ...
logit: _UFunc11g[L["logit"]] = ...

# F->F; D->D
wofz: _UFunc11c[L["wofz"], L[0]] = ...

# f->f; d->d; F->F; D->D
_cospi: _UFunc11fc[L["_cospi"]] = ...
_sinpi: _UFunc11fc[L["_sinpi"]] = ...
_riemann_zeta: _UFunc11fc[L["_riemann_zeta"], L[0]] = ...
dawsn: _UFunc11fc[L["dawsn"], L[0]] = ...
erf: _UFunc11fc[L["erf"], L[0]] = ...
erfc: _UFunc11fc[L["erfc"], L[0]] = ...
erfcx: _UFunc11fc[L["erfcx"], L[0]] = ...
erfi: _UFunc11fc[L["erfi"], L[0]] = ...
exp1: _UFunc11fc[L["exp1"]] = ...
expi: _UFunc11fc[L["expi"]] = ...
expm1: _UFunc11fc[L["expm1"], L[0]] = ...
gamma: _UFunc11fc[L["gamma"]] = ...
log1p: _UFunc11fc[L["log1p"], L[0]] = ...
log_ndtr: _UFunc11fc[L["log_ndtr"], L[0]] = ...
loggamma: _UFunc11fc[L["loggamma"]] = ...
ndtr: _UFunc11fc[L["ndtr"], L[0]] = ...
psi: _UFunc11fc[L["psi"]] = ...
rgamma: _UFunc11fc[L["rgamma"]] = ...
spence: _UFunc11fc[L["spence"], L[0]] = ...
wrightomega: _UFunc11fc[L["wrightomega"], L[0]] = ...

# ld->d
eval_hermite: _UFunc21ld[L["eval_hermite"], L[0]] = ...
eval_hermitenorm: _UFunc21ld[L["eval_hermitenorm"], L[0]] = ...

# ff->f; (l|d)d->d
_igam_fac: _UFunc21f[L["_igam_fac"], L[0]] = ...
_iv_ratio: _UFunc21f[L["_iv_ratio"], L[0]] = ...
_iv_ratio_c: _UFunc21f[L["_iv_ratio_c"], L[0]] = ...
_nbinom_mean: _UFunc21f[L["_nbinom_mean"], L[0]] = ...
_nbinom_variance: _UFunc21f[L["_nbinom_variance"], L[0]] = ...
_nbinom_skewness: _UFunc21f[L["_nbinom_skewness"], L[0]] = ...
_nbinom_kurtosis_excess: _UFunc21f[L["_nbinom_kurtosis_excess"], L[0]] = ...
_nct_mean: _UFunc21f[L["_nct_mean"], L[0]] = ...
_nct_variance: _UFunc21f[L["_nct_variance"], L[0]] = ...
_nct_skewness: _UFunc21f[L["_nct_skewness"], L[0]] = ...
_nct_kurtosis_excess: _UFunc21f[L["_nct_kurtosis_excess"], L[0]] = ...
_smirnovc: _UFunc21f[L["_smirnovc"], L[0]] = ...
_smirnovci: _UFunc21f[L["_smirnovci"], L[0]] = ...
_smirnovp: _UFunc21f[L["_smirnovp"], L[0]] = ...
_stirling2_inexact: _UFunc21f[L["_stirling2_inexact"]] = ...
agm: _UFunc21f[L["agm"], L[0]] = ...
beta: _UFunc21f[L["beta"], L[0]] = ...
betaln: _UFunc21f[L["betaln"], L[0]] = ...
binom: _UFunc21f[L["binom"]] = ...
boxcox: _UFunc21f[L["boxcox"], L[0]] = ...
boxcox1p: _UFunc21f[L["boxcox1p"], L[0]] = ...
chdtr: _UFunc21f[L["chdtr"], L[0]] = ...
chdtrc: _UFunc21f[L["chdtrc"], L[0]] = ...
chdtri: _UFunc21f[L["chdtri"], L[0]] = ...
chdtriv: _UFunc21f[L["chdtriv"], L[0]] = ...
ellipeinc: _UFunc21f[L["ellipeinc"], L[0]] = ...
ellipkinc: _UFunc21f[L["ellipkinc"], L[0]] = ...
expn: _UFunc21f[L["expn"], L[0]] = ...
gammainc: _UFunc21f[L["gammainc"], L[0]] = ...
gammaincc: _UFunc21f[L["gammaincc"], L[0]] = ...
gammainccinv: _UFunc21f[L["gammainccinv"], L[0]] = ...
gammaincinv: _UFunc21f[L["gammaincinv"], L[0]] = ...
huber: _UFunc21f[L["huber"], L[0]] = ...
inv_boxcox1p: _UFunc21f[L["inv_boxcox1p"], L[0]] = ...
inv_boxcox: _UFunc21f[L["inv_boxcox"], L[0]] = ...
kl_div: _UFunc21f[L["kl_div"], L[0]] = ...
kn: _UFunc21f[L["kn"], L[0]] = ...
mathieu_a: _UFunc21f[L["mathieu_a"]] = ...
mathieu_b: _UFunc21f[L["mathieu_b"]] = ...
modstruve: _UFunc21f[L["modstruve"], L[0]] = ...
owens_t: _UFunc21f[L["owens_t"], L[0]] = ...
pdtr: _UFunc21f[L["pdtr"], L[0]] = ...
pdtrc: _UFunc21f[L["pdtrc"], L[0]] = ...
pdtri: _UFunc21f[L["pdtri"], L[0]] = ...
pdtrik: _UFunc21f[L["pdtrik"], L[0]] = ...
poch: _UFunc21f[L["poch"], L[0]] = ...
powm1: _UFunc21f[L["powm1"], L[0]] = ...
pseudo_huber: _UFunc21f[L["pseudo_huber"], L[0]] = ...
rel_entr: _UFunc21f[L["rel_entr"], L[0]] = ...
smirnov: _UFunc21f[L["smirnov"], L[0]] = ...
smirnovi: _UFunc21f[L["smirnovi"], L[0]] = ...
stdtr: _UFunc21f[L["stdtr"], L[0]] = ...
stdtridf: _UFunc21f[L["stdtridf"], L[0]] = ...
stdtrit: _UFunc21f[L["stdtrit"], L[0]] = ...
struve: _UFunc21f[L["struve"], L[0]] = ...
tklmbda: _UFunc21f[L["tklmbda"], L[0]] = ...
yn: _UFunc21f[L["yn"], L[0]] = ...

# fF->F; dD->D
hankel1: _UFunc21c1[L["hankel1"]] = ...
hankel1e: _UFunc21c1[L["hankel1e"]] = ...
hankel2: _UFunc21c1[L["hankel2"]] = ...
hankel2e: _UFunc21c1[L["hankel2e"]] = ...

# ff->f; (l|d)d->d; fF->F; dD->D
eval_chebyc: _UFunc21fc1[L["eval_chebyc"], L[0]] = ...
eval_chebys: _UFunc21fc1[L["eval_chebys"], L[0]] = ...
eval_chebyt: _UFunc21fc1[L["eval_chebyt"], L[0]] = ...
eval_chebyu: _UFunc21fc1[L["eval_chebyu"], L[0]] = ...
eval_laguerre: _UFunc21fc1[L["eval_laguerre"], L[0]] = ...
eval_legendre: _UFunc21fc1[L["eval_legendre"], L[0]] = ...
eval_sh_chebyt: _UFunc21fc1[L["eval_sh_chebyt"], L[0]] = ...
eval_sh_chebyu: _UFunc21fc1[L["eval_sh_chebyu"], L[0]] = ...
eval_sh_legendre: _UFunc21fc1[L["eval_sh_legendre"], L[0]] = ...
hyp0f1: _UFunc21fc1[L["hyp0f1"], L[0]] = ...
iv: _UFunc21fc1[L["iv"]] = ...
ive: _UFunc21fc1[L["ive"]] = ...
jn: _UFunc21fc1[L["jn"]] = ...
jv: _UFunc21fc1[L["jv"]] = ...
jve: _UFunc21fc1[L["jve"]] = ...
kv: _UFunc21fc1[L["kv"]] = ...
kve: _UFunc21fc1[L["kve"]] = ...
yv: _UFunc21fc1[L["yv"]] = ...
yve: _UFunc21fc1[L["yve"]] = ...

# lf->f; ld->d; lF->F; lD->D
_spherical_in: np.ufunc = ...
_spherical_in_d: np.ufunc = ...
_spherical_jn: np.ufunc = ...
_spherical_jn_d: np.ufunc = ...
_spherical_kn: np.ufunc = ...
_spherical_kn_d: np.ufunc = ...
_spherical_yn: np.ufunc = ...
_spherical_yn_d: np.ufunc = ...

# ff->f; dd->d; FF->F; DD->D
elliprc: _UFunc21fc2[L["elliprc"], L[0]] = ...
xlog1py: _UFunc21fc2[L["xlog1py"], L[0]] = ...
xlogy: _UFunc21fc2[L["xlogy"], L[0]] = ...

# ff->f; dd->d; Ff->F; Dd->D
_zeta: np.ufunc = ...

# Flf->F; Dld->D
_lambertw: np.ufunc = ...

# fff->f; ddd->d; ffF->F; (l|d)dD->D
# TODO
eval_gegenbauer: np.ufunc = ...
eval_genlaguerre: np.ufunc = ...
hyp1f1: np.ufunc = ...

# fff->f; ddd->d; FFF->F; DDD->D
# TODO
elliprd: np.ufunc = ...
elliprf: np.ufunc = ...
elliprg: np.ufunc = ...

# ffff->f; dddd->d
_hypergeom_pmf: _UFunc41f[L["_hypergeom_pmf"], L[0]] = ...
_hypergeom_cdf: _UFunc41f[L["_hypergeom_cdf"], L[0]] = ...
_hypergeom_sf: _UFunc41f[L["_hypergeom_sf"], L[0]] = ...
_ncf_pdf: _UFunc41f[L["_ncf_pdf"], L[0]] = ...
_ncf_cdf: _UFunc41f[L["_ncf_cdf"], L[0]] = ...
_ncf_ppf: _UFunc41f[L["_ncf_ppf"], L[0]] = ...
_ncf_sf: _UFunc41f[L["_ncf_sf"], L[0]] = ...
_ncf_isf: _UFunc41f[L["_ncf_isf"], L[0]] = ...
_skewnorm_cdf: _UFunc41f[L["_skewnorm_cdf"], L[0]] = ...
_skewnorm_ppf: _UFunc41f[L["_skewnorm_ppf"], L[0]] = ...
_skewnorm_isf: _UFunc41f[L["_skewnorm_isf"], L[0]] = ...
ncfdtr: _UFunc41f[L["ncfdtr"], L[0]] = ...
ncfdtri: _UFunc41f[L["ncfdtri"], L[0]] = ...
ncfdtridfd: _UFunc41f[L["ncfdtridfd"], L[0]] = ...
ncfdtridfn: _UFunc41f[L["ncfdtridfn"], L[0]] = ...
ncfdtrinc: _UFunc41f[L["ncfdtrinc"], L[0]] = ...

# (qq|ff)ff->F; (qq|dd)dd->D
# NOTE: Deprecated in SciPy 1.15.0
sph_harm: _UFuncSphHarm = ...

# ffff->f; (l|d)ddd->d; fffF->F; dddD->D
eval_jacobi: _UFunc41fc1[L["eval_jacobi"], L[0]] = ...
eval_sh_jacobi: _UFunc41fc1[L["eval_sh_jacobi"], L[0]] = ...
hyp2f1: _UFunc41fc1[L["hyp2f1"]] = ...

# ffff->f; dddd->d; FFFF->F; DDDD->D
# TODO
elliprj: np.ufunc = ...

# fffffff->f; dd(ll|dd)ddd->d
_ellip_harm: np.ufunc = ...

# fff->f; (ll|dl|dd)d->d
_beta_pdf: _UFunc31f[L["_beta_pdf"], L[0]] = ...
_beta_ppf: _UFunc31f[L["_beta_ppf"], L[0]] = ...
_binom_pmf: _UFunc31f[L["_binom_pmf"], L[0]] = ...
_binom_cdf: _UFunc31f[L["_binom_cdf"], L[0]] = ...
_binom_ppf: _UFunc31f[L["_binom_ppf"], L[0]] = ...
_binom_sf: _UFunc31f[L["_binom_sf"], L[0]] = ...
_binom_isf: _UFunc31f[L["_binom_isf"], L[0]] = ...
_cauchy_ppf: _UFunc31f[L["_cauchy_ppf"], L[0]] = ...
_cauchy_isf: _UFunc31f[L["_cauchy_isf"], L[0]] = ...
_hypergeom_mean: _UFunc31f[L["_hypergeom_mean"], L[0]] = ...
_hypergeom_variance: _UFunc31f[L["_hypergeom_variance"], L[0]] = ...
_hypergeom_skewness: _UFunc31f[L["_hypergeom_skewness"], L[0]] = ...
_invgauss_ppf: _UFunc31f[L["_invgauss_ppf"], L[0]] = ...
_invgauss_isf: _UFunc31f[L["_invgauss_isf"], L[0]] = ...
_landau_pdf: _UFunc31f[L["_landau_pdf"], L[0]] = ...
_landau_cdf: _UFunc31f[L["_landau_cdf"], L[0]] = ...
_landau_ppf: _UFunc31f[L["_landau_ppf"], L[0]] = ...
_landau_sf: _UFunc31f[L["_landau_sf"], L[0]] = ...
_landau_isf: _UFunc31f[L["_landau_isf"], L[0]] = ...
_nbinom_cdf: _UFunc31f[L["_nbinom_cdf"], L[0]] = ...
_nbinom_isf: _UFunc31f[L["_nbinom_isf"], L[0]] = ...
_nbinom_pmf: _UFunc31f[L["_nbinom_pmf"], L[0]] = ...
_nbinom_ppf: _UFunc31f[L["_nbinom_ppf"], L[0]] = ...
_nbinom_sf: _UFunc31f[L["_nbinom_sf"], L[0]] = ...
_ncf_mean: _UFunc31f[L["_ncf_mean"], L[0]] = ...
_ncf_variance: _UFunc31f[L["_ncf_variance"], L[0]] = ...
_ncf_skewness: _UFunc31f[L["_ncf_skewness"], L[0]] = ...
_ncf_kurtosis_excess: _UFunc31f[L["_ncf_kurtosis_excess"], L[0]] = ...
_nct_cdf: _UFunc31f[L["_nct_cdf"], L[0]] = ...
_nct_ppf: _UFunc31f[L["_nct_ppf"], L[0]] = ...
_nct_sf: _UFunc31f[L["_nct_sf"], L[0]] = ...
_nct_isf: _UFunc31f[L["_nct_isf"], L[0]] = ...
_ncx2_pdf: _UFunc31f[L["_ncx2_pdf"], L[0]] = ...
_ncx2_cdf: _UFunc31f[L["_ncx2_cdf"], L[0]] = ...
_ncx2_ppf: _UFunc31f[L["_ncx2_ppf"], L[0]] = ...
_ncx2_sf: _UFunc31f[L["_ncx2_sf"], L[0]] = ...
_ncx2_isf: _UFunc31f[L["_ncx2_isf"], L[0]] = ...
bdtr: _UFunc31f[L["bdtr"], L[0]] = ...
bdtrc: _UFunc31f[L["bdtrc"], L[0]] = ...
bdtri: _UFunc31f[L["bdtri"], L[0]] = ...
bdtrik: _UFunc31f[L["bdtrik"], L[0]] = ...
bdtrin: _UFunc31f[L["bdtrin"], L[0]] = ...
besselpoly: _UFunc31f[L["besselpoly"], L[0]] = ...
betainc: _UFunc31f[L["betainc"], L[0]] = ...
betaincc: _UFunc31f[L["betaincc"], L[0]] = ...
betainccinv: _UFunc31f[L["betainccinv"], L[0]] = ...
betaincinv: _UFunc31f[L["betaincinv"], L[0]] = ...
btdtria: _UFunc31f[L["btdtria"], L[0]] = ...
btdtrib: _UFunc31f[L["btdtrib"], L[0]] = ...
chndtr: _UFunc31f[L["chndtr"], L[0]] = ...
chndtridf: _UFunc31f[L["chndtridf"], L[0]] = ...
chndtrinc: _UFunc31f[L["chndtrinc"], L[0]] = ...
chndtrix: _UFunc31f[L["chndtrix"], L[0]] = ...
fdtr: _UFunc31f[L["fdtr"], L[0]] = ...
fdtrc: _UFunc31f[L["fdtrc"], L[0]] = ...
fdtri: _UFunc31f[L["fdtri"], L[0]] = ...
fdtridfd: _UFunc31f[L["fdtridfd"], L[0]] = ...
gdtr: _UFunc31f[L["gdtr"], L[0]] = ...
gdtrc: _UFunc31f[L["gdtrc"], L[0]] = ...
gdtria: _UFunc31f[L["gdtria"], L[0]] = ...
gdtrib: _UFunc31f[L["gdtrib"], L[0]] = ...
gdtrix: _UFunc31f[L["gdtrix"], L[0]] = ...
hyperu: _UFunc31f[L["hyperu"], L[0]] = ...
log_wright_bessel: _UFunc31f[L["log_wright_bessel"]] = ...
lpmv: _UFunc31f[L["lpmv"], L[0]] = ...
nbdtr: _UFunc31f[L["nbdtr"], L[0]] = ...
nbdtrc: _UFunc31f[L["nbdtrc"], L[0]] = ...
nbdtri: _UFunc31f[L["nbdtri"], L[0]] = ...
nbdtrik: _UFunc31f[L["nbdtrik"], L[0]] = ...
nbdtrin: _UFunc31f[L["nbdtrin"], L[0]] = ...
nctdtr: _UFunc31f[L["nctdtr"], L[0]] = ...
nctdtridf: _UFunc31f[L["nctdtridf"], L[0]] = ...
nctdtrinc: _UFunc31f[L["nctdtrinc"], L[0]] = ...
nctdtrit: _UFunc31f[L["nctdtrit"], L[0]] = ...
nrdtrimn: _UFunc31f[L["nrdtrimn"], L[0]] = ...
nrdtrisd: _UFunc31f[L["nrdtrisd"], L[0]] = ...
obl_cv: _UFunc31f[L["obl_cv"]] = ...
pro_cv: _UFunc31f[L["pro_cv"]] = ...
radian: _UFunc31f[L["radian"], L[0]] = ...
voigt_profile: _UFunc31f[L["voigt_profile"], L[0]] = ...
wright_bessel: _UFunc31f[L["wright_bessel"]] = ...

# f->ff; d->dd
iti0k0: _UFunc12f[L["iti0k0"]] = ...
it2i0k0: _UFunc12f[L["it2i0k0"]] = ...
itj0y0: _UFunc12f[L["itj0y0"]] = ...
it2j0y0: _UFunc12f[L["it2j0y0"]] = ...

# f->FF; d->DD
modfresnelm: _UFunc12c[L["modfresnelm"]] = ...
modfresnelp: _UFunc12c[L["modfresnelp"]] = ...

# f->ff; d->dd; f->FF; D->DD
fresnel: _UFunc12fc[L["fresnel"]] = ...
shichi: _UFunc12fc[L["shichi"], L[0]] = ...
sici: _UFunc12fc[L["sici"], L[0]] = ...

# ff->ff; dd->dd
# TODO
pbdv: np.ufunc = ...
pbvv: np.ufunc = ...
pbwa: np.ufunc = ...

# fff->ff; ddd->dd
# TODO
mathieu_cem: np.ufunc = ...
mathieu_modcem1: np.ufunc = ...
mathieu_modcem2: np.ufunc = ...
mathieu_sem: np.ufunc = ...
mathieu_modsem1: np.ufunc = ...
mathieu_modsem2: np.ufunc = ...

# ddl->dd
_struve_asymp_large_z: np.ufunc = ...
_struve_bessel_series: np.ufunc = ...
_struve_power_series: np.ufunc = ...

# ffff->ff; dddd->dd
# TODO
obl_ang1: np.ufunc = ...
obl_rad1: np.ufunc = ...
obl_rad2: np.ufunc = ...
pro_ang1: np.ufunc = ...
pro_rad1: np.ufunc = ...
pro_rad2: np.ufunc = ...

# fffff->ff; ddddd->dd
# TODO
obl_ang1_cv: np.ufunc = ...
obl_rad1_cv: np.ufunc = ...
obl_rad2_cv: np.ufunc = ...
pro_ang1_cv: np.ufunc = ...
pro_rad1_cv: np.ufunc = ...
pro_rad2_cv: np.ufunc = ...

# f->ffff; d->dddd
# TODO
itairy: np.ufunc = ...

# f->FFFF; d->DDDD
# TODO
kelvin: np.ufunc = ...

# f->ffff; d->dddd; F->FFFF; D->DDDD
# TODO
airy: np.ufunc = ...
airye: np.ufunc = ...

# ff->ffff; dd->dddd
# TODO
ellipj: np.ufunc = ...
