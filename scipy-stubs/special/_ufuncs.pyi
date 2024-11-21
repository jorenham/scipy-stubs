# mypy: disable-error-code="explicit-override, override"
# pyright: reportIncompatibleMethodOverride=false, reportIncompatibleVariableOverride=false

from typing import Any, Generic, Literal as L, TypeAlias, TypedDict, final, overload, type_check_only
from typing_extensions import LiteralString, TypeVar, Unpack

import numpy as np
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
_IdentityT_co = TypeVar("_IdentityT_co", bound=L[0] | None, covariant=True)
_OutT = TypeVar("_OutT", bound=onp.ArrayND[np.number[Any]])

_Tuple2: TypeAlias = tuple[_T, _T]
_Tuple3: TypeAlias = tuple[_T, _T, _T]

_Float: TypeAlias = np.float32 | np.float64
_ToSubFloat: TypeAlias = float | np.float16 | np.integer[Any] | np.bool_
_Inexact: TypeAlias = _Float | np.complex64 | np.complex128
_FloatNDT = TypeVar("_FloatNDT", bound=_Float | onp.ArrayND[_Float])
_InexactNDT = TypeVar("_InexactNDT", bound=_Inexact | onp.ArrayND[_Inexact])

_ToFloatDType: TypeAlias = onp.AnyFloat32DType | onp.AnyFloat64DType
_ToInexactDType: TypeAlias = _ToFloatDType | onp.AnyComplex64DType | onp.AnyComplex128DType

@type_check_only
class _KwBase(TypedDict, total=False):
    order: OrderKACF
    casting: Casting
    subok: onp.ToBool
    where: onp.ToBool | onp.ToBoolND

@type_check_only
class _UFuncBase(np.ufunc, Generic[_NameT_co, _IdentityT_co]):  # type: ignore[misc]  # pyright: ignore[reportGeneralTypeIssues]
    @property
    def __class__(self) -> type[np.ufunc]: ...
    @__class__.setter
    def __class__(self, t: type[np.ufunc], /) -> None: ...
    @property
    def __name__(self) -> _NameT_co: ...
    @property
    def identity(self) -> _IdentityT_co: ...
    @property
    def signature(self) -> None: ...

@type_check_only
class _Kw11f(_KwBase, TypedDict, total=False):
    dtype: _ToFloatDType | None
    signature: L["f->f", "d->d"] | _Tuple2[_ToFloatDType]

@type_check_only
@final
class _UFunc11f(_UFuncBase[_NameT_co, _IdentityT_co], Generic[_NameT_co, _IdentityT_co]):  # type: ignore[misc]
    @property
    def nin(self) -> L[1]: ...
    @property
    def nout(self) -> L[1]: ...
    @property
    def nargs(self) -> L[2]: ...
    @property
    def ntypes(self) -> L[2]: ...
    @property
    def types(self) -> list[L["f->f", "d->d"]]: ...

    #
    @overload
    def __call__(self, x: _ToSubFloat, /, out: None = None, **kw: Unpack[_Kw11f]) -> _Float: ...
    @overload
    def __call__(self, x: _FloatNDT, /, out: None = None, **kw: Unpack[_Kw11f]) -> _FloatNDT: ...
    @overload
    def __call__(self, x: onp.ToFloatND, /, out: None = None, **kw: Unpack[_Kw11f]) -> onp.ArrayND[_Float]: ...
    @overload
    def __call__(self, x: onp.ToFloat | onp.ToFloatND, /, out: _OutT, **kw: Unpack[_Kw11f]) -> _OutT: ...

    # TODO: at

@type_check_only
class _Kw11fc(_KwBase, TypedDict, total=False):
    dtype: _ToInexactDType | None
    signature: L["f->f", "d->d", "F->F", "D->D"] | _Tuple2[_ToInexactDType]

@type_check_only
@final
class _UFunc11fc(_UFuncBase[_NameT_co, _IdentityT_co], Generic[_NameT_co, _IdentityT_co]):  # type: ignore[misc]
    @property
    def nin(self) -> L[1]: ...
    @property
    def nout(self) -> L[1]: ...
    @property
    def nargs(self) -> L[2]: ...
    @property
    def ntypes(self) -> L[4]: ...
    @property
    def types(self) -> list[L["f->f", "d->d", "F->F", "D->D"]]: ...

    #
    @overload
    def __call__(self, x: _ToSubFloat, /, out: None = None, **kw: Unpack[_Kw11fc]) -> _Float: ...
    @overload
    def __call__(self, x: complex | _ToSubFloat, /, out: None = None, **kw: Unpack[_Kw11fc]) -> _Inexact: ...
    @overload
    def __call__(self, x: _InexactNDT, /, out: None = None, **kw: Unpack[_Kw11fc]) -> _InexactNDT: ...
    @overload
    def __call__(self, x: onp.ToFloatND, /, out: None = None, **kw: Unpack[_Kw11fc]) -> onp.ArrayND[_Float]: ...
    @overload
    def __call__(self, x: onp.ToComplexND, /, out: None = None, **kw: Unpack[_Kw11fc]) -> onp.ArrayND[_Inexact]: ...
    @overload
    def __call__(self, x: onp.ToComplex | onp.ToComplexND, /, out: _OutT, **kw: Unpack[_Kw11fc]) -> _OutT: ...

    # TODO: at

@type_check_only
class _Kw21f(_KwBase, TypedDict, total=False):
    dtype: _ToFloatDType | None
    signature: L["ff->f", "dd->d"] | _Tuple3[_ToFloatDType]

@type_check_only
@final
class _UFunc21f(_UFuncBase[_NameT_co, _IdentityT_co], Generic[_NameT_co, _IdentityT_co]):  # type: ignore[misc]
    @property
    def nin(self) -> L[2]: ...
    @property
    def nout(self) -> L[1]: ...
    @property
    def nargs(self) -> L[3]: ...
    @property
    def ntypes(self) -> L[2]: ...
    @property
    def types(self) -> list[L["ff->f", "dd->d"]]: ...

    #
    @overload
    def __call__(self, a: _ToSubFloat, b: _ToSubFloat, /, out: None = None, **kwargs: Unpack[_Kw21f]) -> _Float: ...
    @overload
    def __call__(self, a: _FloatNDT, b: _FloatNDT, /, out: None = None, **kwargs: Unpack[_Kw21f]) -> _FloatNDT: ...
    @overload
    def __call__(
        self,
        a: onp.ToFloat,
        b: onp.ToFloatND,
        /,
        out: None = None,
        **kwargs: Unpack[_Kw21f],
    ) -> onp.ArrayND[_Float]: ...
    @overload
    def __call__(
        self,
        a: onp.ToFloatND,
        b: onp.ToFloat | onp.ToFloatND,
        /,
        out: None = None,
        **kwargs: Unpack[_Kw21f],
    ) -> onp.ArrayND[_Float]: ...
    @overload
    def __call__(
        self,
        a: onp.ToFloat | onp.ToFloatND,
        b: onp.ToFloat | onp.ToFloatND,
        /,
        out: _OutT,
        **kwargs: Unpack[_Kw21f],
    ) -> _OutT: ...

    #
    @overload
    def reduce(
        self,
        /,
        array: onp.ToFloatND,
        axis: None,
        dtype: onp.AnyFloat32DType | onp.AnyFloat64DType | None = None,
        out: None = None,
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
        out: None = None,
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
        out: None = None,
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
        out: _OutT,
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
        out: _OutT,
        keepdims: bool = False,
        initial: onp.ToFloat = ...,
        where: onp.ToBool | onp.ToBoolND = True,
    ) -> _OutT: ...

    # TODO: at
    # TODO: accumulate
    # TODO: reduceat
    # TODO: outer

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
_scaled_exp1: _UFunc11f[L["_scaled_exp1"], None]
bei: _UFunc11f[L["bei"], None]
beip: _UFunc11f[L["beip"], None]
ber: _UFunc11f[L["ber"], None]
berp: _UFunc11f[L["berp"], None]
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
exprel: _UFunc11f[L["exprel"], None]
gammaln: _UFunc11f[L["gammaln"], None]
gammasgn: _UFunc11f[L["gammasgn"], L[0]]
i0: _UFunc11f[L["i0"], L[0]]
i0e: _UFunc11f[L["i0e"], L[0]]
i1: _UFunc11f[L["i1"], L[0]]
i1e: _UFunc11f[L["i1e"], L[0]]
it2struve0: _UFunc11f[L["it2struve0"], None]
itmodstruve0: _UFunc11f[L["itmodstruve0"], None]
itstruve0: _UFunc11f[L["itstruve0"], None]
j0: _UFunc11f[L["j0"], L[0]]
j1: _UFunc11f[L["j1"], L[0]]
k0: _UFunc11f[L["k0"], L[0]]
k0e: _UFunc11f[L["k0e"], L[0]]
k1: _UFunc11f[L["k1"], L[0]]
k1e: _UFunc11f[L["k1e"], L[0]]
kei: _UFunc11f[L["kei"], None]
keip: _UFunc11f[L["keip"], None]
ker: _UFunc11f[L["ker"], None]
kerp: _UFunc11f[L["kerp"], None]
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
# TODO (identity=None)
expit: np.ufunc
log_expit: np.ufunc
logit: np.ufunc

# F->F; D->D
# TODO (identity=0)
wofz: np.ufunc

# f->f; d->d; F->F; D->D
_cospi: _UFunc11fc[L["_cospi"], None]
_sinpi: _UFunc11fc[L["_sinpi"], None]
dawsn: _UFunc11fc[L["dawsn"], L[0]]
erf: _UFunc11fc[L["erf"], L[0]]
erfc: _UFunc11fc[L["erfc"], L[0]]
erfcx: _UFunc11fc[L["erfcx"], L[0]]
erfi: _UFunc11fc[L["erfi"], L[0]]
exp1: _UFunc11fc[L["exp1"], None]
expi: _UFunc11fc[L["expi"], None]
expm1: _UFunc11fc[L["expm1"], L[0]]
gamma: _UFunc11fc[L["gamma"], None]
log1p: _UFunc11fc[L["log1p"], L[0]]
log_ndtr: _UFunc11fc[L["log_ndtr"], L[0]]
loggamma: _UFunc11fc[L["loggamma"], None]
ndtr: _UFunc11fc[L["ndtr"], L[0]]
psi: _UFunc11fc[L["psi"], None]
rgamma: _UFunc11fc[L["rgamma"], None]
spence: _UFunc11fc[L["spence"], L[0]]
wrightomega: _UFunc11fc[L["wrightomega"], L[0]]

# ff->f; dd->d
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
_stirling2_inexact: _UFunc21f[L["_stirling2_inexact"], None]
_zeta: _UFunc21f[L["_zeta"], L[0]]
agm: _UFunc21f[L["agm"], L[0]]
beta: _UFunc21f[L["beta"], L[0]]
betaln: _UFunc21f[L["betaln"], L[0]]
binom: _UFunc21f[L["binom"], None]
boxcox: _UFunc21f[L["boxcox"], L[0]]
boxcox1p: _UFunc21f[L["boxcox1p"], L[0]]
chdtr: _UFunc21f[L["chdtr"], L[0]]
chdtrc: _UFunc21f[L["chdtrc"], L[0]]
chdtri: _UFunc21f[L["chdtri"], L[0]]
chdtriv: _UFunc21f[L["chdtriv"], L[0]]
ellipeinc: _UFunc21f[L["ellipeinc"], L[0]]
ellipkinc: _UFunc21f[L["ellipkinc"], L[0]]
gammainc: _UFunc21f[L["gammainc"], L[0]]
gammaincc: _UFunc21f[L["gammaincc"], L[0]]
gammainccinv: _UFunc21f[L["gammainccinv"], L[0]]
gammaincinv: _UFunc21f[L["gammaincinv"], L[0]]
huber: _UFunc21f[L["huber"], L[0]]
inv_boxcox1p: _UFunc21f[L["inv_boxcox1p"], L[0]]
inv_boxcox: _UFunc21f[L["inv_boxcox"], L[0]]
kl_div: _UFunc21f[L["kl_div"], L[0]]
mathieu_a: _UFunc21f[L["mathieu_a"], None]
mathieu_b: _UFunc21f[L["mathieu_b"], None]
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
stdtr: _UFunc21f[L["stdtr"], L[0]]
stdtridf: _UFunc21f[L["stdtridf"], L[0]]
stdtrit: _UFunc21f[L["stdtrit"], L[0]]
struve: _UFunc21f[L["struve"], L[0]]
tklmbda: _UFunc21f[L["tklmbda"], L[0]]

# ld->d
# TODO (identity=0)
eval_hermite: np.ufunc
eval_hermitenorm: np.ufunc

# ld->d; ff->f; fF->F; dd->d; dD->D
# TODO (identity=0)
eval_chebyc: np.ufunc
eval_chebys: np.ufunc
eval_chebyt: np.ufunc
eval_chebyu: np.ufunc
eval_laguerre: np.ufunc
eval_legendre: np.ufunc
eval_sh_chebyt: np.ufunc
eval_sh_chebyu: np.ufunc
eval_sh_legendre: np.ufunc

# fff->f; ddd->d
# TODO
_beta_pdf: np.ufunc
_beta_ppf: np.ufunc
_binom_cdf: np.ufunc
_binom_isf: np.ufunc
_binom_pmf: np.ufunc
_binom_ppf: np.ufunc
_binom_sf: np.ufunc
_hypergeom_mean: np.ufunc
_hypergeom_skewness: np.ufunc
_hypergeom_variance: np.ufunc
_invgauss_isf: np.ufunc
_invgauss_ppf: np.ufunc
_nbinom_cdf: np.ufunc
_nbinom_isf: np.ufunc
_nbinom_pmf: np.ufunc
_nbinom_ppf: np.ufunc
_nbinom_sf: np.ufunc
_ncf_kurtosis_excess: np.ufunc
_ncf_mean: np.ufunc
_ncf_skewness: np.ufunc
_ncf_variance: np.ufunc
_nct_cdf: np.ufunc
_nct_isf: np.ufunc
_nct_ppf: np.ufunc
_nct_sf: np.ufunc
_ncx2_cdf: np.ufunc
_ncx2_isf: np.ufunc
_ncx2_pdf: np.ufunc
_ncx2_ppf: np.ufunc
_ncx2_sf: np.ufunc
bdtrik: np.ufunc
bdtrin: np.ufunc
besselpoly: np.ufunc
betainc: np.ufunc
betaincc: np.ufunc
betainccinv: np.ufunc
betaincinv: np.ufunc
btdtr: np.ufunc
btdtri: np.ufunc
btdtria: np.ufunc
btdtrib: np.ufunc
chndtr: np.ufunc
chndtridf: np.ufunc
chndtrinc: np.ufunc
chndtrix: np.ufunc
fdtr: np.ufunc
fdtrc: np.ufunc
fdtri: np.ufunc
fdtridfd: np.ufunc
gdtr: np.ufunc
gdtrc: np.ufunc
gdtria: np.ufunc
gdtrib: np.ufunc
gdtrix: np.ufunc
hyperu: np.ufunc
log_wright_bessel: np.ufunc
lpmv: np.ufunc
nbdtrik: np.ufunc
nbdtrin: np.ufunc
nctdtr: np.ufunc
nctdtridf: np.ufunc
nctdtrinc: np.ufunc
nctdtrit: np.ufunc
nrdtrimn: np.ufunc
nrdtrisd: np.ufunc
obl_cv: np.ufunc
pro_cv: np.ufunc
radian: np.ufunc
voigt_profile: np.ufunc
wright_bessel: np.ufunc

# ffff->f; dddd->d
# TODO
_hypergeom_cdf: np.ufunc
_hypergeom_pmf: np.ufunc
_hypergeom_sf: np.ufunc
_ncf_cdf: np.ufunc
_ncf_isf: np.ufunc
_ncf_pdf: np.ufunc
_ncf_ppf: np.ufunc
_ncf_sf: np.ufunc
_skewnorm_cdf: np.ufunc
_skewnorm_isf: np.ufunc
_skewnorm_ppf: np.ufunc
ncfdtr: np.ufunc
ncfdtri: np.ufunc
ncfdtridfd: np.ufunc
ncfdtridfn: np.ufunc
ncfdtrinc: np.ufunc

# TODO
_ellip_harm: np.ufunc
_lambertw: np.ufunc
_smirnovc: np.ufunc
_smirnovci: np.ufunc
_smirnovp: np.ufunc
_spherical_n: np.ufunc
_spherical_in_d: np.ufunc
_spherical_jn: np.ufunc
_spherical_jn_d: np.ufunc
_spherical_kn: np.ufunc
_spherical_kn_d: np.ufunc
_spherical_yn: np.ufunc
_spherical_yn_d: np.ufunc
_struve_asymp_large_z: np.ufunc
_struve_bessel_series: np.ufunc
_struve_power_series: np.ufunc
airy: np.ufunc
airye: np.ufunc
bdtr: np.ufunc
bdtrc: np.ufunc
bdtri: np.ufunc
ellipj: np.ufunc
elliprc: np.ufunc
elliprd: np.ufunc
elliprf: np.ufunc
elliprg: np.ufunc
elliprj: np.ufunc
eval_gegenbauer: np.ufunc
eval_genlaguerre: np.ufunc
eval_jacobi: np.ufunc
eval_sh_jacobi: np.ufunc
expn: np.ufunc
fresnel: np.ufunc
hankel1: np.ufunc
hankel1e: np.ufunc
hankel2: np.ufunc
hankel2e: np.ufunc
hyp0f1: np.ufunc
hyp1f1: np.ufunc
hyp2f1: np.ufunc
it2i0k0: np.ufunc
it2j0y0: np.ufunc
itairy: np.ufunc
iti0k0: np.ufunc
itj0y0: np.ufunc
iv: np.ufunc
ive: np.ufunc
jn: np.ufunc
jv: np.ufunc
jve: np.ufunc
kelvin: np.ufunc
kn: np.ufunc
kv: np.ufunc
kve: np.ufunc
mathieu_cem: np.ufunc
mathieu_modcem1: np.ufunc
mathieu_modcem2: np.ufunc
mathieu_modsem1: np.ufunc
mathieu_modsem2: np.ufunc
mathieu_sem: np.ufunc
modfresnelm: np.ufunc
modfresnelp: np.ufunc
nbdtr: np.ufunc
nbdtrc: np.ufunc
nbdtri: np.ufunc
obl_ang1: np.ufunc
obl_ang1_cv: np.ufunc
obl_rad1: np.ufunc
obl_rad1_cv: np.ufunc
obl_rad2: np.ufunc
obl_rad2_cv: np.ufunc
pbdv: np.ufunc
pbvv: np.ufunc
pbwa: np.ufunc
pro_ang1: np.ufunc
pro_ang1_cv: np.ufunc
pro_rad1: np.ufunc
pro_rad1_cv: np.ufunc
pro_rad2: np.ufunc
pro_rad2_cv: np.ufunc
shichi: np.ufunc
sici: np.ufunc
smirnov: np.ufunc
smirnovi: np.ufunc
sph_harm: np.ufunc
xlog1py: np.ufunc
xlogy: np.ufunc
yn: np.ufunc
yv: np.ufunc
yve: np.ufunc
