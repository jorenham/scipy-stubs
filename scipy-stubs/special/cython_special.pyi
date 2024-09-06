from types import CodeType
from typing import Final, Literal, Protocol, TypeAlias, TypedDict, overload, type_check_only
from typing_extensions import CapsuleType, LiteralString, Never

import numpy as np

_X_b: TypeAlias = Literal[False, 0, True, 1] | np.bool_
_X_i: TypeAlias = int | np.intp
_X_f: TypeAlias = float | np.float64
_X_c: TypeAlias = complex | np.complex128
_X_if: TypeAlias = int | float | np.intp | np.float64
_X_fc: TypeAlias = float | complex | np.float64 | np.complex128

@type_check_only
class _BaseCythonFunctionOrMethod(Protocol):
    __name__: LiteralString
    __qualname__: str  # cannot be `LiteralString` (blame typeshed)
    __module__: str  # cannot be `Literal["scipy.special.cython_special"]` (blame typeshed)

    __annotations__: dict[str, Never]  # and that's why were here :)
    __defaults__: tuple[()] | tuple[Literal[0]] | None
    __kwdefaults__: None  # kw-only params aren't used

    __closure__: None
    __code__: CodeType

    _is_coroutine: Literal[False]
    func_defaults: tuple[()] | tuple[Literal[0]] | None
    # like `r'See the documentation for scipy.special.{self.__name__}'`
    func_doc: str | None

    # NOTE: __call__ should be defined in the subtype

@type_check_only
class _CythonFunctionOrMethod_1f(_BaseCythonFunctionOrMethod, Protocol):
    def __call__(self, /, x0: _X_f) -> float: ...

@type_check_only
class _CythonFunctionOrMethod_1c(_BaseCythonFunctionOrMethod, Protocol):
    def __call__(self, /, x0: _X_c) -> complex: ...

@type_check_only
class _CythonFunctionOrMethod_1fc(_BaseCythonFunctionOrMethod, Protocol):
    @overload
    def __call__(self, /, x0: _X_f) -> float: ...
    @overload
    def __call__(self, /, x0: _X_c) -> complex: ...

@type_check_only
class _CythonFunctionOrMethod_2f(_BaseCythonFunctionOrMethod, Protocol):
    def __call__(self, /, x0: _X_f, x1: _X_f) -> float: ...

@type_check_only
class _CythonFunctionOrMethod_2fc(_BaseCythonFunctionOrMethod, Protocol):
    def __call__(self, /, x0: _X_fc, x1: _X_fc) -> float | complex: ...

@type_check_only
class _CythonFunctionOrMethod_2_poly(_BaseCythonFunctionOrMethod, Protocol):
    @overload
    def __call__(self, /, x0: _X_if, x1: _X_f) -> float: ...
    @overload
    def __call__(self, /, x0: _X_f, x1: _X_c) -> complex: ...

@type_check_only
class _CythonFunctionOrMethod_2_hankel(_BaseCythonFunctionOrMethod, Protocol):
    def __call__(self, /, x0: _X_f, x1: _X_c) -> complex: ...

@type_check_only
class _CythonFunctionOrMethod_2_spherical(_BaseCythonFunctionOrMethod, Protocol):
    @overload
    def __call__(self, /, n: _X_i, z: _X_f, derivative: _X_b = 0) -> float: ...
    @overload
    def __call__(self, /, n: _X_i, z: _X_c, derivative: _X_b = 0) -> complex: ...

@type_check_only
class _CythonFunctionOrMethod_3f(_BaseCythonFunctionOrMethod, Protocol):
    def __call__(self, /, x0: _X_f, x1: _X_f, x2: _X_f) -> float: ...

@type_check_only
class _CythonFunctionOrMethod_3fc(_BaseCythonFunctionOrMethod, Protocol):
    def __call__(self, /, x0: _X_fc, x1: _X_fc, x2: _X_fc) -> float | complex: ...

@type_check_only
class _CythonFunctionOrMethod_3_poly(_BaseCythonFunctionOrMethod, Protocol):
    @overload
    def __call__(self, /, x0: _X_if, x1: _X_f, x2: _X_f) -> float: ...
    @overload
    def __call__(self, /, x0: _X_f, x1: _X_f, x2: _X_c) -> complex: ...

@type_check_only
class _CythonFunctionOrMethod_4f(_BaseCythonFunctionOrMethod, Protocol):
    def __call__(self, /, x0: _X_f, x1: _X_f, x2: _X_f, x3: _X_f) -> float: ...

@type_check_only
class _CythonFunctionOrMethod_4fc(_BaseCythonFunctionOrMethod, Protocol):
    def __call__(self, /, x0: _X_fc, x1: _X_fc, x2: _X_fc, x3: _X_fc) -> float | complex: ...

@type_check_only
class _CythonFunctionOrMethod_4_poly(_BaseCythonFunctionOrMethod, Protocol):
    @overload
    def __call__(self, /, x0: _X_if, x1: _X_f, x2: _X_f, x3: _X_f) -> float: ...
    @overload
    def __call__(self, /, x0: _X_f, x1: _X_f, x2: _X_c, x3: _X_fc) -> complex: ...

# TODO: Use `ReadOnly[CapsuleType]` once mypy supports PEP 705: https://github.com/python/mypy/pull/17644
@type_check_only
class _CApiDict(TypedDict):
    agm: CapsuleType
    bdtrik: CapsuleType
    bdtrin: CapsuleType
    bei: CapsuleType
    beip: CapsuleType
    ber: CapsuleType
    berp: CapsuleType
    besselpoly: CapsuleType
    beta: CapsuleType
    betaln: CapsuleType
    binom: CapsuleType
    boxcox: CapsuleType
    boxcox1p: CapsuleType
    btdtr: CapsuleType
    btdtri: CapsuleType
    btdtria: CapsuleType
    btdtrib: CapsuleType
    cbrt: CapsuleType
    chdtr: CapsuleType
    chdtrc: CapsuleType
    chdtri: CapsuleType
    chdtriv: CapsuleType
    chndtr: CapsuleType
    chndtridf: CapsuleType
    chndtrinc: CapsuleType
    chndtrix: CapsuleType
    cosdg: CapsuleType
    cosm1: CapsuleType
    cotdg: CapsuleType
    ellipe: CapsuleType
    ellipeinc: CapsuleType
    ellipj: CapsuleType
    ellipkinc: CapsuleType
    ellipkm1: CapsuleType
    ellipk: CapsuleType
    entr: CapsuleType
    erfcinv: CapsuleType
    eval_hermite: CapsuleType
    eval_hermitenorm: CapsuleType
    exp10: CapsuleType
    exp2: CapsuleType
    exprel: CapsuleType
    fdtr: CapsuleType
    fdtrc: CapsuleType
    fdtri: CapsuleType
    fdtridfd: CapsuleType
    gammainc: CapsuleType
    gammaincc: CapsuleType
    gammainccinv: CapsuleType
    gammaincinv: CapsuleType
    gammaln: CapsuleType
    gammasgn: CapsuleType
    gdtr: CapsuleType
    gdtrc: CapsuleType
    gdtria: CapsuleType
    gdtrib: CapsuleType
    gdtrix: CapsuleType
    hankel1: CapsuleType
    hankel1e: CapsuleType
    hankel2: CapsuleType
    hankel2e: CapsuleType
    huber: CapsuleType
    hyperu: CapsuleType
    i0: CapsuleType
    i0e: CapsuleType
    i1: CapsuleType
    i1e: CapsuleType
    inv_boxcox: CapsuleType
    inv_boxcox1p: CapsuleType
    it2i0k0: CapsuleType
    it2j0y0: CapsuleType
    it2struve0: CapsuleType
    itairy: CapsuleType
    iti0k0: CapsuleType
    itj0y0: CapsuleType
    itmodstruve0: CapsuleType
    itstruve0: CapsuleType
    j0: CapsuleType
    j1: CapsuleType
    k0: CapsuleType
    k0e: CapsuleType
    k1: CapsuleType
    k1e: CapsuleType
    kei: CapsuleType
    keip: CapsuleType
    kelvin: CapsuleType
    ker: CapsuleType
    kerp: CapsuleType
    kl_div: CapsuleType
    kolmogi: CapsuleType
    kolmogorov: CapsuleType
    lpmv: CapsuleType
    mathieu_a: CapsuleType
    mathieu_b: CapsuleType
    mathieu_cem: CapsuleType
    mathieu_modcem1: CapsuleType
    mathieu_modcem2: CapsuleType
    mathieu_modsem1: CapsuleType
    mathieu_modsem2: CapsuleType
    mathieu_sem: CapsuleType
    modfresnelm: CapsuleType
    modfresnelp: CapsuleType
    modstruve: CapsuleType
    nbdtrik: CapsuleType
    nbdtrin: CapsuleType
    ncfdtr: CapsuleType
    ncfdtri: CapsuleType
    ncfdtridfd: CapsuleType
    ncfdtridfn: CapsuleType
    ncfdtrinc: CapsuleType
    nctdtr: CapsuleType
    nctdtridf: CapsuleType
    nctdtrinc: CapsuleType
    nctdtrit: CapsuleType
    ndtri: CapsuleType
    nrdtrimn: CapsuleType
    nrdtrisd: CapsuleType
    obl_ang1: CapsuleType
    obl_ang1_cv: CapsuleType
    obl_cv: CapsuleType
    obl_rad1: CapsuleType
    obl_rad1_cv: CapsuleType
    obl_rad2: CapsuleType
    obl_rad2_cv: CapsuleType
    owens_t: CapsuleType
    pbdv: CapsuleType
    pbvv: CapsuleType
    pbwa: CapsuleType
    pdtr: CapsuleType
    pdtrc: CapsuleType
    pdtrik: CapsuleType
    poch: CapsuleType
    pro_ang1: CapsuleType
    pro_ang1_cv: CapsuleType
    pro_cv: CapsuleType
    pro_rad1: CapsuleType
    pro_rad1_cv: CapsuleType
    pro_rad2: CapsuleType
    pro_rad2_cv: CapsuleType
    pseudo_huber: CapsuleType
    radian: CapsuleType
    rel_entr: CapsuleType
    round: CapsuleType
    sindg: CapsuleType
    stdtr: CapsuleType
    stdtridf: CapsuleType
    stdtrit: CapsuleType
    struve: CapsuleType
    tandg: CapsuleType
    tklmbda: CapsuleType
    voigt_profile: CapsuleType
    wofz: CapsuleType
    y0: CapsuleType
    y1: CapsuleType
    zetac: CapsuleType
    wright_bessel: CapsuleType
    log_wright_bessel: CapsuleType
    ndtri_exp: CapsuleType

    __pyx_fuse_0spherical_jn: CapsuleType
    __pyx_fuse_1spherical_jn: CapsuleType
    __pyx_fuse_0spherical_yn: CapsuleType
    __pyx_fuse_1spherical_yn: CapsuleType
    __pyx_fuse_0spherical_in: CapsuleType
    __pyx_fuse_1spherical_in: CapsuleType
    __pyx_fuse_0spherical_kn: CapsuleType
    __pyx_fuse_1spherical_kn: CapsuleType
    __pyx_fuse_0airy: CapsuleType
    __pyx_fuse_1airy: CapsuleType
    __pyx_fuse_0airye: CapsuleType
    __pyx_fuse_1airye: CapsuleType
    __pyx_fuse_0bdtr: CapsuleType
    __pyx_fuse_1bdtr: CapsuleType
    __pyx_fuse_2bdtr: CapsuleType
    __pyx_fuse_0bdtrc: CapsuleType
    __pyx_fuse_1bdtrc: CapsuleType
    __pyx_fuse_2bdtrc: CapsuleType
    __pyx_fuse_0bdtri: CapsuleType
    __pyx_fuse_1bdtri: CapsuleType
    __pyx_fuse_2bdtri: CapsuleType
    __pyx_fuse_0betainc: CapsuleType
    __pyx_fuse_1betainc: CapsuleType
    __pyx_fuse_0betaincc: CapsuleType
    __pyx_fuse_1betaincc: CapsuleType
    __pyx_fuse_0betaincinv: CapsuleType
    __pyx_fuse_1betaincinv: CapsuleType
    __pyx_fuse_0betainccinv: CapsuleType
    __pyx_fuse_1betainccinv: CapsuleType
    __pyx_fuse_0dawsn: CapsuleType
    __pyx_fuse_1dawsn: CapsuleType
    __pyx_fuse_0elliprc: CapsuleType
    __pyx_fuse_1elliprc: CapsuleType
    __pyx_fuse_0elliprd: CapsuleType
    __pyx_fuse_1elliprd: CapsuleType
    __pyx_fuse_0elliprf: CapsuleType
    __pyx_fuse_1elliprf: CapsuleType
    __pyx_fuse_0elliprg: CapsuleType
    __pyx_fuse_1elliprg: CapsuleType
    __pyx_fuse_0elliprj: CapsuleType
    __pyx_fuse_1elliprj: CapsuleType
    __pyx_fuse_0erf: CapsuleType
    __pyx_fuse_1erf: CapsuleType
    __pyx_fuse_0erfc: CapsuleType
    __pyx_fuse_1erfc: CapsuleType
    __pyx_fuse_0erfcx: CapsuleType
    __pyx_fuse_1erfcx: CapsuleType
    __pyx_fuse_0erfi: CapsuleType
    __pyx_fuse_1erfi: CapsuleType
    __pyx_fuse_0erfinv: CapsuleType
    __pyx_fuse_1erfinv: CapsuleType
    __pyx_fuse_0_0eval_chebyc: CapsuleType
    __pyx_fuse_0_1eval_chebyc: CapsuleType
    __pyx_fuse_1_0eval_chebyc: CapsuleType
    __pyx_fuse_1_1eval_chebyc: CapsuleType
    __pyx_fuse_2_0eval_chebyc: CapsuleType
    __pyx_fuse_2_1eval_chebyc: CapsuleType
    __pyx_fuse_0_0eval_chebys: CapsuleType
    __pyx_fuse_0_1eval_chebys: CapsuleType
    __pyx_fuse_1_0eval_chebys: CapsuleType
    __pyx_fuse_1_1eval_chebys: CapsuleType
    __pyx_fuse_2_0eval_chebys: CapsuleType
    __pyx_fuse_2_1eval_chebys: CapsuleType
    __pyx_fuse_0_0eval_chebyt: CapsuleType
    __pyx_fuse_0_1eval_chebyt: CapsuleType
    __pyx_fuse_1_0eval_chebyt: CapsuleType
    __pyx_fuse_1_1eval_chebyt: CapsuleType
    __pyx_fuse_2_0eval_chebyt: CapsuleType
    __pyx_fuse_2_1eval_chebyt: CapsuleType
    __pyx_fuse_0_0eval_chebyu: CapsuleType
    __pyx_fuse_0_1eval_chebyu: CapsuleType
    __pyx_fuse_1_0eval_chebyu: CapsuleType
    __pyx_fuse_1_1eval_chebyu: CapsuleType
    __pyx_fuse_2_0eval_chebyu: CapsuleType
    __pyx_fuse_2_1eval_chebyu: CapsuleType
    __pyx_fuse_0_0eval_gegenbauer: CapsuleType
    __pyx_fuse_0_1eval_gegenbauer: CapsuleType
    __pyx_fuse_1_0eval_gegenbauer: CapsuleType
    __pyx_fuse_1_1eval_gegenbauer: CapsuleType
    __pyx_fuse_2_0eval_gegenbauer: CapsuleType
    __pyx_fuse_2_1eval_gegenbauer: CapsuleType
    __pyx_fuse_0_0eval_genlaguerre: CapsuleType
    __pyx_fuse_0_1eval_genlaguerre: CapsuleType
    __pyx_fuse_1_0eval_genlaguerre: CapsuleType
    __pyx_fuse_1_1eval_genlaguerre: CapsuleType
    __pyx_fuse_2_0eval_genlaguerre: CapsuleType
    __pyx_fuse_2_1eval_genlaguerre: CapsuleType
    __pyx_fuse_0_0eval_jacobi: CapsuleType
    __pyx_fuse_0_1eval_jacobi: CapsuleType
    __pyx_fuse_1_0eval_jacobi: CapsuleType
    __pyx_fuse_1_1eval_jacobi: CapsuleType
    __pyx_fuse_2_0eval_jacobi: CapsuleType
    __pyx_fuse_2_1eval_jacobi: CapsuleType
    __pyx_fuse_0_0eval_laguerre: CapsuleType
    __pyx_fuse_0_1eval_laguerre: CapsuleType
    __pyx_fuse_1_0eval_laguerre: CapsuleType
    __pyx_fuse_1_1eval_laguerre: CapsuleType
    __pyx_fuse_2_0eval_laguerre: CapsuleType
    __pyx_fuse_2_1eval_laguerre: CapsuleType
    __pyx_fuse_0_0eval_legendre: CapsuleType
    __pyx_fuse_0_1eval_legendre: CapsuleType
    __pyx_fuse_1_0eval_legendre: CapsuleType
    __pyx_fuse_1_1eval_legendre: CapsuleType
    __pyx_fuse_2_0eval_legendre: CapsuleType
    __pyx_fuse_2_1eval_legendre: CapsuleType
    __pyx_fuse_0_0eval_sh_chebyt: CapsuleType
    __pyx_fuse_0_1eval_sh_chebyt: CapsuleType
    __pyx_fuse_1_0eval_sh_chebyt: CapsuleType
    __pyx_fuse_1_1eval_sh_chebyt: CapsuleType
    __pyx_fuse_2_0eval_sh_chebyt: CapsuleType
    __pyx_fuse_2_1eval_sh_chebyt: CapsuleType
    __pyx_fuse_0_0eval_sh_chebyu: CapsuleType
    __pyx_fuse_0_1eval_sh_chebyu: CapsuleType
    __pyx_fuse_1_0eval_sh_chebyu: CapsuleType
    __pyx_fuse_1_1eval_sh_chebyu: CapsuleType
    __pyx_fuse_2_0eval_sh_chebyu: CapsuleType
    __pyx_fuse_2_1eval_sh_chebyu: CapsuleType
    __pyx_fuse_0_0eval_sh_jacobi: CapsuleType
    __pyx_fuse_0_1eval_sh_jacobi: CapsuleType
    __pyx_fuse_1_0eval_sh_jacobi: CapsuleType
    __pyx_fuse_1_1eval_sh_jacobi: CapsuleType
    __pyx_fuse_2_0eval_sh_jacobi: CapsuleType
    __pyx_fuse_2_1eval_sh_jacobi: CapsuleType
    __pyx_fuse_0_0eval_sh_legendre: CapsuleType
    __pyx_fuse_0_1eval_sh_legendre: CapsuleType
    __pyx_fuse_1_0eval_sh_legendre: CapsuleType
    __pyx_fuse_1_1eval_sh_legendre: CapsuleType
    __pyx_fuse_2_0eval_sh_legendre: CapsuleType
    __pyx_fuse_2_1eval_sh_legendre: CapsuleType
    __pyx_fuse_0exp1: CapsuleType
    __pyx_fuse_1exp1: CapsuleType
    __pyx_fuse_0expi: CapsuleType
    __pyx_fuse_1expi: CapsuleType
    __pyx_fuse_0expit: CapsuleType
    __pyx_fuse_1expit: CapsuleType
    __pyx_fuse_2expit: CapsuleType
    __pyx_fuse_0expm1: CapsuleType
    __pyx_fuse_1expm1: CapsuleType
    __pyx_fuse_0expn: CapsuleType
    __pyx_fuse_1expn: CapsuleType
    __pyx_fuse_2expn: CapsuleType
    __pyx_fuse_0fresnel: CapsuleType
    __pyx_fuse_1fresnel: CapsuleType
    __pyx_fuse_0gamma: CapsuleType
    __pyx_fuse_1gamma: CapsuleType
    __pyx_fuse_0hyp0f1: CapsuleType
    __pyx_fuse_1hyp0f1: CapsuleType
    __pyx_fuse_0hyp1f1: CapsuleType
    __pyx_fuse_1hyp1f1: CapsuleType
    __pyx_fuse_0hyp2f1: CapsuleType
    __pyx_fuse_1hyp2f1: CapsuleType
    __pyx_fuse_0iv: CapsuleType
    __pyx_fuse_1iv: CapsuleType
    __pyx_fuse_0ive: CapsuleType
    __pyx_fuse_1ive: CapsuleType
    __pyx_fuse_0jv: CapsuleType
    __pyx_fuse_1jv: CapsuleType
    __pyx_fuse_0jve: CapsuleType
    __pyx_fuse_1jve: CapsuleType
    __pyx_fuse_0kn: CapsuleType
    __pyx_fuse_1kn: CapsuleType
    __pyx_fuse_2kn: CapsuleType
    __pyx_fuse_0kv: CapsuleType
    __pyx_fuse_1kv: CapsuleType
    __pyx_fuse_0kve: CapsuleType
    __pyx_fuse_1kve: CapsuleType
    __pyx_fuse_0log1p: CapsuleType
    __pyx_fuse_1log1p: CapsuleType
    __pyx_fuse_0log_expit: CapsuleType
    __pyx_fuse_1log_expit: CapsuleType
    __pyx_fuse_2log_expit: CapsuleType
    __pyx_fuse_0log_ndtr: CapsuleType
    __pyx_fuse_1log_ndtr: CapsuleType
    __pyx_fuse_0loggamma: CapsuleType
    __pyx_fuse_1loggamma: CapsuleType
    __pyx_fuse_0logit: CapsuleType
    __pyx_fuse_1logit: CapsuleType
    __pyx_fuse_2logit: CapsuleType
    __pyx_fuse_0nbdtr: CapsuleType
    __pyx_fuse_1nbdtr: CapsuleType
    __pyx_fuse_2nbdtr: CapsuleType
    __pyx_fuse_0nbdtrc: CapsuleType
    __pyx_fuse_1nbdtrc: CapsuleType
    __pyx_fuse_2nbdtrc: CapsuleType
    __pyx_fuse_0nbdtri: CapsuleType
    __pyx_fuse_1nbdtri: CapsuleType
    __pyx_fuse_2nbdtri: CapsuleType
    __pyx_fuse_0ndtr: CapsuleType
    __pyx_fuse_1ndtr: CapsuleType
    __pyx_fuse_0pdtri: CapsuleType
    __pyx_fuse_1pdtri: CapsuleType
    __pyx_fuse_2pdtri: CapsuleType
    __pyx_fuse_0powm1: CapsuleType
    __pyx_fuse_1powm1: CapsuleType
    __pyx_fuse_0psi: CapsuleType
    __pyx_fuse_1psi: CapsuleType
    __pyx_fuse_0rgamma: CapsuleType
    __pyx_fuse_1rgamma: CapsuleType
    __pyx_fuse_0shichi: CapsuleType
    __pyx_fuse_1shichi: CapsuleType
    __pyx_fuse_0sici: CapsuleType
    __pyx_fuse_1sici: CapsuleType
    __pyx_fuse_0smirnov: CapsuleType
    __pyx_fuse_1smirnov: CapsuleType
    __pyx_fuse_2smirnov: CapsuleType
    __pyx_fuse_0smirnovi: CapsuleType
    __pyx_fuse_1smirnovi: CapsuleType
    __pyx_fuse_2smirnovi: CapsuleType
    __pyx_fuse_0spence: CapsuleType
    __pyx_fuse_1spence: CapsuleType
    __pyx_fuse_0sph_harm: CapsuleType
    __pyx_fuse_1sph_harm: CapsuleType
    __pyx_fuse_2sph_harm: CapsuleType
    __pyx_fuse_0wrightomega: CapsuleType
    __pyx_fuse_1wrightomega: CapsuleType
    __pyx_fuse_0xlog1py: CapsuleType
    __pyx_fuse_1xlog1py: CapsuleType
    __pyx_fuse_0xlogy: CapsuleType
    __pyx_fuse_1xlogy: CapsuleType
    __pyx_fuse_0yn: CapsuleType
    __pyx_fuse_1yn: CapsuleType
    __pyx_fuse_2yn: CapsuleType
    __pyx_fuse_0yv: CapsuleType
    __pyx_fuse_1yv: CapsuleType
    __pyx_fuse_0yve: CapsuleType
    __pyx_fuse_1yve: CapsuleType

class _TestDict(TypedDict): ...

__pyx_capi__: Final[_CApiDict]
__test__: Final[_TestDict]

agm: Final[_CythonFunctionOrMethod_2f]
bdtr: Final[_CythonFunctionOrMethod_3f]
bdtrc: Final[_CythonFunctionOrMethod_3f]
bdtri: Final[_CythonFunctionOrMethod_3f]
bdtrik: Final[_CythonFunctionOrMethod_3f]
bdtrin: Final[_CythonFunctionOrMethod_3f]
bei: Final[_CythonFunctionOrMethod_1f]
beip: Final[_CythonFunctionOrMethod_1f]
ber: Final[_CythonFunctionOrMethod_1f]
berp: Final[_CythonFunctionOrMethod_1f]
besselpoly: Final[_CythonFunctionOrMethod_3f]
beta: Final[_CythonFunctionOrMethod_2f]
betainc: Final[_CythonFunctionOrMethod_3f]
betaincc: Final[_CythonFunctionOrMethod_3f]
betainccinv: Final[_CythonFunctionOrMethod_3f]
betaincinv: Final[_CythonFunctionOrMethod_3f]
betaln: Final[_CythonFunctionOrMethod_2f]
binom: Final[_CythonFunctionOrMethod_2f]
boxcox: Final[_CythonFunctionOrMethod_2f]
boxcox1p: Final[_CythonFunctionOrMethod_2f]
btdtr: Final[_CythonFunctionOrMethod_3f]
btdtri: Final[_CythonFunctionOrMethod_3f]
btdtria: Final[_CythonFunctionOrMethod_3f]
btdtrib: Final[_CythonFunctionOrMethod_3f]
cbrt: Final[_CythonFunctionOrMethod_1f]
chdtr: Final[_CythonFunctionOrMethod_2f]
chdtrc: Final[_CythonFunctionOrMethod_2f]
chdtri: Final[_CythonFunctionOrMethod_2f]
chdtriv: Final[_CythonFunctionOrMethod_2f]
chndtr: Final[_CythonFunctionOrMethod_3f]
chndtridf: Final[_CythonFunctionOrMethod_3f]
chndtrinc: Final[_CythonFunctionOrMethod_3f]
chndtrix: Final[_CythonFunctionOrMethod_3f]
cosdg: Final[_CythonFunctionOrMethod_1f]
cosm1: Final[_CythonFunctionOrMethod_1f]
cotdg: Final[_CythonFunctionOrMethod_1f]
dawsn: Final[_CythonFunctionOrMethod_1fc]
ellipe: Final[_CythonFunctionOrMethod_1f]
ellipeinc: Final[_CythonFunctionOrMethod_2f]
ellipk: Final[_CythonFunctionOrMethod_1f]
ellipkinc: Final[_CythonFunctionOrMethod_2f]
ellipkm1: Final[_CythonFunctionOrMethod_1f]
elliprc: Final[_CythonFunctionOrMethod_2fc]
elliprd: Final[_CythonFunctionOrMethod_3fc]
elliprf: Final[_CythonFunctionOrMethod_3fc]
elliprg: Final[_CythonFunctionOrMethod_3fc]
elliprj: Final[_CythonFunctionOrMethod_4fc]
entr: Final[_CythonFunctionOrMethod_1f]
erf: Final[_CythonFunctionOrMethod_1fc]
erfc: Final[_CythonFunctionOrMethod_1fc]
erfcinv: Final[_CythonFunctionOrMethod_1f]
erfcx: Final[_CythonFunctionOrMethod_1fc]
erfi: Final[_CythonFunctionOrMethod_1fc]
erfinv: Final[_CythonFunctionOrMethod_1f]
eval_chebyc: Final[_CythonFunctionOrMethod_2_poly]
eval_chebys: Final[_CythonFunctionOrMethod_2_poly]
eval_chebyt: Final[_CythonFunctionOrMethod_2_poly]
eval_chebyu: Final[_CythonFunctionOrMethod_2_poly]
eval_gegenbauer: Final[_CythonFunctionOrMethod_3_poly]
eval_genlaguerre: Final[_CythonFunctionOrMethod_3_poly]
eval_hermite: Final[_CythonFunctionOrMethod_2_poly]
eval_hermitenorm: Final[_CythonFunctionOrMethod_2_poly]
eval_jacobi: Final[_CythonFunctionOrMethod_4_poly]
eval_laguerre: Final[_CythonFunctionOrMethod_2_poly]
eval_legendre: Final[_CythonFunctionOrMethod_2_poly]
eval_sh_chebyt: Final[_CythonFunctionOrMethod_2_poly]
eval_sh_chebyu: Final[_CythonFunctionOrMethod_2_poly]
eval_sh_jacobi: Final[_CythonFunctionOrMethod_4_poly]
eval_sh_legendre: Final[_CythonFunctionOrMethod_2_poly]
exp1: Final[_CythonFunctionOrMethod_1fc]
exp2: Final[_CythonFunctionOrMethod_1f]
exp10: Final[_CythonFunctionOrMethod_1f]
expi: Final[_CythonFunctionOrMethod_1fc]
expit: Final[_CythonFunctionOrMethod_1f]
expm1: Final[_CythonFunctionOrMethod_1fc]
expn: Final[_CythonFunctionOrMethod_2f]
exprel: Final[_CythonFunctionOrMethod_1f]
fdtr: Final[_CythonFunctionOrMethod_3f]
fdtrc: Final[_CythonFunctionOrMethod_3f]
fdtri: Final[_CythonFunctionOrMethod_3f]
fdtridfd: Final[_CythonFunctionOrMethod_3f]
gamma: Final[_CythonFunctionOrMethod_1fc]
gammainc: Final[_CythonFunctionOrMethod_2f]
gammaincc: Final[_CythonFunctionOrMethod_2f]
gammainccinv: Final[_CythonFunctionOrMethod_2f]
gammaincinv: Final[_CythonFunctionOrMethod_2f]
gammaln: Final[_CythonFunctionOrMethod_1f]
gammasgn: Final[_CythonFunctionOrMethod_1f]
gdtr: Final[_CythonFunctionOrMethod_3f]
gdtrc: Final[_CythonFunctionOrMethod_3f]
gdtria: Final[_CythonFunctionOrMethod_3f]
gdtrib: Final[_CythonFunctionOrMethod_3f]
gdtrix: Final[_CythonFunctionOrMethod_3f]
hankel1: Final[_CythonFunctionOrMethod_2_hankel]
hankel1e: Final[_CythonFunctionOrMethod_2_hankel]
hankel2: Final[_CythonFunctionOrMethod_2_hankel]
hankel2e: Final[_CythonFunctionOrMethod_2_hankel]
huber: Final[_CythonFunctionOrMethod_2f]
hyp0f1: Final[_CythonFunctionOrMethod_2fc]
hyp1f1: Final[_CythonFunctionOrMethod_3fc]
hyp2f1: Final[_CythonFunctionOrMethod_4fc]
hyperu: Final[_CythonFunctionOrMethod_3f]
i0: Final[_CythonFunctionOrMethod_1f]
i0e: Final[_CythonFunctionOrMethod_1f]
i1: Final[_CythonFunctionOrMethod_1f]
i1e: Final[_CythonFunctionOrMethod_1f]
inv_boxcox: Final[_CythonFunctionOrMethod_2f]
inv_boxcox1p: Final[_CythonFunctionOrMethod_2f]
it2struve0: Final[_CythonFunctionOrMethod_1f]
itmodstruve0: Final[_CythonFunctionOrMethod_1f]
itstruve0: Final[_CythonFunctionOrMethod_1f]
iv: Final[_CythonFunctionOrMethod_2fc]
ive: Final[_CythonFunctionOrMethod_2fc]
j0: Final[_CythonFunctionOrMethod_1f]
j1: Final[_CythonFunctionOrMethod_1f]
jv: Final[_CythonFunctionOrMethod_2fc]
jve: Final[_CythonFunctionOrMethod_2fc]
k0: Final[_CythonFunctionOrMethod_1f]
k0e: Final[_CythonFunctionOrMethod_1f]
k1: Final[_CythonFunctionOrMethod_1f]
k1e: Final[_CythonFunctionOrMethod_1f]
kei: Final[_CythonFunctionOrMethod_1f]
keip: Final[_CythonFunctionOrMethod_1f]
ker: Final[_CythonFunctionOrMethod_1f]
kerp: Final[_CythonFunctionOrMethod_1f]
kl_div: Final[_CythonFunctionOrMethod_2f]
kn: Final[_CythonFunctionOrMethod_2f]
kolmogi: Final[_CythonFunctionOrMethod_1f]
kolmogorov: Final[_CythonFunctionOrMethod_1f]
kv: Final[_CythonFunctionOrMethod_2fc]
kve: Final[_CythonFunctionOrMethod_2fc]
log1p: Final[_CythonFunctionOrMethod_1fc]
log_expit: Final[_CythonFunctionOrMethod_1f]
log_ndtr: Final[_CythonFunctionOrMethod_1fc]
log_wright_bessel: Final[_CythonFunctionOrMethod_3f]
loggamma: Final[_CythonFunctionOrMethod_1fc]
logit: Final[_CythonFunctionOrMethod_1f]
lpmv: Final[_CythonFunctionOrMethod_3f]
mathieu_a: Final[_CythonFunctionOrMethod_2f]
mathieu_b: Final[_CythonFunctionOrMethod_2f]
modstruve: Final[_CythonFunctionOrMethod_2f]
nbdtr: Final[_CythonFunctionOrMethod_3f]
nbdtrc: Final[_CythonFunctionOrMethod_3f]
nbdtri: Final[_CythonFunctionOrMethod_3f]
nbdtrik: Final[_CythonFunctionOrMethod_3f]
nbdtrin: Final[_CythonFunctionOrMethod_3f]
ncfdtr: Final[_CythonFunctionOrMethod_4f]
ncfdtri: Final[_CythonFunctionOrMethod_4f]
ncfdtridfd: Final[_CythonFunctionOrMethod_4f]
ncfdtridfn: Final[_CythonFunctionOrMethod_4f]
ncfdtrinc: Final[_CythonFunctionOrMethod_4f]
nctdtr: Final[_CythonFunctionOrMethod_3f]
nctdtridf: Final[_CythonFunctionOrMethod_3f]
nctdtrinc: Final[_CythonFunctionOrMethod_3f]
nctdtrit: Final[_CythonFunctionOrMethod_3f]
ndtr: Final[_CythonFunctionOrMethod_1fc]
ndtri: Final[_CythonFunctionOrMethod_1f]
ndtri_exp: Final[_CythonFunctionOrMethod_1f]
nrdtrimn: Final[_CythonFunctionOrMethod_3f]
nrdtrisd: Final[_CythonFunctionOrMethod_3f]
obl_cv: Final[_CythonFunctionOrMethod_3f]
owens_t: Final[_CythonFunctionOrMethod_2f]
pdtr: Final[_CythonFunctionOrMethod_2f]
pdtrc: Final[_CythonFunctionOrMethod_2f]
pdtri: Final[_CythonFunctionOrMethod_2f]
pdtrik: Final[_CythonFunctionOrMethod_2f]
poch: Final[_CythonFunctionOrMethod_2f]
powm1: Final[_CythonFunctionOrMethod_2f]
pro_cv: Final[_CythonFunctionOrMethod_3f]
pseudo_huber: Final[_CythonFunctionOrMethod_2f]
psi: Final[_CythonFunctionOrMethod_1fc]
radian: Final[_CythonFunctionOrMethod_3f]
rel_entr: Final[_CythonFunctionOrMethod_2f]
rgamma: Final[_CythonFunctionOrMethod_1fc]
round: Final[_CythonFunctionOrMethod_1f]
sindg: Final[_CythonFunctionOrMethod_1f]
smirnov: Final[_CythonFunctionOrMethod_2f]
smirnovi: Final[_CythonFunctionOrMethod_2f]
spence: Final[_CythonFunctionOrMethod_1fc]
sph_harm: Final[_CythonFunctionOrMethod_4fc]
stdtr: Final[_CythonFunctionOrMethod_2f]
stdtridf: Final[_CythonFunctionOrMethod_2f]
stdtrit: Final[_CythonFunctionOrMethod_2f]
struve: Final[_CythonFunctionOrMethod_2f]
tandg: Final[_CythonFunctionOrMethod_1f]
tklmbda: Final[_CythonFunctionOrMethod_2f]
voigt_profile: Final[_CythonFunctionOrMethod_3f]
wofz: Final[_CythonFunctionOrMethod_1c]
wright_bessel: Final[_CythonFunctionOrMethod_3f]
wrightomega: Final[_CythonFunctionOrMethod_1fc]
xlog1py: Final[_CythonFunctionOrMethod_2fc]
xlogy: Final[_CythonFunctionOrMethod_2fc]
y0: Final[_CythonFunctionOrMethod_1f]
y1: Final[_CythonFunctionOrMethod_1f]
yn: Final[_CythonFunctionOrMethod_2f]
yv: Final[_CythonFunctionOrMethod_2fc]
yve: Final[_CythonFunctionOrMethod_2fc]
zetac: Final[_CythonFunctionOrMethod_1f]

spherical_in: Final[_CythonFunctionOrMethod_2_spherical]
spherical_jn: Final[_CythonFunctionOrMethod_2_spherical]
spherical_kn: Final[_CythonFunctionOrMethod_2_spherical]
spherical_yn: Final[_CythonFunctionOrMethod_2_spherical]
