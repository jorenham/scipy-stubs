from collections.abc import Callable, Mapping
from typing import Final, TypeAlias

from typing_extensions import CapsuleType

__all__ = [
    "__pyx_capi__",
    "agm",
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
    "it2struve0",
    "itmodstruve0",
    "itstruve0",
    "iv",
    "ive",
    "j0",
    "j1",
    "jv",
    "jve",
    "k0",
    "k0e",
    "k1",
    "k1e",
    "kei",
    "keip",
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
    "obl_cv",
    "owens_t",
    "pdtr",
    "pdtrc",
    "pdtri",
    "pdtrik",
    "poch",
    "powm1",
    "pro_cv",
    "pseudo_huber",
    "psi",
    "radian",
    "rel_entr",
    "rgamma",
    "round",
    "sindg",
    "smirnov",
    "smirnovi",
    "spence",
    "sph_harm",
    "spherical_in",
    "spherical_jn",
    "spherical_kn",
    "spherical_yn",
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

__pyx_capi__: Final[Mapping[str, CapsuleType]]

_CythonFunctionOrMethod: TypeAlias = Callable[..., object]

agm: _CythonFunctionOrMethod
bdtr: _CythonFunctionOrMethod
bdtrc: _CythonFunctionOrMethod
bdtri: _CythonFunctionOrMethod
bdtrik: _CythonFunctionOrMethod
bdtrin: _CythonFunctionOrMethod
bei: _CythonFunctionOrMethod
beip: _CythonFunctionOrMethod
ber: _CythonFunctionOrMethod
berp: _CythonFunctionOrMethod
besselpoly: _CythonFunctionOrMethod
beta: _CythonFunctionOrMethod
betainc: _CythonFunctionOrMethod
betaincc: _CythonFunctionOrMethod
betainccinv: _CythonFunctionOrMethod
betaincinv: _CythonFunctionOrMethod
betaln: _CythonFunctionOrMethod
binom: _CythonFunctionOrMethod
boxcox: _CythonFunctionOrMethod
boxcox1p: _CythonFunctionOrMethod
btdtr: _CythonFunctionOrMethod
btdtri: _CythonFunctionOrMethod
btdtria: _CythonFunctionOrMethod
btdtrib: _CythonFunctionOrMethod
cbrt: _CythonFunctionOrMethod
chdtr: _CythonFunctionOrMethod
chdtrc: _CythonFunctionOrMethod
chdtri: _CythonFunctionOrMethod
chdtriv: _CythonFunctionOrMethod
chndtr: _CythonFunctionOrMethod
chndtridf: _CythonFunctionOrMethod
chndtrinc: _CythonFunctionOrMethod
chndtrix: _CythonFunctionOrMethod
cosdg: _CythonFunctionOrMethod
cosm1: _CythonFunctionOrMethod
cotdg: _CythonFunctionOrMethod
dawsn: _CythonFunctionOrMethod
ellipe: _CythonFunctionOrMethod
ellipeinc: _CythonFunctionOrMethod
ellipk: _CythonFunctionOrMethod
ellipkinc: _CythonFunctionOrMethod
ellipkm1: _CythonFunctionOrMethod
elliprc: _CythonFunctionOrMethod
elliprd: _CythonFunctionOrMethod
elliprf: _CythonFunctionOrMethod
elliprg: _CythonFunctionOrMethod
elliprj: _CythonFunctionOrMethod
entr: _CythonFunctionOrMethod
erf: _CythonFunctionOrMethod
erfc: _CythonFunctionOrMethod
erfcinv: _CythonFunctionOrMethod
erfcx: _CythonFunctionOrMethod
erfi: _CythonFunctionOrMethod
erfinv: _CythonFunctionOrMethod
eval_chebyc: _CythonFunctionOrMethod
eval_chebys: _CythonFunctionOrMethod
eval_chebyt: _CythonFunctionOrMethod
eval_chebyu: _CythonFunctionOrMethod
eval_gegenbauer: _CythonFunctionOrMethod
eval_genlaguerre: _CythonFunctionOrMethod
eval_hermite: _CythonFunctionOrMethod
eval_hermitenorm: _CythonFunctionOrMethod
eval_jacobi: _CythonFunctionOrMethod
eval_laguerre: _CythonFunctionOrMethod
eval_legendre: _CythonFunctionOrMethod
eval_sh_chebyt: _CythonFunctionOrMethod
eval_sh_chebyu: _CythonFunctionOrMethod
eval_sh_jacobi: _CythonFunctionOrMethod
eval_sh_legendre: _CythonFunctionOrMethod
exp1: _CythonFunctionOrMethod
exp2: _CythonFunctionOrMethod
exp10: _CythonFunctionOrMethod
expi: _CythonFunctionOrMethod
expit: _CythonFunctionOrMethod
expm1: _CythonFunctionOrMethod
expn: _CythonFunctionOrMethod
exprel: _CythonFunctionOrMethod
fdtr: _CythonFunctionOrMethod
fdtrc: _CythonFunctionOrMethod
fdtri: _CythonFunctionOrMethod
fdtridfd: _CythonFunctionOrMethod
gamma: _CythonFunctionOrMethod
gammainc: _CythonFunctionOrMethod
gammaincc: _CythonFunctionOrMethod
gammainccinv: _CythonFunctionOrMethod
gammaincinv: _CythonFunctionOrMethod
gammaln: _CythonFunctionOrMethod
gammasgn: _CythonFunctionOrMethod
gdtr: _CythonFunctionOrMethod
gdtrc: _CythonFunctionOrMethod
gdtria: _CythonFunctionOrMethod
gdtrib: _CythonFunctionOrMethod
gdtrix: _CythonFunctionOrMethod
hankel1: _CythonFunctionOrMethod
hankel1e: _CythonFunctionOrMethod
hankel2: _CythonFunctionOrMethod
hankel2e: _CythonFunctionOrMethod
huber: _CythonFunctionOrMethod
hyp0f1: _CythonFunctionOrMethod
hyp1f1: _CythonFunctionOrMethod
hyp2f1: _CythonFunctionOrMethod
hyperu: _CythonFunctionOrMethod
i0: _CythonFunctionOrMethod
i0e: _CythonFunctionOrMethod
i1: _CythonFunctionOrMethod
i1e: _CythonFunctionOrMethod
inv_boxcox: _CythonFunctionOrMethod
inv_boxcox1p: _CythonFunctionOrMethod
it2struve0: _CythonFunctionOrMethod
itmodstruve0: _CythonFunctionOrMethod
itstruve0: _CythonFunctionOrMethod
iv: _CythonFunctionOrMethod
ive: _CythonFunctionOrMethod
j0: _CythonFunctionOrMethod
j1: _CythonFunctionOrMethod
jv: _CythonFunctionOrMethod
jve: _CythonFunctionOrMethod
k0: _CythonFunctionOrMethod
k0e: _CythonFunctionOrMethod
k1: _CythonFunctionOrMethod
k1e: _CythonFunctionOrMethod
kei: _CythonFunctionOrMethod
keip: _CythonFunctionOrMethod
ker: _CythonFunctionOrMethod
kerp: _CythonFunctionOrMethod
kl_div: _CythonFunctionOrMethod
kn: _CythonFunctionOrMethod
kolmogi: _CythonFunctionOrMethod
kolmogorov: _CythonFunctionOrMethod
kv: _CythonFunctionOrMethod
kve: _CythonFunctionOrMethod
log1p: _CythonFunctionOrMethod
log_expit: _CythonFunctionOrMethod
log_ndtr: _CythonFunctionOrMethod
log_wright_bessel: _CythonFunctionOrMethod
loggamma: _CythonFunctionOrMethod
logit: _CythonFunctionOrMethod
lpmv: _CythonFunctionOrMethod
mathieu_a: _CythonFunctionOrMethod
mathieu_b: _CythonFunctionOrMethod
modstruve: _CythonFunctionOrMethod
nbdtr: _CythonFunctionOrMethod
nbdtrc: _CythonFunctionOrMethod
nbdtri: _CythonFunctionOrMethod
nbdtrik: _CythonFunctionOrMethod
nbdtrin: _CythonFunctionOrMethod
ncfdtr: _CythonFunctionOrMethod
ncfdtri: _CythonFunctionOrMethod
ncfdtridfd: _CythonFunctionOrMethod
ncfdtridfn: _CythonFunctionOrMethod
ncfdtrinc: _CythonFunctionOrMethod
nctdtr: _CythonFunctionOrMethod
nctdtridf: _CythonFunctionOrMethod
nctdtrinc: _CythonFunctionOrMethod
nctdtrit: _CythonFunctionOrMethod
ndtr: _CythonFunctionOrMethod
ndtri: _CythonFunctionOrMethod
ndtri_exp: _CythonFunctionOrMethod
nrdtrimn: _CythonFunctionOrMethod
nrdtrisd: _CythonFunctionOrMethod
obl_cv: _CythonFunctionOrMethod
owens_t: _CythonFunctionOrMethod
pdtr: _CythonFunctionOrMethod
pdtrc: _CythonFunctionOrMethod
pdtri: _CythonFunctionOrMethod
pdtrik: _CythonFunctionOrMethod
poch: _CythonFunctionOrMethod
powm1: _CythonFunctionOrMethod
pro_cv: _CythonFunctionOrMethod
pseudo_huber: _CythonFunctionOrMethod
psi: _CythonFunctionOrMethod
radian: _CythonFunctionOrMethod
rel_entr: _CythonFunctionOrMethod
rgamma: _CythonFunctionOrMethod
round: _CythonFunctionOrMethod
sindg: _CythonFunctionOrMethod
smirnov: _CythonFunctionOrMethod
smirnovi: _CythonFunctionOrMethod
spence: _CythonFunctionOrMethod
sph_harm: _CythonFunctionOrMethod
spherical_in: _CythonFunctionOrMethod
spherical_jn: _CythonFunctionOrMethod
spherical_kn: _CythonFunctionOrMethod
spherical_yn: _CythonFunctionOrMethod
stdtr: _CythonFunctionOrMethod
stdtridf: _CythonFunctionOrMethod
stdtrit: _CythonFunctionOrMethod
struve: _CythonFunctionOrMethod
tandg: _CythonFunctionOrMethod
tklmbda: _CythonFunctionOrMethod
voigt_profile: _CythonFunctionOrMethod
wofz: _CythonFunctionOrMethod
wright_bessel: _CythonFunctionOrMethod
wrightomega: _CythonFunctionOrMethod
xlog1py: _CythonFunctionOrMethod
xlogy: _CythonFunctionOrMethod
y0: _CythonFunctionOrMethod
y1: _CythonFunctionOrMethod
yn: _CythonFunctionOrMethod
yv: _CythonFunctionOrMethod
yve: _CythonFunctionOrMethod
zetac: _CythonFunctionOrMethod
