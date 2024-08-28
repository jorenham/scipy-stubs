from typing import Final

from typing_extensions import deprecated

from ._distn_infrastructure import rv_continuous

__all__ = [
    "alpha",
    "anglit",
    "arcsine",
    "argus",
    "beta",
    "betaprime",
    "bradford",
    "burr",
    "burr12",
    "cauchy",
    "chi",
    "chi2",
    "cosine",
    "crystalball",
    "dgamma",
    "dweibull",
    "erlang",
    "expon",
    "exponnorm",
    "exponpow",
    "exponweib",
    "f",
    "fatiguelife",
    "fisk",
    "foldcauchy",
    "foldnorm",
    "gamma",
    "gausshyper",
    "genexpon",
    "genextreme",
    "gengamma",
    "genhalflogistic",
    "genhyperbolic",
    "geninvgauss",
    "genlogistic",
    "gennorm",
    "genpareto",
    "gibrat",
    "gompertz",
    "gumbel_l",
    "gumbel_r",
    "halfcauchy",
    "halfgennorm",
    "halflogistic",
    "halfnorm",
    "hypsecant",
    "invgamma",
    "invgauss",
    "invweibull",
    "irwinhall",
    "jf_skew_t",
    "johnsonsb",
    "johnsonsu",
    "kappa3",
    "kappa4",
    "ksone",
    "kstwo",
    "kstwobign",
    "laplace",
    "laplace_asymmetric",
    "levy",
    "levy_l",
    "loggamma",
    "logistic",
    "loglaplace",
    "lognorm",
    "loguniform",
    "lomax",
    "maxwell",
    "mielke",
    "moyal",
    "nakagami",
    "ncf",
    "nct",
    "ncx2",
    "norm",
    "norminvgauss",
    "pareto",
    "pearson3",
    "powerlaw",
    "powerlognorm",
    "powernorm",
    "rayleigh",
    "rdist",
    "recipinvgauss",
    "reciprocal",
    "rel_breitwigner",
    "rice",
    "rv_histogram",
    "semicircular",
    "skewcauchy",
    "skewnorm",
    "studentized_range",
    "t",
    "trapezoid",
    "trapz",
    "triang",
    "truncexpon",
    "truncnorm",
    "truncpareto",
    "truncweibull_min",
    "tukeylambda",
    "uniform",
    "vonmises",
    "vonmises_line",
    "wald",
    "weibull_max",
    "weibull_min",
    "wrapcauchy",
]

class ksone_gen(rv_continuous): ...
ksone: Final[ksone_gen]

class kstwo_gen(rv_continuous): ...
kstwo: Final[kstwo_gen]

class kstwobign_gen(rv_continuous): ...
kstwobign: Final[kstwobign_gen]

class norm_gen(rv_continuous): ...
norm: Final[norm_gen]

class alpha_gen(rv_continuous): ...
alpha: Final[alpha_gen]

class anglit_gen(rv_continuous): ...
anglit: Final[anglit_gen]

class arcsine_gen(rv_continuous): ...
arcsine: Final[arcsine_gen]

class beta_gen(rv_continuous): ...
beta: Final[beta_gen]

class betaprime_gen(rv_continuous): ...
betaprime: Final[betaprime_gen]

class bradford_gen(rv_continuous): ...
bradford: Final[bradford_gen]

class burr_gen(rv_continuous): ...
burr: Final[burr_gen]

class burr12_gen(rv_continuous): ...
burr12: Final[burr12_gen]

class fisk_gen(burr_gen): ...
fisk: Final[fisk_gen]

class cauchy_gen(rv_continuous): ...
cauchy: Final[cauchy_gen]

class chi_gen(rv_continuous): ...
chi: Final[chi_gen]

class chi2_gen(rv_continuous): ...
chi2: Final[chi2_gen]

class cosine_gen(rv_continuous): ...
cosine: Final[cosine_gen]

class dgamma_gen(rv_continuous): ...
dgamma: Final[dgamma_gen]

class dweibull_gen(rv_continuous): ...
dweibull: Final[dweibull_gen]

class expon_gen(rv_continuous): ...
expon: Final[expon_gen]

class exponnorm_gen(rv_continuous): ...
exponnorm: Final[exponnorm_gen]

class exponweib_gen(rv_continuous): ...
exponweib: Final[exponweib_gen]

class exponpow_gen(rv_continuous): ...
exponpow: Final[exponpow_gen]

class fatiguelife_gen(rv_continuous): ...
fatiguelife: Final[fatiguelife_gen]

class foldcauchy_gen(rv_continuous): ...
foldcauchy: Final[foldcauchy_gen]

class f_gen(rv_continuous): ...
f: Final[f_gen]

class foldnorm_gen(rv_continuous): ...
foldnorm: Final[foldnorm_gen]

class weibull_min_gen(rv_continuous): ...
weibull_min: Final[weibull_min_gen]

class truncweibull_min_gen(rv_continuous): ...
truncweibull_min: Final[truncweibull_min_gen]

class weibull_max_gen(rv_continuous): ...
weibull_max: Final[weibull_max_gen]

class genlogistic_gen(rv_continuous): ...
genlogistic: Final[genlogistic_gen]

class genpareto_gen(rv_continuous): ...
genpareto: Final[genpareto_gen]

class genexpon_gen(rv_continuous): ...
genexpon: Final[genexpon_gen]

class genextreme_gen(rv_continuous): ...
genextreme: Final[genextreme_gen]

class gamma_gen(rv_continuous): ...
gamma: Final[gamma_gen]

class erlang_gen(gamma_gen): ...
erlang: Final[erlang_gen]

class gengamma_gen(rv_continuous): ...
gengamma: Final[gengamma_gen]

class genhalflogistic_gen(rv_continuous): ...
genhalflogistic: Final[genhalflogistic_gen]

class genhyperbolic_gen(rv_continuous): ...
genhyperbolic: Final[genhyperbolic_gen]

class gompertz_gen(rv_continuous): ...
gompertz: Final[gompertz_gen]

class gumbel_r_gen(rv_continuous): ...
gumbel_r: Final[gumbel_r_gen]

class gumbel_l_gen(rv_continuous): ...
gumbel_l: Final[gumbel_l_gen]

class halfcauchy_gen(rv_continuous): ...
halfcauchy: Final[halfcauchy_gen]

class halflogistic_gen(rv_continuous): ...
halflogistic: Final[halflogistic_gen]

class halfnorm_gen(rv_continuous): ...
halfnorm: Final[halfnorm_gen]

class hypsecant_gen(rv_continuous): ...
hypsecant: Final[hypsecant_gen]

class gausshyper_gen(rv_continuous): ...
gausshyper: Final[gausshyper_gen]

class invgamma_gen(rv_continuous): ...
invgamma: Final[invgamma_gen]

class invgauss_gen(rv_continuous): ...
invgauss: Final[invgauss_gen]

class geninvgauss_gen(rv_continuous): ...
geninvgauss: Final[geninvgauss_gen]

class norminvgauss_gen(rv_continuous): ...
norminvgauss: Final[norminvgauss_gen]

class invweibull_gen(rv_continuous): ...
invweibull: Final[invweibull_gen]

class jf_skew_t_gen(rv_continuous): ...
jf_skew_t: Final[jf_skew_t_gen]

class johnsonsb_gen(rv_continuous): ...
johnsonsb: Final[johnsonsb_gen]

class johnsonsu_gen(rv_continuous): ...
johnsonsu: Final[johnsonsu_gen]

class laplace_gen(rv_continuous): ...
laplace: Final[laplace_gen]

class laplace_asymmetric_gen(rv_continuous): ...
laplace_asymmetric: Final[laplace_asymmetric_gen]

class levy_gen(rv_continuous): ...
levy: Final[levy_gen]

class levy_l_gen(rv_continuous): ...
levy_l: Final[levy_l_gen]

class logistic_gen(rv_continuous): ...
logistic: Final[logistic_gen]

class loggamma_gen(rv_continuous): ...
loggamma: Final[loggamma_gen]

class loglaplace_gen(rv_continuous): ...
loglaplace: Final[loglaplace_gen]

class lognorm_gen(rv_continuous): ...
lognorm: Final[lognorm_gen]

class gibrat_gen(rv_continuous): ...
gibrat: Final[gibrat_gen]

class maxwell_gen(rv_continuous): ...
maxwell: Final[maxwell_gen]

class mielke_gen(rv_continuous): ...
mielke: Final[mielke_gen]

class kappa4_gen(rv_continuous): ...
kappa4: Final[kappa4_gen]

class kappa3_gen(rv_continuous): ...
kappa3: Final[kappa3_gen]

class moyal_gen(rv_continuous): ...
moyal: Final[moyal_gen]

class nakagami_gen(rv_continuous): ...
nakagami: Final[nakagami_gen]

class ncx2_gen(rv_continuous): ...
ncx2: Final[ncx2_gen]

class ncf_gen(rv_continuous): ...
ncf: Final[ncf_gen]

class t_gen(rv_continuous): ...
t: Final[t_gen]

class nct_gen(rv_continuous): ...
nct: Final[nct_gen]

class pareto_gen(rv_continuous): ...
pareto: Final[pareto_gen]

class lomax_gen(rv_continuous): ...
lomax: Final[lomax_gen]

class pearson3_gen(rv_continuous): ...
pearson3: Final[pearson3_gen]

class powerlaw_gen(rv_continuous): ...
powerlaw: Final[powerlaw_gen]

class powerlognorm_gen(rv_continuous): ...
powerlognorm: Final[powerlognorm_gen]

class powernorm_gen(rv_continuous): ...
powernorm: Final[powernorm_gen]

class rdist_gen(rv_continuous): ...
rdist: Final[rdist_gen]

class rayleigh_gen(rv_continuous): ...
rayleigh: Final[rayleigh_gen]

class reciprocal_gen(rv_continuous):
    fit_note: str
loguniform: Final[reciprocal_gen]
reciprocal: Final[reciprocal_gen]

class rice_gen(rv_continuous): ...
rice: Final[rice_gen]

class irwinhall_gen(rv_continuous): ...
irwinhall: Final[irwinhall_gen]

class recipinvgauss_gen(rv_continuous): ...
recipinvgauss: Final[recipinvgauss_gen]

class semicircular_gen(rv_continuous): ...
semicircular: Final[semicircular_gen]

class skewcauchy_gen(rv_continuous): ...
skewcauchy: Final[skewcauchy_gen]

class skewnorm_gen(rv_continuous): ...
skewnorm: Final[skewnorm_gen]

class trapezoid_gen(rv_continuous): ...
@deprecated("will be removed in SciPy 1.16.0.")
class trapz_gen(trapezoid_gen): ...
trapezoid: Final[trapezoid_gen]
trapz: Final[trapz_gen]  # pyright: ignore[reportDeprecated]

class triang_gen(rv_continuous): ...
triang: Final[triang_gen]

class truncexpon_gen(rv_continuous): ...
truncexpon: Final[truncexpon_gen]

class truncnorm_gen(rv_continuous): ...
truncnorm: Final[truncnorm_gen]

class truncpareto_gen(rv_continuous): ...
truncpareto: Final[truncpareto_gen]

class tukeylambda_gen(rv_continuous): ...
tukeylambda: Final[tukeylambda_gen]

class uniform_gen(rv_continuous): ...
uniform: Final[uniform_gen]

class vonmises_gen(rv_continuous): ...
vonmises: Final[vonmises_gen]
vonmises_line: Final[vonmises_gen]

class wald_gen(invgauss_gen): ...
wald: Final[wald_gen]

class wrapcauchy_gen(rv_continuous): ...
wrapcauchy: Final[wrapcauchy_gen]

class gennorm_gen(rv_continuous): ...
gennorm: Final[gennorm_gen]

class halfgennorm_gen(rv_continuous): ...
halfgennorm: Final[halfgennorm_gen]

class crystalball_gen(rv_continuous): ...
crystalball: Final[crystalball_gen]

class argus_gen(rv_continuous): ...
argus: Final[argus_gen]

class rv_histogram(rv_continuous): ...

class studentized_range_gen(rv_continuous): ...
studentized_range: Final[studentized_range_gen]

class rel_breitwigner_gen(rv_continuous): ...
rel_breitwigner: Final[rel_breitwigner_gen]
