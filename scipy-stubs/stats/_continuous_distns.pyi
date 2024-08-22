from ._censored_data import CensoredData as CensoredData
from ._distn_infrastructure import get_distribution_names as get_distribution_names, rv_continuous as rv_continuous
from ._ksstats import kolmogn as kolmogn, kolmogni as kolmogni, kolmognp as kolmognp
from scipy import integrate as integrate, optimize as optimize
from scipy._lib._ccallback import LowLevelCallable as LowLevelCallable
from scipy._lib.doccer import (
    extend_notes_in_docstring as extend_notes_in_docstring,
    inherit_docstring_from as inherit_docstring_from,
    replace_notes_in_docstring as replace_notes_in_docstring,
)
from scipy._typing import Untyped
from scipy.interpolate import BSpline as BSpline
from scipy.optimize import root_scalar as root_scalar
from scipy.stats._warnings_errors import FitError as FitError

class ksone_gen(rv_continuous): ...

ksone: Untyped

class kstwo_gen(rv_continuous): ...

kstwo: Untyped

class kstwobign_gen(rv_continuous): ...

kstwobign: Untyped

class norm_gen(rv_continuous):
    def fit(self, data, **kwds) -> Untyped: ...

norm: Untyped

class alpha_gen(rv_continuous): ...

alpha: Untyped

class anglit_gen(rv_continuous): ...

anglit: Untyped

class arcsine_gen(rv_continuous): ...

arcsine: Untyped

class FitDataError(ValueError):
    args: Untyped
    def __init__(self, distr, lower, upper) -> None: ...

class FitSolverError(FitError):
    args: Untyped
    def __init__(self, mesg) -> None: ...

class beta_gen(rv_continuous):
    def fit(self, data, *args, **kwds) -> Untyped: ...

beta: Untyped

class betaprime_gen(rv_continuous): ...

betaprime: Untyped

class bradford_gen(rv_continuous): ...

bradford: Untyped

class burr_gen(rv_continuous): ...

burr: Untyped

class burr12_gen(rv_continuous): ...

burr12: Untyped

class fisk_gen(burr_gen): ...

fisk: Untyped

class cauchy_gen(rv_continuous): ...

cauchy: Untyped

class chi_gen(rv_continuous): ...

chi: Untyped

class chi2_gen(rv_continuous): ...

chi2: Untyped

class cosine_gen(rv_continuous): ...

cosine: Untyped

class dgamma_gen(rv_continuous): ...

dgamma: Untyped

class dweibull_gen(rv_continuous): ...

dweibull: Untyped

class expon_gen(rv_continuous):
    def fit(self, data, *args, **kwds) -> Untyped: ...

expon: Untyped

class exponnorm_gen(rv_continuous): ...

exponnorm: Untyped

class exponweib_gen(rv_continuous): ...

exponweib: Untyped

class exponpow_gen(rv_continuous): ...

exponpow: Untyped

class fatiguelife_gen(rv_continuous): ...

fatiguelife: Untyped

class foldcauchy_gen(rv_continuous): ...

foldcauchy: Untyped

class f_gen(rv_continuous): ...

f: Untyped

class foldnorm_gen(rv_continuous): ...

foldnorm: Untyped

class weibull_min_gen(rv_continuous):
    def fit(self, data, *args, **kwds) -> Untyped: ...

weibull_min: Untyped

class truncweibull_min_gen(rv_continuous): ...

truncweibull_min: Untyped

class weibull_max_gen(rv_continuous): ...

weibull_max: Untyped

class genlogistic_gen(rv_continuous): ...

genlogistic: Untyped

class genpareto_gen(rv_continuous): ...

genpareto: Untyped

class genexpon_gen(rv_continuous): ...

genexpon: Untyped

class genextreme_gen(rv_continuous): ...

genextreme: Untyped

class gamma_gen(rv_continuous):
    def fit(self, data, *args, **kwds) -> Untyped: ...

gamma: Untyped

class erlang_gen(gamma_gen):
    def fit(self, data, *args, **kwds) -> Untyped: ...

erlang: Untyped

class gengamma_gen(rv_continuous): ...

gengamma: Untyped

class genhalflogistic_gen(rv_continuous): ...

genhalflogistic: Untyped

class genhyperbolic_gen(rv_continuous): ...

genhyperbolic: Untyped

class gompertz_gen(rv_continuous): ...

gompertz: Untyped

class gumbel_r_gen(rv_continuous):
    def fit(self, data, *args, **kwds) -> Untyped: ...

gumbel_r: Untyped

class gumbel_l_gen(rv_continuous):
    def fit(self, data, *args, **kwds) -> Untyped: ...

gumbel_l: Untyped

class halfcauchy_gen(rv_continuous):
    def fit(self, data, *args, **kwds) -> Untyped: ...

halfcauchy: Untyped

class halflogistic_gen(rv_continuous):
    def fit(self, data, *args, **kwds) -> Untyped: ...

halflogistic: Untyped

class halfnorm_gen(rv_continuous):
    def fit(self, data, *args, **kwds) -> Untyped: ...

halfnorm: Untyped

class hypsecant_gen(rv_continuous): ...

hypsecant: Untyped

class gausshyper_gen(rv_continuous): ...

gausshyper: Untyped

class invgamma_gen(rv_continuous): ...

invgamma: Untyped

class invgauss_gen(rv_continuous):
    def fit(self, data, *args, **kwds) -> Untyped: ...

invgauss: Untyped

class geninvgauss_gen(rv_continuous): ...

geninvgauss: Untyped

class norminvgauss_gen(rv_continuous): ...

norminvgauss: Untyped

class invweibull_gen(rv_continuous): ...

invweibull: Untyped

class jf_skew_t_gen(rv_continuous): ...

jf_skew_t: Untyped

class johnsonsb_gen(rv_continuous): ...

johnsonsb: Untyped

class johnsonsu_gen(rv_continuous): ...

johnsonsu: Untyped

class laplace_gen(rv_continuous):
    def fit(self, data, *args, **kwds) -> Untyped: ...

laplace: Untyped

class laplace_asymmetric_gen(rv_continuous): ...

laplace_asymmetric: Untyped

class levy_gen(rv_continuous): ...

levy: Untyped

class levy_l_gen(rv_continuous): ...

levy_l: Untyped

class logistic_gen(rv_continuous):
    def fit(self, data, *args, **kwds) -> Untyped: ...

logistic: Untyped

class loggamma_gen(rv_continuous): ...

loggamma: Untyped

class loglaplace_gen(rv_continuous):
    def fit(self, data, *args, **kwds) -> Untyped: ...

loglaplace: Untyped

class lognorm_gen(rv_continuous):
    def fit(self, data, *args, **kwds) -> Untyped: ...

lognorm: Untyped

class gibrat_gen(rv_continuous): ...

gibrat: Untyped

class maxwell_gen(rv_continuous): ...

maxwell: Untyped

class mielke_gen(rv_continuous): ...

mielke: Untyped

class kappa4_gen(rv_continuous): ...

kappa4: Untyped

class kappa3_gen(rv_continuous): ...

kappa3: Untyped

class moyal_gen(rv_continuous): ...

moyal: Untyped

class nakagami_gen(rv_continuous): ...

nakagami: Untyped

class ncx2_gen(rv_continuous): ...

ncx2: Untyped

class ncf_gen(rv_continuous): ...

ncf: Untyped

class t_gen(rv_continuous): ...

t: Untyped

class nct_gen(rv_continuous): ...

nct: Untyped

class pareto_gen(rv_continuous):
    def fit(self, data, *args, **kwds) -> Untyped: ...

pareto: Untyped

class lomax_gen(rv_continuous): ...

lomax: Untyped

class pearson3_gen(rv_continuous):
    def fit(self, data, *args, **kwds) -> Untyped: ...

pearson3: Untyped

class powerlaw_gen(rv_continuous):
    def fit(self, data, *args, **kwds) -> Untyped: ...

powerlaw: Untyped

class powerlognorm_gen(rv_continuous): ...

powerlognorm: Untyped

class powernorm_gen(rv_continuous): ...

powernorm: Untyped

class rdist_gen(rv_continuous): ...

rdist: Untyped

class rayleigh_gen(rv_continuous):
    def fit(self, data, *args, **kwds) -> Untyped: ...

rayleigh: Untyped

class reciprocal_gen(rv_continuous):
    fit_note: str
    def fit(self, data, *args, **kwds) -> Untyped: ...

loguniform: Untyped
reciprocal: Untyped

class rice_gen(rv_continuous): ...

rice: Untyped

class irwinhall_gen(rv_continuous):
    def fit(self, data, *args, **kwds): ...

irwinhall: Untyped

class recipinvgauss_gen(rv_continuous): ...

recipinvgauss: Untyped

class semicircular_gen(rv_continuous): ...

semicircular: Untyped

class skewcauchy_gen(rv_continuous): ...

skewcauchy: Untyped

class skewnorm_gen(rv_continuous):
    def fit(self, data, *args, **kwds) -> Untyped: ...

skewnorm: Untyped

class trapezoid_gen(rv_continuous): ...

deprmsg: str

class trapz_gen(trapezoid_gen):
    def __call__(self, *args, **kwds) -> Untyped: ...

trapezoid: Untyped
trapz: Untyped

class _DeprecationWrapper:
    msg: Untyped
    method: Untyped
    def __init__(self, method) -> None: ...
    def __call__(self, *args, **kwargs) -> Untyped: ...

class triang_gen(rv_continuous): ...

triang: Untyped

class truncexpon_gen(rv_continuous): ...

truncexpon: Untyped

class truncnorm_gen(rv_continuous): ...

truncnorm: Untyped

class truncpareto_gen(rv_continuous):
    def fit(self, data, *args, **kwds) -> Untyped: ...

truncpareto: Untyped

class tukeylambda_gen(rv_continuous): ...

tukeylambda: Untyped

class FitUniformFixedScaleDataError(FitDataError):
    args: Untyped
    def __init__(self, ptp, fscale) -> None: ...

class uniform_gen(rv_continuous):
    def fit(self, data, *args, **kwds) -> Untyped: ...

uniform: Untyped

class vonmises_gen(rv_continuous):
    def rvs(self, *args, **kwds) -> Untyped: ...
    def expect(
        self,
        func: Untyped | None = None,
        args=(),
        loc: int = 0,
        scale: int = 1,
        lb: Untyped | None = None,
        ub: Untyped | None = None,
        conditional: bool = False,
        **kwds,
    ) -> Untyped: ...
    def fit(self, data, *args, **kwds) -> Untyped: ...

vonmises: Untyped
vonmises_line: Untyped

class wald_gen(invgauss_gen): ...

wald: Untyped

class wrapcauchy_gen(rv_continuous): ...

wrapcauchy: Untyped

class gennorm_gen(rv_continuous): ...

gennorm: Untyped

class halfgennorm_gen(rv_continuous): ...

halfgennorm: Untyped

class crystalball_gen(rv_continuous): ...

crystalball: Untyped

class argus_gen(rv_continuous): ...

argus: Untyped

class rv_histogram(rv_continuous):
    def __init__(self, histogram, *args, density: Untyped | None = None, **kwargs): ...

class studentized_range_gen(rv_continuous): ...

studentized_range: Untyped

class rel_breitwigner_gen(rv_continuous):
    def fit(self, data, *args, **kwds) -> Untyped: ...

rel_breitwigner: Untyped
pairs: Untyped
