from typing import ClassVar, Final

from ._distn_infrastructure import rv_discrete

__all__ = [
    "bernoulli",
    "betabinom",
    "betanbinom",
    "binom",
    "boltzmann",
    "dlaplace",
    "geom",
    "hypergeom",
    "logser",
    "nbinom",
    "nchypergeom_fisher",
    "nchypergeom_wallenius",
    "nhypergeom",
    "planck",
    "poisson",
    "randint",
    "skellam",
    "yulesimon",
    "zipf",
    "zipfian",
]

class binom_gen(rv_discrete): ...
class bernoulli_gen(binom_gen): ...
class betabinom_gen(rv_discrete): ...
class nbinom_gen(rv_discrete): ...
class betanbinom_gen(rv_discrete): ...
class geom_gen(rv_discrete): ...
class hypergeom_gen(rv_discrete): ...
class nhypergeom_gen(rv_discrete): ...
class logser_gen(rv_discrete): ...
class poisson_gen(rv_discrete): ...
class planck_gen(rv_discrete): ...
class boltzmann_gen(rv_discrete): ...
class randint_gen(rv_discrete): ...
class zipf_gen(rv_discrete): ...
class zipfian_gen(rv_discrete): ...
class dlaplace_gen(rv_discrete): ...
class skellam_gen(rv_discrete): ...
class yulesimon_gen(rv_discrete): ...
class _nchypergeom_gen(rv_discrete):
    rvs_name: ClassVar = None
    dist: ClassVar = None
class nchypergeom_fisher_gen(_nchypergeom_gen):
    rvs_name: ClassVar = "rvs_fisher"
    dist: ClassVar[type] = ...  # scipy.stats._biasedurn._PyFishersNCHypergeometric
class nchypergeom_wallenius_gen(_nchypergeom_gen):
    rvs_name: ClassVar = "rvs_wallenius"
    dist: ClassVar[type] = ...  # scipy.stats._biasedurn._PyWalleniusNCHypergeometric

binom: Final[binom_gen]
bernoulli: Final[bernoulli_gen]
betabinom: Final[betabinom_gen]
nbinom: Final[nbinom_gen]
betanbinom: Final[betanbinom_gen]
geom: Final[geom_gen]
hypergeom: Final[hypergeom_gen]
nhypergeom: Final[nhypergeom_gen]
logser: Final[logser_gen]
poisson: Final[poisson_gen]
planck: Final[planck_gen]
boltzmann: Final[boltzmann_gen]
randint: Final[randint_gen]
zipf: Final[zipf_gen]
zipfian: Final[zipfian_gen]
dlaplace: Final[dlaplace_gen]
skellam: Final[skellam_gen]
yulesimon: Final[yulesimon_gen]
nchypergeom_fisher: Final[nchypergeom_fisher_gen]
nchypergeom_wallenius: Final[nchypergeom_wallenius_gen]
