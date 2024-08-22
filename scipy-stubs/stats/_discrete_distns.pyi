from ._distn_infrastructure import get_distribution_names as get_distribution_names, rv_discrete as rv_discrete
from scipy import special as special
from scipy._lib._util import rng_integers as rng_integers
from scipy._typing import Untyped
from scipy.interpolate import interp1d as interp1d
from scipy.special import betaln as betaln, entr as entr, logsumexp as logsumexp, zeta as zeta

class binom_gen(rv_discrete): ...

binom: Untyped

class bernoulli_gen(binom_gen): ...

bernoulli: Untyped

class betabinom_gen(rv_discrete): ...

betabinom: Untyped

class nbinom_gen(rv_discrete): ...

nbinom: Untyped

class betanbinom_gen(rv_discrete): ...

betanbinom: Untyped

class geom_gen(rv_discrete): ...

geom: Untyped

class hypergeom_gen(rv_discrete): ...

hypergeom: Untyped

class nhypergeom_gen(rv_discrete): ...

nhypergeom: Untyped

class logser_gen(rv_discrete): ...

logser: Untyped

class poisson_gen(rv_discrete): ...

poisson: Untyped

class planck_gen(rv_discrete): ...

planck: Untyped

class boltzmann_gen(rv_discrete): ...

boltzmann: Untyped

class randint_gen(rv_discrete): ...

randint: Untyped

class zipf_gen(rv_discrete): ...

zipf: Untyped

class zipfian_gen(rv_discrete): ...

zipfian: Untyped

class dlaplace_gen(rv_discrete): ...

dlaplace: Untyped

class skellam_gen(rv_discrete): ...

skellam: Untyped

class yulesimon_gen(rv_discrete): ...

yulesimon: Untyped

class _nchypergeom_gen(rv_discrete):
    rvs_name: Untyped
    dist: Untyped

class nchypergeom_fisher_gen(_nchypergeom_gen):
    rvs_name: str
    dist: Untyped

nchypergeom_fisher: Untyped

class nchypergeom_wallenius_gen(_nchypergeom_gen):
    rvs_name: str
    dist: Untyped

nchypergeom_wallenius: Untyped
pairs: Untyped
