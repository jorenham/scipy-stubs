from . import _basic, _ellip_harm, _lambertw, _logsumexp, _orthogonal, _sf_error, _spfun_stats, _spherical_bessel, _ufuncs
from ._basic import *
from ._ellip_harm import *
from ._lambertw import *
from ._logsumexp import *
from ._orthogonal import *
from ._sf_error import *
from ._spfun_stats import *
from ._spherical_bessel import *
from ._ufuncs import *

__all__: list[str] = [
    "c_roots",
    "cg_roots",
    "h_roots",
    "he_roots",
    "j_roots",
    "js_roots",
    "l_roots",
    "la_roots",
    "p_roots",
    "ps_roots",
    "s_roots",
    "t_roots",
    "ts_roots",
    "u_roots",
    "us_roots",
]
__all__ += _basic.__all__
__all__ += _ellip_harm.__all__
__all__ += _lambertw.__all__
__all__ += _logsumexp.__all__
__all__ += _orthogonal.__all__
__all__ += _sf_error.__all__
__all__ += _spfun_stats.__all__
__all__ += _spherical_bessel.__all__
__all__ += _ufuncs.__all__

p_roots = roots_legendre
t_roots = roots_chebyt
u_roots = roots_chebyu
c_roots = roots_chebyc
s_roots = roots_chebys
j_roots = roots_jacobi
l_roots = roots_laguerre
la_roots = roots_genlaguerre
h_roots = roots_hermite
he_roots = roots_hermitenorm
cg_roots = roots_gegenbauer
ps_roots = roots_sh_legendre
ts_roots = roots_sh_chebyt
us_roots = roots_sh_chebyu
js_roots = roots_sh_jacobi
