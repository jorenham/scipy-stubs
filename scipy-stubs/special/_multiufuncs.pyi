from ._gufuncs import (
    assoc_legendre_p_all as assoc_legendre_p_all,
    legendre_p_all as legendre_p_all,
    sph_harm_y_all as sph_harm_y_all,
    sph_legendre_p_all as sph_legendre_p_all,
)
from ._special_ufuncs import (
    assoc_legendre_p as assoc_legendre_p,
    legendre_p as legendre_p,
    sph_harm_y as sph_harm_y,
    sph_legendre_p as sph_legendre_p,
)
from scipy._typing import Untyped

class MultiUFunc:
    def __init__(self, ufunc_or_ufuncs, doc: Untyped | None = None, *, force_complex_output: bool = False, **default_kwargs): ...
    @property
    def __doc__(self) -> Untyped: ...
    def __call__(self, *args, **kwargs) -> Untyped: ...

def _(diff_n) -> Untyped: ...
