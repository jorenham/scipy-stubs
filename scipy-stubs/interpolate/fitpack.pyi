# This module is not meant for public use and will be removed in SciPy v2.0.0.
from typing_extensions import deprecated

__all__ = [
    "BSpline",
    "bisplev",
    "bisplrep",
    "insert",
    "spalde",
    "splantider",
    "splder",
    "splev",
    "splint",
    "splprep",
    "splrep",
    "sproot",
]

@deprecated("will be removed in SciPy v2.0.0")
def splprep(
    x: object,
    w: object = ...,
    u: object = ...,
    ub: object = ...,
    ue: object = ...,
    k: object = ...,
    task: object = ...,
    s: object = ...,
    t: object = ...,
    full_output: object = ...,
    nest: object = ...,
    per: object = ...,
    quiet: object = ...,
) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def splrep(
    x: object,
    y: object,
    w: object = ...,
    xb: object = ...,
    xe: object = ...,
    k: object = ...,
    task: object = ...,
    s: object = ...,
    t: object = ...,
    full_output: object = ...,
    per: object = ...,
    quiet: object = ...,
) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def splev(x: object, tck: object, der: object = ..., ext: object = ...) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def splint(a: object, b: object, tck: object, full_output: object = ...) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def sproot(tck: object, mest: object = ...) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def spalde(x: object, tck: object) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def insert(x: object, tck: object, m: object = ..., per: object = ...) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def splder(tck: object, n: object = ...) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def splantider(tck: object, n: object = ...) -> object: ...

# bsplines
@deprecated("will be removed in SciPy v2.0.0")
class BSpline:
    def __init__(self, t: object, c: object, k: object, extrapolate: object = ..., axis: object = ...) -> None: ...
    @classmethod
    def construct_fast(cls, t: object, c: object, k: object, extrapolate: object = ..., axis: object = ...) -> object: ...
    @property
    def tck(self) -> object: ...
    @classmethod
    def basis_element(cls, t: object, extrapolate: object = ...) -> object: ...
    @classmethod
    def design_matrix(cls, x: object, t: object, k: object, extrapolate: object = ...) -> object: ...
    def __call__(self, x: object, nu: object = ..., extrapolate: object = ...) -> object: ...
    def derivative(self, nu: object = ...) -> object: ...
    def antiderivative(self, nu: object = ...) -> object: ...
    def integrate(self, a: object, b: object, extrapolate: object = ...) -> object: ...
    @classmethod
    def from_power_basis(cls, pp: object, bc_type: object = ...) -> object: ...
    def insert_knot(self, x: object, m: object = ...) -> object: ...

# fitpack_impl
@deprecated("will be removed in SciPy v2.0.0")
def bisplrep(
    x: object,
    y: object,
    z: object,
    w: object = ...,
    xb: object = ...,
    xe: object = ...,
    yb: object = ...,
    ye: object = ...,
    kx: object = ...,
    ky: object = ...,
    task: object = ...,
    s: object = ...,
    eps: object = ...,
    tx: object = ...,
    ty: object = ...,
    full_output: object = ...,
    nxest: object = ...,
    nyest: object = ...,
    quiet: object = ...,
) -> object: ...
def bisplev(x: object, y: object, tck: object, dx: object = ..., dy: object = ...) -> object: ...
