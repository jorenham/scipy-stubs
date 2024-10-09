# This file is not meant for public use and will be removed in SciPy v2.0.0.

from ._decomp_qr import qr, qr_multiply, rq
from .lapack import get_lapack_funcs

__all__ = ["get_lapack_funcs", "qr", "qr_multiply", "rq"]
