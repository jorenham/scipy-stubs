# This file is not meant for public use and will be removed in SciPy v2.0.0.

from ._basic import inv, solve
from ._decomp_schur import rsf2csf, schur
from ._decomp_svd import svd
from ._matfuncs import (
    coshm,
    cosm,
    expm,
    expm_cond,
    expm_frechet,
    fractional_matrix_power,
    funm,
    khatri_rao,
    logm,
    signm,
    sinhm,
    sinm,
    sqrtm,
    tanhm,
    tanm,
)
from ._misc import norm

__all__ = [
    "coshm",
    "cosm",
    "expm",
    "expm_cond",
    "expm_frechet",
    "fractional_matrix_power",
    "funm",
    "inv",
    "khatri_rao",
    "logm",
    "norm",
    "rsf2csf",
    "schur",
    "signm",
    "sinhm",
    "sinm",
    "solve",
    "sqrtm",
    "svd",
    "tanhm",
    "tanm",
]
