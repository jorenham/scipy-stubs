# This file is not meant for public use and will be removed in SciPy v2.0.0.
from typing_extensions import deprecated

from ._quadpack_py import dblquad, nquad, quad, tplquad

__all__ = ["IntegrationWarning", "dblquad", "nquad", "quad", "tplquad"]

@deprecated("will be removed in SciPy v2.0.0")
class IntegrationWarning(UserWarning): ...
