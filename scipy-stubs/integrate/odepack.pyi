# This file is not meant for public use and will be removed in SciPy v2.0.0.
from typing_extensions import deprecated

from ._odepack_py import odeint

__all__ = ["ODEintWarning", "odeint"]

@deprecated("will be removed in SciPy 2.0.0.")
class ODEintWarning(Warning): ...
