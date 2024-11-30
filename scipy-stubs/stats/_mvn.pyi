# NOTE: see `scipy/stats/mvn.pyf` for the f2py interface and `scipy/stats/mvndst.f` for the Fortran code

from typing import Final, type_check_only
from typing_extensions import LiteralString

import numpy as np
import optype.numpy as onp

@type_check_only
class _FortranObject_dkblck:
    ivls: onp.Array0D[np.int32]

@type_check_only
class _FortranFunction_mvnun:
    @staticmethod
    def __call__(
        lower: onp.ToFloatND,
        upper: onp.ToFloatND,
        means: onp.ToFloatND,
        covar: onp.ToFloatND,
        maxpts: onp.ToInt = ...,
        abseps: onp.ToFloat = 1e-6,
        releps: onp.ToFloat = 1e-6,
    ) -> tuple[float, int]: ...

@type_check_only
class _FortranFunction_mvnun_weighted:
    @staticmethod
    def __call__(
        lower: onp.ToFloatND,
        upper: onp.ToFloatND,
        means: onp.ToFloatND,
        weights: onp.ToFloatND,
        covar: onp.ToFloatND,
        maxpts: onp.ToInt = ...,
        abseps: onp.ToFloat = 1e-6,
        releps: onp.ToFloat = 1e-6,
    ) -> tuple[float, int]: ...

@type_check_only
class _FortranFunction_mvndst:
    @staticmethod
    def __call__(
        lower: onp.ToFloatND,
        upper: onp.ToFloatND,
        infin: onp.ToIntND,
        correl: onp.ToFloatND,
        maxpts: onp.ToInt = 2000,
        abseps: onp.ToFloat = 1e-6,
        releps: onp.ToFloat = 1e-6,
    ) -> tuple[float, float, int]: ...

__f2py_numpy_version__: Final[LiteralString] = ...

dkblck: Final[_FortranObject_dkblck] = ...
mvnun: Final[_FortranFunction_mvnun] = ...
mvnun_weighted: Final[_FortranFunction_mvnun_weighted] = ...
mvndst: Final[_FortranFunction_mvndst] = ...
