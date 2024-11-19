# NOTE: see `scipy/stats/mvn.pyf` for the f2py interface and `scipy/stats/mvndst.f` for the Fortran code

from typing import Final, type_check_only
from typing_extensions import LiteralString

import numpy as np
import optype.numpy as onp
from numpy._typing import _ArrayLikeFloat_co, _ArrayLikeInt_co

@type_check_only
class _FortranObject_dkblck:
    ivls: np.ndarray[tuple[()], np.dtype[np.int32]]

@type_check_only
class _FortranFunction_mvnun:
    @staticmethod
    def __call__(
        lower: _ArrayLikeFloat_co,
        upper: _ArrayLikeFloat_co,
        means: _ArrayLikeFloat_co,
        covar: _ArrayLikeFloat_co,
        maxpts: onp.ToInt = ...,
        abseps: onp.ToFloat = 1e-6,
        releps: onp.ToFloat = 1e-6,
    ) -> tuple[float, int]: ...

@type_check_only
class _FortranFunction_mvnun_weighted:
    @staticmethod
    def __call__(
        lower: _ArrayLikeFloat_co,
        upper: _ArrayLikeFloat_co,
        means: _ArrayLikeFloat_co,
        weights: _ArrayLikeFloat_co,
        covar: _ArrayLikeFloat_co,
        maxpts: onp.ToInt = ...,
        abseps: onp.ToFloat = 1e-6,
        releps: onp.ToFloat = 1e-6,
    ) -> tuple[float, int]: ...

@type_check_only
class _FortranFunction_mvndst:
    @staticmethod
    def __call__(
        lower: _ArrayLikeFloat_co,
        upper: _ArrayLikeFloat_co,
        infin: _ArrayLikeInt_co,
        correl: _ArrayLikeFloat_co,
        maxpts: onp.ToInt = 2000,
        abseps: onp.ToFloat = 1e-6,
        releps: onp.ToFloat = 1e-6,
    ) -> tuple[float, float, int]: ...

__f2py_numpy_version__: Final[LiteralString] = ...

dkblck: Final[_FortranObject_dkblck] = ...
mvnun: Final[_FortranFunction_mvnun] = ...
mvnun_weighted: Final[_FortranFunction_mvnun_weighted] = ...
mvndst: Final[_FortranFunction_mvndst] = ...
