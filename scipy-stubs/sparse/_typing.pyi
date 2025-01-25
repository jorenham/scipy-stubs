# NOTE(scipy-stubs): This ia a module only exists `if typing.TYPE_CHECKING: ...`
from typing import Literal, TypeAlias
from typing_extensions import TypeAliasType, Unpack

import numpy as np
import optype as op
import optype.numpy as onp
from optype.numpy.compat import complexfloating as CFloating, integer as Integer

__all__ = (
    "CFloating",
    "Floating",
    "Index1D",
    "Integer",
    "Numeric",
    "SPFormat",
    "ToShape1D",
    "ToShape1D",
    "ToShape2D",
    "ToShapeMin1D",
    "ToShapeMin3D",
)

###

# NOTE: The `TypeAliasType`s are used to avoid long error messages.
Floating = TypeAliasType("Floating", np.float32 | np.float64 | np.longdouble)
# NOTE: This (almost always) matches `scipy.sparse._sputils.supported_dtypes`
Numeric = TypeAliasType("Numeric", np.bool_ | Integer | Floating | CFloating)

Index1D: TypeAlias = onp.Array1D[np.int32 | np.int64]

ToShape1D: TypeAlias = tuple[op.CanIndex]  # ndim == 1
ToShape2D: TypeAlias = tuple[op.CanIndex, op.CanIndex]  # ndim == 2
ToShapeMin1D: TypeAlias = tuple[op.CanIndex, Unpack[tuple[op.CanIndex, ...]]]  # ndim >= 1
ToShapeMin3D: TypeAlias = tuple[op.CanIndex, op.CanIndex, op.CanIndex, Unpack[tuple[op.CanIndex, ...]]]  # ndim >= 2

SPFormat: TypeAlias = Literal["bsr", "coo", "csc", "csr", "dia", "dok", "lil"]
