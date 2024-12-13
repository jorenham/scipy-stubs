# pyright: reportUnannotatedClassAttribute=false

# needed (once) for `numpy>=2.2.0`
# mypy: disable-error-code="overload-overlap"

from typing import Generic, overload
from typing_extensions import Self, TypeVar

import numpy as np
import optype.numpy as onp
from scipy._typing import Untyped
from ._bsr import bsr_matrix
from ._coo import coo_matrix
from ._csc import csc_matrix
from ._csr import csr_matrix
from ._dia import dia_matrix
from ._dok import dok_matrix
from ._lil import lil_matrix
from ._typing import Scalar, SPFormat, ToShape2D

_SCT_co = TypeVar("_SCT_co", bound=Scalar, default=Scalar, covariant=True)

###

class spmatrix(Generic[_SCT_co]):
    @property
    def _bsr_container(self, /) -> bsr_matrix[_SCT_co]: ...
    @property
    def _coo_container(self, /) -> coo_matrix[_SCT_co]: ...
    @property
    def _csc_container(self, /) -> csc_matrix[_SCT_co]: ...
    @property
    def _csr_container(self, /) -> csr_matrix[_SCT_co]: ...
    @property
    def _dia_container(self, /) -> dia_matrix[_SCT_co]: ...
    @property
    def _dok_container(self, /) -> dok_matrix[_SCT_co]: ...
    @property
    def _lil_container(self, /) -> lil_matrix[_SCT_co]: ...

    #
    @property
    def shape(self, /) -> tuple[int, int]: ...
    def get_shape(self, /) -> tuple[int, int]: ...
    def set_shape(self, /, shape: ToShape2D) -> None: ...

    #
    def __mul__(self, other: Untyped, /) -> Untyped: ...
    def __rmul__(self, other: Untyped, /) -> Untyped: ...
    def __pow__(self, power: Untyped, /) -> Untyped: ...

    #
    def getmaxprint(self, /) -> int: ...
    def getformat(self, /) -> SPFormat: ...
    # NOTE: `axis` is only supported by `{coo,csc,csr,lil}_matrix`
    def getnnz(self, /, axis: None = None) -> int: ...
    def getH(self, /) -> Self: ...
    def getcol(self, /, j: onp.ToJustInt) -> csc_matrix[_SCT_co]: ...
    def getrow(self, /, i: onp.ToJustInt) -> csr_matrix[_SCT_co]: ...

    # NOTE: mypy reports a false positive for overlapping overloads
    @overload
    def asfptype(self: spmatrix[np.bool_ | np.int8 | np.int16 | np.uint8 | np.uint16], /) -> spmatrix[np.float32]: ...
    @overload
    def asfptype(self: spmatrix[np.int32 | np.int64 | np.uint32 | np.uint64], /) -> spmatrix[np.float64]: ...
    @overload
    def asfptype(self, /) -> Self: ...
