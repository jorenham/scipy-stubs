from collections.abc import Callable, Mapping
from typing import Any, Concatenate, Final, Literal, TypeAlias, TypedDict, overload, type_check_only

import numpy as np
import numpy.typing as npt
import optype.numpy as onpt
from numpy._typing import _ArrayLikeFloat_co, _ArrayLikeInt
from scipy._typing import AnyBool

__all__ = ["ODR", "Data", "Model", "OdrError", "OdrStop", "OdrWarning", "Output", "RealData", "odr", "odr_error", "odr_stop"]

_Int_co: TypeAlias = np.integer[Any] | np.bool_
_Float_co: TypeAlias = np.floating[Any] | _Int_co

_VectorF8: TypeAlias = onpt.Array[tuple[int], np.float64]
_MatrixF8: TypeAlias = onpt.Array[tuple[int, int], np.float64]
_ArrayF8: TypeAlias = npt.NDArray[np.float64]
_FCN: TypeAlias = Callable[Concatenate[_VectorF8, _ArrayF8, ...], npt.NDArray[_Float_co]]

_01: TypeAlias = Literal[0, 1]  # noqa: PYI042
_012: TypeAlias = Literal[0, 1, 2]  # noqa: PYI042
_0123: TypeAlias = Literal[0, 1, 2, 3]  # noqa: PYI042

@type_check_only
class _FullOutput(TypedDict):
    delta: _VectorF8
    eps: _VectorF8
    xplus: _VectorF8
    y: _VectorF8
    res_var: float
    sum_square: float
    sum_square_delta: float
    sum_square_eps: float
    inc_condnum: float
    rel_error: float
    work: _VectorF8
    work_ind: dict[str, int]
    iwork: onpt.Array[tuple[int], np.int32]
    info: int

###

odr_error = OdrError
odr_stop = OdrStop

class OdrWarning(UserWarning): ...
class OdrError(Exception): ...
class OdrStop(Exception): ...

class Data:
    x: Final[npt.NDArray[_Float_co]]
    y: Final[_Float_co | npt.NDArray[_Float_co] | None]
    we: Final[_Float_co | npt.NDArray[_Float_co] | None]
    wd: Final[_Float_co | npt.NDArray[_Float_co] | None]
    fix: Final[npt.NDArray[_Int_co] | None]
    meta: Final[Mapping[str, object]]

    def __init__(
        self,
        /,
        x: _ArrayLikeFloat_co,
        y: _ArrayLikeFloat_co | None = None,
        we: _ArrayLikeFloat_co | None = None,
        wd: _ArrayLikeFloat_co | None = None,
        fix: _ArrayLikeInt | None = None,
        meta: Mapping[str, object] | None = None,
    ) -> None: ...
    def set_meta(self, **kwds: object) -> None: ...

class RealData(Data):
    sx: Final[npt.NDArray[_Float_co] | None]
    sy: Final[npt.NDArray[_Float_co] | None]
    covx: Final[npt.NDArray[_Float_co] | None]
    covy: Final[npt.NDArray[_Float_co] | None]

    @overload
    def __init__(
        self,
        /,
        x: _ArrayLikeFloat_co,
        y: _ArrayLikeFloat_co | None = None,
        sx: _ArrayLikeFloat_co | None = None,
        sy: _ArrayLikeFloat_co | None = None,
        covx: None = None,
        covy: None = None,
        fix: _ArrayLikeInt | None = None,
        meta: Mapping[str, object] | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self,
        /,
        x: _ArrayLikeFloat_co,
        y: _ArrayLikeFloat_co | None,
        sx: None,
        sy: _ArrayLikeFloat_co | None,
        covx: _ArrayLikeFloat_co,
        covy: None = None,
        fix: _ArrayLikeInt | None = None,
        meta: Mapping[str, object] | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self,
        /,
        x: _ArrayLikeFloat_co,
        y: _ArrayLikeFloat_co | None,
        sx: _ArrayLikeFloat_co | None,
        sy: None,
        covx: None,
        covy: _ArrayLikeFloat_co,
        fix: _ArrayLikeInt | None = None,
        meta: Mapping[str, object] | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self,
        /,
        x: _ArrayLikeFloat_co,
        y: _ArrayLikeFloat_co | None,
        sx: None,
        sy: None,
        covx: _ArrayLikeFloat_co,
        covy: _ArrayLikeFloat_co,
        fix: _ArrayLikeInt | None = None,
        meta: Mapping[str, object] | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self,
        /,
        x: _ArrayLikeFloat_co,
        y: _ArrayLikeFloat_co | None = None,
        sx: None = None,
        sy: _ArrayLikeFloat_co | None = None,
        *,
        covx: _ArrayLikeFloat_co,
        covy: None = None,
        fix: _ArrayLikeInt | None = None,
        meta: Mapping[str, object] | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self,
        /,
        x: _ArrayLikeFloat_co,
        y: _ArrayLikeFloat_co | None = None,
        sx: _ArrayLikeFloat_co | None = None,
        sy: None = None,
        *,
        covx: None = None,
        covy: _ArrayLikeFloat_co,
        fix: _ArrayLikeInt | None = None,
        meta: Mapping[str, object] | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self,
        /,
        x: _ArrayLikeFloat_co,
        y: _ArrayLikeFloat_co | None = None,
        sx: None = None,
        sy: None = None,
        *,
        covx: _ArrayLikeFloat_co,
        covy: _ArrayLikeFloat_co,
        fix: _ArrayLikeInt | None = None,
        meta: Mapping[str, object] | None = None,
    ) -> None: ...

class Model:
    fcn: Final[_FCN]
    fjacb: Final[_FCN]
    fjacd: Final[_FCN]
    extra_args: Final[tuple[object, ...]]
    covx: Final[npt.NDArray[_Float_co] | None]
    implicit: Final[AnyBool]
    meta: Final[Mapping[str, object]]

    def __init__(
        self,
        /,
        fcn: _FCN,
        fjacb: _FCN | None = None,
        fjacd: _FCN | None = None,
        extra_args: tuple[object, ...] | None = None,
        estimate: _ArrayLikeFloat_co | None = None,
        implicit: AnyBool = 0,
        meta: Mapping[str, object] | None = None,
    ) -> None: ...
    def set_meta(self, **kwds: object) -> None: ...

class Output:
    beta: Final[onpt.Array[tuple[int], _Float_co]]
    sd_beta: Final[onpt.Array[tuple[int], _Float_co]]
    cov_beta: Final[onpt.Array[tuple[int], _Float_co]]
    stopreason: Final[list[str]]

    def __init__(self, /, output: npt.NDArray[_Float_co]) -> None: ...
    def pprint(self, /) -> None: ...

class ODR:
    data: Final[Data]
    model: Final[Model]
    beta0: Final[onpt.Array[tuple[int], _Float_co]]
    delta0: Final[onpt.Array[tuple[int], _Float_co] | None]
    ifixx: Final[onpt.Array[tuple[int], np.int32] | None]
    ifixb: Final[onpt.Array[tuple[int], np.int32] | None]
    errfile: Final[str | None]
    rptfile: Final[str | None]
    ndigit: Final[int | None]
    taufac: Final[float | None]
    sstol: Final[float | None]
    partol: Final[float | None]
    stpb: Final[onpt.Array[tuple[int], _Float_co] | None]
    stpd: Final[onpt.Array[tuple[int], _Float_co] | None]
    sclb: Final[onpt.Array[tuple[int], _Float_co] | None]
    scld: Final[onpt.Array[tuple[int], _Float_co] | None]

    job: int | None
    iprint: int | None
    maxit: int | None
    work: onpt.Array[tuple[int], _Float_co] | None
    iwork: onpt.Array[tuple[int], _Int_co] | None
    output: Output | None

    def __init__(
        self,
        /,
        data: Data,
        model: Model,
        beta0: _ArrayLikeFloat_co | None = None,
        delta0: _ArrayLikeFloat_co | None = None,
        ifixb: _ArrayLikeInt | None = None,
        ifixx: _ArrayLikeInt | None = None,
        job: int | None = None,
        iprint: int | None = None,
        errfile: str | None = None,
        rptfile: str | None = None,
        ndigit: int | None = None,
        taufac: float | None = None,
        sstol: float | None = None,
        partol: float | None = None,
        maxit: int | None = None,
        stpb: _ArrayLikeFloat_co | None = None,
        stpd: _ArrayLikeFloat_co | None = None,
        sclb: _ArrayLikeFloat_co | None = None,
        scld: _ArrayLikeFloat_co | None = None,
        work: _ArrayLikeFloat_co | None = None,
        iwork: _ArrayLikeInt | None = None,
        overwrite: bool = False,
    ) -> None: ...
    def set_job(
        self,
        /,
        fit_type: _012 | None = None,
        deriv: _0123 | None = None,
        var_calc: _012 | None = None,
        del_init: _01 | None = None,
        restart: _01 | None = None,
    ) -> None: ...
    def set_iprint(
        self,
        /,
        init: _012 | None = None,
        so_init: _012 | None = None,
        iter: _012 | None = None,
        so_iter: _012 | None = None,
        iter_step: _012 | None = None,
        final: _012 | None = None,
        so_final: _012 | None = None,
    ) -> None: ...
    def run(self, /) -> Output: ...
    def restart(self, /, iter: int | None = None) -> Output: ...

@overload
def odr(
    fcn: _FCN,
    beta0: _ArrayLikeFloat_co,
    y: _ArrayLikeFloat_co,
    x: _ArrayLikeFloat_co,
    we: _ArrayLikeFloat_co | None = None,
    wd: _ArrayLikeFloat_co | None = None,
    fjacb: _FCN | None = None,
    fjacd: _FCN | None = None,
    extra_args: tuple[object, ...] | None = None,
    ifixx: _ArrayLikeInt | None = None,
    ifixb: _ArrayLikeInt | None = None,
    job: int = 0,
    iprint: int = 0,
    errfile: str | None = None,
    rptfile: str | None = None,
    ndigit: int = 0,
    taufac: float = 0.0,
    sstol: float = -1.0,
    partol: float = -1.0,
    maxit: int = -1,
    stpb: _ArrayLikeFloat_co | None = None,
    stpd: _ArrayLikeFloat_co | None = None,
    sclb: _ArrayLikeFloat_co | None = None,
    scld: _ArrayLikeFloat_co | None = None,
    work: _ArrayLikeFloat_co | None = None,
    iwork: _ArrayLikeInt | None = None,
    full_output: Literal[False, 0] = 0,
) -> tuple[_VectorF8, _VectorF8, _MatrixF8]: ...
@overload
def odr(
    fcn: _FCN,
    beta0: _ArrayLikeFloat_co,
    y: _ArrayLikeFloat_co,
    x: _ArrayLikeFloat_co,
    we: _ArrayLikeFloat_co | None = None,
    wd: _ArrayLikeFloat_co | None = None,
    fjacb: _FCN | None = None,
    fjacd: _FCN | None = None,
    extra_args: tuple[object, ...] | None = None,
    ifixx: _ArrayLikeInt | None = None,
    ifixb: _ArrayLikeInt | None = None,
    job: int = 0,
    iprint: int = 0,
    errfile: str | None = None,
    rptfile: str | None = None,
    ndigit: int = 0,
    taufac: float = 0.0,
    sstol: float = -1.0,
    partol: float = -1.0,
    maxit: int = -1,
    stpb: _ArrayLikeFloat_co | None = None,
    stpd: _ArrayLikeFloat_co | None = None,
    sclb: _ArrayLikeFloat_co | None = None,
    scld: _ArrayLikeFloat_co | None = None,
    work: _ArrayLikeFloat_co | None = None,
    iwork: _ArrayLikeInt | None = None,
    *,
    full_output: Literal[True, 1],
) -> tuple[_VectorF8, _VectorF8, _MatrixF8, _FullOutput]: ...
