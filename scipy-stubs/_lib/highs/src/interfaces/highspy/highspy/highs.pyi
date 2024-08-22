from .highs_bindings import (
    HIGHS_VERSION_MAJOR as HIGHS_VERSION_MAJOR,
    HIGHS_VERSION_MINOR as HIGHS_VERSION_MINOR,
    HIGHS_VERSION_PATCH as HIGHS_VERSION_PATCH,
    BasisValidity as BasisValidity,
    CallbackTuple as CallbackTuple,
    HessianFormat as HessianFormat,
    HighsBasis as HighsBasis,
    HighsBasisStatus as HighsBasisStatus,
    HighsHessian as HighsHessian,
    HighsInfo as HighsInfo,
    HighsLogType as HighsLogType,
    HighsLp as HighsLp,
    HighsModel as HighsModel,
    HighsModelStatus as HighsModelStatus,
    HighsOptions as HighsOptions,
    HighsSolution as HighsSolution,
    HighsSparseMatrix as HighsSparseMatrix,
    HighsStatus as HighsStatus,
    HighsVarType as HighsVarType,
    MatrixFormat as MatrixFormat,
    ObjSense as ObjSense,
    SolutionStatus as SolutionStatus,
    _Highs,
    kHighsInf as kHighsInf,
)

class Highs(_Highs):
    def __init__(self) -> None: ...
    def setLogCallback(self, func, callback_data): ...
