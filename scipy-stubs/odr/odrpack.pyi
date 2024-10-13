# This module is not meant for public use and will be removed in SciPy v2.0.0.
from typing_extensions import deprecated, override

__all__ = ["ODR", "Data", "Model", "OdrError", "OdrStop", "OdrWarning", "Output", "RealData", "odr", "odr_error", "odr_stop"]

@deprecated("will be removed in SciPy v2.0.0")
class OdrWarning(UserWarning): ...

@deprecated("will be removed in SciPy v2.0.0")
class OdrError(Exception): ...

@deprecated("will be removed in SciPy v2.0.0")
class OdrStop(Exception): ...

@deprecated("will be removed in SciPy v2.0.0")
class Data:
    def __init__(
        self,
        x: object,
        y: object = ...,
        we: object = ...,
        wd: object = ...,
        fix: object = ...,
        meta: object = ...,
    ) -> None: ...
    def set_meta(self, **kwds: object) -> None: ...
    def __getattr__(self, attr: object) -> object: ...

@deprecated("will be removed in SciPy v2.0.0")
class RealData(Data):
    def __init__(
        self,
        x: object,
        y: object = ...,
        sx: object = ...,
        sy: object = ...,
        covx: object = ...,
        covy: object = ...,
        fix: object = ...,
        meta: object = ...,
    ) -> None: ...
    @override
    def __getattr__(self, attr: object) -> object: ...

@deprecated("will be removed in SciPy v2.0.0")
class Model:
    def __init__(
        self,
        fcn: object,
        fjacb: object = ...,
        fjacd: object = ...,
        extra_args: object = ...,
        estimate: object = ...,
        implicit: object = ...,
        meta: object = ...,
    ) -> None: ...
    def set_meta(self, **kwds: object) -> None: ...
    def __getattr__(self, attr: object) -> object: ...

@deprecated("will be removed in SciPy v2.0.0")
class Output:
    def __init__(self, output: object) -> None: ...
    def pprint(self) -> None: ...

@deprecated("will be removed in SciPy v2.0.0")
class ODR:
    def __init__(
        self,
        data: object,
        model: object,
        beta0: object = ...,
        delta0: object = ...,
        ifixb: object = ...,
        ifixx: object = ...,
        job: object = ...,
        iprint: object = ...,
        errfile: object = ...,
        rptfile: object = ...,
        ndigit: object = ...,
        taufac: object = ...,
        sstol: object = ...,
        partol: object = ...,
        maxit: object = ...,
        stpb: object = ...,
        stpd: object = ...,
        sclb: object = ...,
        scld: object = ...,
        work: object = ...,
        iwork: object = ...,
        overwrite: object = ...,
    ) -> None: ...
    def set_job(
        self,
        fit_type: object = ...,
        deriv: object = ...,
        var_calc: object = ...,
        del_init: object = ...,
        restart: object = ...,
    ) -> None: ...
    def set_iprint(
        self,
        init: object = ...,
        so_init: object = ...,
        iter: object = ...,
        so_iter: object = ...,
        iter_step: object = ...,
        final: object = ...,
        so_final: object = ...,
    ) -> None: ...
    def run(self) -> object: ...
    def restart(self, iter: object = ...) -> object: ...

odr: object
odr_error = OdrError
odr_stop = OdrStop
