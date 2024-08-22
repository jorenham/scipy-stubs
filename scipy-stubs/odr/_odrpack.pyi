from scipy._typing import Untyped

odr: Untyped

class OdrWarning(UserWarning): ...
class OdrError(Exception): ...
class OdrStop(Exception): ...

odr_error = OdrError
odr_stop = OdrStop

class Data:
    x: Untyped
    y: Untyped
    we: Untyped
    wd: Untyped
    fix: Untyped
    meta: Untyped
    def __init__(
        self,
        x,
        y: Untyped | None = None,
        we: Untyped | None = None,
        wd: Untyped | None = None,
        fix: Untyped | None = None,
        meta: Untyped | None = None,
    ): ...
    def set_meta(self, **kwds): ...
    def __getattr__(self, attr) -> Untyped: ...

class RealData(Data):
    x: Untyped
    y: Untyped
    sx: Untyped
    sy: Untyped
    covx: Untyped
    covy: Untyped
    fix: Untyped
    meta: Untyped
    def __init__(
        self,
        x,
        y: Untyped | None = None,
        sx: Untyped | None = None,
        sy: Untyped | None = None,
        covx: Untyped | None = None,
        covy: Untyped | None = None,
        fix: Untyped | None = None,
        meta: Untyped | None = None,
    ): ...
    def __getattr__(self, attr) -> Untyped: ...

class Model:
    fcn: Untyped
    fjacb: Untyped
    fjacd: Untyped
    extra_args: Untyped
    estimate: Untyped
    implicit: Untyped
    meta: Untyped
    def __init__(
        self,
        fcn,
        fjacb: Untyped | None = None,
        fjacd: Untyped | None = None,
        extra_args: Untyped | None = None,
        estimate: Untyped | None = None,
        implicit: int = 0,
        meta: Untyped | None = None,
    ): ...
    def set_meta(self, **kwds): ...
    def __getattr__(self, attr) -> Untyped: ...

class Output:
    beta: Untyped
    sd_beta: Untyped
    cov_beta: Untyped
    stopreason: Untyped
    def __init__(self, output) -> None: ...
    def pprint(self): ...

class ODR:
    data: Untyped
    model: Untyped
    beta0: Untyped
    delta0: Untyped
    ifixx: Untyped
    ifixb: Untyped
    job: Untyped
    iprint: Untyped
    errfile: Untyped
    rptfile: Untyped
    ndigit: Untyped
    taufac: Untyped
    sstol: Untyped
    partol: Untyped
    maxit: Untyped
    stpb: Untyped
    stpd: Untyped
    sclb: Untyped
    scld: Untyped
    work: Untyped
    iwork: Untyped
    output: Untyped
    def __init__(
        self,
        data,
        model,
        beta0: Untyped | None = None,
        delta0: Untyped | None = None,
        ifixb: Untyped | None = None,
        ifixx: Untyped | None = None,
        job: Untyped | None = None,
        iprint: Untyped | None = None,
        errfile: Untyped | None = None,
        rptfile: Untyped | None = None,
        ndigit: Untyped | None = None,
        taufac: Untyped | None = None,
        sstol: Untyped | None = None,
        partol: Untyped | None = None,
        maxit: Untyped | None = None,
        stpb: Untyped | None = None,
        stpd: Untyped | None = None,
        sclb: Untyped | None = None,
        scld: Untyped | None = None,
        work: Untyped | None = None,
        iwork: Untyped | None = None,
        overwrite: bool = False,
    ): ...
    def set_job(
        self,
        fit_type: Untyped | None = None,
        deriv: Untyped | None = None,
        var_calc: Untyped | None = None,
        del_init: Untyped | None = None,
        restart: Untyped | None = None,
    ): ...
    def set_iprint(
        self,
        init: Untyped | None = None,
        so_init: Untyped | None = None,
        iter: Untyped | None = None,
        so_iter: Untyped | None = None,
        iter_step: Untyped | None = None,
        final: Untyped | None = None,
        so_final: Untyped | None = None,
    ): ...
    def run(self) -> Untyped: ...
    def restart(self, iter: Untyped | None = None) -> Untyped: ...
