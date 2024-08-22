from scipy._typing import Untyped

class DCSRCH:
    stage: Untyped
    ginit: Untyped
    gtest: Untyped
    gx: Untyped
    gy: Untyped
    finit: Untyped
    fx: Untyped
    fy: Untyped
    stx: Untyped
    sty: Untyped
    stmin: Untyped
    stmax: Untyped
    width: Untyped
    width1: Untyped
    ftol: Untyped
    gtol: Untyped
    xtol: Untyped
    stpmin: Untyped
    stpmax: Untyped
    phi: Untyped
    derphi: Untyped
    def __init__(self, phi, derphi, ftol, gtol, xtol, stpmin, stpmax) -> None: ...
    def __call__(self, alpha1, phi0: Untyped | None = None, derphi0: Untyped | None = None, maxiter: int = 100) -> Untyped: ...

def dcstep(stx, fx, dx, sty, fy, dy, stp, fp, dp, brackt, stpmin, stpmax) -> Untyped: ...
