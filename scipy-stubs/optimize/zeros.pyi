# This file is not meant for public use and will be removed in SciPy v2.0.0.

from typing_extensions import deprecated

__all__ = ["RootResults", "bisect", "brenth", "brentq", "newton", "ridder", "toms748"]

@deprecated("will be removed in SciPy v2.0.0")
class RootResults:
    def __init__(self, root: object, iterations: object, function_calls: object, flag: object, method: object) -> None: ...

@deprecated("will be removed in SciPy v2.0.0")
def newton(
    func: object,
    x0: object,
    fprime: object = ...,
    args: object = ...,
    tol: object = ...,
    maxiter: object = ...,
    fprime2: object = ...,
    x1: object = ...,
    rtol: object = ...,
    full_output: object = ...,
    disp: object = ...,
) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def bisect(
    f: object,
    a: object,
    b: object,
    args: object = ...,
    xtol: object = ...,
    rtol: object = ...,
    maxiter: object = ...,
    full_output: object = ...,
    disp: object = ...,
) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def ridder(
    f: object,
    a: object,
    b: object,
    args: object = ...,
    xtol: object = ...,
    rtol: object = ...,
    maxiter: object = ...,
    full_output: object = ...,
    disp: object = ...,
) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def brentq(
    f: object,
    a: object,
    b: object,
    args: object = ...,
    xtol: object = ...,
    rtol: object = ...,
    maxiter: object = ...,
    full_output: object = ...,
    disp: object = ...,
) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def brenth(
    f: object,
    a: object,
    b: object,
    args: object = ...,
    xtol: object = ...,
    rtol: object = ...,
    maxiter: object = ...,
    full_output: object = ...,
    disp: object = ...,
) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def toms748(
    f: object,
    a: object,
    b: object,
    args: object = ...,
    k: object = ...,
    xtol: object = ...,
    rtol: object = ...,
    maxiter: object = ...,
    full_output: object = ...,
    disp: object = ...,
) -> object: ...
