from scipy._typing import Untyped

class DegenerateDataWarning(RuntimeWarning):
    args: Untyped
    def __init__(self, msg: Untyped | None = None): ...

class ConstantInputWarning(DegenerateDataWarning):
    args: Untyped
    def __init__(self, msg: Untyped | None = None): ...

class NearConstantInputWarning(DegenerateDataWarning):
    args: Untyped
    def __init__(self, msg: Untyped | None = None): ...

class FitError(RuntimeError):
    args: Untyped
    def __init__(self, msg: Untyped | None = None): ...
