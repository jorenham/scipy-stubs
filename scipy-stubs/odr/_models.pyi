from scipy._typing import Untyped
from scipy.odr._odrpack import Model as Model

class _MultilinearModel(Model):
    def __init__(self) -> None: ...

multilinear: Untyped

def polynomial(order) -> Untyped: ...

class _ExponentialModel(Model):
    def __init__(self) -> None: ...

exponential: Untyped

class _UnilinearModel(Model):
    def __init__(self) -> None: ...

unilinear: Untyped

class _QuadraticModel(Model):
    def __init__(self) -> None: ...

quadratic: Untyped
