from typing import Any

from scipy._typing import Untyped

txt2002: str
txt2006: str
txt2010: str
txt2014: str
txt2018: str
physical_constants: dict[str, tuple[float, str, float]]

def parse_constants_2002to2014(d: str) -> dict[str, tuple[float, str, float]]: ...
def parse_constants_2018toXXXX(d: str) -> dict[str, tuple[float, str, float]]: ...

class ConstantWarning(DeprecationWarning): ...

def value(key: str) -> float: ...
def unit(key: str) -> str: ...
def precision(key: str) -> float: ...
def find(sub: str | None = None, disp: bool = False) -> Any: ...

c: Untyped
mu0: Untyped
epsilon0: Untyped
exact_values: Untyped
val: Untyped
