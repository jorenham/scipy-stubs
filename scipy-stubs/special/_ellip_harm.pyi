from typing import Literal

__all__ = ["ellip_harm", "ellip_harm_2", "ellip_normal"]

def ellip_harm(
    h2: float,
    k2: float,
    n: int,
    p: float,
    s: float,
    signm: Literal[-1, 1] = ...,
    signn: Literal[-1, 1] = ...,
) -> float: ...
def ellip_harm_2(h2: float, k2: float, n: int, p: int, s: float) -> float: ...
def ellip_normal(h2: float, k2: float, n: int, p: int) -> float: ...
