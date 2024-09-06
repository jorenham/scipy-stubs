from typing import Literal

import numpy as np
import scipy._typing as spt

def ellip_harm(
    h2: spt.AnyReal,
    k2: spt.AnyReal,
    n: spt.AnyInt,
    p: spt.AnyReal,
    s: spt.AnyReal,
    signm: Literal[-1, 1] = ...,
    signn: Literal[-1, 1] = ...,
) -> np.float64: ...
def ellip_harm_2(h2: spt.AnyReal, k2: spt.AnyReal, n: spt.AnyInt, p: spt.AnyInt, s: spt.AnyReal) -> spt.Array0D[np.float64]: ...
def ellip_normal(h2: spt.AnyReal, k2: spt.AnyReal, n: spt.AnyReal, p: spt.AnyReal) -> spt.Array0D[np.float64]: ...
