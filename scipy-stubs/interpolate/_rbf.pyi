from scipy import linalg as linalg
from scipy._typing import Untyped
from scipy.spatial.distance import cdist as cdist, pdist as pdist, squareform as squareform
from scipy.special import xlogy as xlogy

class Rbf:
    xi: Untyped
    N: Untyped
    mode: Untyped
    di: Untyped
    norm: Untyped
    epsilon: Untyped
    smooth: Untyped
    function: Untyped
    nodes: Untyped
    def __init__(self, *args, **kwargs) -> None: ...
    @property
    def A(self) -> Untyped: ...
    def __call__(self, *args) -> Untyped: ...
