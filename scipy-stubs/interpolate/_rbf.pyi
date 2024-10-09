from scipy._typing import Untyped

__all__ = ["Rbf"]

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
    @property
    def A(self) -> Untyped: ...
    def __init__(self, *args: Untyped, **kwargs: Untyped) -> None: ...
    def __call__(self, *args: Untyped) -> Untyped: ...
