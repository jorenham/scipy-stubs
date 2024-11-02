from scipy._typing import Untyped

__all__ = ["load_npz", "save_npz"]

PICKLE_KWARGS: Untyped

def save_npz(file: Untyped, matrix: Untyped, compressed: bool = True) -> None: ...
def load_npz(file: Untyped) -> Untyped: ...
