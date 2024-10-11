from scipy._typing import Untyped, UntypedArray

__all__ = ["max_len_seq"]

def max_len_seq(
    nbits: int,
    state: Untyped | None = None,
    length: int | None = None,
    taps: Untyped | None = None,
) -> tuple[UntypedArray, UntypedArray]: ...
