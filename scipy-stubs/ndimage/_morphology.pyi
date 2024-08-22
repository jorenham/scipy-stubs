from scipy._typing import Untyped

def iterate_structure(structure, iterations, origin: Untyped | None = None) -> Untyped: ...
def generate_binary_structure(rank, connectivity) -> Untyped: ...
def binary_erosion(
    input,
    structure: Untyped | None = None,
    iterations: int = 1,
    mask: Untyped | None = None,
    output: Untyped | None = None,
    border_value: int = 0,
    origin: int = 0,
    brute_force: bool = False,
) -> Untyped: ...
def binary_dilation(
    input,
    structure: Untyped | None = None,
    iterations: int = 1,
    mask: Untyped | None = None,
    output: Untyped | None = None,
    border_value: int = 0,
    origin: int = 0,
    brute_force: bool = False,
) -> Untyped: ...
def binary_opening(
    input,
    structure: Untyped | None = None,
    iterations: int = 1,
    output: Untyped | None = None,
    origin: int = 0,
    mask: Untyped | None = None,
    border_value: int = 0,
    brute_force: bool = False,
) -> Untyped: ...
def binary_closing(
    input,
    structure: Untyped | None = None,
    iterations: int = 1,
    output: Untyped | None = None,
    origin: int = 0,
    mask: Untyped | None = None,
    border_value: int = 0,
    brute_force: bool = False,
) -> Untyped: ...
def binary_hit_or_miss(
    input,
    structure1: Untyped | None = None,
    structure2: Untyped | None = None,
    output: Untyped | None = None,
    origin1: int = 0,
    origin2: Untyped | None = None,
) -> Untyped: ...
def binary_propagation(
    input,
    structure: Untyped | None = None,
    mask: Untyped | None = None,
    output: Untyped | None = None,
    border_value: int = 0,
    origin: int = 0,
) -> Untyped: ...
def binary_fill_holes(input, structure: Untyped | None = None, output: Untyped | None = None, origin: int = 0) -> Untyped: ...
def grey_erosion(
    input,
    size: Untyped | None = None,
    footprint: Untyped | None = None,
    structure: Untyped | None = None,
    output: Untyped | None = None,
    mode: str = "reflect",
    cval: float = 0.0,
    origin: int = 0,
) -> Untyped: ...
def grey_dilation(
    input,
    size: Untyped | None = None,
    footprint: Untyped | None = None,
    structure: Untyped | None = None,
    output: Untyped | None = None,
    mode: str = "reflect",
    cval: float = 0.0,
    origin: int = 0,
) -> Untyped: ...
def grey_opening(
    input,
    size: Untyped | None = None,
    footprint: Untyped | None = None,
    structure: Untyped | None = None,
    output: Untyped | None = None,
    mode: str = "reflect",
    cval: float = 0.0,
    origin: int = 0,
) -> Untyped: ...
def grey_closing(
    input,
    size: Untyped | None = None,
    footprint: Untyped | None = None,
    structure: Untyped | None = None,
    output: Untyped | None = None,
    mode: str = "reflect",
    cval: float = 0.0,
    origin: int = 0,
) -> Untyped: ...
def morphological_gradient(
    input,
    size: Untyped | None = None,
    footprint: Untyped | None = None,
    structure: Untyped | None = None,
    output: Untyped | None = None,
    mode: str = "reflect",
    cval: float = 0.0,
    origin: int = 0,
) -> Untyped: ...
def morphological_laplace(
    input,
    size: Untyped | None = None,
    footprint: Untyped | None = None,
    structure: Untyped | None = None,
    output: Untyped | None = None,
    mode: str = "reflect",
    cval: float = 0.0,
    origin: int = 0,
) -> Untyped: ...
def white_tophat(
    input,
    size: Untyped | None = None,
    footprint: Untyped | None = None,
    structure: Untyped | None = None,
    output: Untyped | None = None,
    mode: str = "reflect",
    cval: float = 0.0,
    origin: int = 0,
) -> Untyped: ...
def black_tophat(
    input,
    size: Untyped | None = None,
    footprint: Untyped | None = None,
    structure: Untyped | None = None,
    output: Untyped | None = None,
    mode: str = "reflect",
    cval: float = 0.0,
    origin: int = 0,
) -> Untyped: ...
def distance_transform_bf(
    input,
    metric: str = "euclidean",
    sampling: Untyped | None = None,
    return_distances: bool = True,
    return_indices: bool = False,
    distances: Untyped | None = None,
    indices: Untyped | None = None,
) -> Untyped: ...
def distance_transform_cdt(
    input,
    metric: str = "chessboard",
    return_distances: bool = True,
    return_indices: bool = False,
    distances: Untyped | None = None,
    indices: Untyped | None = None,
) -> Untyped: ...
def distance_transform_edt(
    input,
    sampling: Untyped | None = None,
    return_distances: bool = True,
    return_indices: bool = False,
    distances: Untyped | None = None,
    indices: Untyped | None = None,
) -> Untyped: ...
