from scipy._lib._array_api import array_namespace as array_namespace
from scipy._typing import Untyped

def affine_transform_dispatcher(
    input, matrix, offset: float = 0.0, output_shape: Untyped | None = None, output: Untyped | None = None, *args, **kwds
) -> Untyped: ...
def binary_closing_dispatcher(
    input, structure: Untyped | None = None, iterations: int = 1, output: Untyped | None = None, *args, **kwds
) -> Untyped: ...

binary_opening_dispatcher = binary_closing_dispatcher

def binary_dilation_dispatcher(
    input,
    structure: Untyped | None = None,
    iterations: int = 1,
    mask: Untyped | None = None,
    output: Untyped | None = None,
    *args,
    **kwds,
) -> Untyped: ...

binary_erosion_dispatcher = binary_dilation_dispatcher

def binary_fill_holes_dispatcher(
    input, structure: Untyped | None = None, output: Untyped | None = None, origin: int = 0
) -> Untyped: ...

label_dispatcher = binary_fill_holes_dispatcher

def binary_hit_or_miss_dispatcher(
    input, structure1: Untyped | None = None, structure2: Untyped | None = None, output: Untyped | None = None, *args, **kwds
) -> Untyped: ...
def binary_propagation_dispatcher(
    input, structure: Untyped | None = None, mask: Untyped | None = None, output: Untyped | None = None, *args, **kwds
) -> Untyped: ...
def convolve_dispatcher(input, weights, output: Untyped | None = None, *args, **kwds) -> Untyped: ...

correlate_dispatcher = convolve_dispatcher

def convolve1d_dispatcher(input, weights, axis: int = -1, output: Untyped | None = None, *args, **kwds) -> Untyped: ...

correlate1d_dispatcher = convolve1d_dispatcher

def distance_transform_bf_dispatcher(
    input,
    metric: str = "euclidean",
    sampling: Untyped | None = None,
    return_distances: bool = True,
    return_indices: bool = False,
    distances: Untyped | None = None,
    indices: Untyped | None = None,
) -> Untyped: ...
def distance_transform_cdt_dispatcher(
    input,
    metric: str = "chessboard",
    return_distances: bool = True,
    return_indices: bool = False,
    distances: Untyped | None = None,
    indices: Untyped | None = None,
) -> Untyped: ...
def distance_transform_edt_dispatcher(
    input,
    sampling: Untyped | None = None,
    return_distances: bool = True,
    return_indices: bool = False,
    distances: Untyped | None = None,
    indices: Untyped | None = None,
) -> Untyped: ...
def find_objects_dispatcher(input, max_label: int = 0) -> Untyped: ...
def fourier_ellipsoid_dispatcher(input, size, n: int = -1, axis: int = -1, output: Untyped | None = None) -> Untyped: ...

fourier_uniform_dispatcher = fourier_ellipsoid_dispatcher

def fourier_gaussian_dispatcher(input, sigma, n: int = -1, axis: int = -1, output: Untyped | None = None) -> Untyped: ...
def fourier_shift_dispatcher(input, shift, n: int = -1, axis: int = -1, output: Untyped | None = None) -> Untyped: ...
def gaussian_filter_dispatcher(input, sigma, order: int = 0, output: Untyped | None = None, *args, **kwds) -> Untyped: ...
def gaussian_filter1d_dispatcher(
    input, sigma, axis: int = -1, order: int = 0, output: Untyped | None = None, *args, **kwds
) -> Untyped: ...
def gaussian_gradient_magnitude_dispatcher(input, sigma, output: Untyped | None = None, *args, **kwds) -> Untyped: ...

gaussian_laplace_dispatcher = gaussian_gradient_magnitude_dispatcher

def generate_binary_structure_dispatcher(rank, connectivity) -> Untyped: ...
def generic_filter_dispatcher(
    input, function, size: Untyped | None = None, footprint: Untyped | None = None, output: Untyped | None = None, *args, **kwds
) -> Untyped: ...
def generic_filter1d_dispatcher(
    input, function, filter_size, axis: int = -1, output: Untyped | None = None, *args, **kwds
) -> Untyped: ...
def generic_gradient_magnitude_dispatcher(input, derivative, output: Untyped | None = None, *args, **kwds) -> Untyped: ...
def generic_laplace_dispatcher(input, derivative2, output: Untyped | None = None, *args, **kwds) -> Untyped: ...
def geometric_transform_dispatcher(
    input, mapping, output_shape: Untyped | None = None, output: Untyped | None = None, *args, **kwds
) -> Untyped: ...
def histogram_dispatcher(input, min, max, bins, labels: Untyped | None = None, index: Untyped | None = None) -> Untyped: ...
def iterate_structure_dispatcher(structure, iterations, origin: Untyped | None = None) -> Untyped: ...
def labeled_comprehension_dispatcher(input, labels, *args, **kwds) -> Untyped: ...
def laplace_dispatcher(input, output: Untyped | None = None, *args, **kwds) -> Untyped: ...
def map_coordinates_dispatcher(input, coordinates, output: Untyped | None = None, *args, **kwds) -> Untyped: ...
def maximum_filter1d_dispatcher(input, size, axis: int = -1, output: Untyped | None = None, *args, **kwds) -> Untyped: ...

minimum_filter1d_dispatcher = maximum_filter1d_dispatcher
uniform_filter1d_dispatcher = maximum_filter1d_dispatcher

def maximum_dispatcher(input, labels: Untyped | None = None, index: Untyped | None = None) -> Untyped: ...

minimum_dispatcher = maximum_dispatcher
median_dispatcher = maximum_dispatcher
mean_dispatcher = maximum_dispatcher
variance_dispatcher = maximum_dispatcher
standard_deviation_dispatcher = maximum_dispatcher
sum_labels_dispatcher = maximum_dispatcher
sum_dispatcher = maximum_dispatcher
maximum_position_dispatcher = maximum_dispatcher
minimum_position_dispatcher = maximum_dispatcher
extrema_dispatcher = maximum_dispatcher
center_of_mass_dispatcher = extrema_dispatcher

def median_filter_dispatcher(
    input, size: Untyped | None = None, footprint: Untyped | None = None, output: Untyped | None = None, *args, **kwds
) -> Untyped: ...

minimum_filter_dispatcher = median_filter_dispatcher
maximum_filter_dispatcher = median_filter_dispatcher

def morphological_gradient_dispatcher(
    input,
    size: Untyped | None = None,
    footprint: Untyped | None = None,
    structure: Untyped | None = None,
    output: Untyped | None = None,
    *args,
    **kwds,
) -> Untyped: ...

morphological_laplace_dispatcher = morphological_gradient_dispatcher
white_tophat_dispatcher = morphological_gradient_dispatcher
black_tophat_dispatcher = morphological_gradient_dispatcher
grey_closing_dispatcher = morphological_gradient_dispatcher
grey_dilation_dispatcher = morphological_gradient_dispatcher
grey_erosion_dispatcher = morphological_gradient_dispatcher
grey_opening_dispatcher = morphological_gradient_dispatcher

def percentile_filter_dispatcher(
    input, percentile, size: Untyped | None = None, footprint: Untyped | None = None, output: Untyped | None = None, *args, **kwds
) -> Untyped: ...
def prewitt_dispatcher(input, axis: int = -1, output: Untyped | None = None, *args, **kwds) -> Untyped: ...

sobel_dispatcher = prewitt_dispatcher

def rank_filter_dispatcher(
    input, rank, size: Untyped | None = None, footprint: Untyped | None = None, output: Untyped | None = None, *args, **kwds
) -> Untyped: ...
def rotate_dispatcher(
    input, angle, axes=(1, 0), reshape: bool = True, output: Untyped | None = None, *args, **kwds
) -> Untyped: ...
def shift_dispatcher(input, shift, output: Untyped | None = None, *args, **kwds) -> Untyped: ...
def spline_filter_dispatcher(input, order: int = 3, output=..., *args, **kwds) -> Untyped: ...
def spline_filter1d_dispatcher(input, order: int = 3, axis: int = -1, output=..., *args, **kwds) -> Untyped: ...
def uniform_filter_dispatcher(input, size: int = 3, output: Untyped | None = None, *args, **kwds) -> Untyped: ...
def value_indices_dispatcher(arr, *args, **kwds) -> Untyped: ...
def watershed_ift_dispatcher(input, markers, structure: Untyped | None = None, output: Untyped | None = None) -> Untyped: ...
def zoom_dispatcher(input, zoom, output: Untyped | None = None, *args, **kwds) -> Untyped: ...
