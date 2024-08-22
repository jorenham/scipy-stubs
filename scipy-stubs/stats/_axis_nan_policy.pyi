from scipy._lib._array_api import array_namespace as array_namespace, is_numpy as is_numpy
from scipy._lib._docscrape import FunctionDoc as FunctionDoc, Parameter as Parameter
from scipy._lib._util import AxisError as AxisError

too_small_1d_not_omit: str
too_small_1d_omit: str
too_small_nd_not_omit: str
too_small_nd_omit: str

class SmallSampleWarning(RuntimeWarning): ...
