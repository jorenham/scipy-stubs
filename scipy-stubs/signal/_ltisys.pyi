# mypy: disable-error-code="explicit-override"
import abc
from typing import ClassVar, Final, Literal, TypeAlias, final, overload, type_check_only
from typing_extensions import Never, Self, Unpack, override

import numpy as np
import optype as op
import optype.numpy as onp
import optype.numpy.compat as npc
from ._lti_conversion import _DiscretizeMethod

__all__ = [
    "StateSpace",
    "TransferFunction",
    "ZerosPolesGain",
    "bode",
    "dbode",
    "dfreqresp",
    "dimpulse",
    "dlsim",
    "dlti",
    "dstep",
    "freqresp",
    "impulse",
    "lsim",
    "lti",
    "place_poles",
    "step",
]

###

# numerator, denominator
_ToTFContReal: TypeAlias = tuple[onp.ToFloat1D | onp.ToFloat2D, onp.ToComplex1D]
_ToTFContComplex: TypeAlias = tuple[onp.ToComplex1D | onp.ToComplex2D, onp.ToComplex1D]
# numerator, denominator, dt
_ToTFDiscReal: TypeAlias = tuple[onp.ToFloat1D | onp.ToFloat2D, onp.ToComplex1D, onp.ToFloat]

# zeros, poles, gain
_ToZPKContReal: TypeAlias = tuple[onp.ToFloat1D, onp.ToFloat1D, onp.ToFloat]
_ToZPKContComplex: TypeAlias = tuple[onp.ToComplex1D, onp.ToComplex1D, onp.ToFloat]
# zeros, poles, gain, dt
_ToZPKDiscReal: TypeAlias = tuple[onp.ToFloat1D, onp.ToFloat1D, onp.ToFloat, onp.ToFloat]

# A, B, C, D
_ToSSContReal: TypeAlias = tuple[onp.ToFloat2D, onp.ToFloat2D, onp.ToFloat2D, onp.ToFloat2D]
_ToSSContComplex: TypeAlias = tuple[onp.ToComplex2D, onp.ToComplex2D, onp.ToComplex2D, onp.ToComplex2D]
# A, B, C, D, dt
_ToSSDiscReal: TypeAlias = tuple[onp.ToFloat2D, onp.ToFloat2D, onp.ToFloat2D, onp.ToFloat2D, onp.ToFloat]

_ToLTIReal: TypeAlias = _ToTFContReal | _ToZPKContReal | _ToSSContReal
_ToLTIComplex: TypeAlias = _ToTFContComplex | _ToZPKContComplex | _ToSSContComplex
_ToDLTI: TypeAlias = _ToTFDiscReal | _ToZPKDiscReal | _ToSSDiscReal

###

# TODO(jorenham): Generic scalar type
class LinearTimeInvariant:
    inputs: Final[int]
    outputs: Final[int]

    def __new__(cls, *system: Never, **kwargs: Never) -> Self: ...

    #
    @abc.abstractmethod
    @type_check_only
    def to_tf(self, /) -> TransferFunction: ...
    @abc.abstractmethod
    @type_check_only
    def to_zpk(self, /) -> ZerosPolesGain: ...
    @abc.abstractmethod
    @type_check_only
    def to_ss(self, /) -> StateSpace: ...

    #
    @property
    def dt(self, /) -> float | None: ...
    @property
    def zeros(self, /) -> onp.Array1D[npc.number] | onp.Array2D[npc.number]: ...
    @property
    def poles(self, /) -> onp.Array1D[npc.number]: ...

class lti(LinearTimeInvariant, metaclass=abc.ABCMeta):
    @overload
    def __new__(cls, *system: Unpack[tuple[onp.ToFloat1D | onp.ToFloat2D, onp.ToFloat1D]]) -> TransferFunctionContinuous: ...
    @overload
    def __new__(cls, *system: Unpack[tuple[onp.ToComplex1D, onp.ToComplex1D, onp.ToFloat]]) -> ZerosPolesGainContinuous: ...
    @overload
    def __new__(
        cls,
        *system: Unpack[tuple[onp.ToComplex2D, onp.ToComplex2D, onp.ToComplex2D, onp.ToComplex2D]],
    ) -> StateSpaceContinuous: ...

    #
    def __init__(self, /, *system: Never) -> None: ...

    #
    def impulse(
        self,
        /,
        X0: onp.ToFloat1D | None = None,
        T: onp.ToFloat1D | None = None,
        N: onp.ToJustInt | None = None,
    ) -> tuple[onp.Array1D[npc.floating], onp.Array1D[npc.floating]]: ...
    def step(
        self,
        /,
        X0: onp.ToComplex1D | None = None,
        T: onp.ToFloat1D | None = None,
        N: onp.ToJustInt | None = None,
    ) -> tuple[onp.Array1D[npc.floating], onp.Array1D[npc.inexact]]: ...
    def output(
        self,
        /,
        U: onp.ToFloat1D | onp.ToFloat2D | onp.ToFloat | None,
        T: onp.ToFloat1D,
        X0: onp.ToComplex1D | None = None,
    ) -> tuple[onp.Array1D[npc.floating], onp.Array1D[npc.inexact], onp.Array1D[npc.inexact] | onp.Array2D[npc.inexact]]: ...
    def bode(
        self,
        /,
        w: onp.ToFloat1D | None = None,
        n: onp.ToJustInt = 100,
    ) -> tuple[onp.Array1D[npc.floating], onp.Array1D[npc.floating], onp.Array1D[npc.floating]]: ...
    def freqresp(
        self,
        /,
        w: onp.ToFloat1D | None = None,
        n: onp.ToJustInt = 10_000,
    ) -> tuple[onp.Array1D[npc.floating], onp.Array1D[npc.complexfloating]]: ...

    #
    @abc.abstractmethod
    def to_discrete(
        self,
        /,
        dt: onp.ToFloat,
        method: _DiscretizeMethod = "zoh",
        alpha: onp.ToJustFloat | None = None,
    ) -> dlti: ...

class dlti(LinearTimeInvariant, metaclass=abc.ABCMeta):
    @overload
    def __new__(
        cls,
        *system: Unpack[tuple[onp.ToFloat1D | onp.ToFloat2D, onp.ToFloat1D]],
        dt: onp.ToFloat = True,
    ) -> TransferFunctionDiscrete: ...
    @overload
    def __new__(
        cls,
        *system: Unpack[tuple[onp.ToComplex1D, onp.ToComplex1D, onp.ToFloat]],
        dt: onp.ToFloat = True,
    ) -> ZerosPolesGainDiscrete: ...
    @overload
    def __new__(
        cls,
        *system: Unpack[tuple[onp.ToComplex2D, onp.ToComplex2D, onp.ToComplex2D, onp.ToComplex2D]],
        dt: onp.ToFloat = True,
    ) -> StateSpaceDiscrete: ...

    #
    def __init__(self, /, *system: Never, dt: onp.ToFloat, **kwargs: Never) -> None: ...

    #
    @property
    @override
    def dt(self, /) -> float: ...

    #
    def impulse(
        self,
        /,
        x0: onp.ToFloat1D | None = None,
        t: onp.ToFloat1D | None = None,
        n: onp.ToJustInt | None = None,
    ) -> tuple[onp.Array1D[npc.floating], onp.Array1D[npc.floating]]: ...
    def step(
        self,
        /,
        x0: onp.ToFloat1D | None = None,
        t: onp.ToFloat1D | None = None,
        n: onp.ToJustInt | None = None,
    ) -> tuple[onp.Array1D[npc.floating], onp.Array1D[npc.floating]]: ...
    def output(
        self,
        /,
        u: onp.ToFloat1D | onp.ToFloat2D | onp.ToFloat | None,
        t: onp.ToFloat1D,
        x0: onp.ToFloat1D | None = None,
    ) -> tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]]: ...
    def bode(
        self,
        /,
        w: onp.ToFloat1D | None = None,
        n: onp.ToJustInt = 100,
    ) -> tuple[onp.Array1D[npc.floating], onp.Array1D[npc.floating], onp.Array1D[npc.floating]]: ...
    def freqresp(
        self,
        /,
        w: onp.ToFloat1D | None = None,
        n: onp.ToJustInt = 10_000,
        whole: op.CanBool = False,
    ) -> tuple[onp.Array1D[npc.floating], onp.Array1D[npc.complexfloating]]: ...

class TransferFunction(LinearTimeInvariant):
    @overload
    def __new__(cls, *system: Unpack[tuple[lti]]) -> TransferFunctionContinuous: ...
    @overload
    def __new__(cls, *system: Unpack[tuple[dlti]]) -> TransferFunctionDiscrete: ...
    @overload
    def __new__(cls, *system: Unpack[tuple[onp.ToFloat1D | onp.ToFloat2D, onp.ToFloat1D]]) -> TransferFunctionContinuous: ...
    @overload
    def __new__(
        cls,
        *system: Unpack[tuple[onp.ToFloat1D | onp.ToFloat2D, onp.ToFloat1D]],
        dt: onp.ToFloat,
    ) -> TransferFunctionDiscrete: ...

    #
    @overload
    def __init__(self, system: LinearTimeInvariant, /) -> None: ...
    @overload
    def __init__(self, numerator: onp.ToFloat1D | onp.ToFloat2D, denominator: onp.ToFloat1D, /) -> None: ...

    #
    @property
    def num(self, /) -> onp.Array1D[npc.number] | onp.Array2D[npc.number]: ...
    @num.setter
    def num(self, /, num: onp.ToComplex1D | onp.ToComplex2D) -> None: ...

    #
    @property
    def den(self, /) -> onp.Array1D[npc.number]: ...
    @den.setter
    def den(self, /, den: onp.ToComplex1D) -> None: ...

    #
    @override
    def to_tf(self, /) -> Self: ...
    @override
    def to_zpk(self, /) -> ZerosPolesGain: ...
    @override
    def to_ss(self, /) -> StateSpace: ...

@final
class TransferFunctionContinuous(TransferFunction, lti):
    @override
    def to_discrete(
        self,
        /,
        dt: onp.ToFloat,
        method: _DiscretizeMethod = "zoh",
        alpha: onp.ToJustFloat | None = None,
    ) -> TransferFunctionDiscrete: ...

@final
class TransferFunctionDiscrete(TransferFunction, dlti):
    @overload
    def __init__(self, system: LinearTimeInvariant, /) -> None: ...
    @overload
    def __init__(
        self,
        numerator: onp.ToFloat1D | onp.ToFloat2D,
        denominator: onp.ToFloat1D,
        /,
        *,
        dt: onp.ToFloat = ...,
    ) -> None: ...

class ZerosPolesGain(LinearTimeInvariant):
    @overload
    def __new__(cls, *system: Unpack[tuple[lti]]) -> ZerosPolesGainContinuous: ...
    @overload
    def __new__(cls, *system: Unpack[tuple[dlti]]) -> ZerosPolesGainDiscrete: ...
    @overload
    def __new__(
        cls,
        *system: Unpack[tuple[onp.ToComplex1D | onp.ToComplex2D, onp.ToComplex1D, onp.ToFloat]],
    ) -> ZerosPolesGainContinuous: ...
    @overload
    def __new__(
        cls,
        *system: Unpack[tuple[onp.ToComplex1D | onp.ToComplex2D, onp.ToComplex1D, onp.ToFloat]],
        dt: onp.ToFloat,
    ) -> ZerosPolesGainDiscrete: ...

    #
    @overload
    def __init__(self, system: LinearTimeInvariant, /) -> None: ...
    @overload
    def __init__(self, zeros: onp.ToComplex1D | onp.ToComplex2D, poles: onp.ToComplex1D, gain: onp.ToFloat, /) -> None: ...

    #
    @property
    @override
    def zeros(self, /) -> onp.Array1D[npc.number] | onp.Array2D[npc.number]: ...
    @zeros.setter
    def zeros(self, zeros: onp.ToComplex1D | onp.ToComplex2D, /) -> None: ...

    #
    @property
    @override
    def poles(self, /) -> onp.Array1D[npc.number]: ...
    @poles.setter
    def poles(self, gain: onp.ToComplex1D, /) -> None: ...

    #
    @property
    def gain(self, /) -> float: ...
    @gain.setter
    def gain(self, gain: float, /) -> None: ...

    #
    @override
    def to_tf(self, /) -> TransferFunction: ...
    @override
    def to_zpk(self, /) -> Self: ...
    @override
    def to_ss(self, /) -> StateSpace: ...

@final
class ZerosPolesGainContinuous(ZerosPolesGain, lti):
    @override
    def to_discrete(
        self,
        /,
        dt: onp.ToFloat,
        method: _DiscretizeMethod = "zoh",
        alpha: onp.ToJustFloat | None = None,
    ) -> ZerosPolesGainDiscrete: ...

@final
class ZerosPolesGainDiscrete(ZerosPolesGain, dlti):
    @overload
    def __init__(self, system: ZerosPolesGain, /) -> None: ...
    @overload
    def __init__(
        self,
        zeros: onp.ToComplex1D | onp.ToComplex2D,
        poles: onp.ToComplex1D,
        gain: onp.ToFloat,
        /,
        *,
        dt: onp.ToFloat = ...,
    ) -> None: ...

class StateSpace(LinearTimeInvariant):
    __array_priority__: ClassVar[float] = 100.0
    __array_ufunc__: ClassVar[None] = None

    @overload
    def __new__(cls, *system: Unpack[tuple[lti]]) -> StateSpaceContinuous: ...
    @overload
    def __new__(cls, *system: Unpack[tuple[dlti]]) -> StateSpaceDiscrete: ...
    @overload
    def __new__(
        cls,
        *system: Unpack[tuple[onp.ToComplex2D, onp.ToComplex2D, onp.ToComplex2D, onp.ToComplex2D]],
    ) -> StateSpaceContinuous: ...
    @overload
    def __new__(
        cls,
        *system: Unpack[tuple[onp.ToComplex2D, onp.ToComplex2D, onp.ToComplex2D, onp.ToComplex2D]],
        dt: onp.ToFloat,
    ) -> StateSpaceDiscrete: ...

    #
    @overload
    def __init__(self, system: StateSpace, /) -> None: ...
    @overload
    def __init__(
        self,
        A: onp.ToComplex2D,
        B: onp.ToComplex2D,
        C: onp.ToComplex2D,
        D: onp.ToComplex2D,
        /,
    ) -> None: ...

    #
    def __neg__(self, /) -> Self: ...
    def __add__(self, other: Self | complex | npc.number | onp.ArrayND[npc.number], /) -> Self: ...
    def __sub__(self, other: Self | complex | npc.number | onp.ArrayND[npc.number], /) -> Self: ...
    def __mul__(self, other: Self | complex | npc.number | onp.ArrayND[npc.number], /) -> Self: ...
    def __truediv__(self, other: complex | npc.number, /) -> Self: ...
    # ehh mypy, u ok?
    def __radd__(self, other: complex | npc.number | onp.ArrayND[npc.number], /) -> Self: ...  # type: ignore[misc]
    def __rsub__(self, other: complex | npc.number | onp.ArrayND[npc.number], /) -> Self: ...  # type: ignore[misc]
    def __rmul__(self, other: complex | npc.number | onp.ArrayND[npc.number], /) -> Self: ...  # type: ignore[misc]

    #
    @property
    def A(self, /) -> onp.Array2D[npc.number]: ...
    @A.setter
    def A(self, /, A: onp.ToComplex2D) -> None: ...
    #
    @property
    def B(self, /) -> onp.Array2D[npc.number]: ...
    @B.setter
    def B(self, /, B: onp.ToComplex2D) -> None: ...
    #
    @property
    def C(self, /) -> onp.Array2D[npc.number]: ...
    @C.setter
    def C(self, /, C: onp.ToComplex2D) -> None: ...
    #
    @property
    def D(self, /) -> onp.Array2D[npc.number]: ...
    @D.setter
    def D(self, /, D: onp.ToComplex2D) -> None: ...
    #
    @override
    def to_tf(self, /, *, input: onp.ToInt = 0) -> TransferFunction: ...
    @override
    def to_zpk(self, /, *, input: onp.ToInt = 0) -> ZerosPolesGain: ...
    @override
    def to_ss(self, /) -> Self: ...

@final
class StateSpaceContinuous(StateSpace, lti):
    @override
    def to_discrete(
        self,
        /,
        dt: onp.ToFloat,
        method: _DiscretizeMethod = "zoh",
        alpha: onp.ToJustFloat | None = None,
    ) -> StateSpaceDiscrete: ...

@final
class StateSpaceDiscrete(StateSpace, dlti):
    @overload
    def __init__(self, system: StateSpace, /) -> None: ...
    @overload
    def __init__(
        self,
        A: onp.ToComplex2D,
        B: onp.ToComplex2D,
        C: onp.ToComplex2D,
        D: onp.ToComplex2D,
        /,
        *,
        dt: onp.ToFloat = ...,
    ) -> None: ...

# NOTE: Only used as return type of `place_poles`.
class Bunch:
    gain_matrix: onp.Array2D[np.float64]
    computed_poles: onp.Array1D[np.float64]
    requested_poles: onp.Array1D[np.float64]
    X: onp.Array2D[np.complex128]
    rtol: float
    nb_iter: int

    def __init__(self, /, **kwds: Never) -> None: ...

#
def place_poles(
    A: onp.ToFloat2D,
    B: onp.ToFloat2D,
    poles: onp.ToComplex1D,
    method: Literal["YT", "KNV0"] = "YT",
    rtol: float = 0.001,
    maxiter: int = 30,
) -> Bunch: ...

#
@overload
def lsim(
    system: _ToLTIReal,
    U: onp.ToFloat1D | onp.ToFloat2D | onp.ToFloat | None,
    T: onp.ToFloat1D,
    X0: onp.ToFloat1D | None = None,
    interp: op.CanBool = True,
) -> tuple[onp.Array1D[npc.floating], onp.Array1D[npc.floating], onp.Array1D[npc.floating] | onp.Array2D[npc.floating]]: ...
@overload
def lsim(
    system: lti | _ToLTIComplex,
    U: onp.ToFloat1D | onp.ToFloat2D | onp.ToFloat | None,
    T: onp.ToFloat1D,
    X0: onp.ToComplex1D | None = None,
    interp: op.CanBool = True,
) -> tuple[onp.Array1D[npc.floating], onp.Array1D[npc.inexact], onp.Array1D[npc.inexact] | onp.Array2D[npc.inexact]]: ...

#
@overload
def dlsim(
    system: StateSpaceDiscrete,
    u: onp.ToFloat1D | onp.ToFloat2D | onp.ToFloat | None,
    t: onp.ToFloat1D | None = None,
    x0: onp.ToFloat1D | None = None,
) -> tuple[onp.Array1D[np.float64], onp.Array1D[np.float64], onp.Array1D[np.float64]]: ...
@overload
def dlsim(
    system: _ToDLTI,
    u: onp.ToFloat1D | onp.ToFloat2D | onp.ToFloat | None,
    t: onp.ToFloat1D | None = None,
    x0: onp.ToFloat1D | None = None,
) -> tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]]: ...

#
@overload
def impulse(
    system: _ToLTIReal,
    X0: onp.ToFloat1D | None = None,
    T: onp.ToFloat1D | None = None,
    N: onp.ToJustInt | None = None,
) -> tuple[onp.Array1D[npc.floating], onp.Array1D[npc.floating]]: ...
@overload
def impulse(
    system: lti | _ToLTIComplex,
    X0: onp.ToComplex1D | None = None,
    T: onp.ToFloat1D | None = None,
    N: onp.ToJustInt | None = None,
) -> tuple[onp.Array1D[npc.floating], onp.Array1D[npc.inexact]]: ...

#
def dimpulse(
    system: dlti | _ToDLTI,
    x0: onp.ToFloat1D | None = None,
    t: onp.ToFloat1D | None = None,
    n: onp.ToJustInt | None = None,
) -> tuple[onp.Array1D[npc.floating], onp.Array1D[npc.floating]]: ...

#
@overload
def step(
    system: _ToLTIReal,
    X0: onp.ToFloat1D | None = None,
    T: onp.ToFloat1D | None = None,
    N: onp.ToJustInt | None = None,
) -> tuple[onp.Array1D[npc.floating], onp.Array1D[npc.floating]]: ...
@overload
def step(
    system: lti | _ToLTIComplex,
    X0: onp.ToComplex1D | None = None,
    T: onp.ToFloat1D | None = None,
    N: onp.ToJustInt | None = None,
) -> tuple[onp.Array1D[npc.floating], onp.Array1D[npc.inexact]]: ...

#
def dstep(
    system: dlti | _ToDLTI,
    x0: onp.ToFloat1D | None = None,
    t: onp.ToFloat1D | None = None,
    n: onp.ToJustInt | None = None,
) -> tuple[onp.Array1D[npc.floating], onp.Array1D[npc.floating]]: ...

#
def bode(
    system: lti | _ToLTIComplex,
    w: onp.ToFloat1D | None = None,
    n: onp.ToJustInt = 100,
) -> tuple[onp.Array1D[npc.floating], onp.Array1D[npc.floating], onp.Array1D[npc.floating]]: ...

#
def dbode(
    system: dlti | _ToDLTI,
    w: onp.ToFloat1D | None = None,
    n: onp.ToJustInt = 100,
) -> tuple[onp.Array1D[npc.floating], onp.Array1D[npc.floating], onp.Array1D[npc.floating]]: ...

#
def freqresp(
    system: lti | _ToLTIComplex,
    w: onp.ToFloat1D | None = None,
    n: onp.ToJustInt = 10_000,
) -> tuple[onp.Array1D[npc.floating], onp.Array1D[npc.complexfloating]]: ...

#
def dfreqresp(
    system: dlti | _ToDLTI,
    w: onp.ToFloat1D | None = None,
    n: onp.ToJustInt = 10_000,
    whole: op.CanBool = False,
) -> tuple[onp.Array1D[npc.floating], onp.Array1D[npc.complexfloating]]: ...
