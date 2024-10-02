import abc
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Any, Final, Generic, TypeAlias
from typing_extensions import Self, TypeVar, override

import numpy as np
import optype.numpy as onpt
from scipy._lib._util import MapWrapper as MapWrapper
from scipy._typing import Untyped, UntypedTuple

_SCT = TypeVar("_SCT", bound=np.number[Any], default=np.floating[Any])
_SCT_co = TypeVar("_SCT_co", bound=np.number[Any], covariant=True, default=np.floating[Any])

_Vector: TypeAlias = onpt.Array[tuple[int], _SCT]
_VectorLike: TypeAlias = tuple[float | _SCT, ...] | _Vector[_SCT]

_ScalarField: TypeAlias = Callable[[_Vector[_SCT]], float | _SCT]
_VectorField: TypeAlias = Callable[[_Vector[_SCT]], _Vector[_SCT]]
_ScalarFieldCons: TypeAlias = Callable[[_Vector[_SCT]], Untyped]  # TODO

class VertexBase(Generic[_SCT_co], metaclass=abc.ABCMeta):
    x: _VectorLike[_SCT_co]
    x_a: _Vector[_SCT_co]  # lazy
    hash: Final[int]
    index: Final[int | None]
    nn: set[Self]

    st: set[Self]  # might not be set
    feasible: bool  # might not be set

    def __init__(self, /, x: _VectorLike[_SCT_co], nn: Iterable[Self] | None = None, index: int | None = None) -> None: ...
    @abc.abstractmethod
    def connect(self, /, v: Self) -> None: ...
    @abc.abstractmethod
    def disconnect(self, /, v: Self) -> None: ...
    def star(self, /) -> set[Self]: ...

class VertexScalarField(VertexBase[_SCT_co], Generic[_SCT_co]):
    check_min: bool
    check_max: bool
    # TODO: Support non-empty `field_args` and `g_cons_args`
    def __init__(
        self,
        /,
        x: _VectorLike[_SCT_co],
        field: _ScalarField[_SCT_co] | None = None,
        nn: Iterable[Self] | None = None,
        index: int | None = None,
        field_args: tuple[()] = (),
        g_cons: _ScalarFieldCons[_SCT_co] | None = None,
        g_cons_args: tuple[()] = (),
    ) -> None: ...
    @override
    def connect(self, /, v: Self) -> None: ...
    @override
    def disconnect(self, /, v: Self) -> None: ...
    def minimiser(self, /) -> bool: ...
    def maximiser(self, /) -> bool: ...

class VertexVectorField(VertexBase[_SCT_co], Generic[_SCT_co], metaclass=abc.ABCMeta):
    # NOTE: The implementaiton is a WIP
    # TODO: Support non-empty `field_args`, `vfield_args`, and `g_cons_args`
    def __init__(
        self,
        /,
        x: _VectorLike[_SCT_co],
        sfield: _ScalarField[_SCT_co] | None = None,
        vfield: _VectorField[_SCT_co] | None = None,
        field_args: tuple[()] = (),
        vfield_args: tuple[()] = (),
        g_cons: _ScalarFieldCons[_SCT_co] | None = None,
        g_cons_args: tuple[()] = (),
        nn: Iterable[Self] | None = None,
        index: int | None = None,
    ) -> None: ...
    @override
    def connect(self, /, v: Self) -> None: ...
    @override
    def disconnect(self, /, v: Self) -> None: ...

class VertexCube(VertexBase[_SCT_co], Generic[_SCT_co]):
    def __init__(self, /, x: _VectorLike[_SCT_co], nn: Iterable[Self] | None = None, index: int | None = None) -> None: ...
    @override
    def connect(self, /, v: Self) -> None: ...
    @override
    def disconnect(self, /, v: Self) -> None: ...

_KT = TypeVar("_KT", default=Untyped)  # TODO: Select a decent default
_VT = TypeVar("_VT", bound=VertexBase, default=VertexBase)

class VertexCacheBase(Generic[_KT, _VT]):
    cache: OrderedDict[_KT, _VT]
    nfev: int
    index: int
    def __init__(self, /) -> None: ...
    def __iter__(self, /) -> Iterator[_VT]: ...
    def size(self, /) -> int: ...
    def print_out(self, /) -> None: ...

class VertexCacheIndex(VertexCacheBase[_KT, _VT], Generic[_KT, _VT]):
    Vertex: type[_VT]
    def __getitem__(self, x: _KT, /, nn: None = None) -> _VT: ...

class VertexCacheField(VertexCacheBase[_KT, _VT], Generic[_KT, _VT, _SCT_co]):
    index: int
    Vertex: type[_VT]
    field: _ScalarField[_SCT_co]
    field_args: tuple[()]
    wfield: FieldWrapper[_SCT_co]
    g_cons: Sequence[_ScalarFieldCons[_SCT_co]]
    g_cons_args: tuple[()]
    wgcons: ConstraintWrapper[_SCT_co]
    gpool: set[UntypedTuple]
    fpool: set[UntypedTuple]
    sfc_lock: bool
    workers: int
    process_gpool: Callable[[], None]
    process_fpool: Callable[[], None]
    def __init__(
        self,
        /,
        field: Untyped | None = None,
        field_args: tuple[()] = (),
        g_cons: Sequence[_ScalarFieldCons[_SCT_co]] | None = None,
        g_cons_args: tuple[()] = (),
        workers: int = 1,
    ) -> None: ...
    def __getitem__(self, x: _KT, /, nn: Iterable[_VT] | None = None) -> _VT: ...
    def process_pools(self, /) -> None: ...
    def feasibility_check(self, /, v: _VT) -> bool: ...
    def compute_sfield(self, /, v: _VT) -> None: ...
    def proc_gpool(self, /) -> None: ...
    def pproc_gpool(self, /) -> None: ...
    def proc_fpool_g(self, /) -> None: ...
    def proc_fpool_nog(self, /) -> None: ...
    def pproc_fpool_g(self, /) -> None: ...
    def pproc_fpool_nog(self, /) -> None: ...
    def proc_minimisers(self, /) -> None: ...

class ConstraintWrapper(Generic[_SCT]):
    g_cons: Sequence[_ScalarFieldCons[_SCT]]
    g_cons_args: Sequence[UntypedTuple]
    def __init__(self, /, g_cons: Sequence[_ScalarFieldCons[_SCT]], g_cons_args: Sequence[UntypedTuple]) -> None: ...
    def gcons(self, /, v_x_a: _Vector[_SCT]) -> bool: ...

class FieldWrapper(Generic[_SCT]):
    # TODO: Support non-empty `g_cons_args`
    field: _ScalarField[_SCT] | _VectorField[_SCT]
    field_args: tuple[()]
    def __init__(self, /, field: _ScalarField[_SCT] | _VectorField[_SCT], field_args: tuple[()]) -> None: ...
    def func(self, /, v_x_a: _Vector[_SCT]) -> _SCT: ...
