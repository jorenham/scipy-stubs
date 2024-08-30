from contextlib import _GeneratorContextManager
from typing import NamedTuple

from scipy._typing import Untyped, UntypedCallable

__version__: str

def get_init(cls) -> Untyped: ...

class ArgSpec(NamedTuple):
    args: Untyped
    varargs: Untyped
    varkw: Untyped
    defaults: Untyped

def getargspec(f) -> Untyped: ...

DEF: Untyped

class FunctionMaker:
    shortsignature: Untyped
    name: Untyped
    doc: Untyped
    module: Untyped
    annotations: Untyped
    signature: Untyped
    dict: Untyped
    defaults: Untyped
    def __init__(
        self,
        func: Untyped | None = None,
        name: Untyped | None = None,
        signature: Untyped | None = None,
        defaults: Untyped | None = None,
        doc: Untyped | None = None,
        module: Untyped | None = None,
        funcdict: Untyped | None = None,
    ): ...
    def update(self, func, **kw): ...
    def make(self, src_templ, evaldict: Untyped | None = None, addsource: bool = False, **attrs) -> Untyped: ...
    @classmethod
    def create(
        cls,
        obj,
        body,
        evaldict,
        defaults: Untyped | None = None,
        doc: Untyped | None = None,
        module: Untyped | None = None,
        addsource: bool = True,
        **attrs,
    ) -> Untyped: ...

def decorate(func: UntypedCallable, caller) -> Untyped: ...
def decorator(caller, _func: UntypedCallable | None = None) -> UntypedCallable: ...

class ContextManager(_GeneratorContextManager[Untyped]): ...

init: Untyped
n_args: Untyped

def __init__(self, g, *a, **k): ...

contextmanager: UntypedCallable

def append(a, vancestors): ...
def dispatch_on(*dispatch_args) -> Untyped: ...
