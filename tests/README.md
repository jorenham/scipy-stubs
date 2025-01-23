# type-testing `scipy-stubs`

The `scipy-stubs` annotations are tested through the use of `.pyi` stubs.
These can be "run" (i.e. type-checked with mypy and pyright) with:

```shell
uv run mypy tests && uv run basedpyright tests
```

## A simple example

To illustrate, consider the following (Python 3.12+) function implementation:

```py
# example/__init__.py
def add_one(x, /):
    return x + 1
```

The stubs for the `example` package are as follows:

```pyi
# example-stubs/__init__.pyi
from typing import Literal, Protocol

class CanAdd[T, R](Protocol):
    def __add__(self, x: T, /) -> R: ...

def add_one[R](x: CanAdd[Literal[1], R]: , /) -> R: ...
```

A type-equivalent Python 3.10+ implementation of this `CanAdd` Protocol can be found in the
[`optype`][optype] package as [`optype.CanAdd`][optype-binops], a global dependency of `scipy-stubs`

### Positive tests

The type-test equivalent of `assert` in `pytest` unit-tests is [`typing.assert_type`][assert_type].
We can use it to validate whether certain input to `add_one` produces output we expect:

```pyi
# tests/test_example.pyi
from typing_extensions import assert_type
from example import add_one

assert_type(add_one(False), int)
assert_type(add_one(42), int)
assert_type(add_one(3.14), float)
assert_type(add_one(1 - 1j), complex)
```

### Negative tests

To validate that `add_one` rejects invalid input, we use `# type: ignore` to "assert"
a specific mypy error code, and likewise `# type: error` for pyright:

```pyi
# tests/test_example.pyi (continued)

add_one()  # type: ignore[call-arg]  # pyright: ignore[reportCallIssue]
add_one(None)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
add_one("")  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
add_one([])  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
```

If the input is accepted, mypy will report that there's an unused ignore comment, because
[`warn_unused_ignores = true`][mypy-warn-ignore] has been set in the mypy configuration.
In the same way, pyright requires that the ignored error is actually reported because it
is configured with [`reportUnnecessaryTypeIgnoreComment = true`][bpr-rules] and
[`enableTypeIgnoreComments = false`][bpr-rules].

[assert_type]: https://docs.python.org/3/library/typing.html#typing.assert_type
[bpr-rules]: https://docs.basedpyright.com/latest/configuration/config-files/#type-check-rule-overrides
[mypy-warn-ignore]: https://kotlinisland.github.io/basedmypy/config_file.html#confval-warn_unused_ignores
[optype]: https://github.com/jorenham/optype
[optype-binops]: https://github.com/jorenham/optype?tab=readme-ov-file#binary-operations
