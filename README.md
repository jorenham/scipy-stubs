<h1 align="center">scipy-stubs</h1>

<p align="center">
    Precise type hints for <i>all</i> of <a href="https://github.com/scipy/scipy">SciPy</a>.
</p>

<p align="center">
    <a href="https://pypi.org/project/scipy-stubs/">
        <img
            alt="scipy-stubs - PyPI"
            src="https://img.shields.io/pypi/v/scipy-stubs?style=flat&color=olive"
        />
    </a>
    <a href="https://anaconda.org/conda-forge/scipy-stubs">
        <img
            alt="scipy-stubs - conda-forge"
            src="https://anaconda.org/conda-forge/scipy-stubs/badges/version.svg"
        />
    </a>
    <a href="https://github.com/jorenham/scipy-stubs">
        <img
            alt="scipy-stubs - Python Versions"
            src="https://img.shields.io/pypi/pyversions/scipy-stubs?style=flat"
        />
    </a>
    <a href="https://github.com/jorenham/scipy-stubs">
        <img
            alt="scipy-stubs - license"
            src="https://img.shields.io/github/license/jorenham/scipy-stubs?style=flat"
        />
    </a>
</p>
<p align="center">
    <a href="https://github.com/jorenham/scipy-stubs/actions?query=workflow%3ACI">
        <img
            alt="scipy-stubs - CI"
            src="https://github.com/jorenham/scipy-stubs/workflows/CI/badge.svg"
        />
    </a>
    <a href="https://github.com/pre-commit/pre-commit">
        <img
            alt="scipy-stubs - pre-commit"
            src="https://img.shields.io/badge/pre--commit-enabled-teal?logo=pre-commit"
        />
    </a>
    <a href="https://github.com/KotlinIsland/basedmypy">
        <img
            alt="scipy-stubs - basedmypy"
            src="https://img.shields.io/badge/basedmypy-checked-fd9002"
        />
    </a>
    <a href="https://detachhead.github.io/basedpyright">
        <img
            alt="scipy-stubs - basedpyright"
            src="https://img.shields.io/badge/basedpyright-checked-42b983"
        />
    </a>
    <a href="https://github.com/astral-sh/ruff">
        <img
            alt="scipy-stubs - ruff"
            src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json"
        />
    </a>
</p>

______________________________________________________________________

## Highlights

- Works out-of-the-box
  - all that's needed is to [install `scipy-stubs`](#installation)
  - does not require a `mypy` plugin or other configuration
  - available on [PyPI](https://pypi.org/project/scipy-stubs/) and [conda-forge](https://anaconda.org/conda-forge/scipy-stubs)
- Improves IDE suggestions and autocompletion
  - ... even if you don't use static typing in your code
  - no additional plugins required
- 0% runtime overhead
  - not even a single import is required
- 100% coverage of the [public SciPy API](https://docs.scipy.org/doc/scipy-1.14.1/reference/index.html)
  - also covers most of the private API
- Precise type-hinting of dtypes and [shape-types](https://github.com/numpy/numpy/issues/16544)
  - works with all "array-likes" and "dtype-likes"
  - many of the functions that return an array are *shape-typed*
  - shape-typing is optional: all functions still accept arrays with unknown shape-type
- Type-checker agnostic
  - works with at least [`mypy`](https://github.com/KotlinIsland/basedmypy),
    [`pyright`](https://github.com/DetachHead/basedpyright)/pylance and [`ruff`](https://github.com/astral-sh/ruff)
  - ... even in the strict mode
  - compatible with the [Python Typing Spec](https://typing.readthedocs.io/en/latest/spec/index.html)
- [SPEC 0](https://scientific-python.org/specs/spec-0000/) compliant
  - Supports Python ≥ 3.10
  - Supports NumPy ≥ 1.23.5

<!-- NOTE: SciPy permalinks to the following `#installation` anchor; don't modify it! -->

## Installation

The source code is currently hosted on GitHub at [github.com/jorenham/scipy-stubs](https://github.com/jorenham/scipy-stubs/).

Binary distributions are available at the [Python Package Index (PyPI)](https://pypi.org/project/scipy-stubs/) and on
[conda-forge](https://anaconda.org/conda-forge/scipy-stubs).

### Using pip (PyPI)

To install from the [PyPI](https://pypi.org/project/scipy-stubs/), run:

```bash
pip install scipy-stubs
```

In case you haven't installed `scipy` yet, both can be installed with:

```bash
pip install scipy-stubs[scipy]
```

### Using conda (conda-forge)

To install using Conda from the [conda-forge channel](https://anaconda.org/conda-forge/scipy-stubs), run:

```bash
conda install conda-forge::scipy-stubs
```

## Supported static type-checkers

1. [`basedpyright`](https://github.com/DetachHead/basedpyright) (recommended)
1. [`basedmypy`](https://github.com/KotlinIsland/basedmypy)
1. [`pyright`](https://pyright.readthedocs.io/en/latest/index.html)
1. [`mypy`](https://mypy.readthedocs.io/en/stable/index.html) (not recommended, see [erictraut/mypy_issues](https://github.com/erictraut/mypy_issues))

For validation and testing, `scipy-stubs` primarily uses [`basedmypy`](https://github.com/KotlinIsland/basedmypy) (a `mypy` fork)
and [`basedpyright`](https://github.com/DetachHead/basedpyright) (a `pyright` fork).
They are in generally stricter than `mypy` and `pyright`, so you can assume compatibility with `mypy` and `pyright` as well.
But if you find that this isn't the case, then don't hesitate to open an issue or submit a pull request.

## Versioning and requirements

The versioning scheme of `scipy-stubs` includes the compatible `scipy` version as `{scipy_version}.{stubs_version}`.
Even though `scipy-stubs` doesn't enforce an upper bound on the `scipy` version, later `scipy` versions aren't guaranteed to be
fully compatible.

The supported range of `numpy` versions are specified in [`SPEC 0`](https://scientific-python.org/specs/spec-0000/), which
`scipy-stubs` aims to follow as close as feasible.

Currently, `scipy-stubs` has one required dependency: [`optype`](https://github.com/jorenham/optype).
This is essential for `scipy-stubs` to work properly, as it relies heavily on it for annotating (shaped) array-likes,
scalar-likes, shape-typing in general, and much more.
At the moment, `scipy-stubs` requires the latest version `optype`.

The exact version requirements are specified in the [`pyproject.toml`](pyproject.toml).

## `scipy` coverage

The entire public API of `scipy` is **fully annotated** and **verifiably valid**.
For the most part, this can also be said about `scipy`'s private API and other internal machinery.

However, a small portion uses `Untyped` (an alias of `Any`) as "placeholder annotations".
In those cases static type-checkers won't do any type-checking, and won't bother you with errors or warnings.

The following table shows the (subjective) proportion of `scipy-stubs` that is(n't) annotated with `Untyped`, ranging
from 🌑 (100% `Untyped`) to 🌕 (0% `Untyped`).

| `scipy._`     | `ruff` & `flake8-pyi` | `stubtest` | `basedmypy` | `basedpyright` | phase |
| :------------ | :-------------------: | :--------: | :---------: | :------------: | :---: |
| `cluster`     |          ✔️           |     ✔️     |     ✔️      |       ✔️       |  🌕   |
| `constants`   |          ✔️           |     ✔️     |     ✔️      |       ✔️       |  🌝   |
| `datasets`    |          ✔️           |     ✔️     |     ✔️      |       ✔️       |  🌝   |
| `fft`         |          ✔️           |     ✔️     |     ✔️      |       ✔️       |  🌕   |
| `fftpack`     |          ✔️           |     ✔️     |     ✔️      |       ✔️       |  🌕   |
| `integrate`   |          ✔️           |     ✔️     |     ✔️      |       ✔️       |  🌕   |
| `interpolate` |          ✔️           |     ✔️     |     ✔️      |       ✔️       |  🌕   |
| `io`          |          ✔️           |     ✔️     |     ✔️      |       ✔️       |  🌕   |
| `linalg`      |          ✔️           |     ✔️     |     ✔️      |       ✔️       |  🌕   |
| ~`misc`~      |          ✔️           |     ✔️     |     ✔️      |       ✔️       |  🌕   |
| `ndimage`     |          ✔️           |     ✔️     |     ✔️      |       ✔️       |  🌕   |
| `odr`         |          ✔️           |     ✔️     |     ✔️      |       ✔️       |  🌕   |
| `optimize`    |          ✔️           |     ✔️     |     ✔️      |       ✔️       |  🌕   |
| `signal`      |          ✔️           |     ✔️     |     ✔️      |       ✔️       |  🌔   |
| `sparse`      |          ✔️           |     ✔️     |     ✔️      |       ✔️       |  🌓   |
| `spatial`     |          ✔️           |     ✔️     |     ✔️      |       ✔️       |  🌕   |
| `special`     |          ✔️           |     ✔️     |     ✔️      |       ✔️       |  🌕   |
| `stats`       |          ✔️           |     ✔️     |     ✔️      |       ✔️       |  🌕   |
| *`_lib`*      |          ✔️           |     ✔️     |     ✔️      |       ✔️       |  🌕   |

Currently, only `signal` and `sparse` contain `Untyped` annotations.

## Podcast (AI generated)

<https://github.com/user-attachments/assets/adbec640-2329-488b-bda2-d9687c6b1f7b>

## See also

- [scipy/scipy#21614](https://github.com/scipy/scipy/issues/21614): On why `scipy-stubs` is a separate package, and not part of
  `scipy` (yet).
- [microsoft/python-type-stubs#321](https://github.com/microsoft/python-type-stubs/pull/321): The removal of Microsoft's
  `scipy-stubs` — that used to be bundled with Pylance — in favor of `scipy-stubs`.
- [`optype`](https://github.com/jorenham/optype): The fundamental typing package that made `scipy-stubs` possible.
- [`basedpyright`](https://github.com/detachhead/basedpyright): The recommended type-checker to use with `scipy-stubs`.
- [`basedmypy`](https://github.com/KotlinIsland/basedmypy): A [less-broken](https://github.com/erictraut/mypy_issues) `mypy` fork,
  with a bunch of cool extra features.
