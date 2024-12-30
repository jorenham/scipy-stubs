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
- 100% coverage of the [public SciPy API](https://docs.scipy.org/doc/scipy/reference/index.html)
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

It's also possible to install both `scipy` and `scipy-stubs` together through the bundled
[`scipy-typed`](https://anaconda.org/conda-forge/scipy-typed) package:

```bash
conda install conda-forge::scipy-typed
```

### Packages overview

<table>
  <tr>
    <th rowspan="2" colspan="2"></th>
    <th colspan="2">Python packages</th>
  </tr>
  <tr>
    <th><code>scipy-stubs</code></th>
    <th><code>scipy</code> + <code>scipy-stubs</code></td>
  </tr>
  <tr>
    <th>PyPI</th>
    <th align="right"><code>pip install {}</code></th>
    <td><code>scipy-stubs</code></td>
    <td><code>scipy-stubs[scipy]</code></td>
  </tr>
  <tr>
    <th>conda-forge</th>
    <th align="right"><code>conda install conda-forge::{}</code></th>
    <td><code>scipy-stubs</code></td>
    <td><code>scipy-typed</code></td>
  </tr>
</table>

## Versioning and requirements

The versioning scheme of `scipy-stubs` includes the compatible `scipy` version as `{scipy_version}.{stubs_version}`.
Even though `scipy-stubs` doesn't enforce an upper bound on the `scipy` version, later `scipy` versions aren't guaranteed to be
fully compatible.

There are no additional restrictions enforced by `scipy-stubs` on the `numpy` requirements.
For `scipy[-stubs]` `1.14.*` and `1.15.*` that is `numpy >= 1.23.5`.

Currently, `scipy-stubs` has one required dependency: [`optype`](https://github.com/jorenham/optype).
This is essential for `scipy-stubs` to work properly, as it relies heavily on it for annotating (shaped) array-likes,
scalar-likes, shape-typing in general, and much more.
At the moment, `scipy-stubs` requires the latest version `optype`.

The exact version requirements are specified in the [`pyproject.toml`](pyproject.toml).

## Supported static type-checkers

1. [`basedpyright`](https://github.com/DetachHead/basedpyright) (recommended)
1. [`basedmypy`](https://github.com/KotlinIsland/basedmypy)
1. [`pyright`](https://pyright.readthedocs.io/en/latest/index.html)
1. [`mypy`](https://mypy.readthedocs.io/en/stable/index.html) (not recommended, see
   [mypy_issues](https://github.com/erictraut/mypy_issues))

For validation and testing, `scipy-stubs` primarily uses [`basedmypy`](https://github.com/KotlinIsland/basedmypy) (a `mypy` fork)
and [`basedpyright`](https://github.com/DetachHead/basedpyright) (a `pyright` fork).
They are in generally stricter than `mypy` and `pyright`, so you can assume compatibility with `mypy` and `pyright` as well.
But if you find that this isn't the case, then don't hesitate to open an issue or submit a pull request.

## `scipy` coverage

The entire public API of `scipy` is **fully annotated** and **verifiably valid**.
For the most part, this can also be said about `scipy`'s private API and other internal machinery.

Note that this does not mean that all annotations are optimal, and some might even be incorrect. If you encounter this, it would
help a lot if you could open an issue or a PR for it.

## Contributing

There are many ways that you can help improve `scipy-stubs`, for example

- reporting issues, bugs, or other unexpected outcomes
- improving the `.pyi` stubs (see [CONTRIBUTING.md](https://github.com/jorenham/scipy-stubs/blob/master/CONTRIBUTING.md))
- type-testing (see the `README.md` in [`scipy-stubs/tests`](https://github.com/jorenham/scipy-stubs/tree/master/tests) for the
  specifics)
- write new documentation (usage examples, guides, tips & tricks, FAQ, etc.), or e.g. improve this `README.md`
- help spread the word on `scipy-stubs`, so that more can benefit from using it

## AI generated Podcast

### Typing in SciPy

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
