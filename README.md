<h1 align="center">scipy-stubs</h1>

<p align="center">
    Typing stubs for <a href="https://github.com/scipy/scipy">SciPy</a>.
</p>

<p align="center">
    <a href="https://pypi.org/project/scipy-stubs/">
        <img
            alt="scipy-stubs - PyPI"
            src="https://img.shields.io/pypi/v/scipy-stubs?style=flat&color=olive"
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
            alt="scipy-stubs - dependencies"
            src="https://img.shields.io/librariesio/github/jorenham/scipy-stubs?style=flat&color=violet"
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
    <!-- TODO -->
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

## Installation

```shell
pip install scipy-stubs
```

## Development Status

| `scipy._`     | `ruff` & `flake8-pyi` | `stubtest` | `basedmypy` | `basedpyright` | phase |
| :------------ | :-------------------: | :--------: | :---------: | :------------: | :---: |
| `_lib`        |          ✔️           |     ✔️     |     ✔️      |       ✔️       |  🌕   |
| `cluster`     |          ✔️           |     ✔️     |     ✔️      |       ✔️       |  🌕   |
| `constants`   |          ✔️           |     ✔️     |     ✔️      |       ✔️       |  🌝   |
| `datasets`    |          ✔️           |     ✔️     |     ✔️      |       ✔️       |  🌝   |
| `fft`         |          ✔️           |     ✔️     |     ✔️      |       ✔️       |  🌔   |
| `fftpack`     |          ✔️           |     ✔️     |     ✔️      |       ✔️       |  🌔   |
| `integrate`   |          ✔️           |     ✔️     |     ✔️      |       ✔️       |  🌔   |
| `interpolate` |          ✔️           |     ✔️     |     ✔️      |       ✔️       |  🌓   |
| `io`          |          ✔️           |     ✔️     |     ✔️      |       ✔️       |  🌕   |
| `linalg`      |          ✔️           |     ✔️     |     ✔️      |       ✔️       |  🌕   |
| ~`misc`~      |          ✔️           |     ✔️     |     ✔️      |       ✔️       |  🌕   |
| `ndimage`     |          ✔️           |     ✔️     |     ✔️      |       ✔️       |  🌕   |
| `odr`         |          ✔️           |     ✔️     |     ✔️      |       ✔️       |  🌕   |
| `optimize`    |          ✔️           |     ✔️     |     ✔️      |       ✔️       |  🌓   |
| `signal`      |          ✔️           |     ✔️     |     ✔️      |       ✔️       |  🌒   |
| `sparse`      |          ✔️           |     ✔️     |     ✔️      |       ✔️       |  🌒   |
| `spatial`     |          ✔️           |     ✔️     |     ✔️      |       ✔️       |  🌕   |
| `special`     |          ✔️           |     ✔️     |     ✔️      |       ✔️       |  🌓   |
| `stats`       |          ✔️           |     ✔️     |     ✔️      |       ✔️       |  🌕   |

## Version Compatibility

### Type-checkers

For validation and testing, `scipy-stubs` primarily uses [`basedmypy`](https://github.com/KotlinIsland/basedmypy) (a `mypy` fork)
and [`basedpyright`](https://github.com/DetachHead/basedpyright) (a `pyright` fork).
They are in generally stricter than `mypy` and `pyright`, so you can assume compatibility with `mypy` and `pyright` as well.
But if you find that this isn't the case, then don't hesitate to open an issue or submit a pull request.

### Required dependencies

The versioning scheme of `scipy-stubs` includes the compatible `scipy` version as `{scipy_version}.{stubs_version}`.
Even though `scipy-stubs` doesn't enforce an upper bound on the `scipy` version, later `scipy` versions aren't guaranteed to be
fully compatible.

Apart from `scipy`'s own dependencies, (e.g. `numpy`), the only other required dependency is
[`optype`](https://github.com/jorenham/optype), which itself only depends on `typing_extensions`.

The exact version requirements are specified in the [`pyproject.toml`](pyproject.toml).
