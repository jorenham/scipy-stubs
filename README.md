<h1 align="center">scipy-stubs</h1>

<p align="center">
    Type stubs for <a href="https://github.com/scipy/scipy">SciPy</a>.
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
    <!-- <a href="https://github.com/pre-commit/pre-commit">
        <img
            alt="scipy-stubs - pre-commit"
            src="https://img.shields.io/badge/pre--commit-enabled-teal?logo=pre-commit"
        />
    </a> -->
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

---

> [!NOTE]
> This project is in the alpha stage, so some annotations are missing, and some might
> be (slightly) incorrect.
> But either way, it's no problem to `scipy-stubs` at the moment:
> Type-checkers will (already) understand `scipy` a lot better *with* `scipy-stubs`,
> than without it.

## Installation

The `scipy-stubs` package is available as on PyPI:

```shell
pip install scipy-stubs
```

## Version Compatibility

### Type-checkers

For validation and testing, `scipy-stubs` primarily uses
[`basedmypy`](https://github.com/KotlinIsland/basedmypy) (a `mypy` fork) and
[`basedpyright`](https://github.com/DetachHead/basedpyright) (a `pyright` fork).
Because they are in generally stricter than `mypy` and `pyright`, they should be
compatible as well.
If you find that this is not the case, then don't hesitate to open an issue.

### Required dependencies

The versioning scheme of `scipy-stubs` includes the compatible `scipy` version.
Later versions might work too, but in case of API-changes, the stubs could be outdated.

Apart from `scipy`'s own dependencies (e.g. `numpy`), the only other
required dependency is [`optype`](https://github.com/jorenham/optype).

The exact version requirements are specified in the [`pyproject.toml`](pyproject.toml).

## Development Progress

| Package or module                 | Stubs status    |
|---------------------------------- |---------------- |
| `scipy.__init__`                  | 3: ready        |
| `scipy._lib`                      | 2: partial      |
| `scipy.cluster.vq`                | **4: done**     |
| `scipy.cluster.hierarchy`         | **4: done**     |
| `scipy.constants`                 | **4: done**     |
| `scipy.datasets`                  | **4: done**     |
| `scipy.fft`                       | 2: partial      |
| `scipy.fftpack`                   | 2: partial      |
| `scipy.integrate`                 | **4: done**     |
| `scipy.interpolate`               | 2: partial      |
| `scipy.io`                        | 2: partial      |
| `scipy.io.arff`                   | 2: partial      |
| `scipy.io.matlab`                 | 2: partial      |
| `scipy.linalg`                    | **4: done**     |
| ~`scipy.misc`~                    | **4: done**     |
| `scipy.ndimage`                   | 2: partial      |
| `scipy.odr`                       | 1: skeleton     |
| `scipy.optimize`                  | 2: partial      |
| `scipy.signal`                    | 2: partial      |
| `scipy.signal.windows`            | 1: skeleton     |
| `scipy.sparse`                    | 2: partial      |
| `scipy.sparse.csgraph`            | 2: partial      |
| `scipy.sparse.linalg`             | 2: partial      |
| `scipy.spatial`                   | 2: partial      |
| `scipy.spatial.distance`          | 3: ready        |
| `scipy.special`                   | **4: done**     |
| `scipy.special.cython_special`    | **4: done**     |
| `scipy.stats`                     | 2: partial      |
| `scipy.stats.contingency`         | 1: skeleton     |
| `scipy.stats.distributions`       | **4: done**     |
| `scipy.stats.mstats`              | 1: skeleton     |
| `scipy.stats.qmc`                 | 2: partial      |
| `scipy.stats.sampling`            | 1: skeleton     |
| `scipy.version`                   | **4: done**     |

Status labels:

- 0: missing (failed stubgen)
- 1: skeleton (mostly succesful stubgen)
- 2: partial (incomplete/broad annotations)
- 3: ready (complete & valid annotations, untested)
- 4: done (complete, valid, tested, and production-ready)
