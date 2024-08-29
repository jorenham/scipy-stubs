<h1 align="center">scipy-stubs</h1>

<p align="center">
    Building blocks for precise & flexible type hints.
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
    <!-- TODO -->
    <!-- <a href="https://github.com/jorenham/scipy-stubs/actions?query=workflow%3ACI">
        <img
            alt="scipy-stubs - CI"
            src="https://github.com/jorenham/scipy-stubs/workflows/CI/badge.svg"
        />
    </a> -->
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

## Installation

The `scipy-stubs` package is available as on PyPI:

```shell
pip install scipy-stubs
```

> [!IMPORTANT]
> This project is in the early development stage, and is not ready for production use.

## Development Progress

| Module                            | Stubs status    |
|---------------------------------- |---------------- |
| `scipy`                           | 1: skeleton     |
| `scipy._lib`                      | 1: skeleton     |
| `scipy._lib.uarray`               | 1: skeleton     |
| `scipy._lib.array_api_compat`     | 1: partial      |
| `scipy.cluster`                   | 1: skeleton     |
| `scipy.constants`                 | 3: ready        |
| `scipy.datasets`                  | 1: skeleton     |
| `scipy.fft`                       | 1: skeleton     |
| `scipy.fft._pocketfft`            | 1: skeleton     |
| `scipy.fftpack`                   | 1: skeleton     |
| `scipy.integrate`                 | 1: skeleton     |
| `scipy.integrate._ivp`            | 1: skeleton     |
| `scipy.interpolate`               | 1: skeleton     |
| `scipy.io`                        | 1: skeleton     |
| `scipy.io.arff`                   | 1: skeleton     |
| `scipy.io.matlab`                 | 1: skeleton     |
| `scipy.linalg`                    | 3: ready        |
| `scipy.misc`                      | 0: missing      |
| `scipy.ndimage`                   | 1: skeleton     |
| `scipy.odr`                       | 1: skeleton     |
| `scipy.optimize`                  | 1: skeleton     |
| `scipy.optimize.cython_optimize`  | 0: missing      |
| `scipy.optimize.zeros`            | 0: missing      |
| `scipy.signal`                    | 1: skeleton     |
| `scipy.signal.windows`            | 1: skeleton     |
| `scipy.sparse`                    | 1: skeleton     |
| `scipy.sparse.csgraph`            | 1: skeleton     |
| `scipy.sparse.linalg`             | 1: skeleton     |
| `scipy.spatial`                   | 2: partial      |
| `scipy.spatial.distance`          | 3: ready        |
| `scipy.special`                   | 3: ready        |
| `scipy.special.cython_special`    | 2: partial      |
| `scipy.stats`                     | 2: partial      |
| `scipy.stats.contingency`         | 1: skeleton     |
| `scipy.stats.distributions`       | 3: ready        |
| `scipy.stats.mstats`              | 1: skeleton     |
| `scipy.stats.qmc`                 | 1: skeleton     |
| `scipy.stats.sampling`            | 1: skeleton     |

Status labels:

0. missing (failed stubgen)
1. skeleton (mostly succesful stubgen)
2. partial (incomplete/broad annotations)
3. ready (complete & valid annotations, untested)
4. done (complete, valid, tested, and production-ready)
