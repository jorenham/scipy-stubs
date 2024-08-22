# scipy-stubs

Typing stubs for [scipy](https://github.com/scipy/scipy).

> [!IMPORTANT]
> This project is in the early development stage, and is not ready for production use.

## Installation

```bash
pip install scipy-stubs
```

## Development Progress

| Module                            | Stubs status    |
|---------------------------------- |---------------- |
| `scipy`                           | 1: skeleton     |
| `scipy._lib`                      | 1: skeleton     |
| `scipy._lib._uarray`              | 0: missing      |
| `scipy.cluster`                   | 1: skeleton     |
| `scipy.constants`                 | 1: skeleton     |
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
| `scipy.linalg`                    | 1: skeleton     |
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
| `scipy.stats`                     | 1: skeleton     |
| `scipy.stats.contingency`         | 1: skeleton     |
| `scipy.stats.distributions`       | 1: skeleton     |
| `scipy.stats.mstats`              | 1: skeleton     |
| `scipy.stats.qmc`                 | 1: skeleton     |
| `scipy.stats.sampling`            | 1: skeleton     |

Status labels:

1. missing (failed stubgen)
2. skeleton (mostly succesful stubgen)
3. partial (incomplete/broad annotations)
4. ready (complete & valid annotations, untested)
5. done (complete, valid, tested, and production-ready)
