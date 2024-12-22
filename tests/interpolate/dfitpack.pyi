# test the deprecations of the "fortran functions" of the deprecated `scipy.interpolate.dfitpack` namespace

from scipy.interpolate import dfitpack

dfitpack.spalde(1, 1, 1, 1)  # pyright: ignore[reportDeprecated]
dfitpack.bispeu(1, 1, 1, 1, 1, 1, 1)  # pyright: ignore[reportDeprecated]
