from typing_extensions import deprecated

@deprecated("will be removed in SciPy 1.15.0")
def rvs_ratio_uniforms(
    pdf: object,
    umax: object,
    vmin: object,
    vmax: object,
    size: object = 1,
    c: object = 0,
    random_state: object = None,
) -> object: ...
