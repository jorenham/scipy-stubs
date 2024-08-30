from typing import Final
from typing_extensions import LiteralString

__all__ = "distcont", "distdiscrete", "invdistcont", "invdistdiscrete"

distcont: Final[
    list[
        tuple[
            LiteralString,
            # 0 - 4 parameters (`'gausshyper'`)
            tuple[()] | tuple[float] | tuple[float, float] | tuple[float, float, float] | tuple[float, float, float, float],
        ]
    ]
]
invdistcont: Final[
    list[
        tuple[
            LiteralString,
            # 0 - 4 parameters (`'gausshyper'`)
            tuple[()] | tuple[float] | tuple[float, float] | tuple[float, float, float] | tuple[float, float, float, float],
        ]
    ]
]

distdiscrete: Final[
    list[
        tuple[
            LiteralString,
            # 1 - 4 parameters (`'nchypergeom_fisher'` and `'nchypergeom_wallenius'`)
            tuple[float] | tuple[float, float] | tuple[int, float, float] | tuple[int, int, int, float],
        ]
    ]
]
invdistdiscrete: Final[
    list[
        tuple[
            LiteralString,
            # 1 - 4 parameters (`'nchypergeom_fisher'` and `'nchypergeom_wallenius'`)
            tuple[float] | tuple[float, float] | tuple[int, float, float] | tuple[int, int, int, float],
        ]
    ]
]
