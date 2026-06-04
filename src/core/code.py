from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class PolarCode:
    """
    Polar code definition
    """

    N: int
    K: int
    frozen_positions: frozenset[int]
    n: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.frozen_positions = frozenset(self.frozen_positions)
        _n = int(np.log2(self.N))

        if 2**_n != self.N:
            raise ValueError("N must be a power of 2.")
        if self.K >= self.N:
            raise ValueError("K must be smaller than N.")
        self.n = _n

    @property
    def rate(self) -> float:
        return self.K / self.N
