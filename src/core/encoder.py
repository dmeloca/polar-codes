import numpy as np

from .code import PolarCode
from .generator import build_generator


class PolarEncoder:
    def __init__(self, code: PolarCode) -> None:
        self.code = code
        self.G = build_generator(code.n)
        self._info_positions = [
            i for i in range(code.N) if i not in code.frozen_positions
        ]

    def _build_u(self, information_bits: np.ndarray) -> np.ndarray:
        if len(information_bits) != self.code.K:
            raise ValueError("Incorrect information length.")
        u = np.zeros(self.code.N, dtype=np.uint8)
        u[self._info_positions] = information_bits
        return u

    def encode(self, information_bits: np.ndarray) -> np.ndarray:
        u = self._build_u(information_bits)
        x = (u @ self.G) % 2
        return x.astype(np.uint8)
