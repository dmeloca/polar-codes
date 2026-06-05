import numpy as np


class BECChannel:
    """
    Binary Erasure Channel.

    Transmit pipeline: binary codeword -> erase each bit with prob epsilon -> LLRs.
    Erased bits get LLR = 0.0 (no information). Received bits get LLR = +-inf.
    LLR sign convention: positive = likely 0, negative = likely 1.
    """

    def __init__(self, epsilon: float) -> None:
        self.epsilon = epsilon

    def transmit(self, codeword: np.ndarray) -> np.ndarray:
        llrs = np.where(codeword == 0, np.inf, -np.inf).astype(np.float64)
        erased = np.random.random(len(codeword)) < self.epsilon
        llrs[erased] = 0.0
        return llrs
