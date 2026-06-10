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

    def transmit(self, codeword: np.ndarray, mode: str = "llrs") -> np.ndarray:
        if mode == "lrs":
            #* 0 => (1-e)/0 = inf  |  1 => 0/(1-e) = 0
            lrs = np.where(codeword == 0, np.inf, 0.0).astype(np.float64)
            erased = np.random.random(len(codeword)) < self.epsilon
            lrs[erased] = 1.0 #*Uncertainty
            return lrs
        else:
            llrs = np.where(codeword == 0, np.inf, -np.inf).astype(np.float64)
            erased = np.random.random(len(codeword)) < self.epsilon
            llrs[erased] = 0.0
            return llrs
    
    def w_rule(self, target_bit: float, given_bit: float) -> float:
        if (target_bit == 0 and given_bit == 1) or (target_bit == 1 and given_bit == 0):
            return 0
        elif target_bit != 0 and target_bit != 1:
            return self.epsilon
        else:
            return 1 - self.epsilon
