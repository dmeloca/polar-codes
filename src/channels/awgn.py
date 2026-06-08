import numpy as np


class AWGNChannel:
    """
    BPSK-modulated AWGN channel.

    Transmit pipeline: binary codeword → BPSK symbols → additive noise → LLRs.
    LLR sign convention: positive = likely 0, negative = likely 1.
    """

    def __init__(self, snr_db: float) -> None:
        self.snr_db = snr_db
        snr = 10.0 ** (snr_db / 10.0)
        # For unit-amplitude BPSK: Eb = 1, σ² = 1/(2·Eb/N0)
        self._sigma = np.sqrt(1.0 / (2.0 * snr))

    def transmit(self, codeword: np.ndarray) -> np.ndarray:
        """
        Modulate, add noise, and return LLRs.

        Parameters
        ----------
        codeword : np.ndarray of uint8, shape (N,)

        Returns
        -------
        llrs : np.ndarray of float64, shape (N,)
            LLR_i = log P(y_i | x_i=0) / P(y_i | x_i=1) = 2·y_i / σ²
        """
        x = 1.0 - 2.0 * codeword.astype(np.float64)  # 0 → +1, 1 → −1
        y = x + np.random.normal(0.0, self._sigma, x.shape)
        return 2.0 * y / self._sigma ** 2
    
    def w_rule(self, target_bit: float, given_bit: float) -> float:
        pass
