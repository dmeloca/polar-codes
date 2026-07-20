import numpy as np


class AWGNChannel:
    """
    BPSK-modulated AWGN channel.

    Transmit pipeline: binary codeword → BPSK symbols → additive noise → LLRs.
    LLR sign convention: positive = likely 0, negative = likely 1.

    The SNR is **Es/N0** (energy per transmitted *symbol*), not Eb/N0 (energy per
    *information bit*). A channel carries symbols and has no notion of a code
    rate, so it cannot convert between the two on its own. For a rate R = K/N
    code the relation is

        Es/N0 (dB) = Eb/N0 (dB) + 10·log10(R)

    i.e. Es/N0 sits 3.01 dB *below* Eb/N0 at R = 1/2. Convert at the experiment
    level, where R is known. Published polar-code curves are conventionally
    plotted against Eb/N0, so compare against them only after converting.

    `awgn_frozen_set` in core/construction.py takes the same Es/N0 convention;
    the two must be given matching values or the code is designed for a
    different operating point than the channel delivers.
    """

    def __init__(self, esn0_db: float) -> None:
        self.esn0_db = esn0_db
        esn0 = 10.0 ** (esn0_db / 10.0)
        # For unit-amplitude BPSK: Es = 1, σ² = 1/(2·Es/N0)
        self._sigma = np.sqrt(1.0 / (2.0 * esn0))

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
