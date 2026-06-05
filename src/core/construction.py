import numpy as np

# ── BEC (Bhattacharyya bounds) ────────────────────────────────────────────────


def _bec_reliabilities(n: int, epsilon: float) -> np.ndarray:
    z = np.array([epsilon])
    for _ in range(n):
        z_new = np.empty(2 * len(z))
        z_new[0::2] = 2.0 * z - z**2  # degraded (bad) sub-channel
        z_new[1::2] = z**2  # upgraded (good) sub-channel
        z = z_new
    return z


def bec_frozen_set(N: int, K: int, epsilon: float) -> frozenset[int]:
    """
    Bhattacharyya construction for a BEC with erasure probability epsilon.
    Freezes the N-K least reliable synthetic channels.
    """
    n = int(np.log2(N))
    z = _bec_reliabilities(n, epsilon)
    # ascending sort: first K positions are most reliable (info bits)
    return frozenset(int(i) for i in np.argsort(z)[K:])


# ── AWGN (Gaussian approximation) ────────────────────────────────────────────


def _phi(mu: float) -> float:
    """φ(μ) ≈ E[tanh(Z/2)] for Z ~ N(μ, 2μ); piecewise approximation."""
    if mu < 1e-10:
        return 1.0
    if mu >= 10.0:
        return float(
            np.sqrt(np.pi / mu) * np.exp(-mu / 4.0) * (1.0 - 10.0 / (7.0 * mu))
        )
    return float(np.exp(-0.4527 * mu**0.86 + 0.0218))


def _phi_inv(y: float) -> float:
    """Inverse of _phi via bisection."""
    if y >= 1.0:
        return 0.0
    if y <= 0.0:
        return float("inf")
    lo, hi = 0.0, 1000.0
    for _ in range(60):
        mid = (lo + hi) * 0.5
        if _phi(mid) > y:
            lo = mid
        else:
            hi = mid
    return (lo + hi) * 0.5


def _awgn_reliabilities(n: int, snr_db: float) -> np.ndarray:
    """
    Evolve LLR means through n polarization stages.
    snr_db is Eb/N0 in dB for BPSK-AWGN.
    """
    snr = 10.0 ** (snr_db / 10.0)
    # initial LLR mean for BPSK: μ₀ = 2/σ² = 4·(Eb/N0)
    mu = np.array([4.0 * snr])
    for _ in range(n):
        mu_new = np.empty(2 * len(mu))
        mu_new[0::2] = 2.0 * mu  # bad (−): μ⁻ = 2μ
        mu_new[1::2] = np.array(
            [_phi_inv(_phi(m) ** 2) for m in mu]
        )  # good (+): φ(μ⁺) = φ(μ)²
        mu = mu_new
    return mu


def awgn_frozen_set(N: int, K: int, snr_db: float) -> frozenset[int]:
    """
    Gaussian approximation construction for AWGN with BPSK.
    snr_db is Eb/N0 in dB.
    Freezes the N-K least reliable synthetic channels.
    """
    n = int(np.log2(N))
    mu = _awgn_reliabilities(n, snr_db)
    # ascending sort: first N-K positions have smallest μ (least reliable → frozen)
    return frozenset(int(i) for i in np.argsort(mu)[: N - K])
