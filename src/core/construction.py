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
    # φ is decreasing, so grow the bracket until it actually contains the root
    # rather than assuming a fixed cap: μ⁺ = 2μ doubles every stage, so the means
    # run past any constant bound once n is large enough (μ > 1000 at N=128, 3 dB)
    lo, hi = 0.0, 1.0
    while _phi(hi) > y and hi < 1e12:
        lo, hi = hi, hi * 2.0
    for _ in range(60):
        mid = (lo + hi) * 0.5
        if _phi(mid) > y:
            lo = mid
        else:
            hi = mid
    return (lo + hi) * 0.5


def _awgn_reliabilities(n: int, esn0_db: float) -> np.ndarray:
    """
    Evolve LLR means through n polarization stages.
    esn0_db is Es/N0 in dB for BPSK-AWGN, matching AWGNChannel's convention
    (see its docstring for the Eb/N0 relation).
    """
    esn0 = 10.0 ** (esn0_db / 10.0)
    # initial LLR mean for BPSK: μ₀ = 2/σ² = 4·(Es/N0)
    mu = np.array([4.0 * esn0])
    for _ in range(n):
        mu_new = np.empty(2 * len(mu))
        # bad (−): φ(μ⁻) = 1 − (1 − φ(μ))², which is < μ. Evaluated in the factored
        # form φ·(2 − φ) because 1 − (1 − φ)² cancels to exactly 0 once φ is small
        # enough to vanish against the 1.0, which would send μ⁻ to +inf.
        mu_new[0::2] = np.array([_phi_inv(p * (2.0 - p)) for p in map(_phi, mu)])
        mu_new[1::2] = 2.0 * mu  # good (+): μ⁺ = 2μ
        mu = mu_new
    return mu


def awgn_frozen_set(N: int, K: int, esn0_db: float) -> frozenset[int]:
    """
    Gaussian approximation construction for AWGN with BPSK.
    esn0_db is Es/N0 in dB, the same convention AWGNChannel takes -- pass both
    the same value, or the code is designed for a different operating point
    than the channel delivers.
    Freezes the N-K least reliable synthetic channels.
    """
    n = int(np.log2(N))
    mu = _awgn_reliabilities(n, esn0_db)
    # ascending sort: first N-K positions have smallest μ (least reliable → frozen)
    return frozenset(int(i) for i in np.argsort(mu)[: N - K])
