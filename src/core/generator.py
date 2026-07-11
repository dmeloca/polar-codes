import numpy as np


def _bit_reverse(x: int, n_bits: int) -> int:
    r = 0
    for _ in range(n_bits):
        r = (r << 1) | (x & 1)
        x >>= 1
    return r


def build_generator(n: int) -> np.ndarray:
    """
    Build G_N = B_N @ F^{(kron) n}, the standard Arikan polar code generator
    matrix. The bit-reversal permutation B_N is required so that the
    recursive SC decoder (which splits the received LLRs into contiguous
    halves at every stage) lines up with the generator's recursive
    structure; without it, G_N = F^{(kron) n} alone maps u's odd/even
    indices onto the codeword's odd/even indices instead of onto its
    first/second half, and the decoder cannot recover about half the bits.
    """
    F = np.array([[1, 0], [1, 1]], dtype=np.uint8)
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F)
    G = G % 2
    N = G.shape[0]
    perm = [_bit_reverse(i, n) for i in range(N)]
    return G[perm]
