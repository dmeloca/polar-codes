import numpy as np


def build_generator(n: int) -> np.ndarray:
    F = np.array([[1, 0], [1, 1]], dtype=np.uint8)
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F)
    return G % 2
