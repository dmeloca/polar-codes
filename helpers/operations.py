import numpy as np


def kronecker_power(kernel: np.ndarray, n: int) -> np.ndarray:
    result = np.array([[1]], dtype=int)
    for _ in range(n):
        result = np.kron(result, kernel)
    return result % 2


def polarize(z: float, n: int) -> list[float]:
    if n == 0:
        return [z]
    left: list[float] = polarize(2 * z - z**2, n - 1)
    right: list[float] = polarize(z**2, n - 1)
    return left + right
