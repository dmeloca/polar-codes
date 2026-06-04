import numpy as np

from src.core import PolarCode, PolarEncoder

code = PolarCode(N=8, K=4, frozen_positions=frozenset({0, 1, 2, 4}))
encoder = PolarEncoder(code)

info_bits = np.array([1, 0, 1, 1], dtype=np.uint8)
codeword = encoder.encode(info_bits)

print(f"Code rate : {code.rate}")
print(f"Info bits : {info_bits}")
print(f"Codeword  : {codeword}")
