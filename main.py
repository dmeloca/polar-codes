import numpy as np

from src.channels import AWGNChannel, BECChannel
from src.core import PolarCode, PolarEncoder, awgn_frozen_set, bec_frozen_set

N, K = 8, 4
info_bits = np.array([1, 0, 1, 1], dtype=np.uint8)

# ── BEC ──────────────────────────────────────────────────────────────────────
epsilon = 0.3
frozen_bec = bec_frozen_set(N, K, epsilon)
codeword_bec = PolarEncoder(PolarCode(N, K, frozen_bec)).encode(info_bits)
llrs_bec = BECChannel(epsilon).transmit(codeword_bec)

print("-- BEC --")
print(f"Frozen   : {sorted(frozen_bec)}")
print(f"Codeword : {codeword_bec}")
print(f"LLRs     : {llrs_bec}")

# ── AWGN ─────────────────────────────────────────────────────────────────────
snr_db = 2.0
frozen_awgn = awgn_frozen_set(N, K, snr_db)
codeword_awgn = PolarEncoder(PolarCode(N, K, frozen_awgn)).encode(info_bits)
llrs_awgn = AWGNChannel(snr_db).transmit(codeword_awgn)

print("\n-- AWGN --")
print(f"Frozen   : {sorted(frozen_awgn)}")
print(f"Codeword : {codeword_awgn}")
print(f"LLRs     : {np.round(llrs_awgn, 2)}")
