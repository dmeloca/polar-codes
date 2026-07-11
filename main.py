import numpy as np

from src.channels import AWGNChannel, BECChannel
from src.core import PolarCode, PolarEncoder, awgn_frozen_set, bec_frozen_set, sc_decode, sc_decode_llr

N, K = 8, 4
info_bits = np.array([1, 0, 1, 0], dtype=np.uint8)

# ── BEC ──────────────────────────────────────────────────────────────────────
epsilon = 0.3
frozen_bec = bec_frozen_set(N, K, epsilon)
codeword_bec = PolarEncoder(PolarCode(N, K, frozen_bec)).encode(info_bits)
# lrs_bec = BECChannel(epsilon).transmit(codeword_bec, mode='lrs')
llrs_bec = BECChannel(epsilon).transmit(codeword_bec, mode='llrs') #*Log LRs
# raw_estimate = sc_decode(frozen_bec, codeword_bec, lrs_bec, BECChannel(epsilon))
raw_estimate = sc_decode_llr(frozen_bec, llrs_bec)
clean_estimate = np.array([int(raw_estimate[i]) for i in range(raw_estimate.size) if i not in frozen_bec])

print("-- BEC --")
print(f"Target   : {info_bits}")
print(f"Frozen   : {sorted(frozen_bec)}")
print(f"Codeword : {codeword_bec}")
print(f"LLRs     : {llrs_bec}")
print(f"Raw estimate: {raw_estimate}")
print(f"Estimate : {clean_estimate}")

# ── AWGN ─────────────────────────────────────────────────────────────────────
snr_db = 2.0
frozen_awgn = awgn_frozen_set(N, K, snr_db)
codeword_awgn = PolarEncoder(PolarCode(N, K, frozen_awgn)).encode(info_bits)
llrs_awgn = AWGNChannel(snr_db).transmit(codeword_awgn)

print("\n-- AWGN --")
print(f"Frozen   : {sorted(frozen_awgn)}")
print(f"Codeword : {codeword_awgn}")
print(f"LLRs     : {np.round(llrs_awgn, 2)}")
