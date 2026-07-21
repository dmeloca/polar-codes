"""
5G NR CRC-11 (3GPP TS 38.212 section 5.1)
    gCRC11(D) = D^11 + D^10 + D^9 + D^5 + 1
Convention: MSB-first, zero init, no bit reflection, no output XOR.
Bits are plain Python ints (0/1); message is a list/array, MSB first (a0 first).
"""
import numpy as np

from .polys import GEN_POLYS, full_form_int

class CRC:
    def __init__(self, crc_type: str, K: int):
        if crc_type not in GEN_POLYS:
            raise ValueError(f"The requested CRC type, {crc_type} is not in the 5G standards. All available are {list(GEN_POLYS.keys())}")
        self.type: str = crc_type
        self.R, self.taps = GEN_POLYS[self.type]
        self.G: int = full_form_int(self.type)
        self.K: int = K
        self.H: np.ndarray = self.parity_check_matrix(K)

    def crc_bits(self, msg: str) -> list[int]:
        """
        Streaming LFSR division: return the R CRC bits (MSB-first) for `msg`,
        which are the
        """
        reg = 0 #*register
        #*Divide the message (m)
        for b in msg: #*most-significant bit to least
            reg = (reg << 1) | int(b)
            if (reg >> self.R) & 1: #*degree-R term appeared -> subtract g (XOR)
                reg ^= self.G
        #*Continue the division by appending r zeros to the message, which is
        #*multiplying it by x^r
        for _ in range(self.R):
            reg <<= 1
            if (reg >> self.R) & 1:
                reg ^= self.G
        rem = reg & ((1 << self.R) - 1)
        return [(rem >> (self.R - 1 - k)) & 1 for k in range(self.R)]
    
    def encode(self, msg: str):
        """
        Systematic codeword = [message | 11 CRC bits]. Creates a list of the
        integers inside msg and appends the CRC bits.
        """
        return list(map(int, msg)) + self.crc_bits(msg)

    def check(self, codeword: list[int]):
        """Receiver test: divide the whole word; True iff remainder is zero."""
        reg = 0
        for b in codeword:
            reg = (reg << 1) | int(b)
            if (reg >> self.R) & 1:
                reg ^= self.G
        return (reg & ((1 << self.R) - 1)) == 0


    def parity_check_matrix(self, K):
        """H (size R x K) such that `H x codeword == 0 (mod 2)`. Its column j is
        computed as = x^(K-1-j) mod g."""
        H = np.zeros((self.R, K), dtype=int)
        for j in range(K):
            power = K - 1 - j
            reg = 1
            for _ in range(power): #*computes remainder by repeated shift-reduce
                reg <<= 1
                if (reg >> self.R) & 1:
                    reg ^= self.G
            for r in range(self.R):
                H[r, j] = (reg >> (self.R - 1 - r)) & 1
        return H
    
    def check_batch(self, candidates: np.ndarray) -> np.ndarray:
        """
        Checks which codewords in the L `candidates` belong to the code by
        calculating their syndromes.

        Parameters
        ----------
        - candidates: np.ndarray
            (L, K) array of 0/1. Each candidate is expected to be the
            information bits of a codeword.
        """
        candidates = np.asarray(candidates, dtype=np.uint8)
        syndromes = (self.H @ candidates.T) % 2 #*(R, K) x (K, L) = (R, L)
        return ~syndromes.any(axis=0) #*arr of size L. True where syndrome == 0



# # ---- independent reference: explicit "append zeros then divide" ----
# #!Ask later what this was. Seems like another way to compute the CRC bits
# def crc_bits_reference(msg, G=G, R=R):
#     L = len(msg)
#     val = 0
#     for b in msg:
#         val = (val << 1) | int(b)
#     val <<= R                                   # append R zeros
#     for pos in range(L + R - 1, R - 1, -1):     # long division, high to low
#         if (val >> pos) & 1:
#             val ^= G << (pos - R)
#     rem = val & ((1 << R) - 1)
#     return [(rem >> (R - 1 - k)) & 1 for k in range(R)]


# if __name__ == "__main__":
#     rng = np.random.default_rng(0)
#     print(f"g(D) taps {TAPS} -> full int {G} = {hex(G)}, "
#           f"normal-form 11-bit poly = {hex(G & 0x7FF)}")

#     ok_methods = ok_zero = ok_H = ok_detect = True
#     for m in [256, 100, 1, 512]:
#         for _ in range(2000):
#             msg = rng.integers(0, 2, size=m).tolist()
#             c1 = crc_bits(msg)
#             c2 = crc_bits_reference(msg)
#             ok_methods &= (c1 == c2)                     # two engines agree
#             cw = encode(msg)
#             ok_zero &= check(cw)                         # clean codeword passes
#             H = parity_check_matrix(len(cw))
#             ok_H &= not (H @ np.array(cw) % 2).any()     # H @ c == 0
#             e = cw.copy(); e[rng.integers(0, len(cw))] ^= 1
#             ok_detect &= not check(e)                    # single-bit error caught

#     print(f"two independent CRC engines agree ...... {ok_methods}")
#     print(f"encoded codewords divide to zero ........ {ok_zero}")
#     print(f"parity-check matrix H @ codeword == 0 ... {ok_H}")
#     print(f"all single-bit errors detected .......... {ok_detect}")

#     demo = [1, 0, 1, 1, 0, 0, 1, 0]
#     print(f"\nexample msg {demo}\n  CRC bits  {crc_bits(demo)}\n"
#           f"  codeword  {encode(demo)}\n  check()   {check(encode(demo))}")