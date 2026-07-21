import numpy as np

from ..channels.awgn import AWGNChannel
from ..channels.bec import BECChannel
from .crc.crc import CRC

FROZEN_BIT_VALUE: int = 0

class SyntheticVectorChannel:
    def __init__(self, size: int, base_channel: AWGNChannel | BECChannel):
        self.base_channel: AWGNChannel | BECChannel = base_channel
        self.size: int = size
        if np.log2(size) != int(np.log2(size)):
            raise ValueError(f"Size of the channel must be a power of 2, but is: {self.size}")
        self.n: int = int(np.log2(size))
        self.G_N: np.matrix = None
        self.B_N: np.matrix = None
        self.F_N: np.matrix = None

    def get_G_N(self, u_1N: np.ndarray) -> np.matrix:
        """
        Compute u_1N @ G_N using the fact that G_N = B_N @ F^n
        """
        u_permuted: np.ndarray = self._reverse(u_1N)
        u_polarized: np.ndarray = self.calc_F_N(u_permuted)
        return u_polarized
        
    def _reverse(self, vect: np.ndarray) -> np.matrix:
        """
        Bit-reverse a vector.
        """
        def _bit_reverse(x: int, n_bits: int) -> int:
            """
            Generate the bit-reversion of a number, i.e., convert it to binary,
            flip it, and convert it back to decimal representation. 
            
            For example, 4 -> 2, because 4 = (0100)_2 -> (0010)_2 = 2
            """
            r = 0
            for _ in range(n_bits):
                r = (r << 1) | (x & 1)
                x >>= 1
            return r
        
        perm = [_bit_reverse(i, self.n) for i in range(self.size)]
        new_vect: np.ndarray = np.zeros(shape=vect.shape)
        for i in range(vect.shape):
            new_vect[i] = vect[perm[i]]

        return new_vect

    def calc_F_N(self, vect: np.ndarray) -> np.matrix:
        """
        Calculates F_N from scratch if not store in self. Else, returns the
        previously calculated matrix.
        """
        if self.F_N == None:
            #*
            F: np.ndarray = np.array(data=[[1, 0], [1, 1]])
            for i in range(self.n - 1): #*Getting F^{n}
                self.F_N = np.kron(self.F_N, F)

        return vect @ self.F_N

def L_i(i: int, codeword: np.ndarray, lr_1N: np.ndarray, u_estimate: np.ndarray, base_channel: AWGNChannel | BECChannel) -> float:
    """

    Params
    ------
    - u_estimate: np.ndarray
        Corresponds to u_1^{i-1} or u_1^{2*half_i - 2}
    """
    N: int = int(lr_1N.size)
    half_N: int = int(N/2)

    # if N == 1: #*Base case
    #     w1 = base_channel.w_rule(lr_1N[0], 1)
    #     w0 = base_channel.w_rule(lr_1N[0], 0)
    #     return np.inf if w1 == 0 else w0 / w1

    half_i: int = 0

    # print("u_estimate:", u_estimate)
    # print("codeword:", codeword)

    # import pdb; pdb.set_trace()
    # print("-------")
    # print("u_estimate:", u_estimate, u_estimate.size, "i:", i)
    # print(N, "lr_estimate", lr_1N)
    # if u_estimate.size == 0:
    #     print("Warning, u_estimate is size 0:", u_estimate)
    if i % 2 == 0:
        half_i = int(i / 2)
        #*The following naming conventions may sound contradictory. However,
        #*they are done that way since the indices start on 1 in the paper.
        #*The slicing goes up to u_estimate.size - 1 (exclusive) because we are
        #*taking u_1^{2i-2} from u_1^{2i-1} (u_estimate)
        # print("Even i", i)
        #*Taking u_1^{2i-2} (prev) as estimate and u_1^{2i-1} for the exponent
        u_prev: np.ndarray = u_estimate[:-1]
        u_exp: np.ndarray = u_estimate[-1]
        if u_prev.size == 0:
            sum_estimate: np.ndarray = np.array([])
            even_estimate: np.ndarray = np.array([])
        else:
            odd_estimate: np.ndarray = u_prev[::2] #*0, 2, ... -> 1, 3, ...
            even_estimate: np.ndarray = u_prev[1::2] #* 1, 3, ... -> 2, 4, ...
            sum_estimate: np.ndarray = (odd_estimate + even_estimate) % 2
        # print("Odd estimate:", odd_estimate)
        # print("Even estimate:", even_estimate)
        # print("Sum estimate:", even_estimate)
        exp = (1 - 2*u_exp)
        left = L_i(half_i, codeword[0:half_N], lr_1N[0:half_N], sum_estimate, base_channel)
        right = L_i(half_i, codeword[half_N:N], lr_1N[half_N:N], even_estimate, base_channel)
        # print("Left:", left)
        # print("Right:", right)

        if left == np.inf:
            if right == 0:
                if exp == 1:
                    result = 1
                else:
                    result = 0
            elif right == np.inf and exp == -1:
                result = 1
            else:
                result = (float(left)**exp) * right
        elif left == 0:
            if right == 0 and exp == -1:
                result = np.inf
            elif right == np.inf:
                if exp == 1:
                    result = 1
                else:
                    result = np.inf
            else:
                result = (float(left)**exp) * right
        else:
            result = (float(left)**exp) * right

        # print("Result:", result)
        return result
        
        # if left == 0 and u_exp == 1:
        #     left = np.inf
        # else:
        #     left = left**(1 - 2*u_exp)
        # print("First prod", left)
        # print("First prod:", left, "| Second prod:", right)
        # print(result)
        # print(f"L_i(i={i}, N={N}) -> {result}")
        
    else:
        if N != 1: #*the i != 1 condition is redundant
            # print("Odd i", i)
            half_i: int = int((i + 1) / 2) #*Because i = ((i + 1) / 2) - 1
            #*The slicing goes up to u_estimate.size (exclusive) because
            #*2 * half_i - 2 is even
            # print(type(half_N), half_N)
            if u_estimate.size == 0:
                even_estimate: np.ndarray = np.array([])
                sum_estimate: np.ndarray = np.array([])
            else:
                odd_estimate: np.ndarray = u_estimate[::2] #*0, 2, ... -> 1, 3, ...
                even_estimate: np.ndarray = u_estimate[1::2] #* 1, 3, ... -> 2, 4, ...
                sum_estimate: np.ndarray = (odd_estimate + even_estimate) % 2
            # print("Odd estimate:", odd_estimate)
            # print("Even estimate:", even_estimate)
            # print("Sum estimate:", even_estimate)
            left: float = L_i(half_i, codeword[0:half_N], lr_1N[0:half_N], sum_estimate, base_channel)
            right: float = L_i(half_i, codeword[half_N:N], lr_1N[half_N:N], even_estimate, base_channel)
            # if (left == np.inf and right == 0) or (right == np.inf and left == 0):
            #     print("warning")
            if left == np.inf and right == 0:
                result = 0
            elif left == 0 and right == np.inf:
                result = 0
            elif left == np.inf and right == np.inf:
                result = np.inf
            elif (left == np.inf and right == 1) or (left == 1 and right == np.inf):
                result = 1
            elif left == 0 and right == 0:
                result = np.inf
            else:
                num = left*right+1
                den = left+right
                # print("Left:", left)
                # print("Right:", right)
                if den == 0 and num != 0:
                    result = np.inf
                else:
                    result = num / den

            # print("Result:", result)
            return result
            
        else: #*i = 1 AND N = 1; last recursive step
            return lr_1N[0]
            # lr = lr_1N[0]
            # if lr == 0:
            #     return 1.0 #*uncertainty
            # elif lr == np.inf: #*Received 1
            #     return np.inf
            # else: #*Received 0 (lr == -np.inf)
            #     return 0.0
            
            # one_rule: float = base_channel.w_rule(lr_1N[0], 1)
            # if one_rule == 0:
            #     # print(f"L_i(i={i}, N={N}) -> {np.inf}")
            #     return np.inf
            # else:
            #     zero_rule: float = base_channel.w_rule(lr_1N[0], 0)
            #     # print(f"L_i(i={i}, N={N}) -> {zero_rule / one_rule}")
            #     return zero_rule / one_rule
                

def h_i(i: int, codeword: np.ndarray, lr_1N: np.ndarray, u_estimate: np.ndarray, base_channel: AWGNChannel | BECChannel) -> int:
    # print("----------------")
    lr: float = L_i(i, codeword, lr_1N, u_estimate, base_channel)
    if lr >= 1:
        return 0
    else:
        return 1

def sc_decode(frozen_bits: np.ndarray, codeword: np.ndarray, lr_1N: np.ndarray, base_channel: AWGNChannel | BECChannel) -> np.ndarray:
    # print(y.size)
    u_i: float = None
    u_is: list[float] = []

    for i in range(lr_1N.size): #*Traverse the indices
        u_i = None
        if i in frozen_bits:
            u_i = FROZEN_BIT_VALUE
        else:
            # if i == 7:
            u_i = h_i(i+1, codeword, lr_1N, np.array(u_is), base_channel)
        u_is.append(u_i)

    return np.array(u_is)

def f_llr(a: float, b: float) -> float:
    """
    Check-node LLR combining rule ("boxplus"), used for the odd branch
    (i.e., u_i XOR u_{i+1}) of the L_i recursion.

    Exact formula: 2 * atanh( tanh(a/2) * tanh(b/2) ). tanh saturates
    smoothly to +-1 as its argument grows without bound, so this expression
    stays well-defined for a, b in {-inf, ..., +inf} and never needs the
    inf/0 special-casing that the likelihood-ratio version of L_i requires.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(2 * np.arctanh(np.tanh(a / 2) * np.tanh(b / 2)))

def g_llr(a: float, b: float, u: int) -> float:
    """
    Bit-node LLR combining rule, used for the even branch (i.e., u_{i+1}
    alone, given a hard decision on u_i) of the L_i recursion.

    result = (-1)^u * a + b. The only case this can't resolve is when a and
    b are +inf and -inf after applying the sign: that means the two branches
    are certain of contradictory bit values, which is treated as complete
    uncertainty (LLR = 0) rather than left as NaN.
    """
    total: float = ((1 - 2 * u) * a) + b
    return 0.0 if np.isnan(total) else float(total)

def llr_i(i: int, llr_1N: np.ndarray, u_estimate: np.ndarray) -> float:
    """
    Recursively compute the LLR L_N^{(i)}(llr_1N, u_estimate).

    This follows the exact same index bookkeeping as L_i (same halving of N,
    same odd/even-i case split, same odd/even/sum estimate slicing) but
    combines the two branch LLRs with f_llr/g_llr instead of multiplying
    likelihood ratios, so no base_channel/codeword bookkeeping is needed.

    Params
    ------
    - u_estimate: np.ndarray
        Corresponds to u_1^{i-1} or u_1^{2*half_i - 2}
    """
    N: int = int(llr_1N.size)
    half_N: int = int(N / 2)

    if N == 1: #*Base case: the LLR straight from the channel
        return float(llr_1N[0])

    if i % 2 == 0:
        half_i: int = int(i / 2)
        #*Taking u_1^{2i-2} (prev) as estimate and u_1^{2i-1} for the sign
        u_prev: np.ndarray = u_estimate[:-1]
        u_exp: int = int(u_estimate[-1])
        if u_prev.size == 0:
            sum_estimate: np.ndarray = np.array([])
            even_estimate: np.ndarray = np.array([])
        else:
            odd_estimate: np.ndarray = u_prev[::2] #*0, 2, ... -> 1, 3, ...
            even_estimate: np.ndarray = u_prev[1::2] #* 1, 3, ... -> 2, 4, ...
            sum_estimate: np.ndarray = (odd_estimate + even_estimate) % 2

        left: float = llr_i(half_i, llr_1N[0:half_N], sum_estimate)
        right: float = llr_i(half_i, llr_1N[half_N:N], even_estimate)
        return g_llr(left, right, u_exp)
    else:
        half_i: int = int((i + 1) / 2) #*Because i = ((i + 1) / 2) - 1
        if u_estimate.size == 0:
            even_estimate: np.ndarray = np.array([])
            sum_estimate: np.ndarray = np.array([])
        else:
            odd_estimate: np.ndarray = u_estimate[::2] #*0, 2, ... -> 1, 3, ...
            even_estimate: np.ndarray = u_estimate[1::2] #* 1, 3, ... -> 2, 4, ...
            sum_estimate: np.ndarray = (odd_estimate + even_estimate) % 2

        left: float = llr_i(half_i, llr_1N[0:half_N], sum_estimate)
        right: float = llr_i(half_i, llr_1N[half_N:N], even_estimate)
        return f_llr(left, right)

def h_i_llr(i: int, llr_1N: np.ndarray, u_estimate: np.ndarray) -> int:
    """
    Hard-decide u_i from its LLR. LLR = ln(W(y|0)/W(y|1)), so a
    non-negative LLR favors 0 and a negative LLR favors 1.
    """
    llr: float = llr_i(i, llr_1N, u_estimate)
    return 0 if llr >= 0 else 1

def sc_decode_llr(frozen_bits: np.ndarray, llr_1N: np.ndarray, ca: bool = False) -> np.ndarray:
    """
    Successive Cancellation decoder operating on log-likelihood ratios.

    Workflow: decide u_i in index order, force frozen
    bits to FROZEN_BIT_VALUE, and grow u_estimate one hard decision at a
    time, but drives the recursion through llr_i/h_i_llr instead of the
    likelihood-ratio based L_i/h_i.
    """
    u_i: int = None
    u_is: list[int] = []

    for i in range(llr_1N.size): #*Traverse the indices
        u_i = None
        if i in frozen_bits:
            u_i = FROZEN_BIT_VALUE
        else:
            u_i = h_i_llr(i + 1, llr_1N, np.array(u_is))
        u_is.append(u_i)

    return np.array(u_is)

def ca_scl_finish(path_bits, path_metrics, crc: CRC, m: int):
    """
    Runs CRC on all candidates and returns the first candidate to pass the CRC
    along with a boolean indicating it or the closest one to the codeword.
    path_bits: (L, K), path_metrics: (L,) lower = more likely.
    """
    passed = crc.check_batch(path_bits)          # (L,) bool, one matmul
    if not passed.any():
        best = np.argmin(path_metrics)            # no CRC survivor -> fall back to best metric
    else:
        candidates = np.where(passed)[0]
        best = candidates[np.argmin(path_metrics[candidates])]
    return path_bits[best, :m], passed[best]

if __name__ == "__main__":
    frozen_bits: list[int] = [0, 1, 2, 4]
    y_received: np.ndarray = np.array([0, 0, 1, 0, 0, 0, 0, 0])

    llrs = BECChannel(0.3).transmit(y_received, mode="llrs")
    llr_estimate: np.ndarray = sc_decode_llr(np.array(frozen_bits), llrs)
    print("--------------")
    print("LLR-domain final estimate:", llr_estimate)
    no_frozen_llr_estimate = [int(llr_estimate[i]) for i in range(llr_estimate.size) if i not in frozen_bits]
    print("Without the frozen bits:", no_frozen_llr_estimate)
    print("--------------")

    lrs = BECChannel(0.3).transmit(y_received, mode="lrs")
    estimate: np.ndarray = sc_decode(np.array(frozen_bits), np.array(lrs), BECChannel(epsilon=0.5))
    print("--------------")
    print("Final estimate:", estimate)
    no_frozen_estimate = [int(estimate[i]) for i in range(estimate.size) if i not in frozen_bits]
    print("Without the frozen bits:", no_frozen_estimate)
    print("--------------")
