import math
from typing import Tuple

import numpy as np

try:
    import cupy as cp  # optional; only needed for use_gpu=True
    _HAS_CUPY = True
except ImportError:
    cp = None
    _HAS_CUPY = False


def _xp_of(a):
    """
    Return the array module (numpy or cupy) that owns `a`.
    Falls back to numpy when cupy isn't installed.
    """
    if _HAS_CUPY:
        return cp.get_array_module(a)
    return np

FROZEN_BIT_VALUE: int = 0


def _f_llr(a, b):
    """
    Batched, exact log-domain check-node combining rule (boxplus).

    Mirrors sc.f_llr's formula 2 * atanh(tanh(a/2) * tanh(b/2)) but works
    elementwise over arrays and dispatches to numpy or cupy via
    cp.get_array_module, so the same code runs on either backend.
    """
    xp = _xp_of(a)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 2.0 * xp.arctanh(xp.tanh(a / 2.0) * xp.tanh(b / 2.0))


def _g_llr(a, b, u):
    """
    Batched, exact log-domain bit-node combining rule: (-1)^u * a + b.
    u is expected to be a 0/1 array broadcast-compatible with a and b.
    NaN (from inf + -inf) collapses to 0.0, matching sc.g_llr's convention.
    """
    out = (1.0 - 2.0 * u) * a + b
    xp = _xp_of(out)
    return xp.where(xp.isnan(out), 0.0, out)


def _softplus(x):
    """
    Numerically stable log(1 + exp(x)) via logaddexp(0, x).
    """
    xp = _xp_of(x)
    return xp.logaddexp(xp.zeros_like(x), x)


def _trailing_zeros(j: int) -> int:
    """Index of the lowest set bit of j (j must be > 0)."""
    return (j & -j).bit_length() - 1


def _scl_core(frozen_bits: frozenset, llr_1N, L: int, xp, clip: bool):
    """
    Shared O(N log N) SC/SCL engine, in the log-LR domain.

    Computes exactly the same quantities as sc.llr_i's recursion (same f/g
    rules, same index bookkeeping, same +-inf conventions), but keeps the
    intermediate LLRs between successive bit decisions instead of rebuilding
    the whole recursion tree from the channel at every bit.

    The layout follows Arikan's index recursion directly. Depth d holds one
    LLR per *node* (2**d of them), not one per channel position: depth 0 is
    the root, whose single LLR decides u_j, and depth n is the channel. Node
    t at depth d covers channel positions [t*N/2**d, (t+1)*N/2**d) and has
    children 2t and 2t+1 at depth d+1. The LLR that node t needs is the one
    for its own bit index floor(j / 2**d), so depth d only goes stale once
    every 2**d bits. Recomputing it then costs 2**d combines, i.e. N per
    depth across the whole word, hence O(N log N) overall rather than the
    O(N**2) of re-descending from the channel per bit.

    Returns (u_paths, PM, L_active): the full length-N decision vector per
    surviving path, their path metrics (lower = more likely), and how many
    of the L slots are live.
    """
    llr = xp.asarray(llr_1N, dtype=xp.float64)
    if clip:
        #*Clip infinities (BEC emits +-inf on unerased positions) so metrics stay finite.
        #*A single +-inf cost would saturate the path metric and destroy discrimination
        #*between candidates on later bits (argsort ties break arbitrarily). +-30 matches
        #*the clip bp.py's _initialize_L uses and keeps arctanh/tanh well away from +-1.
        llr = xp.where(llr == xp.inf, 30.0, llr)
        llr = xp.where(llr == -xp.inf, -30.0, llr)
    N: int = int(llr.size)
    n: int = int(round(math.log2(N)))

    #*alpha[d]: current LLR of each of the 2**d nodes at depth d, per path.
    #*beta[d]: that node's own last decided pair of bits (even slot, odd slot);
    #*the even slot is what its g-combine consumes on the following bit.
    #*Depth n is the channel itself, which is path-independent and never
    #*rewritten, so it stays a plain (N,) array outside these lists.
    alpha = [xp.zeros((L, 1 << d), dtype=xp.float64) for d in range(n)]
    beta = [xp.zeros((L, 1 << d, 2), dtype=xp.int8) for d in range(n)]
    u_paths = xp.zeros((L, N), dtype=xp.int8)
    PM = xp.zeros(L, dtype=xp.float64)
    L_active: int = 1

    #*Children of the deepest node layer are channel observations; slicing them
    #*once here keeps them out of the per-bit path.
    chan_even = llr[0::2]
    chan_odd = llr[1::2]

    arange_cache: dict = {}

    def _arange(m: int):
        if m not in arange_cache:
            arange_cache[m] = xp.arange(m)
        return arange_cache[m]

    for j in range(N):
        #*Refresh every depth whose bit index floor(j / 2**d) just advanced.
        #*j and j-1 differ exactly in bits 0..tz, so depths above tz are still
        #*holding the LLR they were asked for last bit.
        d_start: int = n - 1 if j == 0 else min(_trailing_zeros(j), n - 1)
        for d in range(d_start, -1, -1):
            if d == n - 1:
                a, b = chan_even, chan_odd
            else:
                a = alpha[d + 1][:L_active, 0::2]
                b = alpha[d + 1][:L_active, 1::2]
            if (j >> d) & 1:
                alpha[d][:L_active] = _g_llr(a, b, beta[d][:L_active, :, 0])
            else:
                alpha[d][:L_active] = _f_llr(a, b)

        llrs = alpha[0][:L_active, 0]  #*root LLR of u_j, one per path

        cost0 = _softplus(-llrs)  #*cost of choosing u_j = 0
        cost1 = _softplus(llrs)   #*cost of choosing u_j = 1

        if j in frozen_bits:
            #*No branching: force u_j = FROZEN_BIT_VALUE and pay the associated cost
            PM[:L_active] += cost0
            u_paths[:L_active, j] = FROZEN_BIT_VALUE
        else:
            #*Branching: each path spawns two candidates (u_j = 0, u_j = 1).
            cand_metrics = xp.concatenate([PM[:L_active] + cost0,
                                           PM[:L_active] + cost1])
            cand_u = xp.concatenate([xp.zeros(L_active, dtype=xp.int8),
                                     xp.ones(L_active, dtype=xp.int8)])
            cand_parent = xp.concatenate([_arange(L_active), _arange(L_active)])

            #*Secondary tiebreak: when candidate metrics are equal (e.g., both +inf
            #*because internal LLRs saturated to +-inf via g-node cascades even with
            #*clipped channel input), prefer the u that matches sign(llr) so L=1 SCL
            #*still agrees with sc.h_i_llr's `0 if llr >= 0 else 1` rule.
            h = (llrs < 0).astype(xp.int8)  #*SC's hard decision per active path
            tie_keys = xp.concatenate([h, 1 - h])  #*0 = preferred, 1 = alternative
            #*lexsort: LAST key is primary, so we sort by metric first, then by tie_key.
            order = xp.lexsort(xp.stack([tie_keys, cand_metrics]))

            new_L: int = min(L, 2 * L_active)
            keep = order[:new_L]
            parent_idx = cand_parent[keep]

            #*Surviving paths inherit their parent's whole decoder state, so the
            #*kept LLRs/partial sums have to be gathered alongside the decisions.
            gathered = u_paths[parent_idx]
            gathered[:, j] = cand_u[keep]
            u_paths[:new_L] = gathered
            PM[:new_L] = cand_metrics[keep]
            for d in range(n):
                alpha[d][:new_L] = alpha[d][parent_idx]
                beta[d][:new_L] = beta[d][parent_idx]

            L_active = new_L

        #*Push the decision back down the tree. A node hands its children a
        #*completed pair (b_even, b_odd) as (b_even ^ b_odd) on the left and
        #*b_odd on the right -- the same u_o ^ u_e / u_e split sc.llr_i slices
        #*out of its history -- and only once that pair exists, i.e. while the
        #*bit of j at that depth is 1.
        if n:
            beta[0][:L_active, 0, j & 1] = u_paths[:L_active, j]
            d = 0
            while d < n - 1 and (j >> d) & 1:
                b0 = beta[d][:L_active, :, 0]
                b1 = beta[d][:L_active, :, 1]
                parity = (j >> (d + 1)) & 1
                beta[d + 1][:L_active, 0::2, parity] = b0 ^ b1
                beta[d + 1][:L_active, 1::2, parity] = b1
                d += 1

    return u_paths, PM, L_active


def scl_decode(
    frozen_bits: frozenset,
    llr_1N: np.ndarray,
    L: int = 8,
    use_gpu: bool = False,
) -> Tuple[np.ndarray, bool]:
    """
    Successive Cancellation List decoder in the log-LR domain.

    Generalises sc.sc_decode_llr by carrying up to L candidate paths and pruning
    to the L most-likely at every non-frozen bit. With L=1 this reproduces
    sc_decode_llr bit-for-bit (same f_llr/g_llr, same tie-breaking).

    Parameters
    ----------
    frozen_bits : frozenset of int
        Indices of frozen positions in the u vector.
    llr_1N : np.ndarray of float, shape (N,)
        Channel log-likelihood ratios (positive = bit likely 0).
    L : int, default 8
        List size. L=1 => plain SC.
    use_gpu : bool, default False
        If True, allocate on the GPU via cupy. Note that the sweep is inherently
        sequential over the N bits and the per-bit arrays are small, so kernel
        launch overhead dominates and this is typically *slower* than the CPU
        path; bpl_decode is the decoder that actually benefits from the GPU.

    Returns
    -------
    info_bits : np.ndarray of uint8, shape (K,)
        Decoded information bits with frozen positions stripped, matching
        the shape/contract of bp_decode and bpl_decode.
    success : bool
        Always True. There is no CRC or parity check available to invalidate
        a survivor, so the best-metric path is returned unconditionally.
    """
    if use_gpu and not _HAS_CUPY:
        raise RuntimeError("scl_decode(use_gpu=True) requires cupy to be installed.")
    xp = cp if use_gpu else np

    u_paths, PM, L_active = _scl_core(frozen_bits, llr_1N, L, xp, clip=True)

    best = int(xp.argmin(PM[:L_active]))
    u_hat = u_paths[best]
    if use_gpu:
        u_hat = cp.asnumpy(u_hat)

    N = int(u_hat.size)
    info_mask = np.ones(N, dtype=bool)
    info_mask[list(frozen_bits)] = False
    return u_hat[info_mask].astype(np.uint8), True
