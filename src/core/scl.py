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


def _batched_llr_i(i: int, llr_1N_batch, u_estimate_batch):
    """
    Batched version of sc.llr_i.

    Parameters
    ----------
    i : int
        1-indexed position (same convention as sc.llr_i).
    llr_1N_batch : array of shape (L_active, N)
        Channel LLRs sliced to the current sub-channel, replicated per path.
    u_estimate_batch : array of shape (L_active, k)
        Decision prefix per path (k = i-1 at the top level, halves as recursion descends).
    Returns
    -------
    array of shape (L_active,) with the LLR of u_i for each path.
    """
    xp = _xp_of(llr_1N_batch)
    L_active, N = llr_1N_batch.shape
    half_N = N // 2

    if N == 1:
        return llr_1N_batch[:, 0]

    if i % 2 == 0:
        half_i = i // 2
        u_prev = u_estimate_batch[:, :-1]
        u_exp = u_estimate_batch[:, -1]

        if u_prev.shape[1] == 0:
            sum_estimate = xp.zeros((L_active, 0), dtype=xp.int8)
            even_estimate = xp.zeros((L_active, 0), dtype=xp.int8)
        else:
            odd_estimate = u_prev[:, ::2]
            even_estimate = u_prev[:, 1::2]
            sum_estimate = (odd_estimate + even_estimate) % 2

        left = _batched_llr_i(half_i, llr_1N_batch[:, :half_N], sum_estimate)
        right = _batched_llr_i(half_i, llr_1N_batch[:, half_N:], even_estimate)
        return _g_llr(left, right, u_exp)

    half_i = (i + 1) // 2
    if u_estimate_batch.shape[1] == 0:
        sum_estimate = xp.zeros((L_active, 0), dtype=xp.int8)
        even_estimate = xp.zeros((L_active, 0), dtype=xp.int8)
    else:
        odd_estimate = u_estimate_batch[:, ::2]
        even_estimate = u_estimate_batch[:, 1::2]
        sum_estimate = (odd_estimate + even_estimate) % 2

    left = _batched_llr_i(half_i, llr_1N_batch[:, :half_N], sum_estimate)
    right = _batched_llr_i(half_i, llr_1N_batch[:, half_N:], even_estimate)
    return _f_llr(left, right)


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
        If True, allocate on the GPU via cupy and dispatch every helper through
        cp.get_array_module, mirroring the pattern used in bp.bpl_decode.

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

    llr_1N = xp.asarray(llr_1N, dtype=xp.float64)
    #*Clip infinities (BEC emits +-inf on unerased positions) so metrics stay finite.
    #*A single +-inf cost would saturate the path metric and destroy discrimination
    #*between candidates on later bits (argsort ties break arbitrarily). +-30 matches
    #*the clip bp.py's _initialize_L uses and keeps arctanh/tanh well away from +-1.
    llr_1N = xp.where(llr_1N == xp.inf, 30.0, llr_1N)
    llr_1N = xp.where(llr_1N == -xp.inf, -30.0, llr_1N)
    N = int(llr_1N.size)

    #*Broadcast the channel LLRs across all L path slots; only the first
    #*L_active rows are meaningful at any given moment.
    llr_1N_batch = xp.tile(llr_1N.reshape(1, N), (L, 1))

    #*Full-length decision matrix; column i is filled at loop iteration i.
    u_paths = xp.zeros((L, N), dtype=xp.int8)
    PM = xp.zeros(L, dtype=xp.float64)
    L_active = 1

    arange_L_cache: dict = {}

    def _arange(n: int):
        if n not in arange_L_cache:
            arange_L_cache[n] = xp.arange(n)
        return arange_L_cache[n]

    for i in range(N):
        #*LLR at u_i for each active path
        llrs = _batched_llr_i(i + 1, llr_1N_batch[:L_active], u_paths[:L_active, :i])

        cost0 = _softplus(-llrs)  #*cost of choosing u_i = 0
        cost1 = _softplus(llrs)   #*cost of choosing u_i = 1

        if i in frozen_bits:
            #*No branching: force u_i = FROZEN_BIT_VALUE and pay the associated cost
            PM[:L_active] += cost0
            u_paths[:L_active, i] = FROZEN_BIT_VALUE
            continue

        #*Branching: each path spawns two candidates (u_i = 0, u_i = 1).
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

        new_L = min(L, 2 * L_active)
        keep = order[:new_L]
        parent_idx = cand_parent[keep]

        #*Gather the prefixes of the surviving paths from their parents.
        #*fancy indexing returns a fresh array so the write below is safe.
        gathered = u_paths[parent_idx]
        gathered[:, i] = cand_u[keep]
        u_paths[:new_L] = gathered
        PM[:new_L] = cand_metrics[keep]

        L_active = new_L

    best = int(xp.argmin(PM[:L_active]))
    u_hat = u_paths[best]

    if use_gpu:
        u_hat = cp.asnumpy(u_hat)

    info_bits = np.array(
        [int(u_hat[j]) for j in range(N) if j not in frozen_bits],
        dtype=np.uint8,
    )
    return info_bits, True
