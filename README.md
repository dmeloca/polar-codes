# LLR-domain Successive Cancellation decoder

`src/core/sc.py` already contained a Successive Cancellation (SC) decoder,
`sc_decode`, that works with plain **likelihood ratios** (LR = W(y|0)/W(y|1))
and combines them multiplicatively inside the recursive function `L_i`. This
document explains the new decoder that was added alongside it —
`sc_decode_llr` — which performs the exact same recursion but in the
**log-likelihood ratio** (LLR) domain.

## Why an LLR version?

LR values naturally range over `[0, +inf]`, which forces `L_i` to special-case
combinations of `0` and `inf` to avoid `0 * inf` / `inf / inf` (see the long
`if/elif` chains in `L_i`). LLR values (`LLR = ln(LR)`) range over
`(-inf, +inf)` instead, and the two standard polar-code combining rules
(`f` for "check nodes", `g` for "bit nodes") are expressed with `tanh`/`atanh`
and plain addition, both of which saturate smoothly at `+-inf` — so almost
none of that special-casing is needed. This is also the representation used
in practice (e.g. the `AWGNChannel`/`BECChannel` classes already emit LLRs,
with the convention **positive LLR ⇒ bit is more likely 0**).

## New functions

### `f_llr(a, b)` — check-node combining rule

```python
f_llr(a, b) = 2 * atanh( tanh(a/2) * tanh(b/2) )
```

This is the exact LLR of `u_i XOR u_{i+1}` given the LLRs `a`, `b` of `u_i`
and `u_{i+1}` individually (the "boxplus" operator, `⊞`). It replaces the
odd-`i` branch of `L_i`, which computed the equivalent quantity in the LR
domain as `(left*right + 1) / (left + right)`.

Because `tanh` saturates to `+-1` as its argument grows, `f_llr` is
well-defined for any combination of finite values and `+-inf` without extra
branches (e.g. `f_llr(inf, b) == b`, `f_llr(inf, inf) == inf`,
`f_llr(inf, -inf) == -inf`).

### `g_llr(a, b, u)` — bit-node combining rule

```python
g_llr(a, b, u) = (-1)^u * a + b
```

This is the LLR of `u_{i+1}` given the LLRs `a`, `b` of the two branches and
a hard decision `u` on `u_i`. It replaces the even-`i` branch of `L_i`, which
computed `left^(1-2u) * right` in the LR domain.

The only case plain addition can't resolve is `a = +inf` combined with
`b = -inf` (after applying the sign) — two branches that are *certain* of
contradictory bit values. That is mathematically undefined (`nan` in
floating point), and `g_llr` resolves it to `0.0` (complete uncertainty)
instead of propagating a `nan`.

### `llr_i(i, llr_1N, u_estimate)` — the recursive LLR

This is the LLR analogue of `L_i`. It keeps **exactly the same index
bookkeeping** as `L_i`:

- `N = llr_1N.size`, recursion bottoms out at `N == 1` by returning the raw
  channel LLR (`llr_1N[0]`) — the log-domain equivalent of `L_i`'s
  `return lr_1N[0]` base case.
- For even `i`: split `u_estimate` (which represents `u_1^{i-1}`) into
  `u_prev = u_estimate[:-1]` and the sign bit `u_exp = u_estimate[-1]`, derive
  `sum_estimate`/`even_estimate` from `u_prev`'s odd/even sub-indices exactly
  as `L_i` does, recurse on both halves of `llr_1N`, and combine with
  `g_llr(left, right, u_exp)`.
- For odd `i` (`N != 1`): derive `sum_estimate`/`even_estimate` directly from
  `u_estimate`'s odd/even sub-indices, recurse on both halves of `llr_1N`,
  and combine with `f_llr(left, right)`.

Unlike `L_i`, `llr_i` doesn't need the `codeword` or `base_channel`
parameters — they were unused in `L_i`'s actual logic (only ever sliced and
passed down, or referenced in dead/commented-out code), so they're dropped
here.

### `h_i_llr(i, llr_1N, u_estimate)` — hard decision

```python
return 0 if llr_i(...) >= 0 else 1
```

Mirrors `h_i`, but compares against `0` (the LLR domain's neutral point)
instead of `1` (the LR domain's neutral point, since `LLR = ln(LR)` and
`ln(1) = 0`).

### `sc_decode_llr(frozen_bits, llr_1N)` — top-level decode loop

Line-for-line the same traversal as `sc_decode`: walk `i` from `0` to
`N - 1`, force frozen positions to `FROZEN_BIT_VALUE`, otherwise call
`h_i_llr(i + 1, llr_1N, u_is)` (the `+1` converts the 0-indexed loop variable
to the 1-indexed position `L_i`/`llr_i` expect), and append the hard decision
to the growing estimate before moving to the next position.

## Usage

```python
from src.channels import BECChannel, AWGNChannel
from src.core import sc_decode_llr

llrs = AWGNChannel(snr_db=4.0).transmit(codeword)     # or BECChannel(eps).transmit(codeword, mode="llrs")
estimate = sc_decode_llr(frozen_bits, llrs)
info_bits = [int(estimate[i]) for i in range(estimate.size) if i not in frozen_bits]
```

## A pre-existing bug this surfaced (now fixed)

While validating `sc_decode_llr` against `PolarEncoder`, non-frozen bit
positions `1` and `3` (0-indexed) never affected the decoded output for an
`N=8` code — regardless of channel noise. The same happened with the
original `sc_decode`, so it wasn't specific to the LLR version; it was a
mismatch between `src/core/generator.py`'s generator matrix and the index
convention `L_i`/`llr_i` assume.

`build_generator` used to return `G_N = F^{⊗n}` with **no bit-reversal
permutation**. Because of how `np.kron` associates
(`G_n = kron(G_{n-1}, F)`), that generator maps `u`'s *even/odd* indices onto
the codeword's *even/odd* indices. But `L_i`/`llr_i` (and the standard
Arikan SC recursion they implement) assume the generator maps `u`'s odd/even
indices onto the codeword's *contiguous first/second half* instead — i.e.
they assume the standard construction `G_N = B_N F^{⊗n}`, with `B_N` the
bit-reversal permutation.

This was almost certainly the root cause behind the
`bug(sc/): not decoding correctly` commit. `build_generator` now permutes the
rows of `F^{⊗n}` by bit-reversed index before returning it, so
`PolarEncoder`/`build_generator` produce the standard `G_N = B_N F^{⊗n}`
without any change needed in `encoder.py`. Both `sc_decode` and
`sc_decode_llr` now decode all 16 messages correctly for a noiseless
`N=8, K=4` BEC, and get 100%/99.99% correct over 8000 randomized trials on
AWGN (4 dB) and BEC (ε=0.05) respectively.

# BP decoder
Equations used to update the `R` and `L` matrices (or the graph's nodes' values):

> R-sweep (left-to-right)
>
> - if i is top (j = i + d):   R[i][s] = f( R[i][s-1] , L[j][s] + R[j][s-1] )
> 
> - if i is bottom (j = i - d):  R[i][s] = f( R[j][s-1] , L[j][s] ) + R[i][s-1]

> L-sweep (right-to-left):
>
> - if i is top   (j = i + d):   L[i][s-1] = f( L[i][s] , L[j][s] + R[j][s-1] )
>
> - if i is bottom (j = i - d):  L[i][s-1] = f( R[j][s-1] , L[j][s] ) + L[i][s]

Note: here, "top" means it is a "+" node, while bottom means it is an "=" node.