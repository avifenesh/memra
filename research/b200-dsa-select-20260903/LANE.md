# lane/b200-dsa-select-20260903: MEMRA_B200_DSA_SELECT, the exact multi-CTA k-pool selector

Follows `lane/b200-dsa-decode-20260902` (PRs #104/#110). That lane took `attn_gathered` and
`kpool_score` down ~10x and 4-7.6x on the pair and named this kernel as what the scorer had been
hiding. No B200 in this worktree; the local RTX 5090 was used for EXACTNESS and direction only.

## 1. What was wrong

`memra_mla_kpool_select_kernel` grids `t_q` blocks, so plain decode runs the whole selection on
**ONE CTA** - 0.68% of a 148-SM die - sweeping `n_pools` up to ten times (8 MSB-first radix
passes, an optional unique-resolution scan, the membership count, the emit). Depth-LINEAR in
`n_pools = t_kv / pool`. Measured on the 5090 at t_q=1: 83.0 us/layer at 128k, 149.8 at 256k,
**557.0 at 1M** - against a byte floor of `n_pools * 4 B` = 1 MB per sweep, ~0.13 us at 8 TB/s.
The gap is parallelism, nothing else.

## 2. Exact, not banded - and why that is a construction

The emitted plane is a pure function of ONE 64-bit number: `thr`, the `select_k`-th smallest
order key `(desc32(score) << 32) | pool_index`. The shipped kernel picks `thr` and then emits
every pool with `key(p) <= thr`. `desc32` is a strictly decreasing injection from finite f32 to
u32 and the low word is a unique pool index, so **keys are distinct** and "the k-th smallest" is
unambiguous. Reproducing `thr` bit-for-bit therefore reproduces the selection bit-for-bit. The
parallel pipeline computes the same key and runs the same membership test: this is a
launch-geometry change with an exact answer, not a tolerance. No banded class was needed and none
is claimed.

## 3. The pipeline: six launches, no grid barrier

1. `clear` - zero the histogram and the control words (the workspace is an uninitialised
   allocation, so this is what makes pass 0 well-defined).
2. `hist` pass 0 - 65536-bin histogram of the key's high word, top half, plus the finite count;
   the last-arriving CTA runs the descent and leaves the bins zeroed for pass 1.
3. `hist` pass 1 - the low half within the resolved prefix. `thr_s` is now exact.
4. `tie` - per-CTA count of the tie group, then the last CTA locates the rank-r smallest index
   among the tying pools. `thr_p` is now exact.
5. `count` - per-CTA membership counts; the last CTA exclusive-scans them.
6. `emit` - ascending, two levels of contiguous range, pool expansion, tail, -1 pad.

**Only 32 bits take a radix descent, not 64.** The key's low word is the pool index, which is
unique, so it never needs a histogram: once the score is resolved the threshold's index is simply
the rank-r smallest index among the pools that TIE at that score, and a rank selection over an
ascending contiguous range is a count plus a scan. Ties are the common case, not a corner - ReLU
zeroes every head whose query-pool dot is non-positive, so exact 0.0 ties are ORDINARY (the same
fact that makes the scorer's bit-identity load-bearing).

**No grid barrier and no spin.** Each multi-CTA pass ends with the last CTA to arrive running the
epilogue (`atomicAdd` on a done counter plus `__threadfence()`), the standard two-phase-reduction
idiom. Deliberately not a cooperative launch or a persistent-grid residency assumption: a
spin-wait that deadlocks if occupancy ever changes is not something to put on a serving path,
even behind a default-OFF door.

The 65536-bin descent is run by the last CTA with **all 256 threads** (each sums a 256-bin slice,
thread 0 scans the slice sums, the owning thread re-walks its slice). A single-threaded walk of
65536 bins would have cost ~131 us on its own - more than the kernel being replaced.

## 4. The gate, and its red arm

`crates/memra-engine/src/bin/dsa_select_gate.rs`, three bars:

- **EXACTNESS** - byte-identical `idx` at every score shape x context x width. The shapes hit the
  paths that decide the answer: heavy exact-0.0 ties, a plane where EVERY finite score ties (the
  threshold is then decided entirely by pool index), a sparse plane with fewer finite pools than
  the budget (the rank clamp), and an all-`-INFINITY` plane (nothing selected, tail only).
- **RED ARM** - the same pipeline with the resolved threshold deliberately lowered by one must
  produce a DIFFERENT plane; if it matches, the gate exits 1 however green everything else is.
- **ANCHOR** - the shipped radix kernel vs the in-tree reference selector, so the chain the door
  is compared against is pinned to the order definition rather than to itself.

**The red arm earned its keep on the first run.** The initial version bumped the threshold by
`+1` and silently MATCHED: raising the threshold admits pool `thr_p + 1` only if that pool TIES
at the threshold score, and otherwise the set is unchanged. Lowering by one always drops the
threshold pool itself, because `key(thr_p) == thr` by construction. A red arm that can no-op is
not a red arm, and for one run every exactness cell below it was vacuous.

## 5. Receipts (RTX 5090, N=5 interleaved, 2026-09-03)

`research/b200-dsa-select-20260903/gate-5090-20260903.txt`. **PASS**: red arm DIFFERS, anchor
IDENTICAL, **40/40 exactness cells EXACT**.

Timing, direction only under the rig law (the 5090 throttles and this door is sm_100a-gated, so
it cannot even engage there - the gate reaches the kernels through raw FFI):

| n_pools | context | shipped t_q=1 | parallel | ratio | t_q=4 ratio |
|---|---|---|---|---|---|
| 4096 | 16k | 20.5 us | 102.3 | 0.20x | 0.19x |
| 8192 | 32k | 26.0 us | 101.9 | 0.26x | 0.23x |
| 32768 | 128k | 83.0 us | 92.8 | 0.89x | 0.45x |
| **65536** | **256k** | **149.8 us** | **99.9** | **1.50x** | **1.20x** |
| 262144 | 1M | **557.0 us** | **175.8** | **3.17x** | **1.90x** |

Six launches against one kernel is a real fixed cost, so this is a DEPTH door: it loses below
128k and pays from 256k up. `MLA_DSA_SELECT_MIN_POOLS` is therefore **65536**, measured. The
constant was initially 4096, copied from the scorer's `MLA_DSA_SCORE_MIN_POOLS`; that would have
shipped a measured **4.6x regression at 32k**, and the gate's own regression bar is what caught
it.

## 6. Predicted saving, for the box cell to beat

Per token = per layer x 11 MLA/DSA layers, at t_q=1, applying the 5090 ratios:

| context | shipped | with the door | **saving** |
|---|---|---|---|
| 256k | 1.65 ms/token | 1.10 ms/token | **0.55 ms/token** |
| 1M | 6.13 ms/token | 1.93 ms/token | **4.19 ms/token** |

Those absolute numbers are 5090 microseconds and will not transfer; the RATIOS are the
prediction. Scaling by the previous lane's measured 5090-to-B200 factor on the sibling scorer
(3093.7 -> 852.0 us at 1M, ~3.6x), a B200 shipped selector at 1M should sit near ~155 us/layer =
~1.7 ms/token, and the parallel arm should do RELATIVELY BETTER there than on an 82-SM part
because the whole point is filling SMs. So the box cell has two numbers to beat: **>= 1.5x at
256k and >= 3.2x at 1M**, both at t_q=1.

## 7. Open

1. The interleaved plain-route B200 A/B at 256k and 1M (x3 fresh boots, vendor sampling, effort
   low), queued with the coordinator behind the running chain. Door stays default OFF until it
   lands.
2. `dsa-select-gate <dev> 5` on the pair, to confirm or move `MLA_DSA_SELECT_MIN_POOLS` - the
   crossover is a launch-overhead-vs-sweep trade and both terms differ on that silicon.
