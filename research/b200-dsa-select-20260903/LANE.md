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

## 4b. The race review caught, and why no gate could have

`memra_sel_last_arrival` originally read `__threadfence(); if (tid == 0) atomicAdd(&done, 1);`
with **no block barrier first**. That is wrong, and it was wrong in exactly the pass that matters:

`__threadfence()` orders only the CALLING THREAD's prior global writes. The histogram pass's data
is written by all 256 threads (`atomicAdd(&hist[..])`) and by eight warp leaders (the finite
count), with only warp-scoped `__shfl_down_sync` between those writes and the arrival call. So a
CTA's thread 0 could publish that CTA's arrival while its own warp 7 had not yet issued its
histogram atomics. Whichever CTA then observed the full count -- on another SM, synchronised only
through the counter -- would run the descent on a histogram missing those bins and an
undercounted `n_fin`, producing a wrong `ctrl[HI]` and a wrong rank clamp. A **silent,
non-deterministic break of the exact byte-identity this whole door exists to provide.**

The sibling `tie` and `count` passes were correct by accident of shape: they reduce through
shared memory, `__syncthreads()`, and let thread 0 alone write `cta[blockIdx.x]`, so thread 0's
own fence covered the only write that mattered. The histogram pass had no such funnel.

Fixed by putting the `__syncthreads()` inside the helper, ahead of the fence, so every call site
gets the CUDA Programming Guide's `threadFenceReduction` idiom in full rather than three call
sites each getting it right or wrong on their own.

**Nothing in the tree could have caught this.** `compute-sanitizer racecheck` is shared-memory
only; `synccheck` looks for barrier divergence; and this gate runs on an idle device where the
window is vanishingly small. The original 40/40 EXACT receipt did not speak to it, and the
re-run after the fix does not either -- **the ordering argument is the evidence, and the gate
only shows the fix costs nothing.** The exactness cells now repeat 20x each (800 comparisons) as
a net rather than a proof, and the reasoning is written on the helper so it is not re-derived.

Found in review on PR #115, not by a gate. Worth remembering when the next lane reaches for a
last-CTA epilogue: the idiom is `__syncthreads()` THEN `__threadfence()` THEN the counter, and
the leading barrier is the half that is easy to drop.

## 5. Receipts (RTX 5090, N=5 interleaved, 2026-09-03)

`research/b200-dsa-select-20260903/gate-5090-20260903.txt`. **PASS**: red arm DIFFERS, anchor
IDENTICAL, **40/40 exactness cells EXACT at 20 repeats each (800 comparisons)**.

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

## 6. Predicted saving, for the box cell to beat (PREDICTIONS, not measurements)

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

## 7. Composition audit (2026-09-03, static)

Written out because a sibling lane's default flip was withdrawn (#114) for a compose-time defect
that no tape-identity gate catches, and because this lane's own `memra_sel_last_arrival` bug was
the same class: correct in isolation, wrong in combination, invisible to every gate in the tree.

**Buffers this path touches.** `score` (read-only; allocated and filled immediately before the
call, hybrid_forward.rs:8658), `idx` (written here, allocated at :8673, consumed later by
`mla_attn_gathered`), `ws` (allocated per call inside the wrapper, touched only by the six
kernels, never escaping).

| conjunct | verdict |
|---|---|
| `MEMRA_B200_DSA_DECODE=1` / `=2` | The ONLY door that meets these buffers: `=1` may replace the producer of `score`, `=2` the consumer of `idx`. Same stream, strict producer -> select -> consumer order, no shared scratch. Composes. |
| `MEMRA_B200_MATVEC_ARM`, `MEMRA_GLM5_W8`, `MEMRA_B200_GEMV_V2` | Change the `mm()` producing `q_index`/`head_weights`, i.e. the VALUES in `score`. Input data, not a shared buffer; the exactness claim is conditional on the score plane handed in, not on how it was computed. Composes. |
| `MEMRA_HC_FUSED_PRE`, `MEMRA_KDA_FUSED_PROJ`, `MEMRA_GLM5_Q8_FUSE` | Do not appear anywhere in the kpool call chain (hybrid_forward.rs:8439-8690). Disjoint. |
| `MEMRA_B200_PRIME_V2` arms 1+2 | The mHC PRIME schedule (hybrid_forward.rs:678-1405), a prefill mechanism; this door is `t_q <= 8` only. Disjoint by width AND by code path. |
| `MEMRA_GLM5_DECODE_GRAPH` | `ws` goes through `uninit_i32` -> `alloc_uninit` -> stream-ordered `alloc` + `keep_if_capturing`, the identical contract `score`, `idx` and the sibling lane's `part_*` buffers already use here. Composes for the same reason they do. |
| streams | The engine has two: `stream()` and `copy_stream` (MoeSlotCache weight prefetch only). All six kernels launch on `stream()`; none of these buffers is touched by `copy_stream`. |
| slice arithmetic (#114's failure mode) | Kernels index only through `memra_mla_kpool_select_ctas`/`_ws_ints`, the same entry points the host sizes from; verified numerically over n_pools 0/1/255/4096..1M including the 1024-CTA clamp, emit worst case (pool slot 2047, tail slot 2050) under `width` 2051. **No Rust-side slicing of a device buffer anywhere in the path, so there is no `slice_mut` to panic.** |

**What the audit could NOT check, stated rather than assumed:** runtime composition on sm_100a.
Nothing has exercised `mla_dsa_select_on() == true` end to end, because no sm_100a hardware is
available (below). The kernels are measured; the door's composition is ARGUED.

## 7b. First sm_100a serving signal (2026-09-03, ONE pair, provisional)

B200 pair, best posture on main `3908a431`, vendor sampling, fresh boot per arm, 1M plain route:

| rung | off | on |
|---|---|---|
| 1M decode | 34.70 | **41.32 tok/s (+19.1%)** |
| 1M TTFT | 486.8 s | 487.0 s (flat) |
| short code decode | 50.81 | 50.62 (flat -- cannot engage) |

```
[mla-b200-dsa-select] engaged kpool_select t=1 pools=256756 ctas=1003 class=exact
  (sm_100a; MEMRA_B200_DSA_SELECT=1)
```

TTFT flat is the right shape for a decode-only door, and the short rung is flat because it sits
below the floor. The rig-derived prediction (~4.19 ms/token over 11 layers, which on the off
arm's 28.7 ms/token lands near 41 tok/s) transferred almost exactly.

**This is one pair, not a median.** The interleaved x3 run is still in flight; its medians are
what this doc should eventually quote, and until then the section 6 per-token figures stay
PREDICTIONS.

**The `RemoteDisconnected` in that run was harness fratricide, not this door.** A follow-up knee
cell waited on a `grep -q "ALL DONE"` marker that an earlier aborted pass had already written to
the same log, so its wait loop fell through immediately and it `pkill`ed the live server
mid-request while booting its own beside it. The engine behaved correctly throughout
(`SIGTERM: draining (1 in flight, deadline 30s)`), the client simply saw the socket close. No
panic, no defect in this path. Recorded here so it is not later misread as a stability question
about the door.

## 7c. The floor is 262_144 tokens, and a "256k" rung is under it

`MLA_DSA_SELECT_MIN_POOLS` is 65536 pools, which at `pool = 4` is **262_144 tokens exactly**. A
serving rung called "256k" is usually ~256_756 tokens = **64_189 pools, 2.06% under the floor**,
so it engages nothing. That cost a real B200 rung on 2026-09-03, sized from an earlier note of
mine that read "n_pools >= 65536 (256k context at pool=4)" -- true only if "256k" means the
binary 262_144. Every context figure in this lane is now an exact token count.

Full rig band (RTX 5090, N=5 interleaved, `gate-5090-band-20260903.txt`), t_q=1 / t_q=4:

| n_pools | tokens | t_q=1 | t_q=4 |
|---|---|---|---|
| 4096 | 16_384 | 0.20x | 0.19x |
| 8192 | 32_768 | 0.25x | 0.25x |
| 32768 | 131_072 | 0.90x | 0.51x |
| 49152 | 196_608 | **1.13x** | 0.64x |
| 64189 | 256_756 | **1.44x** | 1.02x |
| 65536 | 262_144 | **1.53x** | **1.07x** |
| 262144 | 1_048_576 | **3.13x** | **1.95x** |

**`t_q=4` is what holds the floor up.** Plain decode crosses into profit around 49152 pools and
is a clear 1.44x by 64189; the DFlash2 spec-verify width is still 0.64x at 49152 and only reaches
parity near 64189. The door engages uniformly across `t_q`, so the floor is set by the worse
width, and 65536 is the first cell that wins at both.

So the door refuses a 256k rung the rig measures at **1.44x on the plain route**. Closing that
needs the floor KEYED ON `t_q` -- a lower floor for plain decode, this one kept for spec-verify --
the way the sibling `MLA_DSA_ATTN_ARM` table is keyed. Deliberately NOT done here: a default-ON PR
for this door is in flight, and widening which shapes engage underneath it is the wrong order of
operations. It wants its own PR and a B200 gate run.

## 8. Open, and why it is blocked

**No box receipt exists for this door, and none can be scheduled.** As of 2026-09-03 the vast.ai
account is out of credit (`billing_creditonly`, so no auto-topup charges a card) and every
instance was stopped - the B200 pair, the glm5 prod box, q38 and ornith. Only the owner can
restore it.

1. The interleaved plain-route sm_100a A/B at 256k and 1M (x3 fresh boots, vendor sampling,
   effort low) is **OWED**. The per-token figures in section 6 are PREDICTIONS derived from 5090
   ratios - the numbers that cell is meant to test, not numbers it produced. Door stays default
   OFF.
2. `dsa-select-gate <dev> 5` on the pair, to confirm or move `MLA_DSA_SELECT_MIN_POOLS` - the
   crossover is a launch-overhead-vs-sweep trade and both terms differ on that silicon.
3. Rig work remains available (the 5090 is up), so any further exactness or composition work that
   does not need sm_100a can proceed.
