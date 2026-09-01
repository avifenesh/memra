# qwen4_exp PROFILE-11 — the 262,144 window: the host indexer top-k moves device-side and
# the ~131k cliff DIES

Target, per the owner's scope change: **262,144 tokens — the model's native window — and
best performance THERE.** 1M is dropped; YaRN stays a banked capability and is not a target
(LADDER.md "SCOPE CHANGE 2026-08-31").

Box: the round-2/round-3 lane boxes, **2x RTX PRO 6000 Blackwell Server Edition, 97,887 MiB,
600 W** each — the same card class as every prior receipt in this lane, which is what makes
the round-2 numbers the comparison. Provider, region, instance class and instance ids are
fleet state and live in darklanes, not here. Artifact `q48fn-yarn1m` (hardlink twin of
`q48fn-nvfp4`, `rope_type=yarn factor=3.814697265625 original=262144 mpe=1000000`), corpus
`corpus_commit=84a9d5b6a`, chunk 2048, one card.

Receipts: `../round2-box-receipts/kvq2/` and `../round2-box-receipts/logs/`. Every receipt
header carries its cache arm (`# cache kv_quant=... idxq=... golden_pin=... seams_env=...`),
which is the instrument fix PROFILE-10 §4 forced; quote it with any number below.

## THE VERDICT (the sentence to read alone)

**The ~131k cliff is DEAD, and it was the host indexer top-k.** One seam — `idxsel`,
`qsa_index_topk_u32`, the QSA indexer's top-512 block selection moved from the host to the
GPU — takes the 131,072 rung from **17.41 to 30.69 tok/s (1.76x)** and the 150,000 rung from
**17.15 to 30.11 (1.76x)**, collapses the deep prefill chunk's dominant section by **48x**,
and removes the discontinuity itself: 100,000 -> 131,072 was 1.9x slower for 1.31x depth and
is now a gentle **-7.6%**. Decode across the target window goes from "priced by one boundary"
to monotone and shallow. Prefill per chunk becomes **flat in depth**.

## 1. What moved, and why the obvious reuse would have been slower

`set_idx_dev` (the yarn lane) already computed the block SCORES on the GPU. It then **dtoh'd
the whole score slab** — up to 128 MB per sub-batch — so the HOST could run
`top_blocks_ascending` (top-512 per row under the pinned tie rule) over it. PROFILE-10 §4c
attributed the entire 262k performance problem to that host half: at a 131,072 fill
`qsa.idx_host` was **51,235 ms, 83.0% of a prefill chunk**, while every GPU section stayed
flat within 4%, and it jumped **16x for a 9% depth increase** between 120,000 and 131,072.
The sub-batch-boundary hypothesis for the same cliff had already been **tested and killed**
by its own knob (`MEMRA_Q4E_IDX_SCORE_CAP_MF=128` removed the 2->3 sub-batch transition
entirely and bought 4.4%), so the cost was in the host top-k work itself.

`qsa_index_topk_u32` runs the selection on device and reads back `rows x budget` u32 (4 MB at
the target geometry) instead of `rows x blocks` f32. The host never touches a score.

**Exactness is by construction, not by tolerance.** The key is
`(~f32_total_asc_u32(score) << 32) | block_index`, and ascending u64 order IS the host
`sel_cmp` — score descending under `total_cmp`, then block index ascending. Keys are
distinct because the low 32 bits are the unique block index, so the k-th smallest key is a
single well-defined element and `key <= threshold` selects exactly k blocks; the emitted
order is ascending block index, which is what the host twin's `candidates.sort_unstable()`
provides.

### Corpus-worthy: the prior art to reuse was the KEY, not the LOOP

The natural move was to copy the devtwin lane's `qwen4exp_route_topk_f32` — k rounds of
block-wide strict min over `(~bits(w) << 32) | idx`, the pattern PROFILE-9 finding 1 proved
took a route launch from 54 to 9 us. **Copied here it would have been SLOWER than the host it
replaces.** That loop is O(k*n) and the two geometries differ by three orders:

| | k | n | comparisons per row |
|---|---|---|---|
| route (devtwin) | 10 | 512 experts | 5,120 |
| indexer (this lane) | 512 | up to 65,536 blocks | **33.5 M** |

At 2,048 rows x 12 QSA layers per prefill chunk that is ~823 G comparisons per chunk. So the
kernel is an **8-pass radix select** (one byte per pass, 256 smem bins) that fixes the k-th
smallest key exactly, followed by ONE ordered warp-ballot compaction pass. What transferred
from the route lane is the KEY ENCODING — the insight that a u64 total-order key makes a
parallel selection bit-exact under any evaluation order, ties included.

Two smaller carries in the same direction:

- **The tie class is structural, not an edge case.** The scores are a relu-sum, so a deep row
  really does carry long runs of exact `+0.0`, and the whole selection among them is decided
  by the index tiebreak. A tie-blind kernel is green on random data and silently attends a
  different KV set on real data. The oracle's `all-zero` case exists for exactly that.
- **The route kernel's key made an implicit DOMAIN assumption; this one does not.** See §1a.

### 1a. `f32_total_asc_u32` — a reusable ordering primitive, and why the implicit version is a trap

Both kernels turn "compare floats under the host's `total_cmp`, break ties by index" into
"compare u64 keys as unsigned", because a total order enumerated by a parallel reduction is
bit-exact under ANY evaluation order — which is what makes a device selection provably equal to
a host one rather than approximately equal. They differ in the float half of the key:

| | float → ascending-u32 map | valid on |
|---|---|---|
| `qwen4exp_route_topk_f32` (devtwin) | `~bits(w)` | **only `w >= +0.0`** (post-exp, post-positive-division softmax weights) |
| `qsa_index_topk_u32` (this lane) | `f32_total_asc_u32(v)` | **the whole f32 domain** |

```
// Rust f32::total_cmp VERBATIM, then shifted into the unsigned domain:
//   let mut l = bits as i32;  l ^= (((l >> 31) as u32) >> 1) as i32;  compare as i32
unsigned f32_total_asc_u32(float v) {
    unsigned u = __float_as_uint(v);
    unsigned m = ((unsigned)((int)u >> 31)) >> 1;   // 0, or 0x7fffffff when the sign bit is set
    return (u ^ m) ^ 0x80000000u;                   // i32 total order -> ascending u32
}
```

Strictly monotone across both zeros (it puts `-0.0` below `+0.0`, as `total_cmp` does), both
infinities, subnormals, and every NaN payload. Two properties make it worth reusing verbatim
rather than re-deriving:

1. **It removes a claim from the kernel's contract.** `~bits(w)` is correct *given* a domain
   invariant that lives somewhere else in the program — in the route kernel's case, in the fact
   that softmax weights come out of `exp` and a positive division. That invariant is true today
   and nothing enforces it. Had it been copied into the indexer, where scores pass through
   `fmaxf(dot, 0.0f)` and could in principle carry an inf or a NaN from upstream, the failure
   would be a *silently different attended KV set* on rare rows: no assert, no NaN, fluent
   output. The full map costs two extra integer ops per key and deletes the claim entirely.
2. **It is the same three lines everywhere**, so the oracle can gate the map itself. This
   lane's arm 0i carries a `total-cmp-domain` case (signed zeros, subnormals, negatives, NaN)
   precisely so the encoding is proven on the whole domain rather than on the domain the caller
   happens to supply.

**Reuse rule:** any device top-k that must reproduce a host `total_cmp` ordering uses
`f32_total_asc_u32`, and a `~bits(x)` shortcut needs the domain invariant written into the
kernel doc *and* an oracle case that would fail if it were violated. The devtwin key has the
former and not the latter; it is correct and it is the one to migrate first if that kernel is
ever retuned.

## 2. The attribution, measured — same instrument, one seam moved

`--ladder 100000,131072,150000 --profile 1` with `MEMRA_Q4E_SEAMS=idxsel`
(`kvq2/ladder-r2prof-step-idxsel.tsv`) against the banked OFF arm
(`kvq2/ladder-r2prof-step.tsv`). Sections at the profiled chunk at fill **~131,072**:

| section | OFF | ON | change |
|---|---|---|---|
| **`qsa.idx_host`** | **51,235.1 ms (83.0%)** | **1,066.5 ms (9.6%)** | **48.0x less** |
| `moe.sel_grouped` | 2,563.4 (4.2%) | 2,512.1 (22.6%) | -2.0% |
| `qsa.sdpa` | 2,153.7 (3.5%) | 2,004.4 (18.0%) | -6.9% |
| `moe.router` | 1,478.4 (2.4%) | 1,327.8 (11.9%) | -10.2% |
| `hyper.read` | 1,357.3 (2.2%) | 1,322.2 (11.9%) | -2.6% |
| `gdn.proj` | 990.4 (1.6%) | 984.6 (8.8%) | -0.6% |
| `gdn.conv_scan` | 863.5 (1.4%) | 860.4 (7.7%) | -0.4% |
| `qsa.proj` | 261.3 (0.4%) | 261.3 (2.3%) | 0 |
| chunk total | **~61,700 ms** | **~11,100 ms** | **5.6x** |

Every GPU section is where it was — three move by single digits, plausibly less host
contention, and `qsa.proj` is identical to the tenth of a millisecond. That is the same
reading the diagnosis made in the other direction: the cliff was ONE host section and
nothing else. Absolute ms under `--profile` are sync-bounded and inflated; shares and ratios
are the signal.

**The profile at fill ~131,072 with the seam ON is a flat, broad, GPU-bound profile with no
dominant section** (22.6 / 18.0 / 11.9 / 11.9 / 9.6 / 8.8 / 7.7%), and its absolutes are
essentially the fill-~0 profile plus the one-time bounded-attention step in `qsa.sdpa`
(933.7 -> 2,004.4 ms, which then holds). That is the healthy shape, and it is what makes the
next claim possible.

## 3. Prefill per chunk is now FLAT in depth

Per-16,384-token segment wall from the two profiled runs' own `# ladder-progress` lines:

| fill reached | OFF s/16k | ON s/16k |
|---|---|---|
| 16,384 | 84.1 | 82.9 |
| 32,768 | 89.2 | 86.1 |
| 49,152 | 91.3 | 85.9 |
| 65,536 | 94.2 | 86.7 |
| 81,920 | 98.1 | 86.1 |
| 98,304 | 101.5 | 87.0 |
| **116,421 (8 chunks)** | 105.7 | **88.2** |
| **131,072 (7 chunks)** | 97.3 | **79.4** |
| **147,493 (8 chunks)** | **297.2** | **90.1** |

PROFILE-10 §3b's headline was "prefill cost per chunk STEPS UP ~4.8x at ~131k depth". On the
ON arm there is **no step and no curve**: ~11 s per 2,048-token chunk from the first chunk to
the last, across a 9x depth increase. The +25% sub-131k growth the OFF arm showed below the
cliff was ALSO the host top-k (it is O(fill) per chunk by construction), which the earlier
receipt correctly called "gently, near-linearly" and correctly did not call a bug.

Cumulative prefill to 150,000: **1,348.5 -> 795.1 s**.

## 4. The rungs — and the discontinuity is gone from DECODE too

Ladder timing rows (unprofiled, `rounds=3x12`, medians named, escalation automatic):

| depth | OFF tok/s | ON tok/s | speedup | OFF ms/tok | ON ms/tok | OFF spread | ON spread |
|---|---|---|---|---|---|---|---|
| 100,000 | 33.18 | 33.22 | 1.001 | 30.1 | 30.1 | 0.30% (x3) | 0.14% (x3) |
| **131,072** | **17.41** | **30.69** | **1.76x** | 57.4 | 32.6 | 2.74% (**x5**) | 0.36% (x3) |
| **150,000** | **17.15** | **30.11** | **1.76x** | 58.3 | 33.2 | 1.62% (**x5**) | 0.02% (x3) |

Four findings from this table.

1. **The decode discontinuity is GONE.** 100,000 -> 131,072 was 1.9x slower for 1.31x depth;
   it is now **-7.6%**. The 16x step was a host top-k step in decode as well as in prefill.
2. **The variance was the host top-k too.** Both deep rungs auto-escalated to x5 on the OFF
   arm; on the ON arm they sit at 0.36% and 0.02% and no escalation is owed. A host thread
   pool contending over a growing score array is exactly the shape that produces
   round-to-round spread, and removing it removed the spread. Worth keeping as a reading
   rule: **a rung that escalates is telling you something about the mechanism, not just about
   the box.**
3. **At 100,000 this seam is a PREFILL lever and a decode WASH** — 33.18 vs 33.22, inside the
   spread, with prefill -8.0%. Mechanism from the decode profile at that fill:
   `qsa.idx_host` is 2.5 ms / 7.3% with the seam ON against round 1's 2.2 ms host figure,
   because a single decode row is ONE CUDA block over 25,000 blocks and the device path is
   launch-and-sync-latency-bound rather than work-bound. A prefill chunk carries 2,048 rows,
   which is where the 48x lives. **Do not quote this seam as a shallow decode win.**
4. **The greedy chain at a 100,000-token fill is BYTE-IDENTICAL across the seam.** Both arms'
   banked `continuation_ids` open
   `21609,8,7813,198,92,271,6182,6292,3840,313,198,262,5003,491,3052,25,594,485,8,1411,9815,313,198,285,9815`
   — all 25 ids the OFF arm recorded, in order. An at-depth exactness receipt that fell out
   of a perf cell for free.

## 5. The NEW residual — the profile that orders the systems levers

Owner direction 2026-08-31: the 262k window is a systems problem, not one hot function, with a
named lever list (graphs, prefetch, expert speculation, cache-friendly layout, context
switches / sticky threads, "and so on"). The rule is profile first, so here is the deep decode
profile with `idxsel` armed, beside the OFF arm at the same fill (`--profile 1` at rung
150,000; sync-bounded absolutes, shares are the signal):

| section | OFF | ON | share ON |
|---|---|---|---|
| `qsa.sdpa` | 10.6 ms | 10.3 ms | **27.3%** |
| **`ple.host_ngram_gather`** | 7.9 ms | **7.3 ms** | **19.5%** (HOST) |
| `hyper.read` (96 calls) | 3.3 | 3.2 | 8.5% |
| **`qsa.idx_host`** | **27.0 ms (42.6%)** | **3.2 ms** | **8.4%** (HOST) |
| `moe.sel_grouped` (48) | 2.8 | 2.6 | 7.0% |
| `gdn.proj` (36) | 2.6 | 2.6 | 6.8% |
| `gdn.norm_gate_out` (36) | 1.4 | 1.3 | 3.6% |
| `moe.shared` (48) | 1.2 | 1.2 | 3.1% |
| `moe.router` (48) | 1.1 | 1.1 | 2.9% |
| `hyper.write` (96) | 1.0 | 0.9 | 2.4% |
| `qsa.proj` (12) | 0.9 | 0.9 | 2.4% |
| `lm_head` | — | 0.8 | 2.3% |

`qsa.idx_host` in DECODE goes **27.0 ms (42.6%) -> 3.2 ms (8.4%)**, which is the same
attribution as the prefill table at section level and confirms the decode cliff had the same
cause. What is left is a broad profile whose top two entries are one GPU kernel and one HOST
section, and **27.9% of a deep decode token is still host work** (`ple.host_ngram_gather` +
`qsa.idx_host`).

### Lever verdicts, in the order the profile puts them

1. **PREFETCH (`ple.host_ngram_gather`, 19.5%) — the profile MOVED the target, and that is the
   finding.** The lever assumed the cost was the synchronous gather from the 102 GB
   host-resident table, to be overlapped with compute (the vendor's cookbook design point).
   The gather is `t x 16` random rows — 16 reads of 160 f32 at decode, microseconds. The 7.3 ms
   is `host_ngram_ids`, a `ngram_ids` twin over the **FULL token history** whose last `t` rows
   the caller then slices: a decode step at a 150,000-token fill rebuilds 150,000 rows of
   hashes (plus `max_ngram` shifted copies of the whole history) to consume ONE. **Async
   prefetching the table would have bought ~nothing.** Landed as the `plecache` seam (default
   OFF, `gate_ple_ngram_cache` EXACT over 69,635 cumulative-sequence comparisons incl. eos
   segment resets and repeated rewinds to diverging prefixes), and handed to the agent who owns
   the PLE host path for its A/B and default. See §5a — this is the **third** time on this family
   that "the host half is O(context) per token" was the real mechanism.
2. **`qsa.sdpa` (27.3%) — the largest single section and now the top GPU item.** It is already
   bounded (the block-list kernel reads only the <= 2,052 selected KV rows at any depth, flat
   from 4k to 262k), so this is a *bandwidth/layout* question, not a complexity one: 10.3 ms
   for 12 layers = 0.86 ms/layer to gather ~2,052 quantized KV rows. That is owner lever 4
   (cache-friendly layout), and it is NOT this agent's territory — named here with its number
   so the owner of the KV layout starts from a measurement.
3. **GRAPHS AT DEPTH (owner lever 1) — measuring, not arguing.** Round 4 measured decode
   graphs a wash and PROFILE-9 §3a found the setting stopped mattering once the devtwin stack
   removed the host boundaries (13.57 ON vs 13.60 OFF) — both SHALLOW. At 131k the GPU work per
   token is 32.6 ms against 13.6 ms shallow, so the launch-issue share is *smaller* and the
   prediction is "even more of a wash". A prediction is not a receipt: x3 interleaved pairs at
   131,072 under the measurement lock, ON vs `graph=0`, are running.
4. **EXPERT SPECULATION (owner lever 3) — structurally NOT APPLICABLE in this configuration,
   with the reason.** The lever hides the latency of *staging/uploading* a predicted expert
   set. On this deployment the whole model is device-resident (89,971 MiB post-load on one
   card, NVFP4 trunk expert banks included) — there is no upload to hide. The profile agrees
   that there is little to win even if there were: `moe.router` is 2.9% and `moe.sel_grouped`
   7.0% of a deep decode token, and the router already runs on device with no readback
   (`routerdev`). Where it WOULD pay is a configuration with a non-resident or peer-resident
   expert bank — i.e. TP2, whose peer half-bank makes 99.93% of layer-tokens cross cards — and
   **TP2 cannot reach this window at all** (it OOMs during the fill below 100k while one card
   reaches ~731k). Banked as a receipted non-lever for the single-card 262k route, and as a
   live lever for the co-activation placement lane if TP2 depth is ever solved.
### 5a. FAMILY PATTERN: on qwen4_exp, suspect "the host half is O(context) per token" FIRST

Three independent perf investigations on this model have now ended at the same shape, each time
after a different mechanism was named first and each time in a different section:

| lane | section | what the host half was doing | measured before | after |
|---|---|---|---|---|
| yarn (round 1) | `qsa.idx_host` | scoring every micro-block per query row on the host | 29.3 ms at a 32k fill, **52% of the token**, quadratic across a prefill | 24.1 ms total token (device scorer) |
| this lane (§2) | `qsa.idx_host` | top-512 per row over the dtoh'd score slab | **51,235 ms, 83.0% of a deep chunk**, 16x step at ~131k | 1,066 ms, 9.6% |
| PLE (§5.1, handed off) | `ple.host_ngram_gather` | rebuilding n-gram ids over the FULL history to use the last `t` rows | **7.3 ms, 19.5% of a deep decode token** | predicted ~0 (cached; A/B owed) |

Why it keeps happening, stated as mechanism rather than coincidence: this model's execution
doctrine deliberately keeps CONTROL decisions on the host as exact twins of the reference code
(MoE routing top-k, QSA micro-block selection, PLE n-gram hashing, PLE gate scalars — module
doc). Every one of those twins was written against the reference, where the natural formulation
is "compute the quantity over the whole sequence, then take the part you need". That is free at
reference scale and it is O(context) per token at product scale. **The host twin is not slow
because host code is slow; it is slow because the reference formulation it faithfully copies is
whole-sequence.**

Three practical carries:

- **First guess for any depth regression on this family: find the host section and ask whether
  it recomputes over the whole history.** In all three cases the fix was either "append instead
  of rebuild" or "move the reduction to the device", and in none of them was the named-first
  mechanism (allocator thrash, sub-batch boundaries, dtoh count, async prefetch) the answer.
- **The shape hides from shallow profiling.** All three were invisible or minor below ~8-32k of
  fill and dominant past it, which is why the cliff sat "just past the deepest depth any
  previous receipt in this lane reached". A residual list built at short context will not rank
  these correctly.
- **Two host twins remain by design and should NOT be attacked the same way:** the PLE host
  n-gram GATHER itself (the 102 GB table is host-resident by design, and the gather is 16 rows)
  and the TP2 route (it consumes host expert ids by construction). Knowing which host work is
  structural is what keeps this heuristic from becoming a licence to move everything.

5. **CONTEXT SWITCHES / STICKY THREADS (owner lever 5) — the variance it targets has already
   collapsed, so it is deprioritized on evidence.** The metric the lever asks for is per-token
   variance, and the ladder measures it: on the OFF arm both deep rungs auto-escalated to x5
   (2.74% and 1.62% spread) and on the ON arm they sit at **0.36% and 0.02%** with no
   escalation owed. That is the signature being described — a 48-thread host pool contending
   over a growing score array — and `idxsel` removed the pool from the deep path
   (`top_blocks_ascending` was the only caller passing `available_parallelism()` on the device
   scoring path). What host threading remains is single-threaded appends. Re-open only if a
   later arm shows spread returning; the number to beat is 0.02%.

## 6. The exactness receipts for `idxsel`

Selection is an EXACTNESS contract, not a tolerance one: a differing selection changes which KV
rows attention reads, and the output stays fluent. Four instruments, each covering something the
others do not, and the gaps are named.

| instrument | scope | result |
|---|---|---|
| tiny arm 0i `gate_qsa_index_topk` | budget 512 up to **65,536 blocks** (the 262k window's fill/4), ragged sub-batch slabs, tie classes | ids **and ascending order EXACT**, 13 rows / 7 cases, **rig AND box** |
| live audit at depth (`--ladder 131072`) | real checkpoint, ship-default cache | **rows=1,549,452 mismatched=0 deepest_blocks=32,793** (a 131,172-token history) |
| live audit, decode-row volume (`--ladder 8192 --ladder-decode 6000`) | 5x2000 steps x 12 QSA layers | **120,000 decode-row selections**, 0 mismatches (193,692 rows total) |
| greedy chain at a 100,000-token fill | the perf cell's own `continuation_ids` | **BYTE-IDENTICAL** across the seam, all 25 ids the OFF arm banked |
| four rule gates with `idxsel` armed | verify-bit 24, spec byte-identity 256 raw, tp2-gate 24, tp2-prefill-gate 8 | all green, **identical to the previous battery to the digit** |

Two things stated so the battery is not read as coverage it lacks. The four rule gates seed from
the 10-token goldens probe, which never crosses the 2,048-token selection horizon, so `idxsel` is
a structural no-op in all of them — they prove no regression on the shipped path and nothing about
depth, which is precisely why the depth instruments above exist. And the two audits split the
contract deliberately: the 131,072 cell has the DEPTH (`deepest_blocks=32,793`) and the 8,192 cell
has the decode-row COUNT (`deepest_blocks=4,548`). Neither alone satisfies "zero mismatches over
>=100k real decode rows at depth"; together they do, and saying which is which is the point.

### The audit cell re-proved the attribution by accident

The audit deliberately restores the score-slab dtoh and the host top-k, and its own rung row came
back at **58.1 ms / 17.21 tok/s** — essentially the OFF arm's 57.4 ms / 17.41 at the same depth,
from a binary with the device selection armed and verified correct. Device selection ON plus host
top-k added back equals the old number. That is the cliff mechanism demonstrated a second time, in
the opposite direction, on a different cell — and it is also the strongest possible proof that the
audit is not a no-op.

## 7. Measurement protocol: two findings that cost three A/B attempts

This lane runs on a box shared by three agents working the owner's lever list in parallel. Both
findings are measured, and both are the kind that silently produce a defensible-looking number.

**The measurement lock covers timed rounds; the contention is in the PREFILLS.** At 20:52Z `ps`
showed THREE `qwen4exp_real_gate` processes computing simultaneously (100% / 92.7% / 27.9% CPU)
while an interleaved A/B pair held `flock` EXCLUSIVE on the shared lock. The lock did what it
says: the in-instrument form wraps the timed rounds only, so every lane's load and prefill runs
unlocked, and on a shared 48-vCPU box those starve the HOST half of somebody else's timed decode.
The arm most exposed to it is the one that matters here — the OFF arm is a 48-thread host top-k.
Fix, one word on each side and it closes the hole both ways: `flock -s` (shared) for untimed
correctness work, `flock -x` (exclusive) for anything quoted. Never nested — a shell holding `-s`
whose child requests `-x` on the same path blocks on its own ancestor forever.

**The harder constraint is capacity, and no lock protocol touches it.** qwen4_exp is **89,971 MiB
post-load**, so one model fills one card. With state at the measured 11.08 KiB/token: 131,072
allocates at 91,475 MiB, 262,144 at 92,883, and a FILLED 262,144 rung peaks at **95,805 of 97,887
with 2,082 MiB free**. A deep cell needs a WHOLE card with nothing else on it; even a shallow cell
is 90 GB of trunk. **Two cards hold at most two concurrent lanes, never three** — so for a
three-way parallel lever split a second box is not an optimisation, it is the precondition.

What the banked cells rest on, stated rather than assumed: the §2 attribution pair and the §4 rung
table were taken before the other lanes had built their binaries (theirs are timestamped 20:23 and
20:28; process listings at 18:47 and 19:05 show only this lane), and their own receipts corroborate
sole occupancy — monotone segment walls (82.9 / 86.1 / 85.9 / 86.7 / 86.1 / 87.0 s per 16k) and
decode spreads of 0.14% / 0.36% / 0.02%, where the contended arm later showed exactly the spread
that this class of interference produces. The re-run A/B under `-x` with a capacity guard is the
belt to that braces.

## 8. THE 262,144 TABLE — the deliverable

Ship defaults, one card, chunk 2048, `corpus_commit=84a9d5b6a`, artifact `q48fn-yarn1m`. Both
rows self-evidencing via `# cache kv_quant=q8_0/q5_1 idxq=q8 golden_pin=false`. The ON row was
taken under exclusive `flock -x` around the ENTIRE invocation with card 1 idle at 595 MiB.

| 262,144 tokens | OFF (round 2) | **ON (`idxsel`)** | change |
|---|---|---|---|
| **tok/s** | 15.21 | **23.44** | **1.54x** |
| ms/token mean / med / p90 | 65.7 / 65.0 / 66.6 | **42.7 / 40.4 / 40.5** | |
| **prefill wall** | 4,779.1 s (79.7 min) | **1,439.2 s (24.0 min)** | **3.32x** |
| chunks | 128 | 128 | |
| card-0 VRAM at rung | 95,805 MiB | 96,669 MiB | |
| round medians | [64.9, 64.2, 65.1, 65.9, 65.2] | **[40.5, 40.4, 40.3]** | |
| spread | 2.56% (**escalated x5**) | **0.30% (x3)** | |

`looped=false`, host RSS 167,796 MiB.

**The whole window, before and after** (OFF rows from LADDER.md §4c):

| depth | OFF tok/s | ON tok/s |
|---|---|---|
| 100,000 | 33.18 | 33.22 |
| 120,000 | 32.01 | (not re-run) |
| 131,072 | **17.41** | **30.69** |
| 150,000 | 17.15 | 30.11 |
| **262,144** | **15.21** | **23.44** |

The window used to be priced by one boundary — ~32 tok/s below it and 15-18 above. It is now a
single monotone curve from 33.2 at 100k to 23.4 at 262k: **-29% for 2.6x the context**, against
the old -54%.

### Prefill is flat through the old cliff, in one continuous fill

Per-16,384 segment wall from this run's own `# ladder-progress` lines: 82.8, 82.8, 85.8, 85.7,
88.3, 88.3, **89.4 (131,072 crossed here)**, 90.4, 93.2, 94.2, 95.5, 96.4 s. The OFF arm went
84 -> 105 and then **stepped to 475**. What remains is a gentle curve — **+16% across a 16x depth
increase** — which is the residual O(fill) term (pooled-key extension plus the device kernel's own
growth). Prefill is no longer the dominant cost at the target window: 24 minutes, not 80.

### The one wrinkle, named rather than averaged away

The `# ladder-jitter` line reads `n=34 mean=42.66 med=40.36 sd=12.959 cv=30.38% p99=117.09
max=117.09 outliers_1.5x=1` — a SINGLE 117.09 ms step among 34 samples, against a median of 40.36
and three round medians inside 0.5% of each other. One event, not a regime, but it drags the mean
by 2.3 ms and inflates cv to 30%. The table quotes **23.44 tok/s from the mean** because the
banked OFF row quotes its mean too (65.7 -> 15.21), so the 1.54x is apples-to-apples; the
median-based rate is **24.75 tok/s**. The outlier is UNATTRIBUTED and owed one cheap look — host
page-fault and allocator events are the candidates, and contention is ruled out (`-x` held, card 1
idle at 595 MiB).

## 9. The interleaved A/B, and the DEFAULT FLIP

Three two-process attempts at this A/B were voided by contention (§7). The version that settled
it uses `--ladder-ab-seam` (a sibling lane's instrument): both arms interleaved on ONE 131,072
prefill, exclusive `flock -x` around the entire invocation, sole tenant on card 0.

```
# ladder-ab  rung=131072 seam=idxsel arm=off median_ms=56.64 tok_per_s=17.66 reps=7
    medians=[56.64, 56.73, 55.88, 57.02, 57.26, 56.55, 56.12] spread=2.40% n=224
# ladder-ab  rung=131072 seam=idxsel arm=on  median_ms=32.25 tok_per_s=31.01 reps=7
    medians=[32.06, 32.08, 32.25, 32.04, 32.73, 32.69, 32.71] spread=2.12% n=224
# ladder-ab-verdict rung=131072 seam=idxsel off_ms=56.64 on_ms=32.25
    speedup=1.7562x delta_pct=43.06% reps_per_arm=7 (escalated) vol_cs=4316
# ladder-ab-restore seam=idxsel arm=on
```

**1.7562x**, 7 reps per arm (escalated past 5 by the instrument), 224 warm samples per arm,
within-arm spreads 2.40% and 2.12% — the verdict is ~18x the pooled spread. Three properties make
this the strongest available form:

1. **It reproduces both arms of the independent two-process cells** — on 31.01 vs 30.69, off 17.66
   vs 17.41, speedup 1.7562x vs 1.76x. Two instruments, two sessions, one answer.
2. **Same fill**, so the two arms share the literal same KV cache, indexer cache and pooled-key
   plane; no prefill-wall difference can leak into the decode comparison.
3. **It is its own POSITIVE CONTROL.** A same-state seam flip is not automatically sound — the
   `selv2` seam note records that captured decode graphs bake some kernel choices — so pointing
   the instrument at a seam with a KNOWN 1.76x effect is what licenses trusting it on the next
   cell, graphs at depth, whose expected result is a wash and where a wash is therefore
   ambiguous between "no effect" and "no engagement". `ladder-ab-restore arm=on` confirms the
   seam was returned to its entry state, so nothing measured after the A/B ran the wrong arm.

The same process's rung row reproduces cleanly: 131,072 at **30.14 tok/s** (32.1 ms median,
`rounds=3x12 medians=[32.1, 32.0, 32.1] spread=0.21%`, prefill 686.2 s, `looped=false`).

### The default is FLIPPED ON (2026-09-01, commit 8c4bdbe78)

`IDX_SEL_DEFAULT = true`, FLAGS row rewritten in the same commit (default, both arms, rollback
seam, receipts pointer). What it rides: the 1.7562x interleaved verdict above; the 1.54x target
window with prefill 3.32x; the cliff's removal; the oracle EXACT on both card classes; the
1,549,452-row at-depth audit and 120,000 decode-row audit at zero mismatches; the byte-identical
greedy chain at a 100,000-token fill; four rule gates identical to the prior battery.

**No pairing requirement**, stated because the devtwin pair set the opposite precedent
(`routerdev` alone measured 0.906x with decode graphs on, so those two flip together). This seam
wins alone on every measured surface, on top of the shipped `routerdev` + `idxcache` + `kvq`
stack. Rollback stays one word: `MEMRA_Q4E_SEAMS=idxsel=0`.

**Owed for the flip and not claimed by it:** the no-env engagement proof (PROFILE-9 §7's
discipline — run with no `MEMRA_Q4E_SEAMS` at all and assert `idxsel-audit rows > 0` at a fill
past the horizon, because a silently no-op default reports rows=0). It needs a binary built from
the flip commit, so it is a cell, not a sentence. Parked on card 1 deliberately: it carries no
timing, and consecutive card-0 cells never leave that card free long enough for a capacity guard
to catch.

### A recurring host stall, now with two data points

Both deep timed rows carry exactly one outlier, at the same ratio to their medians:

| rung | n | median | max | cv | outliers_1.5x |
|---|---|---|---|---|---|
| 262,144 | 34 | 40.36 ms | **117.09 ms (2.9x)** | 30.38% | 1 |
| 131,072 | 34 | 32.06 ms | **69.57 ms (2.2x)** | 19.10% | 1 |

Round medians in both cases sit inside 0.21-0.30%, so these are single events, not regimes — but
two of them at the same shape is a class, not a coincidence. Unattributed; contention is ruled out
(`-x` held, the other card idle or owned). Host page-fault and allocator events are the
candidates. Named as a follow-up rather than averaged into a mean.

<!-- SECTIONS 10+ (graphs at depth, the depth gates, spec-at-depth per shape, traces, and what a
     262k perf CLAIM would still need) land as their cells complete. -->
