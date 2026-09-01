# lane/kv256-capacity — 262,144-window session capacity on the 96GB pair

Base: f2b03e63. Owner directive: the 262,144 window is the product identity — capacity comes
from engine economics, never window reduction. lane/cx-ctxcharge owns per-request charge +
alloc-OOM reclaim; this lane must not duplicate those.

## Mission

More concurrent full-262k sessions on the box1 PRO 6000 pair. Rank/design/implement:

1. Lazy/growable KV (VMM extent growth) — deep fix, design + prototype minimum.
2. Step35 SWA geometry — do SWA(512) layers hold full-ctx slabs today? (verify, trim) — FIRST.
3. KV format on full-attn layers only — fast-fail gate run.
4. Spec-reserve right-sizing on the PP-2 plain-only path — FIRST.
5. Host offload for parked sessions — design/measure if reachable.

## State

- [x] Worktree entered, anatomy read: memra-kv Cache::new_inner, cache_bytes_per_token,
      step35_attn_pre_wo (prime view trim), admission gate (worker.rs ~2660-2820).
- [x] PRELIMINARY (code-read, receipts pending): Step35 SWA layers are
      LayerKind::FullAttention; new_inner allocates max_ctx-token slabs for all 45 trunk
      layers; the SWA prime read view is off = base_len-(win-1) (aligned down 32) — only
      the last ~512+chunk keys are ever read on SWA layers. If decode reads match, the
      slab below the window is dead capacity on 33 of 45 layers.
- [x] PRELIMINARY (code-read, receipts pending): on the PP-2 plain-only shape the
      admission reserve is `cost` (a full second 21,894 MB at 262k), not
      SPEC_SHRINK_RESERVE — admit_reserve_override applies only to the spec branch.
- [ ] Geometry receipt: exact full/SWA byte split from the artifact's config (arithmetic
      receipt, then box1 confirmation).
- [x] Verify decode/spec/rollback paths never read SWA KV below the window (else trim is
      not free) — then design the SWA ring/trim. DONE — full reader-audit table below.
- [ ] Option 4: implement policy-aware reserve, unit tests, local gates.
- [ ] Option 2: implement SWA window-sized allocation, local gates.
- [ ] Local 5090: kernel-check, run-gen argmax, run-spec K=1..8 (affected models).
- [ ] Box1: step35 exactness + PP-2 serve-shape gates.
- [ ] Box1: capacity before/after at MEMRA_CTX=262144 (honest sessions row).
- [ ] Option 1: VMM growable-KV design + measured prototype.
- [ ] Option 3: fp8-on-full-attn-layers fast-fail gate run on step35.
- [ ] RESULTS.md with ranked option table.

## SWA reader audit (Option 2 feasibility) — 2026-08-10

Question: does ANY path read SWA-layer KV below the attention window? If not, a window-sized
("ring") allocation is semantics-free for that path. Verdicts:
- **window-safe** — reads only rows `>= len - win` (or `>= aligned(len-(win-1))`) on SWA layers.
- **reads-below-window** — dereferences rows older than the window; a trimmed allocation breaks it.
- **needs-ring-semantics** — copies/stores the FULL `[0..len)` byte range; correct today because
  the slab holds all rows, so a ring must either keep this path on full-history bytes or change
  its contract.

All readers of the per-layer `KvLayer.k/.v` byte planes were enumerated via `view_u8_range` /
`view_u8` / `copy_u8_into` / `memcpy_dtod` / `dtoh_u8` call sites (exhaustive grep, this commit).
step35 = the 262k target (SWA on 33/45 trunk layers + the SWA-type MTP block 45); gemma4 rows are
included because the allocator (`Cache::new_inner`) is arch-shared and any ring flag lands there.

### A. Attention read paths (engine)

| # | Reader | file:line | SWA view arithmetic | Verdict |
|---|--------|-----------|--------------------|---------|
| A1 | step35 prime (chunked + monolithic) | `crates/memra-engine/src/hybrid_forward.rs:8684-8694` (`step35_attn_pre_wo`) | `off = (base_len-(win-1)) & !31`; view `[off, off+t_kv)` | window-safe (oldest touched row = `align32(base_len-win+1)`, i.e. <=31 rows below `len-win+1`, never older) |
| A2 | step35 eager decode T=1 | `crates/memra-engine/src/hybrid_forward.rs:8900-8904` (`step35_decode_attn`) | `off = len-win` once `len > win` | window-safe |
| A3 | step35 batched decode (B>1 serve) | `crates/memra-engine/src/decode_batch.rs:1415-1424` (`step35_decode_batch_layers`) | verbatim copy of A2's arithmetic | window-safe |
| A4 | step35 spec verify | `crates/memra-engine/src/spec.rs:2649-2694` (`step35_verify`) — per-row REPLAY of `step35_decode_attn` (spec.rs:2688) | inherits A2 | window-safe |
| A5 | step35 MTP draft block (SWA-type blk 45, scratch KV) | `crates/memra-engine/src/spec.rs:952-960` (`mtp_step35_attn`) | `off = len-window` on the SEPARATE `MtpScratch` KV (cap = max_ctx, spec.rs:3075-3080) | window-safe read; NOTE the scratch is a second full-ctx SWA-class allocation — in scope for the same trim |
| A6 | step35 dc/graph decode | `crates/memra-engine/src/decode.rs:1914-1920`, `decode.rs:2477-2486`, `spec.rs:1286-1298`, `worker.rs:3018-3028` | REFUSED by construction (`fa_decode_dc` cannot express the offset view) | window-safe (path does not exist for step35) |
| A7 | generic (non-SWA) decode/verify full-range views | `decode.rs:2043-2050,2777-2778,2934-2935`, `spec.rs:full_attn_verify` | `[0..t_kv)` full-history views | not SWA layers (qwen/M3/Hy3 full-attn; window trim never applies) — out of scope by definition |
| A8 | gemma4 eager decode / dc-eager / batched-dc | `hybrid_forward.rs:7270-7274` (`gemma4_decode_attn` fallback), `7680-7686` (`gemma4_decode_attn_dc` dc-eager fallback) | `off = len-win` | window-safe |
| A8b | gemma4 rows_w twins (decode/verify/dc) | `hybrid_forward.rs:7250-7259, 7660-7676` and `gemma4_verify_attn:8305-8312` | view is `[0..len)` FULL-PREFIX (`view_u8(&kvl.k, (base_len+t)*ktb)`) but `fa_decode_rows_w` masks per-row to the last `win` keys (absolute-index geometry) | window-safe in MASK terms, **reads-below-window in ADDRESS terms**: the view base is row 0, kernel indexes absolute positions. A ring for gemma4 must rebase these views or stay step35-only |
| A8c | gemma4 dc GRAPH capture arm | `hybrid_forward.rs:7696-7700` (full-buffer views, `b_swa` bucket) | full-buffer view + device counter | same address-term caveat as A8b; graph replay bakes row-0-based addresses |
| A9 | gemma4 verify per-token fallback | `hybrid_forward.rs:8316-8320` (`gemma4_verify_attn` loop) | `off = avail-win` per row | window-safe |
| A10 | gemma4 fresh prime | `hybrid_forward.rs:6490-6494` (`gemma4_attn_prime`: fresh-prompt only, attends f32 q/k/v, cache append is write-only) | no cache read at all | window-safe |
| A11 | KV L2 prefetch (opt-in `MEMRA_KV_PREFETCH=1`) | `decode.rs:2666-2670` | prefetches `[0..t_kv)` from row 0 | reads-below-window in address terms, but (a) step35 returns at decode.rs:2651 BEFORE this point, (b) prefetch is value-free. No correctness impact for step35; must be range-checked if ever enabled under a ring |
| A12 | debug kvsum readback (`MEMRA_BURST_VCHECK=1`) | `gemma_spec.rs:1320-1326` | `dtoh_u8` of the WHOLE plane, sums `[pos0, pos0+kr+1)` | diagnostic-only, gemma4-only; full-plane dtoh would need the ring size instead — trivial, not a blocker |

### B. Rollback / snapshot (state managers)

| # | Reader | file:line | Mechanism | Verdict |
|---|--------|-----------|-----------|---------|
| B1 | `Cache::snapshot`/`snapshot_into` | `crates/memra-kv/src/lib.rs:373-428` | records per-layer `len` ONLY (no KV byte copy) | window-safe (no byte reads) |
| B2 | `Cache::rollback` | `crates/memra-kv/src/lib.rs:437-463` | `len = saved + accept_len` truncation, `len_d` restamp; NO byte copy | window-safe — BUT see the ring-rewind hazard below |
| B3 | spec `commit_verified_prefix` (~line 2327) | `crates/memra-engine/src/spec.rs:2346-2352` | `kvl.len = saved + j`; verify's appended rows for the kept columns stay in place (bit-identical contract); no re-read of old rows | window-safe — the verify re-attends via A4, whose oldest reach at the ROLLED-BACK len is `saved+j-win+1 > 0`; never below the pre-round window |
| B4 | `spec_rollback_kv` (device twin) | `crates/memra-engine/src/lib.rs:1908-1921`, called `spec.rs:4853` | writes `len_d = saved + base + n_acc` on device; no KV byte access | window-safe |
| B5 | `spec_rewind_to_checkpoint` | `crates/memra-engine/src/spec.rs:3114-3135` | `rollback(accept_len=0)` + `scratch.set_len` (spec.rs:670-673) — pure len truncation both | window-safe as len-arithmetic; ring-rewind hazard below applies |

RING-REWIND HAZARD (the one real semantic constraint from this section): every rollback is a
len TRUNCATION over a slab that still HOLDS the old rows. Under a ring of R rows, rows are
overwritten mod R — a rewind from len=L to len=P keeps working (the rows in `[P-win+1, P)` are
still resident as long as `L - (P - win + 1) <= R`, i.e. the forward distance since the
checkpoint has not lapped the ring). Spec rounds move len forward by <= K+1 (~<=9) per round and
roll back within the round: never laps a ~5k-row ring. Plain-affinity rewind (B5-class, worker
e.cache.rollback at worker.rs:4429) rewinds a PARKED session from end-of-generation back to the
pre-generation boundary — forward distance = the whole generated turn (max_tokens, can be 4k-32k).
A ring must therefore either (a) size R >= win + max rewind distance it wants to support, or
(b) invalidate checkpoints older than `len - (R - win)` (decline the resume, full re-prime —
the existing decline path at worker.rs:4444-4446 already handles exactly this shape).

### C. Prefix cache / affinity / reuse (worker + prime graph)

| # | Reader | file:line | Mechanism | Verdict |
|---|--------|-----------|-----------|---------|
| C1 | `prefix_snapshot` | `crates/memra-server/src/worker.rs:1934-1983` (byte copy at 1948-1953) | copies the FULL `[0..len)` K/V byte range of every layer into a `PrefixEntry` | **needs-ring-semantics** |
| C2 | `prefix_restore` | `crates/memra-server/src/worker.rs:1988-2033` (byte copy at 2012-2013) | copies FULL `[0..src.len)` back into a fresh cache, `dst.len = src.len` | **needs-ring-semantics** (pairs with C1) |
| C3 | `prefix_insert_from_session` / `maybe_prefix_seed` | `worker.rs:2035-2046, 2085-2097` | wrappers over C1 | inherits C1 |
| C4 | prefix-dedup fanout restore | `worker.rs:5324-5328` | `prefix_restore` into siblings | inherits C2 |
| C5 | `maybe_plain_checkpoint` (affinity capture) | `worker.rs:2059-2083` | `Cache::snapshot` = len-only (B1) | window-safe (no bytes) |
| C6 | ReuseEntry park/resume (exact-extension) | `worker.rs:3995-4000` (park moves the WHOLE `Cache` — no byte copy), resume `worker.rs:4342-4356` | ownership transfer of the live slab; continuation prime appends at `len` | window-safe as-is (the parked object IS the ring if the ring is inside `KvLayer`) |
| C7 | PlainCheckpoint affinity rewind | `worker.rs:4429-4446` | `Cache::rollback` (B2) | len-safe; RING-REWIND HAZARD applies (see B) — bounded rewind or decline |
| C8 | prime-graph scratch copy-out | `crates/memra-engine/src/prime_graph.rs:136-143` | `memcpy_dtod` rows `[0, t)` scratch -> session | **needs-ring-semantics** for t > R; today gemma4-only (`prime_chunk_captured` path), t <= bucket (~prefill tick) — small, but the copy is position-0-based |

### C semantics naming (task (c)): the prefix cache stores POSITION-ADDRESSED FULL-HISTORY
bytes ("flat-history semantics"). A window-sized SWA allocation cannot serve C1/C2 for entries
whose `pos > win` unless the entry either (i) stores only the last `win`-aligned rows per SWA
layer + full rows on full-attn layers ("ring-aware entry" — restore rebases into the ring), or
(ii) is re-derived by re-priming. Option (i) is mechanical (the entry already stores per-layer
lens; add per-layer row ranges), but it is NEW cache-entry format work, not a flag.

### Audit verdict — Option 2 scoping

1. Every ATTENTION read on step35 SWA layers is window-safe (A1-A6). The 32-alignment in A1
   means the resident requirement is `win - 1 + chunk` rows plus <=31 alignment slack ->
   the 543-row + chunk figure in the geometry receipt stands as the ring floor.
2. gemma4's rows_w/dc-graph arms (A8b/A8c) address from row 0 even though they mask to the
   window — a ring that rebases the buffer breaks their ADDRESS math. Scope the flag to
   step35-class SWA layers (LayerGeometry.window on step35 arch) OR rebase those views.
3. All rollback paths are len-truncations (B) — safe under a ring iff the rewind distance since
   the oldest needed row hasn't lapped the ring. Spec rounds: trivially safe. Plain-affinity
   rewind: needs a lap-check (decline resume when lapped — existing decline path fits).
4. The prefix cache (C1/C2/C8) has flat-history semantics and is the ONE structural exclusion:
   session-local KV can ring-trim behind a flag; prefix-cache entries must stay full-history
   (they already deep-copy, so they'd simply keep allocating `len`-sized planes) or adopt a
   ring-aware entry format later. EXCLUSION RECEIPT: with `MEMRA_SWA_RING=1`, sessions whose
   KV may be prefix-snapshotted (prefix cache eligible / fanout leaders) either keep full
   allocation or the snapshot declines — the flag's first cut targets session-local KV only.
5. The MTP draft scratch (A5) is a second max_ctx-sized SWA-class allocation on step35
   (1 layer's rows, cap = max_ctx) — same trim applies and is simpler (no prefix cache ever
   touches the scratch; parked SpecReuseEntry moves it whole).

Option 2 is FEASIBLE scoped as: step35 SWA trunk layers + step35 MTP scratch, session-local
KV, flag `MEMRA_SWA_RING=1` default OFF, prefix-cache-eligible paths excluded (decline or full
alloc), affinity-rewind lap-check. gemma4 stays full-allocation until A8b/A8c views are rebased.

## Receipts land under research/kv256-20260809/raw/.
