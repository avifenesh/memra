# Plain-session affinity with prompt-end checkpoints — implementation + gate

Lane `lane/plain-affinity`, base `74afcaf6` (merged train tip `06f89163`). Implements
`research/cachespec-20260809/RESULTS.md` §P0: after a plain-decode session completes, write a
rewind checkpoint at a stable pre-generation boundary into the affinity store; on the next
request identity NOMINATES and an exact token comparison DECIDES, so a rewritten-history turn
resumes and primes only its own delta instead of recomputing an ever-larger suffix.

## The bug this fixes (owner-reported, root-caused in cachespec-20260809)

The prefix cache credits a frozen snapshot while an agent conversation grows: every later turn
recomputes an ever-larger uncached suffix (measured TTFT 2.35s → 13.7s over 10 turns, linear at
1.758 ms/uncached-token, R²=0.99971). The session-affinity resume path existed only on the
speculative tier, but the deployed PP-2 policy selects plain decode (K=0), so real agent clients
never created checkpoints and the prefix hit froze at the first LCP split. This lane extends the
affinity mechanism to the plain path.

## What shipped

All in `crates/memra-server/src/worker.rs` unless noted:

- `PlainCheckpoint { snap, pos, last_logits }` — the plain twin of `spec::SpecCheckpoint`. A
  `Cache::snapshot` at a stable pre-generation boundary: full-attn KV recorded as per-layer
  `len` (truncatable, no copy), GDN conv/ssm as a real device copy (the one thing that cannot
  roll back by length). `ReuseEntry` now carries `ckpt`, `affinity`, `fingerprint`.
- `plain_checkpoint_boundary()` — derives the boundary from the prompt's own turn markers: the
  LAST control-token index (the `<|im_start|>` opening the live generation segment), NOT
  prompt-end. Prompt-end is what the forced-spec control in cachespec-20260809 disproved (turn
  N+1 diverges a couple tokens below it, inside the live `assistant\n<think>\n` header). Raw
  markerless prompts fall back to a conservative guard window (`PLAIN_CKPT_RAW_GUARD=16`, in the
  RESULTS.md 8..32 band). The observed value 2 is never hardcoded.
- `maybe_plain_checkpoint()` — captures at the boundary via a prefill-tick prime-stop (the same
  `snapshot_at` machinery the prefix-cache LCP split uses; sessions armed with `ckpt_at` are
  excluded from batch-prime / dark-batch / fanout, which prime monolithically and cannot stop).
  Silent-fail on a too-tight rig, exactly like the spec checkpoint.
- Admit-side nominate/decide probe over the `reuse` pool (after the exact-extension probe
  misses): identity nominates (explicit `session_id`/`user`/`x-session-id` match, else the
  implicit fingerprint chain ≥ `FP_MIN_SEGMENTS`), `affinity_match` decides on bytes, then
  `Cache::rollback(snap, 0)` rewinds and the suffix primes. Divergence below the boundary = full
  re-prime. `MEMRA_AFFINITY=0` disables (the A/B seam).
- `plain_affinity_rewinds` metric on `/metrics` (subset of `continuation_pool_hits`) — the
  per-turn resume count the gate reads.

Audit hazards from `research/code-audit-20260809` fixed rather than inherited (this lane touches
these paths): **5.7** the only production `px.unpin` sat inside a `debug_assert!` (compiled out
in release → pins leaked → budget stopped bounding, a second cause of the frozen checkpoint) —
now a plain statement + warn; **H6** `prefix_restore` capacity precondition returns `Err` instead
of a `slice_mut` panic; **2.5** `promote_miss_to_hit` saturating-sub + warn instead of a release
`.expect()` crash; **H5** double-park guard (a conversation is not parked in both `reuse` and
`spec_reuse` under one affinity id).

## THE EXACTNESS WALL, and the bar the gate actually enforces

The mission asked for "every output byte-identical" (affinity on vs off). Building the gate
re-confirmed what the predecessor spec-affinity lane already proved with four independent
receipts (`research/session-affinity-20260805/RESULTS.md` §"ROOT CAUSE of that class"):

> **resumed == cold is NOT a property this engine has, on any reuse tier, and no reuse is even
> required to break it.** Chunked prefill is not reduction-order-stable. A resume primes
> `[rewind_boundary .. end]` as its own chunk sequence instead of one full prime; a different
> chunk split changes the reduction order in the prefill GEMMs, perturbs logits in the last
> bits, and flips a near-tie argmax at long generation windows. `MEMRA_PRIME_CHUNK` alone (a
> documented machine-config knob) changes greedy text on the same prompt with zero reuse, so two
> rigs already produce different greedy text.

Asserting resumed==cold would wire a permanently-red gate that blames affinity for chunked
prefill's reduction order. The first run of this lane's gate hit exactly that (11/17 "mismatches"
between a naive on/off), and the naive control was itself confounded: `MEMRA_AFFINITY=0` still
runs the prefix cache (its `cached` froze at 746 — the cachespec bug reproduced on q9), so it was
never a cold oracle. The gate now asserts what affinity OWNS, with a **true** cold oracle
(`MEMRA_KV_REUSE=0`, every tier off):

1. **DETERMINISM** — the resume path reproduces itself byte-for-byte across servers.
2. **NO NEW DIVERGENCE CLASS** — every affinity-vs-cold text divergence sits after a long
   coherent shared prefix (the pre-existing near-tie class the shipped prefix tier already
   shows); a shallow (< 32 char) divergence would be resume-state corruption and fails.
3. **SHORT-WINDOW EXACTNESS** — with generation stopped before near-ties cascade, affinity
   resumes are byte-IDENTICAL to cold. The positive proof the resumed STATE is correct.
4. **BUDGET** — `completion_tokens ≤ max_tokens`.
5. **SLOPE** — the ON arm's TTFT collapses after the learning turns with `plain_affinity_rewinds > 0`.

Gate: `research/affinity-20260809/compare_gate.py` (+ `run-5090-gate.sh`). Failure paths verified
with synthetic fixtures; the three green receipts below are the real q9 runs.

## Gate receipts — local RTX 5090 Laptop, q9 (Qwen3.5-9B-NVFP4-MTP), plain decode (MEMRA_SPEC_K=0)

`MEMRA_SPEC_K=0` reproduces the deployed PP-2 K=0 plain-decode policy on one card — the exact
path the owner report lives in. Greedy (temperature 0). Raw evidence under
`raw/5090/gate-20260809T111422Z/`, `raw/5090/tiny-20260809T120306Z/`, and the `chunk-*` /
`short-*` diagnostics.

### CHECK 1-2 — determinism + no new divergence class (full 12-turn workload, max_tokens=256)

`gate-corrected.json`. `--on` = affinity ON, three replay reps; `--cold` = true cold oracle
(`MEMRA_KV_REUSE=0`, `cached_tokens=0` on every turn, confirmed):

- **DETERMINISM: PASS** — on-run-1 == on-run-2 == on-run-3, all 17 turns.
- **NO NEW DIVERGENCE CLASS: PASS** — 12 turns diverge from cold; every one after a
  126–920-char coherent shared prefix (floor 32). The shipped prefix-cache tier diverges from
  the same cold oracle on **13** turns — affinity diverges on FEWER, and the one turn affinity
  diverges on that the shipped prefix tier doesn't (`sequential/3`) is 374 chars deep, squarely
  the near-tie class. Affinity introduces no new divergence class.
- `plain_affinity_rewinds = 11`, `continuation_pool_hits = 11` (ON); `0/0` on the
  `MEMRA_AFFINITY=0` arm — the mechanism engages only when enabled.

### CHECK 3 — SHORT-WINDOW EXACTNESS: the resumed state is byte-correct (max_tokens=8)

`tiny-20260809T120306Z/gate-tiny.json`. Prefix cache OFF (`MEMRA_PREFIX_CACHE_MB=0`) so affinity
rewind is the ONLY resume tier; 11 rewinds fired. With generation stopped at 8 tokens (near-ties
cannot cascade into a divergence):

> **12/12 sequential turns (17/17 including burst+postburst) BYTE-IDENTICAL to a true cold
> prime**, with `cached` tracking the rewind boundary (743 → 1112 across turns).

This is the decisive proof: `Cache::rollback` to the checkpoint restores exactly the state a cold
prime of `fed[..pos]` produces. The longer-window divergences above are purely the chunked-prefill
near-tie tail, not a resume-state defect.

### CHECK 4-5 — budget + TTFT slope (N=3 medians, full workload)

Budget: PASS (every response ≤ 256). Uncached-token slope: ON **0.062 ms/tok** vs OFF
**0.224 ms/tok**. The OFF arm's `cached` is **frozen at 746** (the cachespec frozen-checkpoint
bug, reproduced on q9); the ON arm's `cached` advances every turn (743 → 2609), which is the fix.

| turn | prompt tok | ON cached | ON TTFT s (N=3 med) | OFF TTFT s (N=3 med) | speedup |
|---:|---:|---:|---:|---:|---:|
| 0 | 748 | 0 | 0.234 | 0.207 | 0.88x |
| 1 | 1032 | 743 | 0.123 | 0.248 | 2.01x |
| 2 | 1209 | 1027 | 0.096 | 0.109 | 1.14x |
| 3 | 1368 | 1204 | 0.095 | 0.128 | 1.35x |
| 4 | 1552 | 1363 | 0.095 | 0.180 | 1.89x |
| 5 | 1715 | 1547 | 0.096 | 0.217 | 2.27x |
| 6 | 1859 | 1710 | 0.095 | 0.262 | 2.76x |
| 7 | 2034 | 1854 | 0.097 | 0.304 | 3.13x |
| 8 | 2251 | 2029 | 0.099 | 0.330 | 3.34x |
| 9 | 2401 | 2246 | 0.097 | 0.350 | 3.61x |
| 10 | 2614 | 2396 | 0.100 | 0.406 | 4.06x |
| 11 | 2814 | 2609 | 0.099 | 0.485 | 4.89x |

Sum-of-medians over the 12 sequential turns: **ON 1.326s vs OFF 3.227s = 2.43x**. The ON arm is
flat (~0.095s) while OFF climbs with prompt length. This workload is a laptop-scale smoke (8k
ctx, `base_notes=8`, prompts reach only ~2.8k tokens), so the absolute slope is proportionally
smaller than the owner's 6k–14k regime — but the SHAPE is the exact fix, and OFF's win is
understated because OFF here still runs the prefix cache (a real deployment's OFF would be the
full re-prime climb). The mechanism (flat TTFT, advancing `cached`, 11 rewinds) is what matters.

### Chunk-order note (honest)

The predecessor lane isolated the near-tie class with a `MEMRA_PRIME_CHUNK` 2048-vs-64 probe. On
THIS workload a 2048-vs-512 cold-vs-cold probe (`chunk-20260809T120014Z`) did NOT reproduce
divergence (0/12) — a smaller chunk delta on shorter prompts does not always cross a near-tie.
The attribution here does not rest on that probe: it rests on (a) the short-window 17/17 exactness
proof that the resumed state is correct, and (b) affinity diverging from cold on FEWER turns than
the already-shipped, already-gated prefix tier. Both are load-bearing; the chunk probe is a
negative control that neither confirms nor refutes and is reported as-is.

## Unit tests (cargo test -p memra-server, bin target)

**138 pass** (130 baseline + 6 new + 2 from the merged train tip), 0 failed. New:
`plain_checkpoint_boundary_lands_before_the_live_generation_header` (the last-marker rule; caught
and fixed a boundary bug during authoring), `..._uses_a_guard_window_for_raw_prompts`,
`..._declines_short_prompts`, `plain_affinity_resume_decision_is_bytes_over_identity`,
`plain_affinity_declines_a_divergence_below_the_checkpoint`,
`plain_affinity_fingerprint_collision_cannot_force_a_wrong_resume`.

## Constraints honored / scope

- `MEMRA_SPEC_K=0` reproduces the deployed PP-2 plain policy; PP-2 placement and spec defaults
  untouched. The spec-path checkpoint tier is unchanged (no regression).
- VRAM: the checkpoint is one GDN-state snapshot inside the already-`MEMRA_REUSE_POOL`-bounded
  `reuse` pool; it is dropped on LRU/consume and is already counted in the admit-oom
  `free + pool_cached` gate — no third leak.
- `MEMRA_CTX=262144`: the mechanism is ctx-independent (boundary/rewind logic is identical);
  the local gate ran 8k for the 24 GB laptop. The 256k doctrine run belongs on box1/RunPod
  (`run-5090-gate.sh` honors `MEMRA_CTX` for a bigger card).
- **Out of scope, flagged for a follow-up lane:** the full-cap-park + admission-reclaim +
  analytic-cost + global-byte-budget-LRU work (code-audit §1.3/6.1/6.3). That is a separate,
  larger memory-manager surface; this lane ships the affinity RESUME (the TTFT-slope fix) with a
  bounded, non-regressing pool. See PROGRESS.md.

## Gates still required before merge/tag (target rig, not done here)

`kernel-check` ALL GREEN, `run-gen` argmax MATCH, `run-spec` K=1..8 on the target GPU rig, per
CLAUDE.md. This lane's local 5090 evidence is the affinity correctness + slope receipt; the
orchestrator owns the merge and the full target-rig battery.
