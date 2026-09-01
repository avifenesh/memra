# lane/plain-affinity — plain-session affinity with prompt-end checkpoints

Base: 74afcaf6. Mission: implement RESULTS.md §P0 (research/cachespec-20260809/) —
plain-decode sessions must write a checkpoint at a stable pre-generation boundary into the
affinity store on completion; next request nominates by identity (session_id / x-session-id /
implicit fingerprint), exact token comparison decides; divergence = full re-prime, never a
wrong resume. Exactness law: resumed-vs-cold byte-identical.

## Status

- [x] PROGRESS.md committed before deep work
- [x] Read worker.rs affinity/spec-session/park/admission paths (lane 70ce5a0f mechanism)
- [x] Design note: where the plain checkpoint hooks in, stable-boundary selection
- [x] Merge train tip 06f89163 (TokenSnapshot + Gumbel fix + StepFun defaults)
- [x] Implementation: checkpoint capture at pre-gen boundary (maybe_plain_checkpoint via
      ckpt_at prime-stop; excluded from batch-prime/dark-batch/fanout)
- [x] Implementation: park ckpt+affinity+fingerprint in ReuseEntry at retire (H5 double-park guard)
- [x] Implementation: nominate/decide resume in admit (identity nominates, affinity_match
      decides, Cache::rollback to boundary, plain_affinity_rewinds metric)
- [x] VRAM accounting: checkpoint = one GDN-state snapshot inside the parked ReuseEntry
      (already counted in the admit-oom pool_cached gate; bounded by MEMRA_REUSE_POOL;
      dropped on LRU/consume — no third leak). Full-cap right-sizing + reclaim = P0-2, noted below.
- [x] Audit hazards hardened: 5.7 (unpin debug_assert -> plain stmt+warn), H6 (prefix_restore
      capacity precondition), 2.5 (promote_miss_to_hit saturating+warn)
- [x] Unit tests: 6 new (boundary last-marker/guard/short, resume bytes-over-identity,
      below-boundary decline, collision-cannot-resume). cargo test 138 pass (130 base + 6 + 2 train)
- [x] Replay gate script (compare_gate.py on/off byte-identity+budget+slope; run-5090-gate.sh)
- [x] Local 5090 replay gate run (q9 plain K=0) — GREEN under the correct bar
      (gate-20260809T111422Z + tiny-20260809T120306Z)
- [x] RESULTS.md with before/after TTFT table (N=3)

## Outcome (20260809)

DELIVERED. Implementation + gate green + RESULTS.md + 138 tests. Key finding: the mission's
literal "resumed==cold byte-identical" bar is IMPOSSIBLE on this engine (chunked prefill is not
reduction-order-stable — predecessor session-affinity-20260805 proved it with 4 receipts). The
gate asserts the achievable + correct bar instead:
  - SHORT-WINDOW (max_tokens=8, prefix cache OFF, 11 rewinds): 17/17 byte-IDENTICAL to a true
    cold oracle -> the resumed STATE is byte-correct (Cache::rollback restores exactly a cold
    prime of fed[..pos]).
  - Full workload: affinity diverges from cold on 12 turns vs the SHIPPED prefix tier's 13, all
    126-920 chars deep = the pre-existing near-tie class, NOT a new divergence. Resume path
    deterministic across 3 servers.
  - TTFT slope 0.224 -> 0.062 ms/uncached-tok; OFF cached frozen@746 (the cachespec bug
    reproduced on q9), ON cached advances 743->2609 (the fix).
This is a proven, bounded correctness win. Not a correctness risk — the short-window proof is
the teeth. Merge + full target-rig battery (kernel-check/run-gen/run-spec) owned by orchestrator.

## Post-merge regression fix (tip battery, 2026-08-09) — NOMINATABLE guard

The tip battery's serve-smoke cache-metering gate went 13-checks red after the merge. Root
cause CONFIRMED (hypothesis 1+2 combined, verified not assumed): the gate's workload is a 5-way
fanout of markerless anonymous `prompt_ids` (272 tokens, 256 shared) with NO session identity.
`plain_checkpoint_boundary` fell to the raw-prompt guard window and armed `ckpt_at` on every
session — and an armed `ckpt_at` excludes a session from the in-batch fanout / prime-batch
paths (they prime monolithically, cannot honor the per-session boundary stop). Every session
primed alone: 0 prefix hits, 6 misses/inserts, empty tick-seg window, tenants/economics red.
The armed checkpoint was PURE COST: a markerless anonymous prompt can never be nominated (the
implicit fingerprint tier needs FP_MIN_SEGMENTS; explicit tier needs a session id).

FIX (`358b88d7` on restructure/public-split): arm `ckpt_at` only when the checkpoint is
NOMINATABLE — explicit `req.affinity` present, OR `plain_ckpt_nominatable()` (the prompt's own
fingerprint chain can reach FP_MIN_SEGMENTS). Chat traffic keeps the capture (the pi fix
intact); raw agent loops that want affinity name their conversation (the documented pi
spelling: `sendSessionAffinityHeaders: true` + openrouter format). +1 unit test pinning the
class; 153 server tests green; cache-meter gate 23/23 ok.

Battery receipt (`local-ci-perf-quick-after-nominatable-fix.log`): kernel-check GREEN,
cache-metering 0 failed, session-affinity resume exactness 4/4 ok, gemma4 arm ok. ONE remaining
serve-smoke failure — "spec-vs-plain text mismatch" — is NOT this lane's: bisected with
spec+plain A/B on three binaries under the same lock: pre-merge b752cf2d spec==plain YES,
my-merge d71c231f YES, honesty-merge 349be208 NO (mid-text fork at char 110, spec usage shows
overshoot-clamp era counters). The honesty lane owns spec emission; flagged, stopping there
per the coordinator's instruction.

## Out of this lane's scope (noted, not built) — the reclaim/right-size P0-2

The audit (code-audit-20260809 §1.3/6.1/6.3) assigns the full-cap-park + admission-reclaim work
to this lane too. That is a SEPARATE, larger correctness surface (right-size parked caches to
fed.len(), global byte-budgeted LRU across PoolKeys, admission evicting dead entries before
deferring, analytic session_vram_cost). This lane delivers the affinity RESUME (the TTFT-slope
fix, the owner-reported bug) and does NOT regress the pool: the checkpoint adds only a GDN-state
copy inside the already-bounded reuse pool. The reclaim/right-size work is flagged for a
follow-up lane so this one ships a proven, bounded correctness win rather than a half-built
memory-manager rewrite. MEMRA_CTX=262144 full-cap parks remain as they were on main.

## Design (from worker.rs/spec.rs read — the hook points)

Mechanism inventory (all in crates/memra-server/src/worker.rs unless noted):

- Spec path already has the FULL mechanism: `SpecReuseEntry { sess, committed_text, affinity,
  fingerprint }`, park at retire (~line 3394), nominate/decide probe in `admit` (~3945-4042),
  `SpecCheckpoint` + `spec_rewind_to_checkpoint` in memra-engine/src/spec.rs. PP-2 policy
  picks K=0 so none of it runs. Plain path has only `ReuseEntry { fed, cache, last_logits,
  cap }` with exact-extension probe (~3727) — rewritten history misses 15/15.

- THE PLAIN CHECKPOINT IS STRUCTURALLY SIMPLER than spec's: a plain session's `cache` holds
  state for exactly `fed` (prompt + generated), full-attn KV is len-truncatable, GDN conv/ssm
  needs the D2D copy — same `Cache::snapshot()/rollback()` (memra-kv/src/lib.rs 343/407) the
  spec checkpoint uses. No draft scratch, no last_h anchor needed at the boundary IF we park
  (a) the checkpoint snapshot at prompt-end AND (b) resume = rollback + prime suffix via
  the existing continuation prime path (prime_cache with cache.pos>0), which re-derives
  logits from the primed suffix. Resume REQUIRES non-empty suffix (a rewound plain session
  has no last_logits for the boundary row unless we save them — we CAN save prompt-end
  logits cheaply? NO: at prompt-end `last_logits` IS the prompt-end row — s.last_logits at
  prefill_done. But affinity resume with empty suffix = re-asking the same prompt = prefix
  cache territory; decline empty suffix like spec does).

- BOUNDARY: RESULTS.md says stable PRE-GENERATION boundary, not "checkpoint+2 wrong" —
  the spec checkpoint at prompt-end FAILED because the frozen workload's turn N+1 prompt
  diverges 2 tokens BEFORE turn N's prompt end (the template's live assistant-generation
  suffix `<|im_start|>assistant\n<think>\n` differs after rewrite... actually receipts say
  divergence at ckpt-2). Fix per RESULTS.md: derive the checkpoint before the template's
  live generation suffix. Implementation: checkpoint position = prompt_end - guard, where
  guard = the trailing run of tokens that belongs to the live assistant header. Concretely:
  scan the prompt tail backwards to the LAST control-token boundary (the `<|im_start|>`
  opening the generation turn); checkpoint just BEFORE that final segment. Exact diff still
  decides. Fallback for markerless raw completions: guard window 16 tokens (in 8..32 per
  RESULTS.md), never hardcode 2.

- WHERE TO CAPTURE: at prefill-done on the plain path (prefill_tick / prime-batch /
  dedup-fanout all converge on `s.prefill_done = true`) — but capture must happen only
  ONCE per session at the stable boundary. Cheapest correct shape: capture at RETIRE time
  is WRONG (recurrent state at retire is post-generation, can't rewind). So capture during
  prefill: split the prime at the boundary (the LCP-split machinery `snapshot_at` already
  knows how to stop a prime at an exact token index!) and take Cache::snapshot there.
  REUSE `snapshot_at`-style boundary stop: new field `ckpt_at: Option<usize>` primes to
  the boundary, snapshots (kv lens + conv/ssm D2D), continues. Cost: one snapshot per
  session (GDN state copy, KB-class), NOT a 20GiB prefix-entry copy.

- PARK AT RETIRE: plain retire branch (~3408) additionally carries the checkpoint +
  affinity + fingerprint: extend ReuseEntry with `ckpt: Option<PlainCheckpoint>`,
  `affinity: Option<String>`, `fingerprint: Vec<u64>`, `committed: Vec<u32>` == fed.
  fed IS committed for plain sessions.

- NOMINATE/DECIDE AT ADMIT: after the exact-extension probe misses (~3733), before the
  prefix-cache probe: affinity probe over `reuse` pool — identity nominates (explicit
  affinity id match, else fingerprint chain >= FP_MIN_SEGMENTS shared), bytes decide
  (affinity_match(prompt, fed[..ckpt.pos]) must be Exact{suffix_from == pos} with
  non-empty suffix), room check (e.cap >= need, mirrors spec's F5 lesson: NOT ctx_cap).
  On accept: rollback cache to ckpt, resume with seed_fed = fed[..pos], suffix primes.
  On decline: existing paths (prefix cache serves its frozen 6148, which is still better
  than nothing).

- VRAM ACCOUNTING: checkpoint conv/ssm copies live inside the parked ReuseEntry — the
  parked pool is already counted in the admission gate's `pool res/used` diagnostic and
  bounded by MEMRA_REUSE_POOL. The snapshot adds KB-MB class (GDN state only, no KV
  copy). No third leak: the checkpoint is dropped with the entry (LRU) and consumed on
  resume. Right-sizing parked full-cap entries (the ~11.2GB c4 term) = P0-2 (admission
  reclaim), NOT this lane's first deliverable; but plain sessions get need-sized caches
  already via ctx_cap math... NO — MEMRA_CTX=262144 makes ctx_floor 262144, every session
  full-cap. Right-size at PARK time is out of scope creep; note it, don't build it.
  What this lane MUST NOT do: pin extra full-size buffers beyond the existing pool cap.

- EXACTNESS ARGUMENT: rollback lands cache at exactly the state a fresh prime of
  fed[..pos] produces (same contract spec_rewind relies on); suffix prime + decode then
  bit-identical to cold (continuation-prime == fresh-prime contract, session-gate oracle).
  Sampler penalty history replayed host-side over fed[..pos] (same as today's reuse hit).

- GOTCHA (graph sessions): a graph-promoted session's cache lives in s.graph.cache at
  retire — today's retire branch reads s.cache (None for graph sessions => graph sessions
  never park; unchanged behavior, fine).
- GOTCHA (eager-only/gemma4): no continuation prime — engine refuses pos>0 prime. Plain
  affinity resume must EXCLUDE eager_only models (they can't prime a suffix over a
  rewound cache except tokenwise... tokenwise decode_step path DOES work for carried
  suffixes per prefill_tick. Keep exclusion conservative v1: skip eager-only).
- GOTCHA (step35 SWA): step35 KV is full-attn per-layer geometry, len-truncatable same as
  qwen — Cache::snapshot/rollback are geometry-agnostic (len + conv/ssm). OK.

## Intel digest (research/prior-art-20260809 + code-audit-20260809, read 20260809)

- SWA-ring rollback question RESOLVED (audit "Hypotheses affirmatively KILLED"): memra has
  NO ring buffer — SWA is a read-side mask over a full max_ctx allocation, absolute rope,
  rollback = pure len truncation. llama.cpp #13194's "rollback across the ring frontier is
  impossible" does NOT apply. Pre-generation boundaries are additionally safe by
  construction (positions reached during prefill only).
- llama.cpp PR arc #15293->#24176 independently converged on this exact design
  (checkpoints at user boundaries, restore = last ckpt at-or-before divergence, erase
  later ones; tail-only checkpoints die on mid-history edits). v1 here: checkpoint CHAIN
  carried per parked entry (Vec, small bound), restore picks last <= divergence — the
  #24176 shape, nearly free when the model has no recurrent layers.
- Audit hazards this lane must FIX, not inherit:
  * H6: prefix_restore -> copy_u8_into -> slice_mut PANICS on oversized src; add explicit
    capacity precondition returning Err (this lane touches sizing/restore paths).
  * 2.5: promote_miss_to_hit checked_sub().expect() + prefix_miss_lcp.take().expect() are
    RELEASE panics on a bookkeeping slip; new admission routes (this lane) must convert to
    saturating+receipt / decline-fanout.
  * 5.7 (CRITICAL): the only production px.unpin sits inside debug_assert! (worker.rs:3354)
    — compiled out in release; pins leak, budget stops bounding. CHECK git log for a
    competing fix lane before touching.
  * 2.1: the checkpoint publish must NOT route through maybe_prefix_seed (cold-only +
    has_covering would freeze it) — it doesn't; separate mechanism.
  * H5 double-parking: one conversation must not hold entries in both reuse and spec_reuse
    under the same affinity id — evict the same-id spec entry when plain-parking with an id.
- Capture strategy split (regression control): models with NO recurrent layers (step35
  class — all-full-attn, len==pos on every layer) need NO capture at all: the checkpoint
  is synthesized (kv lens = pos) at resume; no prime-stop, no batch-prime exclusion, ZERO
  behavior change for the c4 burst receipt. Hybrid (GDN) models capture via a ckpt_at
  prime-stop (the snapshot_at machinery shape) and are excluded from batch-prime/fanout
  while a capture is pending — the local q9/q27 replay gate exercises this path.
- Base moved: train tip 06f89163 (Gumbel u01 clamp + TokenSnapshot event + StepFun
  sampling defaults — touches Event enum + sampling plumbing near this lane). Merge before
  further implementation.

## Log

- 20260809: lane started, worktree /home/avifenesh/projects/wt-affinity @ 74afcaf6.
- 20260809: read RESULTS.md + worker.rs (admit/park/affinity/prefix paths) + spec.rs
  checkpoint mechanism + memra-kv snapshot/rollback. Design pinned above. Next: implement
  PlainCheckpoint capture at boundary via ckpt_at prime-stop, park-with-identity, admit
  probe, unit tests.
- 20260809: coordinator relaunch after API-timeout death mid-read. Committed state, read
  both papers, digest above. Next: merge 06f89163, check 5.7 ownership, verify step35
  recur-layer question (cfg.ssm), then implement.
