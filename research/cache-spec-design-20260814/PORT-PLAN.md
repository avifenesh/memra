# Design A port plan — commit-gated cache publication for spec sessions

Scoped 2026-08-14 against 2e37cafbf. Anchors are file:line at that commit.
Survey + rationale: REPORT.md (same dir). Phase 1 = items 1–3; item 4 (K=0
draft-warm burst) = phase 2 behind MEMRA_SPEC_WARM0=1 default-off.

## Mechanism map (verified)

Cache writes (plain-tier only, crates/memra-server/src/worker.rs):
- SEED at prefill-done: maybe_prefix_seed :3693 → prefix_insert_from_session
  :3629 → prefix_snapshot :3220 (from prefill_tick :8276/:8357, prime-batch
  :5840). Key: PoolKey (model, cache_salt ns) :3854 + exact token prefix.
- LCP-SPLIT learning insert: armed at admission :7326-7330 (snapshot_at =
  miss-LCP), captured mid-prime :8348-8351 (boundary stop :8314-8325).
- In-batch FANOUT: dedup_interactive_prefixes :8072, insert_pinned :8229.
Reads: admit() :7154-7335 — whole-entry lookup :2785 + prefix_restore :3443;
partial best_lcp_entry :2801 + prefix_restore_at :3316 gated by
partial_prefix_decision :2521. LAWS: routed-MoE refuses mid-entry split
(:2530-2531); hybrid recurrent restores only at captured endpoint (:3371-3378).

Spec bypass (the 18.4% vs 99.5% cause — research/canonflip-20260813):
- Lookup: worker.rs:7165 `if prefix_on && reused.is_none() && !spec_eligible`
  skips probe AND miss-side arming. Policy comment :1334-1337.
- Publication: spec skips prefill_tick (prompt = turn-1 suffix via
  generate_spec_session, step_session :8867-8897); excluded from fanout
  (prefix_fanout_eligible :8054 requires s.spec.is_none()); retire parks only
  into spec_reuse (:6252-6285, capped 2/ns :648-652) — the residual 18.4%.
- Extra non-cache cost: spec bursts solo serially (phase (a) :5039).

Intra-session commit machinery ALREADY EXISTS (crates/memra-engine/src/spec.rs):
- SpecSession.committed :469; invariant cache.pos == committed.len()
  (:549-551; worker :5300-5302). pending_tok :498 flushed by
  spec_flush_pending :5284 before park/demote.
- Rollback by counters per-round: commit_verified_prefix :4166 (kvl.len =
  saved + j; device via spec_rollback_kv worker :7733); GDN recurrent rebuilt
  from VerifyCkpt :1188.
- Draft KV: MtpScratch :1080; set_len :1152 = only truncation; suffix fill
  :6469-6529 needs trunk hiddens; row-0 anchor = last_h, zeros fallback.
- Prompt-end checkpoint: SpecCheckpoint :615 / turn_ckpt :503; rewind :5146.
- Cold-prime split hook: prime_split :5873-5894 (boundary-stops turn-1 prime).

## Change list (phase 1)

1. Publication:
   - spec.rs prime section (:5843-5905): extend prime_split into boundary
     capture — after prime_cache(&prompt[..split]) snapshot recurrent conv/ssm
     + split logits → SpecBoundaryCapture { pos, recur_snap, logits } returned
     through generate_spec_session_*_prime_split (:5542/:5583). Full-attn KV
     rows [0..split) + draft rows [0..split) append-only → worker slices
     post-burst (true-hidden refresh :5838-5841 rewrites ≥ prompt end only).
   - worker.rs admit(): arm snapshot_at/seed_prefix for spec sessions
     (:7326-7333); thread boundary to step_session burst (merge with affinity
     prime_split :8984-8993 by min, law of :8299-8318).
   - worker.rs new prefix_insert_from_spec_session (twin of :3629): entry from
     capture + live-cache slices; GATE toks == sess.committed[..pos] &&
     pos <= committed_len; refuse under pp_host_bounce_active. PrefixEntry
     gains draft: Option<PrefixPlane> (MtpScratch rows [0..pos)); bump
     PREFIX_ENTRY_LAYOUT_VERSION :1355.
   - Seed case (boundary == prompt end): capture from turn_ckpt + prime
     logits, post-burst.
2. Counter enforcement at the seam (~20 lines): debug_assert cache.pos ==
   committed.len() at capture/publish; refuse publish while has_pending()
   unless flushed; never derive publish length from cache.pos.
3. Capped-restore probe:
   - admit() :7165 drop !spec_eligible. Spec-eligible restores always leave a
     recompute suffix: restore_len = min(lcp, prompt.len() - RECOMPUTE_TAIL)
     (PRIME_MIN_T-aligned page, floor 1; never round to coarser boundary —
     vLLM #38182/#9247 guard). Hybrid entries: restore_len == e.pos or skip.
     Whole-prompt entry → clear spec_eligible, take plain whole-entry hit.
   - spec.rs spec_session_from_restored(cache, committed, draft_plane):
     install cache (pp::new_cache pattern :7250-7257) + scratch copy +
     set_len(restore_len); committed = entry.toks[..restore_len];
     last_h = None; next_pred = None. Reuse spec_resumed machinery (:7581,
     :7710-7714, n_cached :7763). V1: entry without draft plane → declined
     for spec (plain path serves it).

## Gates

Stay green: serve-smoke; cache-meter-gate.py (spec-off; update its bypass
docstring); q35-cold-mixed-gate.py; run-spec K=1..8; accept-gate.sh;
decode-batch-gate --mode ppspec; local-ci tier 2; worker unit suite incl.
assert_prefix_cache_accounting :11338. NOTE
prefix_budget_geometry_reproduces_all_six_measured_q27_q35_entries :9601 pins
entry bytes — draft plane changes geometry; RE-DERIVE, don't delete.

New: (1) headline soldsweep cell — 4,860-tok shared prefix, 60 out, 10% miss,
c=16, N=3: spec-on req/s >= 0.9x spec-off AND hit >= 95% AND greedy anchor sha
equal across arms; sub-cells: production gate config + K-pinned. (2) spec-on
cache-meter twin (1 publisher, N-1 capped restores, cached_tokens ==
restore_len, salt-B cold). (3) restored-hit acceptance within a few pp of cold
spec.

## Size/risks

~700-1,000 lines Rust (worker.rs 350-500, spec.rs 250-400) + gates 200-300.
Risks ranked: (1) boundary placement on routed-MoE — the LCP learning insert,
not the seed, delivers the 95%; (2) recurrent capture exactly at boundary,
sub-PRIME_MIN_T veto (:5880-5885, W1-class); (3) entry geometry/budget
accounting + pinned test :9601; (4) exactness through restored spec sessions
(anchor sha gates); (5) one-row draft-anchor degradation (gate 3); (6)
VRAM/admission arms (evict_all-and-retry :7207, observe :780); (7) NOT fixed:
phase-(a) solo-burst serialization :5039 — measure via default-gate sub-cell.
