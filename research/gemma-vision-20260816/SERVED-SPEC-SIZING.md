# gemma4 drafter serve wiring — SIZING (lane/gemma-batched, 2026-08-16)

Coordinator's stop rule invoked: "if the drafter serve wiring turns out days-class
(new scheduler surface rather than routing), report sizing before digging." It is
days-class. This document is the sizing, with receipts.

## What was verified before calling it days-class

1. **The server's spec machinery is the NextN/MTP program, not a generic spec
   surface.** Admission hard-requires `lm.model.mtp.is_some()` (worker.rs
   `session_admits_spec`); gemma4 trunk GGUFs carry no NextN layers — the assistant
   head is a SEPARATE GGUF the engine attaches via `MEMRA_DRAFT` (the `'+draft'`
   NextN attach path explicitly refuses it: hybrid.rs "gemma assistant drafters
   attach via MEMRA_DRAFT, not '+draft'"). The server never constructs a
   `GemmaDraft` anywhere (grep receipt: zero hits in crates/memra-server).
2. **`SpecSession` is a deep scheduler contract**, not a burst function: pending-
   carry across bursts, demote handoff (`spec_flush_pending` → `into_demoted` hands
   the trunk cache to the plain path), prefix-cache boundary capture, session-
   affinity turn checkpoints, Philox sampled-spec continuity, persistent draft-graph
   contexts, telemetry deltas. The worker consumes this surface at the admission,
   burst (`step_spec_pair` → `generate_spec_session_pair`), demote-gate, prefix-
   reuse, and sizing sites.
3. **gemma4's spec is generation-scoped.** `generate_spec_gemma` (gemma_spec.rs)
   builds its own Cache, primes, runs the round loop to completion, returns the
   final token vec. No suffix-fed burst entry, no session persistence, no on_commit
   streaming, no continue-verdict, no demote shape. The engine's session machinery
   explicitly excludes gemma4 (spec.rs:5961-5963) and its verify core asserts
   non-gemma4 (spec.rs:3280 — "the gemma4 arms have their own decode_step_t twins").

## The cheap shape that does NOT meet the scope (rejected)

Whole-request blocking route: at c1, run `generate_spec_gemma` for the entire
request inside the session slot. Hours-class, but it fails the coordinator's own
acceptance criteria:
- **The mixed cell is unimplementable**: while one spec generation runs, the GPU
  worker thread is inside a single engine call — c8 batch traffic stalls completely.
  "c1 spec alongside c8 batch" requires burst-scoped yielding, which is exactly the
  session surface.
- Head-of-line blocking: a 4k-token generation at ~155 tok/s blocks admission ~26s.
- SSE cadence: all tokens arrive at generation end (no on_commit in the gemma arm).
Q38 serving went burst-scoped (MEMRA_SPEC_BURST) precisely to avoid all three.

## The real arc (days-class, ~3-4 days)

1. **Engine — burst-scoped gemma spec session (~1.5-2 days).** A `GemmaSpecSession`
   (trunk Cache + post-norm h carry + next_pred + GemmaDraft trim-adapt state,
   persisted across calls); refactor `generate_spec_gemma`'s round loop into a
   suffix-fed burst entry mirroring `generate_spec_session`'s contract (max_new
   clamp, on_commit slices, continue-verdict). GREEDY-ONLY v1 (the gemma arm has no
   rejection-sampling verify; non-greedy admission stays plain — the batched arm is
   the fallback, which is already the better-than-flat path). The hard part is
   burst-boundary exactness: the Q38 lane's pending-carry/empty-suffix/prime-split
   bug class all lives at these boundaries and each needs its identity gate.
2. **Worker — admission + routing (~1-1.5 days).** Generalize `s.spec` (enum or
   trait) so a session can hold a gemma spec state; gemma admission predicate
   (dense gemma4 + drafter loaded + greedy + unconstrained + spec-gate concurrency
   low, mirroring the Q38 policy shape where it verifies against gemma4's actual
   arms); demote handoff = drop draft scratch, keep trunk cache (the assistant
   drafter reads the TRUNK's KV — Q-only draft attention — so demote is clean by
   construction, but must be gated, not assumed); boot-time per-model drafter
   attach (MEMRA_DRAFT + MEMRA_GEMMA_DRAFT_RANKS + MEMRA_GEMMA_TRIM_ADAPT +
   MEMRA_SPEC=5 config surface) with the dflash-vs-MTP ambiguity guard (3f4597f02's
   pattern) ported to server boot.
3. **Gates + cells (~0.5-1 day).** Served spec-vs-plain byte identity through
   /v1/chat/completions; batched-path identity re-run (no disturbance); Japan @450W
   interleaved ×5: c1 prose (~155 target) + code (~176-179 target) with the
   served-vs-bench delta reported honestly; c8/c16 re-confirmation (245-257); the
   mixed coexistence cell.

## What IS already done on this lane (this arc's prep)

- lane/gemma-assistant merged into lane/gemma-batched (6e5798580): the engine arm,
  ambiguity guard, conversion tooling, and all A/B receipts are in this lane's tree,
  so whichever option proceeds builds from here.
- The batched serve path (default-on) is unaffected and remains the shipping
  gemma4 serving config; c1 traffic serves plain-batched at 58 tok/s single-stream
  until the spec arc lands.

## Recommendation

Fund the 3-4 day arc as specified (shape 1-3). The whole-request shortcut is not
worth shipping even as an interim: it breaks scheduler coexistence, which is the
one property the mixed cell exists to prove. If a faster interim is wanted, the
only honest sub-step is item 2's boot-time drafter attach + config plumbing landed
behind a refusing seam (spec admission still off), which de-risks the config
surface but serves nothing new.
