# specpp2 — PP-2 speculative decode placement/schedule lane

Branch: `lane/cx-specpp2`, base/starting tip `e874528a`. Rig: box1 hyperscaler pair,
2x RTX PRO 6000 Blackwell Server 96 GB. Model: Step-3.7-flash IQ4_XS + embedded MTP.

## Fixed question and decision rule

Why does forced speculative serving lose to plain batched decode on PP-2, and what
placement or schedule can make it win at concurrency 1 or 2?

- Measure one variable per bounded A/B block, N=5 interleaved.
- Promote only a cell that beats the same-block plain denominator and passes
  `run-spec` K=1..8 plus spec/plain byte identity.
- Keep c>=4 plain regardless of any low-concurrency winner.
- If no arm wins, retain PP-2's K=0 placement policy and name the measured
  structural tax plus the mechanism and effort class needed to remove it.

## Starting evidence and code anatomy

- The #87 fix receipt measured forced spec at 112.5 / 112.3 / 112.1 aggregate
  tok/s for c=1/2/4 versus plain 223.3 / 340.3 / 593.4. Spec is a serial-burst
  queue; plain gains from batched stage-split decode.
- The later v0.72 head-affinity fix already removed the placement-order cliff.
  The serving worker follows the last/head stage, where the sharded loader homes
  `output_norm` and the lm head. Both `0,1` and `1,0` now land in the same
  111-112 tok/s spec class. Reversing device order is therefore not a new lever.
- The current round is host-issued in this order: draft chain on the primary/head
  device; PP verify stage 0; one `[T,n_embd]` peer boundary; PP verify stage 1 +
  output head; primary-device accept and replay-free rollback/refresh. Verify is
  a whole `T=K+1` microbatch per stage; it is not microchunked across the two
  stages, so there is no same-round stage overlap.
- Step-3.7's single reused MTP head is trained for +1. The existing SKU gate has
  K=1 acceptance 14/18 (77.8%) and monotonically worse throughput at K>1. This
  makes K=1 the first candidate, but acceptance alone cannot establish a PP win.

## Lane log

- 2026-08-10: `CLAUDE.md` and `research/pp2spec-crash-20260807/PROGRESS.md`
  read; worktree clean on the dedicated branch.
- 2026-08-10: requested `~/.lanectl/inbox/cx-specpp2.md` is absent locally; a
  bounded search found no alternate lane mailbox. Re-check before every GPU block.
- 2026-08-10: current online primary-source scan found the same structural target:
  pipeline-specific speculative schedules keep stages occupied rather than
  serializing draft then a full pipeline verify. This is background only; memra's
  own timing and exactness receipts decide the lane.
- 2026-08-10: anatomy block completed under one exclusive box1 lock
  (`raw/anatomy/`), c=1, forced K=1, 128 output tokens, 0 errors. Both cards were
  empty before and after. Steady T=2 round decomposition: draft 0.70-0.71 ms;
  stage 0 about 8.44 ms; peer TX + local RX about 0.027 ms total; stage 1 + head
  about 8.72 ms; verify accept about 0.024 ms; commit/rollback 0.14-0.17 ms.
  Verify is 95% of round wall. The boundary copy and rollback hypotheses are
  refuted; the tax is two complete, serial stage walks for every T=2 verify.
  This diagnostic intentionally synchronized the natural PP boundaries, so its
  64.4 tok/s request is anatomy-only, not an uninstrumented performance cell.
- 2026-08-10: c=1 K sweep completed under one exclusive box1 lock
  (`raw/k-sweep/`), N=5 interleaved per arm, four 128-token measured requests
  after one warmup per cell, 0 request errors, empty cards before and after.
  Median aggregate throughput: plain 81.188 tok/s; K=1 65.918 (-18.81%);
  K=2 58.357 (-28.12%); K=3 49.591 (-38.92%). Measured-request acceptance was
  deterministic across all five repeats: 72.97%, 51.59%, and 36.61% for
  K=1/2/3 (73.68%, 52.13%, and 36.93% including warmup). The cheapest depth
  lever does not win; deeper K only extends the serial PP verify bubble.
- 2026-08-10: c=2 best-depth A/B completed under one exclusive box1 lock
  (`raw/c2-k1/`), N=5 interleaved, eight 128-token measured requests after two
  warmups per cell, 0 request errors, empty cards before and after. Median plain
  was 115.230 tok/s versus K=1 at 65.953 tok/s (-42.76%). K=1 is flat versus its
  c=1 65.918 tok/s median while plain scales by 41.93%; a second active session
  queues behind the same whole-round serial schedule instead of filling PP
  bubbles. No c=1 or c=2 policy cell qualifies for exactness/promotion gates.
- 2026-08-10: c=1 verify-shape A/B completed under one exclusive box1 lock
  (`raw/verify-shape/`), K=1, N=5 interleaved, four 128-token measured requests
  after one warmup per cell, 0 request errors, empty cards before and after.
  Default whole-T batching was 65.955 tok/s versus `MEMRA_SPEC_M2=0` sequential
  columns at 65.905 tok/s (-0.08%): flat, and both remain 18.8% below plain.
  The anatomy's measured T=1 stage times also put an optimistic two-token
  cross-stage microchunk schedule at 5.557 + 6.179 + max(5.557, 6.179) =
  17.915 ms, already 4.0% slower than the measured 17.224 ms whole-T median.
  Token microchunking loses the existing within-stage T=2 batching benefit; it
  does not create the needed independent work at c=1.
- 2026-08-10: final verdict written to `RESULTS.md`: hold the PP-2 K=0 placement
  policy. The c=2 removal mechanism is a large stage-resident multi-session
  round scheduler plus batched spec prefill; c=1 needs research-grade
  multi-round speculation. No promotion exactness gate was triggered.
