# Composition gate lane: spec x TP-4 x matvec doors at ctx 262144 vs the 100 tok/s bar

Lane: `lane/glm5-composition` (fresh off origin/main 4131e3a59, full glm53 stack + host-audit).
Charter (coordinator, 2026-09-01): prove or refute the composed route to the owner's serving
bar — 100 tok/s single-stream decode at ctx 262144 for GLM-5.3-Flash (glm5_next) on RTX PRO
6000 class cards — by composing TP-4 x speculative decode (DFlash2) x the matvec doors, and
price it with receipts. Spec x 1M is OUT of scope; this lane is the 262k bar only.

Known numbers going in (sources: corpus card + lane docs, quoted verbatim):
- single-card roofline 99 tok/s; ship 3-card PP3 71.49 greedy (deployed 35.36-35.42 served class)
- public TP4 receipts on this model+card class: 139.4/117.2 (external, not memra)
- verify-batch FLIP head: DFlash2 K3+PMIN0.7 45.654 vs plain 35.408 (1.289x), vendor 46.66 vs 33.56
- TP-2 v1 bare: 22.65 tok/s pool (does not pay); join tax ~13-18 ms/tok, EP prime unported
- composed prediction: TP-4 36-49 base x spec x matvec ~1.25

## Pre-registered protocol (before any cell runs)

- LAW:interleave-x3-default: every A/B interleaved x3 fresh boots, x5 on >0.5% spread.
- Boot-nonce arm identity (A/B arm identity, not liveness).
- Greedy is the instrument; serving verdict rows = vendor-default sampled twin (temp 1.0
  top_p 0.95, reasoning_effort pinned) + 8-turn larger-prompt cache-on twin with per-turn
  TTFT+accept. NOTE prefix-cache snapshot is REFUSED for glm5_next (latent planes) — the
  cache-on twin measures the full-re-prefill reality and says so.
- Spec cells: engagement receipts from the server log (K>0 / accept lines), never boot-log
  "ARMED" trust — the spec route is known to print ARMED then serve plain on unadmitted
  placements (fail-closed violation, control-note Amendment queue item 2).
- Matvec door A/B: OFF arms pinned =0 explicitly so default-ON does not make the gate vacuous;
  engagement proven per arm.
- Real prompts (sxc pools), capped max_tokens, loop-law flagging, greedy loops excluded from
  aggregates and reported separately.
- Every measurement receipt carries `git log -1` of the built tree (rebuild-after-checkout law).
- PR #73 (MEMRA_PP_EXIT_PUBLISH, prime-fingerprint race ~1/3) MUST be folded before any
  acceptance-sensitive cell; 5 ppN bodies still carry the hole post-#73 (queued box row, not
  this lane's blocker unless the composed shape exercises them — check before spec cells).
- Byte-identity gate scope per the tp2 class gate: byte bar on non-MoE shard classes, measured
  band (norm_rel 3-5e-2/layer, saturating) + argmax+margin on EP-MoE; never raw elementwise rel.
- TP arms carry MEMRA_RP=0 (shard builders refuse the A6 rp mirror by name).

## Stages

1. Box: a 4-card RTX PRO 6000 WS class host. Qualify: 4 cards, 600W class, driver >= 580,
   host ST gate (20M-iter python loop <= 0.6s). Provisioning, provider, fleet state and
   pricing live in the PRIVATE ops repo (darklanes `research/glm5-composition-20260901/`),
   never here. Stage the artifact, build at the merged tip.
2. TP-4 bring-up at ctx 262144 resident: boot, fail-closed matrix honored, identity gate vs
   single-stream truth, then baseline decode rows greedy + sampled, interleaved x3.
3. Compose spec: find the spec x TP co-refusal seam, minimal correct enablement, byte-identity
   + accept-rate + round-cost receipts (acc/cycle, tok/cycle, round-wall a+bK fit).
4. Matvec doors on TP-4: prove engagement, pinned-OFF A/B.
5. Bar battery: composed shape at 262144, x3 (x5 on spread), greedy + vendor sampled + 8-turn
   twin. Verdict vs 100 tok/s stated plainly.
6. Merge-early: each gate-green stage merges to origin/main; FLAGS.md row in the same PR for
   any new flag.

## Status log

- 2026-09-01: lane opened. Context loaded (corpus card, tp2/flip-reprice/matvec/transport lane
  docs via extractors). PR #73 OPEN at lane start.
- 2026-09-01: HARDWARE FORK SURFACED to the owner — the 4-card box class this lane's bar
  battery needs is a spend decision; the fork, the interim 2-card plan and all fleet state
  are banked PRIVATELY (darklanes `research/glm5-composition-20260901/HARDWARE.md` + the
  lanectl control note of 2026-09-01). Engine work proceeds regardless; every cell not
  needing 4 physical cards runs on the interim box when it lands.
- 2026-09-01: STAGE "TP-4 widening" RIG-GATE GREEN (this PR). `GLM5_TP_RANKS` const 2 replaced
  by the qualified envelope `GLM5_TP_ALLOWED_RANKS = [2, 4]`; the runtime, transport hop
  shapes (fanout/gather/concat/returns + per-rank pub/rel events + all-pairs pull ladder),
  KDA/MLA sidecars, per-rank state planes (memra-kv `Vec<RecurLayer>` / peer-latent
  `Vec<LatentKvLayer>`), EP slabs/ptr-tables, sequential + dieted + grouped-prime EP walks all
  take N ranks; at TWO ranks every arm reproduces the v1 walk hop-for-hop (gate receipt: the
  TP-2 arms' verdicts, bands 2.8e-5..4.85e-5, red magnitudes 1.429e2/2.705e2/6.742e1 and the
  X1 census peer_pulls=1674 pub_events=6696 are IDENTICAL to the tp2/transport lanes' banked
  runs). `glm5-tp-gate` grew the quad battery (fixture 4 KDA + 4 MLA heads, 8 experts; arms
  Q-B/Q-BD/Q-X/Q-XD/Q-XT/Q-M/Q-R1..R4/Q-N3/Q-N2H/Q-H6): decode BYTE-IDENTICAL at four ranks on
  every green arm (incl. diet + peer-pull), transport-vs-transport byte-identical decode AND
  prime, reds bite 1.5e2..4.0e3, envelope refusals by name. NEW CALIBRATION ROW: the quad
  prime near-tie class measures 4.013e-4 max_rel (identical across all quad green arms and
  across runs — one deterministic shard-shape difference; the 2-rank class was 2.8e-5..4.85e-5)
  -> PRIME_BAND_QUAD = 4e-3 (the 10x-over-worst precedent; quad reds sit 4.6+ orders above).
  Rig logs: rig-gates/01 (band discovery, 5 quad prime-band FAILs at the 2-rank band) and
  rig-gates/02 (ALL ARMS PASS). Spec x TP co-refusal and the serving-worker refusal are
  UNCHANGED in this PR (arm F still green) — lifting them is the next stage, gated separately.
- 2026-09-01: STAGE "spec x TP composition" RIG-GATE GREEN (second PR, stacked). The
  co-refusal at glm_spec.rs `glm5_spec_session_new` becomes a GATED admission behind
  `MEMRA_GLM5_SPEC_TP` (default OFF by design, FLAGS row in this PR): `=1` lifts ONLY the
  co-refusal; the batched verify walk is REQUIRED (per-row seam has no TP arm — refused at
  the session AND at the walk), the draft-source law holds. The wiring: sharded-KDA verify
  rounds run per-rank batched `KdaStash::Rows` calls (per-rank pre-round ssm snapshot +
  rows stash in the ckpt's new `kda_tp` plane) joined column-parallel-over-gather with
  rows-exact `wo`; sharded-MLA rides the TP walk with `rows_exact=true` threaded through;
  rollback restores every rank through its own engine (`kda_verify_rollback_rows_on` per
  rank; MLA truncates canonical + every peer replica in lock-step, pool-key clamps
  included); t=1 rounds ride the TP decode walk. GATE (rig-gates/03 discovery run +
  rig-gates/04 committed-tree receipt, ALL ARMS PASS): S2 + Q-S4 verify logits
  BYTE-IDENTICAL sharded-vs-plain at t=K+1=4 on BOTH rank counts (decode-regime state
  build isolates the walk from the prime near-tie class); rollback keep in {1,2,4} +
  6-step continue BYTE-IDENTICAL (accept-j-then-continue identity through TP shards);
  the no-rollback RED diverges; SF1/SF2/SF3/SW refusals all fire by name. What is NOT in
  this stage: the serving-worker TP refusal stands (worker wiring = the box increment);
  zero real-artifact receipts (the box battery owns the flip decision and the composed
  tok/s rows).
- 2026-09-01: **BOX WINDOW COMPLETE — VERDICT BANKED (VERDICT.md).** Took the 2-card window
  on a shared 2-card dev box under the cleanup-batch protocol (announce + DONE in the
  lanectl note; GPU taken only after 10 sustained free minutes, yielded to 0 MiB between
  cells). Staged the real artifact (20/20 shards sha-verified) + the b33c0347 drafter,
  built at the lane tip, and ran W2-G0 (fixture gate ALL ARMS PASS on box silicon), W2-G1
  (real-artifact class gate: 0 argmax forks, worst norm_rel 2.715e-01 in the pre-registered
  EP band; transport twin 44/44 byte-identical on REAL fabric), W2-T (transport re-price,
  x3 interleaved) and W2-S (composed spec rows, x3 interleaved + K ladder).
  THE NUMBER: composed spec x TP-2 peer-pull + diet = **29.684 tok/s** (spread 0.195%),
  vs plain TP-2 hc 19.754, peer-pull 22.902, +diet 25.784, 1-card 19.529, 1-card spec
  19.720. Composition multiplier on the TP shape **1.151x** (vs 1.010x on one card);
  acceptance banked-identical (tok/cyc 2.921 vs 2.907); composed tapes 6/6 byte-identical
  to the plain sharded tapes. **Bar is 100 -> REFUTED as composed**, and the composed shape
  sits at 0.65x of the already-shipped PP3+spec 45.654. Full arithmetic, caveats
  (host-class ST 0.66-0.74 s: absolute rows caveated, ratios unaffected) and the
  recommendation (do NOT spend the 4-card OD box on this route) in VERDICT.md.
- 2026-09-01: #82 review hardening (second reviewer pass, 15 findings): the ARMED receipt
  moved to immediately before the session RETURNS (four fallible steps sat below its first
  home); the whole MLA rollback arm early-outs on full accept (the canonical store was
  unconditional too); the walk-entry guard's message states model truth, not the env; the
  KDA fused-proj door declines shards per-call (the load-time refusal was bypassable by a
  post-load set — the same class SF4 pins for spec); the RED matcher names the actual guard
  phrases instead of the `blk.` prefix (100+ non-guard errors carry it); a new SD arm asserts
  the verify-batch DEFAULT resolves ON with the env absent (every other arm reads a pin);
  the probe clamps its spec tape at EOS and reports `surplus_dropped` + an honest `ctx`
  finish; provision.sh's P2P gate parses the MATRIX rows (a bare `grep OK` matched
  nvidia-smi's legend and passed on a 1-GPU box), its ST probe writes to its own file
  (broken probe now distinguishable from slow host), and an absent artifact exits 70
  STAGING-INCOMPLETE instead of DISQUALIFIED; the TC FLAGS row documents the shard decline;
  CELLS.md states how to actually OBSERVE that decline (it cannot print on a pinned=0 arm).
  Gate ALL ARMS PASS post-hardening (rig-gates/07).
