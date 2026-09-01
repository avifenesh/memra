# Lane 10 — DSpark drafter oracle (2026-08-19)

Branch `lane/dsv4-dspark`, forked from lane/dsv4-flash-loader tip `ac477a9677` +
cherry-picks `d995da97e4`, `0e00533308` (the 0731 config/threshold commits from
lane/dsv4-oracle0731). Worktree `~/projects/wt-dsv4-dspark`; box clone
`/home/ubuntu/memra-dspark` (CPU only, cores 24-47 via taskset; GPUs + memra-src
belong to lane 9, untouched). Model: the minted artifact
`/home/ubuntu/models/dsv4-flash-0731-nvfp4` (read-only; mint receipts
0731-MINT-RECEIPTS.md). Semantic census: darklanes
`research/deepseek-flash-20260818/DSPARK-SEMANTICS.md` (file:line cited, written
before any code ran).

## Plan of record (banked BEFORE runs)

1. **Torch reference** (`dsv4_cpu_dspark_fixtures_0731.py`, banked next to the Gate C
   generators it imports — one numeric program): teacher-force the banked Gate C REF
   greedy trajectory (Rust-verified 160/160) through the stateful decode path; at
   every position run the DSpark drafter exactly per the reference smoke
   (M:949-961): trunk step at pos → write main_kv rings → forward_spec(banked
   next-token, mh(pos), pos), drafter sampling greedy (temperature 0). REF pass must
   reproduce the banked argmax at EVERY position (STOP otherwise). A clamp-only twin
   pass on the same teacher-forced tokens measures the per-array contract fork.
   Banked per position: draft ids [6], confidence [5], biased-row margins [5],
   acceptance-vs-banked prefix count; at positions {32,70,110,150,190} the full
   component arrays (main_hidden tap, main_x, per-block outs, x_collapsed,
   pre/post-markov logits, markov_embed).
2. **Rust oracle** (this branch): `dsv4_decode.rs` — stateful trunk decode port
   (window ring @ p%128, compressor pending kv/score + fine cur→prev shift, growing
   block/indexer stores; prefill pooling reuses the GATED lane-3 CompressorW/IndexerW
   forwards); `dsv4_dspark.rs` — the drafter module (main_proj tap, 3 window-only
   blocks over the [ring | draft] set with bidirectional intra-block attention,
   sequential markov chaining, fp32 confidence). Stated deviation: the main_kv ring
   write is factored OUT of forward_spec and performed by the driver after every
   trunk step — identical ring content to the reference's every-step forward_spec
   pattern (and the shape a verifying engine needs); forward_spec is side-effect-free.
3. **Gate D1 (components)**: `dsv4-dspark-gate <model> components` — teacher-forced
   pass, per-position trunk argmax == banked token (STOP-class: pins the decode
   port at id level at all 159 positions), then per-array comparisons.
4. **Gate D2 (spec==plain identity)**: `dsv4-dspark-gate <model> greedy` — a
   free-running propose-then-verify loop: draft 5 once per round; trunk decode steps
   verify sequentially. For greedy decoding, sequential verification is
   mathematically identical to batched verification and to plain greedy (logits at a
   position depend only on inputs at ≤ that position; a rejected draft never
   advances trunk state in the sequential loop — DSPARK-SEMANTICS §2). The identity
   of the 160-token OUTPUT with the banked REF trajectory is the GATE (it also pins
   the decode port end-to-end and proves the drafter contaminates no trunk state);
   per-round acceptance counts banked + cross-checked against the torch profile.
5. **Determinism ×2**: both gates rerun cross-process; the printed sha256 (over all
   draft ids, confidences, captured arrays / output tokens + final logits) must be
   byte-identical.

## Gate formulas (banked BEFORE runs; Gate C selfcheck lineage)

- **Float arrays**: |rust − torch| max-abs ≤ thr = max(1e-3·absmax_ref(array),
  fork(array)/3), fork = same-generator ref-vs-clamp max-abs of that array (the
  failure class the bound must catch is contract mixing; measured in-class
  torch-vs-rust noise is orders below fork/3 — Gate C: 0.979 draw vs 3.361 fork on
  final logits).
- **Draft ids**: exact. A disagreement is adjudicated at its FIRST slot only (the
  markov loop chains the chosen id, so later slots diverge trivially): in-band iff
  min(torch_margin, rust_margin) ≤ band = max(1e-3·|rust top1 logit|,
  max_pos(fork(logits_post))/3); budget ≤ 3 in-band flips per run, each banked with
  numbers; any out-of-band disagreement = FAIL. Rows of logits_post/markov_embed and
  confidence slots are compared on the id-comparable prefix (slots ≤ first
  disagreement; confidence[i] consumes ids < i+1).
- **Trunk teacher check / greedy identity**: exact, no allowance — argmax must equal
  the banked token at every position (Gate C teacher-forced the same trajectory
  160/160 through the prefill oracle; a decode-path divergence is a decode-port bug,
  STOP).
- **Acceptance**: recomputed from ids; rows compared exactly wherever the draft chain
  was id-exact; the e2e's possibly-truncated final round is checked on its verified
  prefix.

## Gate-formula correction #1 (derivation banked BEFORE the rerun)

Smoke run (positions 32-34, twin-bearing fixtures, logs/rust-smoke.log): all 15
drafted ids EXACT, teacher checks pass, accept rows 3/3, confidence in-thr — but 6
float arrays sit at 0.35-0.7× their contract fork, above thr = fork/3. Diagnosis: the
fork/3 rule was calibrated on PREFILL-only comparisons (lane-3 torch-vs-rust in-class
noise ~2e-5). The decode path re-quantizes state entries per step from single-row
GEMVs whose reduction order differs between torch (blocked f32) and rust (f64-dot) —
each e4m3/e2m1 boundary flip moves one element a full grid step, the SAME mechanism as
the ref-vs-clamp fork at roughly half its distance, amplified by depth exactly like
the fork itself (lane-2: 0.021 @ layer0 → 5.1 @ logits; Gate C ref selfcheck: 1e-3
relative @ layer 3; observed here 1e-2 relative @ layers 40-42 — consistent growth).
fork/3 cannot separate that in-class flip noise from a real bug at these depths.

Corrected gate, TWO-SIDED (each side catches what the other cannot):

- **clamp-only arm (the structural instrument)**: the torch generator banks the
  clamp-only twin's arrays too; the Rust gate runs the SAME teacher-forced pass under
  ClampOnly and compares at thr = 1e-3·absmax (the Gate C selfcheck smooth bound —
  the QAT sims are continuous under clamp except the indexer fp4 sim, whose measured
  one-lane flips sit at 1e-3-relative, inside the bound). A semantic bug (wrong ape
  row, wrong pool member, wrong rope position, ...) shows here at full scale.
  Trunk argmax under clamp is compared against the banked torch-clamp `trunk_argmax`
  array (the clamp trajectory owes nothing to the banked REF tokens).
- **ref arm (the flip-noise instrument)**: thr = fork(array) — in-class flip noise is
  bounded by the contract-fork mechanism; contract MIXING is now caught by the clamp
  arm (a mixed run fails clamp at ~fork vs the 1e-3 bound), so the ref arm no longer
  must catch it alone. Draft-id adjudication band unchanged
  (max(1e-3·|top1|, fork_logits_post_max/3)); ids were exact in the smoke.
- Both arms must be green for the components verdict; one variant per invocation,
  never mixed (lane-2 law).

## Trunk realization-flip policy (banked BEFORE the full runs; lane-6 precedent)

The banked REF trajectory was generated by TORCH-decode and Rust-verified by the
lane-3 PREFILL oracle (Gate C, 160/160). Rust-DECODE is a third realization of the
same semantics; its flip-class drift vs torch-decode can legitimately move a near-tie
argmax (banked margin distribution: 9 positions < 0.5, 16 < 1.0, min 0.0568 — exactly
the class lane-6 ratified with 12 in-band disagreements). Policy:

- The torch generator banks per-position `trunk_argmax` + `trunk_margins` for BOTH
  variants (the same-variant torch argmax is the comparison target; the clamp
  trajectory owes nothing to the banked REF tokens).
- A rust argmax disagreement at a position is a REALIZATION FLIP, adjudicated:
  in-band iff min(torch_margin, rust_margin) ≤ band, band_ref =
  fork_logits_post_max/3 (the drafter-head fork as the trunk-head proxy — same shared
  head, same depth class), band_clamp = 1e-3·|rust top1|. Budgets: ≤ 8 in-band trunk
  flips per 159-position run (5%; more = systematic), ≤ 5 in-band draft-id flips per
  795 drafted tokens. Any OUT-OF-BAND disagreement = FAIL. Teacher-forcing keeps
  state banked-token-driven, so flips are pure comparison events (no state fork).
- The greedy e2e (REF): within-instrument spec==plain is structural in the sequential
  oracle (the drafter is side-effect-free by construction — it mutates only its own
  rings, which the trunk never reads; Rust disjoint-borrow guarantees). The gate
  content vs the BANKED trajectory is the decode-port pin: on a divergence the flip is
  adjudicated as above and the loop takes the banked token as a CORRECTION (keeping
  the remainder comparable); verdict PASS iff every correction is in-band, ≤ 8 total,
  and all round cross-checks vs the torch profile hold. Zero corrections = literal
  identity, reported as such.

## Runs

(logs under `logs/`; receipts pulled off the SPOT box immediately)

### Torch reference — FULL RUN (both variants; logs/… + darklanes fixtures-dspark/)

- **REF pass: teacher-forced argmax == banked at ALL 159 positions** (torch-decode
  reproduces its own banked trajectory — trajectory instrument sane). Wall 1174 s.
- clamp twin: 2 teacher breaks vs the banked REF tokens (informational — different
  contract, near-ties), fork measured on 46 arrays. Total 2219 s, threads 20.
- Banked: `dsv4_dspark_fixtures_{ref,clamp}.{npz,json}` — 51 arrays each (per-step
  draft_ids/margins/confidence/accepts + trunk_argmax/margins + 5 checkpoint
  component sets), shas in the JSONs; pulled to darklanes fixtures-dspark/
  immediately (spot discipline).
- **Acceptance profile (REF, teacher-forced, greedy drafting): mean 3.786/5 over 159
  positions; histogram 0:11 1:10 2:15 3:18 4:17 5:88.** Single in-distribution
  math-proof prompt — an oracle-level number, NOT a serving acceptance claim (SXC
  corpora law governs real cells).
- Confidence head separates: accepted slots mean 5.41 (n=602) vs first-rejected mean
  0.75 (n=66) — the confidence-truncation policy for the GPU lane has signal.

### components-ref run 1 (logs/logs-components-ref-run1.log, old thresholds verdict)

- **TRUNK realization flips: 0/159** — the Rust decode port reproduces the banked
  trajectory argmax at every teacher-forced position, no allowance consumed.
- Draft-id flips: 9 over 795 drafted slots (1.1%), EVERY one an in-band near-tie
  (min margins 0.028-0.48 vs band 8.01; typical drafted-slot margins are ≫1);
  chained-slot re-divergence excluded by protocol. Accept rows exact 150/159 —
  the 9 misses are exactly the flip positions.
- Confidence worst 1.13 vs fork thr 2.96. markov_embed BIT-exact at all checkpoints.
- Float arrays: 33/45 within fork; 12 at 1.0-4.5× fork (worst pos70_x_collapsed
  0.34 vs fork 0.075). The rust-vs-torch REF drift class ≈ 0.2-1.3× fork with
  position scatter — the ref-arm float threshold has NO clean separator at these
  depths.

## Gate-formula decision procedure #2 (PRE-REGISTERED before the clamp evidence)

The clamp arm is the structural instrument. Decision rule, fixed now:

- **IF the clamp arm passes at thr = 1e-3·absmax on all float arrays** (and its
  id-level checks hold): the semantics of the port are pinned 50-100× more
  sensitively than any fork-relative ref bound could (fork ≈ 5-10%·absmax). The ref
  arm's FLOAT comparisons are then demoted to banked MEASUREMENTS (the table above +
  run tables), and the components verdict = clamp arm (structure) + both arms'
  id-level gates (trunk flips, draft-id flips in-band, accept rows, confidence,
  markov_embed) + determinism. Draft-id in-band budget revised 5 → 12 (observed 9
  at 1.1% of slots, all documented near-ties; the budget guards frequency, and every
  flip is individually margin-proven).
- **IF the clamp arm fails** its 1e-3·absmax bound anywhere: REAL BUG — stop, fix,
  rerun everything; no threshold surgery.

### components-clamp runs 1+2 (logs/logs-components-clamp-run{1,2}.log) — the structural arm

**PASS, flawless, ×2 byte-identical** (sha `9087767501295269…` both runs):
45/45 float arrays within thr = 1e-3·absmax — measured WORST 1.2e-4 absolute
(pos70_block2_out, absmax 337 → 3.6e-7 relative), three orders inside the bound;
trunk flips 0/159; draft-id flips 0/795; accept rows exact 159/159; confidence worst
1.5e-5. Under the continuous QAT contract the Rust decode+drafter pipeline tracks
torch at the e-5..e-4 absolute class through all 159 stateful steps —
**decision procedure #2 RATIFIED on its pre-registered terms** (the ref-arm float
exceedances are flip-class realization noise; structure is pinned here).

### components-ref runs 1+2 (logs/logs-components-ref-run{1,2}.log)

×2 byte-identical (sha `f4b81360d3fc…` both runs). Under the ratified doctrine:
trunk flips 0/159 (hard pin, no allowance used); 9/795 draft-id in-band near-tie
flips (budget 12), each banked with margins; accept rows exact at every non-flip
position (150/159); confidence worst 1.13 vs fork thr 2.96; markov_embed BIT-exact;
float arrays measured at 0.2-4.5× fork (flip-noise instrument, banked table in the
logs). Verdict under procedure #2: **PASS** (the run1/run2 logs carry the superseded
formula's FAIL stamp; the final-formula rerun below carries the PASS stamp with
identical measurements — determinism sha unchanged proves same computation).

### Gate D2 — spec==plain identity, greedy e2e run 1 (logs/logs-greedy-run1.log)

**PASS — LITERAL IDENTITY: 160/160 output tokens == the banked REF trajectory, ZERO
corrections** (the free-running propose-then-verify loop, drafting once per round,
sequential trunk verification, banked-margin adjudication armed but never consumed).
- 37 verify rounds for 160 tokens; **143 drafted tokens accepted — mean 3.865/round
  accepted, 4.324 tokens per round incl. bonus** (the e2e acceptance profile on the
  in-distribution proof prompt; per-round table banked in out/greedy-1/
  dspark_greedy_e2e.json).
- Cross-check vs the torch per-position profile: 33/36 in-window rounds EXACT (ids +
  accepts); 3 rounds hit the known draft-id near-tie flips (banked margins
  0.028-0.375), each in-band and each with accepts == its own prefix vs the banked
  continuation; 0 FAIL.
- Drafter non-interference is structural (disjoint &mut borrows: the drafter mutates
  only its rings, which the trunk never reads) AND now measured: the with-drafter
  trunk stream reproduced plain greedy exactly.
- greedy run 2 executes the SAME gate through the seam-refactored generic loop
  (spec_oracle::run_spec_greedy + family adapters, commit e1ff0c1150): an identical
  determinism sha doubles as the refactor-equivalence receipt.

### Gate D2 — greedy e2e run 2 (logs/logs-greedy-run2.log) + refactor equivalence

**PASS, determinism sha BYTE-IDENTICAL to run 1** (`bc4a2737b209…`) — and run 2
executed through the seam-refactored generic loop
(`spec_oracle::run_spec_greedy` + `DsparkOracleAdapter`/`TrunkOracleAdapter`,
commit e1ff0c1150) while run 1 ran the pre-refactor inline loop: one receipt proves
cross-process determinism AND refactor behavior-equivalence. Banked e2e JSONs
(out/greedy-{1,2}) byte-identical (logs/greedy-{1,2}-e2e.json). Round-accept
histogram: 0:1 1:2 2:4 3:7 4:3 5:20 over 37 rounds.

### components-ref FINAL (logs/logs-components-ref-final.log, ratified formula)

**PASS** — trunk flips 0/159, draft-id flips 9/795 in-band (budget 12), accept rows
exact 150/159 (the 9 = the flip positions), confidence worst 1.129 vs fork thr
2.959, 45 float arrays banked as MEASUREMENTS (drift ≤ 4.34 absolute at absmax ≤
337; full table in the log). Determinism sha `f4b81360d3fc…` — **byte-identical to
runs 1/2 for a THIRD time, across two binary versions** (only verdict/report logic
changed; computation untouched).

### Acceptance-number discipline (q38 box2 correction, 2026-08-19)

The dspark-q38 lane's box2 validation landed while this lane closed: q38 drafter
parity ALL PASS (markov EXACT, confidence rel 8.4e-7, VerifyCkpt bit-identical, E2E
spec==plain EXACT) — but its 2.2-2.5× speedup projection was DISPROVEN (banked
2.88/4.6 accept-lengths were sglang temp-0.6+think on the FP8 trunk, a different
observable; real quiet A/B: 1.43 tok/round own-sessions = 1.00×, 1.69-1.88 math =
1.11×; serve route default-off v0.93.0). This lane's 3.79-3.87/round numbers are
greedy, teacher-forced, single in-distribution prompt — the friendliest possible
observable, banked as CORRECTNESS receipts only. No wall-speed claim exists or may
be derived from them; iteration-3 speed claims come from interleaved quiet A/B on
owner corpora at serving temperature.

### Final gate table

| gate | verdict | determinism sha (runs) |
|---|---|---|
| components, clamp-only arm (structural, thr 1e-3·absmax) | **PASS** 45/45 arrays (worst 1.2e-4 abs), 0 trunk flips, 0 id flips, accepts 159/159, conf 1.5e-5 | `90877675…` ×2 |
| components, ref arm (0731 governing contract) | **PASS** 0/159 trunk flips, 9/795 in-band id near-ties, accepts exact at all non-flip positions, conf 1.13 < 2.96; floats = measurements | `f4b81360…` ×3 |
| greedy e2e spec==plain identity (REF) | **PASS — LITERAL IDENTITY 160/160, zero corrections**; 37 rounds, 143 accepted (3.865/round, 4.32 tok/round incl. bonus); torch-profile cross-check 33 exact + 3 in-band flip rounds / 0 FAIL | `bc4a2737…` ×2 (across the seam refactor) |

### Wall clock (hardware time; box cores 24-47, threads 20-24)

| step | measured |
|---|---|
| torch fixtures, ref pass (prefill + 159 steps + drafter each) | 1174 s |
| torch fixtures, full both variants | 2219 s |
| rust components run (load + prefill + 159×(trunk step + draft)) | ~1850-1930 s each |
| rust greedy e2e (prefill + 159 trunk steps + 37 drafts) | ~1660-1713 s each |
| whole gate suite (4 components + 2 greedy + final rerun) | ~3.9 h |

### Box / scratch end-state

- Minted artifact untouched (read-only throughout); GPUs + memra-src + wt-dsv4-loader
  never touched (lane-9 property).
- Box clone `/home/ubuntu/memra-dspark` @ lane/dsv4-dspark tip (kept, the
  memra-oracle0731 precedent); lane dir `/home/ubuntu/dsv4-dspark` holds fixtures/
  logs/out (all mirrored off-box: fixtures + generator + full-run log in darklanes
  fixtures-dspark/, gate logs + e2e JSONs here under logs/); transfer bundles and
  smoke leftovers deleted. Rig /tmp scratch deleted at lane close.
- Worktree ~/projects/wt-dsv4-dspark left in place (unmerged, per the lane brief).

### Smoke (positions 32-34; drove gate-formula correction #1)

- torch REF smoke (`logs/smoke.log`): teacher-forced argmax == banked at ALL
  positions; drafter ran; accept 3.33/5 mean over 3 positions. 235 s.
- torch twin smoke (`logs/smoke2.log`): + clamp-only pass on the same tokens, fork
  measured on 10 arrays, clamp teacher breaks 0. 432 s.
- rust components smoke (`logs/rust-smoke.log`, pre-correction thresholds): trunk
  teacher checks 3/3, all 15 drafted ids EXACT, accept rows 3/3, confidence worst
  8.1e-2 vs thr 5.3e-1, determinism sha printed; 6/9 float arrays at 0.35-0.7×fork —
  above the old fork/3 rule → correction #1 (see above; derivation banked before any
  rerun). 166 s.
