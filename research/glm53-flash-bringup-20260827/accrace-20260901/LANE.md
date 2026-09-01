# lane/glm5-accrace — THE SPEC ACCEPTANCE RACE

**Owner lane opened 2026-09-01. Base: `origin/lane/glm53-flash-bringup` @ `216ffd114`.
Worktree `~/projects/wt-glm5-accrace`, branch `lane/glm5-accrace`. Rig: local RTX 5090,
exactness only (no timing number is read from any log in this lane).**

Escalated here by the debt-payment lane as follow-up 6 of
`../dedup-20260831/LANE.md` §6: `glm5-spec-ppn-gate [E forced-rejection sweep K=7]` is
load-sensitively nondeterministic, perfectly bimodal — **14/42 accepted → PASS, 13/42 →
the e2e tape diverges.** One acceptance, silently lost. Present at stages=2 AND stages=3
(the deployed serving shape), reproducing on the merge target with none of the dedup lane's
code, and invisible to every unloaded gate.

---

## 1. VERDICT

**ROOT CAUSE: a ppN body that returns to its caller published only the PRODUCING stage
stream, so the caller resumed and ALLOCATED while work was still queued on a
caller-co-resident stage stream. `glm5_verify_rollback` had neither an entry fence nor any
exit ordering at all — it was the load-bearing leak.**

The lost acceptance is the last link of the chain, not the defect. The defect's first
observable effect is much earlier and much worse:

> **Under per-stage streams the hc ppN PRIME stopped being a function of its inputs.**
> The same 24-token prompt, the same weights, a fresh cache: repeated primes inside ONE
> process returned **three distinct logit fingerprints**, ~1 in 3 of all primes
> non-canonical. Every one of them kept the same argmax, so nothing was ever loud about it.

Downstream, a drift that happens to land on a close pair of verify logits flips one accept
comparison, the round commits `j-1` drafts instead of `j`, the forced-rejection override's
own cursor model goes stale, and the tape diverges. That is the 14/42 → 13/42 signal.

**FIX: apply the exit half of the boundary law — `PpNRt::publish_all_to`, event waits over
EVERY stage stream, at each body's exit.** Wired at `glm5_verify_rollback` (the measured
leak), `glm5_verify_rows_ppn` and `prime_cache_hyper_ppn`. Rollback/control seam
`MEMRA_PP_EXIT_PUBLISH=0` (default ON, `docs/FLAGS.md` row in the same commit).

**RESULT: 48/48 loaded runs green on every signal across stages=2 and stages=3 and both door
arms, against a control that reproduces (§6.2). 14/14 standing batteries green, clippy zero,
fmt clean, flags census clean (§6.4).**

**Secondary defect (also fixed): the arm's FAIL detail string printed "tape identical" on a
failing line.** Every tape-identity arm in the gate now derives its message from the
comparison it ran and names the first divergence index.

---

## 2. WHY `publish_to(last_stage)` WAS NOT ENOUGH

`PpNRt::fence_stages_behind` (the #87 fix) orders the STAGE streams behind the CALLER at a
body's entry. Its mirror, `publish_to`, orders the caller behind ONE stage at the exit. Two
gaps followed:

1. **A body's terminal drain reaches earlier stages only as far as their `ev_tx`.** The hc
   ppN bodies end with a `dtoh` inside the last-stage scope, which host-syncs that stream;
   the TX-wait chain then covers each earlier stage's work up to the moment it recorded
   `ev_tx`. It does NOT cover what each earlier stage's stream still holds AFTER its tx —
   the stage-scope locals (`pos`, the embedded/expanded rows, the boundary residual, every
   per-layer transient, a verify round's ckpt clones) are dropped with the stage override
   still active, so their `free_async` enqueues on the STAGE stream after `ev_tx`.
2. **`glm5_verify_rollback` published nothing and drained nothing.** Its doc-comment
   reasoned that "the next walk's own entry fence covers the primary-stream seam". It does
   not: the entry fence points the other way. Everything the round does after the rollback —
   the MTP plane reset, the h_seed rows, the next round's whole draft chain, the next
   SESSION's cache allocation and prime — runs on the caller's stream and allocates.

Per the `fence_stages_behind` anatomy already in `pp.rs`, cudarc's drops carry no read
guard, so the pool can hand the caller a block whose stage-stream lifetime has not retired,
and the caller's writes land under queued stage work. An event wait is sufficient (not a
device sync): allocation is a host-side pool operation, and it is the caller's *kernels*
that must be ordered.

### Blast radius, stated precisely

The hazard needs the caller and a stage stream to share a device/context.

| placement | exposure |
|---|---|
| one-card split (`MEMRA_PP_STAGES` with no `MEMRA_PP_DEVICES`) — every gate arm on this rig | EVERY stage shares the caller's device: fully exposed |
| a placement with a REPEATED device (`MEMRA_PP_DEVICES=0,1,0`) | the repeated non-head stages are exposed |
| distinct-device split (the real multi-card serving shape) | only the HEAD stage shares the caller's context, and a body's terminal drain already covered it — **except `glm5_verify_rollback`, which had no drain, so its head-stage restores were unpublished on EVERY placement, including multi-card** |

So the prime/walk half of this fix hardens the one-card and repeated-device regimes; the
rollback half is a real serving-shape correctness fix at any stage count and any placement.
This is also one named mechanism of the "same Engine kernels concurrent on two streams of
one device" root cause that `pp::pp_multi_stream_same_device`'s doc has carried as OPEN since
2026-08-02. Stated narrowly: measured on the SERIAL arm. Whether the refused PIPELINED arm's
remaining flake is the same mechanism is **unmeasured**, and that refusal stands.

---

## 3. THE HUNT, step by step (receipts in `receipts/`)

Everything below ran `flock /tmp/memra-5090.lock`, `NVIDIA_TF32_OVERRIDE=0`, on the
`./target/debug/` binary the gate matrices actually use, under deliberate capped host load
(`nice -n 19` spinners in systemd user scopes with a CPUQuota).

### 3.1 Reproduced — `receipts/baseline-n2-OFF/`

12 reps, stages=2, dedup doors pinned `=0` (never merely unset): **3/12 FAIL**, every
failure exactly `13/42` and every pass exactly `14/42`. The bimodality is confirmed on this
tree and this load window.

*(For the record: `14/42` is the CORRECT value, not a suspicious one. Rounds accept
j=0,1,2,3,4 and then round 5 is capped by the tape's own length, so `0+1+2+3+4+4 = 14`.)*

### 3.2 The accept path is INNOCENT — `receipts/hunt/probe-*.log`

A gate instrument (`Glm5SpecKnobs::accept_probe`, kept) traces every greedy round: the
DEVICE accept row (`argmax_token_device_col` + the one u32 readback the walk consumes)
against a HOST argmax over the same `vlogits`, plus a per-row (argmax, FNV-1a of the row's
f32 bits) census. The host read is issued after the device path's own `dtoh_u32` has already
synchronized, so the probe can only observe the race, never mask it — and indeed the flake
still reproduced 6/20 with it on.

Result: **`agree=true` in every round of every PASS and every FAIL.** The device accept
walk always published exactly what the logits buffer justified. The logits themselves were
wrong.

### 3.3 It is a CROSS-STREAM race, not a nondeterministic kernel

| arm | forced-rejection FAILS |
|---|---|
| default (per-stage streams) | 6/20 |
| `MEMRA_PP_STREAMS=0` (same-stream seam) | **0/20** |

And the `STREAMS=0` round-0 row hashes are **bit-identical** to the multi-stream PASS runs'
— so `STREAMS=0` is the truth and the multi-stream FAIL runs are corrupted, not merely
different. Four separate PASS runs were bit-identical to each other, so the program IS
deterministic when it is not raced.

### 3.4 Every failure diverges at ROUND 0

Six independent FAIL traces, first differing round against the canonical PASS trace: **0, 0,
0, 0, 0, 0.** The corruption is established before the session's first verify round, which
kills "round N's rollback corrupts round N+1" as the *proximate* story and points upstream,
at the session's own setup.

### 3.5 The prime is nondeterministic — `receipts/hunt/prime-determinism-controls.txt`

A hash of the prime's own logits, printed once per session (11 sessions per gate run), on
the pre-fix tree:

* `MEMRA_PP_STREAMS=0`: **one fingerprint, 11/11.**
* `MEMRA_PP_OVERLAP=0`, per-stage streams: **three fingerprints.** Double-buffered slot
  alternation is exonerated.
* default, stages=3: **two fingerprints.** Nondeterministic at the deployed stage count too.

`receipts/hunt/prime-vs-verdict-correlation.txt` closes the loop over 14 reps: the FIRST
prime in a process is ALWAYS canonical (nothing is in flight yet), a non-canonical
forced-rejection prime always moves round 0's row hash with it, and the arm fails exactly
when that drift happens to flip an argmax. `rep12` is the instructive one — a corrupted
prime that PASSED. The gate's failure rate is the probability of a coincidence, which is why
8 base arms and ~34 settled reps never saw it.

**This is 11x more sensitive than the end-to-end tape and it became the new gate arm (§5).**

### 3.6 Bisect of the missing dependency — `receipts/hunt/settle-bisect-summary.txt`

A temporary bitmask scaffold placed candidate publications one at a time, interleaved in one
load window per block:

| candidate publication point | non-canonical primes | arm FAILS |
|---|---|---|
| none (control) | 20/110 · 37/110 | 3/10 · 2/12 |
| verify WALK exit only | — | 1/12 (does not close it) |
| `prime_cache_hyper_ppn` own exit only | 31/110 | 3/10 (**flat**) |
| **after `glm5_verify_rollback`** | **2/110** | **0/12** |

The rollback exit is the load-bearing seam. The residual 2/110 is the earlier stages'
post-`ev_tx` tails at the other bodies' exits, which the shipped fix closes at the walk and
the prime as well. The scaffold is REMOVED from the shipped tree; the receipts are what
remains of it.

---

## 4. THE FIX (shipped in this lane)

| file | change |
|---|---|
| `crates/memra-engine/src/pp.rs` | **new** `PpNRt::publish_all_to(dst)` — `publish_to` over every stage, with the full anatomy and the measurement in its doc; **new** `pp_exit_publish()` reading `MEMRA_PP_EXIT_PUBLISH` (default ON, `0` = the known-racy control arm); a narrow cross-reference added to `pp_multi_stream_same_device`'s long-open root-cause note. |
| `crates/memra-engine/src/glm_spec.rs` | `glm5_publish_stages()` helper (no-op with the door shut or on the `STREAMS=0` seam); exit publication at `glm5_verify_rollback` and at `glm5_verify_rows_ppn`; the stale "the next walk's own entry fence covers the primary-stream seam" reasoning replaced with why it does not; `Glm5SpecKnobs::accept_probe` + `glm5_accept_probe()` kept as documented gate instruments. |
| `crates/memra-engine/src/hybrid_forward.rs` | exit publication at `prime_cache_hyper_ppn` (the same law; measured FLAT on its own, kept because its earlier stages are caller-co-resident on any one-card or repeated-device placement). |
| `crates/memra-engine/src/bin/glm5_spec_ppn_gate.rs` | **new arm P0 prime-determinism**; `tape_verdict()` / `first_diff()` so every tape arm reports the comparison it ran; the accept probe and the P0 rep count as positional args (arg 4, arg 5) — deliberately not env flags. |
| `docs/FLAGS.md` | the `MEMRA_PP_EXIT_PUBLISH` row, with both arms, the rollback seam and the receipts pointer, in the same commit as the flag (the new-flags law). |

Not a sleep, not a retry, not a device sync: an explicit event dependency, exactly the
primitive `pp.rs` already documents for the other direction.

---

## 5. THE ARM THAT WAS MISSING, and the standing gate

Every arm in `glm5-spec-ppn-gate` compared a door-ON walk against a door-OFF walk ONCE.
None of them could see a walk that is merely *not a function of its inputs*. **Arm P0
`prime-determinism`** re-primes the same prompt R times (default 8) under the split and
holds every one to the door-OFF prime bit for bit. It runs BEFORE the walk arms so a
non-deterministic prime is named as such instead of surfacing many rounds later as a
confusing tape failure.

**PASS-line count of a fully green run moves 23 → 24.** Any runner asserting 23 will now
fail loudly with a wrong-count message rather than silently pass — including the banked
`../dedup-20260831/receipts/run-matrices.sh`, which is a closed lane's historical receipt
and was deliberately NOT edited.

`repro.sh` is the banked regression harness (CI has no GPU, so the loaded protocol IS the
gate). It asserts, per rep: exit 0, the sweep arm PASS, the P0 arm PASS, and the pass-line
count. Its load window is raised INSIDE the rig lock and dropped before the lock is
released, so a co-tenant lane holding `/tmp/memra-5090.lock` is never perturbed by it —
learned during this lane, when another lane's `graph-warmup-stress` gate held the rig for
over half an hour while an earlier version of this harness kept spinners up the whole time.

```
# build OUTSIDE the lock (the sccache-under-flock deadlock trap)
cargo build -p memra-engine --bin glm5-spec-ppn-gate
research/glm53-flash-bringup-20260827/accrace-20260901/repro.sh fix-n2 2 12
research/glm53-flash-bringup-20260827/accrace-20260901/repro.sh fix-n3 3 12
research/glm53-flash-bringup-20260827/accrace-20260901/repro.sh ctl-n2 2 12 MEMRA_PP_EXIT_PUBLISH=0
```

---

## 6. RESULTS

### 6.1 The load window had to be built twice, and the first version proved nothing

Stated up front because it is the most reusable lesson here. The first interleaved A/B ran
12x2 with capped host spinners only and returned **FIX 0/12 and CTL 0/12** — and `ab.sh`
correctly **refused itself** (`exit=1`) on its own liveness clause rather than banking a
"green" that had tested nothing. Two causes, both fixed:

1. **The original reproduction had an unrelated CUDA context on the card.** Once the harness
   raised its load window strictly inside its own rig lock, the GPU was idle during every
   measured rep and the race window closed. The window now also runs a SECOND CUDA CONTEXT
   of our own gate binary as load (output discarded — load, never a sample), which is the
   brief's own suggested generator and never another lane's live work.
2. **Arm P0 was placed in the quiet regime.** It ran before any spec session, i.e. before any
   rollback or stage-owned-cache teardown was in flight — the same regime in which the hunt
   measured the FIRST prime of a process canonical in 14/14 reps. A first pass of P0 went
   0/96 deviations on a tree that still had the defect. **Arm `P1
   prime-determinism-post-spec` was added after arm E and is the actual detector.**

### 6.2 The loaded interleaved A/B — `receipts/ab-*/ab.txt`

Fix arm = default (`MEMRA_PP_EXIT_PUBLISH=1`), control arm = `MEMRA_PP_EXIT_PUBLISH=0`,
alternating rep by rep in ONE load window per rep, 12 reps per arm per cell.

| cell | arm | sweep FAILS | P0 FAILS (dev) | P1 FAILS (dev) |
|---|---|---|---|---|
| stages=2, doors pinned `=0` | **FIX** | **0/12** | **0/12 (0/96)** | **0/12 (0/96)** |
| | CTL | 3/12 | 0/12 (0/96) | 5/12 (5/96) |
| stages=3, doors pinned `=0` (the deployed shape) | **FIX** | **0/12** | **0/12 (0/96)** | **0/12 (0/96)** |
| | CTL | 3/12 | 0/12 (0/96) | 0/12 (0/96) |
| stages=2, doors E ON | **FIX** | **0/12** | **0/12 (0/96)** | **0/12 (0/96)** |
| | CTL | 0/12 | 0/12 (0/96) | 2/12 (2/96) |
| stages=3, doors E ON | **FIX** | **0/12** | **0/12 (0/96)** | **0/12 (0/96)** |
| | CTL | 0/12 | 0/12 (0/96) | 0/12 (0/96) |

**FIX: 48/48 runs green on every signal, across both stage counts and both door arms.**

Three of the four cells have a reproducing control; the fourth (stages=3 + doors E) does NOT,
so `ab.sh` marked that cell `exit=1` and **it is not counted as A/B evidence** — its FIX arm
is banked as a plain green run and nothing more. Recording it that way is the point: a cell
whose control stayed silent cannot testify about the fix.

Two things the table says that are worth keeping:

* **The two signals are complementary, not redundant.** At stages=2 the control shows up on
  P1 (5/12) more than on the sweep (3/12); at stages=3 it shows up ONLY on the sweep (3/12);
  with doors E ON it shows up ONLY on P1 (2/12). Neither alone would have covered all three.
  That also explains, without hand-waving, why the debt lane measured a LOWER rate with the
  doors ON (2/12) than with them pinned `=0` (5/12) — the doors were never the variable.
* **The blast radius is wider than the arm it was reported on.** In the control probe
  (`receipts/probe-ctl/rep1.log`) the pre-lane program failed **`E natural K=7` and BOTH
  `E forced-accept` arms** in one run, each diverging at tape index 2 (`got Some(29), want
  Some(2)`). The forced-rejection sweep is simply where it was first caught.

### 6.3 The repaired FAIL message, in the wild

The old arm printed `... tape identical (13/42)` under the word FAIL. The same failure now
reads:

```
glm5-spec-ppn gate FAIL [E natural K=7]: natural drafter, TAPE DIVERGED from door-OFF
plain greedy at index 2 (got Some(29), want Some(2); lens 20/20) (0/133)
```

and the forced-accept arm additionally states `full-accept path NOT exercised — accepted*2 <
drafted` instead of claiming it was exercised.

### 6.4 Standing batteries on the final tree — `receipts/batteries/`

The hc ppN prime is SHARED, so its own gates re-ran, not only the glm5 spec ones. **14/14
green, TOTAL FAILS=0.**

| | result |
|---|---|
| `glm5-hyper-ppn-gate` 2/6/8, 3/6/8, 2/6/8 `STREAMS=0`, 2/16/24 long | 4/4 arms PASS, 6 PASS lines each |
| `glm5-spec-ppn-gate` 2/24/20, 3/24/20, 2/24/20 `STREAMS=0` | 3/3 arms PASS, 25 `gate PASS` lines each (was 23) |
| `glm5_tparallel_verify_gpu` | 9 passed |
| `glm5_verify_batch_gpu` | 4 passed |
| `glm5_spec_session_gpu` | 10 passed |
| `glm5_dflash_session_gpu` | 10 passed |
| `hyper_connections_gpu` | 7 passed |
| `glm5_chunked_prime_gpu` | 6 passed |
| `glm5_mtp_head_gpu` | 6 passed |

`cargo fmt --all --check` clean · `cargo clippy --all-targets` **ZERO lints** ·
`tools/check-flags.sh` 755 runtime literal reads, no uncovered names, no grandfather list.

**`tools/local-ci.sh --perf` RAN ON THE FINAL TREE AND IS GREEN** — not skipped, not deferred
(`receipts/batteries/local-ci-perf.log`, 931 lines banked). Correctness stage: `kernel-check
ALL GREEN (105 cells, 11 skipped)`, `decode-batch-gate` 4/4 ALL GREEN (9B NVFP4 + Q8_0,
config B=8 and strict B=4 equalized), `graph-warmup-stress` ALL GREEN (10 cycles x 4 arms +
overlap, bit-identical, no fault) and its gate, `serve-stress-gate ALL GREEN (c=64)`,
`SPEC-ON-CACHE-HIT GATE ALL GREEN (qwen)` incl. r1/r2/r3/g1/g2 spec==plain byte identity.
Perf stage: **0 fail, 0 warn**, `qwen9b-plain-short 138.23 tok/s [OK]` (inside the last
measured-green band ~135-138.6). Every `FAIL` string in that log is a red arm asserting its
own expected failure (`rc=1 (expected 1)`), grepped and checked rather than assumed. Per the
rig law no timing number from this lane's own cells is read as a claim; the perf row above is
local-ci's own regression check, quoted as it printed.

### 6.5 RE-VERIFIED ON THE MERGED TIP (the base moved 117 commits mid-lane)

`origin/lane/glm53-flash-bringup` advanced from `216ffd114` to `469e03a81` while this lane
ran, touching `pp.rs` (6 commits), `hybrid_forward.rs` (21) and `docs/FLAGS.md` (38). The lane
was merged onto that tip and **everything was re-measured there** — a pre-merge green does not
carry across 117 commits.

* The one merge conflict was `research/tune-data/perf-ci.jsonl`, resolved as the append-only
  log it is: a **timestamp-ordered UNION**, never a side chosen (the four upstream rows plus
  this lane's own), and the file re-validated as JSONL line by line.
* Interleaved A/B on the merged tree, stages=2, doors pinned `=0`, 12 reps per arm:
  **FIX 0/12 sweep, 0/96 P0 dev, 0/96 P1 dev; CTL sweep 1/12, P1 1/12** — the control still
  reproduces, so the cell is evidence.
* Standing batteries on the merged tree: **12/12, TOTAL FAILS=0**
  (`receipts/batteries-merged/SUMMARY.txt`) — `glm5-hyper-ppn-gate` 2/6/8, 3/6/8, 2/16/24;
  `glm5-spec-ppn-gate` 2/24/20, 3/24/20, 2/24/20 `STREAMS=0` at 25 `gate PASS` lines each;
  `glm5_tparallel_verify_gpu` 9, `glm5_verify_batch_gpu` 4, `glm5_spec_session_gpu` 10,
  `glm5_dflash_session_gpu` 10, `hyper_connections_gpu` 7, `glm5_chunked_prime_gpu` 6.
* Merged tree: `cargo fmt --all --check` clean, `cargo clippy --all-targets` **ZERO lints**,
  `tools/check-flags.sh` 765 runtime literal reads, no uncovered names, no grandfather list.
  Rebuilt from scratch after the merge before any of the above ran
  (rebuild-after-checkout-attribution).

### 6.6 RE-GATED AGAIN ON `origin/main` (the owner's merge-to-main workflow, 2026-09-01)

Mid-lane the owner changed the workflow: gate-green stages merge to `origin/main` rather than
living on a side branch. `origin/main` was merged in (19 commits ahead, 5 conflicts) and the
WHOLE gate set re-ran on that tree. `receipts/MAINMERGE-REGATE.out`,
`receipts/batteries-mainmerged/`, `receipts/ab-mainmerged-n2-OFF/`.

* **rerere is a shared cache and it was NOT trusted.** Four of the five conflicts were
  auto-resolved "using previous resolution" from another lane's merge, which is exactly how
  `origin/main`'s own head commit came to exist (a `MEMRA_PARALLEL_EP_Q8_SCOPE` row dropped by
  a take-theirs resolution, caught by the census). Every flag `origin/main` added in those 19
  commits was enumerated from its own diff and grepped back out of the merged
  `docs/FLAGS.md` — **46/46 present**, and `tools/check-flags.sh` re-ran clean at 772 runtime
  reads. The fifth conflict (`perf-ci.jsonl`) was resolved as the append-only log it is: a
  timestamp-ordered UNION, re-validated line by line as JSONL.
* Full rebuild after the merge before anything was measured
  (rebuild-after-checkout-attribution).
* **Loaded interleaved A/B, stages=2, doors pinned `=0`, 12 reps per arm — the strongest
  control window of the lane:**

| arm | sweep FAILS | P0 FAILS (dev) | P1 FAILS (dev) |
|---|---|---|---|
| **FIX** | **0/12** | **0/12 (0/96)** | **0/12 (0/96)** |
| CTL | 5/12 | 0/12 (0/96) | 5/12 (5/96) |

  The control also produced a SECOND, larger failure signature here — `accepted=(10/63)`
  instead of the bimodal `13/42`, with `pass_lines` down to 22-24 — i.e. the corruption is not
  always a single lost acceptance; it can restructure the whole round schedule and take several
  arms with it.
* **Standing batteries: 12/12, TOTAL FAILS=0** (`glm5-hyper-ppn-gate` 2/6/8, 3/6/8, 2/16/24;
  `glm5-spec-ppn-gate` 2/24/20, 3/24/20, 2/24/20 `STREAMS=0`, 25 `gate PASS` lines each;
  tparallel 9, verify-batch 4, spec-session 10, dflash-session 10, hc 7, chunked-prime 6).
* **`tools/local-ci.sh --perf` exit 0**: `kernel-check ALL GREEN`, `decode-batch-gate` 4/4 ALL
  GREEN, `graph-warmup-stress` + gate ALL GREEN, `serve-stress-gate ALL GREEN (c=64)`,
  `SPEC-ON-CACHE-HIT GATE ALL GREEN (qwen)`, perf stage **0 fail, 0 warn**,
  `qwen9b-plain-short 137.06 tok/s [OK]`.
* `cargo fmt --all --check` clean, `tools/check-flags.sh` clean at 772 reads.

### 6.7 Final tip re-gate (main advanced again during the gate window)

`origin/main` moved 4 more commits (`79bfd84e1`, EP/TP compose + hy3 shared-expert overlap)
between the previous re-gate and the push. Merged, rebuilt, re-gated a third time — a
non-fast-forward race is the expected shape of merge-to-main, not a reason to push a stale
tree. `receipts/FINAL-TIP-REGATE.out`, `receipts/batteries-final/`,
`receipts/ab-final-n2-OFF/`.

The single conflict (`hybrid_forward.rs`) was again rerere-applied and again audited rather
than trusted: both incoming commits' added lines were enumerated and grepped back out of the
merged file, and the one apparent miss was a `cargo fmt` rewrap of `SHEXP_OV_AUTO` with
identical semantics, confirmed line by line against `origin/main`'s own copy.

| | result |
|---|---|
| standing batteries | **12/12, TOTAL FAILS=0** |
| loaded A/B stages=2 doors=0, 12 reps/arm | **FIX 0/12 sweep, 0/96 P0, 0/96 P1** · CTL sweep 2/12, P1 2/12 (control reproduces) |
| `tools/local-ci.sh --perf` | **exit 0**, perf stage 0 fail 0 warn, `qwen9b-plain-short 138.85 tok/s [OK]` |
| fmt / check-flags | clean / no uncovered names |

Across the three trees this fix was measured on (lane base, `origin/main`-merged, final tip)
the FIX arm is **84/84 loaded runs green on all three signals**, and the control reproduced on
every tree.

### 6.8 Fourth re-gate, and the merge to `origin/main`

The owner's merge-early doctrine (gate-green = the merge bar) supersedes this lane's original
"no self-merge" brief, so the lane merges to `main` itself. `origin/main` had moved twice more
— the full glm53 bringup stack at `3347e4bbd`, then host-audit's 7 commits at `4131e3a59`
(`apply_penalties_dense` included). Merged, resolved, rebuilt, re-gated a fourth time.
`receipts/MERGE4-REGATE.out`, `receipts/batteries-merge4/`, `receipts/ab-merge4-n2-OFF/`.

**The journal resolution was rewritten to be provable, not careful.** Three earlier merges
conflicted on `research/tune-data/perf-ci.jsonl` and my second pass duplicated a row across a
hunk boundary. It is now resolved from the THREE VERSIONS rather than from conflict markers —
merge-base, ours, theirs — as: base rows in base order, plus the deduped chronological union of
what each side ADDED relative to the base. Then it ASSERTS, and the assertion is the point:

```
base=1065 ours_added=3 theirs_added=3 union_added=6 final=1071
rows missing from OURS=0  rows missing from origin/main=0  duplicates=0  valid JSONL
```

A resolution that cannot state that it lost nothing and duplicated nothing is not a resolution.
The one pre-existing out-of-order row at base index 1036 is main's own and is left alone rather
than silently rewritten.

**`docs/FLAGS.md` auto-merged, which is exactly where this repo has been bitten**, so it was
audited three ways rather than trusted: the `MEMRA_PP_EXIT_PUBLISH` row is present (1); main's
four incoming names (`MEMRA_AFFINITY`, `MEMRA_DEBUG_AFFINITY`, `MEMRA_WORKER_AFFINITY`,
`MEMRA_WORKER_CPUSET`) are present; and FULL SET CONTAINMENT holds in both directions over all
**876** flag names. `pp.rs` and `hybrid_forward.rs` also auto-merged, so every line main's
incoming commits added to them was enumerated and grepped back out — zero missing.

| | result |
|---|---|
| standing batteries | **12/12, BATTERY TOTAL FAILS=0** |
| loaded A/B stages=2 doors=0, 12 reps/arm | **FIX 0/12 sweep, 0/96 P0, 0/96 P1** · CTL sweep 2/12, **P1 6/12** |
| `tools/local-ci.sh --perf` | **exit 0**, 12 ALL-GREEN gates, perf stage 0 fail 0 warn, `qwen9b-plain-short 138.67 tok/s [OK]` |
| fmt / clippy / check-flags | clean / zero lints / 774 reads, no uncovered names |

Across the **four** trees this fix has now been measured on, the FIX arm is **96/96 loaded runs
green on all three signals**, with a reproducing control on every tree.

### 6.9 Fifth re-gate, the boundary-gate collision, and a red `main` repaired

`origin/main` moved twice more (PR #74 `lane/nvfp4-quad-symbol-20260831`). Two things in it
matter to this lane:

**(a) CI was red on `main` itself, for the reason this PR now fixes.** `main` @ `4131e3a59`,
`3347e4bbd`: job `build`, step `Public-boundary policy check`, `PermissionError` on
`.../mv-battery-20260831/receipts/c3/off-1 -> /root/out-mv/c2/off-1`. `evaluate()` calls
`is_file()`, which STATS; the CI runner's Python raises when an absolute target crosses an
unreadable parent, this workstation's returned False. **The gate therefore scanned nothing after
that file** — the worst shape a censusing gate can have. Fixed two ways here: an unstat-able
tracked path is now its own reported finding (never a traceback, never a silent skip), and the
seven tracked symlinks that pointed OUT of the repo (six at `/root/out-mv/`, one at
`/opt/dl-image/nvme/`) were retargeted **losslessly** at their in-repo twins, which existed beside
them all along (`../c2/off-N`, `../c2/don-N`, `build-20260812T065133Z`). An absolute target is
unresolvable on every machine but the one that wrote it, so those six control arms were never
receipts; they resolve now for the first time.

**(b) Another lane fixed the same crash differently, and the merge kept both halves.** `main`'s
`40a97ee79 fix(ci): scan symlink blobs without dereferencing` reworked `evaluate()` to read the
tracked symlink BLOB (`worktree_blob_bytes`) instead of following the link, and to scan every
link because `git grep` cannot prefilter them — strictly better than my version for symlinks,
and it closes a real blind spot. Their non-symlink arm still `raise`d, though, which kills the
scan for every later file. The conflict was resolved as a **synthesis, not a side**: their
symlink handling, plus my report-and-continue for a non-symlink that cannot be stat'd. Teeth
re-proven against the merged form — reverting my half to their raising version turns
`UnstatablePathTests` red (`FAILED (errors=2)`), restoring it green.

`tools/test_public_boundary.py`: **51/51 OK** · `check` 687 matches / 0 new ·
`verify-allowlist` 687 entries all live · no tracked symlink absolute or dangling.

| merge5 tip | result |
|---|---|
| standing batteries | **11/11, BATTERY TOTAL FAILS=0** |
| loaded A/B stages=2 doors=0, 12 reps/arm | **FIX 0/12 sweep, 0/96 P0, 0/96 P1** · CTL sweep 2/12, P1 2/12 |
| `tools/local-ci.sh --perf` | **exit 0**, perf 0 fail 0 warn, `qwen9b-plain-short 137.60 tok/s [OK]` |
| fmt / check-flags / FLAGS+journal audit | clean / clean / 876 flag names and every journal row of both parents present, 0 duplicates |

Across the **five** trees this fix has been measured on, the FIX arm is **108/108 loaded runs
green on all three signals**, with a reproducing control on every tree.

**NOT MERGED BY ME.** The owner's 2026-09-01 policy amendment supersedes the earlier
merge-early instruction: every stable point opens a PR, gets a review, passes CI, and only then
merges. PR #73 carries a reviewer checklist naming the four risk seams (publish-hole call sites
and their `rt.enter` scoping, the FLAGS row, the journal-union assertions, the boundary-gate
synthesis). `release-arch-mirror (89)/(90a)` had been red on `main` before this branch carried
that content and now go **pass** — another lane closed the cause; see §7 item 4.

### 6.10 Sixth re-gate — and a receipt note that said the evidence was absent when it was not

`origin/main` moved 7 more commits (PRs #75, #78). Two of them (`571c25516`, `b25d021a3`) had
independently made the SAME symlink relink this lane made, which is a good sign about the
diagnosis. One of them also went further and got a fact wrong, and that is worth recording:

PR #75's sweep replaced `research/requal-20260812/raw/setup/latest-build` with a note reading
*"the target string is the receipt; the build tree itself was never committed."* **The build tree
is committed, and always was** — four files under
`research/requal-20260812/raw/setup/build-20260812T065133Z/` (`build.log`, `build.ok`,
`git-status.txt`, `runtime-binaries.sha256`), verified with `git ls-tree -r origin/main`, and
that directory's basename is exactly the last component the box-absolute link named. So the
correct fix is the repo-relative relink, not a note declaring the evidence gone. The merge
therefore keeps the working link and REWRITES the note with the correction, because a receipt
note that says raw runs were never committed, when they were, is how a real receipt gets thrown
away later by someone who takes the note at its word. The absolute target string is preserved in
the note for provenance.

(For the mv-battery links the two lanes agreed: relink relative to the committed `c2` arm, which
`c4.console` independently confirms was c4's K-ladder baseline.)

| merge6 tip | result |
|---|---|
| standing batteries | **11/11, BATTERY TOTAL FAILS=0** |
| loaded A/B stages=2 doors=0, 12 reps/arm | **FIX 0/12 sweep, 0/96 P0, 0/96 P1** · CTL sweep 3/12, P1 2/12 |
| `tools/local-ci.sh --perf` | **exit 0**, perf 0 fail 0 warn, `qwen9b-plain-short 137.83 tok/s [OK]` |
| boundary gate | `test_public_boundary` 51/51, `check` 687/0 new, no tracked symlink absolute or dangling |
| fmt / check-flags / FLAGS+journal audit | clean / clean / 876 names both directions, both parents' journal rows present, 0 duplicates |

Across the **six** trees this fix has been measured on, the FIX arm is **120/120 loaded runs
green on all three signals**, with a reproducing control on every tree.

### 6.11 The pre-existing-head check

The defect is **pre-existing, proven by diff rather than by a rebuild**: every site this lane
touched is byte-identical between the pre-doors head `92ea07376` and the lane base
`216ffd114` —

```
git diff --shortstat 92ea07376 216ffd114 -- crates/memra-engine/src/glm_spec.rs   # (empty)
git diff --shortstat 92ea07376 216ffd114 -- crates/memra-engine/src/pp.rs         # (empty)
# hybrid_forward.rs changes in that range are all at ~line 13005+, far from
# prime_cache_hyper_ppn (~line 2332)
```

so the `MEMRA_PP_EXIT_PUBLISH=0` control arm **is** that head's program for this defect, and
it is the arm that reproduces above. This also matches the debt lane's own attribution: the
doors-pinned-`=0` control failed at a HIGHER rate than the doors-ON arm.

---

## 7. SIBLING SWEEP

The same "returns with stage work enqueued" shape, audited across every ppN body
(`fence_stages_behind` call sites plus `glm5_verify_rollback`):

**Closed by this lane**

* `glm5_verify_rollback` — had NOTHING at either end. The measured leak. Exposed on every
  placement (its head-stage restores share the caller's context even cross-device).
* `glm5_verify_rows_ppn` — last-stage host drain only.
* `prime_cache_hyper_ppn` — last-stage `dtoh` only.

**Already correct**

* `prime_chunk_ppn`, `step35_prime_cache_batch`, `prime_cache_pp2_pipelined`,
  `decode_step_batch_dual`, `gemma4_batch_attn`, `spec.rs`'s verify issue — all publish at
  the exit, though only the LAST stage (see below).

**NAMED, NOT CLOSED HERE — needs its own lane and its own measurement**

1. **Five ppN bodies still exit with no publication at all, relying on a terminal `dtoh` of
   the last stage:** `decode.rs::decode_step_h_ppn`,
   `decode_batch.rs::decode_step_batch_ppn`,
   `decode_batch.rs::decode_step_batch_hyper_ppn`,
   `hybrid_forward.rs::forward_hyper_ppn`, `hybrid_forward.rs::decode_step_hyper_ppn`.
   Same hole, same class of silent per-token corruption, and by §2's blast-radius table
   their exposure is confined to caller-co-resident NON-HEAD stages (one-card and
   repeated-device placements). Not fixed blind here for one honest reason: they are
   **per-token hot**, `publish_to` allocates a fresh CUDA event per call, and this rig is
   exactness-only, so I cannot price the guard on the card that would pay for it. The fix
   is three lines per body; it needs a box row, not a bigger diff.
2. **The bodies that publish only `publish_to(n_st-1, …)` carry the residual §3.6 hole**
   (earlier stages' post-`ev_tx` tails). Same argument, same lane.
3. **`publish_to` creates a CUDA event per call.** One cached event per stage, re-recorded,
   would make the guard essentially free and remove the perf objection in items 1-2. Not
   taken here because re-recording a shared event is only safe under a single-host-thread
   discipline that the dual-PP walkers would have to be checked against first — a
   correctness question that belongs in its own diff, not riding a race fix.
4. ~~**`qmatvec_nvfp4_q8_ep_down_slots` is in NO fatbin at sm_89/sm_90a.**~~ **CLOSED BY
   ANOTHER LANE 2026-09-01, and the entry is retired rather than left standing** (exception
   lists and named items rot the moment their reason dies). `release-arch-mirror (89)/(90a)`
   had been red on `main` since at least `79bfd84e1` — `Engine::func` panics `kernel <name>
   not in any fatbin` on those arches, and an engine panic in the GPU worker takes every
   session with it. PR #74 (`lane/nvfp4-quad-symbol-20260831`,
   `0c174def9 fix(ep): restore W4A16 multirow kernel contract`) fixed it in the `.cu`, and
   both arch mirrors go **pass** on this lane's PR. Worth recording HOW: it was fixed in the
   kernel, NOT declared away — `tools/fatbin-lookup-exceptions.txt` carries no entry for it,
   which is the outcome the third remedy would have quietly bought.
5. **`glm5_tparallel_verify_gpu`'s forced-rejection sweep was not run under load.** It is
   the single-device (door-shut) twin, so this defect cannot reach it, but the loaded
   protocol is cheap and would close the argument by construction rather than by reasoning.

---

## 8. STATUS LOG

* **2026-09-01** — battery closed: interleaved A/B FIX 48/48 green on 4 cells (3 with a
  reproducing control, the 4th self-refused as non-evidence), 14/14 standing batteries green,
  `local-ci.sh --perf` run on the final tree. Load window rebuilt once mid-lane after it
  measured 0/96 on a known-racy arm (§6.1) — the harness caught its own vacuity.
* **2026-09-01** — lane opened on `216ffd114`; defect reproduced 3/12 under capped load;
  accept path cleared; cross-stream nature established against the `STREAMS=0` truth;
  round-0 localization; prime nondeterminism found and correlated with the verdict; missing
  dependency bisected to `glm5_verify_rollback`'s exit; fix + arm P0 + FAIL-message repair
  landed; `cargo fmt` clean, `cargo clippy --all-targets` ZERO lints, `tools/check-flags.sh`
  755 runtime literal reads, no uncovered names, no grandfather list.
