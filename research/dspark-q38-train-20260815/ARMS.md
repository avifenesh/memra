# B200 block arm matrix (owner allocation: 3 arms x 2 cards + 2 cards memra recon)

Box: `b200-1` (8x B200 183GB, N-Virginia-c, block <capacity-block>,
window ends 2026-08-16T11:30Z). Phase 1 (regen) borrows all 8 cards; arms start after.

| Arm | Cards | Init | Geometry | Corpus | Bet it tests |
|---|---|---|---|---|---|
| A own-cold | 0-1 | cold | RadixArk (block 7, taps [4,16,28,40,52], MLP 10240, 40q/8kv, markov+conf) | own think+nothink exploded + pb-30k mix | corpus-match closes the 2.6 -> 3.4+ gap |
| B pb-control | 2-3 | cold | same as A | pb-30k both modes only | isolates corpus effect (A vs B = corpus delta at fixed everything) |
| C warm-3.6 | 4-5 | `model.draft_checkpoint_path` = z-lab/Qwen3.6-27B-DFlash (MIT, clean) | z-lab (block 16, taps [1,16,31,46,61], MLP 17408, 32q/8kv; heads fresh) | own mix | cross-target warm-start collapses training cost 10x |
| memra recon | 6-7 | — | — | — | sm_100 build + untimed correctness on v0.83.0 |

Topology per arm: 1 capture server (FP8 target, patched sglang 0.5.14, `extra_buffer`) +
1 FSDP trainer. Shared: temp-0 target-regenerated labels, `data.train_only_last_turn: true`,
chat template qwen3.5-class, loss CE 0.1/L1 0.9/conf 1.0, decay gamma 4.0, lr 6e-4 27B recipe.
Kill gates per arm (frozen before results): heldout accept trend must beat RadixArk's own-corpus
2.607 within its compute window or the arm stops; STS/ECE checks ride the eval stage, not training.

License chain for the deliverable: SpecForge MIT harness, target apache-2.0, labels = target's own
outputs, arm C init MIT. Every arm's export is commercially servable.

## step-125 interim evals (2026-08-15 ~12:4xZ, B200 cards 6-7, co-tenant caveat)

First exported checkpoints, eval plumbing green end-to-end (export -> normalize -> DSPARK serve ->
serving-gate -> own-sessions + gsm8k). Numbers are EARLY-TRAINING readings, not verdicts:

| arm | own chat-short | own agentic | gsm8k accept | gsm8k tok/s |
|---|---|---|---|---|
| arm-a own-cold @125 | 1.130 | 1.138 | 1.307 | 538.8 |
| arm-b pb-control @125 | 1.103 | 1.092 | 1.283 | 532.4 |

RadixArk control bar: own-sessions 2.607 (think), gsm8k 4.57. Direction note (noise-level, n=64):
arm-a >= arm-b on all three metrics at equal step — the own-corpus signal G3 predicts, visible but
not yet callable. arm-c (warm36) trains at ~half pace; step-50 trainer acc 0.358 vs arm-a 0.087
@100 (warm init pays early, as designed). Next read at later checkpoints via ckptwatch auto-fire.

## step-250 evals (arm-b landed 2026-08-15 ~17:1xZ; arm-a queued; arm-c step-125 relaunched)

| arm | own chat-short | own agentic | gsm8k accept | gsm8k tok/s |
|---|---|---|---|---|
| arm-b pb-control @250 | 1.253 | 1.212 | 1.696 | 653.2 |
| (arm-b @125, for slope) | 1.103 | 1.092 | 1.283 | 532.4 |

Trainer acc at equal step keeps arm-a ahead of arm-b (0.210 vs ~0.19 @200) and arm-c far ahead
of both (0.436 @100 — warm z-lab init). Ops: ckpt-watch fires on first checkpoint DIRECTORY
appearance and raced arm-c's mid-write save — export failed, serve-health loop hung; killed and
relaunched with the arm-a@250 eval queued behind it on GPU 6. Later-checkpoint evals are manual
by design (ckpt-watch is fire-once); keep archiving receipts per step
(/scratch/receipts/eval/<arm>-step125/ pattern) before refiring.

## arm-c warm36 @125 (DFLASH serve mode, 2026-08-15 ~18:5xZ) — TRAIN/SERVE INVERSION

own chat-short 1.046 / agentic 1.045, gsm8k accept 1.051 at 341.7 tok/s (spec overhead makes it
SLOWER than arm-b's 653 — rejected 16-blocks are pure waste). Yet arm-c's trainer acc (0.436
@100) is the best of all three arms. Inversion hypotheses, unresolved: (a) DFLASH serving
integration mis-drives the block-16 drafter (first eval ever through this path — serve fix
was fresh: projector relabel + --speculative-algorithm DFLASH + --speculative-dflash-block-size);
(b) warm-3.6-init predicts teacher-forced blocks well but collapses autoregressively on 3.8.
Control worth one cell: re-serve the SAME export with a shorter verify window (dflash-block-size
8) — a pure serve-knob change; if accept jumps, it is (a). Numbers recorded as measured.

## step-250 full table + arm-c window control (2026-08-15 ~19:5xZ)

| arm @step | own chat-short | own agentic | gsm8k accept | gsm8k tok/s |
|---|---|---|---|---|
| arm-a own-cold @250 | **1.307** | **1.370** | **1.721** | 654.4 |
| arm-b pb-control @250 | 1.253 | 1.212 | 1.696 | 653.2 |
| arm-c warm36 @125, blk16 | 1.046 | 1.045 | 1.051 | 341.7 |
| arm-c warm36 @125, blk8 control | 1.044 | 1.042 | — | — |

arm-a's own-corpus lead over pb-control WIDENS with training: +4.3% chat-short, +13.0% agentic
at equal step (G3 signal strengthening; agentic per-session max hit 3.20). arm-c: the blk8
verify-window control is FLAT -> not a window-size serve knob; remaining hypotheses are warm-3.6
transfer failure at autoregressive serving vs a deeper DFLASH tap/serve mismatch. arm-c keeps
training (its own step-250 eval will re-read); it is not blocking the a-vs-b decision lane.

## RUN 2 — restart after the 22:36Z stall (2026-08-15 ~23:3xZ)

All three trainer pipelines died simultaneously (process-group kill, logs cut mid-write; cause
not conclusively pinned — the one un-pinned GPU toucher, specforge export in the eval phase, is
now device-pinned as a precaution). SpecForge managed_local REFUSES training.resume_from
(schema.py: "managed_local does not support resume"), so a plain relaunch silently retrains from
step 0 — caught before any run-1 checkpoint was overwritten.

Recovery: run-1 checkpoints preserved at /scratch/ckpt-preserved/<arm>-run1; arms relaunched with
model.draft_checkpoint_path = the step-250 (a, b) / step-125 (c) HF exports — WEIGHTS-ONLY warm
start. Consequences, recorded for honesty: optimizer state and LR schedule RESET at the seam;
run-2 step numbering restarts at 0 (run-2 step N ~ run-1 step 250+N for a/b, 125+N for c); the
data stream re-shuffles. Cross-run curves are therefore two segments, not one continuous run.
Both a and b get the identical treatment, so the a-vs-b comparison stays internally valid within
each segment. Watcher: ckpt-watch-v2 reset to run-2 numbering, receipts archived per run+step.

### Timeline correction (17:45Z, after clock reconciliation)

The run-1 death was at 16:36:47Z (all three pipelines, process-group kill, cause still unpinned —
correlates in time with the arm-c eval relaunch cycle; export step now GPU-pinned). Detected
~17:05Z, run-2 warm-relaunched 17:15Z: actual downtime ~40 MINUTES, not hours — the earlier
"stalled for 6h" line in this file was a clock-drift misread on my side and is retracted. Run-1
progress lost at the seam: steps 250->~305 (a/b, ~1.5h) and 125->~150 (c). Block ends
2026-08-16T11:30Z; run-2 has ~17.5h — projected run-2 step ~600+ for a/b on top of the warm
step-250 weights.

## RUN 3 — owner speed call: arm-c stopped, survivors get tp2 capture (2026-08-15 ~18:40Z)

Owner asked for pace. Attempted dual capture servers per arm first: capture side healthy but the
trainer's mooncake fetch from the second segment fails (get_into status -707) — multi-segment
consumer transfer is broken in this stack; timeboxed and abandoned. Working layout instead: ONE
capture server per arm at tp_size=2 — arm-a capture on cards 0+4, trainer 1; arm-b capture on
2+5, trainer 3; single mooncake segment per arm (proven path). arm-c stopped (run-1 checkpoints
at /scratch/ckpt-preserved/, export retained). Both arms warm-started again from the run-1
step-250 exports (run-2's ~45 min had no checkpoint to carry). Pace measurement at step-50 eval.

### Run-3 warm-start proof + pace (20:01Z)

step-50 trainer acc: arm-a 0.304 (== run-1 step-300's 0.301), arm-b 0.281 (== run-1 step-300's
0.276) — the weights-only warm start preserved full training state quality; the seam cost is
optimizer/LR-schedule reset only. Pace: ~45 steps/h (was 36) -> tp2 capture bought ~+22%;
trainer-bound now. DP2 trainer (~1.8x more) offered to owner — doubles effective batch, science
change, awaiting explicit call. Projection: ~cumulative step ~1000/arm by block end, eval per
125-step checkpoint.

## cumulative-375 evals (run-3 step 125; 22:30Z) — acceleration post-seam

| arm | own chat | own agentic | gsm8k | tok/s |
|---|---|---|---|---|
| arm-a own @cum375 | **1.605** | **1.666** | **2.243** | 798.8 |
| arm-b pb @cum375 | 1.411 | 1.355 | 2.133 | 783.4 |
| (arm-a @cum250) | 1.307 | 1.370 | 1.721 | 654 |
| (arm-b @cum250) | 1.253 | 1.212 | 1.696 | 653 |

Own-corpus lead widens again (+14% chat, +23% agentic at equal step). Slope steepened after the
warm seam (plausibly the fresh LR warmup on good weights). Revised projection: own-sessions 2.6
bar reachable ~cum 600-750; gsm8k 3.5-4+ by block end. The earlier "needs 3-5k steps" estimate
was too pessimistic — revise to ~1.5-2k cumulative for parity-class accept on this corpus.

## cumulative-500 evals (run-3 step 250; 01:35Z) — bar within reach

| arm | own chat | own agentic | gsm8k | gsm8k tok/s | trainer acc |
|---|---|---|---|---|---|
| arm-a own | **2.065** | **2.197** | **3.372** | 1121 | 0.518 |
| arm-b pb | 1.704 | 1.579 | 3.173 | — | 0.486 |

Own-corpus lead: +21% chat, +39% agentic. vs RadixArk control (2.607 own-think / 4.57 gsm8k):
arm-a at 84% / 74% of the bar and still climbing steeply. Projection: bar-crossing around
cum 750-900 — before block end. The G3 own-training bet is paying: at equal steps the own-mix
drafter beats the PerfectBlend control BY MORE on own traffic than the two differ on gsm8k.

## cumulative-625 evals (04:25Z) — FIRST BAR CROSSED

| arm | own chat | own agentic | gsm8k | gsm8k tok/s | trainer acc @350 |
|---|---|---|---|---|---|
| arm-a own | 2.453 | **2.788** | 4.067 | 1297 | 0.576 |
| arm-b pb | 1.991 | 1.904 | 4.080 | — | 0.564 |

arm-a agentic-brief EXCEEDS the RadixArk own-think bar (2.788 > 2.607); chat-short at 94% of it.
gsm8k: both arms ~4.07 (89% of the 4.57 bar) — TIED between arms while own-traffic separation is
+23%/+46%. This is G3's prediction realized: own-corpus training buys own-traffic acceptance at
zero measured general-quality cost. Next checkpoints: cum-750 (~07:00Z fire), cum-875 (~09:45Z,
last before block end).

## cumulative-750 evals (07:40Z) — OWN-SESSIONS BAR CROSSED, BOTH CATEGORIES

| arm | own chat | own agentic | gsm8k | gsm8k tok/s | trainer acc @500 |
|---|---|---|---|---|---|
| arm-a own | **2.652** | **2.955** | 4.366 | 1394 | 0.643 |
| arm-b pb | 2.203 | 2.192 | 4.314 | — | 0.629 |

arm-a exceeds the RadixArk own-think bar (2.607) on BOTH own-traffic categories while arm-b does
on neither — at gsm8k the arms remain within 1% of each other (96%/94% of that bar). G3 verdict
callable: OWN-CORPUS TRAINING WINS for our serving traffic, at zero general-quality cost, with a
clean-license drafter. gsm8k bar expected to cross at cum-875 (last in-block checkpoint).

## BLOCK END (box terminated ~09:05Z 2026-08-16 — earlier than the 11:30Z expectation)

The capacity block reclaimed the box between the cum-750 evals and the cum-875 checkpoint. No
response on TCP/ICMP; hyperscaler API unavailable (credentials invalid since ~2026-08-15 15:00Z — owner
action needed there regardless).

SAVED LOCALLY (/home/avifenesh/models/dspark-q38-20260815/):
- run-3 trainer checkpoints (with optimizer state) through step 375 = cumulative 625, both arms
  (ckpt-run3/) — the "agentic bar crossed" level; resumable/exportable offline via SpecForge.
- run-1 checkpoints complete (ckpt-run1/, cum-250 level).
- HF exports at cum-250 (exports/ — the warm-start weights; later eval-time exports did not sync
  before termination).
- Every receipt, eval log, and result table through cum-750 (in-repo, raw/b200/).

LOST WITH THE BOX: the cum-750 weight state (evidence of its quality is banked; the weights are
~125 steps of retraining from the saved cum-625 checkpoint on any B200/H100-class box), and the
never-reached cum-875 checkpoint.

FINAL SCOREBOARD (all receipts in-repo):
| cumulative step | arm-a own chat / agentic / gsm8k | arm-b own chat / agentic / gsm8k |
|---|---|---|
| 250 | 1.31 / 1.37 / 1.72 | 1.25 / 1.21 / 1.70 |
| 375 | 1.61 / 1.67 / 2.24 | 1.41 / 1.36 / 2.13 |
| 500 | 2.07 / 2.20 / 3.37 | 1.70 / 1.58 / 3.17 |
| 625 | 2.45 / 2.79 / 4.07 | 1.99 / 1.90 / 4.08 |
| **750** | **2.65 / 2.96 / 4.37** | 2.20 / 2.19 / 4.31 |
| RadixArk bar | 2.607 / 2.607 / 4.57 | — |

G3 VERDICT: own-corpus training WINS. At cum-750 the own-mix drafter exceeds the RadixArk
control on own traffic in both categories (+20%/+35% over the PerfectBlend control at equal
step) while remaining tied with the control arm on gsm8k. Clean-license (own-trained, SpecForge
MIT, own data) — servable to OpenRouter, which the RadixArk checkpoint is not. gsm8k trend
(1.72 -> 4.37, near-linear in step) indicates the 4.57 bar falls within ~125 further steps.

### Artifact inventory CORRECTION (post-audit, 2026-08-16 ~09:5xZ)

The block-end note above overstated the saved trainer states ("through cum-625") — the 25-min
pull cadence lost the race to the last checkpoints. Audited reality:

| arm | complete TRAINER states (resumable w/ optimizer) | complete WEIGHTS (HF export, verified headers) |
|---|---|---|
| arm-a | run-3 step-125 (= cum-375), run-1 step-125/250 | cum-250 export (2.72 GB, 62 tensors) |
| arm-b | run-1 step-125 only (run-1 s250 + all run-3 partial/empty) | cum-250 export (2.72 GB) |
| arm-c | none (run-1 pull incomplete) | cum-125 export (3.46 GB, 58 tensors) |

Recovery cost to reproduce the bar-crossing (cum-750) quality: warm-start from the cum-250
exports — the mechanism proven lossless at the run-3 seam — and retrain ~500 steps (~5h of
B200-class tp2 time; the entire eval trail predicts the trajectory). All eval evidence and
receipts through cum-750 are intact in-repo. Lesson recorded: artifact pulls must be
CHECKPOINT-TRIGGERED (fire on save, verify sizes), not fixed-cadence, and completeness must be
audited BEFORE the final report — the block-end summary repeated the pull loop's silent-cap
mistake this file's own evidence rules warn about.
