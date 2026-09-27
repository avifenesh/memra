# WP-B day 50: O11's overlap revision, making the exact resume fast where DAY44's E2 failed

OWED.md O11. The owner's direction (2026-09-26): "resume vs rewind - i think its not or or question, but more of we
didnt make it right yet". DAY44 made the resume exact (E1: 0 flips against cold on every `exact` boot on the target
card, against 24 of 60 on `keep`). It is not yet fast everywhere: E2 fails at 8 of 24 cells (DAY44 2.1), all on TTFT.
DAY44 1.4 named the next revision. This day registers it, stage 0 first, before any code.

## 1. Pre-registration

Committed and pushed before any day-50 code and before any day-50 run. Nothing in section 1 changes after a number is
seen; a failed clause is recorded as it reads and a revision is a new, dated addendum pushed before its code.

### 1.1 What the receipts place (DAY44 2.1, the target card, the 27B)

- **M1, zero gap on the plain route:** no settle lands before the next turn, so the resume primes from the in-call
  checkpoint: 64 extra rows at G=32 and 288 at G=256. The prime phase grows by that work: 68.3 to 114.9 ms at 6,144
  G=256, 94.3 to 140.6 at 30,720, 188.7 to 235.4 at 122,880.
- **M2, an arrival waiting for a settle call:** on spec RX at G=256 the resume's own prime phase is faster than `keep`'s
  (84.7 against 149.7 ms at 6,144), but the turn waits for its entry's settle call to land (race case 2). The TTFT excess
  over the prime phase (about 105 ms) matches the settle call's p50 (108 ms). At plain RXg 122,880 G=256 some arrivals
  meet a settle in flight: p50 +25 ms, p95 340 against 196 ms.
- **The cost both pay is a short prime call.** A call of 32 rows costs about 68, 94 and 188 ms at 6,144, 30,720 and
  122,880 context, and each extra row about 0.16 to 0.2 ms (DAY44 2.1's prime phases). The fixed part dominates. What that
  fixed part is spent on (GPU kernels, host launch, syncs) is not yet measured, and it decides the design.

### 1.2 Stage 0 (before the design is chosen): what a short prime call spends its time on

`concat-prime-probe` gains a `callcost` mode: prime `[0, L)` once; then for each R in {32, 64, 288} and N = 5 reps,
take a snapshot at L, prime `[L, L + R)` as one call (the settle and resume shape), time the call's wall around a
device sync, and restore the snapshot. L in {6,144, 30,720, 122,880} on the target card (the 27B) and {6,144, 30,720} on
the 5090 (the 9B). Each (L, R) runs once more under `nsys profile` for the kernel timeline. Readings per (L, R): wall
p50; the GPU-busy time (the union of kernel intervals); the host gaps between kernels; the time in full-attention
kernels, in the GDN scan and in the GEMMs. No clause; stage 0 selects the arm by 1.3's rule.

### 1.3 The arms and the rule that selects them (stated before stage 0)

- **Arm S (short-call overhead).** If the GPU is busy for less than 60% of a 32-row call's wall at any L on the target
  card, the fixed cost is mostly host-side. The revision then removes those gaps from the prime path. That makes every
  short prime faster, `keep`'s resume included, so E2's ratio moves only by the share the gaps took of the extra work.
  Stage 0 decides this first because it is the cheaper change and helps every route.
- **Arm O (overlap on a second stream).** If the GPU is busy for at least 60% of the call, or E2 still fails after arm S,
  settles run on a second compute stream and never block the worker:
  - A settle starts at park time on the side stream. An arrival for another entry runs at once on the main stream,
    which removes M2 for other entries.
  - An arrival for the entry being settled resumes from the checkpoint at `g` on the main stream and drops the
    settle's result, so it never waits (that turn pays M1's cost, bounded by one call).
  - For zero-gap turns (M1), the settle of a turn's own decoded rows starts once the decode crosses the last grid
    point that `max_tokens` allows, so it overlaps the decode's tail.
  - Precondition, as its own census: the engine's shared prime scratch (the prime slabs, the CUTLASS NVFP4 scratch,
    the q8_1 activation scratch, the GDN scan buffers, the cuBLASLt handle) is safe today only because all compute
    serializes on one stream. The side stream gets its own set, and a census test names every buffer the prime path
    touches.
- **Weighed and not chosen:** settling in segments during decode on the same stream puts G / 32 short calls on the
  decode's critical path (about 8 x 68 ms at 6,144 G=256 against a 3,154 ms E2E, +17%). Settling at the turn's end before
  the final frame adds one call to E2E (+15% at 6,144 G=32). Both fail E2's E2E half. A mixed forward that carries
  prime-program rows beside decode rows does not exist in the engine.

### 1.4 Clauses (DAY44's, unchanged, plus E7)

DAY44 1.7's E1 to E6 on DAY44 1.6's cells, both cards, both orders, plus:

- **E7 decode.** On every `exact` boot, the resumed turns' TPOT p50 is at most 1.02 x `keep`'s in the same cell (arm O's
  side stream must not slow the decode it overlaps).

Readings: the settle calls' wall and count, the side stream's busy share (arm O), idle driver-free and pool-cached bytes.

### 1.5 What the reading decides

Nothing moves a default. If E1 to E7 pass on a card class, the exact resume is PROMOTE-ELIGIBLE there and the grid
rewind door's deletion is owed under door hygiene. `MEMRA_RESUME_EXACT` keeps its decide-by (2026-10-10), and the
owner decides at that date.

### 1.6 Price

Stage 0: about 0.3 agent-day (the probe mode and its reader) plus about 1 h on each card. Arm S: about 1 agent-day plus
the E1 to E7 cells (about 12 h on the target card, 5 h on the 5090). Arm O: about 3 to 4 agent-days (the side stream,
its scratch set and census, the settle launcher and the arrival rule, GPU tests of bit-identity against the one-stream
settle) plus the same cells.

### 1.7 Addendum A (2026-09-27, stage 0 as built, before any stage-0 run)

No clause, rule or reading changes.

- **The probe:** `concat-prime-probe callcost` primes `[0, L)` once, takes a snapshot, and per R in {32, 64, 288} runs
  one untimed warm-up and N = 5 timed calls. Before every restore and every call it idles 50 ms, so the trace separates
  setup, restore, call, restore, call. Each timed call is `prime_cache([L, L + R))` between two stream synchronizes,
  with `queued_after = 0`. The trunk prime only: a spec settle's draft fill (`spec_prime_settle`) is not in this shape,
  and that is stated wherever the reading is used for the spec route.
- **The runs (`day50-stage0.sh`):** per L, a wall run and then the same run under `nsys profile -t cuda`, each under
  the card's lock. `day50-trace.py` splits the trace into clusters on 20 ms idle gaps, checks the count against the
  expected shape (and prints that the split was not read when it is off), and reads each timed call. It reports the
  GPU span, the GPU-busy union, the gaps inside the span, the host time outside it, and the busy time by kernel class
  (full attention, GDN/conv, GEMM, other). `busy_share` is busy over the nsys run's own wall. The wall run gives the
  wall without the profiler.
- **Where:** the 5090 (`rtx5090-day50/run.sh`, the 9B, L 6,144 and 30,720) and the target card (the sixteenth sitting,
  `pro-single-b-sitting16.sh`, the 27B, L 6,144, 30,720 and 122,880; it refuses to start without `nsys`).
- **CPU checks:** the probe builds and `cargo clippy -p memra-engine --all-targets -- -D warnings` is clean. The trace
  reader was run on a synthetic trace.

### 1.8 Addendum B (2026-09-27, the trace reader's split, after its first read printed "not read")

On the target card's traces `day50-trace.py` printed `clusters=38 expected=37 (cluster count off: the per-call split
is not read)` at every L. The extra cluster is the weight upload before the setup prime (850 `[CUDA memcpy
Host-to-Device]` events, 3.45 s at 6,144), which 1.2's shape did not count. The reader now takes the calls as the LAST
2 x len(rows) x (reps + 1) clusters. It checks that each pair's first cluster holds only memops (the restore) and its
second holds kernels, and prints `restore_call_shape=off` (not read) when either check fails. The count and
content checks set the split, not any time value. The re-read below ran on the fixed reader before this addendum was
pushed, and it is recorded that way. No clause, rule or reading changes.

## 2. Results

Written after the runs. Section 1 is unchanged.

### 2.1 Stage 0 on the target card (the sixteenth sitting, one RTX PRO 6000 Blackwell Workstation Edition at 600 W, 2026-09-27 02:44 to 02:48Z)

Chain tree `1df7e7852`, the probe built on the box from `5d94e26ae` (sha256 `5b5918a3...`, `probe.sha256`), the 27B,
nsys 2025.4.1. Receipts at `pro-single-day50/box/`: the trace CSVs gzipped, with their raw sha256 in
`stage0/trace-csv-raw.sha256`; the six nsys files are outside git, by hash in `LEAD-EXCLUDED.sha256`. The wall runs
(no profiler), verbatim:

```
callcost L=6144 R=32 N=5 wall_ms p50=68.93 min=68.91 max=68.98 all=[68.93,68.98,68.91,68.92,68.97]
callcost L=6144 R=64 N=5 wall_ms p50=70.20 min=70.15 max=70.22 all=[70.15,70.20,70.21,70.22,70.18]
callcost L=6144 R=288 N=5 wall_ms p50=112.41 min=112.37 max=112.45 all=[112.41,112.37,112.44,112.38,112.45]
callcost L=30720 R=32 N=5 wall_ms p50=94.26 min=94.25 max=94.29 all=[94.29,94.25,94.26,94.28,94.25]
callcost L=30720 R=64 N=5 wall_ms p50=95.70 min=95.61 max=95.73 all=[95.73,95.70,95.65,95.61,95.72]
callcost L=30720 R=288 N=5 wall_ms p50=137.90 min=137.88 max=138.03 all=[137.88,138.03,137.90,137.90,138.00]
callcost L=122880 R=32 N=5 wall_ms p50=189.14 min=189.08 max=189.20 all=[189.20,189.14,189.17,189.11,189.08]
callcost L=122880 R=64 N=5 wall_ms p50=190.21 min=190.14 max=190.44 all=[190.14,190.44,190.20,190.25,190.21]
callcost L=122880 R=288 N=5 wall_ms p50=233.66 min=233.60 max=233.79 all=[233.60,233.60,233.66,233.79,233.67]
```

The traces, read under addendum B (every L: `clusters=38 leading=2 pairs=18 restore_call_shape=ok`):

```
DAY50 S0 card=pro6000 L=6144 R=32 N=5 wall_ms p50=69.39 gpu_span_ms p50=69.22 gpu_busy_ms p50=65.82 busy_share=0.949 in_span_gaps_ms p50=3.41 host_outside_span_ms p50=0.17 attn_ms=0.27 gdn_ms=1.56 gemm_ms=55.89 other_ms=8.10
DAY50 S0 card=pro6000 L=6144 R=64 N=5 wall_ms p50=71.18 gpu_span_ms p50=71.02 gpu_busy_ms p50=66.95 busy_share=0.941 in_span_gaps_ms p50=4.07 host_outside_span_ms p50=0.16 attn_ms=0.28 gdn_ms=1.85 gemm_ms=56.29 other_ms=8.53
DAY50 S0 card=pro6000 L=6144 R=288 N=5 wall_ms p50=112.97 gpu_span_ms p50=112.79 gpu_busy_ms p50=108.90 busy_share=0.964 in_span_gaps_ms p50=3.89 host_outside_span_ms p50=0.18 attn_ms=0.79 gdn_ms=4.12 gemm_ms=91.92 other_ms=12.06
DAY50 S0 card=pro6000 L=30720 R=32 N=5 wall_ms p50=94.96 gpu_span_ms p50=94.80 gpu_busy_ms p50=91.42 busy_share=0.963 in_span_gaps_ms p50=3.39 host_outside_span_ms p50=0.16 attn_ms=0.27 gdn_ms=1.56 gemm_ms=56.20 other_ms=33.38
DAY50 S0 card=pro6000 L=30720 R=64 N=5 wall_ms p50=96.84 gpu_span_ms p50=96.67 gpu_busy_ms p50=92.61 busy_share=0.956 in_span_gaps_ms p50=4.07 host_outside_span_ms p50=0.17 attn_ms=0.28 gdn_ms=1.86 gemm_ms=56.61 other_ms=33.86
DAY50 S0 card=pro6000 L=30720 R=288 N=5 wall_ms p50=138.77 gpu_span_ms p50=138.60 gpu_busy_ms p50=134.73 busy_share=0.971 in_span_gaps_ms p50=3.87 host_outside_span_ms p50=0.17 attn_ms=0.79 gdn_ms=4.13 gemm_ms=92.27 other_ms=37.54
DAY50 S0 card=pro6000 L=122880 R=32 N=5 wall_ms p50=190.67 gpu_span_ms p50=190.51 gpu_busy_ms p50=186.87 busy_share=0.980 in_span_gaps_ms p50=3.63 host_outside_span_ms p50=0.16 attn_ms=0.28 gdn_ms=1.58 gemm_ms=56.70 other_ms=128.32
DAY50 S0 card=pro6000 L=122880 R=64 N=5 wall_ms p50=192.29 gpu_span_ms p50=192.14 gpu_busy_ms p50=188.04 busy_share=0.978 in_span_gaps_ms p50=4.09 host_outside_span_ms p50=0.15 attn_ms=0.28 gdn_ms=1.87 gemm_ms=57.11 other_ms=128.79
DAY50 S0 card=pro6000 L=122880 R=288 N=5 wall_ms p50=235.86 gpu_span_ms p50=235.68 gpu_busy_ms p50=231.78 busy_share=0.983 in_span_gaps_ms p50=3.90 host_outside_span_ms p50=0.18 attn_ms=0.81 gdn_ms=4.17 gemm_ms=93.56 other_ms=133.25
```

- **The rule of 1.3 selects arm O.** The 32-row call is GPU-busy for 94.9%, 96.3% and 98.0% of its wall at the three
  L, far above arm S's 60% line. Host time outside the GPU span is 0.16 to 0.18 ms, and gaps inside it are 3.4 to 4.1 ms.
  The fixed cost is GPU work, not host overhead.
- **What the GPU time is (readings, not a rule input).** Two kernel families make up the short call:
  - The NVFP4 GEMM `mul_mat_q_nvfp4_w4a8<128, 128, ...>` costs 55.4 to 57.1 ms for 32 and for 64 rows alike: the
    128-row tile makes a 32-row call cost what a 128-row call does. At 288 rows it costs 92 to 94 ms.
  - The full-attention prefill `fa_prefill_qw_db` plus `fa_dequant_kv_ws_bf16` grows with the context: about 8 ms at
    6,144, 33 at 30,720, and 113 + 14 ms at 122,880 for 32 rows (16 layers). The reader's `attn` class missed these
    names; they sit in `other`.
  - The GDN scan is 1.6 to 4.2 ms.
- **What this means for the design.** Arm O's side stream has to overlap a call that is GPU-bound. Whether a
  small-M prime kernel can keep the cold prime's exact numbers (the same K-reduction order per output) is a separate
  improvement. It would shorten `keep`'s resume too, so it does not change E2's ratio, and it is recorded as a
  candidate, not an arm of this day. The 5090 half (`rtx5090-day50/`) waits for the card.

### 2.2 Stage 0 on the 5090 (the 9B, `rtx5090-day50/`, 2026-09-27 to 03:31Z)

The probe `5d94e26ae`'s mode, built in the lane checkout at `3c6d98784` (sha256 `d51c26e6...`). The run's own
`read-L*.log` came from the reader before addendum B, and at both L it printed "not read". The re-read under addendum B
is `stage0/reread-L*.log` (at 30,720 the setup prime of 30,720 tokens leaves 38 leading clusters; the pair shape
checks ok at both L). The trace CSVs are gzipped with their raw sha256, and the four nsys files are outside git, by hash
in `EXCLUDED.sha256`. Verbatim:

```
DAY50 S0 card=rtx5090 L=6144 R=32 N=5 wall_ms p50=37.96 gpu_span_ms p50=37.66 gpu_busy_ms p50=37.27 busy_share=0.982 in_span_gaps_ms p50=0.39 host_outside_span_ms p50=0.30 attn_ms=0.15 gdn_ms=0.87 gemm_ms=22.91 other_ms=13.37
DAY50 S0 card=rtx5090 L=6144 R=64 N=5 wall_ms p50=37.98 gpu_span_ms p50=37.72 gpu_busy_ms p50=37.35 busy_share=0.983 in_span_gaps_ms p50=0.38 host_outside_span_ms p50=0.26 attn_ms=0.16 gdn_ms=1.06 gemm_ms=22.10 other_ms=14.02
DAY50 S0 card=rtx5090 L=6144 R=288 N=5 wall_ms p50=88.66 gpu_span_ms p50=88.40 gpu_busy_ms p50=87.98 busy_share=0.992 in_span_gaps_ms p50=0.42 host_outside_span_ms p50=0.26 attn_ms=0.68 gdn_ms=3.36 gemm_ms=53.84 other_ms=30.10
DAY50 S0 card=rtx5090 L=30720 R=32 N=5 wall_ms p50=55.69 gpu_span_ms p50=55.46 gpu_busy_ms p50=55.09 busy_share=0.989 in_span_gaps_ms p50=0.36 host_outside_span_ms p50=0.23 attn_ms=0.15 gdn_ms=0.87 gemm_ms=21.72 other_ms=32.17
DAY50 S0 card=rtx5090 L=30720 R=64 N=5 wall_ms p50=56.68 gpu_span_ms p50=56.48 gpu_busy_ms p50=56.07 busy_share=0.989 in_span_gaps_ms p50=0.41 host_outside_span_ms p50=0.20 attn_ms=0.16 gdn_ms=1.06 gemm_ms=21.88 other_ms=32.74
DAY50 S0 card=rtx5090 L=30720 R=288 N=5 wall_ms p50=103.69 gpu_span_ms p50=103.45 gpu_busy_ms p50=103.07 busy_share=0.994 in_span_gaps_ms p50=0.37 host_outside_span_ms p50=0.24 attn_ms=0.64 gdn_ms=3.24 gemm_ms=49.77 other_ms=49.55
```

- **The rule of 1.3 selects arm O on the 5090 class too.** The 32-row call is GPU-busy for 98.2% and 98.9% of its
  wall. The shape matches the target card's: the GEMM costs the same 22 ms for 32 and 64 rows, and the prefill
  attention grows with the context (in `other`).
- Timings stay on this card and are not compared with the target card's.
