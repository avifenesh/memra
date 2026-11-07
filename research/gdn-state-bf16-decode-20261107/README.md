# bf16 GDN decode class: fp32 prime, one rounding at decode start (2026-11-07)

Lever: keep the Gated DeltaNet recurrent state fp32 through every prime call and every capture,
round it to bf16 once when a session starts decoding, and store and step it in bf16 only in decode
and verify. Model: Qwen3.8-27B (NVFP4+Q5_K GGUF, masked sxc32768 MTP draft). Hardware: GB10
(DGX Spark, `121a`) for quality and speed; the RTX 5090 Laptop (`120a`) for the fp32-class identity
and the GPU tests. Branch `research/gdn-state-bf16-decode`, base main `cf85a23ff5`, on top of the
rejected storage lane (`research/gdn-state-bf16-storage`, `f41085a693`, whose receipts sit next
door in `research/gdn-state-bf16-storage-20261107/`).

**Verdict: negative on the quality gate.** Both exactness laws hold and the fp32 class is byte
identical to main on both devices. GB10 decode tick -9.7% at 16 sessions and -9.2% at 32 (form 1's
win), TTFT flat, no measurable change in the 256K memory floor. Long HumanEval 155/164 in both runs
(the same nine failures), against 158/164 for the same-binary fp32 control and a 156-159 band;
GSM8K 97/100 in all three. The default stays fp32 on every device and nothing lands on main
(`docs/decisions/GDN-STATE-BF16.md` on this branch). The served c1..c32 ABAB was not run: the gate
failed first.

## Why this variant

The storage lane stored the state bf16 everywhere and rounded after every token and after every
32-token WY chunk of the prime. It cut the GB10 decode tick 9.7% at 16 sessions but scored 155/164
on long HumanEval (band 156-159, same-binary fp32 158). The rounding-only research arm, which
rounded at stores only, scored 160. The one place the storage lane rounded more than that arm was
the prime: about 4,100 chunk roundings over a 131K-token prompt. This variant removes all of them.
The prime is the fp32 program bit for bit; the decode tick, where the speed came from, keeps bf16.

## Design

- **The class is per model** (`HybridModel::gdn_decode_bf16`, decided at load). `MEMRA_GDN_STATE_BF16=1|0`
  forces it on an eligible plan (every recurrent layer a Qwen3.5-family GDN layer: d_state 128,
  SiLU z-gate, serial residual). Unset: bf16 on GB10 (family `120a` + unified memory) for the measured
  shape only (dense, 48 GDN layers x 48 value heads: Qwen3.8-27B), fp32 everywhere else.
- **Buffers stay fp32.** The resident state and its ping-pong twin are allocated fp32 as on main.
  `RecurLayer::ssm_bf16` says which class the state is in now; a bf16 state occupies the first half
  of the buffer (element i at byte 2*i).
- **One rounding point.** `Engine::gdn_ensure_decode` (every decode and verify site, per layer) rounds
  an fp32 state to bf16 once with `gdn_state_to_bf16` and copies it back, so buffer addresses never
  move (captured graphs and pointer tables stay valid). Later calls are no-ops. Graph paths bring the
  cache to its class eagerly before capture and launch; a class change inside a capture refuses.
- **Decode and verify** run the storage lane's sequential scans with `st16`, which round after every
  token. The prime kernels have no `st16`: the chunked prime is main's kernel again.
- **Prime entry.** `Engine::gdn_ensure_prime` widens a decoded state back to fp32 (exact) before any
  prime kernel reads it (a multi-turn continuation, a prompt token after the tokenwise door).
- **Captures leave fp32.** Prefix entries, the off-tick capture, grid captures and spec-boundary
  publishes take the fp32 widening of a bf16 layer (`RecurLayer::ssm_clone_f32`,
  `CacheSnapshot::widen_f32`); every restore of an fp32 plane marks the layer fp32.
- **Snapshots keep their class.** `CacheSnapshot` copies only the words that hold the state and
  records each layer's class; a rollback restores class and bits. A verify-prefix commit re-runs the
  verify's own program from the state the verify started on (`verify_start_state`).
- **Decode-internal buffers are class-sized**: verify snapshot rings, per-column checkpoints, the
  dspark stash slabs and their byte accounting.
- **Host tier and handoff.** Entries are fp32 in both classes, but a state minted after a bf16 decode
  is one the fp32 class never produces, so a bf16-class model's host-tier numeric identity gains
  `+gdn-decode-bf16` and a process with such a model sets the handoff layout bit.

## Why the two exactness laws hold

- **decode == verify.** Both run the bf16 sequential scan with per-token rounding, from the same
  state: whichever runs first rounds the fp32 prime state once with the same pure function. A T-row
  verify therefore equals T decode steps bit for bit (kernel-check `gdn_st16_scan/_batched/_tsnap/_dc`,
  `gdn_packed_gpu`, run-spec K=1..8).
- **restore == unrestored.** The prime and every capture are fp32, so a prime split at a capture and
  restored equals the unsplit prime exactly as on main; the decode-start rounding is a function of
  that fp32 state, applied on both paths. A state captured after decoding widens exactly and rounds
  back to the same bits (to_bf16 of a bf16-representable value is the identity), so it can leave and
  re-enter a session any number of times (kernel-check `gdn_st16_round_point`,
  `gdn_st16_prime_then_decode`, `gdn_st16_ensure`; `grid_capture_gpu`
  `gpu_model_decode_class_restore_and_resume_are_exact`).

## Memory

Resident buffers are fp32 again, so the storage lane's capacity gain is gone: the resident state is
288 MiB per session (state + twin) in both classes, and a prefix entry's recurrent part is 149.6 MiB.
What stays halved: the state bytes a decode step streams (288 to 144 MiB per session, 9.0 to 4.5 GiB
per tick at 32 sessions), the verify snapshot rings and column checkpoints, the dspark stash slabs and
the per-round spec snapshot copy.

## Provenance

Every measurement here ran binary `dev1`: commit `7456dac2e3` on main `cf85a23ff5`, built `120a` on the
RTX 5090 Laptop (`gates-rtx5090/bins-dev1-120a.sha256`) and `121a` on GB10 (`gb10/bins-dev1-121a.sha256`).
Its device default turned the class on for the 27B shape on GB10; every arm below forced
`MEMRA_GDN_STATE_BF16=0` or `=1`, so the default never decided a measurement. The branch's last code
commit turns the class into an off-by-default research door (no numeric change in either class).
The fp32 identity baselines are main `cf85a23ff5` binaries (`120a` and `121a`).

## Correctness gates

**fp32 class unchanged, byte for byte** (main `cf85a23ff5` vs `dev1`; `forward_last` prefill logits
sha256 and the run-gen numeric lines: prefill/decode and batched-prime argmax + maxdiff, 128-token
greedy stream). The bf16 class keeps the prefill logits, because the prime is fp32 in both classes;
only its decode lines move.

| device | model | main | `dev1` default | `dev1` class 0 | `dev1` class 1 |
|---|---|---|---|---|---|
| RTX 5090 | Qwen3.8-27B | `a297978ec481bed8` / `9970536dec22031c` | same | same | `a297978ec481bed8` / `f60f5b6876d6371b` |
| RTX 5090 | Ornith-1.5-35B-A3B | `9dbe64f55e95174e` / `ea78b25d4a854a72` | same | same | `9dbe64f55e95174e` / `5682f75a58d46d70` |
| GB10 | Qwen3.8-27B | `f444f2cb935605cd` / `7802b4aec0de6e7f` | | same | `f444f2cb935605cd` / `50061107e1828650` |

The 5090 hashes are the ones the storage lane recorded for main `c2e04719b6`. In the bf16 class the
128-token greedy stream first parts from fp32 at token 112 (Qwen3.8-27B, RTX 5090) and token 123
(Ornith, RTX 5090), and not at all on GB10; the prefill-vs-decode and batched-prime-vs-tokenwise
maxdiffs move (5090 27B: 6.477e-1 to 6.754e-1 and 7.637e-1 to 5.646e-1; GB10: 1.620 to 1.210 and
1.407 to 1.330), argmax MATCH everywhere.

**kernel-check.** RTX 5090: ALL GREEN (120 cells, 615 OK lines, 0 FAIL). GB10 synthetic
(`MEMRA_KC_FAST=1`): 365 OK, 0 FAIL. Both run the nine `gdn_st16_*` cells, all OK:

| cell | what it pins |
|---|---|
| `gdn_st16_scan T=1`, `T=3`, `T=7` | decode and T-row verify scans equal the per-token rounding oracle (fp32 kernel one token per launch, host rounding between) |
| `gdn_st16_batched B=3` | the batched decode scan, per-session pointer arrays |
| `gdn_st16_tsnap T=4` | the packed verify twin: output, final state and the 3 snapshot slots |
| `gdn_st16_dc T=1+2` | the device-count twin equals the plain 3-row scan |
| `gdn_st16_round_point T=160 C=32` | the rounding point: the fp32 chunked prime's state rounded by `gdn_state_to_bf16` equals host RNE, the widening is exact, a second rounding changes nothing |
| `gdn_st16_prime_then_decode` | a prime split at a chunk boundary (96 + 64 rows) equals the unsplit prime in fp32, and both round once and decode the same 3 rows to the oracle's bits |
| `gdn_st16_ensure` | `gdn_ensure_decode` rounds once in place (buffer addresses unchanged), a second call is a no-op, `gdn_ensure_prime` widens exactly, the round trip is the identity, an fp32-class model never converts |

**run-spec K=1..8** (greedy, 32 tokens): identical to the plain target 8/8 in both classes, on the
RTX 5090 for Qwen3.8-27B and Ornith and on GB10 for Qwen3.8-27B. Acceptance (equal to the storage
lane's rows: on these short prompts the two bf16 forms run the same program):

| device | model | class | acceptance K=1..8 |
|---|---|---|---|
| RTX 5090, GB10 | Qwen3.8-27B | fp32 | 12/19 15/32 17/42 17/56 17/70 17/84 17/98 17/112 |
| RTX 5090, GB10 | Qwen3.8-27B | bf16 | 12/19 16/32 18/42 18/56 18/70 18/84 18/98 18/112 |
| RTX 5090 | Ornith-1.5-35B-A3B | fp32 | 14/17 16/30 17/45 17/60 17/75 17/90 17/105 17/120 |
| RTX 5090 | Ornith-1.5-35B-A3B | bf16 | 14/17 17/30 17/45 18/56 18/70 18/84 18/98 18/112 |

**Restore round trip and decode == verify** (GPU tests, both classes; RTX 5090 on Qwen3.5-9B, GB10 on
Qwen3.8-27B): `gdn_packed_gpu` 3/3 (the packed verify twin equals the chained per-row launches word for
word: state, outputs, every snapshot; eager and graph-rebound) and `grid_capture_gpu` 6/6 in each class.
The new `gpu_model_decode_class_restore_and_resume_are_exact` primes a session with a capture inside its
prime, rewinds to it, primes the tail and decodes 12 steps: logits and final states equal the cold
session's bit for bit (restore == unrestored through the decode-start rounding); a decoded state
survives a snapshot + rollback in its class and a widen + round trip unchanged.

**CPU suites** (rig, release): memra-kv 84 passed, memra-engine lib 685 passed, memra-server lib 1166
passed, 0 failed (`gates-rtx5090/cputests-rig.log`). clippy `-D warnings` clean on memra-kv,
memra-engine and memra-server (lib, bins, tests).

## GB10: quality panel

The 264-task long-context panel (`qual_panel.py` under `/usr/bin/python3`: one 131,550-token
session, then 164 long HumanEval and 100 long GSM8K tasks on it, 8 concurrent), binary `dev1` (`121a`),
the `gguf-mtpmask` serving arm (GGUF NVFP4+Q5_K trunk, masked sxc32768 MTP draft), fresh boot per run,
kernel-check preflight before each boot (365 OK, 0 FAIL). Receipts: `gb10/run1/quality/`.

| run | class | long HumanEval | long GSM8K | 131,550-token prime TTFT | prime identity |
|---|---|---:|---:|---:|---|
| 1 | bf16 decode class | 155 / 164 | 97 / 100 | 169.0 s | pass |
| control | fp32, same binary | 158 / 164 | 97 / 100 | 169.4 s | pass |
| 2 | bf16 decode class | 155 / 164 | 97 / 100 | 168.7 s | pass |
| storage lane (form 1, `2c89250343`) | bf16 storage, per-chunk prime rounding | 155 / 164 (twice) | 97 / 100 | 183 s | pass |
| storage lane control (`2c89250343`) | fp32 | 158 / 164 | 97 / 100 | 183 s | pass |
| panel band on the fp32 default | fp32 | 156-159 | 96-98 | | |

The panel is deterministic: both decode-class runs failed the same nine HumanEval tasks (run-to-run
spread 0), and this binary's fp32 control failed exactly the six tasks the storage lane's fp32 control
failed (46, 83, 132, 145, 147, 154), so the fp32 class and the panel did not move between the two
binaries. Against the same-binary control the decode class fails three more tasks and gains none:
HumanEval/91 and /120 (assertion failures) and /129 (the answer hit the token cap). Two-sided exact
p = 0.25 on 3 vs 0 discordant tasks. GSM8K agrees task for task.

The two bf16 forms lose the same count with different tasks: form 1 failed 93, 108, 129 and 160 and
passed 154; this form fails 91, 120 and 129. Only 129 is common.

**Verdict: negative on the quality gate.** 155 is below the 156-159 band in both runs and 3 tasks
below the same-binary fp32 control, more than the measured spread of 0. Removing every prime rounding
(about 4,100 per 131K-token prime in form 1) did not recover a single task: the loss sits in the
per-token rounding of decode and verify, which both forms share and which the exactness law
(verify == decode) requires. The rounding-only arm that scored 160 rounded once per verify launch,
not once per token, so in speculative serving it rounded several times less often.

## GB10: decode tick

`decode-batch-bench`, Qwen3.8-27B GGUF, greedy (`MEMRA_DBB_SAMP=greedy`, the serve tick program), 16 steps
x 3 reps per run (median; rep 0 is an untimed warmup, so the decode-start rounding is outside the
timing), same binary, `MEMRA_GDN_STATE_BF16=0/1`, A B B A A B per cell (`gb10/speed/bench/ab.txt`, one
log per run). Every bf16 run is faster than every fp32 run in every cell. The fp32 fingerprints are
the storage lane's fp32 fingerprints in every cell.

| sessions x depth | fp32 ms/tick | bf16 ms/tick | change | fp32 tok/s | bf16 tok/s | storage lane (form 1) change |
|---|---:|---:|---:|---:|---:|---:|
| 1 x 2048 | 72.98 | 72.22 | -1.0% | 13.7 | 13.8 | -1.0% |
| 8 x 2048 | 90.40 | 84.61 | -6.4% | 88.5 | 94.6 | -6.5% |
| 16 x 2048 | 122.45 | 110.60 | -9.7% | 130.7 | 144.7 | -9.7% |
| 32 x 2048 | 246.21 | 223.63 | -9.2% | 130.0 | 143.1 | -9.5% |
| 16 x 12288 | 162.12 | 150.84 | -7.0% | 98.7 | 106.1 | -7.3% |

The decode tick is form 1's: the decode program and its bytes are the same. At one session the bf16
token fingerprint equals fp32's (`2385a50774f8`, 64 greedy tokens per run unchanged), where form 1's
moved (`0714fa16350a`): the prompt states start from the exact fp32 prime here.

## GB10: coherence (greedy T=0 and sampled T=1 seed 1234, 384 tokens, thinking off)

Same binary, fresh boot per class (`gb10/speed/probes/coherence-bf16{0,1}.jsonl`). Identical in both
classes: greedy and sampled bash one-liner, greedy and sampled SQL, greedy Hebrew translation, sampled
train-trip arithmetic (6 of 16). The rest part after 52 to 1,567 characters and stay on topic. The
train-trip arithmetic: fp32 greedy ends at 225 minutes in 295 tokens, bf16 greedy is still converting
to minutes-from-midnight at the 384-token cap. Sampled Hebrew names the wrong weekday in both classes
(fp32 Friday, bf16 Wednesday); greedy Hebrew is correct and identical. These are form 1's coherence
observations word for word.

## GB10: TTFT and memory at long context

`turnbench.py --plan 32768:1 131072:1 258048:1 --new 1024 --reps 1`, fresh boot per run, A B B A,
MemAvailable sampled once a second over each run (`gb10/speed/probes/ttft-r*`). Cold TTFT is the
whole-context prime of a fresh session; the turn is a 1,024-token new turn on that session.

| run | class | 32K cold | 128K cold | 256K cold (258,048) | 256K turn | MemAvailable floor (1 s trace) | server-life floor (guard, 2 s) |
|---|---|---:|---:|---:|---:|---:|---:|
| r1 | fp32 | 32.27 s | 167.21 s | 426.65 s | 3.55 s | 43.68 GiB | 38.84 GiB |
| r2 | bf16 | 32.23 s | 167.19 s | 426.69 s | 3.65 s | 46.41 GiB | 38.96 GiB |
| r3 | bf16 | 32.21 s | 167.28 s | 427.26 s | 3.57 s | 46.42 GiB | 42.35 GiB |
| r4 | fp32 | 32.27 s | 168.15 s | 426.23 s | 3.56 s | 46.11 GiB | 40.12 GiB |

Medians: fp32 32.27 / 167.68 / 426.44 s, bf16 32.22 / 167.24 / 426.98 s (form 1 on its binary:
36.24 / 182.22 / 455.12 s against fp32 35.98 / 181.99 / 455.23 s).

The prime is the fp32 program in both classes, so TTFT is flat by construction; the new turn runs
the bf16 decode. The resident GDN buffers are fp32-sized in both classes, so the 256K floor no longer
rises the way form 1's did (+1.92 GiB): the trace floors are fp32 43.68 and 46.11 GiB, bf16 46.41 and 46.42 GiB. In every run the minimum is the last
sample of the trace (the run ends inside the 256K session's prefix publish, an 8 GB entry), so the
floor moves with that timing: the two fp32 runs differ by 2.43 GiB, more than the class gap, and the
server-life floors (38.84 / 40.12 fp32, 38.96 / 42.35 bf16) show no class order. By construction the
only resident change is the verify rings and column checkpoints of the turn's speculative rounds
(about 0.2 GiB at this depth). The memory result is: no measurable change, where form 1 gained
1.92 GiB.

## Receipts

- `gates-rtx5090/`: fp32 identity (`ident/`, with `pp-logits.sha256` for the dropped logit dumps),
  kernel-check, the GPU tests and run-spec on the RTX 5090; the run stopped at a step boundary at 10:16
  to free the card for another lane's cell evidence and resumed at 10:39 (`rig5090_gates2.sh`; the
  killed bf16 `gdn_packed_gpu` log is kept as `*.killed-at-stop.log`); `cputests-rig.log`.
- `gb10/run1/`: build, gates (identity, kernel-check, run-spec, GPU tests) and the three panels.
- `gb10/speed/`: decode-tick bench (`bench/`), coherence and TTFT/memory (`probes/`).
- `scripts/`: every driver as run.

## What would change the answer

- **A finer 16-bit format with per-token rounding.** fp16 keeps 10 mantissa bits against bf16's 7,
  so each rounding is about 8x smaller at the same bytes per step (the decode tick win would be the
  same). It needs a range receipt first: the largest GDN state magnitude over long sessions against
  fp16's 65504 and its subnormal floor.
- Rounding less often than once per token is not available to an exact engine: plain decode runs one
  token per launch, so a coarser verify rounding makes verify and decode two programs.
