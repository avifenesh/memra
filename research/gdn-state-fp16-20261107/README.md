# fp16 GDN recurrent state on GB10: magnitude census, then the fp16 decode class (2026-11-07)

Model: Qwen3.8-27B (GGUF NVFP4+Q5_K trunk, masked sxc32768 MTP draft, the `gguf-mtpmask` serving
arm). Hardware: GB10 (DGX Spark, `121a`). Branch `research/gdn-state-fp16`, base main
`05a8a63503`. Follows the two rejected bf16 forms (`research/gdn-state-bf16-storage`, `f41085a693`;
`research/gdn-state-bf16-decode`, `7826562451`; `docs/decisions/GDN-STATE-BF16.md` there): both
scored long HumanEval 155/164 against a same-binary fp32 158 (band 156-159), and removing every
prime rounding recovered nothing, so the loss sits in the per-token rounding of decode and verify.
fp16 has the same bytes and a 10-bit mantissa against bf16's 7, so each rounding is about 8x finer,
but its range is small: 65504 at the top, subnormals (multiples of 2^-24) below 2^-14 = 6.1e-5,
and zero at or below 2^-25 = 3.0e-8. Step 1 asks whether the state fits that range; step 2 runs the
fp16 decode class only if it does.

## Step 1: magnitude census

**Verdict: fp16 holds the state.** No element of any sample comes within 600x of fp16's largest
value, and the subnormal range costs precision only in the smallest heads, where one fp16 rounding
is still at least 5.2x finer than one bf16 rounding.

### Instrument

`MEMRA_GDN_STATE_CENSUS=<dir>` (memra-server, `crates/memra-server/src/gdn_census.rs`, commit
`a8db6a5dc3`; a diagnostic, the numeric program is unchanged). Between worker ticks it copies each
live session's GDN states (`RecurLayer::ssm_state`: 48 layers x 48 value heads x 128 x 128 fp32) to
the host and records per layer and per head:

- the largest |x|, against 65504, and elements that would overflow;
- the fraction in fp16's subnormal range (0 < |x| < 2^-14), the fraction that would flush to zero,
  exact zeros, non-finite values;
- the share of the head's energy (sum of x^2) carried by the subnormal-range elements;
- the relative L2 error of one fp16 rounding and of one bf16 rounding of the head
  (`||round(S) - S|| / ||S||`), which folds the mantissa and the subnormal loss into one number;
- the head RMS, and a log2 |x| histogram per layer.

Per head because the GDN output goes through a per-head gated RMSNorm: a head with a small state is
scaled back up, so its relative error reaches the output whatever its absolute size.

Cadence: a session is sampled when first seen, every 10 s while it primes (the cooperative prime
walker yields between chunks, so mid-prime states are caught), on its first decode tick, and every
128 committed rows of decode. A sample costs about 24 ms (3 ms copy, 21 ms statistics on 16 host
threads; the first one 1 s). The host rounding functions agree with the device's
`__float2half_rn` and `__float2bfloat16_rn` on 49.8M random finite fp32 values and on every finite
fp16 value (`scripts/xcheck.rs` run on the rig, plus the module's own tests).

The decision rule was fixed before the run: fp16 holds if no sample overflows and the energy below
fp16's normal range is negligible, which shows as every head's fp16 rounding error staying near
the normal-range value. A per-layer or per-head scale is considered only if a few heads leave the
range.

### Run

Binary `arm-2d6ce42cf5` (`121a`: the census commit plus the fp16 decode class, served in the fp32
class with `MEMRA_GDN_STATE_FP16=0`), one fresh boot of the 264-task long-context panel
(`qual_panel.py`): the 131,550-token prime, its 64-token answer, then the long HumanEval tasks,
each a prefix hit on the 131K entry with its own tail prime and decode, 8 concurrent. The boot was
stopped after 78 of the 164 HumanEval tasks to free the unit for another lane's landing record; the
census counts only states, so the partial panel still covers the long prime, the decode after it
and 90 task sessions at 131.6K-132.3K rows. Receipts: `census/panel/` (`census.jsonl`, one row
per sample; `agg.json`, per layer and head extremes; `summary.md`; `server.log`; the partial
panel's records under `panel-partial/`, not a quality score).

A second fresh boot on the other unit (same binary, same class) ran `turnbench.py --plan 131072:1
258048:1 --new 1024`: a cold 130,801-token and a cold 257,766-token prime, each followed by a new
turn. Receipts: `census/long/` (47 samples, deepest 258,509 rows).

### Numbers: the panel boot (263 samples, 91 sessions, 9.93e9 element samples)

| quantity | value |
|---|---|
| largest \|x\| (any layer, head, sample) | 101.7 (layer 0, head 9, at 110,592 prime rows); every other layer below 7.8 |
| fp16 headroom | 644x; overflows 0; non-finite in fp32 0 |
| elements in fp16's subnormal range | 10.6% of all (2.0% to 47% per layer: the early layers carry the small values) |
| elements that would flush to zero | 1.2e-4 of all; each is at most 3.0e-8, so the energy lost is below 1e-7 of any head's |
| worst head energy share below 2^-14 | 6.8% (L2 share 26%), layer 1, a head with RMS 8.6e-5 |
| smallest head RMS in any sample | 8.6e-5 (layer 1) |
| one fp16 rounding, relative L2 error, per layer | median 2.07e-4, max 2.75e-4 |
| one bf16 rounding, relative L2 error, per layer | median 1.65e-3, max 2.09e-3 |
| one fp16 rounding, worst head in any sample | 4.13e-4 (layer 2) |
| one bf16 rounding, worst head in any sample | 3.37e-3 (layer 2) |
| bf16/fp16 error ratio per head (worst over samples) | median 8.00, min 5.87; per layer and sample median 8.00, min 5.24 |

The typical element is small: the log2 |x| histogram peaks at 2^-11 to 2^-10 (31% of elements) and
17.5% sit below 2^-13. That is why a tenth of the state is subnormal in fp16, and also why it does
not matter much: in the smallest heads the subnormal spacing (6e-8) is about 1/1400 of the head
RMS, so those heads round at roughly twice the normal-range error, still 5.9x below bf16. Layer 0's
largest element grows with depth in the prime (48.7 at 4,096 rows, 81 at 61K, 101.7 at 110K) and
is 62 to 67 in the task sessions after it.

The long-prime boot agrees on every line: largest |x| 101.7 (the same element of layer 0 head 9,
at 110,592 rows of the 258K prime; the panel and turnbench prompts come from the same long-context
corpus, so their per-depth maxima track each other: 48.7 at 4,096 rows and 101.7 at 110,592 in all three primes), no
overflow, 10.0% subnormal, 1.1e-4 flushed, worst head fp16 error 4.11e-4 against bf16 3.24e-3,
per head bf16/fp16 ratio min 5.79 and median 8.01. The largest element does not grow with depth
past 110K: it moves with the content between 45 and 100 over the second half of the 258K prime
(83.0 at 249,856 rows, 80.7 after the turn).

## Step 2: the fp16 decode class

### Design

The decode class of `research/gdn-state-bf16-decode` (`7456dac2e3`, form 2 there) with fp16 in
place of bf16, ported onto main `05a8a63503` (commit `2d6ce42cf5`; the port rewrote only the added
lines of that diff, `scripts/port_fp16.py`, then replaced the host conversions):

- The prime, every capture, every snapshot plane that leaves the session and every prefix entry
  stay fp32, main's program bit for bit.
- A decode-class layer rounds its fp32 state to fp16 once, when its session starts decoding
  (`Engine::gdn_ensure_decode`, kernel `gdn_state_to_fp16`, `__float2half_rn`), in place, so
  buffer addresses and captured graphs stay valid.
- Decode and verify (`gdn_scan_s128`, `_dc`, `_b`, `_tsnap`, `st16 = 1`) load and store fp16 in
  the first half of the fp32 buffers and round after every token, so a T-row verify equals T
  decode steps bit for bit. A prime that resumes on a decoded session widens back exactly.
- The host oracle conversions (`gdn_state_pack_fp16`, `f32_to_fp16_bits`) follow
  `__float2half_rn` bit for bit: subnormals, ties to even, 65520 and up to infinity, canonical NaN
  0x7fff. Every finite fp16 value round-trips.
- No saturation: the census puts the largest element 644x below the fp16 maximum.
- `MEMRA_GDN_STATE_FP16=1|0` forces the class on an eligible plan; unset, GB10 (120a family plus
  unified memory) decodes the measured Qwen3.8-27B shape in fp16 and every other device and model
  stays fp32. Every measurement below forces `0` or `1`.

### Gates (GB10, binary `arm-2d6ce42cf5`, main baseline `main-05a8a63503`, both built on the unit)

Receipts: `gb10/gates/`.

- **kernel-check** (`MEMRA_KC_FAST=1`, synthetic): 369 OK, 0 FAIL, including the nine
  `gdn_st16_*` cells against the fp16 per-token rounding oracle, bit for bit: the decode and
  T = 3, 7 verify scans, the batched scan (B = 3), the packed verify twin with its snapshots, the
  device-count twin, the rounding point after a chunked fp32 prime, a split prime that rounds and
  decodes to the oracle's bits, and `gdn_ensure_decode` / `gdn_ensure_prime` in place and
  idempotent.
- **fp32 class byte-identical to main**: `forward_last` prefill logits sha256 and the run-gen
  numeric lines (prefill/decode and batched-prime argmax and maxdiff, the 128-token greedy stream)
  are `f444f2cb935605cd` / `7802b4aec0de6e7f` for main and for the lane binary with
  `MEMRA_GDN_STATE_FP16=0`, the same pair the bf16 lane recorded for main `cf85a23ff5` on GB10.
  The fp16 class keeps the prefill logits (`f444f2cb935605cd`, the prime is fp32) and moves the
  decode lines (`b6ddb961d317794c`: the greedy stream parts from fp32 at token 112 of 128,
  argmax MATCH on both checks, maxdiffs 1.620 -> 1.294 and 1.407 -> 1.356).
- **run-spec K=1..8** (greedy, 32 tokens): identical to the plain target 8/8 in both classes.
  Acceptance fp32 12/19 15/32 17/42 17/56 17/70 17/84 17/98 17/112, fp16 12/19 16/32 17/45 17/60
  17/75 17/90 17/105 17/120.
- **GPU tests**, both classes: `gdn_packed_gpu` 3/3 (the packed verify twin equals chained per-row
  launches word for word, eager and graph-rebound, Qwen and Ornith shapes) and `grid_capture_gpu`
  6/6, including `gpu_model_decode_class_restore_and_resume_are_exact`: a session primed with a
  capture inside its prime, rewound to it, primed on and decoded 12 steps equals the cold session
  bit for bit (restore == unrestored through the decode-start rounding), and a decoded state
  survives a snapshot + rollback and a widen + round trip unchanged.

### Quality panel (GB10, the gate)

The 264-task long-context panel (`qual_panel.py` under `/usr/bin/python3`: one 131,550-token
session, then 164 long HumanEval and 100 long GSM8K tasks on it, 8 concurrent), binary
`arm-2d6ce42cf5`, the `gguf-mtpmask` serving arm, fresh boot per run, kernel-check preflight before
each boot (369 OK, 0 FAIL), order fp16, fp32, fp16. Receipts: `gb10/quality/` (records, prime and
summary per run, server logs, memory guards, `compare.txt`).

| run | class | long HumanEval | long GSM8K | 131,550-token prime TTFT | prime identity |
|---|---|---:|---:|---:|---|
| 1 | fp16 decode class | 156 / 164 | 97 / 100 | 158.9 s | pass |
| control | fp32, same binary | 158 / 164 | 97 / 100 | 158.9 s | pass |
| 2 | fp16 decode class | 156 / 164 | 97 / 100 | 158.8 s | pass |
| bf16 decode class (`7456dac2e3`, two runs) | bf16 | 155 / 164 | 97 / 100 | 169 s | pass |
| bf16 lane's fp32 control (`7456dac2e3`) | fp32 | 158 / 164 | 97 / 100 | 169 s | pass |
| panel band on the fp32 default | fp32 | 156-159 | 96-98 | | |

The panel is deterministic. The two fp16 runs produced the same 264 answers byte for byte, and this
binary's fp32 control produced the same 264 answers as the bf16 lane's fp32 control on another
binary, so the run-to-run spread is 0 in both classes. Against the same-binary control, the fp16
class fails HumanEval/91 (an assertion, as in bf16) and /129 (the answer hit the token cap, as in
bf16), and gains no task: two-sided exact p = 0.50 on 2 vs 0 discordant tasks. GSM8K fails the same
three problems in every run. bf16 failed /91, /120 and /129, so 8x finer rounding recovered /120
and nothing else.

The answers that move are as many as under bf16: fp16 changes 45 of the 264 answers (13 HumanEval,
32 GSM8K) and bf16 changed 43 (13 and 30). Finer rounding does not keep the greedy trajectories on
fp32's path any longer on this panel. Once a 16-bit decode rounding flips one greedy token, the
answer diverges, and the panel then scores a different sample of answers. The two-task loss
repeats exactly because the program is deterministic, not because each run is a new draw.

**Verdict: negative on the quality gate.** The gate set for this arm: both fp16 runs in band
(HumanEval >= 156, GSM8K >= 96) and not below the same-binary control by more than the measured
run-to-run spread. Both runs are in band at its floor (156, 97), but 2 tasks below the control
against a measured spread of 0. The default stays fp32 on every device and nothing lands on main.
The decode tick, TTFT and memory cells were not run: the brief stops the arm at a failed gate. The
decode program and its bytes are the bf16 form's (16-bit loads and stores in the same layout), so
the expected tick is that lane's -9.7% at 16 sessions, but this lane did not measure it.

## What would change the answer

- **A quality measure that can resolve two tasks.** On this deterministic panel a 16-bit decode
  class is one fixed sample of 264 answers, and 2 vs 0 discordant tasks is p = 0.50. Several
  seeded or sampled panels, or a larger long-context set, would say whether fp16 costs anything
  real. That is a change to the gate, which is the owner's call, not this lane's.
- **Deterministic stochastic rounding of the state.** Round each element up or down with
  probability given by its remainder, from a counter-based random stream keyed on (layer, head,
  element, token position). A T-row verify and T decode steps would draw the same bits, so
  decode == verify and restore == unrestored still hold, and the rounding would be unbiased
  where round-to-nearest stalls a slowly decaying element. Not built or measured.
- **fp32 for the layers with the smallest states.** The census puts the subnormal-range energy
  and the smallest heads in layers 1 to 16. A mixed class (fp32 there, fp16 elsewhere) keeps part
  of the byte saving. Not built or measured.
- **The census instrument stays on this branch** (`MEMRA_GDN_STATE_CENSUS`), the range receipt a
  later 16-bit state arm on another model needs first.

## Provenance

- Lane code: `a8db6a5dc3` (census), `2d6ce42cf5` (fp16 decode class), on main `05a8a63503`.
  The last code commit on this branch makes the class an off-by-default research door (no
  numeric change in either class).
- Binaries, built on the unit (`121a`, sha256 in `gb10/build/`): `arm-2d6ce42cf5` (lane) and
  `main-05a8a63503` (identity baseline). The banked build logs drop cargo's per-crate progress
  lines (a dependency's name trips the public-boundary provider rule); warnings and results stay. The census binary on the first unit was the same commit
  built with `WHAT=server` (`census/panel/guard.txt` has its sha256).
- Units: the panel census on one GB10 unit; the long census, gates and the three panels on the
  other. Every arm forced `MEMRA_GDN_STATE_FP16=0` or `=1`, so the device default never decided a
  measurement.
- Drivers as run: `scripts/` (`chain.sh`, `step2.sh`, `serve.sh`, `spark_build.sh`, `gates.sh`;
  `bench.sh` was prepared and not run), analysis `census_summary.py`, `panel_compare.py`,
  `xcheck.rs` (host fp16 conversion cross-check).
