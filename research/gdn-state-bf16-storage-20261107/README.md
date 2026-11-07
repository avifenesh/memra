# bf16 storage for the GDN recurrent state (2026-11-07)

Lever: store the Gated DeltaNet recurrent state as bf16 instead of fp32, with fp32 math in
registers. Model: Qwen3.8-27B (NVFP4+Q5_K GGUF), plus Ornith-1.5-35B-A3B and Qwen3.5-9B for the
correctness gates. Hardware: GB10 (DGX Spark, `121a`) and the RTX 5090 Laptop (`120a`).
Branch `research/gdn-state-bf16-storage`, base main `c2e04719b6`.

**Provenance.** Every measurement here ran binary `dev1`: commit `2c89250343` (the lane's first
commit on main `c2e04719b6`), built `120a` on the RTX 5090 Laptop and `121a` on GB10
(`gates-rtx5090/bins-dev1-120a.sha256`, `gb10/bins-dev1-121a.sha256`). The branch was later
rebased onto main `65d5887596` (same lane content as `c95c70ddee`) and took the independent
review's fixes in `af99f7ed36` (one class per process, length guards, handoff layout bit,
per-model host-tier label, a Hopper wgmma cell): no numeric change in either class. The
serve-smoke pin is `e9024f0317`.

**Verdict: negative on the quality gate** (details under GB10: quality panel). GB10 decode tick
-9.7% at 16 sessions and -9.5% at 32, TTFT flat, MemAvailable floor at 256K +1.92 GiB; long
HumanEval 155/164 against a same-binary fp32 158/164 and a 156-159 band. The default stays fp32.

This follows the rounding-only arm of `research/batch-spec-20261101/GDN-STATE-BF16.md`
(branch `research/combo-tc-gdnbf16`), which rounded every state store to bf16 at fp32 storage and
measured quality in band. That arm did not change any allocation, and it rounded only at stores.

## What changed

- **One storage class per engine.** `Engine::gdn_state_bf16` is decided at construction:
  `MEMRA_GDN_STATE_BF16=0/1` forces it, otherwise the device default. Every GDN state kernel takes
  `st16` (`cu/hybrid.cu` `gdn_ld`/`gdn_sv`/`gdn_rn`): fp32 is the old layout byte for byte; bf16
  puts element i at byte 2*i (two per f32 word), stored round-to-nearest-even.
- **Real storage.** In the bf16 class every GDN state buffer is half size: the resident state and
  its ping-pong twin (`memra_kv::recurrent_state_words`, GDN layers only; Kimi delta attention
  stays fp32), verify snapshot rings and per-column checkpoints, the dspark stash slabs, prefix-cache
  entries, their byte accounting and derived budget, host staging, and the host-tier numeric
  identity (`+gdn-state-bf16`, so the two classes never share a host-tier program). Copies are
  sized by buffer length; the commit paths refuse a checkpoint whose length differs from the
  resident plane instead of copying blind.
- **Rounding at every step boundary, not only at stores.** The rounding arm kept the state fp32 in
  registers across the rows of one launch. That breaks two exactness laws: a T-row verify scan no
  longer equals T plain decode steps (the spec gate switches between them inside one request), and
  a chunked prime split at a chunk boundary (a prefix capture, then a restore) no longer equals the
  unsplit prime. The bf16 class therefore rounds the carried state after every token in the
  sequential and packed scans and after every WY chunk in the chunked state pass (f32 K4, K4-mma,
  Hopper K4+K5 wgmma). At T=1 (decode) this is exactly the rounding arm.

## Bytes (Qwen3.8-27B: 48 GDN layers x 48 value heads x 128 x 128)

| | fp32 | bf16 |
|---|---:|---:|
| state per layer | 3.0 MiB | 1.5 MiB |
| state per session (48 layers) | 144 MiB | 72 MiB |
| resident per session (state + ping-pong twin) | 288 MiB | 144 MiB |
| conv ring per session (unchanged) | 5.6 MiB | 5.6 MiB |
| state bytes streamed per decode step per session (read + write) | 288 MiB | 144 MiB |
| the same at 32 sessions | 9.0 GiB | 4.5 GiB |
| verify of T rows per session | (2 + T - 1) x 144 MiB | half |
| recurrent part of one prefix-cache entry | 149.6 MiB | 77.6 MiB |

A prefix entry also holds the 16 attention layers' KV (q8_0 K + q5_1 V: 29,696 B per token):

| entry tokens | fp32 entry | bf16 entry | entries at the same budget |
|---:|---:|---:|---:|
| 2,048 | 217.7 MB | 142.2 MB | x1.53 |
| 8,192 | 400.2 MB | 324.7 MB | x1.23 |
| 16,384 | 643.4 MB | 567.9 MB | x1.13 |
| 32,768 | 1,130.0 MB | 1,054.5 MB | x1.07 |
| 131,072 | 4,049.2 MB | 3,973.7 MB | x1.02 |

## Correctness gates

Binary `dev1`: the lane commit on main `c2e04719b6` (`120a` on the RTX 5090 Laptop, `121a` on GB10,
same source). Raw logs under `gates-rtx5090/` and `gb10/`.

**fp32 class unchanged, byte for byte** (RTX 5090, main `c2e04719b6` binary against this tree
with `MEMRA_GDN_STATE_BF16=0`, `gates-rtx5090/ident/`):

| model | `forward_last` prefill logits (sha256) | run-gen numeric lines (prefill/decode/batched-prime argmax and maxdiff, 128-token greedy stream) |
|---|---|---|
| Qwen3.8-27B | `a297978ec481bed8` both | `9970536dec22031c` both |
| Ornith-1.5-35B-A3B | `9dbe64f55e95174e` both | `ea78b25d4a854a72` both |

The release battery in the fp32 class: kernel-check ALL GREEN (117 cells, 112 executed, 10
required), argmax-margin PASS flips=0 on both models, run-spec K=1..8 PASS 8/8 on both.

**bf16 class equals the rounding arm** (kernel-check `gdn_st16_*`, bitwise, on the RTX 5090 and
on GB10): the oracle is the fp32 kernel run one step per launch (one token, or one C-row chunk)
with the state rounded to bf16 on the host between launches.

| cell | what it pins |
|---|---|
| `gdn_st16_scan T=1` | decode: output and packed state equal the rounding arm |
| `gdn_st16_scan T=3`, `T=7` | a T-row scan equals T chained decode steps of the rounding arm |
| `gdn_st16_batched B=3` | the batched decode scan, per-session pointer arrays |
| `gdn_st16_tsnap T=4` | the packed verify twin: output, final state and the 3 snapshot slots |
| `gdn_st16_dc T=1+2` | the device-count twin equals the plain 3-row scan |
| `gdn_st16_chunked T=200 C=64 f32` | the f32 K4/K5 chunked prime, tail chunk included |
| `gdn_st16_chunked T=160 C=32 mma` | the K4-mma chunked prime (the 120a default) |

Each chunked cell also checks its premise: in fp32 the split at chunk boundaries equals the
unsplit call bit for bit. All 8 cells OK on the RTX 5090 and on GB10 (`kc-fast.log`, 364 OK, 0
FAIL). The Hopper K4+K5 wgmma body carries the same rounding; it compiles for `sm_90a` but no
Hopper box ran it.

**bf16 class gates** (RTX 5090): release battery kernel-check ALL GREEN; argmax-margin PASS on
Qwen3.8-27B (flips=0) and Ornith-1.5-35B-A3B (flips=1, bad=0: one flip inside the calibrated
margin); run-spec K=1..8 PASS 8/8 on both, identical to the plain target: the decode == verify
law holds in the bf16 class. (Store-only rounding, the research arm's form, keeps the state fp32
across the rows of one verify launch, so verify and decode would be two programs; that form was
not run here.)

| model | class | run-spec acceptance K=1..8 (32 tokens) |
|---|---|---|
| Qwen3.8-27B | fp32 | 12/19 15/32 17/42 17/56 17/70 17/84 17/98 17/112 |
| Qwen3.8-27B | bf16 | 12/19 16/32 18/42 18/56 18/70 18/84 18/98 18/112 |
| Ornith-1.5-35B-A3B | fp32 | 14/17 16/30 17/45 17/60 17/75 17/90 17/105 17/120 |
| Ornith-1.5-35B-A3B | bf16 | 14/17 17/30 17/45 18/56 18/70 18/84 18/98 18/112 |

**Restore round trip** (RTX 5090, `grid_capture_gpu` and `gdn_packed_gpu`, both classes, the
model tests on Qwen3.5-9B with the boot class line in the log): a capture inside one prime call
equals the split prime's state, a restore from it primes the next prompt to the cold prime's
logits bitwise, settle-then-resume is cold-exact, and the packed verify equals the chained
per-row launches word for word (state, outputs, every snapshot). 5/5 and 3/3 pass in each class.

**serve-smoke** (RTX 5090, Qwen3.5-9B + draft, prebuilt lane binary): fp32 class 42 ok, 0 fail.
bf16 class 41 ok, 3 fail, all three in the frozen Qwen3.6-35B-A3B cold-mixed arm: every request
of that cell, the 18 hits and the 2 cold misses, returned the same 25 tokens and EOS on its
synthetic periodic prompt, where the fp32 class runs to the 60-token cap. The cold-vs-hit
identity the arm guards holds (all 20 byte-identical). run-gen on the same prompt continues the
pattern 23 tokens in bf16 and 12 in fp32 before leaving it, so the stop is a greedy continuation
of the new class (the batched decode is its own numeric config, `decode_batch_gate.rs` header),
not a state defect. The arm now pins `MEMRA_GDN_STATE_BF16=0`: its frozen workload was qualified
on the fp32 class.

## GB10: quality panel

The 264-task long-context panel (`qual_panel.py`: one 131,550-token session, then 164 long
HumanEval and 100 long GSM8K tasks on it, 8 concurrent), binary `dev1` (`121a`), the
`gguf-mtpmask` serving arm (GGUF NVFP4+Q5_K trunk, masked sxc32768 MTP draft), fresh boot.

Run 1 (bf16 class) graded HumanEval inside the bwrap sandbox with the harness venv's
`sys.executable`, a path the sandbox does not mount, so all 164 programs failed to start
(`bwrap: execvp ... No such file or directory`) before running; generation was unaffected. The
records were re-graded offline with the same judge under the system python
(`scripts/regrade_panel.py`); later runs call the panel with the system python.

| run | class | long HumanEval | long GSM8K | 131,550-token prime TTFT | prime identity | receipts |
|---|---|---:|---:|---:|---|---|
| 1 | bf16 | 155 / 164 | 97 / 100 | 183.2 s | pass | `gb10/run1/quality/` (records re-graded) |
| 2 | bf16 | 155 / 164 | 97 / 100 | 183.5 s | pass | `gb10/run2-bf16/` |
| control | fp32, same binary | 158 / 164 | 97 / 100 | 183.3 s | pass | `gb10/run2-fp32/` |
| panel band on the fp32 default (`qual_panel.py`) | fp32 | 156-159 | 96-98 | | | |
| rounding-only arm (`batch-spec-20261101`) | bf16 rounding at stores | 160 / 164 | 97 / 100 | | | |

The nine bf16 HumanEval misses are real (eight assertion failures, one answer cut at the token cap),
and both bf16 runs missed the same nine: the panel is deterministic for one binary and class.
Against the same-binary fp32 control the discordant HumanEval tasks are 4 fp32-only (93, 108, 129,
160) and 1 bf16-only (154), two-sided exact p = 0.375. GSM8K agrees task for task.

**Verdict: negative on the quality gate.** bf16 storage with exact verify and restore scores 155 on
long HumanEval, one below the 156-159 band, against 158 for fp32 on the same binary. The default
stays fp32 on every device; nothing lands on main (`docs/decisions/GDN-STATE-BF16.md` on this
branch). The served c1..c32 ABAB was not run: the gate failed first, and the decode-tick and TTFT
cells below already price the speed side.

## GB10: coherence (greedy T=0 and sampled T=1 seed 1234, 384 tokens, thinking off)

Same binary, fresh boot per class (`gb10/run1/probes/coherence-bf16{0,1}.jsonl`). Identical: greedy
Hebrew translation and SQL, sampled bash one-liner and SQL (4 of 16). The rest part after 29 to
1,567 characters and stay on topic. The train-trip arithmetic: fp32 greedy ends at 225 minutes in
295 tokens, bf16 greedy is still converting to minutes-from-midnight at the 384-token cap, bf16
sampled reaches 225. Sampled Hebrew names the wrong weekday in both classes (fp32 Friday, bf16
Wednesday); greedy Hebrew is correct and identical in both.

## GB10: TTFT and memory at long context

`turnbench.py --plan 32768:1 131072:1 258048:1 --new 1024 --reps 1`, fresh boot per run, A B B A,
MemAvailable sampled once a second over each run (`gb10/run1/probes/ttft-r*`). Cold TTFT is the
whole-context prime of a fresh session; the turn is a 1,024-token new turn on that session.

| run | class | 32K cold | 128K cold | 256K cold (258,048) | 256K turn | MemAvailable floor |
|---|---|---:|---:|---:|---:|---:|
| r1 | fp32 | 35.97 s | 181.95 s | 455.13 s | 3.94 s | 45.95 GiB |
| r2 | bf16 | 36.48 s | 182.44 s | 455.07 s | 3.54 s | 47.87 GiB |
| r3 | bf16 | 36.00 s | 181.99 s | 455.16 s | 3.59 s | 47.87 GiB |
| r4 | fp32 | 35.99 s | 182.03 s | 455.32 s | 3.68 s | 45.95 GiB |

TTFT does not regress: the prime is attention and GEMM bound, and the chunk-boundary rounding
costs nothing measurable. The 256K floor rises 1.92 GiB in the bf16 class.

## GB10: decode tick

`decode-batch-bench`, Qwen3.8-27B GGUF, greedy, 16 steps x 3 reps per run (median), same binary,
A B B A A B per cell (`gb10/run1/bench/ab.txt`, one log per run). Every bf16 run is faster than every
fp32 run in every cell. The fp32 fingerprints at 16 and 32 sessions (`ffd722c3ca34`,
`ea8db3257f41`) are the ones the batch-spec research recorded on its fp32 builds.

| sessions x depth | fp32 ms/tick | bf16 ms/tick | change | fp32 tok/s | bf16 tok/s |
|---|---:|---:|---:|---:|---:|
| 1 x 2048 | 73.50 | 72.78 | -1.0% | 13.6 | 13.7 |
| 8 x 2048 | 90.88 | 84.98 | -6.5% | 88.0 | 94.1 |
| 16 x 2048 | 122.37 | 110.49 | -9.7% | 130.8 | 144.8 |
| 32 x 2048 | 246.62 | 223.14 | -9.5% | 129.8 | 143.4 |
| 16 x 12288 | 162.89 | 150.93 | -7.3% | 98.2 | 106.0 |

At 32 sessions the state traffic falls from 9.0 to 4.5 GiB per tick; the measured saving, 23.5
ms, is that difference at about 205 GB/s.

## GB10: prefix-cache capacity

Derived budget, same both classes (half of allocatable free on unified memory): 51,292 MB fp32
and 51,318 MB bf16 on the quality boots. The quality panel's own inserts show the entry change:

| entry | fp32 | bf16 | change |
|---|---:|---:|---:|
| spec-boundary insert, 131,520 tokens | 4,306.6 MB | 4,231.1 MB | -75.5 MB |
| lcp-split insert, 131,712 tokens | 4,068.2 MB | 3,992.7 MB | -75.5 MB |
| derived max entry at 262,144 tokens | 7,941.5 MB | 7,866.0 MB | -75.5 MB |

-75.5 MB is 72 MiB, the 48-layer state halving. At this budget a 131K session holds 11.9 entries
in fp32 and 12.1 in bf16 (+1.8%); short entries gain more (x1.23 at 8K tokens, see Bytes).

## What would change the answer

An fp32 prime carry: keep the state fp32 through every prime call and grid capture (prefix entries
fp32, so the prime is exactly the fp32 class), round it once to bf16 at the prime-to-decode
transition, and keep bf16 storage with per-token rounding in decode and verify, where the speed
comes from (-9.7% per tick at 16 sessions). It drops the per-chunk prime rounding, the only place
this build rounds more than the rounding-only arm that scored 160, and keeps both exactness laws.
It is a new lever: prefix entries of two classes and any pooled forward that mixes priming and
decoding sessions need their own gates.
