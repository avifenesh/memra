# bf16 storage for the GDN recurrent state (2026-11-07)

Lever: store the Gated DeltaNet recurrent state as bf16 instead of fp32, with fp32 math in
registers. Model: Qwen3.8-27B (NVFP4+Q5_K GGUF), plus Ornith-1.5-35B-A3B and Qwen3.5-9B for the
correctness gates. Hardware: GB10 (DGX Spark, `121a`) and the RTX 5090 Laptop (`120a`).
Branch `research/gdn-state-bf16-storage`, base main `c2e04719b6`.

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
