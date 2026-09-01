# Q27 + Q35 one-per-card capacity — eu-west PRO pair

Date: 2026-08-12

Lane: `lane/cx-percard`

Rig: 2x RTX PRO 6000 Blackwell Server Edition, one independent server and one model per card

Scored runtime source: `8b2ba8c883152fdbb9f9bbd800a055ad03fe80c4`

## Answer: how many c

**List Qwen3.6-27B at c=16 and Qwen3.6-35B-A3B at c=8.** Those are the first widths to reach
95% of each model's best measured paired throughput while keeping median-window p99 TTFT below
15 seconds. Both servers were generating simultaneously for these receipts.

| Model / physical card | Listing knee | Paired output tok/s | TTFT p99 | Latency p99 | Output tokens/day | Output-only gross/day | Measured-mix gross/day | Headroom over model-pick demand |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.6-27B / GPU 0 | **c=16** | **287.72** | 0.722 s | 7.118 s | 24.86M | $70.00 | $71.19 | **8.72x** the 33 tok/s plan |
| Qwen3.6-35B-A3B / GPU 1 | **c=8** | **606.66** | 0.197 s | 1.687 s | 52.42M | $55.82 | $56.90 | **6.74x** the 90 tok/s plan |
| Derived pair total* | c=16 + c=8 | 894.38 | — | — | **77.27M** | **$125.83** | **$128.09** | clears both plans |

*The pair total adds the two selected per-model paired receipts. The sweep drove equal widths on
both cards, so Q27 c=16 was measured beside Q35 c=16 and Q35 c=8 beside Q27 c=8; it did not add a
separate mixed c16+c8 window after selecting the knees. The component capacities are measured,
while their sum is explicitly derived.

Q27's c=12 result is 94.59% of its best, just below the frozen 95% rule; c=16 is also its
throughput optimum. Q35 reaches 96.15% of its best at c=8. Its throughput-only optimum is c=24
at 630.97 tok/s, only 4.01% above c=8 while latency p99 rises from 1.687 to 4.867 seconds, so c=8
is the better listing default. All measured widths remained below the 15-second TTFT bound.

The dollar columns use the effective input/output prices captured in the
[model-pick report](../modelpick-20260812/REPORT.md): Q27 $0.285/$2.816 per million tokens and Q35
$0.125/$1.065. “Measured mix” values the prompt and completion rates observed at the knee;
Q27 measured 48.19 prompt plus 287.72 output tok/s and Q35 measured 99.53 prompt plus 606.66
output tok/s. “Output-only” values only the completion rate. Both are continuous-utilization
capacity ceilings, not demand, realized revenue, or profit. The model-pick report's approximately
$36/day planning gross remains the demand forecast this capacity receipt clears.

## Simultaneous paired curves

Each row is the median of **N=3** scored windows while both persistent servers were loaded. A
discarded same-width warmup preceded every window. Width order was forward, reverse, then rotated;
paired and peer-resident-idle condition order also rotated. Each request used temperature zero, a
unique cache namespace, zero cached prompt tokens, and exactly 128 completion tokens. Aggregate
output rate spans the global barrier release through that model's final response drain. TTFT and
latency are per-window request quantiles, then medianed across the three windows.

### Qwen3.6-27B — GPU 0, NVFP4 target

| c | Aggregate output tok/s | TTFT p50 | TTFT p99 | Latency p99 |
|---:|---:|---:|---:|---:|
| 1 | 135.18 | 0.073 s | 0.073 s | 0.947 s |
| 2 | 138.59 | 0.119 s | 0.164 s | 1.847 s |
| 4 | 177.81 | 0.294 s | 0.422 s | 2.879 s |
| 8 | 262.74 | 0.476 s | 0.477 s | 3.897 s |
| 12 | 272.17 | 0.590 s | 0.592 s | 5.644 s |
| **16** | **287.72** | **0.720 s** | **0.722 s** | **7.118 s** |
| 24 | 284.56 | 0.922 s | 0.926 s | 10.795 s |

### Qwen3.6-35B-A3B — GPU 1, IQ4_XS target

| c | Aggregate output tok/s | TTFT p50 | TTFT p99 | Latency p99 |
|---:|---:|---:|---:|---:|
| 1 | 355.23 | 0.028 s | 0.028 s | 0.360 s |
| 2 | 355.40 | 0.045 s | 0.061 s | 0.720 s |
| 4 | 451.15 | 0.115 s | 0.167 s | 1.134 s |
| **8** | **606.66** | **0.196 s** | **0.197 s** | **1.687 s** |
| 12 | 563.61 | 0.246 s | 0.247 s | 2.724 s |
| 16 | 621.18 | 0.296 s | 0.299 s | 3.295 s |
| 24 | 630.97 | 0.377 s | 0.380 s | 4.867 s |

## Cross-card interference

**No material cross-card interference was measured.** The control is not a cold or absent peer:
the other model and server stay resident on their card but receive no request. The paired window
then drives the same width on both cards behind one barrier. Positive deltas are regressions for
TTFT/latency and improvements for throughput.

| Model being measured | c | Paired vs peer-idle output | TTFT p99 delta | Latency p99 delta |
|---|---:|---:|---:|---:|
| Q27 | 1 | -0.02% | -0.4 ms | +0.1 ms |
| Q27 | 2 | -0.02% | +0.4 ms | +0.3 ms |
| Q27 | 4 | +0.61% | -19.1 ms | -17.4 ms |
| Q27 | 8 | +0.01% | +1.3 ms | -0.3 ms |
| Q27 | 12 | -0.01% | +2.1 ms | +0.5 ms |
| **Q27** | **16** | **-0.10%** | **+1.5 ms (+0.21%)** | **+6.9 ms** |
| Q27 | 24 | -0.07% | +3.7 ms | +7.2 ms |
| Q35 | 1 | +0.56% | -0.4 ms | -2.0 ms |
| Q35 | 2 | -0.12% | -0.0 ms | +1.0 ms |
| Q35 | 4 | +0.04% | +0.0 ms | -0.7 ms |
| **Q35** | **8** | **+0.06%** | **-0.2 ms (-0.09%)** | **-1.4 ms** |
| Q35 | 12 | +0.17% | -0.0 ms | -5.4 ms |
| Q35 | 16 | -0.31% | +1.7 ms | +9.1 ms |
| Q35 | 24 | +0.31% | -2.0 ms | -16.7 ms |

The frozen materiality rule was a throughput loss greater than 5%, or a TTFT increase exceeding
both 10% and 100 ms, at the selected knee. Q35's p99 TTFT moved **-0.2 ms** when card 0 carried Q27
c=8 traffic; Q27's moved **+1.5 ms** when card 1 carried Q35 c=16 traffic. Across every measured
width, the largest absolute median throughput shift was 0.61% and the largest positive TTFT shift
was 3.7 ms. The paired barrier's worst request-start spread was 6.09 ms. These receipts do not show
meaningful PCIe, CPU, or host-memory interference for the one-model-per-card shape.

## Exactness and speculative state

Correctness ran before timing:

| Gate | Q27 | Q35 |
|---|---|---|
| `run-gen` | prefill/decode MATCH; batched-prime/tokenwise MATCH | prefill/decode MATCH; batched-prime/tokenwise MATCH |
| External-drafter `run-spec` | K=1..8 SELF-CONSISTENCY PASS | K=1..8 SELF-CONSISTENCY PASS |
| Fixed-prompt server golden | 10/10, 516 B, `11cffe49b47503377a370e40e9b9f5b6d888b6c80a97df67386e42edc0cf5c55` | 10/10, 514 B, `6220c20be5847d32f4293f19cfa4e80801435bfc6713dc468359de22eb891a47` |

`kernel-check` independently reported `ALL GREEN (94 cells, 13 skipped)` on each physical GPU.
The 13 skips are named non-candidate fixtures or optional-input skips in the raw logs, not hidden
failures.

Speculative serving was **on for both models**, with the named external drafters attached. The
server's default single-card admission gate was `LOW=2, HIGH=4`: c=1 and c=2 stayed speculative;
at c>=4 it retained a short speculative tail but demoted active sessions to batched plain decode
once four sessions were active. Across the whole campaign Q27 accepted 5,077/7,263 drafted tokens
(69.90%) and Q35 accepted 5,234/7,077 (73.96%). Every paired width recorded speculative activity
in all three replicates, but the high-width throughput is primarily the default batched path, not
an always-speculative result.

## Artifacts and runtime provenance

No model was downloaded or quantized for this campaign. Existing local artifacts were copied to
the eu-west NVMe with resumable verification and then independently re-hashed:

| Role | Bytes | SHA-256 |
|---|---:|---|
| Q27 `Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf` | 15,705,920,064 | `d8d71c7e8a01a1c964fd904a7b496eaef19bdd66827e0949e66c723da742d517` |
| Q27 `draft-daily-owntrim-nvfp4head-q4blk.gguf` | 1,242,867,296 | `b445fbb139e72f9869df06f2f0f91bcaf57527ec34a24bec74d3febd719f3581` |
| Q35 `Qwen3.6-35B-A3B-UD-IQ4_XS.gguf` | 18,209,036,576 | `df27a780435b7b45c2597536112ea3cb091f8544c3d0c3318d9f4258b31f7adf` |
| Q35 `draft-35b-owntrim-nvfp4head-q4blk.gguf` | 944,118,560 | `ae5b7797cc10188bddd00d7e46394e6b8676c1d4e4c6768c8b7b3b10d8870b6a` |

The local `main` bundle was `250ba819e83f868d395c01c6f315a4c6344f54cb` (v0.78.0 class). Its direct
build failed before CUDA compilation because workspace packages were versioned `0.78.0` while
intra-workspace path dependencies remained pinned to `=0.77.0`; the v0.78.0 tag has the same
incomplete one-line bump. The scored build therefore used its parent
`8b2ba8c883152fdbb9f9bbd800a055ad03fe80c4`. `crates/` and `tools/` are byte-identical between
the two commits, and the failed build receipt is retained. This is runtime-code-equivalent v0.78
evidence, not a claim that the exact v0.78 package metadata built successfully.

The release binaries were compiled for auto-detected sm_120a with CUDA 13.2. SHA-256:

- `memra-server`: `924ff232d5c9eca0e0bcff2784fcb1cf49bd5c59b10721edd2a477884e7a8498`
- `kernel-check`: `ab4fffd88d056dd881f25ee2951cdcb7f03365b16daa28e4fe7dd9f8168fb962`
- `run-gen`: `4bed1812cce2f1e8ff5ac784eae75445eb46ee013de23dff4a3bc7d2883a3224`
- `run-spec`: `a143293f99a20f52bb14fa9c082faa4452cda61d2870bbfd8a79ae5f16782903`

## Thermal regime and receipts

The detached campaign held `/tmp/memra-gpu.lock` from 2026-08-12T00:16:09Z through the final
`PERCARD_CAMPAIGN_PASS` at 00:25:11Z. There was no artificial cooldown. Continuous 250 ms sampling
retained 4,318 per-GPU rows, or 2,159 two-card intervals:

| Physical GPU / model | Peak temperature | Peak power | Peak used VRAM | Peak utilization |
|---|---:|---:|---:|---:|
| GPU 0 / Q27 | 53 C | 376.51 W | 28,023 MiB | 100% |
| GPU 1 / Q35 | 46 C | 326.54 W | 23,283 MiB | 99% |

The scored set contains 21 paired windows, 42 peer-idle control windows, 84 model-condition points,
102,912 completion tokens, and zero request errors. Every scored response finished at exactly 128
tokens with `finish_reason=length`; worker counters settled after every window. Host telemetry
recorded zero swap-in and swap-out. Server logs contain no observed OOM, CUDA failure, Xid, panic,
or fatal event. After the completion sentinel both GPUs returned to P8, 0 MiB, 0% utilization with
no compute process.

- [`raw/campaign/summary.json`](raw/campaign/summary.json) is the machine-readable reduction; two
  local reductions reproduced SHA-256 `90905b2e2ffeb6258af8034b1710f93e6f6d39b91f93a228d9fdd94d161345be`.
- [`raw/campaign/MANIFEST.sha256`](raw/campaign/MANIFEST.sha256) verified all 656 remote payload
  files captured before the driver wrote its final pass line. The 680-entry local
  [`raw/SHA256SUMS`](raw/SHA256SUMS), itself SHA-256 `6d09541f7f4faae61289351d83044adfec3d80ad07fa9eb8638214389ac0eec7`,
  also covers the completed driver/launch logs and derived summary.
- [`raw/campaign/perf/`](raw/campaign/perf/) retains every warmup and scored request JSONL, metrics
  delta, thermal snapshot, condition, width, and replicate. [`raw/campaign/driver.log`](raw/campaign/driver.log)
  records all 63 warmups, all 63 scored condition windows, teardown, and the pass sentinel.
- [`raw/gates/`](raw/gates/) contains the two-card kernel checks and per-model `run-gen`/`run-spec`
  receipts. [`raw/setup/artifact-manifest.txt`](raw/setup/artifact-manifest.txt) pins the source and
  remote artifact identities.

This is lane-local research evidence. No runtime code, generated performance board, merge, tag,
push, or formatting surface was changed.
