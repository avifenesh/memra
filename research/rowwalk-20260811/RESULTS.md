# Batched-decode packed-row result

## Verdict

**PROMOTE — keep the persistent-Q packed-row path.** On the designated box1 2x RTX PRO 6000
verification rig, the candidate passed the complete PP-2 exactness battery and the locked
one-hash golden, then improved the interleaved N=5 live-server medians by **+1.142% at c=2,
+1.398% at c=4, and +1.379% at c=8**. Every one of the 15 paired width comparisons was positive.

This satisfies the lane's decision rule: any positive box1 result with exactness green promotes
the arm. The gain is smaller than the original 2–6% expectation, so only the measured values are
claimed. This result authorizes integration of the candidate; it does not itself merge, tag,
release, or move the generated perf board.

## Candidate

The old fallback materialized an owned Q row and an owned attention row around every unchanged
per-session FA launch. The candidate adds a row-view entry to `fa_decode_kvmod` and passes slices
of the existing packed `[B, ...]` Q and attention buffers directly.

For the Step3.7 B>1 walk this means:

- each session still appends its own K/V row in the same order;
- `physical_rows()` remains authoritative for every global/SWA KV view;
- the same split choice, scale, KV encodings, main kernel, combine kernel, and surrounding FP
  chain run in the same order; and
- only two temporary allocations and two arithmetic-free D2D copies per `(layer,row)` disappear.

At B=8 across 45 Step3.7 layers, that removes 720 D2D copy launches per decode step while leaving
the 360 KV appends and 360 per-row FA launches intact. This is the deliberately small
"persistent-Q packed path" candidate, not a multi-row attention kernel.

Code receipts:

- implementation: `188a2d9a` (`perf(decode): keep fallback FA rows packed`);
- focused exactness cell: `eb8abf8a` (`test(decode): pin packed FA row identity`);
- source delta versus lane base `34d6330e`: 15 additions/19 deletions in `decode_batch.rs`, 29/4
  in `lib.rs`, and 51 test additions in `kernel_check.rs`.

## Local 5090 correctness

All GPU blocks held `/tmp/memra-gpu.lock`. A resident ColBERT process occupied 1,390 MiB throughout;
that is fully recorded and makes these correctness-only cells ineligible for timing claims.

| gate | result | raw receipt |
|---|---|---|
| release compile | `kernel-check`, `decode-batch-gate`, `run-gen`, and `run-spec` built under CUDA 13.1/sm_120a, exit 0 | [`local-compile/`](raw/local-compile/) |
| focused FA view contract | hd128, nh64, nkv8, B=4, depth257: copied-row versus packed-row `bitdiff=0`; full `kernel-check` **ALL GREEN**, exit 0 | [`local-kernel-check-rerun/`](raw/local-kernel-check-rerun/) |
| strict multi-session | local 27B, 24 steps each, B=1/2/4/8: gate1, isolated-stream gate2, and device-sampling gate3 all pass; every arm **ALL GREEN**, aggregate exit 0 | [`local-strict-27b/`](raw/local-strict-27b/) |
| generation argmax | prefill/decode **MATCH**; batched-prime/tokenwise **MATCH**, exit 0 | [`run-gen.log`](raw/local-generation-27b/run-gen.log) |
| speculative self-consistency | K=1..8: 8/8 `self-consistency: PASS`, aggregate **SELF-CONSISTENCY PASS**, exit 0 | [`run-spec.log`](raw/local-generation-27b/run-spec.log) |

The strict battery forced the changed generic fallback non-vacuously with
`MEMRA_MMVQ=0 MEMRA_NO_FUSE_NORMQ=1 MEMRA_BATCH_FA=0`. The model was
`Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf` (15,705,920,064 bytes); its draft was
`draft-daily-owntrim-nvfp4head-q4blk.gguf` (1,242,867,296 bytes).

Pinned local binary SHA-256 values:

| binary | SHA-256 |
|---|---|
| `kernel-check` | `c49da014596c01a8001e890a47b5d537978b7148044aa556ca1cf9a02744ea53` |
| `decode-batch-gate` | `550214bd878d62bb188c1d010a7dfa8e42cf56aed93e62a3b96dc7d66d46467b` |
| `run-gen` | `1b98e6da988f6bcba0bb398d71b1163d999def90e4264609a1f0f09e9a7978ec` |
| `run-spec` | `85fbbeb600c5493c37ee44365553bb2eb7f1c370935512ee62ded3f10053a98c` |

## Box1 decision cell

The continuation began only after `cx-opti2` released box1. A nonblocking lock acquisition at
04:39:27Z found the correct host (`<private-host-redacted>`), both GPUs at P8 and 26 C, 0 MiB used, and no
compute applications. The exact candidate source was staged into an isolated checkout and built
under CUDA 13.2 with auto-detected sm_120a.

Pinned candidate provenance:

| item | value |
|---|---|
| source | `fc3a00c939f5deabad2266fbf41609160863ef6f` |
| `memra-server` SHA-256 | `59037f8b46e723b7e4509ddd1c2d3496ba0385b6fdcb75b6036b2fbee09cb964` |
| `kernel-check` SHA-256 | `7ec9d06f7d92ecec3e1066b1055c304bb46b552129e12d1fb9457e3f62bd19fb` |
| `decode-batch-gate` SHA-256 | `53ae8931bfb21a988d59dab70d71c802426ae1d5882da9963780a1a8eeb83da7` |
| `run-gen` SHA-256 | `7225b37f95fc8785fba7649079cc6fe3aab9a339d2eec30afcac862984bc8413` |
| `run-spec` SHA-256 | `466601c6d0e142774ed4c72026418f85d68cdf86b4c4e193648ee08d19cc1051` |

The comparison baseline remained the eagerpar source
`711fbcaaef54491d22488a84d40b7fc35e5a58dd` and server SHA-256
`43ad098d46bb26d644ba0b742d92f3f014d9287ac72e8a0edb8ebf9dac3ba608`.

### Exactness and golden

| gate | box1 result |
|---|---|
| `kernel-check` | **ALL GREEN**; hd128/B=4 copied-row versus packed-row `bitdiff=0` |
| `decode-batch-gate`, PP-2 | B=1/2/4/8, two split reps plus unsplit reference: zero differing logit bits; **0 failing arms** |
| `run-gen` | prefill/decode **MATCH**; batched-prime/tokenwise **MATCH** |
| `run-spec` | K=1..8, 8/8 self-consistency **PASS** |
| fresh-boot golden | expected SHA-256 `21b8293f...445bb6de`; 1/1 match, 0 divergences, 0 errors |

The golden receipt binds the candidate source and server hash and was consumed by the performance
preflight. Raw gates: [`box1-decision-gates/`](raw/box1-decision-gates/).

### Interleaved performance

Metric: aggregate completion tokens divided by live wall time. Each width used five paired rounds,
alternating current/candidate order and a fresh server for every arm. Each measured request
generated 256 tokens; the reducer accepted 40 points and 150 request rows with zero errors,
shedding, or short completions.

| concurrency | current median tok/s | candidate median tok/s | delta | paired wins |
|---:|---:|---:|---:|---:|
| 2 | 119.197 | 120.557 | **+1.142%** | 5/5 |
| 4 | 144.673 | 146.696 | **+1.398%** | 5/5 |
| 8 | 162.001 | 164.235 | **+1.379%** | 5/5 |

Thermal regime: no artificial cooldown, 1,010 samples at 500 ms, 31–45 C. The entire A/B held one
GPU lock from preflight through deterministic reduction. Every server log contains the live
Step3.5 B>1 dispatch receipt and no fatal signature. Raw performance evidence:
[`box1-decision-perf/`](raw/box1-decision-perf/).

## MoESD instrumentation rider

**Still not measured; structurally separate from live rowwalk throughput.** `T_T(B,1)` and
`T_T(B,gamma)` are target-forward timings over a B-by-gamma verify matrix. The ordinary rowwalk
c=2/4/8 server cell is end-to-end plain decode (`gamma=1`), so its aggregate tok/s cannot be
renamed into either target timing.

The audited source has a reduced two-warm-session PP-2 speculative pair path, but no arbitrary
B-by-gamma verify entry and no `moesd-gate` binary. The committed `moesd-harness-20260811` design
requires a standalone bin plus per-layer expert-union telemetry and explicitly excludes live
serving traffic. [`moesd-rider-audit/audit.log`](raw/moesd-rider-audit/audit.log) pins those source
and design contracts. The deferred performance reducer emits explicit null target-timing fields
with this reason, preventing downstream misuse.

The remaining next cell is only the separate MoESD standalone B-by-gamma target-efficiency matrix.
It is neither required for this rowwalk promotion nor implied by the live-serving gain.

## Closeout

- Final decision: **PROMOTE** the persistent-Q packed-row implementation for integration.
- Branch/worktree remained `lane/cx-rowwalk` / `wt-cx-rowwalk`; unrelated work was untouched.
- No `cargo fmt` command ran.
- No origin push, tag, merge, release, or generated perf-board edit occurred in this lane closeout.
- Raw evidence, including the initial busy observations, build attempts, exactness/golden gates,
  request rows, server logs, thermal samples, and deterministic summary, is covered by
  [`raw/SHA256SUMS`](raw/SHA256SUMS).
