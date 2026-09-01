# Multi-row FA decode entry result

## Verdict

**NO-GO — do not promote this lane as a performance win.** The implementation passed every
committed local correctness gate, including the new hd128/B=4 multi-view identity cell, but the
required RTX 5090 N=5 interleaved A/B was below noise. Candidate median deltas were **+0.000% at
B=2, +0.114% at B=4, and +0.131% at B=8**, far below the expected +2–6% class, and paired deltas
crossed zero at every width. B=1 did not regress by the median (+0.075%), but that does not rescue
the missing B>=2 gain.

The local timing artifact is also a non-target proxy: the 32-layer Qwen3.5 9B NVFP4 logs emit no
`[step35-batch]` marker even at B=2/4/8. That marker is unconditional on the first B>1 Step35 walk
and already exists in the branch-point source, so this run does not demonstrate that the changed
Step3.7 dispatch executed. The focused kernel cell is non-vacuous correctness evidence; the local
9B throughput cell is not evidence of the intended Step3.7 launch reduction. A future
reconsideration would require an actual Step3.7 artifact on a suitable rig and a live dispatch
receipt, but this lane's requested timing verdict is NO-GO.

## Candidate and provenance

- Branch-point source: `ba3e70c9af455320dc661ab023e5c653539bc447`.
- Candidate code source: `3845bda8358a6fe5883095250d3d8e6df84fda2a`
  (`perf(decode): batch step35 FA views`).
- Exactness receipt commit: `2241bea916cf3daaf60a805fa3189c611eb80f0f`.
- Pre-existing performance-receipt commit: `5f36611f500b1fedb689d5cd710e295e42a94282`.
- Baseline detached worktree: `/tmp/memra-fadecoderow-base-src.0WnCb2`, at the branch point.
- Baseline separate target: `/tmp/memra-fadecoderow-base-target.0vXLcN`.
- Baseline binary SHA-256: `308e8194e5b15d8ea1dd025cb619a5eb28d436765c53d3edef2b8d3f83aaac36`.
- Candidate binary SHA-256: `3db8c7e9bc9e8a9da745484358443dbf9bd88e1d73dc029a848854d00d6dde7a`.
- Timing artifact: `Qwen3.5-9B-NVFP4-MTP-GGUF.gguf`, 5,657,607,424 bytes.

The harness required both source revisions, required the binary hashes to differ, and rejected any
engine diff after the pinned candidate code commit. Thus the later research-only commits did not
alter the timed engine. In source, the existing `b_n == 1` attention branch remains ahead of the
new B>1 multi-view arm, and the new context builder returns immediately for B<=1.

## Correctness

All evidence below was committed before timing in `2241bea91`.

| gate | result | raw receipt |
|---|---|---|
| focused FA-v3 multi-view cell | hd128, nh64, nkv8, B=4; depths 96/127/257/511 and offsets 0/3/17/31; multi-view row grid versus per-row views: `bitdiff=0` | [`kernel-check.log`](raw/kernel-check/kernel-check.log) |
| full kernel battery | `ALL GREEN (102 cells, 1 skipped)` | [`kernel-check.log`](raw/kernel-check/kernel-check.log) |
| NVFP4 decode-batch config | B=1/2/4/8, 32 steps: gate1, bit-checked isolated-stream gate2, and sampling/lean gate3 all pass; four `ALL GREEN` results | [`decode-batch-gates/`](raw/decode-batch-gates/) |
| NVFP4 strict | B=4 against `decode_step_h`, 32 steps: bit-identity gate1/gate2 and gate3 pass; `ALL GREEN` | [`nvfp4-strict-b4.log`](raw/decode-batch-gates/nvfp4-strict-b4.log) |
| Q8_0 decode-batch config | B=1/2/4/8, 32 steps: gate1, bit-checked isolated-stream gate2, and sampling/lean gate3 all pass; four `ALL GREEN` results | [`decode-batch-gates/`](raw/decode-batch-gates/) |
| Q8_0 strict | B=4 against `decode_step_h`, 32 steps: bit-identity gate1/gate2 and gate3 pass; `ALL GREEN` | [`q8-strict-b4.log`](raw/decode-batch-gates/q8-strict-b4.log) |
| generation argmax | NVFP4 and Q8_0 each report prefill/decode `MATCH` and batched-prime/tokenwise `MATCH` | [`generation-gates/`](raw/generation-gates/) |
| speculative self-consistency | NVFP4 and Q8_0 each pass K=1..8 (8/8) and report aggregate `SELF-CONSISTENCY PASS` | [`generation-gates/`](raw/generation-gates/) |

## Timing

The valid window is [`raw/perf-nvfp4-rerun2/`](raw/perf-nvfp4-rerun2/). It used five paired outer
rounds with alternating arm order, a fresh process per arm, one discarded in-process warmup and
one timed 128-step rep per B=1/2/4/8, context 512, `MEMRA_FAST=1`, `nice -n 15`, and
`ionice -c3`. One exclusive GPU lock covered both discarded process warmups and all ten timed
arms. There was no artificial cooldown or clock control.

Values are aggregate tok/s. Throughput spread is min–max over N=5; paired spread is min–max of
the five per-round candidate deltas.

| B | baseline median (spread) | candidate median (spread) | median delta | paired median (spread) | paired wins |
|---:|---:|---:|---:|---:|---:|
| 1 | 132.9 (132.7–134.1) | 133.0 (132.9–133.4) | **+0.075%** | +0.000% (-0.522–+0.151%) | 2/5 |
| 2 | 209.1 (208.7–210.8) | 209.1 (208.8–210.2) | **+0.000%** | +0.000% (-0.285–+0.048%) | 2/5 |
| 4 | 350.7 (350.0–354.1) | 351.1 (350.1–352.9) | **+0.114%** | +0.085% (-0.339–+0.171%) | 4/5 |
| 8 | 457.8 (456.6–461.0) | 458.4 (456.7–460.2) | **+0.131%** | -0.131% (-0.240–+0.241%) | 2/5 |

Thermal regime: 390 samples at 500 ms in one continuous window, 53–88 °C, 180–2,167 MHz SM
clock, and 9.23–173.64 W. Every warmup and timed process exited 0. The full machine-readable
reduction is [`summary.json`](raw/perf-nvfp4-rerun2/summary.json).

The earlier committed `perf-nvfp4-attempt1/` and `perf-nvfp4/` directories are failed harness
receipts, not measurements: the first stopped on a Bash local-expansion bug and the second stopped
after the baseline warmup because a pre-existing Step35 marker was incorrectly treated as
candidate-only. `perf-nvfp4-rerun1/` preserves the follow-up that proved the 9B artifact emits no
Step35 marker. None contains five timed pairs, so rerunning them as one fresh window was required.

## Closeout

- Correctness: **PASS** on the committed local battery.
- B=1 non-regression: **PASS** by median, with no systematic paired loss.
- B>=2 performance expectation: **FAIL**; observed deltas are below noise and below +2–6%.
- Promotion verdict: **NO-GO**.
- No merge, tag, push, generated-board edit, or `cargo fmt` was performed.
