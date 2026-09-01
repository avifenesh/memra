# PP-N Step35 B=1 correctness default

## Verdict

**PASS — ship the fail-closed correctness fix.** Step3.5/Step3.7 on PP-N now use
`step35_decode_batch_layers` at every live decode width. The fixed binary returned one
326-byte completion class across all 35 fresh boots and all 150 requests in the required
transition matrix:

`21b8293f2298978c74fb89f32d9b14e3ea921f39924cfce88c73b01f445bb6de`

There were zero request errors, zero golden divergences, and no second completion hash.
The load-history-dependent three-class behavior isolated in `research/p0iso-20260810` is
closed for this serve shape.

The correctness price is real and measured plainly: sustained c=1 decode falls from
85.423 to 81.399 tok/s, **-4.710%** at the N=5 median. Short-request TTFT is flat within
the interleaved run (70.281 to 70.125 ms, -0.222%). The decode regression is below the
10% escalation threshold, so it does not block this fix. Restoring eager-path parity
without restoring the numeric-class boundary remains the step-2 performance follow-up.

## Code fix

The PP-N branch now makes the model contract explicit:

- `b1_stage_fast` additionally requires `self.cfg.step35.is_none()`, so non-Step35 models
  retain their existing eager/fused B=1 fast path while Step3.5/Step3.7 cannot enter it:
  [`decode_batch.rs`](../../crates/memra-engine/src/decode_batch.rs#L787).
- `step35_batched` is selected directly from Step35 metadata at every width. If the
  rollback seam disables that trunk on PP-N, B=1 fails closed with the same explicit
  engine error as B>1 instead of falling back to the different eager numeric class:
  [`decode_batch.rs`](../../crates/memra-engine/src/decode_batch.rs#L802).
- No new user flag or scheduler affinity was introduced. The worker may continue grouping
  ready rows freely because every Step35 PP-N width now enters one stage-scoped trunk.
- The engine exactness gate retains its stage-fast arm for eligible non-Step35 models and
  marks it inapplicable for Step35, whose B=1 batched body is already covered by the
  B=1 split/unsplit arms:
  [`decode_batch_gate.rs`](../../crates/memra-engine/src/bin/decode_batch_gate.rs#L747).

## Required one-hash matrix

Every cell was a fresh server boot under one bounded `/tmp/memra-gpu.lock` hold per
condition. The measured prompt was not used as a warmup. The fixed golden is the batched
class from the isolation lane.

| condition | fresh boots | requests | errors | divergences | completion hashes |
|---|---:|---:|---:|---:|---:|
| c=1 | 10/10 | 10/10 | 0 | 0 | 1 |
| c=8 barrier | 10/10 | 80/80 | 0 | 0 | 1 |
| c=2 barrier | 10/10 | 20/20 | 0 | 0 | 1 |
| first-late | 5/5 | 40/40 | 0 | 0 | 1 |
| **total** | **35/35** | **150/150** | **0** | **0** | **1** |

All 150 row receipts independently record `ok=true`, `golden_match=true`, 326 bytes, and
the hash above. Reduced counts are in
[`matrix-summary.json`](raw/matrix-summary.json); the complete per-boot server, scheduler,
GPU, metrics, request, and response receipts are under [`matrix/`](raw/matrix/).

## B=1 performance cost

The A/B used five reps per arm in one lock hold, with a fresh server and an idle-GPU wait
for each arm. Order alternated by pair: base/fixed, fixed/base, base/fixed, fixed/base,
base/fixed. Both GPUs retained their 600 W power limits; between-arm idle snapshots were
27-32 C. The model bytes, prompt, PP-2 placement, context, grouped-MoE setting, prefill
tick, and `MEMRA_SERVE_SPEC=0` were identical.

| c=1 metric | pre-fix eager B=1 | fixed batched B=1 | fixed vs base |
|---|---:|---:|---:|
| sustained decode, tok/s (higher is better) | 85.423 | 81.399 | **-4.710%** |
| short streaming TTFT, ms (lower is better) | 70.281 | 70.125 | -0.222% |

Each table value is an N=5 median. Sustained decode is
`(completion_tokens - 1) / (latency - TTFT)` for a 256-token greedy streaming request;
the short TTFT arm is an 8-token greedy streaming request after one warmup request. The
ten raw request rows, full sample arrays, server logs, and thermal snapshots are under
[`perf/`](raw/perf/), with the deterministic reduction in
[`perf-summary.json`](raw/perf-summary.json).

The 4.710% loss is the honest cost of removing the Step35 eager/fusion incentive from the
live B=1 path. It is not hidden by aggregate wall time or a wider-concurrency result.

## Gate-gap closure

The standing Step35 B2 geometry gate now exercises the live production default rather
than masking it:

1. The server launch explicitly removes inherited `MEMRA_SERVE_B1FAST` and
   `MEMRA_STEP35_BATCH` values and no longer injects `MEMRA_SERVE_B1FAST=0`:
   [`step35-b2-geometry-gate.sh`](../../tools/step35-b2-geometry-gate.sh#L90).
2. Static c=1/c=2/c=4 served-byte comparisons remain.
3. A stateful cell starts one streaming row, waits for its first content-bearing frame,
   admits two late rows, and requires tick-trace evidence of `ready=1` followed by
   `ready>=2`:
   [`step35-b2-geometry-gate.sh`](../../tools/step35-b2-geometry-gate.sh#L132).
4. The early and both late completions must equal the c=1 reference, while chunk-cap and
   first-B>1 log checks prove a batched Step35 walk actually ran.
5. The canary disables `MEMRA_STEP35_BATCH`; the new runtime fails closed and the gate
   passes only because its live assertions break.

The naked gate passed every static and transition assertion. Its tick slice proved
`ready=1 -> ready>=2`, all three transition outputs matched the reference, chunk cap was
8, and the server logged the first B>1 Step35 walk. The canary produced the expected
engine errors and broke five assertions. The archived receipt predates a reporting-only
guard that skips JSON decoding when its c=1 reference is the expected error; the final
script removes that redundant parse/hash failure, still breaks four independent assertion
groups, and avoids the traceback without weakening the canary.

## Standard target-rig battery

| gate | result |
|---|---|
| `kernel-check` | **ALL GREEN**, CPU reference |
| `decode-batch-gate --mode pp --batch 1,2,4,8 --steps 24 --reps 2 --stages 2 --plen 520` | **ALL GREEN**, every split repeat and unsplit comparison bit-identical; 0 failing arms |
| `run-gen` Step3.7 | prefill/decode **MATCH**; batched-prime/tokenwise **MATCH** |
| `run-spec` Step3.7 + MTP | K=1..8 **SELF-CONSISTENCY PASS** |
| Step35 chunk invariance | naked 4096/513/512/256/64 exact; rollback canary **CHUNK-DEPENDENT** as required |
| Step35 tick invariance | naked budgets 0/1024/513/512/256/64 and splits 64/256/512 exact; rollback canary **TICK-DEPENDENT** as required |
| live-default B2 geometry + transition | naked **PASS**; fail-closed canary **PASS** |
| `serve-smoke` | **0 failed** across plain API, concurrency, long generation, cache metering, spec/plain equality, sampled truncation, and affinity replay |

The optional Gemma4 arm in `serve-smoke` skipped because that unrelated artifact was not
present. Every Step35/Step3.7 arm required by this lane ran. Reduced results are in
[`gate-summary.json`](raw/gate-summary.json), with full logs under [`gates/`](raw/gates/).

## Provenance

- Rig: box1 hyperscaler pair, 2x NVIDIA RTX PRO 6000 Blackwell Server Edition, PP stages 2 on
  devices 0,1.
- Serve shape: context 262144, grouped MoE on, prefill tick 2048.
- Trunk artifact: `Step-3.7-flash-IQ4_XS-00001-of-00003.gguf`, 46,483,327,296 bytes.
- MTP artifact: `Step3.7-flash-mtp-Q8_0.gguf`, 3,707,276,416 bytes.
- Fixed runtime source for the matrix: `6e50efdbaefe7167f6c48309abd6892252220eb8`.
- Fixed `memra-server` SHA-256:
  `6a7c2046eb3197773def91baf012abd629e0b0ced239ec2d38016c93be5ca7e5`.
- Perf base source/binary: `188154299064a42b67fc8eb1f41757cf6237300d` /
  `e7e6515e9f47030a7137ba9fdf7c40d43f0764d02699b38959f134ee0ace65b3`.
- Matrix probe SHA-256:
  `6c9e7386e3304deb6b625db1e7bd5089b3f0cf4844c198b17d7173e5c0082e9d`.
- Raw evidence manifest: [`raw/SHA256SUMS`](raw/SHA256SUMS).
- Reproducer: [`qos_probe.py`](../p0iso-20260810/qos_probe.py) and the parameterized
  [`run-box1.sh`](../p0iso-20260810/run-box1.sh).
- Perf driver: [`perf-box1.sh`](perf-box1.sh).

The matrix, performance run, and every gate block held `/tmp/memra-gpu.lock`; block-start
receipts showed no other compute applications. No origin push, tag, or release was made.
