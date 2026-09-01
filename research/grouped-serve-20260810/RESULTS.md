# GROUPED SERVING PROMOTION — box1, 2026-08-10

## Verdict

**PROMOTE for the Step-3.7-Flash PP-2 RTX PRO 6000 serving configuration.** Add
`MEMRA_MOE_GROUPED=1` to that deployment environment. Do **not** change the naked
runtime default: the local RTX 5090 transfer gate remains a valid `-75.3%`
rejection for a global flip.

On the current box1 serving binary, the same-window interleaved 4107-token A/B
moved streaming TTFT from **10.963 s to 7.260 s p50** (N=5 per arm): **-3.703 s,
-33.8%, 1.510x**, with grouped winning all five adjacent pairs. The standalone
prefill A/B independently won all five pairs at every prompt class, by
**+53.4% / +59.9% / +62.3%**. All required exactness gates passed.

The grouped c=4 receipt is healthy but saturated: four cold 4107-token prompts
reached first visible output in **29.029 s p50 wall** across N=3 bursts, at
**565.9 aggregate prompt tok/s**. The trace shows `prefill_batch_calls=0`; these
long prompts remain fairness-sliced single primes. This is an operational
capacity warning, not an exactness or liveness failure, and there is no c=4 OFF
arm here from which to attribute the saturation specifically to grouped mode.

This satisfies the pre-registered serving-config rule: exact, materially faster
on the owned deployment shape, and no c=4 correctness or serving failure. The
promotion is scoped to the resident Step PP-2 pair; other models, single-card
rigs, mixed/spill banks, and the naked command retain the current policy.

## Serving TTFT A/B

Box1 used two RTX PRO 6000 Blackwell Server Edition cards, PP stages/devices
`2` / `0,1`, `MEMRA_CTX=262144`, spec off, and the same `memra-server` binary for
both arms. Each arm used an independent server boot, one excluded warmup, one
measured request, a unique cold cache salt, and alternating pair order inside
one `/tmp/memra-gpu.lock` hold.

| arm | 4107-token streaming TTFT p50 | measured range | N | adjacent wins |
|---|---:|---:|---:|---:|
| grouped OFF | 10.963 s | 10.960-10.988 s | 5 | — |
| grouped ON | **7.260 s** | 7.255-7.279 s | 5 | **5/5** |

The derived same-window change is `-3.702950 s` / `-33.7756%`. The earlier
pp2pipe naked result of 9.771 s is a different commit and measurement window;
it remains historical context, not this decision's denominator. Snapshot
temperatures were 34-38 C, and both cards were at 0 MiB at every arm boundary.

Raw receipt:
[`raw/box1/serve/ttft-client-20260809T234941Z.jsonl`](raw/box1/serve/ttft-client-20260809T234941Z.jsonl)
and the ten paired client/server logs beside it.

## Standalone prefill A/B

Each cell is N=5 independent processes per arm, one warmup per process,
interleaved with alternating arm order. The `pp512` and `pp2048` labels realize
461 and 1833 tokens respectively; `pp4096` realizes 4096.

| shape | grouped OFF median | grouped ON median | throughput change | adjacent wins |
|---|---:|---:|---:|---:|
| pp512, T=461 | 1.3765 s / 334.9 tok/s | **0.8975 s / 513.6 tok/s** | **+53.4%** | 5/5 |
| pp2048, T=1833 | 4.4985 s / 407.5 tok/s | **2.8129 s / 651.6 tok/s** | **+59.9%** | 5/5 |
| pp4096, T=4096 | 9.5995 s / 426.7 tok/s | **5.9134 s / 692.7 tok/s** | **+62.3%** | 5/5 |

All 30 timed arms exited zero. Snapshot temperatures were 29-40 C; both cards
were 0 MiB at the lock boundaries. The controlled results reproduce the prior
Lever C sign and magnitude on today's tree.

Raw receipt:
[`raw/box1/prefill/prefill-results-20260809T232706Z.tsv`](raw/box1/prefill/prefill-results-20260809T232706Z.tsv),
[`raw/box1/prefill/prefill-samples-20260809T232706Z.tsv`](raw/box1/prefill/prefill-samples-20260809T232706Z.tsv),
and all per-process logs beside them.

## Exactness

Every gate ran with `MEMRA_MOE_GROUPED=1` explicitly set on the model-backed
Step artifact.

| gate | result |
|---|---|
| grouped versus sequential oracle | **PASS:** 210/210 `BYTE-IDENTICAL`, zero `MISMATCH`; both `resident-q8-rows` and `resident-q8-clamped-pairs` observed |
| model-backed `kernel-check` | **PASS:** `ALL GREEN: kernels match CPU reference.`; Step IQ4_XS model-backed rows passed at T=16/64/128/512 |
| PP-2 `run-gen` | **PASS:** prefill/decode argmax `6776` MATCH; batched-prime/tokenwise argmax `6776` MATCH |
| serving fault scan | **PASS:** no CUDA error, illegal address, OOM, panic, Xid event, request error, or server death in the corrected scan |

The gate hold ran at 31-37 C and returned both cards to 0 MiB. This lane changes
no engine code and makes no merge/tag; speculative serving was disabled on the
measured PP-2 shape, so no new `run-spec` score is claimed here.

Raw receipt:
[`raw/box1/gates/gates-summary-20260809T222302Z.log`](raw/box1/gates/gates-summary-20260809T222302Z.log)
and its three full gate logs.

## Grouped c=4 interaction

The scored recovery used the same known-visible chat prompt as the TTFT A/B,
one excluded c=1 warmup, then three barrier-synchronized c=4 bursts. All 12
scored requests reported exactly 4107 prompt tokens, `cached_tokens=0`, eight
visible chunks, and a normal `length` finish.

| burst | wall to last first-visible | aggregate prompt tok/s | request TTFT p95 |
|---:|---:|---:|---:|
| 1 | 29.176 s | 563.1 | 29.176 s |
| 2 | 29.012 s | 566.3 | 29.012 s |
| 3 | 29.029 s | 565.9 | 29.029 s |
| **median, N=3** | **29.029 s** | **565.9** | **29.029 s** |

The excluded c=1 warmup was 7.412 s / 554.1 tok/s. Relative to that single
context sample, c=4 is 3.916x the latency for only 2.1% more aggregate prompt
throughput. The scheduler trace explains the shape: every long-prompt tick says
`prefill_single_calls=4` and `prefill_batch_calls=0`, with 1024-token fairness
slices. The later `[step35-batch] ... B=4` record occurs after priming, when the
four ready sessions enter the batched Step walk; it is not evidence that the 4k
primes themselves batched. Scale concurrent long-prime demand with more pairs or
a different prime mechanism; this flag promotion alone does not create c=4
capacity scaling.

The first burst attempt is retained and excluded, not hidden. Its synthetic
4096-token c=1 warmup completed server-side with first SSE byte at 5.904 s and
eight decode ticks, but produced no visible decoded string; the old client raised
`RuntimeError: prime request completed without visible output` before any scored
c=4 row. Its zero-byte JSONL is therefore not evidence. That first driver's
plain `Xid` regex also matched normal gpu-watch startup prose, not an Xid event.
The burst-only recovery changed the client prompt/probe and corrected that scan;
it did not rerun or select the completed TTFT arms.

Raw receipt:
[`raw/box1/serve/burst-c4-20260810T001946Z.jsonl`](raw/box1/serve/burst-c4-20260810T001946Z.jsonl)
and
[`raw/box1/serve/server-burst-grouped-20260810T001946Z.log`](raw/box1/serve/server-burst-grouped-20260810T001946Z.log).
The excluded attempt is beside them under timestamp `20260809T234941Z`.

## Exact deployment env diffs

The actual owned Step PP-2 environment should contain this block:

```bash
MEMRA_PP_STAGES=2
MEMRA_PP_DEVICES=0,1
MEMRA_MOE_GROUPED=1
MEMRA_CTX=262144
```

There is no tracked `serve-env.sh` in base `2d9359df`. The current owned
equivalent is the RunPod provisioner's generated `/etc/memra/runpod.env`, plus
its dry-run contract and API runbook. The exact candidate patch is:

```diff
diff --git a/deploy/runpod/provision.sh b/deploy/runpod/provision.sh
--- a/deploy/runpod/provision.sh
+++ b/deploy/runpod/provision.sh
@@ -198,3 +198,4 @@
   MEMRA_PP_STAGES=2
   MEMRA_PP_DEVICES=0,1
+  MEMRA_MOE_GROUPED=1
   MEMRA_CTX=131072
@@ -586,3 +587,4 @@
 MEMRA_PP_STAGES=2
 MEMRA_PP_DEVICES=0,1
+MEMRA_MOE_GROUPED=1
 MEMRA_CTX=131072
diff --git a/deploy/runpod/API-USAGE.md b/deploy/runpod/API-USAGE.md
--- a/deploy/runpod/API-USAGE.md
+++ b/deploy/runpod/API-USAGE.md
@@ -85,3 +85,4 @@
 MEMRA_PP_STAGES=2
 MEMRA_PP_DEVICES=0,1
+MEMRA_MOE_GROUPED=1
 MEMRA_CTX=131072
```

Those tracked snippets retain their pre-existing 131072 context line because
this lane decides only grouped serving. The measured owned environment was
262144 as shown above. Apply `MEMRA_MOE_GROUPED=1` only where the model/hardware
contract is the Step PP-2 RTX PRO pair. The immediate rollback is
`MEMRA_MOE_GROUPED=0`; removing the line also returns to today's default-off
behavior.

Per the requested stop line, the patch above is recorded but not applied in this
lane. No runtime default, deployment file, origin branch, tag, or release was
changed.

## Provenance and receipts

- Measurement source: `0fc0677a35039d0842b4ce464138371cf4514576` on
  `lane/cx-grouped`, based on `2d9359df353b00f196b124aa62e19ce3bfb7789a`.
- Burst-only client recovery: `188154299064a42b67fc8eb1f41757cf6237300d`;
  server binary unchanged.
- `memra-server` SHA-256:
  `e7e6515e9f47030a7137ba9fdf7c40d43f0764d02699b38959f134ee0ace65b3`.
- Build: CUDA 13.2 / sm_120a, rustc 1.97.1 on box1; full binary, harness,
  four-artifact size/hash, host, and compiler manifests are under
  [`raw/box1/build/`](raw/box1/build/).
- Prompt SHA-256:
  `23c1d8384a16c7c0bcb7736b412d43e64c0b4d8e238703864e928565f824ae11`.
- [`raw/SHA256SUMS`](raw/SHA256SUMS) covers all 85 raw receipt files except
  itself; `sha256sum -c` passed locally.
- The requested `~/.lanectl/inbox/cx-grouped.md` remained absent at every
  bounded-block check.
