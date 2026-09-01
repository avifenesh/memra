# FA3 softmax/PV pipeline result

Date: 2026-08-13
Verdict: **NO-GO**

The baseline profile found a real serialized QK/PV scheduling opportunity, but the tested
sm_120a software pipeline is slower on both served models. The loss is outside run-to-run
spread in both prime time and cold TTFT. The production kernel is therefore unchanged; the
exact negative candidate is retained only as [`candidate.patch`](candidate.patch).

All numbers below are relative-only observations from the local RTX 5090 Laptop GPU under the
owner-imposed 210--1200 MHz cap. They are not absolute-throughput claims and are not transferable
to the PRO target. Because the candidate loses clearly here, a box1 PRO confirmation is not
warranted for this formulation.

## What the profile established

The exact cold 4,860-token serving trace reached `fa_dequant_kv_ws_bf16` followed by
`fa_prefill_qw_db` on both models. Baseline NCU reported 8.33% occupancy, 0.12 eligible
warps/scheduler, 29--34% tensor-pipe activity, and 3.84--3.94 short-scoreboard cycles per issued
instruction. QK and PV together accounted for roughly three quarters of sampled cycles and more
than 94% of short-scoreboard samples; softmax plus P stores accounted for only 5.4--5.7%.

That evidence justified testing cross-warp QK/PV staggering. Full extracted tables and profiler
artifact hashes are in [`PROFILE.md`](PROFILE.md). No NCU or Nsys report file is in the repository.

## Candidate tested

The candidate split the four warps into two cohorts, kept one cohort one KV tile ahead, added a
second P stage, and used generation-tagged shared-memory handshakes so lead QK/softmax could run
while lag PV consumed the preceding tile. Every warp retained its original KV traversal, online
softmax recurrence, and MMA accumulation order.

The measured binary is reproducible from the lane base with
`git apply --unidiff-zero candidate.patch`. The zero-context
[`candidate.patch`](candidate.patch) has SHA-256
`56cee58dfa85fcbce8a06d2e45ea32dbd0cbe1f82042012de2a462bebca30492`.
The baseline and candidate targets were isolated at
`/home/avifenesh/.cache/memra-targets/cx-fa3softmax-base-v0812` and
`/home/avifenesh/.cache/memra-targets/cx-fa3softmax-candidate`.
The candidate server SHA-256 is
`ba3cad87d91a59ceb4ab2db5570b682a564ff5081e69fc6e247ec984c14c6728`;
the separately built baseline server SHA-256 is
`9b6ee7d863d82ce7ea1c18dec3cd0554cce283a6052c8b825de4ba077d3e6881`.

PTXAS exposes costs that the end-to-end result does not isolate individually:

- head-dim-256 `fa_prefill_qw_db` stayed at 255 registers but gained a 24-byte stack frame;
- head-dim-128 `fa_prefill_qw_db` rose from 230 to 246 registers, and its windowed twin from
  233 to 248 registers;
- dynamic shared allocation gained one 4 KiB P stage plus 24 bytes of control state; and
- every tile gained named-barrier, atomic publication, and cohort synchronization work.

The measured regression is the verdict. The resource changes are plausible contributors, not a
claim that any single one was proven causal.

## Exactness

No numeric-class change was observed.

| Gate | Result |
|---|---|
| Required manifests: `kernel-check-27b.cells` + `kernel-check-step35.cells` | `ALL GREEN (106 cells, 1 skipped)` |
| Q27 full model manifest | `ALL GREEN (107 cells, 3 skipped)` |
| Q35 full model manifest | `ALL GREEN (113 cells, 1 skipped)` |
| Q27 / Q35 `run-gen` | prefill/decode `MATCH`; batched-prime/tokenwise `MATCH` |
| Q27 / Q35 `run-spec` | K=1..8: eight `self-consistency: PASS`; terminal PASS |
| Q27 / Q35 chunk invariance | T=97 and T=149, chunks 2048/64/32: logits `EXACT`, stream identical |
| Frozen 4,860-token serving outputs | one text hash per model across all ten baseline/candidate requests |

The registered `tickinv35` cell is specific to the Step-3.7-Flash SWA dispatch and artifact; it
does not exercise either Q27 or Q35's `fa_prefill_qw_db` path, so it is not presented as a pass
for this lane. The directly applicable windowed and unwindowed DB bit-identity cells ran in the
required/full kernel manifests.

Raw gates: [`raw/candidate/gates/`](raw/candidate/gates/). Candidate resource output:
[`raw/candidate/flash-resource-usage.log`](raw/candidate/flash-resource-usage.log).

## Interleaved measurement

One `flock /tmp/memra-5090.lock` covered the complete 20-run window. Every arm checked
`nvidia-smi --query-compute-apps` before starting and found no competing process. Q27 alternated
baseline/candidate; Q35 alternated candidate/baseline to balance the leading arm. Each model/arm
has N=5. Every scored request used the frozen 4,860 token ids, a fresh server and cache namespace,
60 completion tokens, temperature 0, seed 3407, and reported `cached_tokens=0`.

`prime_ms` is the server's request-level prefill span. Prefill rate is mechanically
`4860 / prime_seconds`. Cold TTFT is the client's time to first non-empty streamed content.
Values are min / median / max.

| Model | Arm | Prime ms | Prefill tok/s | Cold TTFT ms |
|---|---|---:|---:|---:|
| Q27 | baseline | 5621.469 / **5622.928** / 5627.221 | 863.659 / **864.318** / 864.543 | 5629.248 / **5630.393** / 5644.523 |
| Q27 | candidate | 5673.743 / **5674.241** / 5674.600 | 856.448 / **856.502** / 856.577 | 5680.997 / **5681.278** / 5683.569 |
| Q35 | baseline | 1460.560 / **1460.635** / 1461.001 | 3326.486 / **3327.320** / 3327.491 | 1467.123 / **1467.416** / 1467.872 |
| Q35 | candidate | 1482.555 / **1483.520** / 1484.591 | 3273.629 / **3275.992** / 3278.125 | 1489.500 / **1490.079** / 1491.506 |

Paired median candidate deltas:

| Model | Prime time | Prefill rate | Cold TTFT |
|---|---:|---:|---:|
| Q27 | **+0.913%** | **-0.904%** | **+0.899%** |
| Q35 | **+1.541%** | **-1.518%** | **+1.544%** |

The Q27 median prefill-rate gap is 7.816 tok/s; the complete baseline and candidate ranges are
0.884 and 0.129 tok/s. Its median cold-TTFT gap is 50.885 ms versus per-arm ranges of 15.274 and
2.573 ms. The Q35 gaps are 51.328 tok/s and 22.663 ms versus ranges of 1.004/4.496 tok/s and
0.749/2.005 ms. The deltas therefore do not fit inside run-to-run spread.

Telemetry sampled every 100 ms. Active clocks stayed within 967--1200 MHz, the observed
temperature range was 52--63 C, and no clock or power setting was changed.

Derived summary: [`measurement-summary.json`](measurement-summary.json). Raw per-request rows,
server TTFT traces, client responses, pre/post compute-app checks, and thermal telemetry:
[`raw/measurement/`](raw/measurement/). Full driver log:
[`raw/measurement-driver.log`](raw/measurement-driver.log).

## Final disposition

**NO-GO for this two-cohort explicit-barrier formulation.** The profile-backed opportunity is
real, and the implementation preserved exactness, but its synchronization and resource costs
more than consume the overlap on both served shapes. No kernel change, flag, dispatch arm, board
edit, merge, tag, push, or live-server action is retained from this lane.
