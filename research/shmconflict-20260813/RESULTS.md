# Cached-prefill shared-memory swizzle result

Date: 2026-08-13

Verdict: **GO at the local RTX 5090 iteration gate; Vast 2x RTX PRO 6000 confirmation required
before merge or default promotion.**

The retained XOR layout removes 99.13% of `fa_prefill_qw_db`'s excessive shared-memory
wavefronts and improves frozen-request prefill and cold TTFT on both served models. Every paired
timing repetition wins, the median gaps are outside per-arm spread, and all exactness gates stay
green. This is a local, relative result under the owner-imposed 210--1200 MHz clock cap; it is not
an absolute-throughput or target-rig claim.

## What was fixed

Per-instruction NCU/SASS attribution found conflicts in every row-major operand:

| Shared operand | Baseline service | Share of total excess | Retained layout change |
|---|---:|---:|---|
| Q `ldmatrix` | 8.00x ideal | 0.17--0.37% | XOR each 16-byte chunk column with `row & 7` |
| K `ldmatrix` | 8.00x ideal | 48.51--48.61% | same `row & 7` permutation |
| P stores and reloads | 4.00x ideal | 2.60% combined | four-chunk `row & 3` permutation |
| V transposed `ldmatrix` | 8.00x ideal | 48.51--48.61% | same `row & 7` permutation |

K and V alone account for 97.03--97.22% of baseline excess. Full PC ranges, access layouts, and
source/SASS mapping are in [`ATTRIBUTION.md`](ATTRIBUTION.md); every extracted counter is retained
in `raw/baseline-shared-pcs.csv` and the per-session source/SASS tables.

Candidate v1 enables the permutation only in the cached/chunk `fa_prefill_qw_db` body. Stores and
matching `ldmatrix` addresses use the same inverse mapping. Tile geometry, shared allocation,
values, MMA order, online-softmax recurrence, and accumulation order are unchanged. The default
template arms used by other kernels remain unswizzled.

## Direct mechanism proof

The frozen serving request reaches two captured target shapes per model. Values are aggregate
shared wavefronts divided by ideal wavefronts.

| Model / shape | Baseline | Candidate v1 | Excess reduction |
|---|---:|---:|---:|
| Q27 / 4,096 | 6.424672x | **1.046995x** | 99.133680% |
| Q27 / 764 | 6.452196x | **1.047378x** | 99.131036% |
| Q35 / 4,096 | 6.424672x | **1.046995x** | 99.133680% |
| Q35 / 764 | 6.452196x | **1.047378x** | 99.131036% |

Q, K, and V reach exactly 1.00x ideal. P stores and reloads retain a 2.00x service ratio and
account for almost all of the aggregate 1.047x remainder. The complete before/after table and
every classified PC are in `raw/shared-attribution-before-after.csv` and
`raw/shared-pcs-before-after.csv`; [`PROFILE-CANDIDATE.md`](PROFILE-CANDIDATE.md) records the
profile setup and artifact boundary.

This is a layout win, not a capacity win. Dynamic shared memory and NCU's 8.33% theoretical
occupancy remain unchanged, so there is no second-CTA claim. Candidate v1 remains at 255 registers
and adds an 8-byte stack frame, two `STL`, and two `LDL` instructions. NCU reports local spill
requests on all four captures. A source-liveness-only v2 moved rather than removed that slot and
was rejected statically. Despite this cost, the v1 profiled kernel duration fell 32--34% in the
single NCU captures; those profiler durations are mechanism context, not the scored timing result.

## Exactness

No numeric-class change was observed.

| Gate | Result |
|---|---|
| Required manifests: `kernel-check-27b.cells` + `kernel-check-step35.cells` | `ALL GREEN (106 cells, 1 skipped)` |
| Q27 full model manifest | `ALL GREEN (107 cells, 3 skipped)` |
| Q35 full model manifest | `ALL GREEN (113 cells, 1 skipped)` |
| Q27 / Q35 `run-gen` | prefill/decode `MATCH`; batched-prime/tokenwise `MATCH` |
| Q27 / Q35 `run-spec` | K=1..8: eight per-K passes and terminal `SELF-CONSISTENCY PASS` |
| Q27 / Q35 chunk invariance | T=97 and T=149; chunks 64/32 are exact to 2048; streams identical |
| Frozen 4,860-token serving output | one text hash per model across all baseline/candidate requests |

The full battery ran under one uninterrupted GPU lock after hash-matching the source, candidate
binaries, models, drafts, and prompts. [`EXACTNESS.md`](EXACTNESS.md) summarizes it; raw logs and a
verified manifest are in [`raw/candidate-v1/gates/`](raw/candidate-v1/gates/).

## Interleaved serving measurement

One `/tmp/memra-5090.lock` hold covered the complete 20-arm window. Q27 alternated
baseline/candidate; Q35 alternated candidate/baseline to reverse the leading arm. Each model/arm
has N=5. Every scored arm used a fresh server process and cache namespace, the frozen 4,860 token
ids, 60 completion tokens, temperature 0, seed 3407, and reported `cached_tokens=0`.

`prime_ms` is the server's request-level prefill span. Prefill rate is mechanically
`4860 / prime_seconds`. Cold TTFT is client time to first non-empty streamed content. Values are
min / **median** / max.

| Model | Arm | Prime ms | Prefill tok/s | Cold TTFT ms |
|---|---|---:|---:|---:|
| Q27 | baseline | 5621.725 / **5622.566** / 5623.781 | 864.187 / **864.374** / 864.503 | 5628.530 / **5629.445** / 5630.636 |
| Q27 | candidate | 5529.286 / **5530.645** / 5538.882 | 877.433 / **878.740** / 878.956 | 5537.454 / **5537.739** / 5545.599 |
| Q35 | baseline | 1459.866 / **1460.351** / 1460.768 | 3327.017 / **3327.967** / 3329.073 | 1466.317 / **1466.786** / 1467.459 |
| Q35 | candidate | 1419.207 / **1419.729** / 1419.838 | 3422.926 / **3423.189** / 3424.448 | 1425.769 / **1426.312** / 1428.306 |

Paired median candidate deltas:

| Model | Prime time | Prefill rate | Cold TTFT |
|---|---:|---:|---:|
| Q27 | **-1.652%** | **+1.680%** | **-1.617%** |
| Q35 | **-2.788%** | **+2.868%** | **-2.747%** |

Candidate v1 wins all ten model/repetition pairs. Q27's median prime gap is 91.921 ms versus
baseline/candidate ranges of 2.056/9.596 ms; its prefill-rate gap is 14.366 tok/s versus ranges of
0.316/1.523 tok/s. Q35's median prime gap is 40.622 ms versus ranges of 0.902/0.631 ms; its
prefill-rate gap is 95.221 tok/s versus ranges of 2.056/1.522 tok/s. These deltas do not fit inside
run-to-run spread.

Active telemetry samples (`utilization.gpu > 0`) observed 960--1200 MHz and 53--64 C. No clock or
power setting was changed. Derived values are in
[`measurement-summary.json`](measurement-summary.json); raw requests, server TTFT traces,
responses, pre/post compute-app checks, 100 ms telemetry, and their verified manifest are under
[`raw/measurement/`](raw/measurement/). The tee-first orchestration log is
[`raw/measurement-driver.log`](raw/measurement-driver.log).

## Reproducibility boundary

- Baseline: unmodified `v0.81.3`; server SHA-256
  `4f0205c2cc2b31cdde89a395b41211f2663caa76b333b55ba7a4a4a085e48521`.
- Candidate v1 server SHA-256:
  `73a049376dc31bb5081d4369d2429a37a02dad34952f482f5019df767baa86ec`.
- Retained `flash_attn.cu` SHA-256:
  `b02e951ebb44aac43220204deac1c88fbd0706131dc965c966a5b1564577ad4b`.
- The successful candidate build used disk-backed `TMPDIR=/home/avifenesh/tmp-lanes`. Earlier
  failures in unchanged `cu/hybrid.cu` occurred during the independently verified `/tmp` tmpfs
  exhaustion and are not candidate evidence.
- NCU report files remain outside the repository. Only tee-first logs, exported CSVs, extracted
  SASS, hashes, and derived tables are committed.

## Final disposition and required follow-up

**GO for candidate v1 at the local development-iteration gate.** The direct conflict metric moves
in the intended direction, both model timings win outside spread, and exactness is bit-preserved.

Before merge, tag, or default promotion, build this exact source on the designated Vast 2x RTX PRO
6000 verification box and run the pre-release battery: `kernel-check` ALL GREEN, Q27/Q35
`run-gen` argmax MATCH, Q27/Q35 `run-spec` K=1..8 self-consistency, same-prompt golden output, and
target-topology memory/throughput measurements. Recheck the 8-byte local spill cost there while
confirming the wavefront reduction transfers. That PRO confirmation is explicitly out of scope
for this lane and has not been run.

No merge, tag, push, generated-board edit, or live-server action was performed.
