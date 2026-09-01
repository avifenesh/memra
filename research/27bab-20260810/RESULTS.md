# Qwen3.6-27B beside Step-3.7 on the Vast pair

Date: 2026-08-10

Rig: 2x RTX PRO 6000 WS Max-Q, CUDA 13.1

Runtime: `dc77de733c1615da8a3c93788ee221032ec3fd2d`

## Verdict

**Not viable as simultaneously active service within the suggested Step QoS bound.** Keeping both
models resident is fine, but steady c=2 Qwen3.6-27B traffic raises Step short-TTFT p50 from 188.4 ms
to 734.9 ms (**+290.2%**) and cuts Step c=1 decode from 73.74 to 23.12 tok/s (**-68.7%**). This is
well outside the suggested 5–8% TTFT limit.

The useful deployment shape is **resident standby with traffic arbitration**: leave Q27 loaded for
fast availability, but admit its work only during Step-idle windows. If both must generate at the
same time, move Q27 to another GPU pair or host. Merely keeping Q27 resident changed Step p50 by
-1.9% and decode by +2.1%, so residency itself is not the problem; concurrent compute is.

## Decision measurements

All table rows are interleaved N=5 medians. “p99” is the N=5 nearest-rank value, therefore the
maximum observation rather than a population tail estimate.

| Step arm | Short TTFT p50 | Short TTFT p99 | c=1 decode | Delta from Step alone |
|---|---:|---:|---:|---:|
| Step alone, host bounce | 188.4 ms | 257.8 ms | 73.74 tok/s | baseline |
| Both resident, Q27 idle | 184.8 ms | 185.5 ms | 75.31 tok/s | -1.9% p50, +2.1% decode |
| Both resident, Q27 steady c=2 | 734.9 ms | 754.9 ms | 23.12 tok/s | **+290.2% p50, -68.7% decode** |

Against the tighter both-resident idle control, active Q27 traffic regressed Step p50 by 297.7%,
nearest-rank p99 by 306.8%, and decode by 69.3%. The result is not explained by the first-load
outlier in the standalone baseline.

Step's 4,107-token TTFT was 6.791 s alone and 8.130 s while the overlapping Q27 request generated,
a 19.7% regression. Every reverse arm has a raw host-monotonic interval assertion proving the two
requests overlapped.

## Qwen3.6-27B listing numbers

Q27 ran on physical card 0 with the NVFP4+MTP trunk, own-trim Q4-block drafter, context 32,768,
speculative serving on, and K=3.

| Arm | Short TTFT p50 / p99 | Decode or aggregate throughput | Weighted acceptance |
|---|---:|---:|---:|
| Q27 alone, c=1 | 173.5 / 361.3 ms | 169.21 tok/s | 72.43% (880/1,215) |
| Q27 alone, c=4 | — | 151.49 aggregate tok/s | 72.33% (3,515/4,860) |
| Both resident, Step idle, c=1 | 173.6 / 174.3 ms | 168.83 tok/s | — |
| Both active, steady Q27 c=2 | — | 117.58 aggregate tok/s | 78.01% (6,225/7,980) |
| Q27 c=1 under Step 4k prime | 368.7 / 369.6 ms | 74.10 tok/s | — |

An idle resident Step was neutral to Q27 (+0.09% TTFT, -0.22% decode versus Q27 alone). An active
Step 4k prime was not: Q27 TTFT rose 112.3% and decode fell 56.1% versus its both-resident idle
control. Contention is severe in both directions.

The standalone TTFT p99 is the cold first request at 361.3 ms; the other four observations were
173.2–175.3 ms.

## VRAM ledger

These are `nvidia-smi` values with both servers resident. “Before” follows initial content checks;
“after” follows the complete forward/reverse campaign and final checks.

| Physical GPU | Before used / free | After used / free | Role |
|---|---:|---:|---|
| 0 | 62,336 / 34,913 MiB | 71,172 / **26,077 MiB** | Step PP stage plus Q27 |
| 1 | 56,339 / 40,910 MiB | 57,621 / **39,628 MiB** | Step PP stage |

The final Q27 metrics reported 16 speculative-pool entries, zero OOM parks, and 27.34 GB
driver-free. The scored c=2 load used exactly two cache namespaces and reported zero cached prompt
tokens, so its throughput is not a prompt-cache result.

There is a separate capacity warning. A deliberately faithful but wrong first traffic generator
created a fresh `cache_salt` namespace on every turn. After the five forward pairs it reached 76
spec-pool entries and failed the next Q27 sanity request with the captured error
`DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")`; three OOM parks were recorded. Reusing
`session_id` alone did not help because memra's cache and spec pools key on `(model, cache_salt)`.
The two excluded raw receipts preserve both reproductions. A public/multi-tenant deployment must
still use distinct salts for isolation, so bound namespace/session fanout at the gateway and do not
treat this pair as an unbounded Q27 tenancy host.

## Recommended co-resident configuration

Keep the measured launch shape:

- Step on `:8002`, PP-2 devices `0,1`, context 262,144, grouped MoE, prefill tick 2,048, and
  **`MEMRA_PP_HOST_BOUNCE=1`**.
- Q27 on `:8003`, `CUDA_VISIBLE_DEVICES=0`, context 32,768, prefix cache 0 MB, speculative serving
  on at K=3, with `Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf` plus
  `draft-daily-owntrim-nvfp4head-q4blk.gguf`.
- At the gateway, queue new Q27 work whenever Step has active or queued interactive work. Do not
  claim simultaneous QoS safety from this configuration. A request already generating on Q27
  cannot be assumed to yield across the two independent server processes.
- Preserve per-user/session `cache_salt` isolation, but place an explicit bound on live/parked Q27
  namespaces. The measured safe shape is two persistent Q27 namespaces; this campaign did not
  establish a higher safe bound.

## Economy line

At continuous c=2 load, Q27 produced 117.58 output tok/s, or **10.16 million output tokens/day**.
Using Google's current standard `gemini-3.1-flash-lite` output price of **$1.50/M tokens** as a
flash-class reference, that is **$15.24/day** of output-token API-equivalent capacity. The current
GA `gemini-3.5-flash-lite` $2.50/M reference would be $25.40/day. Pricing source:
[Google Gemini Developer API pricing](https://ai.google.dev/gemini-api/docs/pricing).

This is throughput-valued, not quality-adjusted, and the continuous figure is incompatible with
the Step QoS requirement. Under the recommended Step-idle scheduler, multiply both 10.16M and
$15.24 by the fraction of the day available to Q27.

## Correctness and evidence

- Scored receipts: 121 clean summary files across A/B/C, 101 settled-counter checks, 64 exact
  known-output hash checks, 15 cross-process overlap assertions, and four empty runtime-failure
  scans. There were zero scored request errors and zero BOS-garbage outputs.
- The Step environment receipt proves `MEMRA_PP_HOST_BOUNCE=1` and PP-2. The release binary SHA-256
  was `de9c201d27993275b1448f778e6942bbaca5e902864e867090366c5fe13087e8`.
- Raw output, request JSONL, server logs, GPU samples, environment receipts, and snapshots are under
  [`raw/A/`](raw/A/), [`raw/B/`](raw/B/), and [`raw/C/`](raw/C/). Excluded diagnostic receipts are
  retained beside them and explained in [`PROGRESS.md`](PROGRESS.md).
- Machine-readable derived values are in [`summary.json`](summary.json); rerun with
  `python3 research/27bab-20260810/summarize.py research/27bab-20260810/raw --out
  research/27bab-20260810/summary.json`.
- Thermal regime: unlocked Max-Q power/clock control with one-second sampling. During C, 244 samples
  per card observed maxima of 59/49 C and 305.5/303.4 W on cards 0/1. Every published median states
  N=5; no single run is presented as a median.

This is lane-local research evidence, not a published perf-board move; no generated README or
`docs/PERFORMANCE.md` numbers were changed.

## Live handoff

At 2026-08-10T17:39:50Z, Step PID 69696 was ready on `:8002`, Q27 PID 69935 was ready on `:8003`,
and `/root/soak.py` PID 70049 had completed nine fresh Step iterations with 97 chunks each and no
errors. Both server failure scans were empty and both reported zero OOM parks. The exact environment,
process, port, GPU, readiness, content-hash, and soak receipts are in [`raw/final/`](raw/final/).
