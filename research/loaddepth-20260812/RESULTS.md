# Naked dual-default deep-load curve — box1

Date: 2026-08-12

Lane: `lane/cx-loaddepth`

Rig: box1, 2x RTX PRO 6000 Blackwell Server Edition

Scored source: `0edc57b9c98720067686cbd60905073d545154b8`

## Verdict

**Use c=16 for this serve shape.** It is the revenue-optimal measured width at
**171.419 aggregate tok/s**, or **14,810,613 tokens/day** at continuous
utilization. That is +9.364% over this campaign's c8 median and +8.449% over the
flip battery's historical c8 anchor of 158.065 tok/s.

The first throughput knee is **c=20**: the N=3 median falls from 171.419 tok/s
at c16 to 164.911 at c20 (-3.796%). c24 recovers to 169.005, but remains 1.408%
below c16. The curve is therefore non-monotonic; sustained depth beyond c16
does not provide the hypothesized 30--60% revenue increase on this shape.

The 15-second tail bar does not bind in the scored curve. Median window p99
TTFT reaches only 1.523 seconds at c24, so the knee verdict is throughput-led.

## Exactness first

The frozen `qos_probe` ran in ascending order on one naked-default server before
any timing point. All 72 responses matched golden SHA-256
`21b8293f2298978c74fb89f32d9b14e3ea921f39924cfce88c73b01f445bb6de`.

| Width | Golden matches | Errors | Divergences | TTFT p99 | Latency p99 |
|---:|---:|---:|---:|---:|---:|
| c12 | 12/12 | 0 | 0 | 7.486 s | 9.418 s |
| c16 | 16/16 | 0 | 0 | 9.646 s | 11.450 s |
| c20 | 20/20 | 0 | 0 | 12.631 s | 14.341 s |
| c24 | 24/24 | 0 | 0 | 15.619 s | 17.324 s |

The exactness probe uses the longer frozen golden prompt and mixed
interactive/judge/harvest lanes. Its c24 TTFT is not substituted into the
revenue curve, whose frozen flip-shaped request is reported below.

## Sustained 128-token curve

Every row is the median of **N=3 interleaved scored windows**; the parenthesized
range contains all three aggregate measurements. Orders were forward, reverse,
and rotated. Each point used a fresh naked server, a discarded same-width
warmup, temperature zero, 128 output tokens per request, and the flip battery's
`Count upward from N` prompt family. Streaming was enabled only to timestamp the
first content-bearing token. Aggregate throughput spans simultaneous barrier
release through final request drain.

Worker step p99 is the server's 30-second engine-truth window. Because every
point used a fresh process and same-width warmup, that window contains only the
reported width. Admission counters are totals across the three scored windows.

| Width | Aggregate tok/s | vs measured c8 | TTFT p50 | TTFT p99 | Step p99 | Admitted / completed | Session / VRAM defers | OOM parks |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| c8 | 156.741 (156.372--157.554) | baseline | 0.475 s | 0.679 s | 62.788 ms | 24 / 24 | 0 / 0 | 0 |
| c10 | 162.593 (162.180--162.665) | +3.733% | 0.520 s | 0.732 s | 76.822 ms | 30 / 30 | 0 / 0 | 0 |
| c12 | 166.557 (166.446--167.900) | +6.263% | 0.520 s | 0.820 s | 90.648 ms | 36 / 36 | 0 / 0 | 0 |
| c16 | **171.419 (170.900--172.488)** | **+9.364%** | 0.609 s | 1.079 s | 117.997 ms | 48 / 48 | 0 / 0 | 0 |
| c20 | 164.911 (164.426--165.168) | +5.213% | 0.938 s | 1.276 s | 152.692 ms | 60 / 60 | 0 / 0 | 0 |
| c24 | 169.005 (167.488--169.514) | +7.824% | 1.521 s | 1.523 s | 180.548 ms | 72 / 72 | 0 / 0 | 0 |

The c8 median is 0.837% below the historical 158.065 flip-battery point. No
formal anchor tolerance was frozen, but the close landing and the narrow N=3
range provide a direct continuity check. Across the full scored sweep,
270/270 requests finished at exactly 128 tokens and worker accounting matched
all **34,560** completion tokens. Sampled queue depth was zero at every width;
dual metrics recorded equal slot use and zero collisions at every point.

## Thermal regime and provenance

The scored run held `/tmp/memra-gpu.lock` continuously from
2026-08-11T22:38:39Z through `LOADDEPTH_ALL_PASS` at 22:59:53Z. There was no
artificial cooldown. Continuous 250 ms sampling retained 4,898 two-GPU
intervals: 27--50 C, 180--2,422 MHz SM clocks, peak power 336.91 W, and peak
used memory 56,691 MiB. The low clock/zero-memory endpoints include the fresh
server boot and teardown periods by design.

The checkout is a descendant of the merged default commit `e94699eba`; its two
later commits contain only the flip battery evidence and documentation. The
scored release server SHA-256 is
`98eed18d98b83be7ebdb34551d7763abb7d3d54cef315db6d6669e33b1f7cdbf`.
The model first-shard, draft, and golden hashes are respectively `b940497a...`,
`469a8166...`, and `21b8293f...`; complete paths and hashes are in
`raw/SHA256SUMS`.

## Receipts and exclusion

- `raw/summary.json` is the machine-readable scored curve; local re-reduction
  reproduced its SHA-256 `b35659a3...` byte-for-byte.
- `raw/MANIFEST.sha256` verifies the 174 scored payload files (175 files with
  the manifest itself). `raw/driver.log` records
  the one lock acquisition, 72-request exactness stage, all 18 score points,
  and the final pass marker. `raw/gpu.csv` is the continuous thermal trace.
- `raw/exactness/` retains every golden row, summary, exit receipt, server log,
  and final dual metrics. `raw/perf/*/{warmup,score}.jsonl` retains every
  request, 250 ms metrics samples, full before/after worker snapshots, and the
  window summary; server and point thermal logs sit beside them.
- `raw-aborted-user-manager-stop-20260811T223712Z/` is excluded preflight
  evidence. The transient user service was externally stopped with exit 143
  during input hashing, before any `SERVER_START`; it produced no measurement
  row and was never combined with the scored run.

No runtime code, performance board, merge, tag, push, or formatting surface was
changed. This measurement branch is intentionally left for the orchestrator.
