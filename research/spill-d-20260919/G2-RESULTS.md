# G2 — pinned/pageable copy envelope on one target-class card

**Measured result:** cacheable pinned memory improves H2D and D2H wall-copy
throughput at 64 KiB through 256 MiB in this campaign. At 4 KiB, pinned H2D is
slower (11.778 vs 11.043 us/copy), while pinned D2H is faster. This is a
size/direction-dependent copy-envelope result, not a universal pinning default.
No engine code, numerical program, runtime flag/default, or performance board changed.

## Scope and immutable evidence

One RTX PRO 6000 Blackwell, 96 GB, 600/600 W; development evidence on the target
card class, **not multi-rig, full G2 size-ladder, storage, serving or release qualification**.
Requested five-size matrix is complete; the probe's other five registered sizes
were not part of this campaign. No timing comparisons with prior-rig receipts.

- Preregistered protocol/worker: `44840aed3` ([G2-PROTOCOL.md](G2-PROTOCOL.md)).
- Native probe build source: `9f849978bac72cf2a4b608873a49341eafb7ab56`;
  engine/probe source unchanged through runner revision. Native build exit 0.
- Binary, probe source, collector, worker and protocol hashes:
  `rented-pro6000-20260920/g2/visits/identity.json`.
- Whole collector window: 269.149 seconds; calibration + scoring held one
  `/tmp/memra-gpu.lock` after bounded waiting. Inherited lock proof retained.
- **N=10 visits per size/direction/arm, 5 AB + 5 BA**, 200 scored samples.
  Fifty-two calibration samples excluded. Every scored arm >=497.164 ms,
  above the preregistered 250 ms floor. Copies/visit do not increase N.
- Raw samples: `rented-pro6000-20260920/g2/visits/samples.jsonl`; per-invocation
  JSONL `.log` files adjacent, with byte/hash controls and raw hashes.
- Capture/telemetry: `rented-pro6000-20260920/g2/collector/`; transfer manifest
  `rented-pro6000-20260920/g2-manifest.json`: **140 files hash-matched**.
- Replay: `python3 research/spill-d-20260919/summarize-g2.py
  research/spill-d-20260919/rented-pro6000-20260920/g2` checks every raw sample,
  matrix/order/N, counts, byte equality, 600/600 W, telemetry and capture integrity.
  Machine-readable result: `rented-pro6000-20260920/G2-SUMMARY.json`.

## Thermal regime T1 (all rows)

Sequential calibration-warmed visits; **no steady-state thermal soak**. Whole
campaign telemetry: 1,074 samples, target/median 250 ms, maximum gap 271 ms;
GPU 37–41 C, SM clocks 180–2355 MHz, memory clocks 405–12481 MHz, power draw
36.01–105.75 W, power limit/max 600/600 W throughout, link x16 with generation
1–5 transitions. These ranges include calibration, setup and idle gaps; they
are not a claim of fixed clocks during every copy. No competing compute process
was present in the between-invocation inventories; all arms stayed in one lock.

## Medians (N=10 each arm, T1 regime)

Wall us/copy includes host submission, per-copy stream fence and CUDA event
instrumentation. Effective GiB/s is median completed bytes / wall interval;
it is **not DMA-only bandwidth**. Allocation, setup and verification are excluded.

| Size | Direction | Pageable us/copy | Pinned us/copy | Pageable GiB/s | Pinned GiB/s | N/arm | Regime |
|---|---|---:|---:|---:|---:|---:|---|
| 4 KiB | H2D | 11.043 | 11.778 | 0.345 | 0.324 | 10 (5 AB+5 BA) | T1 |
| 4 KiB | D2H | 12.403 | 10.970 | 0.308 | 0.348 | 10 (5 AB+5 BA) | T1 |
| 64 KiB | H2D | 15.443 | 12.853 | 3.952 | 4.749 | 10 (5 AB+5 BA) | T1 |
| 64 KiB | D2H | 15.256 | 12.043 | 4.001 | 5.068 | 10 (5 AB+5 BA) | T1 |
| 1 MiB | H2D | 55.884 | 29.885 | 17.475 | 32.678 | 10 (5 AB+5 BA) | T1 |
| 1 MiB | D2H | 53.361 | 29.889 | 18.301 | 32.673 | 10 (5 AB+5 BA) | T1 |
| 16 MiB | H2D | 356.858 | 301.602 | 43.785 | 51.807 | 10 (5 AB+5 BA) | T1 |
| 16 MiB | D2H | 441.990 | 307.802 | 35.352 | 50.763 | 10 (5 AB+5 BA) | T1 |
| 256 MiB | H2D | 6802.038 | 4648.168 | 36.754 | 53.785 | 10 (5 AB+5 BA) | T1 |
| 256 MiB | D2H | 6653.570 | 4756.363 | 37.574 | 52.561 | 10 (5 AB+5 BA) | T1 |

Frozen copies/visit by ascending size: **45,597 / 41,839 / 16,742 / 1,658 / 108**.

Every native probe retains its original `n1-plumbing-not-qualified` label. The
outer protocol supplies the ten observations, not a relabeling of one visit.
The collector and summary retain `qualification: false`: copy-envelope research
does not imply full tiering, model exactness, serving, or release qualification.
