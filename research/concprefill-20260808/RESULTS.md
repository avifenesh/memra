# CONCURRENT-PREFILL SATURATION — box1, 2026-08-08

## Anatomy verdict first

**SATURATED.** Four different-prefix 4096-token primes do not scale aggregate
throughput beyond the single-prime class. With four background decode streams,
aggregate prefill is 568 / 575 / 578 tok/s at c=1/2/4. Removing decode load moves
c=4 only to 580 tok/s.

The worker does serialize these prompts: 4096 tokens exceeds both the 2048-token
fresh-batch ceiling and the 1024-token multi-session tick budget, so every request
runs one 1024-token `prefill_tick` call at a time. But that serialization is not
hiding a second independent compute lane. Stage 1 is already near continuously
occupied, and c=4 prime-only has the same wall as c=4 mixed.

The one-call solo warmup reaches 674 tok/s. Treat the 580-to-674 difference as a
bounded segmentation/fill-drain tax, not a concurrency multiplier. Perfectly
recovering it would still leave the pair 4.5x below the earning model's 3K tok/s
floor. The capacity direction is more pairs or a new per-prime compute mechanism,
not more concurrent-prime scheduler engineering.

## Required table

Step-3.7-Flash trial config on box1: trunk + MTP draft, PP-2 devices 0/1,
`MEMRA_MOE_GROUPED=1`, placement-aware spec default, one exclusive
`/tmp/memra-gpu.lock` hold. Every cell is N=3 barrier bursts; TTFT p95 is across
all requests in those bursts.

| load | simultaneous 4k primes | aggregate prefill tok/s median (range) | TTFT p95 |
|---|---:|---:|---:|
| four decode streams live | 1 | **568.2** (565.4-568.4) | 7.245 s |
| four decode streams live | 2 | **575.3** (574.7-576.0) | 14.255 s |
| four decode streams live | 4 | **577.6** (576.8-577.9) | 28.404 s |
| prime-only control | 4 | **580.5** (579.7-580.5) | 28.265 s |

At mixed c=4, the background streams emitted only 0.7 visible tokens/s aggregate
and saw 7.102 s inter-token p95 because each scheduler tick performs about 7.0 s
of serial prefill before the 27-34 ms decode phase. This is a QoS problem under
co-located long primes, but not unused prefill capacity.

## Trace evidence

- Target-prompt `prime_cache_batch` calls: **0**.
- Typical c=4 tick: four serial calls, 4096 prompt tokens, 7.0-7.1 s prefill wall.
- c=4 GPU utilization medians: mixed 78.5% / 86%; prime-only 80.5% / 86%.
- Thermal regime: 26 C at entry, 54 C maximum, 0 MiB on both cards at entry/exit.
- Run health: no OOM, CUDA error, panic, server death, or request error.

The raw build, client, server, GPU, and derived tables are in `raw/box1/`.

## What landed

No concurrent-prime scheduler or engine change landed. The anatomy commit freezes
the saturated verdict before implementation, and the measured ceiling rejects the
proposed scheduler work as a path to the 3K tok/s target. The only code change is
debug-only timing/accounting behind `MEMRA_TICK_TRACE`; serving policy, batching,
chunking, kernels, and arithmetic are unchanged.

## Target-rig battery

The final battery ran on the same box1 pair. Every production gate passed, every
canary broke its intended assertion, both cards returned to 0 MiB, and the flock
was released.

| gate | result |
|---|---|
| `kernel-check` | ALL GREEN |
| `ppsplit` | PASS: unsplit, serial, and pipelined walks bit-identical and live |
| `ppsplit` canary | PASS: disabled overlap failed the liveness assertion |
| `chunkinv35` | PASS: bit-identical at 4096/513/512/256/64 |
| `chunkinv35` canary | PASS: perturbed path broke the assertion |
| `tickinv35` | PASS: bit-identical across budgets 0/1024/513/512/256/64 and splits 64/256/512 |
| `tickinv35` canary | PASS: perturbed path broke the assertion |
| `run-gen` | MATCH: prefill/decode argmax 6776; batched-prime/tokenwise argmax 6776 |
| `run-spec` | SELF-CONSISTENCY PASS, K=1..8 |
| `b2geo35` | PASS: c=2/c=4 byte-identical to c=1 with B>1 evidence |
| `b2geo35` canary | PASS: B=1 re-pin broke the batched-evidence assertion |

Raw gate output is in `raw/box1/gates/`; the combined ledger is
`raw/box1/gates/gates-summary-20260808T172104Z.log`.
