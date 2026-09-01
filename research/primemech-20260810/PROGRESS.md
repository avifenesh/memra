# primemech — concurrent-prime anatomy on the local 5090

Branch: `lane/primemech`
Base: `5e9b13a4`
Rig: local RTX 5090 Laptop, one sm_120a GPU

## Mission and stop line

Locate where N=1/2/4 simultaneous distinct-prefix, prefill-dominated requests
spend their wall time on the single-GPU serving path. Record aggregate prefill
throughput, each request's prime wall time, sampled GPU SM/memory activity, and
the exact code point that serializes work. Rank 2-3 plausible mechanisms for
true concurrent-prime compute, with measured ceilings and effort classes.

This is an anatomy/design lane. It will not implement or promote a mechanism,
change published perf boards, push, merge, tag, run `nsys`, or attempt a model
that does not fit this card.

## Binding receipts read before work

- `research/concprefill-20260808/RESULTS.md`: four distinct 4096-token primes
  saturate the PP-2 pair at 580.5 aggregate tok/s prime-only versus a 674 tok/s
  solo-warmup class. Typical c=4 ticks contain four serial 1024-token calls.
- `research/vast-trial-20260810/perf-receipt.md`: the live trial box confirms
  the felt-speed exposure: c=4 distinct cold 4k primes reach 26.36 s TTFT p50
  versus 6.94 s solo-class TTFT.
- `research/primebatch-20260808/PROGRESS.md`: the existing cross-request concat
  walker batches weight-streaming work while retaining request-local attention
  and KV state, but only improves T=520 B=2/B=4 wall time by 2.5%/2.3%.
- `research/pp2pipe-20260809/{PROGRESS,RESULTS}.md`: pipeline parallelism removes
  the single-prime stage-idle bill on a two-GPU rig; it is context, not evidence
  for a second compute lane on this single-GPU measurement.

## Pre-registered measurement

1. Trace `crates/memra-server/src/worker.rs` through the engine prime entry and
   identify every host lock, scheduler admission boundary, CUDA stream, and
   synchronization edge before running the benchmark.
2. Use one locally present fitting model and a fixed distinct-prefix prompt
   shape. Launch barrier-synchronized N=1,2,4 cold primes through the real
   server. Keep prompt tokens constant across N, make generation minimal, and
   exclude warmups.
3. Run bounded repeats in an interleaved order. Retain client timing, server
   timing/trace, request results, pre/post thermal state, GPU process state, and
   CUPTI-free `nvidia-smi dmon` samples under `raw/`.
4. Derive aggregate prompt-token throughput from the measured request set and
   per-prime wall from request/server timestamps. Use sampled SM/memory activity
   only as interval evidence, with its cadence stated.
5. Compare timing order and trace spans against the code-path hypothesis. Name
   the confirmed serialization mechanism or explicitly mark it unresolved.

Every published median will state N, repeats, and thermal regime. Any died run
without captured stderr will be labeled cause unknown.

## Pre-run source hypothesis

- There is one CUDA-owning worker thread and one primary compute stream
  (`worker.rs:1-8`, `worker.rs:2317-2338`, `memra-runtime/src/lib.rs:83-92`).
- Interactive prefill iterates active sessions in order and calls
  `prefill_tick` synchronously (`worker.rs:3590-3621`). A 4096-token request is
  excluded from the default concat-prime candidate set by the 2048-token cap
  (`worker.rs:3479-3510`).
- `prefill_tick` calls `prime_cache` directly (`worker.rs:5596-5603`). The prime
  epilogue downloads logits (`hybrid_forward.rs:1342-1346`), and `Engine::dtoh`
  synchronizes that same stream before returning (`memra-engine/src/lib.rs:
  3908-3911`). Therefore the worker cannot enter the next session's prime call
  while the current one is in flight.
- This predicts synchronous scheduler/stream serialization, not waiting on a
  top-level engine mutex. A naive multi-threaded engine entry would additionally
  contend on the shared prime-slab guard held by the layer walk
  (`hybrid_forward.rs:1035-1056`).

The production-default arm leaves the naked scheduler policy intact; the
pre-run expectation was solo widening plus concurrent 1024-token ticks. The
control sets only `MEMRA_PREFILL_TICK=8192`. Post-run trace corrected both shape
assumptions: default-on plain affinity arms a pre-generation checkpoint for
chat-template traffic even with `MEMRA_REUSE_POOL=0`, so production remains at
1024 and the control performs one 4125-token call plus seven tokenwise boundary
tail calls per request. Cross-request reuse is still cold (`cached_tokens=0`),
but the checkpoint boundary is part of the production anatomy and excludes the
existing concat-prime path.

The first excluded warmup used arbitrary raw token ids and produced no visible
text within four decode tokens. It stopped cleanly before any measured burst.
The frozen input was corrected to the repo's normal-text `prompt-pp4096.txt`
(4094 Qwen tokens before a short request-specific prefix/suffix), with eight
decode tokens; raw completions still emitted only non-visible special tokens.
The final harness therefore uses the model's real chat template with
`reasoning.enabled=false` and counts either a content or reasoning delta as the
first visible frame. Both shakeouts stopped before any measured burst;
server-reported prompt counts remain authoritative.

## Coordination

`~/.lanectl/inbox/cx-primemech.md` was absent at lane start. At 2026-08-09
23:04:59Z it assigned the now-shared local 5090 the `/tmp/memra-gpu.lock`
convention; every GPU arm in this lane uses one hold on that path. The exact
inbox path will continue to be checked before each bounded work block.

## Status

- [x] Required predecessor receipts read.
- [x] Measurement intent pre-registered.
- [x] Prime path and synchronization edges traced.
- [x] Model and bounded protocol fixed (local Qwen3.5-9B Q8_0; exact binary and
  model hashes retained with the run).
- [x] N=1/2/4 raw runs captured (one interleaved three-repeat A/B block,
  `20260809T231431Z`; both clients exited 0, no fault-pattern matches, and each
  server teardown returned the card to 31 MiB).
- [x] `RESULTS.md` written and receipts verified.

## Final outcome

Production aggregate prefill stays 4519/4489/4480 tok/s at N=1/2/4 while
client TTFT grows 0.924/1.852/3.704 s. At N=4, about 98.9% of burst wall is the
traced prefill phase; the useful 1 Hz GPU row is 100% NVML busy, 85% GPM SM
busy, and 27% theoretical DRAM bandwidth p50. The named edge is the sole worker's
synchronous per-session call into `prime_cache`, followed by logits D2H and a
same-stream synchronization. The 8192-budget control recovers only 1.1% at
N=4 and moves p95 queue delay from 0.585 s to 2.498 s. See `RESULTS.md`.
