# Qwen3.5-9B serving record, issue #543

All five scoped serving cells passed on 2026-10-02. The record covers the actual
Memra HTTP endpoint and the pinned Qwen3.5-9B NVFP4 GGUF on one RTX 5090 Laptop.
It leaves model support at NativeReference. It does not qualify another artifact,
hardware class, full checkpoint context or numerical program.

## Exact scope

- Runtime source: `2873dd4ca37faa15cd4261e7b398926cced30106`.
- Server SHA256: `eaa0afa71439143ae6c05455d85dd42097053b91dc5fb4e520e639b91d143529`.
- Artifact: `Qwen3.5-9B-NVFP4-MTP-GGUF.gguf`, 5,657,607,424 bytes,
  SHA256 `52c9cceb190055e0591a9a30c21f7200572eaf3ff1c59f6e9a1eda838a8f39de`.
- GPU: NVIDIA GeForce RTX 5090 Laptop GPU, 24,463 MiB, driver 595.91.07.
  CUDA build toolkit: 13.1; native target: `sm_120a`.
- Context envelope: 8192. Short prompt: 241 tokens at offered concurrency 1 and 4.
  Long prompt: 6261 tokens at concurrency 1. Every complete generation has a
  64-token budget. Prefix cache: 2048 MiB; host cache: disabled.
- Requests explicitly use temperature 0 and seed 7. Serving MTP/PMIN defaults
  are unchanged. These are bounded decoded-text tapes, not completed-answer or
  distribution-quality measurements.

The collector used two fresh server processes: cache enabled, then the paired
cache-disabled red arm. Independent cold cells use separate cache salts. A cold
cell means a prefix miss, not a cold CUDA process. The cache and cancellation
cells intentionally preserve state between their control and repeat requests.

## Independently asserted results

| Cell | Observed result | Engagement evidence |
|---|---|---|
| Streaming | 66 JSON frames, one final finish, consistent usage and terminal DONE; decoded bytes, token counts and finish match blocking control | [stream.sse](receipts/serving-v1/cache-on/stream.sse), [streaming.json](receipts/serving-v1/streaming.json) |
| Cache | Cold miss; warm hit restores exactly 224 of 241 prompt tokens; decoded bytes/counts/finish match | [cache.json](receipts/serving-v1/cache.json), server `spec restore` log |
| Concurrency | Four overlapping clients each reproduce their serial control | [concurrency.json](receipts/serving-v1/concurrency.json); server logs `K=0 source=concurrency`, `wave=4`, `active=1..4` |
| Cancellation | Client closes after eight content-bearing frames at 170 ms; worker logs abort at 32 generated tokens and 0.29 s; warm recovery restores 224 tokens and matches control | [cancellation.json](receipts/serving-v1/cancellation.json), actual worker abort log |
| Long context | 6261 prompt tokens, 64 generated; warm hit restores 6240 tokens and matches cold decoded bytes/counts/finish | [long_context.json](receipts/serving-v1/long_context.json); cold K=3 and restored K=2 engagement logs |

The [server log](receipts/serving-v1/cache-on/server.log) records those paths.
Serial short requests use MTP K=3; the concurrent wave uses the default plain
route. This proves the observed admission policy and output agreement across
these requests. It does not prove a four-wide GPU kernel or a mid-request change
of numerical program.

There were 19 completion requests: 18 completed at their 64-token budget and one
deliberate disconnect. No failed request was removed from an aggregate. All five
cell JSON files and every referenced raw-file hash were verified after the run.
The 250 ms telemetry recorded a peak of 12,415 MiB GPU memory over 118 samples.
This is one functional cell, not a performance comparison. The GPU job completed
in 29.606 seconds; cleanup recorded no remaining compute process.

## Red evidence and qualification boundary

The [red-arm record](receipts/serving-v1/red-arms.json) contains four refusals:
missing SSE DONE, malformed stub HTTP completion, invalid usage accounting, and
the real cache-disabled server. The disabled arm returned zero cached tokens and
was rejected by the same cache assertion that passed the enabled arm.

Current inspection passed Config, TokenizerTemplate and TensorCensus. The strict
serve verifier refused the absent checkpoint-parity receipt, with the actual
[error retained](receipts/strict-serve-refusal.log). The inspection directory
still marks TinyParity, CheckpointParity, RewriteParity and Serve pending.

The [five-cell TSV](receipts/serving-v1/qualification.tsv) passes `memra model
qualify` as an internally consistent, hash-backed NativeReference record. This
does not satisfy the separate checkpoint and strict runtime/rewrite prerequisites.
No support registry, runtime source, default, tolerance or release gate changed.
Issue #543 remains open; this completes its scoped real-endpoint evidence task.

## Reproduction and checks

Build the current server and CLI first, then use
`tools/collect-serving-qualification.py` with the pinned artifact, its current
`model inspect` directory, an unused loopback port, and the inherited canonical
GPU lease via `--external-lock`. The exact bounded command is retained in
[job.sh](receipts/gpu-job/job.sh), with the pre-run
[preregistration](PREREGISTRATION.md).

The seven new collector CPU tests passed, including preservation of partial SSE
on a transport failure. They validate the collector, not the model. Native build
output, binary hashes and resource limits are in [build receipts](receipts/build/).
The public queue output omits only private systemd unit/invocation identifiers;
native output and resource summaries are retained.
