# Active C4 host residency, 2026-09-05

Implemented, explicit API only, not a serving default. This extends the parked
prefix tier rather than relabeling it as active offload.

`offload_c4_decode_state` transitions a primed state between closed device
transactions. C4 compressed rows move to cacheable pinned host memory; their
full-capacity GPU allocation is replaced by SWA128 plus the session's transient
rows. C128, compressor pending state, and the indexer's existing f32-QAT keys
remain on device. The initial prefill remains the all-device program.

The native gather copies selected C4 rows once per query into a bounded
per-stage buffer shared by all heads. SWA and transient rows come from the
device allocation. Logical indices, ordering, pads and every f32 bit are
unchanged. Compressor emissions use same-stream asynchronous D2H; subsequent
gathers use the same stream. Rollback changes the live high-water mark, so
rejected rows are inaccessible and are overwritten on re-emission. There is
no approximate top-k, recomputation, weight/KV format substitution or external
runtime dependency. This first implementation has no persistent hot-page LRU;
that optimization requires measured miss/reuse evidence.

Snapshot serializes the same canonical `[SWA, live compressed rows]` as the
device baseline. Restore initially recreates the device representation and may
explicitly offload it again. Host bytes and released device cache bytes are
separate counters. Extra decode gather scratch is exposed by
`c4_device_scratch_bytes`; batched gather scratch is charged to verifier bytes.
CPU reads and host deallocation drain the owning stream. Monolithic prefill on
an already offloaded state is refused; device decode and chunked continuation
are the implemented consumers.

## Evidence and remaining gates

- 398 engine CPU tests pass, 4 GPU-dependent tests ignored in that run.
- Strict clippy passes for the engine library, binaries and tests.
- `active-c4-gather-5090.log` passes on the non-serving RTX 5090 Laptop device:
  query widths 1/6/32, live rows 0/1/40/1025/262144, dead capacity gaps, SWA,
  transient rows, duplicate selection, negative-zero bits, pads, guard regions,
  and asynchronous re-emission into previously read host rows. Test binary:
  `a507d4c9a755978018323e64d5384807be20f6d36ab1f54cdbb4ac5c369817eb`.
- Target model gate `dsv4_c4_host_gate` is staged, not yet passed. Binary:
  `7f15f1c7c7215252e3ec5e04fea0e39cb411f647fa6af02a6a6b5c5692a94ac5`.
  It compares device/host/device logits, nine sampled plain steps, 32 sampled
  speculative outputs and rounds, every live trunk/draft class, suffix lengths
  0/33/129 and canonical snapshot/restore on real 1/160/4097-token prompts.
- Still required: actual target-model PASS, target memory readback, 1M/c1–c16
  admission and fairness, host-RAM bounds, per-phase I/O profiling, persistent
  hot-buffer reuse, and production scheduling/qualification. No throughput or
  concurrency claim follows from the component gate.

## Primary-source checks

Read 2026-09-05:

- [LMSYS V4 integration](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/):
  C4 host extension is distinct from GPU-resident C128 and SWA. Reported B200
  throughput is not transferred to SM120.
- [CUDA unified addressing](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__UNIFIED.html):
  cacheable `cuMemHostAlloc` storage has a common host/device address on UVA
  devices; write-combined memory has a different contract and is not used.
- [CUDA memory API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html):
  page-locked allocation and asynchronous-copy ownership requirements.
