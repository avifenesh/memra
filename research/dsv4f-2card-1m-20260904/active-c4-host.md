# Active C4 host residency, 2026-09-05

Implemented, not a serving default. Active history and parked-prefix storage
are separate tiers. The original transition API remains available; direct-host
fresh allocation and restoration were added on September 6.

`offload_c4_decode_state` transitions a primed state between closed device
transactions. C4 compressed rows move to cacheable pinned host memory; their
full-capacity GPU allocation is replaced by SWA128 plus the session's transient
rows. C128, compressor pending state, and the indexer's existing f32-QAT keys
remain on device. That transition API starts with an all-device prefill.

`alloc_decode_state_host_c4(capacity, transient_rows)` instead creates the
bounded GPU layout and full-capacity pinned C4 history directly. It requires
the matrix program, including position-zero prime, and never allocates then
shrinks a full GPU history. `restore_decode_state_host_c4` reads the canonical
snapshot directly into that layout, uploading only its SWA rows. It preserves
the numerical-program tag; incompatible tags refuse. The pure
`c4_host_bytes_for_capacity` forecast returns exact per-stage host bytes.
Capacity uses complete compressed blocks, so capacities below four require
no host rows. Transient widths are bounded at 512.

The native gather copies selected C4 rows once per query into a bounded
per-stage buffer shared by all heads. SWA and transient rows come from the
device allocation. Logical indices, ordering, pads and every f32 bit are
unchanged. Compressor emissions use same-stream asynchronous D2H; subsequent
gathers use the same stream. Rollback changes the live high-water mark, so
rejected rows are inaccessible and are overwritten on re-emission. There is
no approximate top-k, recomputation, weight/KV format substitution or external
runtime dependency. An explicit, default-OFF recent-row GPU sidecar is now
implemented in [c4-recent.md](c4-recent.md), with hardware gates pending.
It is not a general hot-page LRU; that policy still needs measured miss/reuse
evidence.

Snapshot serializes the same canonical `[SWA, live compressed rows]` as the
device baseline. Original restore recreates the device representation; direct
host restore avoids that intermediate GPU history. Snapshot copies active host
rows directly into its pinned slab, avoiding a full-history pageable vector.
Host bytes and released device cache bytes are
separate counters. Extra decode gather scratch is exposed by
`c4_device_scratch_bytes`; batched gather scratch is charged to verifier bytes.
CPU reads and host deallocation drain the owning stream. Monolithic prefill on
an already offloaded state is refused; device decode and chunked continuation
are the implemented consumers.

Long-prefill observability is opt-in through
`MEMRA_DSV4_PREFILL_PROGRESS=1`. The existing teacher-forced chunk loop then
emits one `PREFILL_PROGRESS` receipt per completed chunk with suffix rows,
absolute position, and width. The default is unset, so serving output,
arithmetic, chunk boundaries, and synchronization are unchanged. This is an
odometer for future 256K/512K/1M cells, not a throughput claim.

## Evidence and remaining gates

- Direct fresh/restore gate `dsv4_c4_direct_gate` passed on both PRO cards at
  2026-09-06 05:35:05 UTC, binary
  `b26e7c8d8decc464541549e650c0fc4ac17b1c5f9d3651f68bf82e4c513503e1`.
  Real-source lengths 1/160/1025/4097 and widths 32/512 cover fresh position
  zero, complete logits/cache/rings, a 129-token suffix, nine sampled plain
  steps and 32 sampled DSpark outputs with confidence/round identity.
  Device-to-host, host-to-device and host-to-host restoration each pass at
  original, enlarged and reduced legal capacities. Actual host bytes equal
  the forecast and replace exactly the corresponding GPU-history bytes.
- Its capacity-only 1048576-token/512-row allocation succeeds: GPU cache
  bytes `[1680457728, 1542225920]`, host C4 `[5905580032, 5368709120]`, matrix
  scratch `[14484064, 14441056]`. These are not total GPU residency: model
  weights, DSpark and other workspaces remain separate. This does not prove
  an actual 1M prompt.
- Direct host-copy GPU unit passes independently on each PRO at 05:43 UTC,
  binary `496dd2ce8b8274e1e70dbdcd6bddb25832b124e2cdaa78e107bcfc6f5c8e20d1`.
  Whole-row/range controls, signed zeros and live-prefix bit preservation are
  exercised without a full GPU-history allocation.
- 398 engine CPU tests pass, 4 GPU-dependent tests ignored in that run.
- Strict clippy passes for the engine library, binaries and tests.
- `active-c4-gather-5090.log` passes on the non-serving RTX 5090 Laptop device:
  query widths 1/6/32, live rows 0/1/40/1025/262144, dead capacity gaps, SWA,
  transient rows, duplicate selection, negative-zero bits, pads, guard regions,
  and asynchronous re-emission into previously read host rows. Test binary:
  `a507d4c9a755978018323e64d5384807be20f6d36ab1f54cdbb4ac5c369817eb`.
- Target model gate `dsv4_c4_host_gate` passes on the two RTX PRO 6000 Max-Q cards,
  including both legacy and device top-k selection. Binary:
  `0c7c2df525a399fbf9963178fa2819208a7b8b042efb0efa44c1b38014d8d355`.
  It compares device/host/device logits, nine sampled plain steps, 32 sampled
  speculative outputs and rounds, every live trunk/draft class, suffix lengths
  0/33/129 and canonical snapshot/restore on real 1/160/4097-token prompts.
- Full-layer one-shot capture also passes under both residency arms with binary
  `dd048ddc5845c6238e2c1235829c08c1c8190a94cfed025d953a5eb146435c0d`.
  This does not qualify persistent replay at changed positions.
- Still required: actual 1M prompts, 1M/c1–c16
  admission and fairness, host-RAM bounds, per-phase I/O profiling, persistent
  hot-buffer reuse, and production scheduling/qualification. No throughput or
  concurrency claim follows from the component gate.

## Experimental serving door

`MEMRA_DSV4_C4_HOST_MB` defaults OFF and independently bounds active C4 history.
Nonzero requires the matrix program and nonzero prefill chunk. The dedicated
FIFO worker checks the sum of both stages' capacity forecasts before consuming
a parked hit or allocating. Over-budget requests receive overload; direct-host
allocation failure never silently switches to a full GPU history. Actual
allocation must equal the forecast. This is one-active-request admission;
concurrent scheduling still needs aggregate reservations. The parked-prefix
budget and transient snapshot workspace require additional host headroom.

The first server candidate is
`b15deaada6e995768d32226a99d53f9f1a94c93919004e9446ec02b919f6d94a`.
604 server CPU tests and strict library/bin clippy pass. The loopback HTTP
GPU/host gate passed at 06:35:15 UTC: vendor-default sampled requests, seeded
cold/warm output identity at 256/8192-token prefixes, positive restore counts,
and a 65536-token request refused against a 128 MiB active-host budget. The
refusal names 704815104 required bytes versus 134217728 configured bytes; a
valid retry still restores its 8192-token prefix and matches the cold output.
Both original request bodies and raw SSE responses are banked with server,
client, controller and process logs in the companion private lane.
This does not establish long HTTP TTFT or concurrency. Long-context HTTP and
1M/c1–c16 gates remain pending. No deployment or default is promoted.

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
