# Native whole-expert EP pair, 2026-09-05

Implemented behind `MEMRA_DSV4_EP=pair`, default OFF. This is the complete trunk
residency/dispatch path, not the earlier six-expert feasibility probe. It is not
yet production qualified.

## Residency and execution

- Global expert ids 0..127 belong to stage 0; 128..255 belong to stage 1.
  Attention, shared experts and DSpark retain the previous layer owners.
- After initial checkpoint loading, alternate old layer owners during relocation
  so one device does not receive all peer halves before releasing its own.
  Code/scale transfers use bounded 64 MiB chunks and both new shards are checked
  byte-for-byte against the original bank before that bank is retired.
- The final routed banks are sharded, not duplicated. The tiny global macro-scale
  metadata is replicated. Unused full-bank pool pages are trimmed after drains.
- Routing and FP8 activation quantization execute once on the layer owner. The
  wire carries exact codes/scales, global ids and routing weights. Both GPUs run
  the existing arithmetic for their owned experts; masked slots do no GEMV work.
- Peer contributions return to their original slots. A bit-preserving overwrite
  precedes the existing ascending-expert combine; there is no new partial-sum
  reduction order. The entire shared expert remains mandatory common work.
- The canonical monolithic prime keeps its old per-expert program and can read
  selected peer-resident bytes. Steady device decode and verification, including
  chunked prefill, use the actual parallel dispatch.
- Scratch is per decode/verify workspace, not per layer; extra decode bytes are
  exposed explicitly and batched bytes are charged to the verifier's stages.
  Initialization and failure cleanup have explicit cross-device drains.

Grouped-prefill math and the single-stage capture probe refuse EP. Neither may
inherit its PP2 qualification. Persistent multi-device graphs, communication
compaction, shared-expert overlap, load balancing and cross-request scheduling
remain further implementation/qualification work.

## Gates

`ep-partition-5090.log` passes against the original full-bank launcher for all
three projections, both reduction variants, row counts 1/6/32, small dimensions
and real 4096/2048 dimensions. It checks global macro scales, owned/unowned slot
behavior, ordered slot restoration and guards. This is component evidence only.

`dsv4_ep_pair_gate` loads PP2 once and stores exact case digests, relocates every
trunk expert, then compares parallel/serialized/parallel EP execution against
those PP2 results. Cases cover canonical prime, widths 1/32/64, warm snapshots,
suffixes, active C4, sampled plain steps, DSpark output/rounds, and all live
trunk/draft state. Target model candidate:
`c0b226980b7d05a8d824f070170e2408154a7581a363074fa84a946e6528a9cf`.
Target execution is pending. Gate wall includes checks and is not serving TPS.

399 CPU tests and strict engine clippy pass before the final scratch-accounting
getter; final exact-head checks are still owed. No default, serving, throughput,
1M-concurrency or release claim follows from these results.

Primary-source checks today:
[DeepEP](https://github.com/deepseek-ai/DeepEP) for dispatch/combine and low-precision
wire design, and [CUDA peer access](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PEER__ACCESS.html).
No external library/kernel is linked or vendored. NVLink/RDMA performance is not
inherited by this PCIe target.
