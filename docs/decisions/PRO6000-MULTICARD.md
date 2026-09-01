# RTX PRO 6000 Blackwell: 2–4 card model placement

Status: capacity-aware automatic placement is implemented behind an experimental door. The HY3
W4A16 EP-4 program is qualified on the named RTX PRO 6000 host; other model x card-count choices
remain receipt-scoped.

## Decision

Use `MEMRA_PARALLEL=auto` as the single shared-model entrypoint on 2–4 RTX PRO 6000 Blackwell
cards. The loader decides from the compiled `ModelPlan`, semantic tensor binding, exact checkpoint
byte census, artifact activation contract, and selected device capacities. A new family does not
receive a handwritten layer list or a bespoke "HY3/Qwen/Gemma parallel loader"; it enters the
same planner once its ordinary ModelPlan and tensor contract compile.

The operation policy is:

| Plan/artifact | Automatic program |
|---|---|
| Dense-only | memory-balanced contiguous PP over every selected card |
| Routed MoE + W4A16 expert boundary | whole-expert EP when every rank preserves the configured reserve |
| Routed MoE whose EP root would not fit | memory-balanced contiguous PP |
| Routed MoE without a qualified device expert backend | contiguous PP |
| Neither legal program preserves capacity | refuse before CUDA allocation |

The current reserve is 6 GiB/card (`MEMRA_PARALLEL_RESERVE_MB=6144`). It is a load-planning floor,
not request admission: KV, context, concurrency, speculative scratch, and prefix-cache state still
pass the server's live admission model. Explicit replica placement remains an operator-level fleet
choice; this door plans one shared model group.

For the sealed HY3 W4A16 artifact, the exact physical census makes the distinction concrete:
EP-2 would place about 100.31 GB on its root before runtime reserve, while legal PP-2 peaks at
90.62 GB. Auto therefore rejects EP-2 and selects the legal PP cut. EP-3 and EP-4 reduce the root
checkpoint estimate to about 73.47 GB and 60.05 GB respectively and are capacity-admissible. These
are placement facts, not throughput claims; only the separately receipted EP-4 execution is
qualified.

## Dense models

Dense shared-model serving uses contiguous PP stages. The planner may cut only at
`ModelPlan::partition_boundaries`; admission must include stage-local weights, KV at the target
context, fixed workspaces, the embedding on the first stage, and norm/head/tail work on the final
stage. Auto uses the same exact header-only checkpoint costs as
`placement-checkpoint-{2,3,4}.tsv`, then adds the runtime reserve before admission. Loader
expansions and optional mirrors remain backend costs; a future backend must account for them before
registering as automatically selectable.

TP-2 remains a latency candidate on a healthy peer pair because it can stream one token from both
cards' HBM. TP-3 and TP-4 are never inferred from card count: heads, KV heads, quantization blocks,
and projection widths must divide, and the measured collectives must beat PP. On a topology that is
two strong pairs joined by a weaker edge, the candidate is TP-2 inside each pair and PP-2 across
pairs, not TP-4 across the whole host.

## MoE models

Layer PP remains the capacity fallback for MoE: whole expert banks remain stage-local, so a token
crosses only stage boundaries instead of dispatching and combining at every routed layer. W4A16
artifacts additionally expose the qualified whole-expert EP backend. The planner distributes every
routed trunk expert bank, leaves router/shared/dense/attention/head work on the root, and includes
local MTP expert residency in the root estimate before selecting EP.

Qualification should nevertheless compare all legal programs:

- PP: whole layers and their expert banks belong to one stage.
- MoE-TP: every expert is tensor-sharded; balanced work, smaller expert matrix shapes.
- EP: whole experts belong to ranks; better expert locality, data-dependent imbalance.
- ETP: expert ownership across groups plus tensor sharding inside a group.

EP is not restricted to high concurrency. Aggregate HBM can reduce a single token's expert weight
time, but it wins only when that saving exceeds PCIe dispatch/combine latency and the slowest-rank
imbalance. Top-k eight over EP-2 or EP-4 commonly touches every rank, so the case for EP at this
scale is expert locality and parallel HBM, not sparse communication destinations. Shared experts
remain root-local in the current backend.

## Runtime shape

`PpNRt` remains the transport and ownership layer: one engine/context/stream per stage,
stage-local weight and cache allocation, double-buffered boundary slots, directed peer-copy
integrity probes, and pinned-host fallback for serial correctness. One generation lease spans each
complete PP walk. Deferred logits retain the generation until the last result is drained; paired
speculation can borrow it only inside its explicitly serialized or coordinated phases. A failed
multi-stage cache wave taints every affected cache, aborts its serving sessions, and cannot enter a
reuse pool.

`MEMRA_PP_WAVE=1` enables the new PP-3/PP-4 schedule. A scheduler tick is split into at most one
wave per stage. One scoped host worker owns each non-head stage and the caller owns the head; waves
flow through explicit stage channels, so concurrent work shares neither a stage engine nor a
request/cache. Each boundary carries two credits, and a slot is returned only after downstream
`rx` records that exact wave's read-complete event. This preserves the shared-scratch,
slot-generation, and per-request ownership laws while allowing independent requests or prompt
microchunks to fill the pipeline. Admission is computed per physical device for two through four
stages. The grow-only boundary buffers are process-global: admission projects only missing
capacity, including the configured concat-prime high-water, rather than charging an already-live
buffer to every session. Generic dense/non-Step concat prime has no cross-device split yet, so the
worker routes those requests through individual `prime_cache` wavefronts.

The door is default **OFF**. It refuses invalid values, host-bounce transport, repeated devices,
and missing double-buffering. `MEMRA_PP_WAVE=0` is the rollback to the existing serial PP-N walk;
PP-2 keeps its independently qualified dual-active default.

## Qualification before a default

Run on the exact non-serving RTX PRO 6000 topology, never a production host:

1. Record GPU UUID/BDF/NUMA topology, negotiated PCIe width, P2P read/write capability, and
   bidirectional byte-integrity/bandwidth results.
2. Run `pp-transport-smoke` for every directed pair.
3. Run `ppn-gate`, `decode-batch-gate --mode pp`, prime split/pipeline gates, and spec verify on
   PP-2, PP-3, and PP-4 in both practical placement orders.
4. Require wave engagement and real host-walker overlap; bit identity without engagement is
   vacuous.
5. Exercise vendor-default sampled serving with concurrency, long context, admission pressure,
   prefix restore, disconnect/rollback, and a PP-2 regression twin.
6. Measure interleaved N≥5 TTFT, E2E, TPOT/ITL p50/p95/p99, request throughput, output-token
   throughput, failures, power, clocks, and thermals. Greedy rows remain exactness instruments.

No PP-3/PP-4 performance or production-support claim exists until those receipts are committed.

## Primary sources

- [NVIDIA RTX PRO 6000 Blackwell family specifications](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000-family/): 96 GB GDDR7 and PCIe Gen5 x16.
- [NVIDIA AI Enterprise GPU table](https://docs.nvidia.com/vgpu/sizing/virtual-workstation/latest/gpus-vws.html): RTX PRO 6000 Blackwell Server Edition lists no NVLink support.
- [Megatron tensor parallelism](https://arxiv.org/abs/1909.08053): column/row parallel transformer
  layers and their collective boundaries.
- [Efficient Large-Scale Language Model Training on GPU Clusters](https://arxiv.org/abs/2104.04473):
  composing tensor, pipeline, and data parallelism around the fabric hierarchy.
- [TD-Pipe](https://arxiv.org/abs/2506.10470): pipeline scheduling on PCIe-only inference nodes.
- [TensorRT-LLM expert parallel scaling](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog04_Scaling_Expert_Parallelism_in_TensorRT-LLM.md): expert locality, communication, and imbalance tradeoffs.
