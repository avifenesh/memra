# WP-D day-1 baseline — 2026-09-19

Repository: `avifenesh/memra`; source inspected at
`c5a33b14ff7b6c75a3cd808dbc0ee4f10aa8b33f`.
Branch: `lane/spill-d-20260919`. CPU scaffold only, no runtime integration or GPU qualification.
DSv4 Flash 0731 stays paused and unrelated: no run, cell, adapter, or oracle in this lane.
No V4.1 model code. Target peer tier is PCIe P2P; RTX PRO 6000 Blackwell Server Edition
has no NVLink. Nominal Gen5 x16 ~64 GB/s/direction is not a measurement.

## Directly inspected transport and placement

| Source at baseline | Existing behavior | Generalization / restriction |
|---|---|---|
| `crates/memra-engine/src/pp.rs:1861–1923` | Context peer-access enable, fixed 16-KiB byte probe before default memory-pool access grants. Grants enumerate owning pool and accessing device separately. | Capability query is not a grant or byte correctness. New tier routes must require context AND pool grants; read-only probes cannot grant them. |
| `crates/memra-engine/src/pp.rs:3025–3151` | `tx_pipelined` alternates two boundary slots; TX waits for RX's previous read; receiving allocation is synchronized before first TX. Explicit-context `memcpy_peer_async`, local copy, and host-bounce are separate arms; TX records an event, RX later waits and produces local storage. | Preserve source and destination lifetimes plus both producer/consumer fences. PP activations are not complete KV/continuation state. Stable graph addresses need longer pins than one DMA. |
| `crates/memra-engine/src/tp_transport.rs:103–120,321–350,404–435,563–622` | Three transport arms: HostCanonical, PeerPull, **DevicePush**. Per-rank publish and release events protect consumer readiness AND source reuse. `pull_f32` carries range checks and release ordering. | Reuse lifetime rules, not TP numerical reductions. Do not call PP or peer capacity TP. Gate hop-heavy as well as optimized walks; a fewer-hop arm hid a historical race. |
| `crates/memra-engine/src/tp_transport.rs:625–765,1193–1347` | Cited 625–765 is fanout/move, **not probing**. `arm_transport` later executes the byte-integrity ladder with direction-specific transport/counters. | Atlas D has stale line/arm inventory despite its stated same base; actual source wins. No edits to this runtime file on day 1. |
| `crates/memra-engine/src/qwen4exp_gpu.rs:2162–2198,10425–10478` | `peer_kv_max_cap()` defaults to 8192 rows. `alloc_state_reserve` refuses deeper peer KV before allocation. Quantized QSA scatter is thread-per-position, with 32 distinct cache rows/sectors per warp; peer reads lose local-L2 reuse. Source documents ~523 GB traffic for one 2048-token chunk at 262k. | Keep this guard untouched. New API accepts contiguous copy extents and returns LOCAL materialization permission; no row-index remote gather or raw peer attention pointer. The historical amplification is a regression sentinel, not a fresh bandwidth observation. |
| `tools/box-health.sh:138–165`; `docs/TESTING.md:438–474` | Current/max PCIe generation and width; P8 generation downshift defers active verification, below-max active link fails. Full health script also launches a peer kernel later. | Do NOT run the whole health script as read-only inventory. Record raw topology/query output first; qualified copy evidence and SM remote-read evidence are different things. |
| `docs/SERVING.md:41–50,513–534`; `docs/TESTING.md:480–523` | PP-3/4 requires distinct devices, native P2P, legal plan cuts, double slots and qualified batch rewrite; no host bounce or repeated devices in wavefront. Gates separate eager, batched, prime, spec and serving crossings. | Step generic PP ladder is D2. Pair evidence can prove four tiers but not all four-card routes. Keep `step-pro` hash-bound gate for Step changes. |

## Inputs / design lineage

Read `AGENTS.md` fully, `docs/ROUTER.md`, `docs/TESTING.md`, PP/placement sections
of `docs/SERVING.md`, and `research/INDEX.md`. Read plan 08 and synth D/E; placement
fixture ports 07 §3 arithmetic only. Relevant private corpus laws inspected read-only:
non-vacuity/red controls; pin against independent truth; source-release fence;
most-hop transport walk; no measuring warm-up traffic; canonical lock with close-on-exec.
No deployment identities or secrets were read or copied.

E tags N8/M4/T4: registered descriptor lists, per-item acceptance, notifications do not
replace completion, retained source/destination ownership. V6: worker acknowledgements
and complete heterogeneous state, not scheduler guesses. P1/T2: bounded slots, measured
end-to-end route budgets and shared read/write interference, not summed label bandwidth.
External systems are design references only; no external runtime dependency is added.

## Topology design, not a fresh hardware observation

`peer/topology.rs` represents directed capability/context/pool/direct-byte observations
separately, unknown explicitly. Device records include NUMA node, CPU affinity and current/max
PCIe generation/width. P8 downshift is `IdleDeferred`, never a healthy measured link.
A four-device fake fixture covers all 12 edges and asymmetric denied grants.

Read-only inventory commands, after the lead authorizes a non-serving rig:

```sh
nvidia-smi topo -m
nvidia-smi topo -p2p r
nvidia-smi --query-gpu=index,pstate,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max --format=csv
lscpu --extended=CPU,NODE,SOCKET
```

Linux adapter can read `/sys/bus/pci/devices/<BDF>/numa_node`, `local_cpulist`,
`current_link_speed`, `current_link_width`, and parent bridge paths. Unknown/missing
fields stay unknown. Querying peer capability does not prove an actual direct route;
owner-thread context/pool grants and patterned native copies belong to D1, after inventory.
No writes to clocks/power/ACS/IOMMU/driver or memory-pool access during inventory.

No SSH attempt from D: the lead reported two failed bounded launch attempts; this Mac has
no NVIDIA GPU. D did not retry or contact a serving instance. Real topology is pending.

## Placement arithmetic boundary

`placement_report` takes explicit per-device census allocations and reservations, owner
record geometry with physical replica devices, and route demand inputs. It does not inspect
architecture names, assume equal layer cost, quarter an MQA cache, choose a record format,
or grant admission. All multiplication/addition is checked. Headroom may be negative.
Estimation annotations survive in `DeviceBudget`. Route rate is measured or unknown, never
filled with nominal PCIe throughput; active traffic is independent of retained KV slope.

Sizing-only fixture from 07 §3:
- Equal 10/10/10/10 weights: 76,367,322,992 / 74,062,760,176 / 73,925,697,008 /
  83,172,210,424 bytes. Sum 307,527,990,600. Attention/indexer residual allocation and
  norm/router residuals are estimates, not a per-layer header census.
- 584+68 opaque record bytes; GPU0 owns two half-rate streams, slope **652 B/token**.
- 528-byte ring floors: 675,840 / 675,840 / 675,840 / 878,592 B/request.
- At T=1,048,576, first overflow **N29 GPU0**; N28 fits. At N29, first context overflow
  **T=1,037,290**. With illustrative 4GB/card reserve first overflow N23 GPU0.
- At N16/1M, remaining bytes: **8,683,118,736 / 16,457,053,968 / 11,124,744,720 /
  12,813,732,104** (matches 07's rounded decimal-GB values).
- Separate opaque 288+68 sizing fixture gives N53; it is NOT a fallback numerical program.
- Tests also cover odd context floors, explicit replicas (2608 B/token rather than 1630),
  alternate 9/11/9/11 arithmetic, loader/scratch/staging charges, invalid owners and overflow.

No predicted bytes have been compared with CUDA allocation peaks yet. G0–G7 remain pending
as an integrated program; CPU fixtures alone do not grant GO.
