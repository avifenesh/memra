# Staged matrix expert chain and EP composition

2026-09-06 UTC. Same active native DSV4 lane. Defaults remain reference MoE
and EP OFF. The composition is implemented and builds; complete-model gates
and performance admission remain separate.

## Shared execution program

The full-bank matrix path and the EP path now use the same `GroupedWork`
operations. `prepare` validates routing, the live assignment count and lossless
FP8 input transport. `gate_up` queues both native matrix projections, their
macro scales, clipped/weighted SwiGLU, and the source intermediate FP8
quantization. `down` validates the intermediate mirror, projects/scales down
and scatters into original slots. No new CUDA arithmetic was introduced.
An internal phase state rejects out-of-order calls and unfinished overwrite.

`modelopt_table` builds projection-indexed pointer tables over expert-major
code/scale banks. Shape, byte lengths and address overflow are checked. The
loader now charges pointer-table bytes and rebuilds local/peer tables after
the existing byte-checked whole-expert relocation. Reference EP does not
allocate these optional matrix tables. A native-EP load without them cannot
silently switch into the matrix program.

EP dispatch retains the exact source FP8 codes/scales, global router ids and
weights. Both partitions prepare first. In the overlapping schedule both
gate/up chains are queued before either intermediate validation/down stage;
the serial gate control drains the owner chain before queuing peer gate/up.
Partition ownership and complete slot coverage are checked. Return is the
existing peer copy plus original-slot overwrite, never a reordered reduction.
The owner driver and CUDA-runtime contexts are restored before the merge and
following shared-expert work. Failure paths drain both streams before borrowed
destinations can be released.

Each verifier owns its local grouped work and peer grouped work, with physical
GPU byte accounting. Matrix plain decode still uses the one-row verifier, so
prefill, plain and speculative verification stay on one numerical program.
Matrix EP requires device routing and reused scratch; the prefill-only grouped
probe and full-layer capture remain refused. Host validation still synchronizes,
so this is not a completed CUDA-graph implementation or an overlap receipt.

## Checks and pinned gates

- Engine library: 404 tests passed, 9 ignored before the new chain GPU test was
  added; that GPU test was subsequently built and run separately.
- Server library: 602 passed. Strict engine/server library and binary clippy
  passed; the expanded engine test target also passes clippy.
- `modelopt_table_addresses_expert_major_banks_by_projection` independently
  checks a literal two-expert pointer table plus bad dimensions/lengths and
  overflowing addresses.
- `cuda_matrix_chain_full_bank_equals_two_partitions` passes on a PRO GPU:
  hidden 4096, intermediate 2048, 16 distinct synthetic experts, top-6, rows
  32/1/6/32, all-first-rank/all-second-rank/mixed selection. It compares the
  complete input/intermediate FP8, weighted SwiGLU, down and original-slot
  pipeline bitwise. Unused scratch is poisoned with NaNs, empty ranks and
  shrinking/growing live prefixes are exercised, and double-down is refused.
  This is a one-device compute/partition component, not a two-device transport
  or full-checkpoint qualification. Raw target logs are banked in the companion
  private lane as `matrix-chain-component-{controller,gpu1}.log`.

The complete-model phase gate now optionally compares 64 forced rows from each
of the reference and matrix programs against the previously frozen raw bank,
in addition to its width/cache/rollback/plain/DSpark checks. This anchors the
refactor outside itself. `dsv4_ep_pair_gate` with the matrix program selected
then compares full-bank versus EP overlap/serial/overlap controls, real
prompts, warm restoration, active C4 and sampled output.

Initial staged phase binary:
`0224ec276fe807fdccd67d0c0bfb6dab93af88fd032d15c1d137a3c0a95304f2`.
Initial staged EP binary:
`0a13d6a4039d2190116dbdcfeb679e6cb6f870b3f662159278fa230555bba177`.
Their first launcher refused before loading because the pair was occupied.
They are historical candidates; the final explicit runtime-context restoration
is being rebuilt and must have its own hash-bound invocation.

Final context-restoring candidate hashes:

- Phase/frozen-row gate: `c5be98ef0b8cc00d81da22ca84084b7db020a0ee2398b3e29f1ade6130b7c44a`.
- Full matrix/EP gate: `a04bcc491c65ebd457e48c416fb7701968267d8efcd1f7a9e35bdf2ed469b241`.
- Subsequent profile/long gate: `46ff0244c1b5d32fe26e2131edd6d04a93ef162f89c859eddd5ada8e6de235da`.

Staging is not a target pass. The companion ops lane records transfer and
controller state; keep the full-model/frozen-row gates pending until their
actual terminal receipts exist.

The final phase gate subsequently completed on the PRO pair at 2026-09-06
02:53:57 UTC, status 0. It passed both 64-row frozen walks, all cold widths,
fresh/reused scratch, complete cache/ring checks, plain/verify commit and
rollback, and 32 sampled plain/DSpark-identical tokens (15 rounds). This closes
the refactored full-bank phase gate for the c5be98ef binary. The following
full-checkpoint matrix/EP comparison is a distinct gate, not implied by this pass.

That full-model matrix/EP gate also completed at 03:04:05 UTC, status 0, on
binary a04bcc49. All five real-source cases pass in overlap/serial/overlap order,
including 4097-token prefill, warm continuation, host restoration, active C4,
full logits/cache/ring hashes, 41 sampled output tokens per case and DSpark
round/confidence identity. All 43 expert-bank relocations were byte-verified.
Raw: `matrix-ep-stages2-ep.log`, its process audit and controller log in the
private companion lane. This is complete-model correctness for that bounded
scope, not 1M/HTTP/concurrency or general checkpoint-quality admission.

## Evidence still required

The source-window distribution diagnostic completed and was independently
audited; see `matrix-distribution.md`. It is not general quality admission.
Complete model/frozen-bank matrix and matrix-EP gates, same-window sampled
performance, real scheduling and fairness, live graph metadata, exact host-C4
working-set serving, complete TP2 and actual 1M endpoint qualification remain
part of the unchanged objective. No serving claim follows from component passes.
