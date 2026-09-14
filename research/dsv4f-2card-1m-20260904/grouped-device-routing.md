# Device-owned routing for the grouped matrix path

2026-09-05 UTC. Active lane, source base `3b1d0a88b6881607d5297e897b804d383e77351a`
plus the current uncommitted EP/capture/routing work. No default, artifact,
numerical-class, serving, or release promotion.

## Implemented

`MEMRA_DSV4_GROUPED_ROUTE=host|device` chooses routing metadata inside the
separately opt-in grouped-prefill or whole-request matrix class. Default is
host. Device routing:

- Counts, prefix-sums and stably scatters expert assignments on GPU. Original
  slot order is retained within each expert; route weights and all three outer
  scales are copied bit-for-bit. Empty expert groups are explicit.
- Keeps metadata in per-VerifyWs buffers rather than allocating it per layer.
  Charged bytes are `4 * (3 * experts + 2 + 6 * slot_capacity)` per workspace.
- Passes null host offsets only for the ModelOpt split-plane matrix ABI.
  Persistent 32-row, 128-row and deep-tail visitors derive exact tile counts
  from their existing device prefix. Weight decoding and MMA chains are unchanged.
- Returns a checked device status for invalid expert IDs before metadata is used
  by the engine. CPU routing remains an explicit one-load control.
- Counts successful device preparation for gate engagement checks. The exclusive
  setter drains streams; request-time policy mutation is not supported.

Current primary design reference: DeepGEMM's masked grouped decode keeps changing
expert populations device-side for graph execution:
https://github.com/deepseek-ai/DeepGEMM/tree/559d79fb6994a58b8a15b4b93bf13ccc16edf247
No external kernel code or runtime dependency was introduced.

## Completed local checks

`tools/dsv4-grouped-route-gate.cu` is a correctness-only CUDA executable. Its
SHA256 is `37946ced86169294698fb0303e258b60625b2a19ca2fc4ccd5902217e0fa8bf3`.

On the RTX 5090 Laptop development GPU, 43 cells pass with five routing patterns
each: concentrated first/last expert, distributed, skewed and permuted. Coverage
includes rows 1/6/32/33/128/512, 256-expert metadata, singleton groups, empty
groups, all three visitor policies, both tail settings, exact route/scale bits,
independent output-tail canaries and two invalid-ID controls per cell. Seven
cells capture once and replay with changed routing, producing 35 checked graph
replays. Device-count matrix outputs match the host-count control bit-for-bit.

Raw evidence:

- `grouped-route-5090.log`: all 43 cells PASS, no timing claim.
- `grouped-route-memcheck.log`: same cells, `ERROR SUMMARY: 0 errors`.
- `grouped-route-synccheck.log`: same cells, `ERROR SUMMARY: 0 errors`.
- `grouped-route-final-cpu.log`: 400 passed, 0 failed, 7 ignored.
- `grouped-route-final-clippy.log`: strict engine library and wide-gate clippy PASS.
- `grouped-route-final-build.log`: optimized full-model gate build PASS.
- Formatting, diff whitespace and runtime flag census PASS.

Build command:

```sh
/usr/local/cuda-13.1/bin/nvcc -O3 -std=c++17 -arch=sm_120a \
  --expt-relaxed-constexpr -fmad=false -Xcompiler=-ffp-contract=off \
  tools/dsv4-grouped-route-gate.cu -lcublasLt -lcublas \
  -o target/dsv4-grouped-route-gate
flock -n /tmp/memra-5090.lock target/dsv4-grouped-route-gate
```

## Target gate and remaining work

`dsv4_wide_prefill_gate` now compares host/device/host metadata on an actual
1025-token source prompt at widths 32/128/512. It compares full logits, live
cache classes, DSpark state, sampled output and rounds, and asserts positive
device-routing calls only in the device arm. Target SHA256:
`d26c7208994c7de3d17722a3471d61a19216b74d5558c8338b5ade887fc633dd`.

The component gate passed on both PRO GPUs. The full-model host/device/host
comparison completed at 2026-09-05 22:53:08 UTC with an explicit scalar-indexer
control. Device-routing calls were 1376/344/86 at widths 32/128/512; the host
controls recorded zero. Full logits, caches, DSpark state and sampled output
agreed. The initial tiled-indexer attempt failed its stale 64-row launch guard,
before comparing routing; that launcher was separately repaired and gated.
Raw target records are banked in the companion private lane as
`routing-scalar-model.log` and `routing-scalar-controller.log`.

This does NOT complete a captured MoE: scalar route validation and FP8-to-half
validation still synchronize. The subsequent persistent-workspace change
removed the per-call half/contribution allocations, and the separate
whole-request matrix program passed its phase/state/storage gate. See
`matrix-request-program.md` and `matrix-workspace-reuse.md`. The prefill-only
grouped probe is still not a phase-consistent serving program. Next qualify
checkpoint behavior and remove host validation from the critical path with an exact
fail-closed publication contract. Full-layer/round graphs, TP/EP comparison,
host-C4 capacity and actual 1M/concurrency qualification remain open.
