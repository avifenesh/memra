# Native grouped routing for expert-ID partitions

2026-09-06 UTC. Progress toward matrix plus expert-ID EP on the pinned DSV4
program. This is a component implementation and local correctness result,
not a target-pair, complete MoE, model-quality or serving qualification.

## Contract

`memra_dsv4_grouped_routes_partition` keeps only assignments in
`[first, first + expert_count)`. It never changes the selected expert or its
routing weight. It emits expert-major CSR with ascending original slot order
inside each expert. Global expert ids address source macro scales; local
group ids address a partition-local weight-pointer table. All three macro
scales and route weights are copied bit-for-bit.

The final offset is the live assignment count. An empty rank is valid. Only
the live prefix of pairs/token ids/weights/scales is written; its tail remains
untouched, not padded with fabricated assignments. Every global selection is
validated, including ids outside the rank's owned interval. The status is
overwritten on each call, so a previous invalid id cannot poison a later valid
smaller input.

The original full-bank entry keeps its original count/prefix/scatter program,
now with compile-time non-partitioned count/scatter specializations. Rust
`GroupedRoutes::new_partition` accounts metadata by local group count but
validates source scales and top-k against global expert count. Its CPU control
performs the same filter and stable order. It reads the live count together
with the existing status check. Host synchronization remains explicit.

The normal full-bank caller additionally requires `live_slots == input_slots`.
This initial component did not enable matrix plus EP. The subsequent staged
executor now rebuilds the local/peer tables and composes both chains; its
full-model gates remain pending. See `matrix-ep-chain.md`. No serving flag or
numerical/format default changed.

## Primary design reference

The current [DeepGEMM interface](https://github.com/deepseek-ai/DeepGEMM/tree/559d79fb6994a58b8a15b4b93bf13ccc16edf247)
distinguishes concatenated M-grouped prefill from masked decode work with
device-known populations. That motivates keeping ownership and valid counts
explicit. Its stated SM90/SM100 support is not an SM120 implementation receipt.
No external kernel or runtime dependency was added. The native implementation
retains our existing FP8-QAT mirror and NVFP4 matrix visitor, not a replacement
activation format or router.

## Completed local checks

- `grouped-ep-route-5090.log`: 56 component cells on RTX 5090 Laptop. Each has
  seven routing populations and all three projections, comparing a local
  expert-pointer table against the global-table grouped-MMA control, bitwise.
  Coverage includes rows 1/32/33/512, both 128-expert halves, singleton and
  non-power-of-two middle partitions, all three visitor variants and both tail
  settings. Empty ranks, local/global id mapping, signed-zero route weights,
  macro-scale bits and independent output-tail canaries pass.
- Every component cell replays a captured routing graph for all seven
  populations (392 metadata-graph replays). This is NOT captured full-MoE
  execution: live activation transport remains host-admitted.
- Invalid global ids -1, global_count and INT_MAX reject; subsequent valid
  preparation clears the status. Six invalid partition/top-k shapes per cell
  reject before launch.
- `grouped-ep-route-regression-5090.log`: the original 43-cell full-bank route
  and matrix-visitor gate still passes, including its full-route/GEMM graph
  replay cases.
- `grouped-ep-route-memcheck.log`: all 56 component cells pass with zero errors.
- `grouped-ep-route-synccheck.log`: the same cells pass with zero synchronization
  errors, including empty-rank device visitors and routing-graph replay.
- `grouped-ep-route-rust-5090.log`: Rust host/device metadata integration passes
  five partition shapes, row counts 32/1/3/32, four populations, invalid global
  ids, empty ranks, stale-tail exclusion and post-error live-prefix clearing.
- `grouped-ep-route-cpu.log`: 403 library tests pass, 9 ignored. The new ignored
  CUDA test is run separately above. The new CPU partition-bounds test covers
  zero, oversized and overflowing dimensions.
- `grouped-ep-route-clippy.log`: strict engine library and all engine binaries
  pass clippy. Formatting, whitespace and runtime-flag census also pass.
- `grouped-ep-route-server.log`: 602 server library tests pass.

### Fixture strengthening

The initial pointer-table comparison used periodic weight bytes that repeated
across opposite expert halves. A deliberately wrong local-table entry therefore
looked identical. `grouped-ep-route-fixture-red.log` records the expected test
failure: `corrupted shard pointer table was invisible to the fixture`. This is
a weakness in that initial table-comparison oracle, not a demonstrated defect
in the routing implementation. The initial hashes/logs below are preserved.

The fixture now varies weight codes and block scales by global expert and
projection. Every multi-expert cell deliberately substitutes the opposite
half's pointer for local expert zero and requires an output mismatch before
restoring the correct table. V2 build and run logs use `grouped-ep-route-v2-*`.
The strengthened component passes all 56 cells and all 55 multi-expert
pointer-corruption controls (the single-expert cell cannot select a wrong
expert). Its SHA256 is
`cc4df7ee5e68b680e5547c024dd7d1c6139f135f99e0ee91c2d50030a9a6b29f`.
Use this v2 component for target qualification, not the initial periodic fixture.
The v2 sanitizer rerun was refused before starting because the local GPU had
an unrelated 3370 MiB allocation. No v2 memcheck/synccheck result is claimed;
the earlier sanitizer passes apply to the initial fixture. Production kernel
code did not change during the fixture correction.

The subsequent PRO-pair qualification completed at 2026-09-06 01:14:06 UTC:
v2 passes plain, memcheck and synccheck separately on each physical card, with
zero sanitizer errors. Thus the local refusal above remains historical, while
target-specific v2 sanitizer evidence now exists. Raw target files are banked
in the companion private lane as `grouped-ep-route-v2-gpu{0,1}-{plain,memcheck,synccheck}.log`
and `grouped-ep-route-v2-controller.log`.

Initial component SHA256 (historical, weaker pointer-table fixture):
`1094f232d937338f3c8006c747d4bb8b4fb8b507e41c0a5d1b260c68e5e41414`.
Full-bank regression binary SHA256:
`5332fbdb25ba4bf6d067bb59fe339d1900211196b864b6d5ab4a88644efc069b`.

Build/run form:

```sh
/usr/local/cuda-13.1/bin/nvcc -O3 -std=c++17 -arch=sm_120a \
  --expt-relaxed-constexpr -fmad=false -Xcompiler=-ffp-contract=off \
  tools/dsv4-grouped-ep-route-gate.cu -lcublasLt -lcublas \
  -o target/dsv4-grouped-ep-route-gate
flock -n /tmp/memra-5090.lock target/dsv4-grouped-ep-route-gate
```

## Remaining composition work

The local/peer table rebuild and admitted-prefix staged execution are now
implemented in `matrix-ep-chain.md`, retaining source FP8 dispatch, routing
weight placement before intermediate QAT and original-slot overwrite return.
The real-dimension one-device compute-chain gate passes. Qualify the complete
two-device chain against the full-bank matrix program and its frozen rows,
then measure overlap and sampled serving before promotion. This routing
component alone proves no overlap or speedup.

The earlier complete matrix phase receipt is bound to binary 7654612e, not to
this newer source. The frozen reference/matrix distribution diagnostic still
needs its target run. Matched sampled performance, full model/phase re-gates,
complete TP2, cross-session scheduling, live graphs, exact host-C4 working-set
service and actual 1M/c1-c16 qualification remain on the original path.
