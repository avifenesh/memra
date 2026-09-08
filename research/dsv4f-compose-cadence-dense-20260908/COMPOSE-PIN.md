# Cadence and dense composition source pin

Base: `001c09e5d451798ef8d570f6087dc11527cbcc19` (fresh origin/main).

Cadence #368: `c16385b02408460b19f311b0e8374b4b07fef5f9`.

Dense #366: `5b66fe9bfd3d8d2b15f31aac484172c848300f0d`.

Both exact reviewed heads were merged with `--no-ff`. No manual conflicts occurred; Git combined the independent FLAGS.md insertions. No reviewed kernel, runtime or helper was rewritten. The table records SHA256 of each complete file and its reviewed source commit. FLAGS.md is the sole two-source composition below. Release metadata from current main is retained. GU #369/#367 are excluded.

| File | Provenance commit | SHA256 |
| --- | --- | --- |
| `crates/memra-engine/src/bin/dsv4_tp_ep_sampled_perf_gate.rs` | `c16385b02408460b19f311b0e8374b4b07fef5f9` | `2566775756ee9958553b421579a8cd9ee212130ae7ff549ff7eed08b062eb4ab` |
| `crates/memra-engine/src/dsv4_full_token_replay_gate.rs` | `c16385b02408460b19f311b0e8374b4b07fef5f9` | `3603accd76a17209cba78009c6e33272545dbc07ef756d638211a19026a3aea5` |
| `crates/memra-engine/src/dsv4_gpu.rs` | `c16385b02408460b19f311b0e8374b4b07fef5f9` | `6bb018fe9a7d10c04517b4da47f66b32758b36d3d5a39bbac61d73f72b17d593` |
| `crates/memra-engine/src/dsv4_graph.rs` | `c16385b02408460b19f311b0e8374b4b07fef5f9` | `c1e344416ae25e1574b5fa9bfe36ca8c2bae2c3da9377f275188fc9234795bcd` |
| `research/dsv4f-replay-cadence-20260908/DESIGN.md` | `c16385b02408460b19f311b0e8374b4b07fef5f9` | `10583c97cda324a9ccae13b2633d93a0d2c0077e0a88b763bbcdc93370e46e6f` |
| `tools/dsv4-replay-cadence-gate.cu` | `c16385b02408460b19f311b0e8374b4b07fef5f9` | `a7f37356eb43adb13e32c92da43c2927d106863297644d41e8bb8026a5d9d1f7` |
| `crates/memra-engine/build.rs` | `5b66fe9bfd3d8d2b15f31aac484172c848300f0d` | `6c88ed3762f45fe59c453fe3e68534b71d74f1ddd23148af7c1fb1213e46c4ea` |
| `crates/memra-engine/cu/dsv4_dense_m1_exact_tail.cuh` | `5b66fe9bfd3d8d2b15f31aac484172c848300f0d` | `9344ebe4f0205b7ca470c5a9da73ecd3d16137a4bf059f5dfdfea7e58550d736` |
| `crates/memra-engine/cu/dsv4_gpu.cu` | `5b66fe9bfd3d8d2b15f31aac484172c848300f0d` | `5c6bd0e5e3058a85db68d713260c8def3d555d86311a94fcb3c27f4d1a7b49b3` |
| `crates/memra-engine/src/bin/dsv4_dense_exact_tail_gate.rs` | `5b66fe9bfd3d8d2b15f31aac484172c848300f0d` | `a0b446b9bb04a96619c6c304bab3c7ba38d84e7c2135c147d2a6c632c76f3a61` |
| `docs/KERNELS.md` | `5b66fe9bfd3d8d2b15f31aac484172c848300f0d` | `0b33ce77a0d1c011e7a48768c86ce67e9fbb51023817d51bceaeb4e9fd6a3bc9` |
| `research/dsv4f-dense-exact-tail-20260908/MODEL-GATE.md` | `5b66fe9bfd3d8d2b15f31aac484172c848300f0d` | `a19d8b2e1bfd9d1594940b350ec15a23ce80b34f0c12101c0d075ec68fc189ab` |
| `research/dsv4f-dense-exact-tail-20260908/README.md` | `5b66fe9bfd3d8d2b15f31aac484172c848300f0d` | `b6850ae472e5c9e0e8787ba807e8037f9e72ddfca3ee28c0b94caf31ad10ae68` |
| `tools/dsv4-dense-exact-tail-gate.cu` | `5b66fe9bfd3d8d2b15f31aac484172c848300f0d` | `5bababfcf632858d208552c7b3d44d0fe0faf9f0cdd857b2d81c217926b1e229` |
| `crates/memra-engine/cu/dsv4_sampler.cu` | `001c09e5d451798ef8d570f6087dc11527cbcc19` | `24997d70078825660ff1fe2b9805be1f0181894d2a370c96bce831e90ec431da` |
| `crates/memra-engine/cu/moe_f16_grouped.cu` | `001c09e5d451798ef8d570f6087dc11527cbcc19` | `d034320fb62091cc83c27a515642ba52b66171eadc49f1306e01a88f00d8aad6` |
| `docs/FLAGS.md` | merged #368 + #366 insertions onto base | `854b15a23d6e73b3d2e9f288a23b038a22d7ec21b317ebb365048eaf4cb241d9` |

The entire `crates/memra-engine/cu/dsv4_gpu.cu` file matches dense #366.
Cadence changes only graph selection/capture host code; its CUDA kernels remain
the existing eager bodies. The dense header, product/scale loads, eight-element
leaves and exact 128-leaf reduction tree remain unchanged. Sampler and expert
GU/down CUDA files match base main exactly.

Both FLAGS.md additions remain present verbatim relative to their source
branches. Cadence is a request-local explicit gate arm; dense uses the existing
thread-local gate selector, initially false. Neither adds an environment door
or changes the serving default. Rollback and decide-by 2026-09-22 entries remain.
No feature performance transfers to this newly linked binary.

Build checkpoint: compile both existing standalone binaries
`dsv4_dense_exact_tail_gate` and `dsv4_tp_ep_sampled_perf_gate` from this tree,
arch120a, MEMRA_DSV4_FMAD=0, two jobs, nice 10. These unchanged helpers prove
source/link integration only. Neither is advertised as the combined A/B
instrument. The draft COMPOSE-PROTOCOL.md identifies the required standalone
adapter and review before any combined model execution. No GPU cell is part of
this source checkpoint.
