# Default-ON PR source provenance

This is the current #374 provenance summary, replacing the inherited claim that
all historical composition hashes still match. The reviewed compiled source is
`7faf840cf75cadb836bd58386db9f2d1da51d944`; subsequent changes are documentation
only. `../dsv4f-cadence-dense-default-on-20260908/PROVENANCE.json` records every
current and inherited whole-file hash plus the additional policy/engagement files.

Of the 16 historically pinned files, **10 remain byte-identical and 6 changed**.
The old `source-provenance.sha256` is retained solely as the historical
composition snapshot; it is not a current verification manifest. Dense ON changes
admitted eager calls as well as captures; cadence requires explicit replay arming.

| Unchanged file | SHA-256 |
| --- | --- |
| `crates/memra-engine/src/dsv4_graph.rs` | `c1e344416ae25e1574b5fa9bfe36ca8c2bae2c3da9377f275188fc9234795bcd` |
| `research/dsv4f-replay-cadence-20260908/DESIGN.md` | `10583c97cda324a9ccae13b2633d93a0d2c0077e0a88b763bbcdc93370e46e6f` |
| `tools/dsv4-replay-cadence-gate.cu` | `a7f37356eb43adb13e32c92da43c2927d106863297644d41e8bb8026a5d9d1f7` |
| `crates/memra-engine/build.rs` | `6c88ed3762f45fe59c453fe3e68534b71d74f1ddd23148af7c1fb1213e46c4ea` |
| `crates/memra-engine/cu/dsv4_gpu.cu` | `5c6bd0e5e3058a85db68d713260c8def3d555d86311a94fcb3c27f4d1a7b49b3` |
| `research/dsv4f-dense-exact-tail-20260908/MODEL-GATE.md` | `a19d8b2e1bfd9d1594940b350ec15a23ce80b34f0c12101c0d075ec68fc189ab` |
| `research/dsv4f-dense-exact-tail-20260908/README.md` | `b6850ae472e5c9e0e8787ba807e8037f9e72ddfca3ee28c0b94caf31ad10ae68` |
| `tools/dsv4-dense-exact-tail-gate.cu` | `5bababfcf632858d208552c7b3d44d0fe0faf9f0cdd857b2d81c217926b1e229` |
| `crates/memra-engine/cu/dsv4_sampler.cu` | `24997d70078825660ff1fe2b9805be1f0181894d2a370c96bce831e90ec431da` |
| `crates/memra-engine/cu/moe_f16_grouped.cu` | `d034320fb62091cc83c27a515642ba52b66171eadc49f1306e01a88f00d8aad6` |

| Changed inherited file | Current SHA-256 | Reason |
| --- | --- | --- |
| `crates/memra-engine/src/bin/dsv4_tp_ep_sampled_perf_gate.rs` | `2b7ec93dca07b2eb277c13bd5328a58efb467cfe51acdfd2ecaf479a55eb1152` | Pins legacy dense OFF and adds actual default/rollback CPU subprocess coverage. |
| `crates/memra-engine/src/dsv4_full_token_replay_gate.rs` | `87125819d5f8b4adb3272c32310be776386feca1666d9206d099fadbb81e2272` | Pins full-replay oracle/profile cadence OFF through the explicit mode API. |
| `crates/memra-engine/src/dsv4_gpu.rs` | `47960d7408c68422698fba3fe45b3907faebed5212703864f0d1b1a2a646edd9` | Adds environment policy reads, explicit mode arming, and host dense selection/readback/restore helpers. |
| `crates/memra-engine/cu/dsv4_dense_m1_exact_tail.cuh` | `2dd12a2c33dbe843d535ecec226105ed83c1649f28b63e25c6aaf437fa2d88ab` | Changes host policy initialization/readback/restore only; device kernel bodies are unchanged. |
| `crates/memra-engine/src/bin/dsv4_dense_exact_tail_gate.rs` | `3188edb6bc2f555ef863ee50e1b4f89cb2e27397a83dac3333e653320ecb5bef` | Pins cadence OFF explicitly so the dense-only instrument stays fixed. |
| `docs/KERNELS.md` | `65e49260c366d6a15ea1e1cc243bcdc95712495984866801cde5fc1772f0cb7f` | Documents ON defaults, eager/replay scope, rollback seams and direct #507/#508/#509 receipts. |

| Additional current file | SHA-256 | Reason |
| --- | --- | --- |
| `crates/memra-engine/src/bin/dsv4_compose_cadence_dense_gate.rs` | `878abc785509dab7ad16c63d095df2fbfe9a0f521fe360df65956a6ef5994517` | Preserves explicit A/B arming and adds the environment-selected engagement mode with real policy restoration. |
| `crates/memra-engine/src/dsv4_default_engagement_gate.rs` | `bfd99820de7f22fa8babff479f3fe46a6e3e629eeb8cd9e3067efb5f23a4d97d` | New 256-step eager identity/census/refusal instrument plus five sanity rows per unset/zero process. |
| `.github/workflows/ci.yml` | `bdd9d011c93fe692f3ec66c8aa1874c104727551845cd6069f163e436256e5c5` | Runs existing sampled/drift/composition CPU contracts with both environment defaults ON. |
| `docs/FLAGS.md` | `f17821cb4c990d3978a198579477bbfe73291e0aec5c6a8a05f25b01cf47309d` | Defines ON/zero rollback semantics, exact eager/replay exposure, direct composition receipts and seam review date. |
| `research/dsv4f-cadence-dense-default-on-20260908/DESIGN.md` | `74e070ea838a9bd52d610c035e636c6185b3e9536abb6a5b3f62b4ea1a648e40` | Scopes this default-policy change, engagement protocol and current provenance separately from old snapshots. |

The unchanged device bodies in the dense header were independently compared
from the reduction-function marker onward. AR, sampler, expert GU/down, replay
ownership and CUDA dispatch/math remain byte-identical. Exact build and
engagement receipts bind 7faf840cf separately from this final docs-only head.
