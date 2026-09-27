# Current-main release qualification on Ubuntu 24.04

Tested source: `4135e7eb5dbd3c1f3e72e386c470e6d8a7ada444` on accepted main `1b354be594acc92d9dc28d6421a93538cecde202`.
Source inputs: `8b8c8048948b0f3f612a5aaef6504d150df36eddfcf28a0b4a1836517366dd35` (58994 entries).
The release producer, input view, validator and native predicates are unchanged; CI and documentation compose accepted main. Runtime and Cargo bytes equal the accepted main. Earlier d31/13467, 53db/9c84, 2554/dd6, a3/8f and PR566/7b evidence remains historical and is not reused to qualify this source.

The fresh controlled v3 build used Rust 1.97.1, CUDA 13.1.115 and the fingerprinted source view inside bubblewrap. Source-view identities matched before/after compilation. `build.json` binds all six ELF outputs. The generic runtime cells execute kernel-check, argmax-margin-probe and run-spec; compiling memra-server, run-gen and tok-parity is not an HTTP-serving or tokenizer-parity run.

All five canonical generic cells passed on one RTX PRO 6000 Blackwell Server Edition, physical GPU0:

| Gate | Result |
|---|---|
| Kernel | ALL GREEN (110 cells, 103 executed, 10 required, 7 skipped; budget 11) |
| ornith-ai/ornith-1.5-35b-a3b argmax | 1 flips, 0 bad under the unchanged calibrated gate |
| ornith-ai/ornith-1.5-35b-a3b | K=1..8 self-consistency, identical to plain target (8/8) |
| qwen/qwen3.8-27b argmax | 0 flips, 0 bad under the unchanged calibrated gate |
| qwen/qwen3.8-27b | K=1..8 self-consistency, identical to plain target (8/8) |

The actual named 9B/35B oracles and roster models are hashed in `run.json`. The sealed `router-baseline-proof.json` binds both exact positive lines in `cells/1-kernel.log` at lines 515 and 516: real35B router, 32m-points 1..2048 and m-invariance, each zero mismatches. No separate router run was needed; counts alone are not its proof.

The completed physical-card lease records wrapper/child exits 0/0, no timeout/interruption and no lingering compute. Source, executable, model and hardware identities match before/after. Raw cells and250ms telemetry are sealed. This is a correctness result with no throughput/timing claim.

Record SHA256: `137f145a35bdda402d877603c9c8e7730bc138a8f7e2af4342cb6cf0f9ae582d`.
Build record SHA256: `86f63187661a594e1631165ca23fd8858979bd2a278e71326d71cc4d7d2f4851`.
Router descriptor SHA256: `da7fed4ac020b7ecb0cd880464a6c292a765df5bbe711d7d176ec594b00c3aef`.

Publication preserves these tested inputs/executables and adds only evidence plus an index append. `current.json` selects the new source-bound record; previously published capsules remain in immutable Git, and the d31 receipt plus all build/native archives remain preserved off-host. Publication does not qualify a rebuilt binary. The scope is Ubuntu24 generic correctness; Ubuntu22 native evidence and an independent matching rebuild remain absent. Actual release/tag still requires both shipping profiles and matching rebuilt ELF bytes. No broader serving, Step, PrimeGraph or unrelated model qualification is implied.
