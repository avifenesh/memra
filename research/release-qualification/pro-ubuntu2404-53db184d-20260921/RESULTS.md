# Current-main release qualification on Ubuntu 24.04

Tested source: `53db184d912c040e14db5ca95a548ff76ff6a25e` on accepted main `9ef2f04d6ece3bbb0f6049ac952df3ada9f173b2`.
Source inputs: `394ba057a7b486261b163edfdfb3db08c0cb2724826e41b02186d9ea0ae7ccaf` (57947 entries).
The release producer, input view, validator and native predicates are unchanged; CI and documentation compose accepted main. Runtime and Cargo bytes equal the accepted main. Earlier 2554/dd6, a3/8f and PR566/7b evidence remains historical and is not reused to qualify this source.

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

Record SHA256: `9c84ed19c55016c1b06613764f910736657352c83c819c9fd13a9a368ec9dae3`.
Build record SHA256: `e261eedab34623188d468e30c9396c168253c1061d08b9fff5ad3e134e49400e`.
Router descriptor SHA256: `27c706e79787c4fbfe6cf6134f10809709a2041ad93f4c3c5a3be870d9666d09`.

Publication preserves these tested inputs/executables and adds only evidence plus an index append. `current.json` selects the new source-bound record; earlier capsules stay historical in immutable Git and verified off-host archives. Publication does not qualify a rebuilt binary. The scope is Ubuntu24 generic correctness; Ubuntu22 native evidence and an independent matching rebuild remain absent. Actual release/tag still requires both shipping profiles and matching rebuilt ELF bytes. No broader serving, Step, PrimeGraph or unrelated model qualification is implied.
