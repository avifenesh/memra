# Current-main release qualification on Ubuntu 24.04

Tested source: `2554577794c0b57b0630c7cf7e0e7b15b8793735` on accepted main `b013885ba7d365516410efd0c013d90607a99bde`.
Source inputs: `fd1b549a56413ae3b95d848c974ddbab047f079652233d4be60a4d60a8dbd41f` (56457 entries).
All reviewed release-tooling paths are unchanged; runtime and Cargo bytes equal the accepted main. Historical a3/8f and PR566/7b records remain intact and are not reused to qualify this changed source.

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

Record SHA256: `dd6fd77a6f9b59fd843ae27c9f2b8975e7495716b7071941bcf778ceeb4448c2`.
Build record SHA256: `27fd493e4e352d5f13b9935a0cbb59c2e09809fb420fe2eb5e11e73a3b7b12e9`.
Router descriptor SHA256: `20d4bebc79d5b65bed11e477f303a249d683df235380bd4aa6383c05480753e7`.

Publication preserves these tested inputs/executables and adds only evidence plus an index append. `current.json` selects the new source-bound record; earlier capsules stay historical. Publication does not qualify a rebuilt binary. The scope is Ubuntu24 generic correctness; Ubuntu22 native evidence and an independent matching rebuild remain absent. Actual release/tag still requires both shipping profiles and matching rebuilt ELF bytes. No broader serving, Step, PrimeGraph or unrelated model qualification is implied.
