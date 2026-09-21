# Standalone release qualification on Ubuntu 24.04

Tested source: `a3dc3cc25835917b45c4b03bf563afb15c1673dc` on accepted main `0e7741b76aff99df65c8db993f45064cab34b42a`.
Source inputs: `a3c9b22fe01f19e6c29247bac32095ca83ea23640a603d1bdbcdc08c5cdf5b17` (55965 entries).
The release-only composition preserves that main's runtime and Cargo inputs. Historical PR566/7b evidence is a separate source-bound result and is not reused here.

The fresh controlled v3 build used Rust 1.97.1, CUDA 13.1.115 and the fingerprinted input view inside bubblewrap. The input-view identities matched before/after compilation. The six ELF outputs are identified in `build.json`; generic runtime cells execute kernel-check, argmax-margin-probe and run-spec. Building memra-server, run-gen and tok-parity is not an HTTP serving or tokenizer-parity run.

All five canonical generic cells passed on one RTX PRO 6000 Blackwell Server Edition, physical GPU0:

| Gate | Result |
|---|---|
| Kernel | 110 cells, 103 executed, all 10 required executed, 7 skips within the unchanged ceiling of 11 |
| Ornith 1.5 35B argmax | 1 flip, 0 bad under the existing calibrated gate |
| Ornith K=1..8 | 8/8 self-consistency, identical to plain target |
| Qwen3.8 27B argmax | 0 flips, 0 bad |
| Qwen K=1..8 | 8/8 self-consistency, identical to plain target |

The actual named 9B/35B oracle files and roster model hashes are in `run.json`. `router-baseline-proof.json` binds both exact positive lines in `cells/1-kernel.log` at lines 515 and 516: real 35B router, 32 m-points 1..2048 and m-invariance, each with zero mismatches. No separate router job was needed, and kernel counts alone are not the router proof.

The completed physical-card lease records wrapper/child exits 0/0, no timeout, no interruption and no lingering compute. Source, all six executables, model files and hardware observations matched before/after. Raw cells and 250 ms telemetry are sealed with the record. This is a correctness result, with no throughput/timing claim.

Sealed record SHA-256: `0142a120855efc26905aa2ca19d88aa3aeb38114352a1e5ffd093258737cb973`.
Build record SHA-256: `41cf55cdd1f43c40f510f15b5568c02f088d23f83483fe7cdbd7d8caadd44af6`.
Router descriptor SHA-256: `764fbde7db248664b1c6a95ee7c64a661a0a8def993213bf23ba37dd805e52e8`.

Publication retains this tested source/build and adds only qualification evidence and an index append. It does not qualify a newly rebuilt binary. The scoped result is Ubuntu24 generic correctness; Ubuntu22 native evidence and an independent matching rebuild are absent. Actual release/tag still requires both shipping profiles and matching rebuilt ELF bytes. No broader model, serving-shape, Step or PrimeGraph qualification is implied.
