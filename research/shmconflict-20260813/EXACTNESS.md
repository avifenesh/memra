# Candidate v1 exactness battery

Date: 2026-08-13

Rig: local RTX 5090 Laptop GPU, owner-capped 210--1200 MHz

Candidate source SHA-256: `b02e951ebb44aac43220204deac1c88fbd0706131dc965c966a5b1564577ad4b`

Raw evidence: `raw/candidate-v1/gates/`

The battery ran under one uninterrupted `/tmp/memra-5090.lock` hold. GPU 0 was idle before the
first command and after every cell. All source, binary, model, draft, and prompt hashes matched
the pinned preflight values.

| Gate | Q27 | Q35 |
|---|---:|---:|
| Model-backed `kernel-check` | ALL GREEN (107 cells, 3 skipped) | ALL GREEN (113 cells, 1 skipped) |
| `run-gen` prefill/decode | argmax 8160 MATCH | argmax 8160 MATCH |
| `run-gen` batched/tokenwise | argmax 8160 MATCH | argmax 8160 MATCH |
| `run-spec` K=1..8 | 8/8 PASS | 8/8 PASS |
| Chunk invariance, T=97 and T=149 | 4/4 comparisons EXACT | 4/4 comparisons EXACT |

The combined required-cell manifests also finished `ALL GREEN (106 cells, 1 skipped)`.
Chunk-invariance compares chunks 64 and 32 against 2048 on each prompt; all logits were
bit-identical and all 48-step token streams were identical. No `MISMATCH`, self-consistency
failure, differing chunk, nonzero process exit, or residual GPU process occurred.

This establishes local candidate correctness. It does not replace the required pre-release
battery on the Vast 2x RTX PRO 6000 verification box.
