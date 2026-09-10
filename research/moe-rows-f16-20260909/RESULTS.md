# F16 rows real-input cell, 2026-09-09

Verdict: **NEGATIVE, numeric no-go**. The complete-chain argmax oracle failed.
Timing was refused, so the >=0.5 ms weighted saving criterion is unmeasured.
The F16 door and all code reachable only through it are removed in this lane.
No serving dispatch or default was installed. PR #294 stays draft.

## Source and scope

- Baseline: main 72aa777c3; candidate measurement source bc89f4d22.
- B200, one visible device, CUDA 13.1.115, sm_100a. Every GPU phase held the
  common lock and used device 0. The replacement qualification host was non-bid.
- Candidate binary SHA256:
  058317b24caf0797d0496a86417676c09fd797643883949e9a225e257dd53c10.
- Mint/drafter staging verified 31/31 files. Staging and input metadata hashes,
  exact token IDs, per-layer f32 captures and binary expert IDs/route weights
  are retained in the full private raw archive; public text receipts retain the
  oracle rows and hashes.
- Inputs came from native short prompt passes through the mint, using prefixes
  of the staged p4k prompt. They are real model activations at the requested
  widths, not captures of production DFlash2 rounds. No serving qualification
  or sampled-generation claim follows this component gate.
- Shape: 4096 -> 2048 -> 4096, top8, preclamp10, all 42 routed layers captured.
  Both arms replay identical activations, expert IDs and routing weights.
  Current uses interleaved NVFP4 ILP rows with Q8 conversions; F16 uses per-row
  V2 weights and lane-major F16 conversions. Each down stage consumes its own
  gate/up output. Resident repack is checked separately from the numeric chain.

## Oracle result

| t | Layers checked | Error-band checks | Complete-chain row argmax | Result |
|---|---|---:|---|---|
| 2 | 3 through 44, all 42 | 84 passed | 84/84 passed | PASS |
| 4 | 3 through 20, 18 | 36 passed | 71 passed, 1 failed | FAIL at layer 20, row 3 |
| 7 | Not run | Not run | Not run | Refused after t4 failure |

The failing tensor is the routed **full-chain output**, t=4, layer=20, row=3.
Current argmax is **4**; F16 argmax is **1819**. The band checks passed:

| Full-chain error at layer 20 | Current | F16 |
|---|---:|---:|
| Normalized mean absolute error | 9.048931134194e-4 | 5.290322992813e-5 |
| Normalized max absolute error | 7.470310070901e-3 | 6.429610422100e-4 |

Errors are relative to the f64 chain of the same bytes, normalized by that
reference tensor's max absolute value. Required mean is <=1.05 times current;
required max is <=2 times max(current max, 1e-3). All values must be finite.
Argmax agreement is an independent required gate; no tolerance was relaxed.
Per-layer numbers are in `per-layer-oracle.tsv`; exact raw rows are in
`raw/t2/oracle.tsv` and `raw/t4/oracle.tsv` inside the archive.

The two synthetic F16 GPU tests also passed at t2/4/7, including accumulation
twins and the wrong-layout red arm. Synthetic success did not replace the
real-input gate.

## Harness correction

The first t2 attempt stopped at layer3, plane0, expert202, resident_repack.
The harness had selected `model::repack_nvfp4_split`, a whole-matrix split-plane
layout, as the reference for the device's per-row V2 layout. The documented
matching reference is `tp::nvfp4_matrix_v2_permute`. Commit bc89f4d22 corrects
that choice and adds replay of already captured inputs/routes. The successful
t2 replay used those exact saved captures, with unchanged numeric bands.
The first failure remains under `t2-initial-layout-mismatch` in the archive.
It is a harness failure, distinct from the subsequent genuine t4 argmax failure.

## Requested timing table

| t | Per-layer timings | Weighted saving, ms/round |
|---|---|---|
| 2 | Not run | Unmeasured |
| 4 | Not run | Unmeasured |
| 7 | Not run | Unmeasured |

No timing archive exists because the oracle failed before the timing phase.
The profile's measured-width weights were 48/162, 103/162 and 11/162 for t2/4/7,
with each routed layer weighted once per round. Another 30/192 observed rounds
used widths 3/5/6 and were explicitly outside this fixed cell. None of these
weights can turn an unmeasured saving into a performance verdict.

## Removal and validation

Removed the F16 and acc32 kernel twins, lane-major converter, Rust launchers,
synthetic and real-input gate code, bench arms, executable research driver,
and active FLAGS/KERNELS entries. The engine tree and KERNELS.md now match
72aa777c3 exactly; only the removed-door ledger and research evidence remain
in the final diff. The measured source remains in git history.
The original interrupted attempt is retained and marked concluded.

Candidate rebuild, formatting, Clippy -D warnings and the two synthetic GPU
checks passed remotely. The earlier engine unit suite passed 449 tests with
19 ignored. Final removal build, formatting and Clippy -D warnings passed; the final
engine unit suite passed 449 tests, 0 failed, 19 ignored. Exact output is in
the raw final-* logs.
No rig cargo, GPU gate, bench, smoke server or CI ran. Push uses
MEMRA_SKIP_PERF_CI=1; the formatting commit hook runs on the qualification host.

## Next exact candidate, plan only

Pair the shared expert's two input projections through the existing
`matmul_decode_exact_dual` at t=2..4. Gate against the current two singles on
identical real layer inputs with byte-identical projection and composed FFN
outputs before warmed, interleaved ABBA x5 timing. The mechanism should remove
42 duplicate quantizations and 42 projection launches per eligible round;
measure the complete chain and target >=0.3 ms saved per eligible round. Keep
t>=5 on the current path. This is the next exact candidate, with no implementation,
serving dispatch or new default in this lane.

## Raw archive

Full oracle and validation archive, including binary activations and routes,
is banked in private Darklanes at
`research/glm5-1m-b200-ship-20260906/receipts/dflash2-20260908/f16-component-20260909/oracle-receipts.tar.gz`,
SHA256 `3d04e9470cbd0250e3c8394b3bf952415028d4514fc85480b0b47b585b2ac01d`.
The public archive `oracle-text-receipts.tar.gz` retains every non-binary member
unchanged, SHA256 `6297642ca48e92dce9165237d936c51f5e8c6ad7dd88830ad741952613a8dca4`. The binary activations produced a
false-positive provider-name match in the public-boundary text scanner.
No allowlist or guard bypass was used. These are correctness receipts, not timing
evidence. `raw-sha256.json` hashes all full-archive members; `summarize.py` reconstructs the
verdict and per-layer table from the raw oracle rows. Each completed t-shape
was copied to the rig before proceeding. No lane process remains on the box.
The target directory is retained because disk space is ample, per owner request.
