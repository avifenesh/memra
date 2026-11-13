# numdiff P1: which kernels a change can make compute different bits, without a GPU (2026-11-13)

Verdict: sound on the history it was checked against (0 false IDENTICAL in 37 cases), with low proof
coverage. `tools/numdiff.py` compares the SASS ptxas emits for two revisions (or two kernels of one
build), with the kernel attributes, constant data, relocations and the host code of `-c` units,
and gives IDENTICAL only when the compiled code cannot compute different bits for the same inputs
and launch parameters. It names the float class that moved (contraction, approximation,
precision, tensor core, reduction, other) when it cannot. The tool's docstring has the full
contract and what it does not cover.

## Validation

37 cases from memra history with GPU ground truth: 14 where a GPU gate saw different bits
(negative) and 23 where one proved equal bits (positive), base-vs-head commits and
kernel-vs-kernel twins. `cases.json` is the public part of each case (commits, files, kernels,
arch, cause); `validate.py` is the harness; `cases-table.md` has the verdict per case.

| kind | n | result |
|---|---|---|
| negative-incident | 9 | 5 caught with the recorded cause, 4 caught |
| negative-rounding | 5 | 4 caught with the cause, 1 uncovered (`rms-block-1024-launch`: a launch-size change in Rust; the kernels are identical and numdiff says the host side is not covered) |
| positive-refactor | 10 | 2 proven identical, 8 not proven (the code moved; equal bits need an equivalence check) |
| positive-twin | 13 | 0 proven (two different kernels; equal bits need an equivalence check) |

- No negative case is IDENTICAL.
- The cross-file `-fmad` incident (`tpwalk9-hcpost-crossfile-fmad`) is caught with fma-contraction.
- Its twin form (`zq8v1-crossfile-fmad`) is caught with the `-fmad=false` flag difference named. The contraction class does not show, because the fused twin's extra work moves every float count the same way.
- A change takes 1 to 26 s when its build commands are cached. A fresh capture (one `cargo check` of memra-engine) takes minutes, mostly waiting on the shared build lock.

Two reviews checked the tool for false IDENTICAL claims, with toy repros:
- The first found 30 issues, among them register tuples, byte selectors, a unit built twice, ADDED/REMOVED kernels, host launchers in `-c` units, twin data, kernel attributes and cache keys.
- The second found 5 more: a host symbol swap, a `.cu` included by a unit, the nvcc environment, `.cargo/config.toml` seams, and non-`.cu` kernel inputs.

All are fixed, each with a test that fails on the code before it. 55 tests in `tools/test_numdiff.py`, run with `python3 -m unittest tools.test_numdiff`.

`validate3.log` ran the first fixed version over all 37 cases. `validate3-dsa.log` reran
`dsa-score-tc-bf16` after the capture learned to accept a tree whose lib does not compile.
`validate4.log` reran the six cases with `-c` units after the second round; only
`dsv4-shared-array-linkage-rename` moved (to PROVEN).

## Next

P2: equivalence over the reals for the kernels P1 marks changed, to prove the 21 positives this
cannot, and to split rounding-only changes from math changes.
