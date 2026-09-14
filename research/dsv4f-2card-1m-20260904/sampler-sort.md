# Exact CPU sampler ordering, 2026-09-06

The current sampled decode gate isolates 14.60/15.11 ms before the first plain
commit from ready logits at 256/8192-token context. Source inspection finds a
full-vocabulary comparison sort followed by ordered f64 probability arithmetic.
This lane now profiles and replaces the ordering operation, not the distribution
or deterministic mapping from position/seed to token.

## Contract and implementation

`MEMRA_DSV4_SAMPLE_SORT=comparison|radix` defaults to comparison. Configuration
is process-lifetime cached and validated by `Dsv4Gpu::load` before artifact or
CUDA access. Unknown and non-Unicode values refuse. A separate thread-local
gate-only override supports one-load interleaved experiments; clearing it restores
the process configuration. It is not a serving scheduler interface.

The native CPU implementation uses four stable byte passes over monotonic,
descending IEEE-f32 keys. Candidate IDs start ascending and stable scattering
preserves the old tie law. Positive and negative zero share a key. NaN-bearing
rows retain the legacy comparator instead of receiving a different total order.
Top-k truncation, ordered f64 max-shifted softmax, both normalization sums,
nucleus cutoff, position-keyed uniform and final CDF traversal are unchanged.
No third-party kernel/runtime is linked or installed.

Current [Rust float documentation](https://doc.rust-lang.org/std/primitive.f32.html)
confirms signed-zero and partial-order distinctions; using `total_cmp` unchanged
would not preserve this sampler's zero ties. The actual build remains Rust
1.97.1, not the newer documentation site's toolchain. The
[FlashInfer sampling design](https://flashinfer.ai/2025/03/10/sampling.html)
is a useful current primary-source comparison: distribution-equivalent unsorted
CDF/rejection sampling does not establish our seeded output identity, so it is
not substituted into this exact-order rewrite.

## Evidence

The original standalone prototype binary
`505400e344b174e28fa55be6737bf435b7db2a7ec67f60c3600f3ff3b93c6cf1`
passed 49,643,520 full candidate-order comparisons and 1536 sampled parameter
cases across 384 captured reference/matrix rows from three real code windows.
Synthetic controls cover signed zeros, infinities, random finite bit patterns,
ties, empty/short/warp/tail lengths and 129280 candidates. The local phase
profile, ABBA x3 over eight fixed rows per pass, places most sampler time in
ordering and shows a material reduction. Raw: `sampler-sort-local.log`.

The integrated gate calls the production radix-order and sampled functions,
retaining an independent copy of the old comparator and probability walk as
its instrument oracle. It repeats the complete order and sampled checks and
passes locally. Binary:
`bc8f4fed4467e39f62589fedcbd8c9c01690feaaf8f022b667af672d412b06a1`;
raw: `sampler-sort-integrated-local.log`. Full engine CPU suite: 407 passing,
11 ignored; server suite: 604 passing. The subsequent eight targeted sampler
tests also cover thread-local override isolation/reversal and child-process
unknown/non-Unicode boot refusal before artifact/CUDA access. Strict clippy,
formatting and the runtime-flag census pass.

## Target sequence and remaining gate

The target controller first hashes all six captured row files, runs the CPU
phase/full-order gate, then loads the model once for comparison/radix x
plain/DSpark measurements. Candidate decode binary:
`8891ec748bdb07ee85d8155f8e8eda298a3443ed1f0c7b830b6543f047a9be66`.
The protocol uses four warmups, then an eight-row balanced block repeated three
times per context: six observations per sorter/decoder combination. Contexts
256/8192, 256 sampled outputs, T=1/top_p=1/top_k=0/seed=20260906, active C4,
matrix/EP and tiled indexer/scorer are fixed. Warm comparison output must also
match the pre-rewrite frozen token hashes. The independent summary checks
all 56 rows, ordering schedule, token identity, loops, time windows and telemetry.

## Target result

The serving-host CPU gate passed at 06:44:39 UTC. Across its six captured-row
panels, median complete sampler cost fell from 14.39–14.88 ms per row to
2.92–3.16 ms. Ordering fell from 13.08–13.44 ms to 1.65–1.70 ms; the unchanged
probability/draw portion remained 1.21–1.46 ms. Raw CPU log SHA256:
`f3ec37f504da568c0c04bcb9b4535442711a1f272277dc48c2885515a14d7152`.

The full-model comparison completed at 07:02:34 UTC, status 0. All 56 rows
pass the independent audit, including both pre-rewrite frozen output hashes;
all 48 measured rows are eligible with no loops or early EOS. One owned GPU
process and 8548 telemetry rows are recorded; phase gaps are at most 263 ms.
DSpark rounds and accepted/drafted counts remain identical between sorters.

| Context | Mode | Comparison output tok/s | Radix output tok/s | Change |
| ---: | --- | ---: | ---: | ---: |
| 256 | plain | 20.5985 | 27.1861 | +31.98% |
| 256 | DSpark | 19.7617 | 25.5735 | +29.41% |
| 8192 | plain | 19.4340 | 25.0810 | +29.06% |
| 8192 | DSpark | 23.9565 | 33.0478 | +37.95% |

Each value is the median of six interleaved observations. Decode-wall includes
generation and final drain, excluding prefill/restoration, allocation, hashing,
detokenization and logging. These are sampled engine rates, not HTTP throughput
or a comparison against the older greedy baseline. Plain first-commit latency
from ready logits falls from 15.05/15.06 ms to 3.20/3.56 ms at the two contexts.

Raw model log SHA256:
`9dc8a1a89523ca111a368a2063fee5f5068479711b1e45f4c425deddc33096da`;
telemetry SHA256:
`3b4672efa182c6b4fe864401ff8935da402ad2db976a6f4b1b996bb49e869e1b`.
Raw rows, process/controller logs and audited summary are banked in the private
companion lane as `sampler-order-20260906-*`.

The subsequent explicit GPU decode profile captures 32 sampled plain steps
after warmup at 8K, with a frozen 96-token output check. Profile binary:
`f6277b88756a7da67729af28f9e8903ffcd2035c99c03ad569e72cff3b317c4e`.
This separates remaining GPU decode costs after the host sampler improvement.

The default remains comparison pending the sampler's HTTP/release gates.
Actual 1M serving, c1–c16 scheduling/fairness, live graphs, broader matrix
quality and the overall peak-speed objective remain open.
