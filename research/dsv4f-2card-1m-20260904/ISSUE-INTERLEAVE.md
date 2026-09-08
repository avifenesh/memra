# TP/EP issue interleave: exact, flat, removed

**NO-GO for this phase-interleaved submission arm.** It removes the rank-start
skew and most rank-0 AR residence, but neither sampled ABBA improves throughput.
The default-OFF gate and phased dispatch were removed after the measurements.
The original rank-serial walk remains. No kernel or numeric class changed.

## Code-confirmed issue order

At base `f805537001642e2028908da23412337d339c70b5`:

- [`dsv4_gpu.rs:9553`](https://github.com/avifenesh/memra/blob/f805537001642e2028908da23412337d339c70b5/crates/memra-engine/src/dsv4_gpu.rs#L9553)
  loops over each layer, then rank 0 and rank 1. Each rank submits its complete
  attention body before the next rank starts.
- [`dsv4_gpu.rs:9663`](https://github.com/avifenesh/memra/blob/f805537001642e2028908da23412337d339c70b5/crates/memra-engine/src/dsv4_gpu.rs#L9663)
  submits the post-attention/MoE bodies in rank order, followed by the expert
  reduction and serial shared tails at line 9712.
- [`dsv4_ep.rs:224`](https://github.com/avifenesh/memra/blob/f805537001642e2028908da23412337d339c70b5/crates/memra-engine/src/dsv4_ep.rs#L224)
  submits the two out-of-place one-shot reductions in rank order.
- [`tp_ar.cu:209`](https://github.com/avifenesh/memra/blob/f805537001642e2028908da23412337d339c70b5/crates/memra-engine/cu/tp_ar.cu#L209)
  waits for both ranks before computing `in_rank0 + in_rank1`, then crosses an
  end barrier before the input buffers can be reused.

The candidate alternated ranks across six attention phases, five post-attention
and expert phases, and three shared-tail phases. Full-program callers used the
same bodies without a phase restriction. Rank-local buffers and each stream's
kernel sequence stayed unchanged. There were no new threads or host joins.
The process-local setter was default OFF, read once per token, with a
2026-09-22 decide-by date. Its gate-only environment selector accepted `0`/`1`.

## First-kernel and AR measurements

The saved baseline is private namespace `compose-sampler-diet-nsys-638eaa0-r1`.
The candidate is `issue-interleave-dcc1e9f-r1`. Each first SQLite capture contains
32 tokens, 43 layers per token and 2,752 AR calls per rank. The first attention
HC dot kernel is paired by token and layer, before considering the AR. Layer 0
is anchored after embedding so the previous token's head cannot be mistaken
for the next layer. All 43 per-layer distributions are retained privately.

| Profile metric | Saved baseline | Interleaved |
|---|---:|---:|
| First-kernel start skew median, us (N=1,376) | 103.102 | 1.0425 |
| First-kernel start skew p10 / p90, us | 49.468 / 194.440 | -0.600 / 1.923 |
| AR start skew median, us (N=2,752) | 49.9435 | 9.9435 |
| AR end skew median, us | 0.7155 | 0.7270 |
| Rank-0 AR duration median, us | 58.112 | 17.504 |
| Rank-0 AR total, ms/token | 5.220086 | 2.022702 |
| Rank-0 other kernels, ms/token | 19.532518 | 19.515338 |
| Rank-0 time without a kernel, ms/token | 6.365735 | 9.474545 |
| Forward GPU span, ms/token | 31.118339 | 31.012585 |

Baseline layer-0 first-start skew median is 130.2195 us (N=32). The lag is
already present at the start of attention, not created only inside the AR.
But eliminating its residence in the reduction does not eliminate step time:
3.197384 ms/token leaves rank-0 AR while 3.108810 ms/token appears in gaps.
Both profiles have the same non-sampler kernel counts, 2,706.25 per token on
rank 0 and 2,712.25 on rank 1.

Forward GPU span runs from the first embedding kernel to the final non-sampler
kernel. Gaps can include copies. The baseline uses device sampling and diet;
the candidate profile uses default radix and diet. These instrumented captures
explain residence, not an unprofiled throughput delta. The ABBA below fixes the
sampler and binary within each comparison.

## Correctness and sampled ABBA

Hardware: two RTX PRO 6000 Blackwell Max-Q GPUs, driver 595.71.05, CUDA 13.1,
sm_120a. All runs held the shared pair lock; the process census validates only
this lane's binaries. Fresh model loads precede every five-row process. Raw
250 ms temperature, clock, power and utilization telemetry is retained.

Both `dsv4_tp_ep_gate` arms passed deterministic continuation, canonical
attention sums, state identity, and all six refusal cells: ranks 0/1, codes
40043/40044, positions 1/3/127, unchanged committed caches and refused retries.
The output SHA is identical:

`bd13bc55fd7fe7a6dc1829b036200a768d6d344b94ca45d8747da76b5835d7f2`

Each ABBA is four fresh processes in new/current/current/new order, five rows
per process. Each row primes 256 source tokens and samples 256 output tokens
with temperature=1, top_p=1, top_k=0, seed=20260907. Attention TP2, radix and
small-kernel diet are fixed ON. The second ABBA also enables `--moe-m1-splitk`
on both arms. The timer is `sample_plus_forward_envelope`; state hashes are
outside timing. Profile runs are excluded. Pooling divides total tokens by
total elapsed time, not by an average of row rates.

| Program | Current tok/s | Interleaved tok/s | Delta | Rows per arm |
|---|---:|---:|---:|---:|
| Radix, split-K OFF | 38.072290 | 37.963496 | -0.285757% | 10 |
| Radix, split-K ON | 41.045194 | 40.991187 | -0.131580% | 10 |

Forward-only prime is also flat: 43.840285 to 43.970330 tok/s with split-K OFF,
and 47.766138 to 47.905911 with split-K ON.
All 40 sampled rows are eligible, non-looping, complete 256 forwards and have
zero AR refusals. Tokens, logits, cache and hidden state are repeatable within
each arm and bit-identical across issue-order arms in each numeric class.
The validator checks actual split-K program/dispatch counts and recomputes
token hashes from raw token IDs.

- Split-K OFF token SHA:
  `35e9e69e90047d273f266b7a4a71e0848d403be66eeebfcb1fe89a37d1caa433`
- Split-K ON token SHA:
  `3479b985a1f5f0bd08b7718781f64d00e665bfac9f2aa1bed59c1b2889816cb2`

The two split-K settings are different numeric classes; their hashes are not
required to agree with each other.

## Evidence and decision

Measured source: `dcc1e9fd56b1b0a44f14106a953039c25e78f6e6`.
Correctness binary SHA256:
`061321bb60ac1ea92285cc887a27fe299859cf56839ef0262a522c42112a64d5`.
Sample/profile binary SHA256:
`c8abda436076b0f66e08a56136ada44e3e0f4a234b6df9fa436acd5844c75302`.
The controller completed with EXIT=0 at 2026-09-08T04:44:20Z.

Raw receipts remain private in `issue-interleave-dcc1e9f-r1`: controller,
build and gate logs; source/binary hashes; hardware/process census; telemetry;
correctness and ABBA summaries; profile pairing and residence analyses; Python
validators; five NVTX reports and the first exported SQLite capture. The
superseded `issue-interleave-f564b8b-r1` contains build-only evidence and no GPU
cells. No raw model output, profile or deployment identity is in this repo.

The result rejects this phase-order accelerator on the measured shape. It does
not establish that host submission overhead is irreducible. Reordering the
same launches aligned the ranks but did not reduce their total submission cost.
The experimental selector, setter, phased wrappers and harness arm are removed;
this report and the FLAGS removed-door ledger retain the decision.

One historical correction: the rejected scoped-worker patch already had a
whole-layer walk inside each worker and joined once per token. Its archived
patch lines 242 and 373-402 and the existing
[`tp-worker-submission-20260907.md`](tp-worker-submission-20260907.md) agree.
The earlier description of a per-layer join was wrong. That result used an
older program and a non-contemporaneous control; it does not qualify or reject
every possible persistent-worker design.

No local cargo, CI or GPU runs were performed. Pushes used exported
`MEMRA_SKIP_PERF_CI=1` with normal hooks. Hosted build, Clippy, engine/server
unit tests, architecture coverage, publish dry-run and policy gates passed on
the measured source. PR #352 remains draft and unmerged because there is no
performance win to promote.
