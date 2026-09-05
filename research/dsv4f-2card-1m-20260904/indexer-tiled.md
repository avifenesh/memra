# Ordered tiled indexer candidate

2026-09-05. Default remains `MEMRA_DSV4_INDEXER_SCORE=scalar`.
The candidate is a Memra-native adaptation of the internal MLA head-blocked
scorer, not an external engine or kernel dependency. DSV4 uses separate rounded
multiply and add, so MLA's explicit FMA is deliberately not carried over.

One thread owns one candidate and all 64 indexer heads. A 128-thread CTA stages
16-wide query/key slabs, preserving x-ascending dot and h-ascending mix order.
The fixed-limit decode and absolute-position chunk masks are unchanged. No
weight, activation, KV quantization, top-k policy or selected-index order changes.

Local RTX 5090 Laptop correctness only, under `/tmp/memra-5090.lock`:

```sh
/usr/local/cuda-13.1/bin/nvcc -O3 -std=c++17 -arch=sm_120a \
  --expt-relaxed-constexpr -fmad=false -Xcompiler=-ffp-contract=off \
  tools/dsv4-indexer-tiled-gate.cu -lcublasLt -o target/dsv4-indexer-tiled-gate
flock -n /tmp/memra-5090.lock target/dsv4-indexer-tiled-gate
flock -n /tmp/memra-5090.lock target/dsv4-indexer-tiled-gate --teeth
```

Initial local gate: 2,045,982 exact score comparisons over candidate counts
1/127/128/129/4096/4103/65536/262147, widths 1/5/8/32/64, partial masks,
all-zero ties, CPU edge anchors, tail canaries and shape refusal.
The one-ULP corruption control exits 1. Raw receipts `indexer-tiled-local.log`
and `indexer-tiled-teeth.log`. Synchronization checking reports zero errors.
The v2 gate additionally captures the actual launcher and checks the function
pointer and 2x1/128 launch geometry; `indexer-tiled-local-v2.log` passes.
No local performance decision.

Candidate binaries, before target-card gates:

- standalone gate: `34081b97cf7fcce81eda2a7cb6048853662cad2d3620d13e0465113e41600cbc`
- model gate: `5f69f8bdcda519d4380dd53fdfdad827c559c4004de0f0cf5653e8b0c2a466be`
- server: `839fabe2aa15ded525b45afea6e5ce72daa88bca33b434a5fb9f9d6200b6e016`

Engine library tests: 391 passed, three ignored. Rewrite-gate tests: seven
passed. Five selected DSV4 serving tests pass. Release model-gate and server
builds pass. Final clippy including the server passes
(`indexer-tiled-final-clippy.log`). These are not full delivery or performance gates.

## RTX PRO 6000 results

Both cards pass 2,045,982 comparisons. Five interleaved component repeats give
the following medians in microseconds (`indexer-tiled-pro-0.log` / `-1.log`):

| candidates | GPU0 scalar / tiled | GPU1 scalar / tiled |
|---|---:|---:|
| 129 | 8.006 / 27.130 | 7.974 / 26.771 |
| 4096 | 85.504 / 27.642 | 85.350 / 27.578 |
| 65536 | 1298.874 / 53.690 | 1297.056 / 53.978 |
| 262147 | 4918.381 / 171.347 | 4920.403 / 171.347 |

Thus the candidate is a short-history loss, about 24x faster at 65536 candidates
and about 29x at 262147 for this scorer at s=1. These are warm component timings,
not whole-model throughput or prefill measurements. Do not force it for short
histories or infer a wider-query dispatch knee without measuring those rows.

The one-load scalar/tiled full-model gate passes plain widths 1/32/64 and both
DSpark fused-MoE arms (`indexer-one-load-pro.log`). Eight short sampled warm/cold
cache twins also pass on the forced-tiled server. The chat fairness diagnostic
reached c9 before a container restart; it is not a complete c1-c16 result.

The scalar 262144-token run was deliberately stopped after 69 minutes without
finishing prefill. That is an incomplete capacity result and a failed TTFT
budget, not an engine crash. A fresh exact-tiled 262144-token engine run is now
active using gate SHA256
`333db6c13b5d7f18e965ccd7e55b40bba4ae630f014f1a882f68161d86328eb4`.
Actual 256K/512K/1M HTTP and performance qualification remain open.

Current primary reference: SGLang's V4 serving article separates active C4 host
offload, C128/SWA residency, fused metadata and long-context scoring/selection:
https://www.lmsys.org/blog/2026-04-25-deepseek-v4/ (read 2026-09-05).
Those architecture directions do not transfer its SM100/Hopper kernel timings
or cluster-launch capability to RTX PRO 6000 SM120.
