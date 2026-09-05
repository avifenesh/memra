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

RTX PRO 6000 component timing, one-load full-model scalar/tiled identity,
sampled serving and the long-context prompt ladder remain required. The separate
scalar 256K run was launched before this candidate and cannot qualify its speed.

Current primary reference: SGLang's V4 serving article separates active C4 host
offload, C128/SWA residency, fused metadata and long-context scoring/selection:
https://www.lmsys.org/blog/2026-04-25-deepseek-v4/ (read 2026-09-05).
Those architecture directions do not transfer its SM100/Hopper kernel timings
or cluster-launch capability to RTX PRO 6000 SM120.
