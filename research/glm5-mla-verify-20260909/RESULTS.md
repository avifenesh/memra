# GLM5 MLA verify split-KV, 2026-09-09

## Current disposition, 2026-09-09

Component KEEP OFF, unserved, code removed 2026-09-09. The owner removed
the default-OFF door, dispatch/workspace allocation, dedicated CUDA entry
points, Rust FFI and oracle/bench source. Measurements below describe the
archived component, not a retained switch. Re-derive from
`c776611e78291aa8ff36c7323d5c216eab961fa7`. No served pair qualification.
rev: 2026-09-23.


Verdict: **KEEP, 3.700307708 ms/round weighted saving**. rev: 2026-09-23

All 18,304 real latent-row argmax checks passed before timing. Warmed ABBA
x5 clears the fixed >=0.5 ms/round bar on both prompt sizes independently.
`MEMRA_GLM5_MLA_VERIFY_SPLITKV` stays default OFF, decide-by: 2026-09-23.
This is selected-attention component qualification; sampled candidate
serving and a default rollout remain separate gates.

## Mechanism and candidate

The current kernel launches one 256-thread CTA per (query, head), all t
rows in one grid. PP1 has 64 heads: t2/4/7 gives 128/256/448 CTAs per MLA
layer, with 11 separate layer launches. Each CTA serially walks 257
8-slot tiles over the captured 2051-slot list, paying two block barriers
per tile. The list expands 512 four-token pools plus tail capacity; heads
share it. This is the recorded serial softmax/barrier and L2 traffic wall,
not a measured HBM bandwidth limit.

The candidate adds eight tile-aligned slot partitions, retaining the
current shuffle-down dot and eight-slot online fold inside each partition.
One combine kernel merges unnormalized (m,l,acc) partials. It does not
re-enable the old per-slot/xor warp-online arm that failed the t4 argmax
gate. No indexer scoring or pool selection code changes. Unsupported head
shards, ranks, RoPE layouts, widths or pool budgets use the current path.

`MECHANISM.md` pins file:line evidence, tptrace9/10's 49 us t1 decode
kernel on 32-head TP shards, the 64-head shard-scaling rule, and PRs
#307/#308. The p32k tally's t2/t4/t7 gathered times are
694.218/656.218/952.051 us per layer. Those profile means and the TP t1
number are different dispatch/head shapes, not a controlled t-scaling
experiment.

Before coding, the t4 screen was 656.218/8 + 10 = 92.02725 us/layer,
or 192.02725 with an additional 100 us overhead allowance: estimated
6.206098 or 5.106098 ms/round saving. Actual t4 saving is about 3.1 ms.
The larger grid still performs the tiled work across additional CTA waves;
the serial-time screen overestimated the gain. The predeclared KEEP bar,
not that estimate, decides the result.

## Identity and capture

Measured source: `c5c2364a50ccdffc67fa02f1541ee907b7f449f8`, engine base
`628521007`. One B200, sm_100a, CUDA 13.1.115. Every capture/oracle/bench
GPU phase held `/tmp/memra-gpu.lock`. No cargo or GPU work ran on the rig.

Oracle/bench executable SHA256:
`dfaef8fd744775f0565906bfd875bc1a5122163ddeb7b11adddeefaa72f23696`.
Capture server SHA256:
`bf0a9cc7d4118dc02e32967adadf3a2f01cc23736a1cc0a3ec96387e88efa005`.
The capture server is the control with `capture-instrumentation.patch`;
the final runtime contains no capture hook. The measured CUDA source is
`abf8526c4a98987b8272431f62dcba6118dd0de8580ce3be2cf12f811e4f317a`.

The captures use the native GLM5 mint and DFlash2. Both requests omit all
sampling parameters, inheriting temperature=1.0/top_p=0.95 from metadata;
K cap=6, PMIN=0.7. First-observed width captures cover trunk MLA layers
3,7,11,15,19,23,27,31,35,39,43. Every captured shape is 64 heads,
rank512, rope0, 2051 slots. Full source cache, index, query and metadata
bytes are retained; both oracle arms upload those same files.

| Prompt | Prompt/output tokens | Spec rounds | Drafted/accepted | HTTP/SSE |
|---|---:|---:|---:|---|
| p32k | 29781/256 | 112 | 262/143 | 200, terminal DONE |
| p128k | 128073/256 | 107 | 237/151 | 200, terminal DONE |

The first p128k attempt returned HTTP400 with the reused metadata's
60000-token maximum. Its retry used a lane-local 140000 limit but returned
HTTP429: `request context 128393 does not fit available KV capacity`.
The log charged 6395MB plus admission reserve against 7562MB available.
The completed p128k text-only capture disables the unused 2.09 GiB image
tower with the existing vision rollback. Text weights, sampling and
attention geometry are unchanged, and admission checks remain active.
Both failed attempts remain in the private archive and supply no timing.

## Oracle before timing

The fixed per-row band is max absolute error <=
`1e-5 + 1e-4 * max_abs(reference row)`, alongside finite outputs and
identical argmax for every (query, head) latent row. Gather validation
compares GPU-gathered values bytewise with CPU copies from the captured
source cache and the original index list. The candidate partitions cover
the same indices, retaining the original slot order within each partition.

| Prompt | t | MLA layers | Latent rows | Argmax flips | Max absolute error | Gather changed bits |
|---|---:|---:|---:|---:|---:|---:|
| p32k | 2 | 11 | 1408 | 0 | 7.9870223999e-06 | 0 |
| p32k | 4 | 11 | 2816 | 0 | 8.46385955811e-06 | 0 |
| p32k | 7 | 11 | 4928 | 0 | 1.45435333252e-05 | 0 |
| p128k | 2 | 11 | 1408 | 0 | 8.94069671631e-06 | 0 |
| p128k | 4 | 11 | 2816 | 0 | 7.7486038208e-06 | 0 |
| p128k | 7 | 11 | 4928 | 0 | 1.25169754028e-05 | 0 |

All bands pass; the worst error/band ratio is 0.058958832. The harness
writes `ORACLE_PASS` only after all 66 cases pass and returns before any
benchmark on an oracle failure. Seven unsupported geometry calls return
40023 before touching null pointers. Raw rows are in `oracle.tsv` and
`gather.tsv`; changed output bits are reported, not called byte-exact.

## Warmed ABBA x5

For each width/context, all 11 layers' captured inputs remain resident.
Twenty complete A/B pairs warm the rotating component, followed by five
ABBA blocks with 20 complete 11-layer rounds per position. Position
boundaries synchronize through CUDA events. Candidate timing includes
both the partition and combine kernels. Uploads, allocations, model load,
other verify phases and HTTP overhead are outside this component timing;
partial workspace is preallocated in the harness.

| Prompt | t | Current ms/round | Split-KV ms/round | Paired saving ms/round |
|---|---:|---:|---:|---:|
| p32k | 2 | 7.277413607 | 2.239275932 | 5.038048744 |
| p32k | 4 | 7.205770731 | 4.114628792 | 3.091277122 |
| p32k | 7 | 10.475944042 | 6.746227980 | 3.729716063 |
| p128k | 2 | 7.256658316 | 2.238322496 | 5.018115759 |
| p128k | 4 | 7.177082300 | 4.112799883 | 3.063913584 |
| p128k | 7 | 10.493181705 | 6.748387814 | 3.744752884 |

Columns are medians of five paired block values. Each block averages its
two A and two B positions; the saving is the median paired difference,
so it need not equal the difference between the two displayed medians.
Conditional width weights 48/162,103/162,11/162 give p32k
**3.711449136 ms/round** and p128k **3.689166279 ms/round**. Equal prompt
weights give **3.700307708 ms/round**. Widths 3/5/6 are outside this fixed
weighting; it is not a whole-traffic throughput estimate.

The additional per-layer ABBA x5 uses 20 warm pairs and 50 launches per
position, recording all 66 hot-layer cells in `per-layer.tsv`. The rotating
component decides KEEP. `bench.tsv` retains all 1440 positions;
`summary.json` retains the five block values and `summarize.py` reproduces
them. Telemetry at 250 ms spans 37-53 C during the successful p128k capture
and full oracle/bench cell; the earlier p32k/failed-attempt cell spans
38-49 C. Power and clocks are receipt metadata.

## Validation and custody

Remote formatting and final all-targets release Clippy with warnings
denied pass. The engine unit suite passes 457 tests, 0 failed, 19 ignored.
The final changes after the measured source are documentation, capture
retry plumbing, and restoring the FFI declaration's documentation anchor;
CUDA arithmetic and the oracle/bench executable are unchanged.

Push uses `MEMRA_SKIP_PERF_CI=1` per the owner's no-rig-cargo instruction.
Hosted checks remain the draft PR gate. No serving-default flip is made.
Raw per-row/per-position records and the member manifest are committed
here. The complete cache/input/source/log archive remains in private
custody beside the Darklanes lane receipt, bound by its SHA256 below.

Archive SHA256: `e11d219bc5820499d3aaca21f7da2ed2deaef0d9f22d9ebcf9b0bdce208d886e`

publicity: skipped - maintenance research record.
