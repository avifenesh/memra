# GLM verify shared-expert dual, 2026-09-09

Verdict: **NEGATIVE against the fixed >=0.5 ms weighted saving criterion**.
The candidate is byte-exact, but saves **0.247601569 ms/round**
on the fixed measured-width mix. The door and its code are removed in this lane.
rev: 2026-09-23

## Plan reading and scope

The banked F16 follow-up means pairing the shared expert's gate/up input
projections through existing `matmul_decode_exact_dual` at t=2..4. The
activation, down projection and final routed-output add retain their current
program. t>=5 stays current. This removes one duplicate input quantization
and one projection launch per eligible routed layer, or 42 of each per round.
It does not introduce a different all-row down kernel. The previous shared
expert side-stream overlap, PR #221, was killed after -1.07% c1 and was not
revived. The owner task's >=0.5 ms weighted criterion supersedes the banked
plan's >=0.3 ms eligible-round target. PLAN.md records this reading before code.

The measured source is `41ba71861353b66a8c7807ff882b5b7ed11de930`, based on
`72aa777c3`. Candidate binary SHA256:
`c8680e865758c09aa627d97c0b759010e2c50b272926925a3add02eb2cea8112`.
Hardware: one B200, CUDA 13.1.115, sm_100a. Every GPU phase held the common lock.
All 31 staged mint/drafter files matched the F16 archive's hash manifest.

## Oracle FIRST

Replay uses the F16 cell's exact t2/t4 binary activations, expert IDs and route
weights, from full archive SHA256
`3d04e9470cbd0250e3c8394b3bf952415028d4514fc85480b0b47b585b2ac01d`.
For t7 the surviving native input trace captures the same mint and p4k-prefix
short-prime recipe. No production DFlash2-round capture or serving claim is
implied. All 42 routed layers, numbered 3..44, use 4096->2048->4096 routed
FFNs, top8, preclamp10. Both arms consume the same captured inputs and routes.

| t | Layers | Gate/up and composed-chain bytes | Complete-FFN row argmax | Direct component engagement checks |
|---|---:|---|---|---:|
| 2 | 42 | All identical | 84/84 PASS | 42/42 |
| 4 | 42 | All identical | 168/168 PASS | 42/42 |
| 7 | 42 | All identical | 294/294 PASS | 0/42, current control |

The four checked stages are gate, up, shared FFN output after activation/down,
and full FFN after routed+shared addition. All values are finite. Every stage
has normalized mean/max delta **0/0**, with zero differing f32 bit patterns.
The exact zero-delta band is stricter than a numeric-class tolerance: equal
bytes necessarily have equal errors against the same fixed f64 reference;
that reference did not need recomputation. No band was relaxed. The raw oracle
contains 336/672/1176 row records at t2/4/7, all PASS. The existing dual reports
eligible at every t2/t4 layer and ineligible at every t7 layer; the runtime
component counter independently proves engagement. Layer20 at t4 also passes.

## Warmed, interleaved complete-FFN timing

All three complete oracles passed before the first timing call. Oracle and
bench binaries have the same SHA256. Each layer warms with 40 current+dual
pairs, then five ABBA blocks, with 100 complete FFN chains per arm position.
There are 840 timing records per width, 2520 total. Same resident weight
copies and route tables are used by both arms, with no weight upload inside
the timed region. CUDA completion is synchronized before and after each
100-chain position. No competing build or GPU phase runs during timing.
250 ms telemetry is retained; power and clocks are metadata only.

Each block's two same-arm positions are averaged per layer, then all 42
layers are summed. Current and dual columns are medians of those five sums.
Saving is the median of the five paired differences, so it need not equal
the difference between the two independently reported medians.

| t | Current, ms/round | Dual, ms/round | Paired saving, ms/round |
|---|---:|---:|---:|
| 2 | 6.089077715 | 5.761392505 | 0.327543050 |
| 4 | 10.381785015 | 10.145291400 | 0.236696805 |
| 7 | 17.214927655 | 17.213959910 | 0.000874255 |

Weights 48/162, 103/162, 11/162 give **0.247601569 ms/round**.
Crediting the unchanged t7 control zero gives **0.247542206 ms/round**.
Both are below 0.5. The t7 paired differences cross zero and are control noise.
The profile's other 30/192 rounds at widths 3/5/6 are outside this fixed cell;
these are conditional measured-width totals, not a whole-traffic speed claim.
The component excludes the router, attention, draft and HTTP work.
`per-layer-timing.tsv` retains all 126 layer-width summaries; `summary.json`
retains every block total and paired difference.

## Removal and validation

Removed `MEMRA_GLM5_SHEXP_DUAL`, both verify dispatch sites, the component
helper/counter, the real-input gate/bench executable and its shell driver.
No new kernels were added, and the existing decode-exact dual remains for its
other callers. The final engine tree and KERNELS.md match base72aa777c3 exactly.
The FLAGS.md Removed doors ledger retains the verdict. No serving config or
default was installed.

The candidate release build and remote formatting passed. Clippy initially
found only a harness length-check style issue; its `.is_multiple_of()` fix
passed Clippy -D warnings without rebuilding the measured binary. The exact
patch is in raw/harness-lint.patch. Both oracle and timing retain the original
binary/source pin. Final removal build, formatting and Clippy -D warnings passed remotely.
Final engine unit suite: 449 passed, 0 failed, 19 ignored. Raw final-* logs
and the empty final-engine-diff.txt retain the evidence.

No cargo or GPU qualification ran on the rig. Push uses MEMRA_SKIP_PERF_CI=1;
formatting hooks and validation execute on the qualification host. PR #383
stays draft. The serving pair was not accessed.

## Raw custody

Full private archive: `raw-receipts.tar.gz`, SHA256
`99aabe104729c52b565113f0fc3ee3bfb96afff9abcf713129f4bccc70406aab`,
under Darklanes `research/glm5-1m-b200-ship-20260906/receipts/dflash2-20260908/shexp-dual-20260909/`.
Public text archive: `raw-text-receipts.tar.gz`, SHA256
`8b2e6f3e77013d143d4cb3243709ee1a5b159543e03f4d3ae645c03476f0d837`. Archive members were checked against the raw manifest.

`summarize.py` reconstructs the tables from raw TSV files; `raw-sha256.json`
hashes the full private raw members. The measured executable source stays in
Git history. The public archive retains text receipts, and the full private
archive retains binary activations and exact routes. All completed shapes were copied back; final archive hashes bind their custody.
