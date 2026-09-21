# Lockstep CPU experts: exact cross-stream program (memra#577 follow-up)

Verdict: the companion's multi-row arm is exact now (bit-identical to M=1, proved on CPU and on
Hy3) and ships opt-in; it is not the throughput winner on the receipt host, so the one-job-per-row
default stands with the default question's missing gate named. The rows kernel already produced per-row bit-identical down projections;
what broke exactness was the fold. `memra_cpu_moe_token_v2` folds each expert as
`sum = fma(y, route_weight * down_scale, sum)` in job order from zero, while the scaled rows
twin returned `y * down_scale * w` and the Rust side added tickets in expert order. The new raw
twin (`memra_cpu_expert_rows_raw_v2`) returns bare `y`, and `cpu_experts::accumulate_expert_exact`
re-folds each row in its own selection order with the one-job arithmetic (`f32::mul_add`, one
f32 scale product, zero start). Exact by construction; proved bitwise on CPU and on Hy3.

## Proof 1: CPU, `cpu_native_check` (runs in ci.yml `engine-tests`)

New arm "multi-row raw + exact accumulate == one-job program": a 3-expert one-job call with
route weights `0.37, -0.61, 0.9` and down scales `0.7, 0.3, 0.9` (non-powers of two on
purpose) against raw rows plus the exact fold, bitwise, for Q2_K, IQ3_S, Q4_K and NVFP4; plus
row 0 of an `m_r = 2` raw call against the `m_r = 1` call (amortization changes no bit).
PASS on the local rig (`raw/cpu-native-check-local.log`) and on the box. The pre-existing
scaled-rows identity arm only ever coincided because its weights (`0.5, -0.25, 0.125`) are
powers of two, which is also what hid the divergence in the lockstep harness until #577.

## Proof 2: Hy3 on a rented RTX PRO 6000 (`raw/abrows/`)

Rig: one rented RTX PRO 6000 Blackwell WS (97887 MiB), AMD EPYC 9B14, 188 GB container RAM,
direct-io disk 2.7 GB/s write and 5.1 GB/s read, 2026-09-21. (A first box with 125 GB RAM
thrashed in disk wait on the first cell and was destroyed; see the acceptance note in the
memory corpus.) `Tiyuvta/Hy3-NVFP4@0af425172b7a`, SHA256SUMS verified (README-only mismatch),
frozen residency, native companion built from this lane, greedy, 32 new tokens, stream 0 = P0.
base = origin/main `9ef2f04d6` (one job per row, #587's default), lane = this branch.
`cpu_native_check` on the box: the raw + exact accumulate arm PASS for all four formats
(`raw/cpu-native-check-box.log`).

| Cell (`raw/abrows/`) | Logits vs base M=1 |
| --- | --- |
| lane M=1 | IDENTICAL, 33 steps |
| lane M=2 mixed | IDENTICAL |
| lane M=3 mixed | IDENTICAL |
| lane M=4 mixed | IDENTICAL |
| lane M=4 same prompt | IDENTICAL |
| lane M=4 mixed, `MEMRA_LOCKSTEP_CPU_ROWS=0` (one job per row) | IDENTICAL |
| base M=4 mixed | IDENTICAL (main's default program, as #587 left it) |

Every M>1 cell reproduces the M=1 bytes with the multi-row arm on: the exact fold holds with
real routing, real sharing across streams, and the companion's amortized decode.

## Cost

Same box, same lane binary, same window (2026-09-21 11:33 to 12:33 UTC, GPU 26 C idle to 38 C),
M=4 mixed prompts, 64 new tokens, one process at a time behind the GPU lock. A = exact multi-row
arm (`MEMRA_LOCKSTEP_CPU_ROWS=1`), B = one job per row (the default). Five AB pairs, then five
BA pairs (`raw/abint/results.tsv`, per-run logs alongside; `raw/ab-interleave.sh`).

| Order | A median (tok/s aggregate) | B median | B over A |
| --- | --- | --- | --- |
| AB x5 | 1.85 | 2.30 | +24% |
| BA x5 | 2.24 | 2.70 | +21% |
| all 10 each | 2.18 | 2.48 | +14% |

The window drifts upward (later runs faster, page cache warming); the interleaving carries
that, and B wins in both orders. On this host (EPYC 9B14, 16 companion threads) the exact
multi-row arm pays one companion call per CPU expert, unshared experts included, where the
one-job program pays one call per row; the decode amortization does not buy that back at M=4.
The earlier Ryzen 9950X single run read the other way (+2.5% for the pre-exact arm). Per the
per-hardware rule the arm stays default off with its missing gate named: the same N>=5 A/B on a
9950X-class host. Both programs are exact, so the choice is throughput only.

Single-cell numbers from the exactness matrix (32 tokens, one run each, `raw/abrows/`): base
M=1 1.42, lane M=1 1.34, lane M=4 mixed rows on 1.83, rows off 2.01, base M=4 mixed 1.81,
lane M=4 same 2.71 tok/s aggregate.

## A/B after the dispatch fix (second box, `raw/abint-dispatchfix/`)

Review of the first measurement (revuto on #604) named a mechanical confound: CPU jobs were
submitted inline through the executor's bounded queue, so the dispatcher parked behind the CPU
work before launching the GPU groups, and the multi-row arm's larger job count made that worse.
Jobs are now prepared inline and submitted from a scoped helper thread. Re-measured on a second
EPYC 9B14 box (220 GB, disk 5.1/6.6 GB/s direct; 2026-09-21 13:39 to 14:47 UTC, GPU 42 C to 46 C),
same protocol (M=4 mixed, 64 tokens, 5 AB then 5 BA), lane head `ebb1fe43a` (the PR head
differs only by the local-ci verdict fix and receipts).

| Order | A median (tok/s aggregate) | B median | B over A |
| --- | --- | --- | --- |
| AB x5 | 2.73 | 3.13 | +15% |
| BA x5 | 2.76 | 3.12 | +13% |
| all 10 each | 2.75 | 3.13 | +14% |

Spread: A 2.69 to 2.77, B 3.10 to 3.16. The dispatch fix moved both
arms up (the first box read A 1.85 to 2.24, B 2.30 to 2.70 medians) and narrowed the gap from
about a quarter to about a seventh, but B, one job per row, still wins in both orders with
non-overlapping ranges. Exactness re-confirmed on this box with the new dispatch: lane M=1 and
lane M=4 mixed (arm on) bit-identical to base M=1 (`raw/abrows-dispatchfix/compare-report.txt`).

Verdict for the default: unchanged. One job per row stays; the exact multi-row arm is opt-in
(`MEMRA_LOCKSTEP_CPU_ROWS=1`). What would flip it: an N>=5 both-orders A/B on a 9950X-class host
showing the arm ahead, before 2026-10-05; otherwise the door and the arm go, keeping the raw twin
and the exact fold only if something else uses them.

## Doors

`MEMRA_LOCKSTEP_CPU_ROWS=1` opts into the exact multi-row arm; default is the one-job-per-row
program. decide-by 2026-10-05: flip or delete on the 9950X-class A/B. The scaled
`memra_cpu_expert_rows_v2` stays exported for ABI compatibility; memra no longer calls it from
the lockstep path.
