# A pooled MTP verify fits the device; evictable memory funds the largest member's round (memra#1266)

Verdict: fixed. On the RTX 5090 Laptop the 29-turn agentic replay at c = 2 on the MTP route
failed requests 19 and 21 with a CUDA out-of-memory in the cohort's pooled verify in every main
boot (3 of 3 here, 4 more before: the issue's three and the diagnostic boot). With the fix the
same boots answer 29 of 29 (3 of 3) with the texts main wrote for its 27 and the texts the
memra#1221 part 2 head wrote for all 29, k=19 and k=21 included. Decision record:
[`docs/decisions/POOLED-VERIFY-MEMORY-CAP.md`](../../docs/decisions/POOLED-VERIFY-MEMORY-CAP.md).

## Binaries

| label | tree | card | binary sha256 (first 16) | build id |
|---|---|---|---|---|
| main | `11b368f7d7` | RTX 5090 (sm_120a) | `3f94cfb8f871c2c9` | `memra-0.141.1-32737cebc309` |
| fix | this branch's code on `11b368f7d7` (`6433a5a264`; later commits change tests and docs only) | RTX 5090 (sm_120a) | `8f1b48774396f5ac` | `memra-0.141.1-4e861bdcb961` |
| diagnostic | main + `harness/rig/diag_probe.py` (log lines only) | RTX 5090 (sm_120a) | `8e7abde6637cc294` | `memra-0.141.1-c200db12045d` |
| main | `11b368f7d7` | GB10 (sm_121a) | `2e1211ff9466a203` | `memra-0.141.1-32737cebc309` |
| fix | as above | GB10 (sm_121a) | `b9c35baaee789aca` | `memra-0.141.1-4e861bdcb961` |
| bench | `decode-batch-bench` from the fix tree | RTX 5090 (sm_120a) | `16903e07c5710f5b` | |

The RTX 5090 binaries were built on the CPU build host (`harness/build-host/dd_build.sh`), the
GB10 ones on the unit (`harness/gb10/build_gb10.sh`, arm M is the fix tree with main's three
changed crate files). Card conditions: RTX 5090 Laptop GPU, driver 615.71.09, on AC with
`nv_tgp` 150 and `ppt_pl1_spl` 140; the enforced power limit read 150 or 155 W at each cell's
start (`guard.log`) and was not changed. GB10: one DGX Spark, driver 580.178.04.

## The shape

Qwen3.8-27B NVFP4 + Q5_K with the masked 32,768-row MTP head, the MTP speculative route, ctx
32,768, 8 sessions, prefix cache 4,000 MB, no host tier, `MEMRA_PREFIX_STABLE_BOUNDARY=1`,
`MEMRA_TICK_TRACE=1`; the 2026-11-05 decision cell's 29 agentic turns (1.4k to 17k prompt
tokens) through its driver (`research/c8-workspace-20261108/harness/agent_turns.py`): a prime
pass at max_tokens 1, then the timed pass at c = 2, thinking on, temperature 0, seed 1234, 768
tokens. The workload derives from private agent sessions and is not here; rows carry lengths,
ids and text hashes only.

## The mechanism (`diag-5090/`)

One boot of the diagnostic arm reproduced the failure at the same point (27 of 29) and printed
the memory around it (`diag-5090/tl-01-D.txt`, times relative to the step OOM):

| when | line | driver free | pool cached | allocatable |
|---|---|---:|---:|---:|
| -0.553 s | admission gate: cost 900 MiB + reserve 1,536 MiB = 2,436 MiB, sufficient | 515 MiB | 2,143 MiB | 2,658 MiB |
| -0.548 s | driver-headroom rung released 1,241 MiB; restore 3,488 of 3,905 tokens | 1,699 MiB | 516 MiB | 2,215 MiB |
| -0.107 s | after the 417-row suffix prime; the cohort enters | 1,539 MiB | 376 MiB | 1,915 MiB |
| -0.016 s | first round gate: graph replay suspended (two burst snapshots, +299 MiB used) | 99 MiB | 1,517 MiB | 1,616 MiB |
| 0.000 s | `alloc-fail bytes=15728640` at the ssm checkpoint slab of `pooled_linear_layer`, layer 61 of 64, rows 12 | 67 MiB | 128 MiB | 195 MiB |

The pooled forward (two K=5 rounds, 12 rows) had allocated 1,453 MiB of checkpoint slabs when
the second member's 15 MiB slab failed. The prefix cache held 2,678 MiB, 2,226 MiB of it
unleased, and two parked spec sessions were resident; the step-OOM teardown evicted them after
both requests had failed. Across the boot's two-member pooled rounds the allocatable reading was
median 2,135, 10th percentile 1,723 and minimum 1,294 MiB (578 rounds).

## Agentic replay, RTX 5090, c = 2 (`agentic-5090-c2/`)

Boots interleaved M F F M M F, one per lease.

| boot | ok | failed | tokens | agg gen tok/s | per-user median tok/s | TTFT p50 s | OOM lines | not parked | capped forwards | pre-cohort reclaims | graph suspended |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 01-M | 27 | 19,21 | 8897 | 42.1 | 48.24 | 2.96 | 2 | 2 | n/a | n/a | 2 |
| 02-F | 29 | - | 9396 | 45.11 | 48.93 | 2.813 | 0 | 0 | 23 | 0 | 0 |
| 03-F | 29 | - | 9396 | 44.03 | 48.04 | 2.949 | 0 | 0 | 23 | 0 | 0 |
| 04-M | 27 | 19,21 | 8897 | 42.61 | 47.97 | 2.926 | 2 | 2 | n/a | n/a | 2 |
| 05-M | 27 | 19,21 | 8897 | 42.65 | 48.3 | 2.859 | 2 | 2 | n/a | n/a | 2 |
| 06-F | 29 | - | 9396 | 44.19 | 47.73 | 2.868 | 0 | 0 | 23 | 0 | 0 |

- 01-M: 27 answered texts equal 02-F's; differing k: none
- 02-F: 29 answered texts equal 02-F's; differing k: none
- 03-F: 29 answered texts equal 02-F's; differing k: none
- 04-M: 27 answered texts equal 02-F's; differing k: none
- 05-M: 27 answered texts equal 02-F's; differing k: none
- 06-F: 29 answered texts equal 02-F's; differing k: none

Main failed requests 19 and 21 in every boot with the issue's lines (`out of memory` 2, `step OOM
NOT parked` 2, `graph replay suspended` 2). The fix answered 29 of 29 in every boot with no OOM,
no step-OOM teardown and no suspension; texts are equal across all six boots on every request
both arms answered, and equal to the memra#1221 part 2 head boots (`rtxag3/02-ST`, `03-ST`, that
lane's receipts) on all 29, k = 19 and 21 included (`ref-1221b-head-texts.jsonl`: 58 of 58).

**The cap's forwards.** 23 capped forwards in each fix boot, all at the same point (the k=19
newcomer beside the decoding k=21): "forward 759 MiB + next member 759 MiB over 1616 MiB
allocatable less the 256 MiB launch floor" (8 lines), the same at 1,500 MiB (7 lines), and one
"next member 607 MiB" line; lines 17 to 23 are not printed (the first 16, then every 256th), the
count is `/metrics`. No pre-cohort reclaim fired: the reading always funded the largest member's
round (about 790 MB) plus the floor, as admission's floor promises. The capped rounds ran three
ticks of K=5 + K=5 at 69 to 80 ms a round (`harness/rig/round_cost.py`); main has no such round
to compare (it fails there), and the pooled-versus-serial verify cost is measured below.

**Throughput.** Aggregate generated tok/s: main 42.10, 42.61, 42.65 over its 27 answered
requests (8,897 tokens); fix 45.11, 44.03, 44.19 over 29 (9,396 tokens). Per-user median tok/s:
main 48.24, 47.97, 48.30; fix 48.93, 48.04, 47.73. The two arms do not answer the same set, so
the aggregate is not a like-for-like speed comparison; the per-user medians overlap.

## Agentic replay, RTX 5090, c = 4 (`agentic-5090-c4/`)

Pending: queued on the RTX 5090 behind another lane's landing lease.

## Agentic replay, GB10, c = 2 (`agentic-gb10-c2/`)

Same shape on the GB10 launcher's model files (`harness/gb10/serve.sh`), boots M F F M.

| boot | ok | failed | tokens | agg gen tok/s | per-user median tok/s | TTFT p50 s | OOM lines | not parked | capped forwards | pre-cohort reclaims | graph suspended |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 01-M | 29 | - | 9665 | 33.34 | 23.59 | 2.403 | 0 | 0 | n/a | n/a | 0 |
| 02-F | 29 | - | 9665 | 33.32 | 23.64 | 2.404 | 0 | 0 | 0 | 0 | 0 |
| 03-F | 29 | - | 9665 | 33.33 | 23.62 | 2.409 | 0 | 0 | 0 | 0 | 0 |
| 04-M | 29 | - | 9665 | 33.29 | 23.61 | 2.404 | 0 | 0 | n/a | n/a | 0 |

- 01-M: 29 answered texts equal 02-F's; differing k: none
- 02-F: 29 answered texts equal 02-F's; differing k: none
- 03-F: 29 answered texts equal 02-F's; differing k: none
- 04-M: 29 answered texts equal 02-F's; differing k: none

GB10 never comes near the cap (unified memory, the allocatable reading counts `MemAvailable`):
0 capped forwards, 0 reclaims, 29 of 29 in all four boots, the same text per request in all
four, 33.34, 33.32, 33.33, 33.29 tok/s. (The GB10 texts differ from the RTX 5090's: another
build target.)

## Normal-concurrency cells (`normal-5090/`, `normal-gb10/`)

The cell #1173 and #1255 used (`research/drafter-admission-20261112/harness/normal_cell.py`:
closed-loop clients at c = 1, 2, 4, 8, 6 requests per client, greedy chat, 160 new tokens,
streamed; ctx 8,192, 8 sessions), on the MTP route this change acts on (the masked MTP head, no
DFlash drafter). Boots M F F M, two per lease.

**RTX 5090.**

| boot | c | requests, errors | completion tokens | agg tok/s | TTFT p50 s | ITL p50 / p99 ms |
|---|---|---|---|---|---|---|
| 01-M | 1 | 6, 0 | 960 | 79.77 | 0.163 | 0.0 / 33.0 |
| 01-M | 2 | 12, 0 | 1920 | 94.77 | 0.223 | 0.0 / 87.1 |
| 01-M | 4 | 24, 0 | 3840 | 119.93 | 0.611 | 28.0 / 33.5 |
| 01-M | 8 | 48, 0 | 7680 | 152.18 | 1.294 | 41.5 / 52.5 |
| 02-F | 1 | 6, 0 | 960 | 79.83 | 0.163 | 0.0 / 33.2 |
| 02-F | 2 | 12, 0 | 1920 | 95.06 | 0.223 | 0.0 / 86.2 |
| 02-F | 4 | 24, 0 | 3840 | 119.46 | 0.618 | 28.0 / 33.9 |
| 02-F | 8 | 48, 0 | 7680 | 149.49 | 1.176 | 42.2 / 54.1 |

- c=1: one set of (prompt, text sha) across the boots
- c=2: one set of (prompt, text sha) across the boots
- c=4: one set of (prompt, text sha) across the boots
- c=8: one set of (prompt, text sha) across the boots
- 01-M: out of memory=0; NOT parked=0; pooled MTP verify capped=0; pre-cohort reclaim=0; graph replay suspended=0; spec-pool] pooled MTP cohort engaged=1
- 02-F: out of memory=0; NOT parked=0; pooled MTP verify capped=0; pre-cohort reclaim=0; graph replay suspended=0; spec-pool] pooled MTP cohort engaged=1

- c=1 agg tok/s: F 79.83 (79.83), M 79.77 (79.77)
- c=2 agg tok/s: F 95.06 (95.06), M 94.77 (94.77)
- c=4 agg tok/s: F 119.46 (119.46), M 119.93 (119.93)
- c=8 agg tok/s: F 149.49 (149.49), M 152.18 (152.18)

**GB10.**

| boot | c | requests, errors | completion tokens | agg tok/s | TTFT p50 s | ITL p50 / p99 ms |
|---|---|---|---|---|---|---|
| 01-M | 1 | 6, 0 | 960 | 28.19 | 0.291 | 0.0 / 90.6 |
| 01-M | 2 | 12, 0 | 1920 | 43.27 | 0.38 | 0.0 / 269.5 |
| 01-M | 4 | 24, 0 | 3840 | 61.62 | 0.423 | 0.0 / 520.0 |
| 01-M | 8 | 48, 0 | 7680 | 79.98 | 1.406 | 87.3 / 102.3 |
| 02-F | 1 | 6, 0 | 960 | 28.19 | 0.291 | 0.0 / 91.1 |
| 02-F | 2 | 12, 0 | 1920 | 43.23 | 0.38 | 0.0 / 268.6 |
| 02-F | 4 | 24, 0 | 3840 | 61.73 | 0.423 | 0.0 / 521.1 |
| 02-F | 8 | 48, 0 | 7680 | 79.9 | 1.409 | 87.4 / 102.5 |

- c=1: one set of (prompt, text sha) across the boots
- c=2: one set of (prompt, text sha) across the boots
- c=4: one set of (prompt, text sha) across the boots
- c=8: one set of (prompt, text sha) across the boots
- 01-M: out of memory=0; NOT parked=0; pooled MTP verify capped=0; pre-cohort reclaim=0; spec-pool] pooled MTP cohort engaged=3
- 02-F: out of memory=0; NOT parked=0; pooled MTP verify capped=0; pre-cohort reclaim=0; spec-pool] pooled MTP cohort engaged=3

- c=1 agg tok/s: F 28.19 (28.19), M 28.19 (28.19)
- c=2 agg tok/s: F 43.23 (43.23), M 43.27 (43.27)
- c=4 agg tok/s: F 61.73 (61.73), M 61.62 (61.62)
- c=8 agg tok/s: F 79.90 (79.9), M 79.98 (79.98)

## What a capped round costs (`bench-5090/`)

Pending: queued on the RTX 5090 behind another lane's landing lease.

## Gates (`gates-5090/`)

Pending: queued on the RTX 5090 behind another lane's landing lease.

## CPU (`cpu/`)

Red arms first (`cpu/red/`, `harness/build-host/red_arms.py` and `dd_red.sh`): each arm changes
one thing and the named test fails (cargo exit 101 with the test's own panic).

| arm | what it changes | the test that fails |
|---|---|---|
| `admits-open` | `pooled_verify_admits` lets every member join (no cap) | `the_first_member_always_joins_and_a_later_one_only_within_the_reading`: `assertion failed: !pooled_verify_admits(k5, k5, 1_616 * MIB)` |
| `main-engine` | main's `spec.rs` with the lockstep census appended | `the_mtp_lockstep_chunk_walk_applies_the_memory_cap`: the lockstep lost its allocatable reading |
| `shortfall-off` | `spec_round_shortfall` never reports a shortfall | `the_reclaim_fires_only_when_the_reading_cannot_fund_the_round` |
| `main-worker` | main's `worker.rs` and server `lib.rs` with the Cohorts-pass census appended | `the_cohorts_pass_reclaims_before_it_forms_cohorts`: no `reclaim_for_spec_round` call |

Green (`cpu/green-*.txt`): the three engine tests and the two worker tests pass; the worker suite
(`worker::`, debug) 458 passed. The memra#1227 phase (a) census
(`spec_newcomers_prime_before_cohorts_is_wired`) now pins the reclaim between the prime quantum
and the cohorts' formation.

Full CPU chain on the CPU build host (`cpu/full-chain.txt`, the branch's crates with the whole
`research/` tree the suites read at run time): `cargo fmt --check`, `cargo clippy --release
--all-targets -D warnings`, the coalescer contract, memra-server 1,227 passed / 0 failed,
memra-kv 87 / 0, memra-engine lib 746 / 0, memra-gguf lib 417 / 0, the portable suites 448
passed / 0 skipped / 0 failed.

## What is here

| path | what |
|---|---|
| `diag-5090/01-D/`, `diag-5090/tl-01-D.txt` | the diagnostic boot (server log with the `[p1266]` lines, per-request rows, telemetry, `guard.log`) and its timeline |
| `agentic-5090-c2/<boot>/`, `agentic-5090-c4/<boot>/`, `agentic-gb10-c2/<boot>/` | `cells.jsonl`, per-request rows, `summary.txt` (line counts), `/metrics`, the server log, telemetry, `guard.log`, `run.log` |
| `normal-5090/<boot>/`, `normal-gb10/<boot>/` | `normal.jsonl` (per request: tokens, TTFT, inter-token gaps, text sha), the server log, telemetry, `guard.log`, `summary.txt` |
| `bench-5090/` | `decode-batch-bench --pooled` output |
| `gates-5090/` | the seeded-invariance and exact-resend gate runs (logs, guard, each run's directory as a tarball) |
| `cpu/` | the red arms (`red/`), the new tests' green runs, the full CPU chain |
| `harness/rig/`, `harness/gb10/`, `harness/build-host/` | the scripts that ran all of it, with the step files; paths and hosts are written generically (`harness/rig/sanitize_receipts.py`) |
