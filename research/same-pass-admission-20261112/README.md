# An admitted session that owns no route state yet keeps its charge booked (memra#1255)

Verdict: fixed. A cold DFlash session allocates nothing in `admit`; its target caches and
drafter plane are built in the tick that follows. On the base tree a second arrival read in the
same admission pass was priced against a reading that still held the first session's whole
charge, and both were admitted on the same bytes. The change books the admitted charge of every
session that owns no route state yet; an arrival that the booking alone keeps out waits one
pass (a settle wait, holding the line like a VRAM defer) instead of being decided on a reading
the tick is about to change. Decision record:
[`docs/decisions/ADMISSION-UNBUILT-SESSION-BOOKING.md`](../../docs/decisions/ADMISSION-UNBUILT-SESSION-BOOKING.md).

Cards: one RTX 5090 Laptop GPU (24 GB) and one DGX Spark (GB10). On the RTX 5090 the enforced
power limit read 150 W at idle and 160 W for most of each timed run, with dips to 140 W; each
run's `gate.log` or `guard.log` and its telemetry record it, and it was not changed. Model:
Qwen3.8-27B NVFP4 + Q5_K with the DFlash2 drafter. Every gate run is a single run.

## Binaries

| label | tree | target | binary sha256 (first 16) | version string |
|---|---|---|---|---|
| base | `445f964ea7` (memra#1248's head) | sm_120a | `2b16d303ec4f96b8` | `memra-0.141.1-740591922061` |
| fix6 | the PR head's code (`worker.rs` sha256 `affcc345dc682d54`), built on `11b368f7d7`; not run on a device (round 3 changed a log line's count and test pins only) | sm_120a | `e9ddcb16f3be773d` | `memra-0.141.1-67cb0215cc40` |
| fix5, "the head" | the code after review round 2 (`worker.rs` sha256 `0a44e1773fcbefc8`), built on `11b368f7d7`; round 3 changed only the partial-reclaim line's session count and test pins, so its decisions are the PR head's | sm_120a | `94dd52f97f6017be` | `memra-0.141.1-61d6d6b1bccd` |
| fix5a | the round-2 build before its last refactor (`worker.rs` `590f55f649c64feb`; the same decisions) | sm_120a | `df6ab6cc7f69818e` | `memra-0.141.1-88d952d164ab` |
| fix3, "round 1" | the code after the first review round, on `445f964ea7` (`worker.rs` `3c6f6d9043c4cea5`) | sm_120a | `58f08d69a1b02938` | `memra-0.141.1-fcef2a79bffb` |
| fix, "the first cut" | this PR's code before the review on `445f964ea7` (`worker.rs` sha256 `6ba2aee6f41d0e59`) | sm_120a | `80d3e8d7e42219d6` | `memra-0.141.1-e906635b927a` |
| base | same tree | sm_121a | `32980dc5fd052485` | `memra-0.141.1-740591922061` |
| fix, "the first cut" | same tree | sm_121a | `ba6cf06e6979acc9` | `memra-0.141.1-e906635b927a` |
| diag | `cdfdfd6006` plus `[diag1255]` logging and an allocation ring (`harness/diag-1255.patch`), never pushed | sm_120a | `47edb1b747021189` | `memra-0.141.1-ac3e5a858ee5` |

Between the first cut, round 1 and the head the change grew in four ways, all from the reviews:
the settle wait holds the line; it waits only for an arrival the booking alone keeps out; its
time ends at the settle check; and the pending-prime term counts an unbuilt session's prime
stack on the armed door while the door-off owed terms leave unbuilt sessions out (with the
reading without the booking built from sessions that own route state only). None of them
changes a decision in the gate, the agentic replay or the cells: nothing queued behind a settle
wait; every settle wait was for an arrival that fit the reading without the booking; and the
pending-prime term is never read there (the memory door is off on both cards, and none of those
runs took a partial reclaim: the agentic replay sets an operator prefix budget,
`MEMRA_PREFIX_CACHE_MB=4000`, which keeps the global flush). Only the settle time's definition
shows in them. The armed door's stack booking is proven on the CPU only (no armed-door card
here). The door-off owed terms run on the device in the reclaim cell (`RECLAIM.md`). The landings
after `445f964ea7` change no file under `crates/memra-server/` and, under `crates/memra-engine/`,
only the MTP route's pre-capture guard (`spec.rs`, `unified_memory.rs`), which no run here reaches.

## What is here

| path | what |
|---|---|
| `GATES.md` | the `tools/same-pass-admission-gate.py` runs on the RTX 5090, and the superseded runs of the tool |
| `RECLAIM.md`, `reclaim/<run>/` | the partial-reclaim cell (`harness/reclaim-cell.py`): `summary.json`, `cell.log` (binary sha, card, power limit), `ballast.log`, the server log |
| `gates/<run>/`, `gates/superseded/<run>/` | `summary.json`, the wrapper's `gate.log` (binary sha, card, power limit), `ballast.log`, the server log |
| `AGENTIC.md` | the agentic replay at c = 2, one row per boot, both cards |
| `agentic-5090/<boot>/`, `agentic-gb10/<boot>/` | `cells.log`, per-request rows (no prompt or generated text, a text sha), `/metrics`, the server log, telemetry, `guard.log` |
| `CELLS.md` | the normal-concurrency cells, one row per boot and c, both cards |
| `cells-5090/<boot>/`, `cells-gb10/<boot>/` | `normal.jsonl` (per request: tokens, TTFT, inter-token gaps, text sha), the server log, telemetry, `guard.log` |
| `diag/` | the diagnostic build's agentic boot (`agentic-c2-145032`) and the gate cell's first cut (`gate-g1`) |
| `cpu/` | the first cut's CPU red and green arms and full checks |
| `cpu-head/` | round 1's red and green arms (each arm's patch and log), its full checks, the gate tool's test; `head-11b368f7d7/` the full checks on the rebased tree |
| `cpu-round2/`, `cpu-round3/` | the red and green arms and the full checks of review rounds 2 and 3, on `11b368f7d7` (round 3 is the PR head's tree) |
| `harness/` | the wrappers that ran every cell, the patch scripts, and the table scripts |

Logs and telemetry over 20 kB are gzipped.

## Gate

`GATES.md` has every line. Two cold prompts (5,772 and 5,741 prompt tokens, charges 816 and
814 MB) fired together beside a decoding keeper. The gate applied a reserve of 1,653 MB (the
boot floor of 1,576 MB plus the verify-graph pool's debt, read from a probe that defers): one
fits at 2,469 MB of effective free, two at 3,283 MB; the ballast placed it at 2,874 MB.

| run | binary | exit | verdicts | what happened |
|---|---|---|---|---|
| `base-g6` | base | 1 | V2 V3 FAIL | Both requests admitted in one pass: the two `[admit-trim]` lines read the same `driver_free_after` 2,137,063,424 and `still_owned` 20,536,576,036 bytes. The first prime chunk ran out of memory with 22 MB of driver free; one session parked and replayed, and the keeper (4,896 tokens streamed) was not parked and answered 503. The pair answered 200 with the reference text after the replay |
| `fix5-g6` | the head | 0 | V1 to V4 PASS | The second request took a settle wait (`effective free 2059MB after 815MB booked ... < cost 814MB + reserve 1653MB`) for 3,215.8 ms, the tick that built the first session, then a VRAM defer until the first finished. No out-of-memory line, both 200 with the reference text, the keeper served 12,000 tokens |
| `base-g5`, `fix3-g5` | base, round 1 | 1, 0 | as above | The same readings and verdicts; the settle wait 3,540.3 ms |
| `base-g4`, `fix-g4` | base, the first cut | 1, 0 | as above | The same verdicts before the gate probed the reserve (band from the floor, 2,392 to 3,206 MB; placement 2,799 MB, inside the applied band too) |

## Agentic replay at c = 2

`AGENTIC.md`. RTX 5090, B F, F B on the first cut and B F, F B on round 1: base ran out of
memory in three of four boots (four step OOMs), each in the gate's shape (two `[admit-trim]`
lines of one pass with the same `driver_free_after` 2,136,342,528 and `still_owned`
21,736,474,480), and each of those boots wrote 2 of 29 turns with another text. The change ran
out of memory in none of its four boots and wrote the same 29 texts in each; three of them took
one settle wait (3,264, 3,190 and 2,182 ms, each one tick). Throughput level: base 35.51, 35.86,
37.08, 36.45 tok/s; the change 36.11, 37.02, 37.94, 35.37. GB10, four boots B F F B on the first
cut: 28.03, 28.01, 28.03, 27.95 tok/s, no settle wait, defer or out-of-memory line, the same 29
texts.

## Normal-concurrency cells

`CELLS.md`. RTX 5090, base and round 1, boots B F F B in one lease: the same text per prompt at
every c in all four boots, no settle wait and no out-of-memory line; tok/s base | round 1: c1 73.35,
72.11 | 72.19, 71.93; c2 81.46, 78.32 | 78.24, 77.05; c4 121.35, 120.94 | 121.09, 121.61; c8
146.50, 146.27 | 146.70, 146.81. The first base boot is the high one at c1 and c2 and the second
base boot sits with round 1's, the shape of drift across the lease rather than of the change,
which took no settle wait in these cells. GB10, base then the first cut: the same text per
prompt at every c, no settle wait, defer or out-of-memory line; tok/s c1 27.16 | 27.03, c2 32.76 |
33.20, c4 46.67 | 46.59, c8 79.13 | 81.00.

## Partial reclaim with an unbuilt session

`RECLAIM.md`. A derived prefix budget above its floor, a cold 2,004-token DFlash arrival A (no
route state yet) and a 7,343-token arrival B in the same pass that fits neither reading, so B goes
to the partial reclaim at once. Base sized it 681 MB (416 MB owed, A counted only by its call and
returned rows) and evicted 3 entries; the head sized it 1,001 MB (A's 362 MB charge booked once
through the unbuilt booking, 375 MB owed, A left out) and evicted 4. Both admitted B in A's pass
on the DFlash route with no out-of-memory line; the head's numbers came out the same in its one run
on the head binary and in two runs on the build before the last refactor.

## CPU

`cpu-round3/`, the PR head's tree on `11b368f7d7` (`harness/dd-fix6.sh`), each arm one line reverted
(`cpu-round2/` holds the same arms' first run, before round 3's pins):

| arm | patch | result |
|---|---|---|
| `red-skip` | the owed terms count unbuilt sessions again (`prime_owed_counts` always true) | `an_unbuilt_session_is_booked_once_and_left_out_of_the_reading_without_the_booking` FAILED |
| `red-share` | the reading without the booking keeps the unbuilt sessions' pending-prime share | `an_unbuilt_session_is_booked_once_and_left_out_of_the_reading_without_the_booking` FAILED |
| `red-guard` | the settle hold without its Interactive guard | `every_route_state_field_counts_as_built_and_the_settle_wait_precedes_every_verdict` FAILED |
| `red-armed` | the armed door's booked pending-prime term skips unbuilt sessions (`false` to `true`) | `every_route_state_field_counts_as_built_and_the_settle_wait_precedes_every_verdict` FAILED |
| `red-built` | the built-only term counts them (`true` to `false`) | `every_route_state_field_counts_as_built_and_the_settle_wait_precedes_every_verdict` FAILED |
| `red-filter` | v2's filter treats every session as built | `every_route_state_field_counts_as_built_and_the_settle_wait_precedes_every_verdict` FAILED |
| `red-walker` | v2 passes `true` to `prime_walker_unstarted` for the owns-route-state argument | `every_route_state_field_counts_as_built_and_the_settle_wait_precedes_every_verdict` FAILED |
| `green` | none | all five admission tests pass |

`cpu-head/`, round 1: `red-stack` (the predicate without the unbuilt sessions), `red-unbooked`
(`settle_wait` without the unbooked fit) and `red-hold` (no hold) each fail their test. `cpu/`,
the first cut: `red-booking` and `red-settle` each fail the first test. The full checks passed on
the head's tree: `cargo fmt --check`, clippy with `-D warnings`, `memra-server` tests (1,219
passed, 14 ignored), the flags census and the numeric-scope check.
`python3 tools/test_same_pass_admission_gate.py`: 11 tests OK.

## Diagnostic build

`diag/agentic-c2-145032`: the agentic replay at c = 2 on the diagnostic build of main
(29/29, no out-of-memory line in that boot). Every cold DFlash admission left the session with
no cache and no drafter session after `admit` (`[diag1255]` lines), and its `dspark_prime_start`
took 518 to 1,017 MiB in the tick. `diag/gate-g1`: the gate cell's first cut on that build. It
placed on a stale reading (an idle worker does not publish `/metrics`) and the pair's first
request, alone in its pass beside parked sessions, ran out of memory in its first prime chunk:
the separate gap filed as memra#1258.
