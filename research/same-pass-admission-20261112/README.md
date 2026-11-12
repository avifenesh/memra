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
| fix3, "the head" | this PR's final code on `445f964ea7` (`worker.rs` sha256 `3c6f6d9043c4cea5`) | sm_120a | `58f08d69a1b02938` | `memra-0.141.1-fcef2a79bffb` |
| fix, "the first cut" | this PR's code before the review on `445f964ea7` (`worker.rs` sha256 `6ba2aee6f41d0e59`) | sm_120a | `80d3e8d7e42219d6` | `memra-0.141.1-e906635b927a` |
| base | same tree | sm_121a | `32980dc5fd052485` | `memra-0.141.1-740591922061` |
| fix, "the first cut" | same tree | sm_121a | `ba6cf06e6979acc9` | `memra-0.141.1-e906635b927a` |
| diag | `cdfdfd6006` plus `[diag1255]` logging and an allocation ring (`harness/diag-1255.patch`), never pushed | sm_120a | `47edb1b747021189` | `memra-0.141.1-ac3e5a858ee5` |

The first cut differs from the head in four places, all from the review: it had no
head-of-line hold on a settle wait; it took a settle wait also for an arrival that did not fit
even without the booking; it timed the settle wait to admission; and the pending-prime term did
not count an unbuilt session's prime stack. The first two are not reached in the gate or at
agentic c = 2 (nothing queues behind a settle wait there, and every settle wait in these runs
was for an arrival that fit the reading without the booking). The fourth is reached on this card
only through the door-off reclaim booking after a partial reclaim, which the agentic replay does
run, so the head's boots carry it and the first cut's do not. Each table names the binary per
row. The PR head carries the head's code rebased onto `11b368f7d7`;
the landings after `445f964ea7` change no file under `crates/memra-server/` and, under
`crates/memra-engine/`, only the MTP route's pre-capture guard (`spec.rs`, `unified_memory.rs`),
which no run here reaches. The CPU checks ran again on that tree (`cpu-head/head-11b368f7d7/`).

## What is here

| path | what |
|---|---|
| `GATES.md` | the `tools/same-pass-admission-gate.py` runs on the RTX 5090, and the superseded runs of the tool |
| `gates/<run>/`, `gates/superseded/<run>/` | `summary.json`, the wrapper's `gate.log` (binary sha, card, power limit), `ballast.log`, the server log |
| `AGENTIC.md` | the agentic replay at c = 2, one row per boot, both cards |
| `agentic-5090/<boot>/`, `agentic-gb10/<boot>/` | `cells.log`, per-request rows (no prompt or generated text, a text sha), `/metrics`, the server log, telemetry, `guard.log` |
| `CELLS.md` | the normal-concurrency cells, one row per boot and c, both cards |
| `cells-5090/<boot>/`, `cells-gb10/<boot>/` | `normal.jsonl` (per request: tokens, TTFT, inter-token gaps, text sha), the server log, telemetry, `guard.log` |
| `diag/` | the diagnostic build's agentic boot (`agentic-c2-145032`) and the gate cell's first cut (`gate-g1`) |
| `cpu/` | the first cut's CPU red and green arms and full checks |
| `cpu-head/` | the head's red and green arms (each arm's patch and log), its full checks, the gate tool's test; `head-11b368f7d7/` the full checks on the rebased tree |
| `harness/` | the wrappers that ran every cell, the patch scripts, and the table scripts |

Logs and telemetry over 20 kB are gzipped.

## Gate

`GATES.md` has every line. Two cold prompts (5,772 and 5,741 prompt tokens, charges 816 and
814 MB) fired together beside a decoding keeper. The gate applied a reserve of 1,653 MB (the
boot floor of 1,576 MB plus the verify-graph pool's debt, read from a probe that defers): one
fits at 2,469 MB of effective free, two at 3,283 MB; the ballast placed it at 2,874 MB.

| run | binary | exit | verdicts | what happened |
|---|---|---|---|---|
| `base-g5` | base | 1 | V2 V3 FAIL | Both requests admitted in one pass: the two `[admit-trim]` lines read the same `driver_free_after` 2,137,063,424 and `still_owned` 20,536,576,036 bytes. The first prime chunk ran out of memory with 22 MB of driver free; one session parked and replayed, and the keeper (4,896 tokens streamed) was not parked and answered 503. The pair answered 200 with the reference text after the replay |
| `fix3-g5` | the head | 0 | V1 to V4 PASS | The second request took a settle wait (`effective free 2059MB after 815MB booked ... < cost 814MB + reserve 1653MB`) for 3,540.3 ms, the tick that built the first session, then a VRAM defer until the first finished. No out-of-memory line, both 200 with the reference text, the keeper served 12,000 tokens |
| `base-g4`, `fix-g4` | base, the first cut | 1, 0 | as above | The same verdicts before the gate probed the reserve (band from the floor, 2,392 to 3,206 MB; placement 2,799 MB, inside the applied band too) |

## Agentic replay at c = 2

`AGENTIC.md`. RTX 5090, B F, F B on the first cut and B F, F B on the head: base ran out of
memory in three of four boots (four step OOMs), each in the gate's shape (two `[admit-trim]`
lines of one pass with the same `driver_free_after` 2,136,342,528 and `still_owned`
21,736,474,480), and each of those boots wrote 2 of 29 turns with another text. The change ran
out of memory in none of its four boots and wrote the same 29 texts in each; three of them took
one settle wait (3,264, 3,190 and 2,182 ms, each one tick). Throughput level: base 35.51, 35.86,
37.08, 36.45 tok/s; the change 36.11, 37.02, 37.94, 35.37. GB10, four boots B F F B on the first
cut: 28.03, 28.01, 28.03, 27.95 tok/s, no settle wait, defer or out-of-memory line, the same 29
texts.

## Normal-concurrency cells

`CELLS.md`. RTX 5090, base and the head, boots B F F B in one lease: the same text per prompt at
every c in all four boots, no settle wait and no out-of-memory line; tok/s base | head: c1 73.35,
72.11 | 72.19, 71.93; c2 81.46, 78.32 | 78.24, 77.05; c4 121.35, 120.94 | 121.09, 121.61; c8
146.50, 146.27 | 146.70, 146.81. The first base boot is the high one at c1 and c2 and the second
base boot sits with the head's, the shape of drift across the lease rather than of the change,
which took no settle wait in these cells. GB10, base then the first cut: the same text per
prompt at every c, no settle wait, defer or out-of-memory line; tok/s c1 27.16 | 27.03, c2 32.76 |
33.20, c4 46.67 | 46.59, c8 79.13 | 81.00.

## CPU

`cpu-head/`, the head's tree on the dev desk (`harness/dd-fix3.sh`), each arm one line of the
change reverted:

| arm | patch | result |
|---|---|---|
| `red-stack` | `prime_walker_unstarted` without the unbuilt sessions | `the_armed_door_books_the_whole_prompt_stack_of_an_unbuilt_session` FAILED |
| `red-unbooked` | `settle_wait` without the unbooked fit | `an_admitted_session_with_no_route_state_books_its_charge_until_the_tick_builds_it` FAILED |
| `red-hold` | the settle wait without its head-of-line hold | `every_route_state_field_counts_as_built_and_the_settle_wait_precedes_every_verdict` FAILED |
| `green` | none | all three pass |

`cpu/`, the first cut: `red-booking` (the booking books nothing) and `red-settle` (`settle_wait`
always false) each fail the first test. The full checks passed on the head's tree and again on
it rebased onto `11b368f7d7`: `cargo fmt --check`, clippy with `-D warnings`, `memra-server`
tests (1,218 passed, 14 ignored), the flags census and the numeric-scope check.
`python3 tools/test_same_pass_admission_gate.py`: 11 tests OK.

## Diagnostic build

`diag/agentic-c2-145032`: the agentic replay at c = 2 on the diagnostic build of main
(29/29, no out-of-memory line in that boot). Every cold DFlash admission left the session with
no cache and no drafter session after `admit` (`[diag1255]` lines), and its `dspark_prime_start`
took 518 to 1,017 MiB in the tick. `diag/gate-g1`: the gate cell's first cut on that build. It
placed on a stale reading (an idle worker does not publish `/metrics`) and the pair's first
request, alone in its pass beside parked sessions, ran out of memory in its first prime chunk:
the separate gap filed as memra#1258.
