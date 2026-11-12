# An admitted session that owns no route state yet keeps its charge booked (memra#1255)

Verdict: fixed. A cold DFlash session allocates nothing in `admit`; its target caches and
drafter plane are built in the tick that follows. On the base tree a second arrival read in the
same admission pass was priced against a reading that still held the first session's whole
charge, and both were admitted on the same bytes. The change books the admitted charge of every
session that owns no route state yet, and an arrival that does not fit the booked reading waits
one pass (a settle wait) instead of being decided on a reading the tick is about to change.
Decision record:
[`docs/decisions/ADMISSION-UNBUILT-SESSION-BOOKING.md`](../../docs/decisions/ADMISSION-UNBUILT-SESSION-BOOKING.md).

Cards: one RTX 5090 Laptop GPU (24 GB) and one DGX Spark (GB10). On the RTX 5090 the enforced
power limit read 150 W at idle and 160 W for most of each timed run, with dips to 140 W; each
run's `gate.log` or `guard.log` and its telemetry record it, and it was not changed. Model:
Qwen3.8-27B NVFP4 + Q5_K with the DFlash2 drafter. Every gate run is a single run.

## Binaries

| label | tree | target | binary sha256 (first 16) | version string |
|---|---|---|---|---|
| base | `445f964ea7` (memra#1248's head; its landing `9d7c086c67` adds records only) | sm_120a | `2b16d303ec4f96b8` | `memra-0.141.1-740591922061` |
| change | this PR's code on `445f964ea7` (`worker.rs` sha256 `6ba2aee6f41d0e59`) | sm_120a | `80d3e8d7e42219d6` | `memra-0.141.1-e906635b927a` |
| base | same tree | sm_121a | `32980dc5fd052485` | `memra-0.141.1-740591922061` |
| change | same tree | sm_121a | `ba6cf06e6979acc9` | `memra-0.141.1-e906635b927a` |
| diag | `cdfdfd6006` plus `[diag1255]` logging and an allocation ring (`harness/diag-1255.patch`), never pushed | sm_120a | `47edb1b747021189` | `memra-0.141.1-ac3e5a858ee5` |

The PR head carries the same patches rebased onto `9d7c086c67`, the landing of memra#1248; that
landing changes no file under `crates/` or `tools/` after `445f964ea7`, so the binaries stand.

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
| `cpu/` | the CPU red and green arms of the two worker tests (each arm's patch and log), the full CPU checks, the gate tool's own test |
| `harness/` | the wrappers that ran every cell, the patch scripts, and the table scripts |

Logs and telemetry over 20 kB are gzipped.

## Gate

`GATES.md` has every line. Two cold prompts (5,772 and 5,741 prompt tokens, charges 816 and
814 MB, transient floor 1,576 MB) fired together beside a decoding keeper, with a ballast
placing effective free at 2,799 MB: one fits (2,392 MB), two do not (3,206 MB).

| run | binary | exit | verdicts | what happened |
|---|---|---|---|---|
| `base-g4` | base | 1 | V2 V3 FAIL | Both requests admitted in one pass: the two `[admit-trim]` lines read the same `driver_free_after` 2,137,063,424 and `still_owned` 20,410,368,036 bytes. The first prime chunk ran out of memory with 22 MB of driver free; one session parked and replayed, and the keeper (3,872 tokens streamed) was not parked and answered 503. The pair answered 200 with the reference text after the replay |
| `fix-g4` | change | 0 | V1 to V4 PASS | The second request took a settle wait (`effective free 1985MB after 814MB booked ... < cost 815MB + reserve 1653MB`), then a VRAM defer until the first finished; admitted 7,637.6 ms after its first wait. No out-of-memory line, both 200 with the reference text, the keeper served 8,000 tokens |

## Agentic replay at c = 2

`AGENTIC.md`. RTX 5090, boots so far in the order B F F B: base 35.51 and 35.86 tok/s, the
change 36.11 and 37.02; 29/29 turns each. The second base boot ran out of memory once, in the
gate's shape (two `[admit-trim]` lines of one pass with the same `driver_free_after`
2,136,342,528 and `still_owned` 21,736,474,480), replayed the parked session and wrote 9,765
tokens and 2 of 29 turns with another text; the other three boots wrote the same 29 texts. The
change took one settle wait in two boots (3,264 ms from the wait to the admission). GB10, four
boots B F F B: 28.03, 28.01, 28.03, 27.95 tok/s, no settle wait, defer or out-of-memory line, the
same 29 texts.

## Normal-concurrency cells

`CELLS.md`. GB10, base then the change: the same text per prompt at every c, no settle wait,
defer or out-of-memory line; tok/s c1 27.16 | 27.03, c2 32.76 | 33.20, c4 46.67 | 46.59, c8
79.13 | 81.00. The RTX 5090 cells (B F F B) are still to run.

## CPU

`cpu/`. The two worker tests on the change's tree, and each with one line of the change
reverted (dev desk, `harness/dd-red.sh`):

| arm | patch | result |
|---|---|---|
| `red-booking` | the booking books nothing (`unbuilt_charge_bytes` filters every session out) | `an_admitted_session_with_no_route_state_books_its_charge_until_the_tick_builds_it` FAILED |
| `red-settle` | `settle_wait` always false | the same test FAILED |
| `green` | none | both tests pass |

The full CPU checks on the change's tree passed (`check-*.log`): `cargo fmt --check`, clippy
with `-D warnings`, `memra-server` tests (1,217 passed, 14 ignored), the flags census and the
numeric-scope check. `python3 tools/test_same_pass_admission_gate.py`: 10 tests OK.

## Diagnostic build

`diag/agentic-c2-145032`: the agentic replay at c = 2 on the diagnostic build of main
(29/29, no out-of-memory line in that boot). Every cold DFlash admission left the session with
no cache and no drafter session after `admit` (`[diag1255]` lines), and its `dspark_prime_start`
took 518 to 1,017 MiB in the tick. `diag/gate-g1`: the gate cell's first cut on that build. It
placed on a stale reading (an idle worker does not publish `/metrics`) and the pair's first
request, alone in its pass beside parked sessions, ran out of memory in its first prime chunk:
the separate gap filed as memra#1258.
