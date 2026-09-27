# WP-B day 46: O6, the enforcing predictive door on the fuller charge

OWED.md O6. DAY24 made the predictive book read the physical gate's own terms (the "fuller charge": `A + W + D` beside
the re-keyed context, `booked_real - kv_hat` equal to the context bracket to the byte) and ran it in shadow mode only:
"the enforcing door was not exercised (it would now refuse on the fuller charge, which is the intended change and needs
its own cell against a budget arm)". This day is that cell, with O4's workspace release (DAY45) as a second enforcing
arm, since a lifetime `W` in the book refuses on workspace that is no longer live. `MEMRA_ADMIT_PREDICT_ENFORCE`
stays default OFF; no default moves and no budget value is chosen.

## 1. Pre-registration

Committed and pushed before any day-46 code and before any day-46 cell. Nothing in section 1 changes after a number is
seen; a failed clause is recorded as it reads and a revision is a new, dated addendum pushed before its code. This day
is text only until DAY37 addendum G's repro and the ninth and tenth sittings have read (the lead's order, 2026-09-26).

### 1.1 The arms (existing doors; no new code is planned)

One binary (the lane tip with DAY45's `MEMRA_ADMIT_W_RELEASE`), the budget at its boot-derived default
(`MEMRA_ADMIT_PREDICT_BUDGET_MB` unset; the boot line's `budget_bytes` is recorded):

- `shadow`: `MEMRA_ADMIT_PREDICT_SHADOW=1` (log only, the control: every request runs).
- `enforce`: `MEMRA_ADMIT_PREDICT_ENFORCE=1` (a `reject-kv` verdict becomes the typed 429 with `Retry-After`).
- `enforce-wrel`: `MEMRA_ADMIT_PREDICT_ENFORCE=1 MEMRA_ADMIT_W_RELEASE=1`.

`MEMRA_ADMIT_BY_MEMORY` unset on every arm (the predictive door alone). Orders O1 (`shadow`, `enforce`,
`enforce-wrel`) and O2 (the reverse): 6 boots per card.

### 1.2 The cell (`day46-client.py`, one boot = one arm)

DAY24 1.1's sequence (one 64-token warm request; (a) four concurrent requests of about 6,000 prompt tokens,
`max_tokens=96`; (b) one of about 12,000; (c) the four of (a) again), then 5 s idle, then a burst of B requests
released together (distinct prompts of L tokens, `max_tokens=64`), then 5 s idle and one 512-token probe. 5090: the 9B
at `MEMRA_CTX=65536`, B = 32, L = 6,144. Target card: the 27B, B = 64, L = 30,720; the sequence's prompt lengths are
DAY24's on both cards.

### 1.3 Clauses

- **P1 typed refusals.** On both enforcing arms every refused request is a 429 with `Retry-After` in 1 to 60 and a
  `[admit-predict] ... verdict=reject-kv ... enforce=1` line with the same id; no other non-200 on any arm.
- **P2 no OOM.** No `CUDA_ERROR_OUT_OF_MEMORY` line, no parked prefill or step OOM, no 503, no crash line on any boot.
- **P3 within the budget.** On both enforcing arms, at every admitted request's `[admit-predict]` line,
  `booked_bytes + kv_hat <= budget_bytes`.
- **P4 identity.** Every request admitted on an enforcing arm has the completion digest of the same request on
  `shadow` in the same order (admission chooses who runs, not what they output).
- **P5 the release reaches the door.** On `enforce-wrel`, every admitted session prints one `w-release` line before its
  retire (DAY45 W2's rule), and the probe's line reads `booked_bytes=0 booked_real=0` on every arm (the books are
  exact at idle).

Readings, no bound: the burst's 200 and 429 counts per arm; the sequence's admitted count per arm (the day-24 rows);
the time to each 429; the admitted requests' TTFT p50 and p95; the peak `booked_bytes`; the `budget_bytes` the boot
derived. Each median states N and the 250 ms regime.

### 1.4 What the reading decides

Nothing moves a default. The reading is the owner's input for the enforcing door: how many of the burst the fuller
charge admits, and how many more the workspace release admits, with no OOM. If P2 fails on an enforcing arm, the
cause is quoted and the door's charge is revised under a new pre-registration.

### 1.5 Price

Code: about 0.2 agent-day (the client and reader; no engine or server change). Cells: about 1 h on the 5090, about
2 h on the target card.

### 1.6 Addendum A (2026-09-26, from DAY45 2.1 and 2.2, before any day-46 code or cell)

Two facts from DAY45 on the target card change this cell before it is built. No bound changes.

- **The shadow arm OOMs at this burst by construction.** DAY45 2.1: with no admission door, the physical gate admits
  all 64 sessions of 30,720 tokens and the batched prime runs out of memory (55 OOM lines, 53 of 64 end `503`), the same
  on both arms. `shadow` here has no door either, so P2 as written would fail on the control for a reason already
  placed. P2 therefore applies to the two enforcing arms, and on `shadow` the OOM, 503 and crash counts are a reading
  (the before). P1's "no other non-200 on any arm" also becomes the enforcing arms only. P4 compares the requests that
  are `200` on both boots of the pair, and prints the status mismatches as a reading (DAY45 addendum B's W3 rule).
- **A burst released together is booked before any prime completes.** DAY45 2.2: a 30,720-token burst session's
  `w-release` lands 217 to 459 s after its booking, so every admission of that burst reads the same books with or
  without the release, and `enforce` and `enforce-wrel` would admit the same burst. So the cell gains a second wave:
  when the burst's first request completes (its prime has completed, so its `W` has been released on
  `enforce-wrel`), B/2 more requests (distinct prompts, the same L and `max_tokens`) are released together. Then 5 s
  idle and the probe, as before. Readings add, per wave and arm, the 200 and 429 counts and the `booked_bytes` at the
  wave's first admission. The second wave's admitted count on `enforce-wrel` against `enforce` is the release's value.
- **P5 uses DAY45 addendum B's W2 rule:** on `enforce-wrel` every booked id prints exactly one of `w-release` or
  `w-retire-unreleased` (bytes equal, none twice), and the probe reads `booked_bytes=0 booked_real=0` on every arm.
- Everything else of section 1 stands. The day stays text only until the ninth sitting (DAY44) has read.

### 1.7 Addendum B (2026-09-27, the cells as built, after the ninth sitting read and before any cell)

The ninth sitting (DAY44 2.1) has read, so the day leaves text only. No clause, bound or reading changes.

- **The binary:** the lane's crates at `8926ccfb3` (DAY45's W release, and every door since, all default off), built
  by `build-arms.sh` (`target/day46` locally, `bins/tip` on the box).
- **The client (`day46-client.py`):** `/v1/completions` with `prompt_ids`, greedy, streamed with usage. The TTFT
  reading is the first streamed token after submit. The DAY24 sequence uses exact token windows: 64 warm tokens; four
  of 6,000 with `max_tokens=96`; one of 12,000; the same four again. `(a)i` and `(c)i` share their cache salt
  `seq<i>` so the prefixes are retained, as DAY24's chat requests were. Every other request has its own salt, which is
  its tenant on the `[admit-predict]` line. The second wave (addendum A) is released when the burst's first request
  completes; its release time is in `wave2.txt`. A 429's `Retry-After` header is recorded per row.
- **The runner (`day46-run.sh`):** `run-day26-cell.sh` arms `MEMRA_ADMIT_PREDICT_SHADOW=1` on every boot; `enforce`
  adds `MEMRA_ADMIT_PREDICT_ENFORCE=1`; `enforce-wrel` adds `MEMRA_ADMIT_W_RELEASE=1` as well. `MEMRA_ADMIT_BY_MEMORY`,
  `MEMRA_ADMIT_OPEN_OUTPUT_TOKENS` and `MEMRA_ADMIT_PREDICT_BUDGET_MB` are unset on every arm.
- **The reader (`day46-read.py`):** P1 maps each 429 to a `verdict=reject-kv ... enforce=1` line of its salt. P3 reads
  `booked_bytes + kv_hat <= budget_bytes` on every `verdict=admit` line of an enforcing boot. P5-PROBE reads the probe's
  line on every arm, and P5 reads the W receipts on `enforce-wrel` with DAY45 addendum B's rule. The readings are per
  wave: the 200 and 429 counts, `booked_bytes` at the wave's first line, TTFT p50 and p95 with N, and time to each 429.
- **Cells:** the 5090 (`rtx5090-day46/run.sh`, B = 32, L = 6,144, the 9B at 65,536) and the target card (the fifteenth
  sitting, `pro-single-b-sitting15.sh`, B = 64, L = 30,720, the 27B at the checkpoint's context), six boots each.

### 1.8 Addendum C (2026-09-27, after 2.1, before the client fix)

2.1 places a client defect: `day46-client.py` releases the second wave when the burst's first request ENDS, and a
typed 429 ends about 1 s after the burst, so the second wave arrived before any admitted prime completed. The value
reading of addendum A (the second wave on `enforce-wrel` against `enforce`) was therefore not measured. The fix: the
second wave is released when the burst's first request completes `200` (its prime completed, so on `enforce-wrel` its
`W` has been released). `wave2.txt` records the release time and the tag of the completion that triggered it. The
whole cell runs again under a new receipt name (`b-day46c` on the target card, `rtx5090-day46c` on the 5090), six
boots each. No clause, bound or reading changes; 2.1's P1 to P5 lines stand as they read.

## 2. Results

Written after the runs. Section 1 is unchanged.

### 2.1 The target card (the fifteenth sitting, one RTX PRO 6000 Blackwell Workstation Edition at 600 W, 2026-09-27 01:57 to 02:41Z)

Chain tree `1df7e7852`; the binary built on the box from `8926ccfb3`, sha256 `263bdc37...b494211443` (the lead mirrored
it by hash). The 27B at the checkpoint's context. The boot-derived budget is `budget_bytes=65877946064` on every boot.
Receipts at `pro-single-day46/box/` (160 files, box manifest OK). Note (2026-09-27, from the lead): the `MANIFEST.sha256` in this mirror is the lead's mirror-time manifest, not the sitting's own. The lead's earlier mirror script wrote its manifest over the sitting's, and the originals are gone (the box copies were destroyed or overwritten). Every mirrored file is verified against the lead's manifest, which hashed the box files at mirror time, so no receipt content is affected. Every boot `rc=0` in `run.log`. The clause lines,
verbatim, with the enforcing boots' lines alike in both orders (O1 shown; O2 reads the same numbers):

```
DAY46 P2 card=pro6000 boot=O1-enforce oom_lines=0 parked_oom_lines=0 crash_lines=0 r503=0 -> PASS
DAY46 P1 card=pro6000 boot=O1-enforce r429=79 without_reject_line=[] retry_after_out_of_1_60=[] other_non200=[] -> PASS
DAY46 P3 card=pro6000 boot=O1-enforce admit_lines=28 over_budget=[] -> PASS
DAY46 P5-PROBE card=pro6000 boot=O1-enforce probe_booked=0 probe_booked_real=0 -> PASS
DAY46 P2 card=pro6000 boot=O1-enforce-wrel oom_lines=0 parked_oom_lines=0 crash_lines=0 r503=0 -> PASS
DAY46 P1 card=pro6000 boot=O1-enforce-wrel r429=79 without_reject_line=[] retry_after_out_of_1_60=[] other_non200=[] -> PASS
DAY46 P3 card=pro6000 boot=O1-enforce-wrel admit_lines=28 over_budget=[] -> PASS
DAY46 P5 card=pro6000 boot=O1-enforce-wrel w_booked=28 w_release=27 w_retire_unreleased=1 twice=[] bytes_mismatch=[] neither=[] -> PASS
DAY46 P2-READING card=pro6000 boot=O1-shadow oom_lines=55 parked_oom_lines=0 crash_lines=0 r503=0 -> READING (the before)
DAY46 P5-PROBE card=pro6000 boot=O1-shadow probe_booked=0 probe_booked_real=0 -> PASS
DAY46 P4 card=pro6000 order=O1 arm=enforce rows_200_both=18 status_mismatch=79 differ=[] -> PASS
DAY46 P4 card=pro6000 order=O1 arm=enforce-wrel rows_200_both=18 status_mismatch=79 differ=[] -> PASS
DAY46 P4 card=pro6000 order=O2 arm=enforce rows_200_both=18 status_mismatch=79 differ=[] -> PASS
DAY46 P4 card=pro6000 order=O2 arm=enforce-wrel rows_200_both=18 status_mismatch=79 differ=[] -> PASS
```

- **P1 to P5 PASS on all four enforcing boots, both orders.** Every refused request is a typed 429 with Retry-After in
  1 to 60 and its own `reject-kv ... enforce=1` line: 79 per boot, 47 of the burst and all 32 of the second wave. No
  OOM, no parked OOM, no 503, no crash. Every admitted line is within the budget (peak `booked_bytes` 64.46 GB
  against 65.88 GB). The probe reads both books 0. On `enforce-wrel` 27 of 28 booked workspaces release at prime
  completion and the probe's retires at once (`same-tick`). The 18 requests that are `200` on both boots of an order
  have equal digests.
- **The before (`shadow`, no door):** 55 OOM lines on both boots. The burst admits all 64, and 11 end `200`. The
  booked book peaks at 231.1 GB against the 65.9 GB budget, which is the door's subject.
- **Readings (N and the 250 ms regime per the boot's samples):** the DAY24 sequence is admitted in full on every
  boot (9 of 9). Enforcing arms: burst 17 of 64 admitted, TTFT p50 151.8 s (N=17; the 17 primes of 30,720 run as one
  batched prime), time to a 429 p50 1.06 s. Shadow: burst TTFT p50 524 to 529 s (N=11), second wave 32 of 32 `200` at
  TTFT p50 312 to 316 s (N=32).
- **The value reading was not measured.** The second wave went out 1 s after the burst, at its first 429, while the
  17 admitted primes were still running. Their `w-release` lines land about 150 s later, at the second wave's
  1790475568278 against 1790475417526. So both enforcing arms read the same book at the second wave (64.46 GB) and
  refuse all 32. That is the client defect addendum C fixes; the P lines above stand.

### 2.2 Addendum C on the target card (the seventeenth sitting, the same card class and box, 2026-09-27 03:10 to 03:59Z)

Chain tree `43dbdb29f` (the fixed client); the binary built on the box from `8926ccfb3`, sha256 `35719237...4cef2bd27d` (the
lead mirrored it by hash). Receipts at `pro-single-day46/box-c/` (the sitting's own `MANIFEST.sha256`, the lead's
`LEAD-MANIFEST.sha256` re-checked). Each boot's `wave2.txt` names the trigger: the burst's first `200` (`burst-0` to
`burst-3`). Every boot `rc=0`. Verbatim, the clause lines (the enforcing boots in both orders read alike; O1 shown):

```
DAY46 P2 card=pro6000 boot=O1-enforce oom_lines=0 parked_oom_lines=0 crash_lines=0 r503=0 -> PASS
DAY46 P1 card=pro6000 boot=O1-enforce r429=78 without_reject_line=[] retry_after_out_of_1_60=[] other_non200=[] -> PASS
DAY46 P3 card=pro6000 boot=O1-enforce admit_lines=29 over_budget=[] -> PASS
DAY46 P5-PROBE card=pro6000 boot=O1-enforce probe_booked=0 probe_booked_real=0 -> PASS
DAY46 P2 card=pro6000 boot=O1-enforce-wrel oom_lines=0 parked_oom_lines=0 crash_lines=0 r503=0 -> PASS
DAY46 P1 card=pro6000 boot=O1-enforce-wrel r429=68 without_reject_line=[] retry_after_out_of_1_60=[] other_non200=[] -> PASS
DAY46 P3 card=pro6000 boot=O1-enforce-wrel admit_lines=39 over_budget=[] -> PASS
DAY46 P5 card=pro6000 boot=O1-enforce-wrel w_booked=39 w_release=38 w_retire_unreleased=1 twice=[] bytes_mismatch=[] neither=[] -> PASS
DAY46 P2-READING card=pro6000 boot=O1-shadow oom_lines=55 parked_oom_lines=0 crash_lines=0 r503=0 -> READING (the before)
DAY46 P4 card=pro6000 order=O1 arm=enforce rows_200_both=19 status_mismatch=78 differ=[] -> PASS
DAY46 P4 card=pro6000 order=O1 arm=enforce-wrel rows_200_both=29 status_mismatch=68 differ=[] -> PASS
DAY46 P4 card=pro6000 order=O2 arm=enforce rows_200_both=19 status_mismatch=78 differ=[] -> PASS
DAY46 P4 card=pro6000 order=O2 arm=enforce-wrel rows_200_both=29 status_mismatch=68 differ=[] -> PASS
```

and the wave readings, verbatim:

```
DAY46 READING card=pro6000 boot=O1-enforce wave=wave2 n=32 ok200=1 r429=31 booked_at_first_line=60602103104 ttft_ms p50=9329.5 p95=9329.5 N=1 time_to_429_ms p50=57.1 max=83.3
DAY46 READING card=pro6000 boot=O1-enforce-wrel wave=wave2 n=32 ok200=11 r429=21 booked_at_first_line=20471002432 ttft_ms p50=97338.0 p95=98386.0 N=11 time_to_429_ms p50=50.6 max=64.5
DAY46 READING card=pro6000 boot=O2-enforce wave=wave2 n=32 ok200=1 r429=31 booked_at_first_line=60602103168 ttft_ms p50=10103.2 p95=10103.2 N=1 time_to_429_ms p50=48.6 max=81.2
DAY46 READING card=pro6000 boot=O2-enforce-wrel wave=wave2 n=32 ok200=11 r429=21 booked_at_first_line=20471002496 ttft_ms p50=98617.5 p95=99449.3 N=11 time_to_429_ms p50=49.5 max=76.5
```

- **P1 to P5 PASS on all four enforcing boots, both orders.** Every refusal is typed, with its own reject line. There is
  no OOM, parked OOM, 503 or crash. Every admitted line is within the budget (peak 64.46 GB against 65.88 GB), and the
  probe reads both books at 0. On `enforce-wrel`, 38 of 39 booked workspaces release at prime completion (the probe's
  retires the same tick). The requests `200` on both boots of a pair have equal digests (19 and 29 rows).
- **The value reading (addendum A), both orders alike:** by the second wave the W release has taken the book from
  60.60 GB down to 20.47 GB. The enforcing door then admits 11 of the 32 second-wave requests with the release,
  against 1 of 32 without it. Over the whole boot that is 39 admitted against 29, with no OOM on either arm.
- **The same as 2.1:** the burst admits 17 of 64 on both enforcing arms. Its TTFT p50 is 150 to 155 s (N=17), because
  the 17 primes of 30,720 tokens run as one batched prime. The DAY24 sequence is admitted 9 of 9 on every boot.
- **The before (`shadow`, no door):** 55 and 56 OOM lines. The burst ends `200` on 11 and 10 of 64, and the second wave
  32 of 32 after the storm.
- **What it decides:** no default moves. This is the owner's input for the enforcing predictive door, and for the W
  release that is its second arm.
