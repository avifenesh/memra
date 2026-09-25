# WP-C day 55 (2026-09-24): an always-admitted prime arm on the RTX 5090 class, pre-registered (OWED C8)

`OWED.md` C8. Sources: `DAY35.md` (pass 1's prime arm inadmissible: the memory admission refused 7 of 10 intruders,
`[admit-oom] capacity reject: model="gate" ctx=5186 does not fit an IDLE box (available 1794MB), HTTP 400
context_length_exceeded`, beside a co-tenant of 12.6 to 13.5 GB that did not hold the lock; pass 2, clean, admitted
10 of 10), `DAY37.md` section 7 (the prime control's low first pass, 229.0 at 51 C on the hold's first boot against
269.7 to 294.2 elsewhere, cause not separated), `DOOR-DECISION-PACKET.md` item 7. Written before the cell ran; tree
at start: `c2ba07aef`.

## 0. Why a shorter prime is not the fix

The refusal's own arithmetic (`DAY35.md`, the `VRAM defer` line): the 5186-token request costs 1,683 MB against a
1,611 MB reserve, 3.3 GB in all. Beside the 9B server alone the card had about 14 GB free (the clean pass-2 prime
boot peaked at 9,753 MiB of 24,463). The refusals came only with the co-tenant, at 1,794 MB available: there the
reserve alone takes 1.6 GB of it, so no prime long enough to stretch a tick (a 1,024-token chunk costs about a fifth
of the 5,120-token prime, about 340 MB, plus the same reserve) is admitted either, while a shorter prime changes the
control (fewer stretched ticks). The day-35 shape (5,120 to 5,123 tokens at `MEMRA_CTX=8192`, `PREFILL_TICK_T =
1024` chunks) is admitted with a 10 GB margin on a card this cell alone holds. So the arm is made always admitted by
making that premise hold and checking it at every boot, not by shrinking the intruder.

## 1. Pre-registration

**The design** (`day55-stall-cell.sh`: `day35-stall-cell.sh` with two additions; boots, budgets, harness
`day35_stall_cell.py`, tenant window and pass order unchanged):

- (a) **A card guard before every boot**: no compute app listed and card `memory.used` at most 2,048 MiB (an idle card
  between this cell's own boots reads 15 MiB, `rtx5090-day35/stall/ev/pass2/*/card.before.csv`), checked up to 16
  times 60 s apart; the boot then starts, recorded `guard=clean` or `guard=timeout` in its `BOOT.txt`, every check in
  `ev/guard.log`. Nothing seen is touched.
- (b) **One warm-up prime boot first** (`--n 1`, part of no quantity), so the first measured prime boot is not the
  hold's first boot on a cold card (day 37's low first pass).
- Unchanged: pass 1 = prime, off, on; pass 2 = on, off, prime; N=5 per arm per order inside every harness run; one
  collector hold on `/tmp/memra-5090.lock` (`day55-local-run.sh`, the day-35 runner), 250 ms telemetry, the cell's own
  1 s CSV; the binary is this lane's `memra-server` (verify digest v3 tree `256c3c640`, SHA-256 in the receipt).

**Acceptance** (`day55-guard.py`, constants fixed here), the verdict `prime_always_admitted` iff all hold:
- every one of the seven boots `guard=clean`;
- no boot's window (its `boot-` to `stopped-` marks) in the 1 s CSV peaks above 14,000 MiB (the 9B server's clean
  footprint read 7,641 to 9,753 MiB; day 35's co-tenant boots read 20,286 to 23,230);
- both measured prime boots print zero `[admit-oom] VRAM defer` and zero `[admit-oom] capacity reject` lines, and
  their receipts carry zero intruder errors.
Otherwise `prime_not_admitted` with every failing term named. The reader was dry-checked on day 35's receipts
(`guard=missing` everywhere since day 35 had no guard; pass 1's co-tenant peaks 23,230 / 20,286 / 20,318 MiB and its
2,633 defers and 7 rejects read as recorded; pass 2 clean).

**Readings beside it**, no clause: `day35-stall-reading.py` unchanged on `ev/` (every arm's stall per pass, the
ON-minus-OFF differences, the attribution lines) and the prime arm per pass against the warm-up's single run, with
each boot's temperature range: day 37's low-first-pass question, read as it comes out.

**What each card can decide.** This is the RTX 5090 class's control arm; nothing is compared with the target card.

## 2. Results (RTX 5090 Laptop GPU, `rtx5090-day55/stall/`)

One collector hold, 01:01:42Z (the warm-up boot) to 01:13:09Z, seven boots, `memra-server-v3` `709f079c...`, the 9B.
Regime (`regime.log`, the collector's 250 ms CSV, N=2725): SM 1335 to 2332 MHz, power 30.6 to 174.0 W, 65 to 89 C
(this card under load throughout, 86 C median). Every receipt `replay=PASS`.

Verbatim (`stall/guard-reading.log`):

`DAY55 BOOT warmup/prime guard=clean admit_defers=0 admit_rejects=0 window_peak_memory_used_mib=9753 cotenant=no prime_run_errors=0`

`DAY55 BOOT pass1/prime guard=clean admit_defers=0 admit_rejects=0 window_peak_memory_used_mib=9753 cotenant=no prime_run_errors=0`

`DAY55 BOOT pass1/off guard=clean admit_defers=0 admit_rejects=0 window_peak_memory_used_mib=7641 cotenant=no`

`DAY55 BOOT pass1/on guard=clean admit_defers=0 admit_rejects=0 window_peak_memory_used_mib=7705 cotenant=no`

`DAY55 BOOT pass2/on guard=clean admit_defers=0 admit_rejects=0 window_peak_memory_used_mib=7705 cotenant=no`

`DAY55 BOOT pass2/off guard=clean admit_defers=0 admit_rejects=0 window_peak_memory_used_mib=7641 cotenant=no`

`DAY55 BOOT pass2/prime guard=clean admit_defers=0 admit_rejects=0 window_peak_memory_used_mib=9782 cotenant=no prime_run_errors=0`

`DAY55 VERDICT -> prime_always_admitted`

C8's acceptance holds: the prime arm is admitted in both passes, every boot started on a clean card, and no co-tenant
appeared. The readings beside it (`stall/reading.log`, `day35-stall-reading.py` unchanged):

`DAY35 STALL VERDICT: prime pass1 255.4 (iqr 27.5) admissible; demote pass1 off 61.8 on 45.3 on-off -16.6 (unc 4.5) inadmissible; promote pass1 off 47.5 on 39.3 on-off -8.2 (unc 1.8) isolated; prime pass2 304.6 (iqr 13.2) admissible; demote pass2 off 68.2 on 46.1 on-off -22.0 (unc 3.5) inadmissible; promote pass2 off 50.4 on 40.6 on-off -9.8 (unc 2.8) isolated; admissible=False`

Read, not tuned: the demote class's ON receipts are `admissible=False` by day 35's own line census (`run 4 demote
without the door's three lines`, and so on: today's tree prints the off-tick demote's lines, not day 35's three), a
reader written for the day-35 program, so no demote difference is claimed here; the promote class reads the door
ON 8.2 and 9.8 ms below OFF (`isolated` both passes). Day 37's low first prime pass persists with the warm-up boot
(255.4 against 304.6, the pass-1 IQR 27.5): the warm-up did not remove it, so it is not the hold's first boot.
