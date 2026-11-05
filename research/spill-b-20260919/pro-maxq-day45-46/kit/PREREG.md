# admit-w-release on the RTX PRO 6000 Blackwell Max-Q: pre-registration

Written 2026-11-04, before any Max-Q boot of this cell. Nothing here changes after a number is seen. A failed clause
is recorded as it reads; a revision is a new, dated addendum written before its code or cell.

## 1. Why

The owner ruled on 2026-11-04 that the RTX PRO 6000 target is the Max-Q. The W-release default
(`docs/decisions/ADMIT-W-RELEASE-DEFAULT.md`, `w_release_for_device` in `crates/memra-server/src/worker.rs`) fires
only on the full-power Workstation Edition, because its deciding cell ran on one Workstation card at 600 W. The
record's Rejected list keeps the Max-Q on whole-life booking "until a cell on them passes". This is that cell.

## 2. The cell

The two halves that settled the Workstation default, replayed at their deciding trees with portability fixes only
(`DEVIATIONS.md`). Arms, orders, N, prompts and client flags are the registered ones.

| Half | Chain tree (scripts, `docs/SERVING.md`) | Arm binary source | Boots, in order | Client flags |
|---|---|---|---|---|
| DAY45 2.2 (addendum B) | `0ac7f6413532b805fbce5ab9763d38b4b373606e` | `09badfe575fc156d177e9fce3835da6bcde5929f` | O1-off, O1-on, O2-on, O2-off | `--burst 64 --length 30720 --max-tokens 64 --wave2-delay-s 10` |
| DAY46 2.2 (addendum C) | `43dbdb29fa2a52be7744d0a03a617930f9426bf5` | `8926ccfb3e8a27cf8f91aa747e2b088251215c7a` | O1-shadow, O1-enforce, O1-enforce-wrel, O2-enforce-wrel, O2-enforce, O2-shadow | `--burst 64 --length 30720 --max-tokens 64` |

Both halves: Qwen3.8-27B-NVFP4-Q5K-mtp.gguf (sha256 `1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a`)
at its declared context (`MEMRA_CTX` unset), `MODEL_KEY=q38`, `N=5`, `MEMRA_TIMEOUT_MS_MAX=3600000`, port 18526,
`MEMRA_ADMIT_PREDICT_SHADOW=1` on every boot. DAY45 arms: `off` and `on` (`MEMRA_ADMIT_W_RELEASE=1`), both with
`MEMRA_ADMIT_BY_MEMORY=1` (`DOOR_MEMORY=1`). DAY46 arms: `shadow`, `enforce` (`MEMRA_ADMIT_PREDICT_ENFORCE=1`),
`enforce-wrel` (`+ MEMRA_ADMIT_W_RELEASE=1`), `MEMRA_ADMIT_BY_MEMORY` unset on every arm.

## 3. The clauses and the rule, quoted from the Workstation pre-registration

### DAY45 (`research/spill-b-20260919/DAY45.md` 1.5, revised by addendum B 1.8)

> - **W1 exact books.** On both arms, the probe's `[admit-predict]` line reads `booked_bytes=0` and `booked_real=0`
>   (nothing in flight is booked), and the final `/metrics` `admission_booked_bytes` is 0.
> - **W2 one release per session.** On `on`, every admitted burst and warm request prints one `w-release` line before its
>   retire, with `bytes` equal to the `W` printed for it at admission (`[admit-book] w-booked id=<id> bytes=<W>`, printed
>   on both arms), and no session releases twice.
> - **W3 identity.** Every request's completion digest equals the other arm's in each order.
> - **W4 health.** No `CUDA_ERROR_OUT_OF_MEMORY` line, no 503, no crash line.

Addendum B (1.8): "W2 then reads: every booked id prints exactly one of `w-release` or `w-retire-unreleased`, bytes
equal, no id twice." "W3 compares the requests that are `200` on both arms and prints the status mismatches as a
reading." "W1 and W4 unchanged. Cells: both cards, both orders, 4 boots each."

The Workstation reading (DAY45.md 2.2): "Every clause PASS on the target card under addendum B".

### DAY46 (`research/spill-b-20260919/DAY46.md` 1.3, revised by addendum A 1.6; addendum C 1.8 fixes the client)

> - **P1 typed refusals.** On both enforcing arms every refused request is a 429 with `Retry-After` in 1 to 60 and a
>   `[admit-predict] ... verdict=reject-kv ... enforce=1` line with the same id; no other non-200 on any arm.
> - **P2 no OOM.** No `CUDA_ERROR_OUT_OF_MEMORY` line, no parked prefill or step OOM, no 503, no crash line on any boot.
> - **P3 within the budget.** On both enforcing arms, at every admitted request's `[admit-predict]` line,
>   `booked_bytes + kv_hat <= budget_bytes`.
> - **P4 identity.** Every request admitted on an enforcing arm has the completion digest of the same request on
>   `shadow` in the same order (admission chooses who runs, not what they output).
> - **P5 the release reaches the door.** On `enforce-wrel`, every admitted session prints one `w-release` line before its
>   retire (DAY45 W2's rule), and the probe's line reads `booked_bytes=0 booked_real=0` on every arm (the books are
>   exact at idle).

Addendum A (1.6): "P2 therefore applies to the two enforcing arms, and on `shadow` the OOM, 503 and crash counts are a
reading (the before). P1's "no other non-200 on any arm" also becomes the enforcing arms only. P4 compares the
requests that are `200` on both boots of the pair, and prints the status mismatches as a reading". "P5 uses DAY45
addendum B's W2 rule". Addendum C (1.8): "the second wave is released when the burst's first request completes
`200` ... No clause, bound or reading changes".

The Workstation reading (DAY46.md 2.2): "P1 to P5 PASS on all four enforcing boots, both orders."

### The rule this cell is judged by

The registered readers (`day45-read.py`, `day46-read.py`, unchanged) print the clause lines. `verdict.py` applies:

- **DAY45 PASS** when these 12 lines read PASS: W4 and W1 on each of the 4 boots, W2 on O1-on and O2-on, W3 for O1
  and O2.
- **DAY46 PASS** when these 24 lines read PASS: P2, P1 and P3 on each of the 4 enforcing boots, P5-PROBE on all 6
  boots (P5's "on every arm" half), P5 on O1-enforce-wrel and O2-enforce-wrel, P4 for each order and enforcing arm.
- A line counts only for a boot that finished (`<boot>.exit` and `finished.txt`). A boot cut by the kit's deadline or
  thermal stop, a server that never became ready, or a missing line makes the half **INCOMPLETE**, and nothing is
  concluded from it ("died, cause unknown, repro needed", AGENTS.md).
- **VERDICT PASS** when both halves PASS. **FAIL** when any finished boot's required line reads FAIL. **INCOMPLETE**
  otherwise. One line, last in `verdict.out`, also in `out-*/VERDICT.txt`:
  `VERDICT admit-w-release-default card=pro6000maxq day45=<state>(<n>/12) day46=<state>(<n>/24) -> <PASS|FAIL|INCOMPLETE>`

The registered readers print no VERDICT line of their own; the Workstation verdicts were written in DAY45.md and
DAY46.md from the same clause lines. `verdict.py` encodes those two sentences. Dry run on the Workstation receipts
(`pro-single-day45/box-b/`, `pro-single-day46/box-c/` at `2ccb038d1c`): the registered readers reproduce both
`read.log` files byte for byte, and `verdict.py --card pro6000` prints
`VERDICT admit-w-release-default card=pro6000 day45=PASS(12/12) day46=PASS(24/24) -> PASS`.

## 4. What each outcome decides

- **PASS:** the W release default may extend to the Max-Q by the change named in the scope (a Max-Q helper beside
  `HardwareTarget::is_pro6000_workstation`, used only in `w_release_for_device`; the other Workstation-keyed defaults
  are not widened). The decision record names this cell, its card index and its power limit.
- **FAIL:** the Max-Q keeps whole-life booking. The failing line is quoted and placed from the receipts before any
  revision.
- **INCOMPLETE:** nothing is concluded; rerun the cut half (`run-part1.sh` or `run-part2.sh`).

## 5. Max-Q specifics

- Card: the index the window holder passes as `CARD` at run time. The server sees only that card
  (`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$CARD`); every idle check, snapshot and telemetry sample reads
  `nvidia-smi -i $CARD` only.
- Power: <box> reads power.limit 250, 250, 280, 280 W on GPU 0 to 3 (default 300 W, max 300 or 325 W), checked read-only
  on 2026-11-04 21:59 Jerusalem; the owner has not changed them. The run records `power.limit` and
  `enforced.power.limit` of its card in `out-*/card.csv`, and each half's `card.csv` (name, power.limit,
  memory.total[, driver]). The Workstation cell ran at 600 W on driver 580.65.06; <box> runs driver 615.71.09.
- Shared box: GPUs 0, 1 and 3 train during the run. Every clause is untimed pass/fail, so it reads validly beside
  them. The readings (TTFT, the second wave's admitted count, booked figures) are labeled "taken on a shared box at
  <power.limit>" and carry no bound; they are not compared with the Workstation's numbers.
- Thermal: the owner's Max-Q rule stops the cell at >= 92 C with fan >= 95 % on 3 consecutive 10 s checks, or at
  >= 95 C. A stop makes the cut half INCOMPLETE.
- Timed clauses: none. Every clause of both halves is untimed pass/fail, so there is no `run-timed.sh`.

## 6. Readings, no bound

As registered: the peak books, the shadow `reject-kv` counts, per-wave 200 and 429 counts, `booked_at_first_line` per
wave, TTFT p50/p95 with N, time to 429, `budget_bytes`. The value reading (DAY46 addendum A): the second wave's
admitted count on `enforce-wrel` against `enforce`, per order. On the Workstation: 11 of 32 against 1 of 32, the
book at the second wave 20.47 GB against 60.60 GB. Plus the kit's labels: telemetry (250 ms, the card only),
`gpu_watch_fault_lines` per half, the box state at start and end.
