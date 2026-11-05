# WP-B day 45: O4, the outstanding-only `W` release at prime completion

OWED.md O4, named in DAY24 (sections 1 and 3: "release the workspace charge when the prime completes, keep the
persistent terms ... needs a prime-completion seam in both books and a retire-side accounting change"). Both admission
books (the predictive shadow book, `AdmissionBook.shadow_booked_bytes` from `shadow_kv_hat`, and the real book,
`booked_bytes` from `booked_kv_bytes`) carry each session's prefill workspace `W` for its whole life, though the
workspace is live only while the session primes. The predictive door (`MEMRA_ADMIT_PREDICT_SHADOW` and `_ENFORCE`,
both default OFF) reads the shadow book, and `/metrics` publishes the real one (`admission_booked_bytes`). Under
`MEMRA_ADMIT_BY_MEMORY` the booked reading already counts `W` only for still-priming sessions (`pending_prime`), so
this item is the two books' side. No default moves.

## 1. Pre-registration

Committed and pushed before any day-45 code and before any day-45 cell. Nothing in section 1 changes after a number is
seen; a failed clause is recorded as it reads and a revision is a new, dated addendum pushed before its code.

### 1.1 The arm (`MEMRA_ADMIT_W_RELEASE`, default unset; decide-by 2026-10-10)

Unset: today's program and today's books, byte for byte. `1`:

- **At the admit seam** each session records `booked_w_bytes`, the prefill-workspace term of its own charge
  (`RequestCharge::from_physical_cost(cost, ..).prefill_workspace_bytes`, the one term both books already carry).
- **The prime-completion seam, one site:** at the tick top, before admission, every active session whose prime has
  completed (`prefill_done`) and whose `W` is still booked releases it from both books (`AdmissionBook::release`,
  model row) and reduces its own `booked_kv_bytes` and `shadow_kv_hat` by it, once. Every prime path reaches
  `prefill_done`, so one site covers them all.
- **Retire side:** the retire seam subtracts the session's (reduced) charges as today, so both books stay exact by
  construction: a session that retires before its prime completes still carries `W` and retires it.
- **Receipt:** `[admit-book] w-release id=<id> model=<m> bytes=<W> booked_real=<after> booked_shadow=<after>`.
- **Stated limits:** a step-OOM park that re-admits a session books it again at re-admission (today's rule); the
  prime-slab high-water (the shared slab, DAY39) is not a per-session term and is not touched.

### 1.2 One numeric program

The arm changes only accounting reads: the predictive door's verdicts (when armed) and the published booked figure.
No token path reads either book, so outputs are unchanged by construction; E3 below measures it.

### 1.3 CPU before the cards

Unit tests: `AdmissionBook::release` (both books, saturating, inflight unchanged), the per-session release is
idempotent, a release then retire equals a retire of the full charge. A census test: the door is read at the admit
seam and the one release site only.

### 1.4 The cell (`day45-client.py`, one boot = one arm)

`MEMRA_ADMIT_PREDICT_SHADOW=1` (log only: no verdict refuses, so both arms serve the same sequence), the prefix cache
at its default. Sequence: 4 warm requests of 512 tokens; then a burst of B requests released together (distinct
prompts of L tokens, `max_tokens = 64`); after the burst drains and 5 s idle, one probe request of 512 tokens. 5090: the
9B at `MEMRA_CTX=65536`, B = 32, L = 6,144. Target card: the 27B, B = 64, L = 30,720. Arms `off` (door unset) and
`on` (`=1`), orders O1 (off, on) and O2 (on, off): 4 boots per card.

### 1.5 Clauses

- **W1 exact books.** On both arms, the probe's `[admit-predict]` line reads `booked_bytes=0` and `booked_real=0`
  (nothing in flight is booked), and the final `/metrics` `admission_booked_bytes` is 0.
- **W2 one release per session.** On `on`, every admitted burst and warm request prints one `w-release` line before its
  retire, with `bytes` equal to the `W` printed for it at admission (`[admit-book] w-booked id=<id> bytes=<W>`, printed
  on both arms), and no session releases twice.
- **W3 identity.** Every request's completion digest equals the other arm's in each order.
- **W4 health.** No `CUDA_ERROR_OUT_OF_MEMORY` line, no 503, no crash line.

Readings, no bound: the peak shadow `booked_bytes` and `booked_real` across the burst's `[admit-predict]` lines per arm;
the count of `verdict=reject-kv` shadow verdicts per arm (what the enforcing door would refuse, O6's input); the time
from each session's admission to its release.

### 1.6 Price

Code: about 0.3 agent-day. Cells: about 40 min on the 5090, about 1.5 h on the target card.

### 1.7 Addendum A (2026-09-26, while coding, before any cell)

The release site is after the tick's command drain and before admission, not at the loop head: a lane-A census window
pins the loop head's first statements, and the drain-then-admit point is the same seam for the admission that
follows (an admission reads the books after the release either way). No clause, bound or reading changes.

## 2. Results

Written after the runs. Section 1 is unchanged.

### 2.1 The target card (the tenth sitting, one RTX PRO 6000 Blackwell Workstation Edition at 600 W, 2026-09-26 13:51 to 14:32Z)

Binary built on the box from `21b081ee1`, sha256 `97d2021d...aa540c0c6`; receipts at `pro-single-day45/box/` (107 files,
the box manifest checked, the binary by hash only). The boot-derived predictive budget reads `budget_bytes=65920413392`.
Verbatim:

```
DAY45 W4 card=pro6000 boot=O1-off oom_lines=55 crash_lines=0 r503=53 -> FAIL
DAY45 W1 card=pro6000 boot=O1-off probe_booked=0 probe_booked_real=0 metrics_booked=0 -> PASS
DAY45 READING card=pro6000 boot=O1-off predict_lines=68 peak_booked_shadow=231005913608 peak_booked_real=231194498056 shadow_reject_kv=47
DAY45 W4 card=pro6000 boot=O1-on oom_lines=55 crash_lines=0 r503=53 -> FAIL
DAY45 W1 card=pro6000 boot=O1-on probe_booked=0 probe_booked_real=0 metrics_booked=0 -> PASS
DAY45 READING card=pro6000 boot=O1-on predict_lines=68 peak_booked_shadow=231005913608 peak_booked_real=231194498056 shadow_reject_kv=47
DAY45 W2 card=pro6000 boot=O1-on w_booked=69 w_release=11 twice=[] bytes_mismatch=[] never_released=['cmpl-1a4bf5d0a3310c03a06b4e3159eb623d', 'cmpl-23dd300f87f0282c34f953f2ada46c52', 'cmpl-27b175e645c90837e0676855049a8927', 'cmpl-2d451e437d86201e87379c00550246d3'] -> FAIL
DAY45 W4 card=pro6000 boot=O2-off oom_lines=55 crash_lines=0 r503=53 -> FAIL
DAY45 W1 card=pro6000 boot=O2-off probe_booked=0 probe_booked_real=0 metrics_booked=0 -> PASS
DAY45 READING card=pro6000 boot=O2-off predict_lines=68 peak_booked_shadow=231005913608 peak_booked_real=231194498056 shadow_reject_kv=47
DAY45 W4 card=pro6000 boot=O2-on oom_lines=55 crash_lines=0 r503=53 -> FAIL
DAY45 W1 card=pro6000 boot=O2-on probe_booked=0 probe_booked_real=0 metrics_booked=0 -> PASS
DAY45 READING card=pro6000 boot=O2-on predict_lines=68 peak_booked_shadow=231005913368 peak_booked_real=231194497816 shadow_reject_kv=47
DAY45 W2 card=pro6000 boot=O2-on w_booked=69 w_release=11 twice=[] bytes_mismatch=[] never_released=['cmpl-01b57d15a858c30c3edfc465a7a8698d', 'cmpl-024f302e88e494c79bd90fd8b4a0f146', 'cmpl-07d03b7ca147010b84e26eed69f4be0c', 'cmpl-0a92c980009220e0eec841d6c8a719ea'] -> FAIL
DAY45 W3 card=pro6000 order=O1 rows=69 differ=['burst-5', 'burst-7'] -> FAIL
DAY45 W3 card=pro6000 order=O2 rows=69 differ=[] -> PASS
```

**Clauses, as they read, and where each failure sits (placed before any fix):**

- **W1 PASS on all four boots**: the probe reads `booked_bytes=0 booked_real=0` and the final `/metrics` booked figure
  is 0 on both arms; both books are exact at idle.
- **W4 FAIL on all four boots, both arms, identically** (55 OOM lines, 53 of the 64 burst requests end `503`
  overloaded). Placed: without an admission door the physical gate admitted all 64 sessions of 30,720 tokens (the real
  book peaks at 231 GB against a 66 GB predictive budget; the shadow door would have refused 47), and the batched prime
  then ran out of memory (`[prime-batch] failed after a partial pipeline wave (DriverError(CUDA_ERROR_OUT_OF_MEMORY,
  ...)); dropping tainted sessions`). That is today's program without `MEMRA_ADMIT_BY_MEMORY` on a burst past the card
  (the door's own subject, O3), the same on both arms; the W release is not in it. The cell's shape was wrong for this
  item: it overloads the card before the books matter.
- **W2 FAIL on both `on` boots** (`w_booked=69 w_release=11`, no double release, no byte mismatch). Placed by the ids:
  the 58 never released are the 53 sessions the prime-batch OOM dropped before their primes completed, and the 5 short
  requests (the 4 warm and the probe) that primed and retired inside one tick. Both exits retire the full charge at the
  retire seam by design (1.1: "a session that retires before its prime completes still carries `W` and retires it");
  W2's wording did not name them. W1's exact books confirm the retire side.
- **W3 FAIL in O1, PASS in O2.** Placed: `burst-5` and `burst-7` are `503` on one arm and `200` on the other (the OOM's
  drop set differs between boots); the 15 requests that are `200` on both arms have equal digests.
- **Readings:** the peak books are equal on both arms (231.0 and 231.2 GB) because the burst's admissions all precede
  its first prime completion; the release moves the books only after that, which this shape never reached.

### 1.8 Addendum B (2026-09-26, after 2.1, before any code of the revision)

- **The cell runs with `MEMRA_ADMIT_BY_MEMORY=1` on both arms**, so the burst is shaped by the memory door (defer or
  the typed 429) and no prime runs past the card; the door's decisions do not read the two books, so both arms serve
  the same admissions. The predictive shadow stays on (log only).
- **A second wave:** 10 s after the first burst's last admission, a second wave of B/2 requests (distinct prompts, the
  same L and `max_tokens`), so admissions arrive after some primes completed and the release can move the shadow
  verdicts. Readings add, per wave and arm, the `reject-kv` shadow verdicts and the booked figures at admission.
- **The retire receipt:** at retire, a session whose `W` is still booked prints `[admit-book] w-retire-unreleased
  id=<id> bytes=<W> reason=<prime-incomplete|same-tick>` (the door on only). W2 then reads: every booked id prints
  exactly one of `w-release` or `w-retire-unreleased`, bytes equal, no id twice.
- **W3 compares the requests that are `200` on both arms** and prints the status mismatches as a reading.
- W1 and W4 unchanged. Cells: both cards, both orders, 4 boots each.

### 2.2 Addendum B on the target card (the eleventh sitting, one RTX PRO 6000 Blackwell Workstation Edition at 600 W, 2026-09-26 17:07 to 17:41Z)

Tree `0ac7f6413`; the arm binary built on the box from `09badfe57` (addendum B's code), sha256 `2efafc68...a1d525ea` (in
`box-binaries.sha256`); receipts at `pro-single-day45/box-b/` (107 files, the box mirror manifest checked, the binary by
hash only). The burst of 64 x 30,720 tokens then a second wave of 32, `MEMRA_ADMIT_BY_MEMORY=1` and the predictive
shadow on both arms. Verbatim (`read.log`; each boot's three READING lines are the same on all four boots and are
printed once here):

```
DAY45 W4 card=pro6000 boot=O1-off oom_lines=0 crash_lines=0 r503=0 -> PASS
DAY45 W1 card=pro6000 boot=O1-off probe_booked=0 probe_booked_real=0 metrics_booked=0 -> PASS
DAY45 READING card=pro6000 boot=O1-off predict_lines=68 peak_booked_shadow=181785235088 peak_booked_real=181933670544 shadow_reject_kv=47
DAY45 READING card=pro6000 boot=O1-off wave=burst predict_lines=64 shadow_reject_kv=47 booked_shadow_at_first=0 booked_shadow_at_last=181785235088
DAY45 READING card=pro6000 boot=O1-off wave=wave2 predict_lines=32 shadow_reject_kv=32 booked_shadow_at_first=181785235088 booked_shadow_at_last=181785235088
DAY45 W4 card=pro6000 boot=O1-on oom_lines=0 crash_lines=0 r503=0 -> PASS
DAY45 W1 card=pro6000 boot=O1-on probe_booked=0 probe_booked_real=0 metrics_booked=0 -> PASS
DAY45 W2 card=pro6000 boot=O1-on w_booked=53 w_release=48 w_retire_unreleased=5 reasons={'same-tick': 5} twice=[] bytes_mismatch=[] neither=[] -> PASS
DAY45 W4 card=pro6000 boot=O2-off oom_lines=0 crash_lines=0 r503=0 -> PASS
DAY45 W1 card=pro6000 boot=O2-off probe_booked=0 probe_booked_real=0 metrics_booked=0 -> PASS
DAY45 W4 card=pro6000 boot=O2-on oom_lines=0 crash_lines=0 r503=0 -> PASS
DAY45 W1 card=pro6000 boot=O2-on probe_booked=0 probe_booked_real=0 metrics_booked=0 -> PASS
DAY45 W2 card=pro6000 boot=O2-on w_booked=53 w_release=48 w_retire_unreleased=5 reasons={'same-tick': 5} twice=[] bytes_mismatch=[] neither=[] -> PASS
DAY45 W3 card=pro6000 order=O1 rows_200_both=53 status_mismatch=0 differ=[] -> PASS
DAY45 W3 card=pro6000 order=O2 rows_200_both=53 status_mismatch=0 differ=[] -> PASS
```

- **Every clause PASS on the target card under addendum B:** W1 and W4 on all four boots (no OOM, no 503, no crash;
  both books 0 at idle), W2 on both `on` boots (each of the 53 booked ids prints exactly one of `w-release` (48) or
  `w-retire-unreleased` (5, all `same-tick`: the 4 warm requests and the probe), bytes equal, none twice), W3 in both
  orders (53 rows `200` on both arms, equal digests, no status mismatch).
- **The memory door shaped the burst identically on every boot** (from each boot's `client.jsonl`): 48 burst requests
  `200` and 16 typed `429`, all 32 second-wave requests `429`, the 4 warm and the probe `200`.
- **Readings.** The peak books are 181.8 GB shadow and 181.9 GB real on every boot; the shadow door would refuse 47 of
  the burst and all 32 of the second wave, on both arms. Time from a burst session's `w-booked` line to its
  `w-release` line (computed from the two `on` server logs' epoch-ms prefixes; the reader prints no such line): O1-on
  N=48 p50 458,444 ms, p95 458,512 ms, min 217,550 ms; O2-on N=48 p50 458,318 ms, p95 458,389 ms, min 217,930 ms.
- **What the shape does not show.** The second wave arrives 10 s after the burst's last admission, and the first burst
  prime completes 217 s after its booking, so every release lands after the second wave was refused. Both books
  therefore read the same at every admission on both arms. On this card and shape the release is exact and harmless.
  Whether it moves a verdict needs arrivals after prime completions. That belongs to O6's `enforce-wrel` arm, the
  one door that reads the book; DAY46 is revised for it before any code.

### 1.9 Addendum C (2026-11-04, the Max-Q pre-registration, written before any Max-Q boot)

Written 2026-11-04, before any Max-Q boot of this cell. Nothing here changes after a number is seen. A failed clause
is recorded as it reads; a revision is a new, dated addendum written before its code or cell.

#### M.1 Why

The owner ruled on 2026-11-04 that the RTX PRO 6000 target is the Max-Q. The W-release default
(`docs/decisions/ADMIT-W-RELEASE-DEFAULT.md`, `w_release_for_device` in `crates/memra-server/src/worker.rs`) fires
only on the full-power Workstation Edition, because its deciding cell ran on one Workstation card at 600 W. The
record's Rejected list keeps the Max-Q on whole-life booking "until a cell on them passes". This is that cell.

#### M.2 The cell

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

#### M.3 The clauses and the rule, quoted from the Workstation pre-registration

##### DAY45 (`research/spill-b-20260919/DAY45.md` 1.5, revised by addendum B 1.8)

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

##### DAY46 (`research/spill-b-20260919/DAY46.md` 1.3, revised by addendum A 1.6; addendum C 1.8 fixes the client)

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

##### The rule this cell is judged by

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

#### M.4 What each outcome decides

- **PASS:** the W release default may extend to the Max-Q by the change named in the scope (a Max-Q helper beside
  `HardwareTarget::is_pro6000_workstation`, used only in `w_release_for_device`; the other Workstation-keyed defaults
  are not widened). The decision record names this cell, its card index and its power limit.
- **FAIL:** the Max-Q keeps whole-life booking. The failing line is quoted and placed from the receipts before any
  revision.
- **INCOMPLETE:** nothing is concluded; rerun the cut half (`run-part1.sh` or `run-part2.sh`).

#### M.5 Max-Q specifics

- Card: the index the window holder passes as `CARD` at run time. The server sees only that card
  (`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$CARD`); every idle check, snapshot and telemetry sample reads
  `nvidia-smi -i $CARD` only.
- Power: the box reads power.limit 250, 250, 280, 280 W on GPU 0 to 3 (default 300 W, max 300 or 325 W), checked read-only
  on 2026-11-04 19:59Z; they have not been changed. The run records `power.limit` and
  `enforced.power.limit` of its card in `out-*/card.csv`, and each half's `card.csv` (name, power.limit,
  memory.total[, driver]). The Workstation cell ran at 600 W on driver 580.65.06; the box runs driver 615.71.09.
- Shared box: GPUs 0, 1 and 3 train during the run. Every clause is untimed pass/fail, so it reads validly beside
  them. The readings (TTFT, the second wave's admitted count, booked figures) are labeled "taken on a shared box at
  <power.limit>" and carry no bound; they are not compared with the Workstation's numbers.
- Thermal: the Max-Q thermal rule stops the cell at >= 92 C with fan >= 95 % on 3 consecutive 10 s checks, or at
  >= 95 C. A stop makes the cut half INCOMPLETE.
- Timed clauses: none. Every clause of both halves is untimed pass/fail, so there is no `run-timed.sh`.

#### M.6 Readings, no bound

As registered: the peak books, the shadow `reject-kv` counts, per-wave 200 and 429 counts, `booked_at_first_line` per
wave, TTFT p50/p95 with N, time to 429, `budget_bytes`. The value reading (DAY46 addendum A): the second wave's
admitted count on `enforce-wrel` against `enforce`, per order. On the Workstation: 11 of 32 against 1 of 32, the
book at the second wave 20.47 GB against 60.60 GB. Plus the kit's labels: telemetry (250 ms, the card only),
`gpu_watch_fault_lines` per half, the box state at start and end.

**Amendment, 2026-11-05, before the first boot: the power limit.**

Owner ruling, 2026-11-05: the cell may raise its card's power limit to 300 W, must note thermal throttling, and
must restore the limit after use. This replaces the statement above that the limits have not been changed. The kit
is unchanged: it still only records the limit it reads.

- Before the cell boots, the runner, which sits outside the kit, sets the card under test to 300 W, the card's
  default limit (`nvidia-smi -i <card> -pl 300`), and reads it back, so the receipts record 300 W. When the cell
  ends, on any path, the runner restores the limit it read before (250 W on index 0 and 280 W on index 2 at kit
  time) and reads it back. The slot guard and the gap scheduler also restore it when they stop.
- Throttle telemetry: every 10 s during the cell the runner records temperature, fan, power draw, power limit, SM
  clock and four clock-event reasons (SW power cap, SW thermal slowdown, HW thermal slowdown, HW power brake) in
  `thermal-<cell>.csv`. It prints one summary line with the maximum temperature and fan and the number of samples
  with each reason active. SW power cap is the normal state of a card running at its limit; it is recorded and not
  judged.
- Reading: every clause is a pass/fail line from the registered readers, so the throttle summary is recorded with the
  receipt and not judged.
- Neighbour rule, asked by the box's training session: this cell runs on GPU 2 beside training on GPUs 0, 1 and 3,
  so every 10 s the runner also reads GPUs 1 and 3. If either reaches 90 C, GPU 2 returns to its previous limit (280
  W) for the rest of the cell, and the summary line records when. The verdict does not depend on power, so the drop
  changes no clause.
- The thermal stop above is unchanged. The runner applies the same rule on its 10 s samples as a
  backstop, and a stopped cell has no verdict.

### 2.3 Addendum C on the RTX PRO 6000 Blackwell Max-Q (`pro-maxq-day45-46/`, 2026-11-05 05:22:49Z to 07:15:32Z)

The cell ran as addendum C and its power amendment registered, in two parts on GPU 2 of the four-card box ("NVIDIA
RTX PRO 6000 Blackwell Max-Q Workstation Edition", driver 615.71.09) while the box's other three cards trained, under
the GPU 2 lock agreed with the training session. Part 1 (`run-part1.sh`, the DAY45 half, 4 boots) ran 05:22:49Z to
06:07:50Z; part 2 (`run-part2.sh`, the DAY46 half, 6 boots) ran 06:07:55Z to 07:15:32Z and read the DAY45 half from
part 1's out dir, as the kit's README states. Binaries: `memra-server` from `09badfe575fc` (`9f40fa7d...`) and
`8926ccfb3e8a` (`e8d9454d...`), the chain trees `0ac7f6413` and `43dbdb29f` as M.2 registered; the model hashed
`1facf36c...` in both parts before the first boot. Receipts mirrored file for file (each part's `MIRROR-CHECK.txt`:
292 `OK` lines in all; the one line that fails in each is `run.out`, which the runner appends to after writing the
manifest, and the box's final copy hashes equal to this mirror's), then exported with
`research/spill-f-20260919/m1-box-sanitize.py`: the card's UUID became `GPU-<card2-uuid>`, and the box name, local time
zone and remote home path their placeholders (`EXPORT-MANIFEST.json`, 29 files). The scheduler's try logs, power summaries and 10 s
thermal samples are in `runner/`.

Power and validity (M.5 and its amendment): the runner set GPU 2 to 300 W (from 280 W, max 300 W) before each part.
The box re-applies its own per-card caps every 10 minutes (a system timer, 280 W on this card), so the card ran at 280
W from 05:30:03Z in part 1 (43 of 268 samples at 300 W) and from 06:10:26Z in part 2 (15 of 402); the runner's
restore found 280 W set both times. The part 1 summary line (`runner/power-w-release-part1-try1.txt`) reports the
limit it read at the start; the samples are the record. No clause depends on power (M.5). Maximum 91 C, fan 91 %; SW
power cap active in 637 of 670 samples, no thermal or power-brake slowdown, no thermal stop; GPUs 1 and 3 at most 89 C,
so the neighbour rule never dropped the limit. No server log of the cell has a `[gpu-watch] CRITICAL` line
(`gpu_watch_fault_lines=0` in both halves). Verbatim (`day45/read.log`, `day46/read.log`, `VERDICT.txt`):

- DAY45, all 4 boots: `DAY45 W4 card=pro6000maxq boot=O1-off oom_lines=0 crash_lines=0 r503=0 -> PASS` and
  `DAY45 W1 card=pro6000maxq boot=O1-off probe_booked=0 probe_booked_real=0 metrics_booked=0 -> PASS`
- `DAY45 W2 card=pro6000maxq boot=O1-on w_booked=54 w_release=49 w_retire_unreleased=5 reasons={'same-tick': 5} twice=[] bytes_mismatch=[] neither=[] -> PASS` (O2-on reads the same)
- `DAY45 W3 card=pro6000maxq order=O1 rows_200_both=53 status_mismatch=2 differ=[] -> PASS` and
  `order=O2 rows_200_both=53 status_mismatch=1 differ=[] -> PASS` (addendum B made the status mismatches a reading)
- DAY46, both `enforce` and both `enforce-wrel` boots: `P2 ... oom_lines=0 parked_oom_lines=0 crash_lines=0 r503=0 -> PASS`,
  `P1 ... r429=78` (`enforce`) and `r429=67` (`enforce-wrel`) `without_reject_line=[] retry_after_out_of_1_60=[] other_non200=[] -> PASS`,
  `P3 ... over_budget=[] -> PASS`, `P5-PROBE ... probe_booked=0 probe_booked_real=0 -> PASS`; both shadow boots
  `P5-PROBE ... -> PASS`
- `DAY46 P5 card=pro6000maxq boot=O1-enforce-wrel w_booked=40 w_release=39 w_retire_unreleased=1 twice=[] bytes_mismatch=[] neither=[] -> PASS` (O2 reads the same)
- `DAY46 P4 ... order=O1 arm=enforce rows_200_both=21 status_mismatch=78 differ=[] -> PASS`, `arm=enforce-wrel rows_200_both=32
  status_mismatch=67 differ=[] -> PASS`; O2: 22 and 32 rows, `differ=[]`, PASS
- `VERDICT admit-w-release-default card=pro6000maxq day45=PASS(12/12) day46=PASS(24/24) -> PASS`

Readings, taken on a shared box at 280 W (no bound, M.6): the second wave's admitted count is 12 of 32 on
`enforce-wrel` against 1 of 32 on `enforce` in both orders, the book at the second wave's first line 20.47 GB against
60.60 GB; second-wave TTFT p50 156.7 s and 155.9 s on `enforce-wrel`, 14.7 s for the one admission on `enforce`. The
before (`shadow`, no door): 53 and 51 OOM lines, the burst ends `200` on 13 and 14 of 64, the second wave 32 of 32.
Budget 65.98 GB on every boot; telemetry 16,081 samples at 250 ms, SM clock p50 1627 MHz.

**Read as registered (M.4): PASS.** The W release becomes the Max-Q default by the change the scope names: a Max-Q
helper beside `HardwareTarget::is_pro6000_workstation`, used only in `w_release_for_device`. It lands as its own
commit. The other Workstation-keyed defaults are not widened; the memory door stays keyed to the Workstation, so on a
Max-Q the release runs in the DAY46 shape this half tested (door unset).
