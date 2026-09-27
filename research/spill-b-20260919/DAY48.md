# WP-B day 48: O8, the verify-graph pool debt in the predictive book (MoE plus linear-attention families)

OWED.md O8. DAY28 1.1 and 3.3 named the gap: the verify-graph pool (`DsparkVgraphs`, keyed `(segment start, vt)`,
backed by the driver's device graph memory pool, never released) grows per new key on the families whose verify graphs
engage by default (`vgraph_family_default`: linear layers AND routed MoE). The physical admission side charges its
projected remaining growth as `vg_debt` on top of the reserve (`dspark_vg_admission_debt`, the `[admission] dspark
verify-graph pool debt:` line); the predictive side does not (its budget subtracts the calibrated floor once). On the
dense 9B and 27B the pool never engages, so the gap was not measurable on this lane's models. This day builds the
predictive twin behind a default-OFF door and measures it on a MoE plus linear-attention model. No default moves.

## 1. Pre-registration

Committed and pushed before any day-48 code and before any day-48 cell. Nothing in section 1 changes after a number is
seen; a failed clause is recorded as it reads and a revision is a new, dated addendum pushed before its code. Text only
until DAY46 (the enforcing door's own cell) has read, because this term matters where the predictive door enforces.

### 1.1 The arm (`MEMRA_ADMIT_PREDICT_VG_DEBT`, default unset; decide-by 14 days after its code lands)

Unset: today's predictive verdict. `1`: the predictive verdict compares `request_kv_hat + booked_kv_hat` against
`budget_bytes - vg_debt`, the same `dspark_vg_admission_debt` the physical side adds to its reserve at the same
admission (one read per admission, printed on the `[admit-predict]` line as `vg_debt=`). The book itself is unchanged
(the debt is a per-model pool term, not a per-request charge).

### 1.2 The model and the cell

The model: Ornith-1.5-35B-A3B NVFP4 with its MTP head (`/data/ai-ml/hf-models/ornith15-gguf/Ornith-1.5-35B-A3B-NVFP4-
Q5K-mtp.gguf`, 20 GB, its sha256 recorded in the cell's receipts before the first boot; a GatedDeltaNet plus routed-MoE
trunk, so the verify-graph pool engages on the spec route by default). The 5090 holds it with little headroom; the
target card holds it with room, and it must be staged there (the lead's box preparation). The cell: `day46-client.py`'s
sequence and burst (DAY46 1.2) on the spec route, with prompts of varied lengths so the pool meets new
`(segment start, vt)` keys during the burst. Arms `enforce` (`MEMRA_ADMIT_PREDICT_ENFORCE=1`) and `enforce-vg`
(`=1` plus `MEMRA_ADMIT_PREDICT_VG_DEBT=1`), both orders, both cards.

### 1.3 Clauses

- **V1 the pool engages.** Each boot prints `[spec-vg]` pool lines and at least one admission with `vg_debt > 0`; a boot
  where it does not is no reading of the arm and says so.
- **V2 no OOM.** No `CUDA_ERROR_OUT_OF_MEMORY` line, no parked OOM, no 503, no crash line on either arm.
- **V3 typed refusals.** Every refused request is a 429 with `Retry-After` in 1 to 60 and its `reject-kv` line.
- **V4 identity.** Every request admitted on both arms has the same completion digest on both.

Readings, no bound: the burst's 200 and 429 counts per arm; `vg_debt` at each admission and the pool's reserved bytes
at the end; the admitted requests' TTFT p50 and p95.

### 1.4 Price

Code: about 0.3 agent-day (the verdict's budget term, its line field, a unit test, a census test). Cells: about 1 h on
the 5090, about 1.5 h on the target card plus the 20 GB staging.

### 1.5 Addendum A (2026-09-27, the implementation and the cell as they will be built, after DAY46 read, before any code)

DAY46 has read (2.2), so the day leaves text only. No clause or bound changes; this names what 1.1 and 1.2 left open.

- **The door (`MEMRA_ADMIT_PREDICT_VG_DEBT=1`)** is read in one place, `admit_predict_vg_debt_on()`. When it is set, the
  predictive seam reads `dspark_vg_admission_debt` for the request's model, the same function the physical side calls
  later in the same admission. Nothing between the two seams grows the pool: no device work runs before prefill. The
  verdict then compares `request_kv_hat + booked` against `budget_bytes - vg_debt` (saturating), and the
  `[admit-predict]` line gains a trailing ` vg_debt=<bytes>`. Unset: nothing is read, and the verdict and the line are
  byte for byte today's. The physical side is unchanged.
- **The cell:** `day48-client.py` is `day46-client.py` (addendum C's trigger) with one change: the burst and second-wave
  requests cycle through eight prompt lengths, L - 256 k for k = 0 to 7. That makes the verify-graph pool meet new
  `(segment start, vt)` keys during the burst (1.2's "varied lengths"). `day48-run.sh` runs the arms `enforce` and
  `enforce-vg` through run-day26-cell.sh on the default (spec) route. `day48-read.py` reads:
  - V1: the boot's `[spec-vg] MTP verify-graph pool ENGAGED` line and the physical `[admission] dspark verify-graph pool
    debt: +<MB>` lines, and on `enforce-vg` at least one `vg_debt=` above 0.
  - V2 to V4 as DAY46's P2, P1 and P4.
  - The readings as registered.
- **Model and shapes:** Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp.gguf (20 GB; its sha256 goes into the receipts before the
  first boot), served as `o15`. The 5090: B = 32, L = 6,144 at `MEMRA_CTX=65536`. The target card: B = 64, L = 30,720 at
  the checkpoint's context. Both orders, 4 boots per card.

### 1.6 Addendum B (2026-09-27, revuto on integ72, before the fix)

The door's read is not read-only. `dspark_vg_admission_debt` calls `DsparkVerifyGraphs::admission_debt`, which records
the pool's `(captures, reserved)` observation whenever captures have grown past the last one. The physical gate calls
the same function later in the same admission. With the door on, the predictive read takes the marginal branch of the
projection and records the observation, so the physical read falls to the bootstrap branch: up to one more pool's worth
of reserve on an export whose pool does not grow per key. So the door can change the physical reserve (its FLAGS row
says it does not) and can print a `vg_debt=` that differs from the physical `pool debt` line. The fix gives each
admission one debt:
- The engine gains a non-mutating peek (`admission_debt_peek`: the same projection, no observation recorded) and the
  model's `dspark_vg_admission_debt_peek`. The predictive seam reads the peek. The physical call is unchanged and
  records the observation as today, so it returns what the peek returned.
- A test drives the pool's debt state across admissions where captures grow between them. On one path the physical
  read runs alone (the door off); on the other the peek runs, then the physical read (the door on). The physical
  values must be equal on both paths, and the peek must equal the physical value on the second path.
- **The eighteenth sitting's reading (2.1) is checked against its logs:** every physical `pool debt` line reads `+34MB`
  on all four boots, both arms, and every predictive `vg_debt=` above 0 reads 33,554,432 bytes (the same 32 MiB), so
  the double read changed no reserve in that run (the pool reached its high-water before the first measured
  admissions). 2.1 stands as it read. The cell reruns on the fix to confirm, with a pool that grows during the burst.

### 1.7 Addendum C (2026-09-27, the rerun on the fix, before it runs)

- **The binary:** the lane's crates at `521fdbbbc` (the peek). The cell, arms and shapes are 1.2's and addendum A's.
- **V5, one debt per admission:** on every `enforce-vg` boot, each `[admit-predict]` line with `vg_debt` above 0 is
  followed, for the same admission, by the physical `[admission] dspark verify-graph pool debt: +<MB>` line. Its MB
  equals the predictive `vg_debt` rounded to MB, and no admission shows the two apart.
- **V6 (a reading):** the physical debt lines' values per boot, on both arms, and how many distinct values the pool
  took during the burst.
- The target card as the nineteenth sitting (`pro-single-b-sitting19.sh`, S48 = `521fdbbbc`, receipts `b-day48b`);
  V1 to V4 as registered.

### 1.8 Addendum D (2026-09-27, the runner, after the nineteenth sitting did not run, before its rerun)

The nineteenth sitting did not run. Its run.log reads `boot O1-enforce: waiting for an idle rig` at 10:28:26Z, then
`boot O1-enforce: rig not idle after 7200 s; not run` and the chain's `boots stopped rc=3` at 12:28:27Z. Another lane's
load took the box lock per run, back to back. day48-run.sh polled for an idle rig (`flock -n`, then no compute app) and
never found the lock free, because the load re-took it at once. The not-run receipts are banked on the box as
`b-day48b-notrun-idlewait-1229` and the lead mirrors them as a not-run record. The runner changes; the cell does not:
- day48-run.sh takes the rig lock first, blocking, bounded by the boot's 7200 s deadline. Under the hold it checks the
  rig idle on 1.2's conditions (no compute app, at least 24 GB host memory available), then boots through
  run-day26-cell.sh with `LOCK=none` (the cell's "held by the collector" path). The cell runs with the hold's fd closed,
  so the server never inherits the lock. The hold is released after the cell, before the yield.
- A rig still busy under the hold after 120 s (a process that does not take the lock) releases the hold and retries
  after 30 s, within the same deadline. Every hold, busy reading, release and not-run is a run.log line, and each
  boot's `.arm.txt` records the hold.
- The helper is a new file, `rig-hold.sh`. `test-rig-hold.sh` checks it against a fixture lock and a stub
  `nvidia-smi`: a lock taker that re-takes the lock back to back (the hold gets in; the old poll's hits are a reading),
  a compute app that clears under the hold, and one that does not (the hold is released and the deadline ends it).
- The binary, the cell, the arms, the shapes and V1 to V6 are addendum C's. The rerun is the same command on the lane
  tip. Closed runners keep the poll they ran with; their receipts record it.

## 2. Results

Written after the runs. Section 1 is unchanged.

### 2.1 The target card (the eighteenth sitting, one RTX PRO 6000 Blackwell Workstation Edition at 600 W, 2026-09-27 05:41 to 05:52Z)

Chain tree `5ceab1a68`; the binary built on the box from `5a6f1898f`, sha256 `dc3e6327...5a19ea367` (the lead mirrored it
by hash). Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp.gguf from the tiyuvta repository at `e058c9f5b`, sha256 `72ff9600...` checked
on the box. Receipts at `pro-single-day48/box/` (the sitting's own `MANIFEST.sha256`, the lead's re-checked). Every boot
`rc=0`. Verbatim (`read.log`; the readings of both orders alike, O1 shown):

```
DAY48 V1 card=pro6000 boot=O1-enforce pool_engaged_lines=1 physical_debt_lines=40 physical_debt_mb_max=34 vg_debt_lines=0 vg_debt_max=0 -> PASS
DAY48 V2 card=pro6000 boot=O1-enforce oom_lines=0 parked_oom_lines=0 crash_lines=0 r503=0 -> PASS
DAY48 V3 card=pro6000 boot=O1-enforce r429=60 without_reject_line=[] retry_after_out_of_1_60=[] other_non200=[] -> PASS
DAY48 READING card=pro6000 boot=O1-enforce wave=burst n=64 ok200=35 r429=29 ttft_ms p50=107928.0 p95=107962.6 N=35
DAY48 READING card=pro6000 boot=O1-enforce wave=wave2 n=32 ok200=1 r429=31 ttft_ms p50=4859.7 p95=4859.7 N=1
DAY48 V1 card=pro6000 boot=O1-enforce-vg pool_engaged_lines=1 physical_debt_lines=41 physical_debt_mb_max=34 vg_debt_lines=107 vg_debt_max=33554432 -> PASS
DAY48 V2 card=pro6000 boot=O1-enforce-vg oom_lines=0 parked_oom_lines=0 crash_lines=0 r503=0 -> PASS
DAY48 V3 card=pro6000 boot=O1-enforce-vg r429=60 without_reject_line=[] retry_after_out_of_1_60=[] other_non200=[] -> PASS
DAY48 READING card=pro6000 boot=O1-enforce-vg wave=burst n=64 ok200=35 r429=29 ttft_ms p50=108644.8 p95=109066.7 N=35
DAY48 READING card=pro6000 boot=O1-enforce-vg wave=wave2 n=32 ok200=1 r429=31 ttft_ms p50=4867.7 p95=4867.7 N=1
DAY48 V4 card=pro6000 order=O1 rows_200_both=45 status_mismatch=4 differ=[] -> PASS
DAY48 V4 card=pro6000 order=O2 rows_200_both=43 status_mismatch=8 differ=[] -> PASS
```

- **V1 to V4 PASS on all four boots, both orders.**
  - V1: the verify-graph pool engaged on every boot, and the physical side charged its debt on 40 to 43 admissions.
    On `enforce-vg`, every predictive line carries `vg_debt=` (107 lines).
  - V2: no OOM, parked OOM, 503 or crash.
  - V3: all 60 refusals are typed, each with its own reject line.
  - V4: every request that is `200` on both arms has an equal digest.
- **What the door changes on this model and card: nothing measurable.** The pool's remaining debt is at most 34 MB
  (33,554,432 bytes on the predictive line), against a budget of tens of GB. Both arms admit 35 of 64 of the burst and
  1 of 32 of the second wave, 47 of 107 over the boot. The 4 and 8 requests whose status differs between the arms of
  an order swap places inside the same counts (which burst requests win a concurrent release); neither arm admits
  more. The DAY28 gap is real but small here: the pool reaches its high-water early in the boot.
- The door stays default off; the decision is the owner's at its decide-by (2026-10-11). The 5090 half runs from
  `rtx5090-day48/run.sh`.

### 2.2 The target card (the nineteenth sitting, addendum C on the fix, one RTX PRO 6000 Blackwell Workstation Edition at 600 W, 2026-09-27 12:44 to 12:56Z)

Chain tree `b094f619d` (addendum D's runner); the binary built on the box from `521fdbbbc`, sha256 `a6e893d2...d63d88d477`
(mirrored by hash). The artifact is 2.1's (sha256 `72ff9600...` checked on the box). Receipts at `pro-single-day48/box-b/`
(the sitting's own `MANIFEST.sha256`, the lead's re-checked); the first attempt's not-run receipts at
`pro-single-day48/box-b-notrun/` (addendum D). Every boot `rc=0`. The runner held the box lock after 9 to 12 s per boot
while another lane's load ran between the boots, and found the card idle under each hold (`run.log`). Verbatim
(`read.log`; the V1 to V3 lines and the readings of O2 read alike, O1 shown):

```
DAY48 V1 card=pro6000 boot=O1-enforce pool_engaged_lines=1 physical_debt_lines=40 physical_debt_mb_max=34 vg_debt_lines=0 vg_debt_max=0 -> PASS
DAY48 V2 card=pro6000 boot=O1-enforce oom_lines=0 parked_oom_lines=0 crash_lines=0 r503=0 -> PASS
DAY48 V3 card=pro6000 boot=O1-enforce r429=60 without_reject_line=[] retry_after_out_of_1_60=[] other_non200=[] -> PASS
DAY48 READING card=pro6000 boot=O1-enforce wave=burst n=64 ok200=35 r429=29 ttft_ms p50=108272.8 p95=108306.2 N=35
DAY48 READING card=pro6000 boot=O1-enforce wave=wave2 n=32 ok200=1 r429=31 ttft_ms p50=4864.7 p95=4864.7 N=1
DAY48 V1 card=pro6000 boot=O1-enforce-vg pool_engaged_lines=1 physical_debt_lines=38 physical_debt_mb_max=34 vg_debt_lines=107 vg_debt_max=33554432 -> PASS
DAY48 V2 card=pro6000 boot=O1-enforce-vg oom_lines=0 parked_oom_lines=0 crash_lines=0 r503=0 -> PASS
DAY48 V3 card=pro6000 boot=O1-enforce-vg r429=60 without_reject_line=[] retry_after_out_of_1_60=[] other_non200=[] -> PASS
DAY48 READING card=pro6000 boot=O1-enforce-vg wave=burst n=64 ok200=35 r429=29 ttft_ms p50=106902.5 p95=108624.5 N=35
DAY48 READING card=pro6000 boot=O1-enforce-vg wave=wave2 n=32 ok200=1 r429=31 ttft_ms p50=5028.1 p95=5028.1 N=1
DAY48 V6 READING card=pro6000 boot=O1-enforce physical_debt_lines=40 distinct_mb=[34]
DAY48 V6 READING card=pro6000 boot=O1-enforce-vg physical_debt_lines=38 distinct_mb=[34]
DAY48 V5 card=pro6000 boot=O1-enforce-vg paired=38 apart=[] predictive_without_physical=67 -> PASS
DAY48 V6 READING card=pro6000 boot=O2-enforce physical_debt_lines=38 distinct_mb=[34]
DAY48 V6 READING card=pro6000 boot=O2-enforce-vg physical_debt_lines=41 distinct_mb=[34]
DAY48 V5 card=pro6000 boot=O2-enforce-vg paired=41 apart=[] predictive_without_physical=64 -> PASS
DAY48 V4 card=pro6000 order=O1 rows_200_both=45 status_mismatch=4 differ=[] -> PASS
DAY48 V4 card=pro6000 order=O2 rows_200_both=45 status_mismatch=4 differ=[] -> PASS
```

- **V1 to V4 PASS on all four boots, both orders**, as in 2.1: the pool engaged on every boot; no OOM, parked OOM, 503
  or crash; all 60 refusals typed; every request `200` on both arms of an order has an equal digest (45 per order).
- **V5 by the reader: PASS.** Every admission that shows both lines shows one debt: 38 and 41 pairs, none apart
  (33,554,432 bytes on the predictive line, `+34MB` on the physical line).
- **V5 by its clause (addendum C): not met, and recorded as it reads.** The clause says each `[admit-predict]` line with
  `vg_debt` above 0 is followed, for the same admission, by the physical line. On 67 and 64 such lines it is not.
  `day48-v5-place.py` (output `pro-single-day48/box-b-v5-place.log`) places them:
  - 60 on each boot are `reject-kv` verdicts. A refused request never reaches the physical gate, so it has no
    physical line.
  - 7 and 4 are admissions with no physical line. The server prints the physical debt line only under `log_estimate`,
    which is false when the admission's (cap, spec, cost) key equals the last logged one. None of the 11 has a
    `request cost` line in its window either, the other line `log_estimate` gates.
  - So those 11 admissions' physical debts are unobserved, not apart. The reader's PASS computes only the equality
    part of the clause. The clause stays as registered: V5 FAILs on its coverage, beside the reader's PASS. A
    revision of the clause after this result is the owner's.
- **V6: the pool did not grow during the burst.** Every physical debt line on every boot is `+34MB`
  (`distinct_mb=[34]`): the pool was at its projected high-water from the first measured admission, as in 2.1. So the
  box did not exercise the case the defect needs (captures growing between the two reads). That case is covered only
  by the engine test `the_peek_leaves_the_physical_debt_unchanged_across_a_growing_pool` (addendum B).
- **Readings, as in 2.1:** both arms admit 35 of 64 of the burst and 1 of 32 of the second wave (47 of 107 over the
  boot); the 4 requests per order whose status differs swap places inside the same counts. Burst TTFT p50 106.9 to
  108.9 s (N = 35 per boot; one boot per arm per order, back to back on one card at 600 W). The door changes nothing
  measurable on this model and card; 2.1 stands.
- What V5's coverage needs to be observed as registered is a physical debt line on every admission, not only when the
  cost key changes. That is a server log change and a sitting, registered as its own addendum before its code.
