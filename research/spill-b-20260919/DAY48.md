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

## 2. Results

Written after the runs. Section 1 is unchanged.
