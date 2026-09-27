Author's review of the full diff `main..lane/spill-integ72-20260927`, posted as a PR comment per the owner rule.

## What the diff is
- `admit_predict.rs`, `worker.rs` (lane B, DAY48): `MEMRA_ADMIT_PREDICT_VG_DEBT`, default off. When set, the
  predictive admission verdict subtracts the verify-graph pool debt the physical side already reserves for the request.
- Lane B's 209 HTTP header captures re-stored with their captured CRLF bytes, with a record of each file's hashes.
- Records: lane B's DAY39 to DAY50 readings and addenda (DAY44 addendum D pending the owner), lane C's DAY85 to DAY87 and
  the door decision packet, lane A's STATE.

## What I checked
- With the door unset, the predictive verdict and its log line are the ones main computes: the new term is read only
  behind the flag, and the FLAGS row carries its default and decide-by date (2026-10-11).
- The CRLF restore changes bytes only in `.hdr` files the record lists; one restored capture checked against its box
  manifest in this tree (`\r\n` line ends, `text: unset`).
- The lane's `.gitattributes` conflict resolution kept main's superset file.
- The fixture pin holds; no provider name, host, id, price or city in the added lines.

## Batteries
- CPU battery 16 of 16 with CI's gates job on the merged head.
- GPU battery on an RTX PRO 6000 (9950X host), running at PR open: integ70's cells, the pause and tier gates, and the
  admit-mem burst gate with the new door on. Its results land as a follow-up commit.

**Hygiene:** no em dash in authored lines.

## Push regime
Server source changed, so the branch goes up with `MEMRA_RELEASE_QUALIFICATION_MODE=development`. No tag. Revuto: if
capped or unavailable, this comment is the review.

## Round 2, after revuto round 1
Revuto found a real defect in the DAY48 door that my review missed: I checked the door is inert when off, not that its
on path calls a function with side effects. `dspark_vg_admission_debt` records a pool observation, so with the door on
the predictive call changed the physical gate's reserve. Lane B's fix (`521fdbbbc`) reads a non-recording peek at the
predictive seam; the physical call is unchanged. What I checked: the peek runs the same projection from the recorded
observation, the physical call still records exactly once per admission, and the engine test fails on the old double
read. Revuto round 2 approved it. Lane B's DAY48 rerun on the fix passed V1 to V5 on the target card (V5 apart=[] on both
enforce-vg boots); its first attempt did not run because I launched it into lane C's load window, and lane B's runner now
holds the lock before its idle check. GPU run 2 on the final code is all green (Q35 is #777); CPU battery 16 of 16.
