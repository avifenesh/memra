# integ9 self-review (lead, 2026-09-21)

Read in full: `kv_tier_gate/fault.rs` (582 lines), `kv_tier_gate/fault_contract.rs` (383), the `kv_tier_gate.rs` and
`cli.rs` diffs, `active.rs` visibility changes, `memra-tier/tests/reclaim/fault.rs`. Receipts spot-checked against
`verify-day11.py`.

## Findings
1. **Serving path untouched.** Everything is inside the gate binary and the tier test crate; `active.rs` changes are
   `pub(super)` visibility for reuse by `fault.rs` plus whole-state admission before any layer is taken and
   `StateBundle::verify` on restore (a stricter roundtrip, not a looser one).
2. **CLI fails closed.** `--fault` needs a value from the closed name list, refuses a duplicate, and is a usage error
   without `--case active --same-program`; unknown names print `REFUSED: unknown fault arm (expected ...)`.
3. **Arms are one injection each, at a named contract call**, and the verdict comes from `Pending::check` rows
   (expected versus observed, written as TSV): `finish` prints `FAULT-ARM PASS <arm>` only when every check holds, a
   typed `REFUSED:` for the two arms whose contract seam does not exist (`cancel-restore`, `require-resident`), and a
   plain failure line otherwise; unit tests cover all three shapes for every arm. A refusal never counts as PASS.
4. **Continuing arms prove the state**: `restores_cache()` arms capture the restored prefix and check it against the
   suspended hash before any decode; the non-continuing arms (`corrupt-host`, `missing-host`, `cancel-restore`)
   print `generated=0` and return before decode.
5. **Findings are honest, not patched.** The two refusals document real gaps (no H2D-source recovery after cancel;
   no required-resident contract at continuation) instead of inventing seams; lead ruling 9 routes them to lane A as
   contract rules.
6. **Holed cache is tainted (revuto finding, fixed in this PR).** The three arms that drop an incomplete layer left
   `cache.kv[i] == None` with a decode path that would `unwrap` it; only the gate's early return stood in the way.
   The drop site now calls `cache.mark_tainted()` (the state `ensure_usable` already refuses) and adds the check
   `holed-cache-refuses-continuation = Err`, so the cache itself records that no token may address it. Extra check
   row, not a required one. Revuto's second round caught that both replays (`tests/reclaim/fault.rs` and
   `verify-day11.py`) rejected any row outside the required set, which would have reddened D's day-12 reruns; both
   now accept extra evidence rows (every row must hold, the required names must be present). 30 Rust tests and the
   day-11 Python replay green on the committed receipts.
7. **Nits (not blocking).** `Arm::NAMES` duplicates the `name()` list as a string; a `join` over `ALL` would keep them
   in sync. The `device-short` arm exercises exhaustion through a second tenant because the governor has no
   post-construction capacity seam; the arm name promises less than a pool shrink and the receipt says so.

## Verification this review relied on
integ9 CPU battery (`integration-day12/integ9-cpu-battery/`), all rc=0; D's BOX3 build receipt (clippy clean, 295
tests, bound binary hash) and the seven cells with `--validate`; `verify-day11.py` replay. This rig cannot run the
model gates; the GPU evidence is the target-card cells.
