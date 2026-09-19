# Generic-spill pre-work — day-2 integration record (lead: agent-c07799, 2026-09-19)

Branch `lane/spill-integ-20260919`, tip after this record's commit (see `git log -1`).
Base `c5a33b14` (main). Merged, in order: `lane/spill-lead-20260919` (259ff819, frozen contracts)
→ A `8fb1738d` → B `bb431f33` → C `2cc1b530` → D `fbcb3d03`. Conflicts: only additive
`crates/memra-tier/src/lib.rs` `pub mod` lines and `Cargo.toml` `[[test]]` entries (union-resolved;
one resolution error — B's `pub mod tier;` dropped by a scripted union — caught by inspection and
restored before the C merge commit). `cargo fmt` then re-sorted the module list.

## Integrated CPU battery on the merged tip (this Mac; no nvcc, no GPU)

| Check | Result |
|---|---|
| `cargo check -p memra-tier -p memra-kv --offline --all-targets` | OK (pre-existing macOS `AsRawFd` unused-import warning in memra-gguf) |
| `cargo test -p memra-tier --offline` | contracts 36 · storage 26 · bank 27 · peer 6 · placement (see lane receipts) · doctests 4 — all pass, 0 ignored |
| `cargo test -p memra-kv --offline` | 40 pass |
| `cargo clippy -p memra-tier -p memra-kv --all-targets --no-deps -- -D warnings` | OK |
| `cargo fmt --all -- --check`, `git diff --check` | OK |
| `bash tools/check-flags.sh` | 864 runtime literal reads, none uncovered; no new `MEMRA_*` reads in any lane |
| `python3 -m py_compile tools/tier-*.py` | OK |

NOT run (stated, not implied): any engine/server build (needs nvcc), any GPU cell, any model load.
B's six-line `crates/memra-server/src/worker.rs` field (`_tier_identity: IdentitySlot`, inert,
`Default`) is source-inspected only — first item for the first rig build.

## Freeze-revision backlog (from lane reports; schedules stay lead-owned)
1. No reusable `PinnedLease` conformance schedule (A) — add.
2. Governor schedules are bound to `support::Governor`, not a generic `BudgetGovernor` schedule (B) — generalize.
3. No reusable `PeerCapacity` schedule; `peer_cancel` covers pre-completion cancel only (D) — add cancel-after-completion + generic capacity schedule; see `research/spill-d-20260919/CONFORMANCE-REVIEW.md` for the full gap list.
4. Sync `ObjectStore` only; thin async wrapper deferred (WP-A follow-up).

## Owner decisions in force
DSv4 Flash 0731 stays PAUSED and is not used by any lane. Target box: 4× RTX PRO 6000 (PCIe Gen5,
no NVLink). Generic spill/tiering before any V4.1 model code; go/no-go G0–G7 per
`~/tiyuvta/dsv41f-serving-viz/08-pre-work-generic-spill-plan.md` §E — all still pending
(no GPU evidence exists yet).
