# Self-review: integ49 (A day 31: the staging back on every post-take refusal, the staging charged, the span-refusal fault cell, the per-slot-class tally)

Author's review of the full diff `main..lane/spill-integ49-20260923`, posted as a PR comment per the owner rule.

## What the diff is
- Engine, under the default-OFF door `MEMRA_KV_HOST_CONTRACTS`: `crates/memra-server/src/worker.rs` (+765/-36 net
  against main, the day-31 code commit `33b1285e0` and its censuses and GPU cells). `HostStaging` (idle buffers, one
  `ResidentCharge` per allocated buffer, the charged bytes, the latch flag), `staging_take` charging before it
  allocates, `staging_put` freeing after the latch, the `StagedSpans` guard in the settle, `unstage` on the driver's two
  early exits, `HostContractFault::SpanAttach`, and the per-slot-class tally on the copy-complete line.
- `docs/FLAGS.md`: `contract-spans` appended to the existing `MEMRA_KV_HOST_FAULT` row. No new `MEMRA_*` name.
- `tools/kv-host-contract-fault-gate.sh`: the `span-refusal` cell (+114), two boots, door ON with the fault and door
  OFF as the byte reference.
- Research: A's DAY31.md, STATE.md, OWNER-THREAD-OFFLOAD.md, HOSTPREFIX-DOOR section B rows, the INDEX row, the
  target-card receipts `pro-single-day31/box/` and the 5090 receipts `rtx5090-day31/`; the integ49 record section
  (ruling 44), the CPU battery and the 5090 battery on the merged tree, and this file.

## What I checked
- Every exit of `host_kv_planes_settle_contract` after the take sits inside the guard's scope (A's census counts 11
  `return Err(`; I read them). The one success exit calls `landed()`. The driver's `!host.armed()` and flip-demote
  exits call `unstage`. The helper's reply path puts the staging back; the reply-mismatch, helper-gone and
  latched-while-hashing exits latch the tier or find it latched, so their buffers free with their holders.
- `staging_take`: refuses on a latched set before any charge; reuses an exact-length idle buffer without a new charge;
  a refused charge allocates nothing; a failed allocation drops its local charge. The governor guard in the dimension
  read is a statement temporary, so `ResidentCharge::reserve` does not re-enter a held lock. `ResidentCharge::drop`
  takes the governor lock; the only production sites that hold that lock (the dimension reads and the D2H admission
  probe) release it before anything that can reach `disable`, so the latch's `clear()` cannot deadlock.
- `clear()` frees the idle buffers before the charges release. A's stated limit stands: after the latch, a buffer on
  a quarantined ticket or with a detached helper outlives its released charge until its holder drops. It applies to
  a latched tier only.
- The fault is one-shot and demote-side (`is_demote` covers `SpanAttach`, unit-tested). The `span-refusal` cell
  byte-compares r1 to r4 against a door-OFF boot; A ran its helpers red and green against a synthetic log.
- One numeric program per request: nothing here changes a hashed, published or promoted byte. The cell's r1 to r4 are
  byte-equal to door OFF on both cards.
- Merge: A's tip `979881aa0` on main `9c07b398b` (`worker.rs` auto-merged beside #655), then main again at
  `f69119ae0` (#556 and the dsv4 PRs #661 to #663; `worker.rs` auto-merged), giving `d1760ac21`. Neither merge
  touches the door's code, so the 5090 battery on the merged binary is the merge check (the integ47 precedent);
  BOX3 is not rerun.
- CPU battery on `d1760ac21`: 15 of 15 rc=0 (fmt, portable suites 360 passed, server 874 passed, engine lib 542
  passed, tier 4, clippy `-D warnings` twice, check-flags, publish census, docs registry, pytest 87, conflict
  markers, workflow keys, perf board, `git diff --check`).
- RTX 5090 on `d1760ac21` (binary `09bbd855`, hashed after serve-smoke's build; one collector hold, 04:10Z to
  04:18Z): serve-smoke `0 failed`; engine `d2d_` and `d2h_span` cells 6 passed; worker `option_b_` and `option_c_`
  cells 13 passed; identity default ON ALL GREEN (12 ok); fault default and plain ALL GREEN (142 ok each,
  `refusal handed back 48 span(s)`, `byte-unequal request(s): none`); hit OFF and ON ALL GREEN (61 and 68 ok).
  The first run on the pre-main merge read the same lines. Its `binary.sha256` names the pre-smoke binary: serve-smoke
  rebuilds `memra-server`, so its later cells ran `920eebe9` (the same sources). The record states this.
- No new `unsafe`. No em dash in authored lines; the ones in the added lines are verbatim server and tool output in
  receipts. `.gitattributes` in every new receipt dir.

## Push regime
The branch went up with `MEMRA_RELEASE_QUALIFICATION_MODE=development` (announced, logged): the release
qualification gate refuses engine-source ranges on a topic branch as `UNQUALIFIED`, and this branch carries
`worker.rs`. Every other hook ran and passed. No tag: an engine change under a default-OFF door, no board move (integ47
precedent). Revuto: if capped or unavailable, this comment is the review.
