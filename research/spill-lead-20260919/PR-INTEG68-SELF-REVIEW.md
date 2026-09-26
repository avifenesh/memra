# Self-review: integ68 (F's items 17 and 18 as default-off doors; C days 75 to 77)

Author's review of the full diff `main..lane/spill-integ68-20260926`, posted as a PR comment per the owner rule.

## What the diff is
- F item 17: `MEMRA_MOE_COLD_BYPASS=staged|mapped` (default off), a cold first miss served once without admission;
  the MoE dispatch goes through `dispatch_source_once`, `payload`, `consumed`.
- F item 18: `MEMRA_KV_HOST_HANDOFF_IO=direct` (default buffered), the handoff file layer `handoff_io.rs`.
- C: the I16 revert, I17 (grouped owner calls), the chunk diagnostic flag; records.
- Integration commits: F's clippy fixes, the moe_cache.rs source re-pin, a location scrub of lane C's records.

## What I checked
- The naked MoE dispatch: with the flag off the bypass branch cannot run (`bypass_allowed` and the mode both guard it),
  `payload` returns the slot's buffer and `0..len`, `consumed` is a no-op, so the kernels read what they read before.
- The handoff `buffered` arm is the historical program (4 MiB BufWriter and BufReader, `sync_all`); the two arms'
  files are byte-identical in the cross-read test.
- The re-pin: the merged moe_cache.rs changes no SLRU statement; the 2,013-row replay passes on the new pin.
- The Q35 serve-smoke failure is main's (same box, main's tree reads the same), filed as #777, not caused here.

## Batteries
- CPU battery 15 of 15 on the fixed head (the first pass's 4 reds were exactly the clippy and pin failures fixed here).
- GPU battery on a rented RTX PRO 6000: every cell green but serve-smoke's Q35 arm (#777, pre-existing on main).

**Hygiene:** no provider name, host, id, price or city in tracked files; the lane C records that carried one are
scrubbed here. No em dash in authored lines.

## Push regime
Engine and server source changed, so the branch goes up with `MEMRA_RELEASE_QUALIFICATION_MODE=development`. No tag.
Revuto: if capped or unavailable, this comment is the review.
