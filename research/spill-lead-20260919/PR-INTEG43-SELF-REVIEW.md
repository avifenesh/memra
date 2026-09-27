# Self-review: integ43 (A day 27: the demote tick cost attributed to the bundle checksum, the three options; C day 33: the 5090 write-combined hash rate, H1 refuted)

Author's review of the full diff `main..lane/spill-integ43-20260922`, posted as a PR comment per the owner rule.

## What the diff is
- `crates/memra-engine/src/bin/hash_micro.rs` (C day 33): an opt-in `--two-step` arm of the diagnostic bin (a memcpy of
  the write-combined buffer into cached pinned memory followed by the hash of the copy, timed beside the single pass,
  both orders, digests compared). No engine path, no runtime flag, the day-18 output byte-unchanged without the flag.
- Research: A DAY27 (section 1's file:line attribution, the three options with the digest micro-cell on both hosts, the
  baseline cell), C DAY33 (the hypotheses, the cell, the verdict), receipts on both cards; `HOSTPREFIX-DOOR.md` section D
  item 6 answered (A's code census plus C's measurement, both kept); `OWNER-THREAD-OFFLOAD.md` owed item 2 restated as
  "the bundle hash off the tick"; INDEX rows; the lead record section (ruling 39); this file; battery receipts.

## What I checked
- No serving path changed: the only Rust is a diagnostic binary's opt-in arm; the battery's fmt, clippy and censuses
  cover it; the default output is pinned by the day-18 receipts A and C re-read.
- The attribution is read from the code with file:line pointers and reconciled with both cards' receipts (74.8 ms
  against a 77.9 ms pass on the target host; 21 to 23 ms against a 37.5 ms per 160 MiB heap pass on the 5090 host), and
  C's independent measurement refutes the write-combined hypothesis on the 5090 without reference to the code.
- The options are pre-registered, measured where measurable without engine code (the digest micro-cell on both hosts,
  byte-identical digests), and only recommended; ruling 39 approves option (a) with its acceptance gate and the receipt
  term unchanged.
- The door doc merge: item 6 edited by both lanes, merged three-way against their common base with both paragraphs kept;
  no duplicated row; markers OK.
- Battery on this tree in the receipts (fmt, portable suites, memra-server suite, cross-target `DOCS_RS=1` clippy,
  censuses, collector pytest, engine CPU lib, tier suite, engine/server/tier clippy `-D warnings`, marker census,
  workflow keys, perf board, diff-check), the local 5090 serve-smoke, the engine `d2d_*` GPU cells, and the hit gate OFF
  and ON (armed) on the 5090 (stated either way).

## Push regime
Engine crate source in the range (a diagnostic bin): pushed with `MEMRA_RELEASE_QUALIFICATION_MODE=development`
(announced, logged). No GPU qualification claimed; every cell executed-not-qualified. Revuto: if capped or unavailable,
this comment is the review.
