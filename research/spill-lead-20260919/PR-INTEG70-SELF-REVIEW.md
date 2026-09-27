Author's review of the full diff `main..lane/spill-integ70-20260927`, posted as a PR comment per the owner rule.

## What the diff is
- `worker.rs` (lane A, P2): the hash helper reserves and refills the demote's payload buffers so the copy does not
  fault on fresh pages, armed only while the copy it replaces would fault.
- `memra-tier` bank and `banked_residency/native.rs` (lane C, I21 and I22): the MoE door's residency and bank reads go
  by catalog position; the SLRU keeps a position view equal to its table at every change.
- `moe_cache.rs`, `spill_pread.rs` (lane F, OWED 26): a demand read waits for a worker buffer instead of falling back
  to mmap.
- Research: the lanes' days and receipts, the integ70 record (ruling 65), the GPU and CPU batteries, this file.
- Lane B is not in it: its lane merges main first (345 commits behind) and goes into integ71.

## What I checked
- P2 is the program its sitting adopted, byte for byte (A's cherry-pick onto main had one conflict: two test
  functions at the top of the test module, both kept). The arming rule reads the job's own page faults, so a hit never
  refills.
- C's position view is checked against the SLRU table after every step of the 250,000-operation randomized trace and
  the day-4 fixture; the positioned adapter equals a twin bank on the hashed path. The day-4 fixture pin holds on the
  merged `moe_cache.rs` (only F edits it, and F re-pinned).
- F's demand wait evicts only a prefetch other than its own block, so a read cannot wait on itself; two new GPU cells
  carry red arms that fail.
- The whitespace fixes touch receipts' `.gitattributes` and two blank lines in F's notes, nothing else.
- No provider name, host, id, price or city in the lanes' new records (grep over every added .md, .txt, .sh, .py).

## Batteries
- CPU battery 15 of 15 on `72fcc4a76` (server lib 957, engine lib 589).
- GPU battery on an RTX PRO 6000 (9950X host): engine cells 26 of 26 (F's seven `spill_pread` cells among them),
  worker cells 19 of 19, every gate ALL GREEN, `tier-transfer-gate` and all seven `kv-tier-gate` fault arms PASS.
  serve-smoke's Q35 arm is main's own #777.

**Hygiene:** no em dash in authored lines.

## Push regime
Engine and server source changed, so the branch goes up with `MEMRA_RELEASE_QUALIFICATION_MODE=development`. No tag.
Revuto: if capped or unavailable, this comment is the review.
