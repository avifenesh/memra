# Self-review: integ22 (B day 20, the LRU decision confirmed on the RTX 5090; lockstep INDEX row restored)

Author's review of the full diff `main..lane/spill-integ22-20260921`, posted as a PR comment per the owner rule
(no human review; revuto is the other half when it runs).

## What the diff is
- `docs/decisions/PREFIX-CACHE-POLICY.md`: the day-16 "did not confirm" paragraph replaced by the day-20 result;
  the Status sentence and the follow-up bullet no longer say the 5090 cell is owed. Read in full: the verdict line
  matches `rtx5090-day20/ab-full-retry1/cell/VERDICT.txt` byte for byte; the `MEMRA_REUSE_POOL=0` deviation and its
  reason are stated in the doc, not only in the lane record; no cross-box timing.
- `research/spill-b-20260919/`: `DAY20.md` (pre-registration committed before the runs, results 1 to 3, checks
  table, boundaries), `STATE.md`, `build-day20.sh`, `run-day20-ab.sh`, `day20-predict.py`, `verify-day20.py`,
  receipts `rtx5090-day20/` (three cells with 250 ms telemetry, build receipt, fmt and census logs, push log).
- `research/INDEX.md`: B's day-20 row, plus the restored `lockstep-cpu-rows-exact-20260921` row (see below).
- `research/spill-lead-20260919/`: the day-12 record's integ22 section and ruling 24, this file, the battery
  summary and logs.

## What I checked
- No engine source in the range (`git diff --name-only main HEAD | grep crates` is empty), so the push is a plain
  push under the #589 gate; nothing is qualified by this PR and no cell claims to be.
- The rule did not move: `run-day20-ab.sh` and `verify-day20.py` add no threshold, tolerance or allow switch; the
  verdict clause is the day-16 clause. The shape change (tokens onto the 32-token grid) is derived from the same
  byte shares and is required by the harness's own seed assertions under the capture law.
- The deviation (`MEMRA_REUSE_POOL=0`) is pre-registered, sized from the day-16 receipts, and controlled by the
  default-pool smoke on the same binary (same rows, same digests, ladder firing). It is a residency change on a
  24 GB card, not a bytes change, and it is stated where the decision is recorded.
- Replays: `verify-day20.py` on all three cells reproduces the rule on the primary (`WINNER=lru`) and the
  predicted `cached_tokens` per request (280/280 per arm on the scored cell, 56/56 on the smokes).
- Lock name in the scripts: `/tmp/memra-5090.lock` only. Public boundary `check`: 0 new matches. No em dashes in
  the added prose. No provider host, id or price in any added tracked file.
- INDEX.md carries no `<<<<<<<`, `=======`, `>>>>>>>` or `|||||||` line; a set difference against #604's INDEX
  (minus its marker) shows the lockstep row was the only line missing on main, and it is back verbatim.

## What I did not do
- No GPU cell of my own; the receipts are B's, replayed here on CPU.
- #523 stays open (its other items).
