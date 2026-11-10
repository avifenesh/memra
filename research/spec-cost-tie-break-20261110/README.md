# The spec cost model's tie between equally ranked slots (memra#1212)

Date: 2026-11-10. CPU only: a unit test and a reading of three boot calibration tables.

**Verdict: fixed.** `SpecCostEntry::priced` scales a prediction by the realized/predicted ratio
of one other slot when the slot it prices has no samples of its own. Among equally ranked slots
it took the first in `HashMap` iteration order, which is random per process. The same state
could then price, and choose, differently on two boots. Ties now break on the slots: the longer
history (more samples), then at a far concurrency a neighbor that is far itself, then the draft
length nearer the one priced, then the lower concurrency, then the shorter draft length. The
verdict log names where each draft length it considered was priced from, through the same
resolver the price uses (`priced_from`).

## Red and green

`spec_cost_tied_slots_price_the_same_on_every_build` (`crates/memra-server/src/worker.rs`)
builds the same entry 64 times, each map with its own hash keys. Two slots that tie in rank and
carry ratios 1.0 and 0.5 price (2, 3) at 23.0 or 11.5. With the old key the 64 builds gave both
prices (`left: [11.5, 23.0]`, `raw/red.log`). With the rule every build gives 23.0
(`raw/green.log`). The test also covers a tie between two draft lengths at the same
concurrency, a longer history beating a lower concurrency, plain neighbors, both sides of the
far-concurrency edge, and the logged source of each price.

## Where a tie can change a choice, per cost-model-owned case

Approximately, a tie decides only when the two tied slots' ratios fall on opposite sides of

    t(n) = plain(n) / best spec(n)

below which a scaled speculative price loses to plain. The approximation assumes plain is priced
at its table value (it can be realized, or scaled from a plain neighbor), that one slot scales
every draft length (each draft length picks its own), and no tie-probe margin (an unmeasured
spec price may still win within two standard errors below plain). `tools/tie_thresholds.py` reads a boot
calibration line and prints the table's predictions and t(n) for n = 1..8, with the server's own
formulas. The acceptance per draft position is the one the server's cost-model tests use
(0.7, 0.5, 0.35); t(n) moves with it. Output: `thresholds.md`. Inputs: `tables/`, one line each
from a GB10 boot of v0.141.0 (memra#1201's sweep), `research/cap-floor-gate-20261106/` (Ornith)
and `research/spec-k-short-5090-20261110/` (the 9B).

| n | Qwen3.8-27B, GB10 | Ornith-1.5-35B-A3B, RTX 5090 | Qwen3.5-9B, RTX 5090 (outside the roster) |
|---:|---:|---:|---:|
| 1 | 0.52 | 0.91 | 0.55 |
| 2 | 0.58 | 1.37 | 0.77 |
| 3 | 0.61 | 1.69 | 0.88 |
| 4 | 0.58 | 1.90 | 0.94 |
| 5 | 0.67 | 2.10 | 0.99 |
| 6 | 0.83 | 2.25 | 1.08 |
| 7 | 0.86 | 2.37 | 1.10 |
| 8 | 0.90 | 2.47 | 1.10 |

- **Qwen3.8-27B on GB10.** The realized rates on record, from memra#1201's normal-concurrency
  cell, against that boot's own table at the same acceptance: 45.1 tok/s at n=2 against 43.3
  (ratio 1.04) and 62.9 at n=4 against 79.9 (0.79), at K=3. Both the ratio and t(n) scale as one
  over the expected tokens per round, so the comparison holds whatever the live acceptance was.
  n=3 sits between them, and both exceed t(3) = 0.61, so both tie outcomes serve K=3 there; the
  rule changes nothing in that state. A flip needs the neighbors to straddle t(n): for example at n=5 (t = 0.67) a
  neighbor at 0.75 against one below 0.67. At n >= 7 only neighbors within one session count.
- **Ornith-1.5-35B-A3B on the RTX 5090.** The model does not pool, so its speculative price is
  flat in n (117.0 tok/s at K=3), and t(n) is 1.37 and above from n=2. A neighbor would have to
  realize 37% above its prediction for speculation to win, so between neighbors plain wins either
  way. At n=1 there is only one neighbor at distance one, so the ties left are between two other
  draft lengths at n=1, or between two other draft lengths at n=2 when (2, K) is unmeasured,
  around t(1) = 0.91.
- **Qwen3.5-9B on the RTX 5090** (a cost-model-owned case outside the roster). The most exposed: from n=3 to n=5, t(n) is 0.88 to 0.99,
  so two neighbors a few percent apart around their predictions can choose differently. No
  realized ratio above n=1 is on record for it.

Whether a tie has flipped a served choice was not measured: before this change the verdict log
did not name the slot that priced it. It does now (`priced from K0=table K1=(1,3) ...`).
