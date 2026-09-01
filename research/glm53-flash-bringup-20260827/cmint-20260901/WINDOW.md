# BALANCE-CONSTRAINED CO-ACTIVATION MINT — box window (lane/glm5-cmint, 2026-08-31)

The retry the tpd-battery window asked for: re-mint the glm5 expert placement map with the
expert-touch imbalance constrained instead of tolerated, and see whether the co-activation
lever survives at the even baseline.

**It does not, and no GPU time was spent finding out.** `--balance-tolerance` is a per-rank
expert-COUNT constraint; the quantity that flipped the lever's sign is per-rank expert-TOUCH.
At the tightest possible setting (exact 144/144 count balance) the busiest rank still runs
**85.0%** of the routing mass and expected max-rank touch is **6.784 vs even's 5.079 (+33.6%)**
— the knob recovers **6.1%** of the imbalance the tpd window measured as the sign-driver, while
giving up 10.4% of the peer-hop benefit it was supposed to preserve. The ladder stopped at the
stats table by its own kill rule.

## 0. Setup and pins

- Cell 1 is **CPU-ONLY**: no GPU, no server boot, no `TIMING-IN-FLIGHT` marker, no port. All
  four cards sat at 1 MiB for the whole window (`receipts/identity-vs-banked.txt`).
- Tool: `tools/build_expert_placement_map.py` at the `lane/glm5-ep-diet` head **25537ca8e**,
  sha256 `7d05a8d4d5cc0a5bf089774cd522715f63a690618b4d1a8d9f104e39021bb460`, byte-identical to
  the copy in the retained clone that minted the refuted artifact. `--selftest` **10/10 green**
  this window. No tool change was made: this window prices the parameter that already exists.
- Input: the **banked** struct-battery decode-filtered agentic twin, not a fresh trace boot.
  `agentic-t1-dec.ids` sha256 `b744e481…0c0219` and `agentic-t1-dec.w` sha256 `1c1a07ee…f6d421`
  — both verified against struct-battery `traces/filtered-receipts.txt` before any mint ran.
  63,462 t=1 rows, 42 layers, 288 experts, k=8.
- Invocation banked verbatim in `box/mint_ladder.sh`; aggregator in `box/agg_mint.py`.

### Control-arm identity: the ladder prices the actual refuted object

| check | result |
|---|---|
| `tol=0.05` assignment vs banked `agentic-t1-coactivation.json` | **42/42 layers identical** |
| `tol=0.05` per-layer `stats` vs banked | **42/42 layers identical** |
| `even` assignment vs banked `agentic-t1-even.json` | **42/42 layers identical** |

The default tolerance *is* 0.05, so the tpd/struct artifact (whole-file sha `56dea5ca…`) is the
`tol=0.05` rung of this ladder. Whole-file sha differs only in the embedded trace receipts
(63,714 full-tap lines vs the 63,462 decode-filtered twin); the placement and every statistic
are bit-identical. The ladder therefore measures a family containing the refuted artifact, not
a look-alike.

## 1. THE DELIVERABLE TABLE — tolerance ladder, 2 ranks

| tolerance | intra-rank co-activation fraction | single-rank fraction | expected max-rank touch | even baseline | peer-touch fraction | per-rank expert counts |
|---|---|---|---|---|---|---|
| **0.00** (tightest) | 0.7552 | 0.3509 | **6.784** | 5.079 | 0.6491 | **144 / 144** |
| 0.01 | 0.7581 | 0.3568 | 6.799 | 5.079 | 0.6432 | 145 / 143 |
| 0.02 | 0.7609 | 0.3627 | 6.815 | 5.079 | 0.6373 | 146 / 142 |
| **0.05** (= the refuted artifact) | 0.7750 | 0.3916 | **6.894** | 5.079 | 0.6084 | 151 / 137 |
| 0.10 | 0.7940 | 0.4312 | 6.999 | 5.079 | 0.5688 | 158 / 130 |
| 0.25 (loose control) | 0.8507 | 0.5592 | 7.301 | 5.079 | 0.4408 | 180 / 108 |
| `even` (engine law) | 0.4983 | 0.0071 | 5.079 | 5.079 | 0.9929 | 144 / 144 |

Monotone in tolerance on every column, and the direction is the problem: **tightening balance
buys touch and sells bundling at a ruinous exchange rate.** Full table with worst-layer rows in
`receipts/ladder-table.txt`.

    tol 0.05 -> 0.00   max-rank touch 6.894 -> 6.784   =  6.1% of the +1.815 gap recovered
                       single-rank    0.3916 -> 0.3509 = 10.4% of the peer-hop benefit lost

The tpd window's bar was "expected max-rank touch **at** even's 5.079 while keeping a
meaningful single-rank fraction". The tight rung keeps the fraction (35.1%, meaningful) and
misses the touch bar by +33.6%. Nothing in the parameter's range is close.

## 2. WHY THE KNOB CANNOT REACH THE BAR — the count/touch confusion, named

`--balance-tolerance` bounds `len(bundle)` (`hard_cap`/`min_size` on expert counts). What sits
on the TP-2 critical path is the slowest rank's expert *work*. Those are different quantities
and the ladder separates them (`receipts/mechanism.txt`):

| map | expert-count share | routing-mass share | pick-count share | expected max-rank touch |
|---|---|---|---|---|
| coact tol=0.00 | **0.5000** (perfect) | **0.8497** | 0.8357 | 6.784 |
| coact tol=0.05 | 0.5243 | 0.8652 | 0.8523 | 6.894 |
| `even` | 0.5000 | 0.5353 | 0.5307 | 5.079 |

At `tol=0.00` the constraint is satisfied **exactly** — 0.5000, a perfect 144/144 split — and
the busiest rank still carries 85.0% of the routing mass, against even's 53.5%. The parameter
is fully engaged and simply does not point at the cost.

### The touch decomposition, and why no tolerance setting could have worked

`receipts/frontier.txt`, same trace, k=8 confirmed (a uniform 2-rank split of 8 picks has
E[max] = 5.094; the even map measures 5.079):

| map | single-rank f | E[max \| single-rank] | E[max \| split] | expected max-rank touch | arithmetic floor 4+4f |
|---|---|---|---|---|---|
| coact tol=0.00 | 0.3509 | **8.000** | 6.126 | 6.784 | **5.404** |
| coact tol=0.05 | 0.3916 | 8.000 | 6.182 | 6.894 | 5.567 |
| `even` | 0.0071 | 8.000 | 5.058 | 5.079 | 4.028 |

Every single-rank token contributes max-touch **8** — all 8 picks on one rank — against even's
5.079 average. Bundling and imbalance are **the same event counted twice**. The floor for a
given single-rank fraction f is `4 + 4f` (attained only if every remaining token splits
perfectly 4/4), and at f = 0.3509 that floor is **5.404, already above even's 5.079**: a 35%
single-rank fraction at the even touch baseline is arithmetically impossible for *any*
objective, not just this one. Using even's own split behaviour (5.058) as the realistic best
for non-bundled tokens gives `em >= 5.058 + 2.942f`, which holds at 5.079 only for
f <= 0.0071 — even's own fraction. **The bar the tpd verdict set is unreachable at ranks=2.**

## 3. RANK SCALING — the direction gets worse with more ranks, not better

The tpd window's fallback hope was that the map's value lives in a future wider-EP/NVLink
shape. Measured on the same trace (`receipts/rank-scaling.txt`), it does not:

| ranks | tol | single-rank fraction | expected max-rank touch | even baseline | em/even | routing-mass share |
|---|---|---|---|---|---|---|
| 2 | 0.00 | 0.3509 | 6.784 | 5.079 | **1.3357** | 0.8497 |
| 2 | 0.05 | 0.3916 | 6.894 | 5.079 | 1.3573 | 0.8652 |
| 4 | 0.00 | **0.0650** | 5.302 | 3.526 | **1.5036** | 0.6043 |
| 4 | 0.05 | 0.0720 | 5.374 | 3.526 | 1.5239 | 0.6183 |
| 8 | 0.00 | **0.0106** | 4.171 | 2.583 | **1.6150** | 0.3978 |
| 8 | 0.05 | 0.0113 | 4.206 | 2.583 | 1.6285 | 0.4045 |

Both axes degrade together: the peer-hop benefit collapses (35.1% -> 6.5% -> 1.1% single-rank)
**and** the relative imbalance penalty grows (1.336x -> 1.504x -> 1.615x). With k=8 over 288
experts, landing all 8 picks inside one bundle of 288/R gets combinatorially rarer as R grows,
while the greedy objective's mass skew persists. The tolerance knob is negligible at every rank
count (0.00 vs 0.05 moves em by ~1% everywhere).

## 4. VERDICTS

- **`--balance-tolerance` is the wrong lever, measured, and the cell stopped at its own kill
  rule.** No GPU time was spent. The A/B was not warranted: the arm it would have timed
  (`tol=0.00`) differs from the already-refuted artifact by 6.1% of the imbalance term while
  giving up 10.4% of the dispatch benefit. Carrying the tpd window's -10.26% into that
  arithmetic predicts roughly -9.6% — the sign cannot flip, and the interleaved x3 spread
  (0.015-0.359% there) was never the limiting factor.
- **`VERDICT:coactivation-mint-must-be-balance-CONSTRAINED` is CLOSED, not satisfied.** The
  named next attempt is refuted at the mint level: for k=8 the constraint and the objective are
  the same quantity with opposite signs, and no tolerance setting — nor any objective at a 35%
  single-rank fraction — can hold expected max-rank touch at even. `MEMRA_GLM5_EP_MAP` stays
  **DO NOT ADOPT**; the engine seam stays as-is (it was never the problem, and byte-exactness
  was already proven).
- **`LAW:coactivation-expert-placement` survives with a sharpened scope.** Co-activation
  structure on real glm5 traffic is real and reproduced (39.2% single-rank t=1 events at
  tol=0.05, 35.1% even under perfect count balance). What is refuted is the *greedy bundle*
  realization of it on a walk whose critical path is per-rank expert compute. A profitable
  placement lever on this routing needs the critical path to be dispatch-bound, not
  compute-bound — or an objective that maximizes bundling **subject to a per-rank routing-mass
  cap**, which is a new objective and a tool change, and whose ceiling is bounded by the
  `4+4f` frontier above.

### Composition note (no serving claim)

None of this reaches a customer surface. TP-2 lost the 100-bar path in the tpd window on its
own merits (best TP-2 ~54.5 projected composed with spec vs **71.489** measured PP-3 + spec
single-stream), and `MEMRA_GLM5_TP` is refused at the serving worker at spawn, so there is no
customer path to flip regardless of how the map mints. This window changes **no** FLAGS
default, no roster entry, no published number, and no performance claim. Its only product is a
closed research direction and the receipts that close it, which is what keeps the next window
off this lever.

## 5. Files

    box/mint_ladder.sh              the exact ladder invocation (CPU, parallel, no GPU)
    box/agg_mint.py                 the stats aggregator, validated against the banked control
    receipts/identity-vs-banked.txt tool sha, trace shas, 42/42 control identity, cards at 1 MiB
    receipts/ladder-table.txt       the full tolerance ladder incl. worst-layer rows
    receipts/mechanism.txt          count share vs routing-mass share vs pick share
    receipts/frontier.txt           the single-rank/max-touch decomposition and the 4+4f floor
    receipts/rank-scaling.txt       ranks 2/4/8 arms
    receipts/mint-tol*.log          per-mint tool output
    maps/coact-tol*.json            the six 2-rank mints (banked so no lane re-mints them)
    maps/r4-*.json maps/r8-*.json   the wider-EP arms
