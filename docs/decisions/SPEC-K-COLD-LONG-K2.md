# Cold long prompts speculate at K=2 on the RTX 5090 Laptop and the RTX PRO 6000 Max-Q (2026-11-09)

**Status:** adopted for the two cards whose cell has K=2 ahead, keyed by device name: the RTX
5090 Laptop GPU and the RTX PRO 6000 Blackwell Max-Q. GB10, whose cell has K=3 ahead, and every
other card keep K=3. Issue memra#1093. Receipts:
[`research/spec-k-cold-long-20261109/`](../../research/spec-k-cold-long-20261109/),
[`research/spec-k-cold-long-c2-20261109/`](../../research/spec-k-cold-long-c2-20261109/),
[`research/spec-cost-draft-step-20261109/`](../../research/spec-cost-draft-step-20261109/).

## Question

The automatic spec K table gives a cold prompt of 1,024 tokens or more (less than 1,024 of them
restored from the prefix cache) the cold-long rung. It shipped at K=3, the depth the 2026-08-08 K
policy measured for cold prompts (`research/kpolicy-20260808/`): on its 27B cell a cold 5,411-token
prompt ran 55.46 tok/s at K=3 against 52.22 at K=2 (+6.2%). That cell does not describe this
traffic: its rate is 128 new tokens over the whole request, prefill included, its drafter accepted
78 to 79% per draft where this model's MTP head accepts 44 to 65%, and it ran on another rig and
an older engine. memra#1093 gave a
sampled resume of a long conversation its cold twin's rung, which cost such resumes about 3% of
their decode rate against the cached-long K=2. That raised the question the other way round: is
K=2 faster for every long prompt, cold ones included?

## Measurement

Qwen3.8-27B NVFP4 + Q5_K GGUF with its untrimmed MTP head, context 8,192, every switch at the
device default. Arms on one binary: the device default (K=3) and `MEMRA_SPEC_K_COLD_LONG=2`.
Twelve boots per cell in the order k3 k2 k2 k3 three times, one boot per GPU lease, the same cold
prompts in every boot, answers capped at 400 tokens, thinking off, 250 ms telemetry. A pair of
short prompts on the cold-short rung (K=3 on both arms) is each boot's clock reference. Ratios are
K=2 over K=3 on per-boot medians; U counts the boot pairs (of 36) K=2 wins.

| card | requests at once | class | decode ratio | U |
|---|---:|---|---:|---:|
| RTX 5090 Laptop | 1 | T=1 1.2k | 0.996 | 16/36 |
| RTX 5090 Laptop | 1 | T=1 3.6k to 3.7k | 1.056 to 1.070 | 31/36 |
| RTX 5090 Laptop | 1 | T=1 5.8k | 1.058 | 36/36 |
| RTX 5090 Laptop | 1 | greedy 3.6k | 1.121 | 36/36 |
| RTX 5090 Laptop | 2 | T=1 1.2k | 1.067 | 36/36 |
| RTX 5090 Laptop | 2 | T=1 3.6k | 1.082 | 33/36 |
| RTX 5090 Laptop | 2 | T=1 5.8k | 1.115 | 34/36 |
| RTX 5090 Laptop | 2 | greedy 3.6k | 1.092 | 36/36 |
| RTX PRO 6000 Max-Q (250 W) | 1 | T=1 1.2k | 1.045 | 36/36 |
| RTX PRO 6000 Max-Q (250 W) | 1 | T=1 3.6k to 3.7k | 1.055 to 1.074 | 36/36 |
| RTX PRO 6000 Max-Q (250 W) | 1 | T=1 5.8k | 1.039 | 36/36 |
| RTX PRO 6000 Max-Q (250 W) | 1 | greedy 1.2k, 3.6k | 1.016, 1.096 | 36/36 |

The RTX 5090 rows are normalized by the boot's reference, because that card throttled through its
runs; the PRO 6000 rows are raw. TTFT did not move on any cell (ratios 0.97 to 1.04, no
separation). A greedy request wrote the same text at both depths in every boot. At T=1 another K
draws another sample stream, so a sampled request's text changes with the depth, as it does with
any other change of K (`research/spec-k-resume-20261109/`).

Why K=2 wins: the third draft position is accepted about one time in five on these prompts (the
MTP head's acceptance by position is about 0.70, 0.50, 0.21), so a K=3 round pays a fourth verify
row and a third draft step for about 0.2 more tokens. Expected tokens per round are 2.20 at K=2
and 2.41 at K=3 (+9.5%); the round is 21.6 ms against 24.4 ms on the PRO 6000 (+13%).

## GB10 keeps K=3

GB10 is in the same `120a` build family, so a family-keyed rung would have moved it. Its own cell
(memra#1145's lane, `research/gb10-decode-20261109/kcl-unit1/`, which lands with that PR: the
served GB10 route with the masked MTP draft, the same harness with a prompt set drawn from both
of this lane's sets, 12 boots k3 k2 k2 k3 x3 on one unit, the clock steady) has K=3 ahead:

| class | K=3 decode (tok/s) | K=2 decode (tok/s) | K=2 over K=3 | U (K=2 faster) | TTFT ratio |
|---|---:|---:|---:|---:|---:|
| T=1 1.3k | 23.37 | 23.06 | 0.987 | 4/36 | 1.002 |
| T=1 3.6k | 23.62 | 22.88 | 0.969 | 0/36 | 1.000 |
| T=1 5.8k | 23.46 | 22.71 | 0.968 | 0/36 | 1.001 |
| greedy 3.6k | 26.34 | 26.29 | 0.998 | 0/36 | 0.999 |

On GB10 the trunk verify dominates a round and costs about the same for one to eight rows, and
the draft is about 6% of it, so K=3's extra row and draft step cost little and its extra accepted
tokens pay. The rung is therefore keyed on the measured cards, not on the build family.

## The PRO 6000's cost model

On a card whose boot calibration fits, the spec cost model owns K: it prices each draft length up
to the rung and serves the best. The PRO 6000 runs it. For this model the RTX 5090 Laptop does
not (the calibration needs 16 sessions at depth 2,048, which do not fit beside a 27B, so it is
skipped and the table's rung serves); a model small enough to calibrate there, Qwen3.5-9B for
one, does run it (Limits). On the PRO 6000 the model picked K=3 for these prompts, the slower
depth. What was found, in
[`research/spec-cost-draft-step-20261109/`](../../research/spec-cost-draft-step-20261109/):

1. **One drafting cost for every K (fixed).** The boot calibration measured the per-round cost
   beyond the verify once, from a K=3 burst, and charged it to every draft length, so K=2 paid
   three draft steps. On this card the calibration now bursts at K=1 and K=3 and fits a
   fixed part plus one draft step (0.35 to 1.08 ms and 1.09 to 1.39 ms over this card's 21
   boots); a round at draft length k pays k steps. The table's round times then match what was
   served: K=3 minus K=2 is 2.6 ms priced against 2.6 to 3.1 ms realized. The two extra bursts and their
   sessions add about 7 s to the boot calibration of an MTP model on this card (19.5 to 20.9 s
   before, 26.5 to 28.2 s after). With the rung still at 3 the model then serves K=2 on 38 of
   50 cold-long requests after a boot's first (0 of 74 before) and decodes T=1 prompts of 1.2k
   and 3.6k tokens 5.5 to 7.2% faster than the base at the same rung.
2. **Realized rates decide between draft lengths on a few bursts (found, not fixed).** After
   three ticks a served (sessions, K) regime is priced from its realized rate, an EMA of per-tick
   samples with weight 0.3. A tick is one 32-token burst of about a dozen rounds, and its tokens
   per round vary from burst to burst by 11% (K=2) to 15% (K=3) of the mean, two to three times the 5%
   that separates the depths, so the served depth's price wanders across the other's stale one.
   That is why the fixed model still returns to K=3 on some requests. A candidate that sampled a
   tick as its round rate times the acceptance window's expected tokens was built and measured
   (`pro6000-maxq/sample-long/`): its prices held within 1 tok/s, but K=2 and K=3 priced as a tie
   (99.8 against 100.0 tok/s where the requests ran 102 and 96), and it served K=2 on 14 of 18
   later requests, no better than the 18 of 24 without it. It was deleted. The window pools the
   per-position acceptance of whatever draft lengths and prompts it holds, and the third
   position's share comes only from the few K=3 requests in it.
3. **Not the calibration depth.** The table is measured at 2,048 tokens and these prompts run at
   1,200 to 6,200; that was the first suspect. The priced difference between K=3 and K=2 rounds
   matches the realized one (item 1), and a served regime is priced from serving, not the table.

The per-prompt picture explains the size of item 2's problem: K=2's lead is an average. Over the
15 long prompts of the fixed-K A/B, K=2 leads by 8 to 12% on the five whose third draft position
is accepted least and is within 1.5% either way on the four where it is accepted most. An online
choice between two depths 5% apart on average, from a handful of requests each, does not settle
on that spread; the 12-boot A/B does. So the rung carries the default, and the model keeps
choosing below it (K=2 against K=1 and plain).

GB10 keeps its calibration as it was: one K=3 burst, its cost charged to every draft length. Its
model picks K=3 for cold long prompts, which its own cell says is right, and the extra bursts were
not timed there.

The final binary on that card (`pro6000-maxq/final-long/`, three boots per arm, the nine prompts
of the long set one at a time): at its default it served K=2 on all 27 requests; with the rung
forced back to 3 the fixed model served K=2 on 8 of 12 cold-long requests after a boot's first;
the base served K=3 on all 27. Four requests at once (`pro6000-maxq/final-c4/`, two boots per
arm): the model demotes to plain batching within about 130 tokens on both binaries; the 3.6k
group's aggregate is about 4% higher on the flip binary, the 1.2k group's 3.6% lower (the gate
probes K=1 once where the base goes straight to plain), the short group's flat.

## Decision

- `SPEC_K_COLD_LONG_MEASURED = 2`: the cold-long rung is K=2 where `spec_k_cold_long_measured`
  holds, by device name: the RTX 5090 Laptop GPU and the RTX PRO 6000 Blackwell Max-Q, as the
  other card-keyed defaults are (`KV-PARK-COMPACT-DEFAULT.md`, `PREFILL-WORKSPACE-HELD.md`).
- GB10 keeps K=3: its cell has K=3 ahead (above). The desktop RTX 5090, the other RTX PRO 6000
  editions and the `100a`, `90a` and `89` builds keep K=3: no cell of theirs says K=2 wins, and
  one-rig evidence sets a one-rig default (`AGENTS.md`, per-hardware arm selection).
- The cost model's draft-step fit is on for the RTX PRO 6000 Blackwell Max-Q only
  (`spec_cost_fits_draft_step`), the one of the two cards that runs the calibration.
- The cached-long rung (K=2 greedy, K=5 with a rank-trimmed MTP head) and the cold-short rung
  (K=3) keep their table values. On the PRO 6000 Max-Q the cost model then serves short prompts
  at K=2 as well while long ones are in the traffic: it prices a draft length only while its
  30 s acceptance window holds rounds that offered that length's last position, and K=2 rounds
  offer no third. Accepted as flat on what was measured: the two short prompts are 3.4% slower
  and 4.9% faster at K=2 in the fixed-K A/B, and the final binary's short class is 1.0% above
  the base (6/9). A rank-trimmed MTP head is not a default on any card (`MEMRA_FRSPEC_TRIM` is
  off), so no trimmed rung was priced here.
- On the two cards a sampled resume of a long conversation now runs K=2 again, through its cold
  twin's rung: the 3% that memra#1093's exactness fix cost such resumes comes back.
- `MEMRA_SPEC_K_COLD_LONG=3` is the rollback and the measurement arm.

## Rejected

- **Leaving K=3 and letting only the cost model move.** The RTX 5090 Laptop does not run the
  model for this checkpoint, so it would keep the slower depth; and the model alone does not
  hold K=2 (item 2 above).
- **Overriding the PRO 6000's model with the rung alone.** With the rung at 2 the model can no
  longer pick K=3 for a cold long prompt, but the flat drafting cost also misprices K=1 against
  K=2 and K=2 against K=3 on the short rung. The table was fixed first.
- **Round-rate realized samples** (item 2 above): flat on the picks, deleted. The patch that ran
  is kept beside its receipts (`research/spec-cost-draft-step-20261109/round-rate-sample.patch.gz`).
- **The draft-step fit on GB10 and on unmeasured cards.** Not measured there; GB10's model
  already picks the depth its cell has ahead.
- **K=2 on every build.** GB10 measured the opposite; the B200 and H100 classes are unmeasured.

## Limits

- One model (Qwen3.8-27B, untrimmed MTP head), answers of 400 tokens, prose prompts. The rung is
  one per device, not per model or head: other MTP models, a rank-trimmed head (its draft step is
  far cheaper, which removes part of K=2's reason) and the GLM5 route under an operator gate
  override take it without a cell of their own. The rung's override is the measurement arm for
  them.
- A model small enough to calibrate on the RTX 5090 Laptop (Qwen3.5-9B does) runs the cost model
  there with this rung and the older flat drafting cost: the combination rejected below for the
  PRO 6000. Not measured. The 2026-08-08 K policy has K=2 ahead of K=3 by 0.7% on that model's
  cold-long cell.
- The PRO 6000 cells are one request at a time, plus one cell of four at once with two boots per
  arm. Two and three sessions on that card, and every concurrency on other MTP models, are not
  measured.
