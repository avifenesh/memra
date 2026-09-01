# qwen4_exp round 2, work item 2 — the CALIBRATED TP2-prefill class gate

Box: the lane box, 2x RTX PRO 6000 Blackwell **Server Edition** 97,887 MiB / 600 W.
Artifact `q48fn-nvfp4`. Receipts in `kvq2/tp2-prefill-gate-*.tsv`, per-arm stdout in `logs/`.

**Cache arm, corrected:** these rows ran the **f32 exactness-instrument caches**
(`kv_quant=f32 idxq=f32`), not the kvq/idxq serving defaults. This file first said "kvq
serving defaults" — wrong, and the receipt could not have contradicted it, because the header
did not record the cache arm at all. It does now (`# cache kv_quant=... idxq=...
golden_pin=... seams_env=...`), and the reason the arm is f32 is stated there:
`--tp2-prefill-gate` takes `--goldens`, and the binary pins golden comparisons to f32.

That is the RIGHT arm for this gate, which is why the numbers stand: a band on the TP2
expert-half split should isolate the split, and quantization noise in the cache would be a
second variable inside it — the same isolation argument that `set_prefill_grouped_all` exists
for. The bands are therefore f32-cache bands, and a kvq-cache TP2 band, if one is ever
wanted, is a separate calibration.

## The calibration found three defects before it could produce a band

The band constants were placeholders (10x a *borrowed* green worst) and the resume order's
step 3 was "run `--tp2-class-calibrate`, read the measured worst, set the constants". The
calibration run could not do that, because the gate as written could not measure what it
claimed to. All three defects are now fixed and each fix carries its own control.

### Defect 1 — `HeadMode::All` did not exist in the TP2 forward

`forward_tp2` allocated its output as one row (`vec![0.0f32; vocab]`) and copied
`plane[(t-1)*hidden]` into the exit segment regardless of `HeadMode`. So `All` was silently
identical to `LastRow`: a caller asking for every row got exactly one, with no error.

The PRIME regime's entire purpose is "compare EVERY row of a full-head forward" — the old
`--tp2-prefill-gate` compared only the chunked prefill's last row, which is a t==1-shaped
read of a t>=2 program, and that is what the two-regime gate was written to fix. It could
not have done so. It surfaced only because the gate length-checks the two logits vectors
before comparing:

```
Error: "tp2 prime: single-card produced 2483200 logits, TP2 248320 (head mismatch)"
```

Without that check the gate would have compared one row and printed a t>=2 verdict. Worth
keeping as a corpus entry: the loud-failure that saved this was a **length assert**, not a
value assert.

Fix: `tp2_seg_exit` takes a `rows` parameter (both `gate_read_inner` and
`launch_qmatvec_bf16w` already take a row count — the kernel's grid y-dim *is* `t`, striding
`x` by `x_tstride` — so this is a parameter that was never threaded, not new math), and the
`All` path runs the exit straight off the planes instead of copying a last row. Chunked
prefill still uses `LastRow` for the same reason the single-card path does: a `[t, vocab]`
block is gigabytes at long-context chunk sizes (it is ~9.9 MB at this gate's 10-token probe).

**Control: at `rows == 1` the launch arguments are byte-for-byte the old ones.** Proven, not
argued — `tp2-gate-g2-tp2.tsv` (24 decode steps through `decode_step_tp2` -> `tp2_seg_exit`)
is **byte-identical** to `tp2-gate-r2base-tp2.tsv` from before the change.

### Defect 2 — the band and the measurement were in different units

The gate computed `max_rel = max |a-b| / max(|a|, 1e-6)`. Every other receipt in this lane,
`--tp2-gate`'s cited 3.0e-5, and the glm5 lane's 4.85e-5 green worst that the placeholders
were borrowed from all use `compare()`, which floors the denominator at **1.0**.

The in-code comment claimed the 1e-6 floor was "the strict form (denominator floored, so a
near-zero logit cannot flatter the comparison)". The sign of that reasoning is backwards: a
near-zero denominator does not fail to flatter, it catastrophically **penalizes**. Over a
248,320-wide vocab there are always logits near zero, so the first calibration run measured
`prime_worst_rel = 2.865e4` on a row whose worst *absolute* difference was 3.975e0 and whose
top-1 matched. Checking 2.865e4 against a band derived from a floor-1.0 measurement is a
category error, not a loose bar.

Fix: the band metric is `compare()`'s, so the gate is in the same units as `--tp2-gate`, the
rest of this lane, and the borrowed glm5 number. The 1e-6-floored quantity is still reported
as `elem_rel` per row and `worst_elemrel` in the verdict, because it does say something real
about near-zero-logit behavior — it is a diagnostic column and never the bar.

### Defect 3 — the PRIME regime straddled two variables, and the wrong one dominated

`HeadMode::All` selects the **per-expert** MoE executor (`grouped = exact || head != All`),
while TP2's `tp2_moe_rows` is **grouped** on both cards. So prime compared per-expert against
grouped: the TP2 expert-half split AND the executor difference at once, with the executor
term ~100x larger.

Measured, on the same run:

| comparison | worst max_rel |
|---|---|
| prime rows, per-expert (single) vs grouped (TP2) | 2.0e-3 .. **4.1e-3** |
| chunked row, grouped on BOTH sides | **1.4e-5** |

Two orders apart, and corroborated independently by the tiny gate's own `prefill-extend` arm,
which prices the grouped-vs-per-expert difference *alone* at **1.865e-4** on a fixture. A band
read off the straddled number would have been ~100x too loose for the question the gate asks.

Fix: `set_prefill_grouped_all` (FLAGS.md row in the same commit; **default OFF**, scoped ON
around the prime forward only and restored immediately). **Control: hidden-goldens and
verify-bit receipts are byte-identical with the seam present and default OFF**
(`g2-*` vs `r2base-*`), so OFF is byte-for-byte the old behavior rather than assumed to be.

## The measured green-worst distribution, and the band it sets

`--tp2-class-calibrate`, 19 rows, argmax **19/19**, tape OK, peer engaged
(`tp2-prefill-gate-cal-green2.tsv`):

| regime | rows | min max_rel | worst max_rel | worst max_abs |
|---|---|---|---|---|
| **prime** (t>=2) | 10 all-rows + 1 chunked | 3.815e-6 | **1.383e-5** | 1.454e-5 |
| **decode** (t==1) | 8 | 6.557e-6 | **1.574e-5** | 1.621e-5 |

Bands set at 10x the measured green worst, in the gate's own metric, on this artifact and
card class — **both tighter than the placeholders they replace** (calibrate downward, never
up):

| constant | placeholder (borrowed) | **calibrated** | basis |
|---|---|---|---|
| `TP2_PRIME_BAND` | 2e-4 | **1.4e-4** | 10 x 1.383e-5 |
| `TP2_DECODE_BAND` | 3e-4 | **1.6e-4** | 10 x 1.574e-5 |
| `TP2_RED_FLOOR` | 1e-3 | 1e-3 (kept) | ~64x green worst, ~6x either band; the RED rows below justify it |

### A finding inside the calibration: decode is NOT tighter than prime here

The placeholder's comment reasoned that decode must be the tighter band because "a t=1 row
has no batched GEMM width variance — only the expert-half join reorder". Measured, decode's
green worst (**1.574e-5**) is slightly **larger** than prime's (**1.383e-5**). The join
reorder alone already puts t==1 in the same order, and the batched width variance the prose
expected to dominate does not. The band follows the measurement, not the prediction. This is
the second place in one gate where prose about the numerics was confidently wrong in a
checkable way, which is itself the argument for calibrating rather than reasoning.

### `decode_byte_identical` is measured, and it is FALSE

Reported as a field rather than assumed as the bar, exactly as designed. The glm5 lane got
byte identity at t==1 because its program was column-parallel-over-gather; ours is an
expert-half split with a join, and it does not. Byte identity here would have been a finding;
its absence confirms the gate was right not to bar on it.

## The band is a bar: the three RED arms land six orders outside it

Each red must land past `RED_FLOOR` **or** break the greedy tape, **and** must have engaged
the peer card — a red that never routed a peer expert is a no-op and proves nothing.

| arm | prime worst | decode worst | vs green worst (1.383e-5) | tape | argmax | peer_slots | verdict |
|---|---|---|---|---|---|---|---|
| **green** (none) | 1.383e-5 | 1.574e-5 | 1x | ok | 19/19 | 6908 | pass, `loud=false` |
| `skip-peer-moe` | **9.930e0** | 8.116e0 | **~7.2e5 x** | BROKEN | 13/19 | 6999 | pass by being loud |
| `peer-local-ids` | **1.003e1** | 9.178e0 | **~7.3e5 x** | BROKEN | 11/19 | 6987 | pass by being loud |
| `reverse-peer-weights` | **8.271e0** | 6.995e0 | **~6.0e5 x** | BROKEN | 13/19 | 6937 | pass by being loud |

Every red is ~4 orders past `RED_FLOOR` and ~5 orders past the band, breaks the tape, and
engaged the peer. There is a four-order gap between where a correct program lands and where
the *least* wrong of these three lands, so the band distinguishes rather than absorbs.

Note `peer-local-ids` is the most damaging of the three by argmax (11/19) while being the most
plausible real bug — right magnitudes, wrong experts. That is the arm worth keeping if one
ever has to be dropped.

## Per-rank engagement: the glm5 lane's DERIVED fractions, measured here

`ROUND2-STATUS.md` records that the glm5 EP fractions (~99.3% peer-touch, ~64% slowest-rank
bytes, ~1.57x effective) are closed-form derivations with no measurement behind them, and that
this lane's counters would be the measured version for the qwen4_exp geometry. They are:

```
# expert-split  peer_slots=6908  home_slots=6532  peer_slot_fraction=0.5140
                layer_tokens=1344  both_card_rows=1343  both_card_fraction=0.9993
                engaged=true
```

**Measured, this geometry (512 experts, even split, this probe): 99.93% of layer-tokens touch
BOTH cards, and the peer card takes 51.40% of dispatched expert slots.** This is the number
the co-activation placement lane exists to reduce, and it is now a counted row rather than a
hypergeometric expectation. It is not comparable to glm5's 99.3% — that was a different expert
count and top-k, and it was derived — so quote them separately.

The near-total both-card fraction is the honest framing of the even split's cost: at 512
experts and this top-k, an even assignment means essentially **every token pays a cross-card
join**, which is exactly why placement is worth measuring and why the even arm is the control.

## Gate verdict, as it now stands

```
# verdict  rows=19  argmax_matches=19  prime_worst_rel=1.383e-5  prime_band=1.4e-4
           decode_worst_rel=1.574e-5  decode_band=1.6e-4  decode_byte_identical=false
           worst_elemrel=2.512e0  tape_ok=true  peer_engaged=true  red_arm=none
           red_floor=1.0e-3  loud=false  calibrate_only=false  pass=true
```

Plus `--tp2-gate 24`: **24/24 argmax, worst_rel 3.016e-5**, byte-identical to round 1.

## Per-commit gate battery with both engine changes in

| gate | result |
|---|---|
| tiny gate, all arms, no seam env | 0 failures; **24 shared arm keys, 0 value differences** vs the pre-change run |
| hidden goldens (single-card default path) | **BYTE-IDENTICAL** to pre-change |
| verify-bit 24 | **BYTE-IDENTICAL** to pre-change; `rows=24 mismatched=0 pass=true` |
| spec byte-identity 256, raw | `pass=true`; rows differ **only** in the two wall-clock columns (7.57 -> 7.56 ms, 14.14 -> 14.12) — accept rates, histograms and first_divergence identical |
| `--tp2-gate 24` | **BYTE-IDENTICAL** to pre-change |
| `--tp2-prefill-gate 8` (calibrated) | green, and the 3 REDs loud + engaged |
