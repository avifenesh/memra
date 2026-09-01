# qwen4_exp round 2, work item 1 — the BOX-BASELINE re-gate (2026-08-31, the lane box)

This is a **different physical machine** from every prior receipt in this lane (the fourth
box; three were lost to preemption). It is the **same card class** — 2x RTX PRO 6000
Blackwell **Server Edition**, 97,887 MiB each, 600 W limit, verified on the box — which is
exactly what makes the round-1 numbers the right comparison and any delta a finding rather
than a new baseline. Provider, region, instance class and instance ids are fleet state and
live in darklanes, not here.

Every arm ran at the **flipped defaults with no seam env** except where a seam is named
(the PROFILE-9 §7 doctrine: a default is a claim about what runs when nobody passes a flag,
so it is verified by arming nothing and asserting an outcome). The binary's reference-parity
pin matters for reading these rows: `--goldens` / `--prompts` force the f32
exactness-instrument cache arms even under the flipped serving defaults, so the no-env
hidden and greedy rows are the **kvq0** rows and `MEMRA_Q4E_SEAMS=kvq` is the armed twin.

Binary: `binary_sha256=c8b1af69dcc05e19c020bcb810a6d7d5f5445df7963716ab550654a0f63ff50e`
(`qwen4exp_real_gate`) / `c256390550e126c25cb7feffae14a239f568337aaf4bf799caf30e1d8ebaa0d2`
(`qwen4exp_gpu_gate`), both at lane tip `69dc19a41`. Round 1's binaries were
`8cb369f06f…` / `aa6ec1d17d…` — **different binaries**, which makes the agreements below
stronger than a re-run of the same bytes.

## Verdict: NO DELTA. Eight arms, all agree; four of them byte-identically.

| # | arm | round-1 receipt | round-2 (this box) | verdict |
|---|---|---|---|---|
| 1 | tiny gate, no seam env | `devtwin/tiny-gate-defaults-on-box.tsv`, 0 failures | 302 rows, **0 failures**, every arm PASS, rc=0 | **AGREES** — 16 shared arm keys, **0 value differences** |
| 2 | real-checkpoint hidden goldens | `kvq/box/hidden-gate-kvq0-defaults.tsv` | `kvq2/hidden-gate-r2base-defaults.tsv` | **AGREES, BYTE-IDENTICAL on every data row**; argmax 10/10 |
| 3 | greedy first-divergence, raw, f32 arm | `-1 / 8 / -1 / 48` (kvq0) | `-1 / 8 / -1 / 48` | **AGREES** |
| 4 | greedy raw, `MEMRA_Q4E_SEAMS=kvq` | `-1 / 8 / -1 / 26` (kvq1) | `-1 / 8 / -1 / 26` | **AGREES** — the arm box 3 ran but never got read |
| 5 | verify-bit 24 (ship admission) | 24/24 bit-identical | `rows=24 mismatched=0 policy=bit-identity pass=true` | **AGREES** |
| 6 | spec byte-identity >= 256, raw | `devtwin/spec-gate-k5-dt4-raw.tsv` | `kvq2/spec-gate-k5-r2base-raw.tsv` | **AGREES, BYTE-IDENTICAL** incl. accept histograms |
| 7 | spec byte-identity >= 256, thinkon | `devtwin/spec-gate-k5-dt4-thinkon.tsv` | `kvq2/spec-gate-k5-r2base-thinkon.tsv` | **AGREES, BYTE-IDENTICAL** — never reached by box 3 |
| 8 | `--tp2-gate 24` | `devtwin/tp2-gate-dt4-tp2.tsv` | `kvq2/tp2-gate-r2base-tp2.tsv` | **AGREES, BYTE-IDENTICAL**: 24/24 argmax, worst_rel 3.016e-5 |

Plus the non-vacuity arm: `MEMRA_Q4E_ROUTER_AUDIT=1` reported
**`# router-audit rows=768 worst_w_ulp=1`**. Rows > 0 is the positive proof that the flipped
device-router default engages; a silent no-op default reports `rows=0`, which is the failure
mode the row counter exists to catch. PROFILE-9's round-1 figure was `rows=129004
worst_w_ulp=3` on a much larger run (spec-gate 256 x 4 prompts vs this arm's 10-token probe
with `--verify-bit-gate 8`), so the counts are **not** comparable and neither is the ulp; what
both prove is engagement. 768 = 48 layers x 16 audited rows, the expected order for this shape.

## The three byte-identical arms are the load-bearing ones, and here is why

Arms 2, 6, 7 and 8 did not merely land inside a tolerance — two different binaries on two
different physical machines wrote **the same bytes**. Some specifics worth quoting:

```
logits        10  248320  6.183e0  4.901e0  2.689e-1  1.950e1     <- round 1 AND round 2
row 0  3.553e-1  3.553e-1  1.231e1  20438  20438  true
row 4  1.883e0   1.825e0   1.888e1    888    888  true
row 9  6.183e0   4.901e0   1.638e1    271    271  true
# logits_argmax_agreement  10/10
```

Structural agreement came with it: `# vram post-load 0, 89971 MiB, 97887 MiB` on both boxes
for the single-card arm, and `0, 92755 MiB | 1, 40211 MiB` on both for the TP2 arm.

The spec arms agreed down to the **accept histogram**, which is the strictest of the four
because the histogram is a record of the whole round structure, not just an endpoint:

```
raw       prompt 0  accept 0.861  len 4.41  hist 2,5,12,9,8,22
thinkon   prompt 3  accept 0.763  len 3.37  hist 6,19,20,13,8,10
```

## A reproducibility receipt that was NOT part of the ask, and is worth keeping

Arm 7's byte identity is also a receipt for `spec/make-shape-prompts.py`. Round 1's thinkon
pack was minted on a box that is long gone; this box re-minted it from the artifact tokenizer
under **transformers 5.16.1** and got prompts whose spec rows are byte-identical to round 1's.
The chat-template render is therefore reproducible across a transformers version bump, not
merely assumed to be. Token counts: thinkon `[83, 91, 104, 105]`, thinkoff `[43, 51, 64, 65]`,
efflow `[71, 79, 92, 93]`.

## Not a delta, and stated so it is not misread as one

The round-2 tiny gate has **302** rows against round 1's 287 (and against the 263 quoted in
`kvq/ROUND2-STATUS.md`, which came from an earlier binary still). The eight extra arm keys
are `kvq-fixture`, `kvq-spec-byte-identity`, `idxq-q8-interleave`, `idxq-q8-envelope`,
`idxq-bf16-interleave`, `idxq-bf16-envelope`, `idxq-q8-spec-byte-identity` and `trunk-diet` —
arms **added** to the gate after the round-1 receipt was banked. Round 1 has **zero** arm keys
absent from round 2, so this is a clean superset: the row-count growth is arms being added,
not numbers moving. Comparing raw row counts across binaries would have read as a delta and
would have been wrong; the value-level compare on shared keys is the honest instrument.

## Process: how these receipts were banked

`run-arm.sh` runs **one arm per ssh invocation** and rsyncs the whole receipt tree the moment
the arm exits, on any rc — a failed arm's receipt is evidence too. This is the direct fix for
the 2026-08-31 loss, where a full phase-0 battery's verdicts existed only as ssh scrollback
when the box went away. The keeper's 2-minute mirror stays as a safety net; it is not the
primary path. Per-arm stdout is in `logs/<arm>.log`.

## Cross-machine tiny gate (rig, 2026-08-31): which arms are CARD-INVARIANT

Run on the rig's RTX 5090 laptop card under `flock /tmp/memra-5090.lock` — correctness only,
never timing (the rig throttles to ~52% clock). `tiny-gate-round2-rig.tsv`, **302 rows, 263
`pass=true`, 0 failures, rc=0**, with the round-2 engine changes in.

Against the box's `kvq2/tiny-gate-g3.tsv`, 24 shared arm keys split **exactly along the
bit-identity / tolerance line** — which makes this a reading rule rather than a curiosity:

**Card-INVARIANT (16 arms, values identical on a 5090 and an RTX PRO 6000):**
`fixture-longatt`, `idxq-q8-interleave`, `idxq-q8-envelope`, `idxq-bf16-interleave`,
`idxq-bf16-envelope`, `idxq-q8-spec-byte-identity`, `kvq-fixture`, `kvq-spec-byte-identity`,
`mtp-armed-prefill-bit`, `mtp-spec-defer`, `mtp-spec-defer-dirbf16`, `mtp-spec-ring`,
`mtp-spec-tiny`, `prefill-extend`, `trunk-diet`, `yarn-identity`.

**Card-VARYING (8 arms, tolerance values differ):** `dir-bf16`, `dir-nvfp4-perexpert`,
`dir-nvfp4-stacked`, `fixture`, `fixture-yarn`, `mtp-dir-bf16`, `mtp-fixture`, `mtp-rewind`.

Every card-varying arm is a comparison against a host or reference twin, where GEMM reduction
splits legally differ per card; every card-invariant arm is a bit/byte-identity arm between two
device programs, which cannot differ by card *by construction*. So:

- **A cross-machine value diff on a card-invariant arm is a real finding.** There are none.
- **A cross-machine value diff on a card-varying arm is not.** Do not file one, and do not
  compare those eight arms' numbers across boxes — compare them to the same card class only.

`prefill-extend` is worth noting: it is a *tolerance* arm (worst abs 1.865e-4) yet its value is
card-invariant, because both sides are device programs and only the chunk boundary differs. It
belongs in the first list despite not being a bit-identity arm.
