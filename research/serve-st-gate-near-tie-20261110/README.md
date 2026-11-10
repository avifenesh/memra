# serve-st-gate check 3 on the block-128 FP8 class: a near-tie between two numeric configs (2026-11-10)

Verdict: not a defect. On the synthetic block-128 FP8 Qwen3-1.7B directory the CLI and the
server part at generated token 34 because they run two numeric configs and that step is a
near-tie: the CLI's own margin there is 0.224, the third smallest of its 64 steps (median 7.7).
After that one step the served stream is the CLI's own argmax path, and the server on the
CLI's config writes the CLI's 64 ids exactly. `tools/serve-st-gate.sh` check 3 asked for
identical streams from two configs that are not bit-identical by design; it now passes on
identical streams, and where they part it requires a near-tie under the CLI's logits, a served
tail that is the CLI's own, and an identical stream from the server on the CLI's config.
memra#1181.

## What the two arms run

For a safetensors directory of this class (`Arch::Qwen3`, block-128 FP8 weights):

| arm | prime | decode |
|---|---|---|
| CLI (`run-gen`) | tokenwise, one `decode_step` per prompt token (GEMV at m=1) | the eager m=1 chain |
| server, `MEMRA_SERVE_SPEC=0` | one batched prime call over the 23 prompt tokens (the per-block FP8 MMA tile at m=23) | the batched body at B=1 |

Both pairs are documented as not bit-identical: the tile against the m=1 path in
`research/fp8blk-20260805/VERDICT.md` and `research/fp8st-20260804/mmq/SLICE3-MODEL-VERDICT.md`,
the batched body against the eager chain in `crates/memra-engine/src/decode_batch.rs`. The
batched-prime gate (#46, `crates/memra-engine/src/forward.rs`) states the law: an argmax flip on
a near-tie under small logit drift is legal between two configs, a flip on a wide margin is
not, with the bound `MEMRA_PRIME_GATE_MARGIN` (default 1.0) on the tokenwise margin.

## Measurements

One RTX 5090 Laptop GPU, binaries built from main 42bd4b07e4's sources for sm_120a (they
carry no git stamp; their sha256 is in the gate logs), the gate's prompt through the
checkpoint's chat template, greedy, 64 tokens. Untimed, single runs; every arm is
deterministic (the gate and the diagnostic passes reproduce the same ids). Tools:
`tools/diag.py`, `tools/diag2.py`, `tools/diag3.py`.

**Which config difference moves the stream** (`run1/diag.log`, one fresh server boot per arm):

| arm | prime | decode | first difference from the CLI |
|---|---|---|---|
| `srv` (the gate's arm) | tile | batched B=1 | token 34: CLI 323, server 13 |
| `srv-b1fast` (`MEMRA_SERVE_B1FAST=1`) | tile | eager | token 25: CLI 13, server 11 |
| `srv-twprime` (`MEMRA_CONFIDENCE_TRACE`) | tokenwise | batched B=1 | token 34: CLI 323, server 13 |
| `srv-both` (both doors) | tokenwise | eager | none: 64 of 64 ids equal |

Each difference alone moves the stream, at a near-tie step each time, and the server on the
CLI's config is the CLI. So the serving stack (template render, tokenizer, scheduler, sampler)
adds nothing: the whole difference is the numeric config.

**The margin at the parting step** (`run2/diag.log`, `run2/cli-margins.json`: the CLI
teacher-forced on its own ids, the raw logit row read at every step):

- The CLI's top-1 minus top-2 over its 64 steps: minimum 0.018, seventh smallest 1.35, median
  7.69, maximum 27.27. Its argmax is the emitted id at 64 of 64 steps.
- Token 34: 323 = 32.5524, 13 = 32.3283, margin 0.224, the third smallest of the 64.
- Token 25 (where the tile prime with the eager chain parts): 13 = 41.594, 11 = 40.692, margin
  0.901, the fifth smallest.

**The size of the tile prime's drift** (`run2/prime-shifts.json`: `verify_shape_bisect` on the
CLI's ids, eager decode in both, the batched prime against `MEMRA_PRIME_TOKENWISE`; the logits
of the CLI's top 8 ids at four steps):

| step | CLI margin | argmax, tokenwise prime | argmax, tile prime | shifts over the top 8 (tile minus tokenwise) |
|---:|---:|---:|---:|---|
| 10 | 17.47 | 9625 | 9625 | -0.695 to +0.527 |
| 25 | 0.901 | 13 | 11 | -0.630 to +0.598 |
| 34 | 0.224 | 323 | 13 | -0.172 to +0.891 |
| 45 | 9.15 | 12095 | 12095 | -0.632 to +0.059 |

32 shifts, rms 0.34, largest 0.89. With the eager chain in both arms, the tile prime alone
moves id 13 by +0.891 at token 34, the size this drift has at the wide-margin steps too
(-0.695 at step 10, -0.632 at step 45), where the argmax does not move. A defect of the
mis-ranking kind would move a wide-margin step; none moved. The batched body alone also flips
token 34 (`srv-twprime`); its logits were not read.

One number for the bound's calibration: the flip at token 25 sits at a tokenwise margin of
0.90. The earlier receipts for this checkpoint had their largest flip at 0.264; the
batched-prime gate's calibration note records first-token flips up to 0.70 and per-position
flips up to 0.92. It is under the 1.0 bound and at the top of what has been recorded.

**The served stream after the parting step** (`run3/diag.log`: the CLI teacher-forced on each
server arm's ids; its EXACTNESS line counts the steps where the CLI's argmax, given that arm's
own prefix, is not the arm's token):

| arm | steps that differ, of 64 | first |
|---|---:|---:|
| `srv` (the gate's arm) | 1 | 34 |
| `srv-b1fast` | 2 | 25 |
| `srv-twprime` | 4 | 34 |
| `srv-both` | 0 | |

After the one near-tie step the gate's arm writes exactly what the CLI would write from the
same prefix: its 29 later tokens are the CLI's argmax path.

## The gate change

`tools/serve-st-gate.sh` check 3:

- identical streams (the trailing EOS tolerated): pass, as before, with no extra work;
- streams that part at a step both arms have a token for: pass only when all three hold.
  (a) The CLI's logit for its token minus its logit for the server's token is at most
  `MEMRA_PRIME_GATE_MARGIN` (default 1.0). (b) `run-gen` teacher-forced on the server's ids
  disagrees with them at the parting step and nowhere else. One forced run gives both: the
  prefix before the parting step is common, so its logit row there is the row the CLI drew
  from. (c) A second server boot on the CLI's config (`MEMRA_SERVE_B1FAST=1`,
  `MEMRA_CONFIDENCE_TRACE`, whose file must have rows) writes the CLI's stream. The pass
  line names the step; the margin is printed above it;
- a margin above the bound, a second differing step, a server on the CLI's config that still
  differs, or a server stream that ends where the CLI goes on: fail, with the reason.

Rule (b) is stricter than the law: a second near-tie later in the window fails and has to be
priced by hand. Rule (c) exists only where the eager door covers the plan; elsewhere a parted
stream fails. The law's full-row drift ceiling (`MEMRA_PRIME_GATE_MAXDIFF`) is not checked,
because no door reads the served logits.

Gate runs with the change (`gates/`, the same binaries):

| run | checkpoint | result |
|---|---|---|
| `fp8-synth` | block-128 FP8 Qwen3-1.7B, synthetic | 0 failed: parts at token 34, margin 0.2241 (bound 1.0), the CLI on the server's ids differs at 1 of 64 steps, the server on the CLI's config identical 64 ids |
| `fp8-synth-tight-bound` | the same, `MEMRA_PRIME_GATE_MARGIN=0.1` | 1 failed: margin 0.2241 above the bound (the red arm of rule a) |
| `bf16-4b` | Qwen3.5-4B BF16, text tensors only | 0 failed: identical 64 ids, no second boot; check 4 equal at 211 and 400 tokens |

Before the change the first row read `1 failed`
(`research/serve-batch-removal-20261110/main-42bd4b07e4/gate-no-mtp-skip/`).

Rules (b) and (c) have no red arm of their own in the gate. `run1` and `run3` show they are
not vacuous: each single door leaves a stream that differs from the CLI, only the two
together reproduce it, and the other two parted arms have 2 and 4 differing steps, which rule
(b) would fail.

## Limits

One checkpoint of the class, one prompt, 64 tokens, one card. The served batched body's logits
were not read (no existing door gives them); its part of the difference is shown at the token
level only (`srv-twprime`). The 27B FP8 checkpoint's 2026-08-05 pass of this check was not
rerun.
