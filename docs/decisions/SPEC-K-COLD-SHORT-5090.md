# The cold-short rung stays at K=3 on the RTX 5090 Laptop (2026-11-10)

**Status:** rejected. A prompt under 1,024 tokens keeps the table's K=3 on the RTX 5090 Laptop
GPU. K=2 was measured there and is not adopted. Receipts:
[`research/spec-k-short-5090-20261110/`](../../research/spec-k-short-5090-20261110/).

## Question

On this card the cold-long rung went to K=2 on its own cell (memra#1093,
[SPEC-K-COLD-LONG-K2.md](SPEC-K-COLD-LONG-K2.md)). Two things then pointed at the short rung:

- Qwen3.5-9B, the model whose boot calibration fits on this card, has its draft length picked
  by the spec cost model, and the model serves K=2 on every prompt once it has acceptance
  history, short ones included. Pinned, K=2 is 3.6% ahead of K=3 there (nine prompts, 141 of
  144 boot pairs).
- Qwen3.8-27B does not calibrate on this card, so the table decides, and its short prompts had
  never been measured at K=2.

## Measurement

Qwen3.8-27B NVFP4 + Q5_K with its MTP head, one RTX 5090 Laptop GPU, the default against
`MEMRA_SPEC_K_COLD_SHORT=2`, one request at a time, 400-token answers, one boot per lease, 12
boots per cell s3 s2 s2 s3 x3. Texts are equal across the arms on every prompt.

- Nine prompts (three short): the short prompts read 0.995, 1.051 and 1.064; the six long ones,
  K=2 in both arms, 0.991 to 1.000.
- Twelve short prompts of different kinds (11 valid boots; one void, the machine left its
  performance power mode during it): geometric mean 1.014, nine prompts ahead, 271 of 360 boot
  pairs. Three prompts are 2 to 3% behind with 0, 6 and 6 of 30 boot pairs; without one K=3
  boot that hit thermal slowdown they lose every pair, and the mean is 1.013.

The sign follows the prompt's acceptance, not its sampler. Over the fifteen short prompts the
ratio falls 3.0% per 0.10 of the acceptance K=3 gets per drafted token (r = -0.88): K=2 gains 3
to 7% at 0.42 or less, is within 3% either way from 0.45 to 0.58 (the fitted line crosses 1.0
at 0.54), and loses 2 to 3% at 0.61 and 0.69.

## What was chosen and what was not

- **The rung stays at K=3.** The criterion: K=2's gains (up to 6.7%) are larger than its
  losses (up to 2.9%), but the loss is reproducible on a quarter of the prompts, and the 1.4%
  mean is the mean of a prompt mix chosen by hand. A default is not flipped on that. A static
  rung takes the gain on hard text only by paying on predictable text.
- **What a flip would need:** a short-prompt set with the served traffic's acceptance
  distribution, the two-at-once cell, the Qwen3.5 MoE class on this card, cached short turns,
  and the cost model's own pick on the 27B once its calibration fits.
- **Not adopted: K=2 for short prompts on this card**, for the reason above. The two-at-once
  cell and the Qwen3.5 MoE cell that a flip would have needed were not run.
- **What the result points at instead:** the faster depth follows acceptance. Where the cost
  model owns K (the 9B) it prices the depth from the model's recent acceptance, one history
  per model, and settles on the depth that wins there on every prompt. The 27B is table-owned
  on this card only because the boot calibration does not fit at its 16-session depth;
  calibrating at the depth and session count that fit would hand it the choice. What it would
  pick on a mixed workload is not measured: with one history per model it may settle on one
  depth, and a per-request choice would need an acceptance estimate per request or session.
  That is a change of its own.
- The draft length that differed between a cold run and its cache hit in the hit gate (the 9B)
  is the model's exploration at the table's rung before it has acceptance history. It settles
  on K=2 within one 400-token request, or six 64-token ones in the hit gate's run. No change.
