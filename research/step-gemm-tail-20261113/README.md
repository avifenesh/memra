# step-gemm-tail-20261113: the step35 batched prime's tail split on Step-3.7-Flash, PP-2 (memra#1275)

Verdict: PASS on the landing pair. With the tail split, the serving prime's last-row logits are
bit-equal to main's at 4,100 and 8,200 tokens and the 16 greedy tokens are equal, on 2x RTX PRO 6000
Blackwell Max-Q (PP-2, the IQ4_XS GGUF route with its Q8_0 MTP head).

## What changes

The step35 batched (GEMM) prime folded a tail of 1 to 15 rows into its last call, so a prompt of
k x 4,096 + 1 to 15 tokens primed a call wider than `PRIME_CHUNK_MAX_TOKENS`. With memra#1275 it
takes `fixed_prime_chunk_ranges` at the cap, as every other prime does. The step35 prime is not
bit-stable across call decompositions in general (the `MEMRA_STEP_GEMM_PRIME_SUFFIX` row of
`docs/FLAGS.md`), so the change is checked on the model.

## The check

`tools/step-check.sh`, with the PASS rule written in its header before the run and applied by
`tools/step-compare.py`. Per arm and length, two processes on the pair
(`MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_PARALLEL=off`):

1. `run-gen` with `MEMRA_PP_ONLY=1 MEMRA_PP_PRIME=1 MEMRA_PP_LOGITS`: the serving prime's last-row
   logits;
2. `run-spec` with `MEMRA_GEN_ONLY=1 MEMRA_NGEN=16` and the external MTP head (`MEMRA_MTP_DRAFT`): the
   serving prime plus 16 plain greedy tokens.

Prompts: the exact 4,100- and 8,200-token prefixes of `research/e2e/prompts/p4-16k.txt` (the
Step-3.7-Flash tokenizer, BOS 0 prepended), passed as ids so both arms prime the same tokens.

PASS, per length: equal last-row argmax, equal 16 greedy tokens, last-row maxdiff within the
documented near-tie band (2.4), the GEMM prime engaged at the full length, every process exit 0.

## Result on the landing pair (`pro6000-pp2/run-084758`, 2026-11-14)

Binaries: base = main 63c351a2f4, head = eb648a2150 (this change rebased onto it). Both carry two
print-only patches, so that the receipt shows every call's width
(`tools/print-patches-*.diff`: the GEMM prime's `ENGAGED` line and the prime walk list their
calls). Their sha256 are in `guard.log`.

| tokens | base calls | head calls | last-row argmax (top-2 margin) | last-row maxdiff, base vs head | 16 greedy tokens |
|---|---|---|---|---|---|
| 4,100 | 4,100 | 4,084 + 16 | 16 (0.2152) in both | 0, bit-equal | equal |
| 8,200 | 4,096 + 4,104 | 4,096 + 4,088 + 16 | 2745 (3.901) in both | 0, bit-equal | equal |

- The logits files of the two arms have the same sha256 at each length (`LOGITS.sha256`; the
  binary files are not in the repo).
- All eight processes exit 0 and `verdict.txt` ends `VERDICT PASS`.
- Single run per arm and length; the cards were shared with another job's pair, and no timing is
  read from these logs.

## Earlier runs (`pro6000-pp2/run-191529`, `run-192323`, 2026-11-13)

On the pair of that day (base 224d374573, head 2e526edfae: the same diff on an older main):

- `run-191529`: the prime half passed with bit-equal last-row logits at both lengths (the same
  sha256 as above). The greedy half had no MTP head on the box, so `run-spec` exited 2 and the
  verdict is FAIL for that reason alone.
- `run-192323`: an attempt to take the greedy tokens from `run-gen`'s plain path. At 4,100 its own
  prefill gate panicked on a near-tie in both arms, and at 8,200 its generation failed on PP-2
  with a cross-device error (memra#1295); the run was stopped. Kept as it ran.

## Files

- `pro6000-pp2/run-*/`: `guard.log` (binaries, prompts, cards), the per-process logs, `verdict.txt`,
  `LOGITS.sha256`. The box's name, account, data mount and card UUIDs read as neutral names.
- `tools/`: `step-check.sh`, `step-compare.py`, `mkids.py` (the prompt ids), the print-only
  patches for both pairs.
