# Official gemma4-assistant drafter — A/B vs dspark (lane/gemma-assistant, 2026-08-16)

Owner directive: "we want every drop we could get, so the bar is our best results, not
other best results, go for it." Bar = OUR dspark baselines (prose 0.549/132.6, code
0.739/190.4 tok/s, Japan @450W, Q4_0 trunk — receipts/accept5.log, accept-code.log).

**VERDICT: the official assistant BEATS dspark on both trunks, both classes, exact
everywhere. It is the drafter to ship.**

## What actually happened to the "days-class arm"

The recon sizing (ASSISTANT-ARM-SPEC.md, superseded header there) missed that
`gemma_spec.rs` has implemented EXACTLY this architecture since 2026-07-10 — Q-only
dual-geometry draft attention over the MAIN model's KV, concat pre-projection, tied
head. The days-class build collapsed into: pin the HF glue (done, candidate_generator
concat order confirmed matching), gate the existing arm on current artifacts, convert
the OFFICIAL checkpoint, and measure. Total: hours.

## Weight-lineage pairing law (new, load-bearing)

Byte gate caught it before any cell: the on-disk `gemma-4-31B-it-Q8_0-MTP.gguf` is the
**QAT-lineage** assistant (layer_scalars/output_norm differ from gg-hf-am bf16;
output_norm max|Δ| 10.2). Fresh conversion of the official bf16 checkpoint minted +
parity-gated EXACT (layer_scalars byte-equal, output_norm 1024/1024):

- `gemma-4-31B-it-official-F16-MTP.gguf` (955MB, bf16→f16 exact-in-range;
  convert_assistant.py clones the proven QAT GGUF metadata, swaps tensors)
- `gemma-4-31B-it-official-Q8_0-MTP.gguf` (511MB, llama-quantize)
- both in `/data/memra/models/gemma4-31b/` (Japan) + local tooluse dir.

Cross-pairing costs HALF the acceptance (5090 + box, same direction):
QAT head on QAT trunk 0.573-0.627 vs official head on QAT trunk 0.283-0.344;
official head on official-weights NVFP4mix trunk 0.584-0.597 (prose).
**Drafter lineage must match trunk lineage.** QAT↔QAT, official↔NVFP4mix.

## Deciding cells — NVFP4mix serving artifact (Japan, device 1, 450W, interleaved ×5)

Receipt-exact prose prompt (accept5.sh ids, n=128); code prompt = same chat frame +
code-class content (28 toks, n=256). Every arm re-measured back-to-back per rep; all
reps dead flat; **every cell byte-exact vs plain greedy** (128/128, 256/256).

| class | plain | dspark dflash | official-F16 K=5 | official-Q8+ranks K=5 | Δ vs dspark |
|---|---|---|---|---|---|
| prose | 58.4 | 110.0 (1.88×, acc 0.566) | 122.7 (2.10×, 0.584) | **154.9 (2.65×, 0.597)** | **+41%** |
| code  | 58.2 | 166.3 (2.86×, acc 0.774) | 147.3 (2.53×, 0.724) | **176.1-178.8 (3.02-3.07×, 0.710-0.724)** | **+6-8%** |

Notes: the assistant wins code despite LOWER acceptance than dspark (0.72 vs 0.77) —
its draft step is far cheaper (4 layers × 1024 hidden reading the trunk's KV vs
DFlash block-16 at 5376). +ranks = the 447k own-gen FR trim (private copy, fresh
`.learned` sidecar) + 512 adaptive slots; it buys ~25% draft-head bandwidth on top of
F16 full-head. K-sweep (K=6,7 ×3): flat vs K=5 — the adaptive-depth policy
self-limits; **K=5 + ranks trim is the config**.

## QAT Q4_0 trunk (receipt-comparable, interleaved ×5, same session)

Stable-condition reps (plain ≈74; some reps saw device-0 lane contention — ratios are
within-rep, arms adjacent in time):

| class | plain | dspark dflash | QAT-Q8 assistant K=5 | official-F16 (cross-pair) |
|---|---|---|---|---|
| prose | 74.3 | 131.5 (1.77×, 0.549 — receipt reproduced exactly) | **205.9-208.6 (2.77-2.80×, 0.627)** | 99.1 (1.33×, 0.283) |
| code  | 74.1 | 142.4-142.7 (1.92×, 0.600) | **235.7 (3.18×, 0.737)** | 93.3 (1.26×, 0.253) |

(dspark's code receipt 190.4/0.739 used a different 37-tok prompt; on THIS prompt,
same-session interleaved, dspark reads 142.7 and the assistant 235.7 — the receipt
number is not contradicted, the comparison is same-cell.)

## Where this leaves the board

- Previous best (dspark, Q4_0): 132.6 prose / 190.4 code.
- Now (QAT assistant, Q4_0): **~206 prose / ~236 code** @450W.
- Serving artifact (official assistant + ranks, NVFP4mix): **154.9 prose /
  176-179 code** @450W — above the 127-tps OR board top on both classes, with the
  NVFP4 single-stream base still at 58 (Q4 is 74): the fused2 lane's 55→72 base fix
  compounds directly into these spec numbers when it lands.
- Exactness law intact everywhere: spec output byte-identical to plain greedy in all
  ~90 cells of this campaign.

## Honest caveats

- Some q4 reps ran while the device-0 lane loaded/ran its own cells (plain dipped to
  35-42); interleaved design keeps arm-vs-arm honest, absolute tok/s from the flagged
  reps should not be quoted. NV cells were quiet and dead flat.
- The code-class prompt is not the receipt code prompt (its ids were never recorded);
  both arms measured on the identical new prompt.
- gguf drafter head trim (`MEMRA_GEMMA_DRAFT_RANKS`) requires a quantized head — the
  F16 arm ran full-head (its 122.7 prose is head-bandwidth-bound; Q8+ranks is the
  shipping config).
- h-seed convention: memra/llama.cpp use POST-output_norm backbone hidden; HF 5.14.1's
  glue seeds PRE-final-norm (hidden_states records per decoder layer). p0 acceptance
  0.84 (properly paired) says the shipped convention performs; a pre-norm-seed
  experiment remains as possible acceptance upside, sized small.

## Protocol / repro

- `assistant_ab.sh` (this dir; box copy /data/memra/assistant_ab.sh), evidence
  receipts/assistant-ab/ + box /data/memra/evidence/gemma-assistant-ab/.
- `convert_assistant.py` (this dir) — official checkpoint → gemma4-assistant GGUF.
- gemma-gate ambiguity guard added: MEMRA_SPEC_DFLASH + MEMRA_DRAFT together now
  refuses instead of silently running dflash.
- Envs (shipping config): `MEMRA_SPEC=5 MEMRA_DRAFT=<official-Q8-MTP>
  MEMRA_GEMMA_DRAFT_RANKS=<ranks> MEMRA_GEMMA_TRIM_ADAPT=512` (defaults: adaptive
  depth floor 4, self-keyed in-round pmin).
