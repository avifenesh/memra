# HY3 native EP tuning — 4× RTX PRO 6000

Status: implementation checkpoint complete. No default flip or `NativeTuned` claim is authorized
by this file; concurrency scaling and long-prompt TTFT remain open in issue #67.

## Bound tuple

- Artifact: the sealed all-expert HY3 ModelOpt W4A16 checkpoint, index
  `0f22f6fc51ac7e39b7510a77c77098c4fd7c722e9e6cfdb9782247c37f1b6afd`.
- Hardware: four NVIDIA RTX PRO 6000 Blackwell Server Edition cards.
- Placement: `MEMRA_PARALLEL=auto`, devices `0,1,2,3`, whole-expert EP.
- Performance-lane base: `1c4280c11b7c33b871cfae218ebb895ecee084ff`.
- Integrated upstream: `origin/main` `d3ac87f80a7b1b94affa1c54ba565a018fb705b6`;
  merge checkpoint `fd26b29e6a6f17b7489a2d5e91468d74b2039046`.
- The generic/proxy rank map was discarded. It is not an admission artifact.

## Masked-MTP corpus law

The rank distribution is formed only from completion text emitted by the served HY3 process.
Prompt text is seed input and is never counted. The corpus server runs the exact W4A16 trunk with
speculation disabled; the request supplies no sampling override, so it exercises the model
configuration served by that Memra process.

A candidate rank map must satisfy all of:

1. at least 131,072 completion tokens after re-tokenizing emitted text with HY3's own tokenizer;
2. deterministic frequency-descending, token-id-ascending tie order;
3. every rank id inside the tokenizer vocabulary;
4. at least 99.0% coverage on the rank-building completion corpus;
5. at least 98.5% coverage on a disjoint held-out HY3 completion corpus.

The 32,768-row candidate was rejected: heldout coverage was 94.76% on the first corpus and
95.68% after 60,634 additional disjoint HY3-served tokens. The smallest admitted width is 81,920
rows: train coverage 100.00%, heldout coverage 98.77%. Intermediate heldout controls were
49,152=96.88%, 65,536=98.01%, and 73,728=98.47%.

The serving candidate is therefore the 81,920-row masked head re-quantized to NVFP4:

```text
MEMRA_FRSPEC_TRIM=<hy3-own-output-ranks-81920.txt>
MEMRA_FRSPEC_TRIM_NVFP4=1
```

The checked-in rank artifacts are bound by:

- GGUF SHA-256 `8bab98c2f86ade8a791eeb3d5c657ba5b05942f721841a2034db8b337a855690`;
- text-rank SHA-256 `4600419c5455bf2008b7644b5614f294a6b5d7796fb92378c59b18c3527d188a`.

BF16-trimmed and untrimmed heads are attribution controls, not the target serving shape.

## Exact kernel admission

The exact candidate hoists invariant NVFP4 row addressing, computes gate and up in one CTA while
sharing the BF16 activation load, and computes adjacent down rows in one CTA. Each output
accumulator must retain its former element order and reduction tree.

Required:

- all-vocabulary first-forward logits bit-identical to the pre-fusion candidate;
- greedy token tape identical;
- MTP self-consistency PASS with nonzero acceptance at K=1 through K=8;
- no non-finite output, CUDA error, Xid, race, or stale-slot result;
- interleaved performance: candidate median must be positive, at least four of five paired wins,
  and no quality or prime regression. A sub-0.5% result is retained only with disjoint
  distributions or kernel-level attribution.

Gate+up fusion is compared first against the pre-fusion binary. Adjacent-row down fusion is then
compared against gate+up fusion, so a losing component is removed rather than hidden in a bundle.

## Internal W4A8 admission

`MEMRA_PARALLEL_EP_Q8_ACT=1` is a separate numeric class and remains default OFF unless every gate
passes. Predeclared full-logit bounds against the exact W4A16 arm, on identical prompt tokens:

- all logits finite;
- argmax equal;
- top-20 overlap at least 18/20;
- cosine at least 0.999;
- RMSE at most 0.25;
- mean absolute error at most 0.10;
- maximum absolute error at most 1.0.

Also require:

- first-token argmax equal on four real prompts;
- symmetric teacher-forced tape checks with mean NLL no more than 1% worse;
- self-consistency PASS and nonzero acceptance at K=1 through K=8;
- masked-NVFP4 MTP engagement in server logs;
- real served-shape c=1 and c=4 rows, long-prompt/cache rows, rollback, and repeated clean boots.

Speed is decided by end-to-end served tok/s, not acceptance alone. Greedy rows are instruments and
looped outputs are excluded from performance aggregates.

## Results so far

- Exact device routing: 23.36 -> 26.80 tok/s median (+14.7%), identical tape. Limiting the
  device-routed width to 32 restored the 34-token prime from 0.483 s to about 0.385 s.
- Exact invariant row walk: green K=1..8 and about +1% decode.
- True gate/up activation-load sharing: green and additive on masked MTP; K1 29.96 -> 30.55,
  K3 22.74 -> 23.37 tok/s.
- Adjacent-row BF16 down: green and additive; K1 30.55 -> 30.73, K3 23.37 -> 23.61.
- Adjacent-row gate/up is admitted only for multi-token widths: using it at t=1 regressed plain
  27.18 -> 26.56. The width-scoped hybrid recovered plain to 27.21 and reached K1 30.85, with
  all 120,832 first-forward logits bit-identical to the two-row baseline.
- Four-row BF16 down and pair-parallel routed pairs were both rejected. Four-row down moved K1
  37.70 -> 36.31 in the final numeric class; pair-parallel c1/c4 was 35.09/34.71 vs the retained
  35.73/34.98 sampled row.
- All-Q8 failed the frozen cosine floor: 0.998650 < 0.999. Down-only failed more strongly:
  cosine 0.996178 and mean absolute error 0.13469.
- Gate/up-only Q8 passed every predeclared logit bound: equal argmax, top-20 overlap 19/20,
  cosine 0.999561, RMSE 0.05654, mean absolute error 0.04479, max absolute error 0.30809.
  Symmetric teacher forcing favored gate/up Q8 on both tapes (mean NLL 0.103942 vs 0.113049 on
  the exact tape; 0.107448 vs 0.115278 on the gate/up tape).
- Masked K=1 gate/up Q8 is stable at 37.33–37.72 tok/s across three fresh process runs, versus
  exact hybrid K1 30.85. Sampled serving measured exact 28.50 c1 / 27.44 c4 aggregate and
  gate/up Q8 35.73 c1 / 34.98 c4 aggregate. First-token identity passed 4/4.
- Prefix cache is correct at 483 prompt tokens: 448 tokens plus the MTP draft plane restored,
  reducing server elapsed 20.52 -> 1.49 s. A 6,553-token request exceeded the 90-second
  first-token deadline and aborted cleanly; long-prompt TTFT remains a blocker.
- Final Nsight K1 attribution after the accepted changes: BF16 expert down 25.5%, prompt-side
  exact gate/up 17.0%, non-expert BF16 projections 16.6%, attention 11.0%, prompt-side exact down
  9.3%, and Q8 gate/up 8.7%.
- Final source checkpoint `4e0b37aedc2f1920794f819a6593d5aa708959a4` passed
  `tools/local-ci.sh --perf`: required real-model kernel coverage, c=64 serving, graph-stress
  canary, cache-hit sampled/rollback gates, and the perf board all finished green.
- The selected rank files and machine receipt are published under `Tiyuvta/Hy3-NVFP4` at Hugging
  Face commit `4e8bbadbdb97b5402cb5a3f997d941946b97c5b5`.

The bank-v2 layout is not a corruption source. The live EP2 defect was the grouped-prefill sktail
`kb+1` prefetch calling `kq_fetch` with its old default `in_f=0`, which selected a packed-code byte
as the V2 scale. This lane has integrated the structural fix from `origin/main`: `in_f` is required
at all ten call sites, and the device-side V1/V2 bank oracle is bit-identical for default, SK128,
SK32, and tail-rollback grouped-prefill arms over both gate/up and down geometries.
