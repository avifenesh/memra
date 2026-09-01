# DSpark Phase 2: 2K smoke trajectory result

Date: 2026-08-11

Lane: `lane/cx-dspark2`

Rigs: local RTX 5090 Laptop for memra corpus generation; dedicated L40S for training

Contract: `research/dspark2-spec-20260811/SPEC.md`

## Verdict

**KILL — do not spend the 30K/Step-scale continuation or integrate this checkpoint.**

The owner-ordered 2K smoke probe crossed every frozen absolute kill threshold after five
epochs:

- held-out temp=0.7 analytical two-deep full survival was **q2=0.1043**, versus the
  `q>=0.60` continuation threshold and `<0.55` kill threshold;
- slot-1 and conditional slot-2 analytical acceptance were **0.3209 / 0.3250**, versus
  the `0.75 / 0.65` no-depth kill floors;
- Sequential Temperature Scaling left cumulative prefix ECE at **7.14%**, versus the
  `>3%` calibration kill threshold;
- fixed-seed sparse rejection sampling agreed at **q2=0.0991**; greedy was better but
  still only **q2=0.1321**.

The final 30% of training moved analytical q2 from 0.0975 at step 900 to 0.1043 at step
1,175: +0.0068 absolute, or a least-squares slope of +0.00253 per 100 steps. That is a
low plateau, not a trajectory toward 0.60–0.70. Confidence ranking itself was strong
(mean cumulative AUROC 0.9390), but its absolute probabilities failed calibration, so
the confidence head has no standalone admission value in this checkpoint.

The cooperative corpus stop sentinel was placed while the active 64-pair window finished.
That window transferred and committed successfully, then both scale services exited or were
stopped cleanly at the exact remotely verified **pair-2,192 boundary**. Training and all
reported metrics used only frozen pairs 0..1,999; pairs 2,000..2,191 never affected ranks,
weights, or evaluation.

This is an intentional staged early stop on the 2K smoke, not evidence that a completed 30K
training run could never improve. Resuming the larger study would be an owner override of the
frozen kill rule; it would require removing the durable stop sentinel, completing the corpus,
refreezing ranks from training responses only, re-exporting shared rows, and training a new
artifact. The 2K checkpoint must not be presented as a 30K-pilot result.

## Frozen inputs

| Input | Frozen value |
|---|---|
| Target GGUF | `Qwen3.5-9B-NVFP4-MTP-GGUF.gguf` |
| Target SHA-256 | `52c9cceb190055e0591a9a30c21f7200572eaf3ff1c59f6e9a1eda838a8f39de` |
| Generation | chat template on, temperature 0.7, max 512 tokens |
| Prompt source | Open-PerfectBlend revision `af60f3c18201652a83a93f46fcfee1b646ba3df7` |
| Prompt text SHA-256 | `20a061ddee54bb3113a25cd2abbb150e7e51c65a05670c3be127c242221fffd9` |
| Stratified assignment SHA-256 | `c2f9504c5761de8bfd88657433e259496240e795ec60cf709e4202d76de58e7f` |
| Training/eval pair range | exact ids 0..1,999 |
| Corpus fingerprint | `48c8b59f1beb85bae80099052f789183fbf4f7c67fcaf691d93b5990e9a00cb7` |
| Anchor records | 7,516 train / 424 heldout |
| Heldout documents | 106, above the 45-document floor |
| d2t rank SHA-256 | `0fdeea0b79f58be3103978b729d5a0d741308e492da9733f2d9d49407281b238` |
| Rank construction | training response tokens only; exact pair range 0..1,999 |
| Token coverage | 99.9272% train / 97.4296% heldout |

The corrected prompt assignment populated all 16 category/mode/split cells. The 2K slice was
500 prompts per category, 988 thinking / 1,012 non-thinking, and 1,894 train / 106 heldout.
Fifteen early-EOS prompts produced no valid gamma-5 anchor; no partial or fabricated records
were used. The aggregate contains 7,940 records, and the sampled target token is present in the
stored target top-64 for 99.9849% of positions.

The first 448-pair `smoke` corpus remains preserved but is explicitly superseded: its periodic
assignment coupled category, mode, and heldout membership. Only the hash-stratified `pilot`
label enters this result.

## Shared-weight and model receipt

The frozen target rows were dequantized directly from the deployed GGUF, in the 32,768-entry
draft-to-target order:

| Table | Source tensor | Source qtype | Export shape / encoding |
|---|---|---|---|
| embedding | `token_embd.weight` | NVFP4 | 32,768 x 4,096 BF16-le |
| LM head | `output.weight` | Q6_K | 32,768 x 4,096 BF16-le |

Each table is 268,435,456 bytes. The shared artifact verifies locally and on the L40S; its
manifest SHA-256 is `38c5601ff84385aa533453613bd2d58fc674710efb6e4f4ffa772100d1478836`.
Frozen tables are non-persistent buffers and are not duplicated inside checkpoints.

The pilot has **318,801,153 trainable parameters**:

- two dense d=4,096 transformer layers;
- 16 query / 4 key-value heads, head dimension 256, bias-free GQA;
- Qwen3.5 text RoPE theta 1e7 with the declared 0.25 partial rotary factor;
- RMSNorm epsilon 1e-6, SwiGLU FFN dimension 8,192;
- one predecessor carrier plus five mutually visible noise positions, wholly inside SWA-128;
- fresh backbone initialization and a dedicated trainable mask latent;
- vanilla rank-256 Markov `W1/W2` correction over the 32,768-row draft vocabulary;
- confidence linear over `[h_k; W1[x_(k-1)]]`.

The current official Qwen3.5 tokenizer has no mask token. A trainable mask latent avoids
fabricating a target id while keeping actual token embedding/head rows frozen and exact.

The objective was frozen as:

```text
0.1 * CE + 0.9 * TVD + 1.0 * confidence-BCE
w_k = exp(-k / 5), k=0..4
```

Draft softmax always covers all 32,768 rows. Target top-64 probability retained inside the trim
is scattered without renormalization; target tail and top-64 ids outside the trim enter one
non-winnable escape bucket. CE is evaluated only for in-trim sampled labels, while TVD prices
every escape miss.

The frozen SPEC contains two incompatible confidence/acceptance phrasings. The implementation
uses the probability identity and current DeepSpec behavior:

```text
acceptance = 1 - TVD = 1 - 0.5 * L1
```

It does not apply a second factor of one-half to an already-halved TVD.

## Environment and preflight

The L40S environment was:

- NVIDIA L40S, sm_89, 46,068 MiB;
- Python 3.12.3;
- PyTorch 2.11.0+cu128;
- accelerate 1.14.0, bitsandbytes 0.50.0, datasets 4.3.0, transformers 5.5.0;
- BF16 CUDA matmul and causal SDPA PASS; `pip check` PASS.

The requested flash-attn 2.7.8 distribution was unavailable from the official package index,
so the recorded attention backend is PyTorch SDPA. This is not silently reported as FA2.

Before training, one full-size real 32-anchor batch passed forward, full-row loss, backward,
gradient clipping, and fused AdamW update. It took a single 0.456-second observation and peaked
at 5,680,248,320 allocated bytes. This is a fit/correctness receipt, not a throughput claim.

## Training receipt

| Field | Value |
|---|---|
| L40S service invocation | `a147b9c9c01146248335c53900cddb19` |
| Code commit recorded by trainer | `468ac177` |
| Seed | 20260811 |
| Epochs / optimizer steps | 5 / 1,175 |
| Batch size | 32 anchor blocks |
| Optimizer | fused AdamW, weight decay 0 |
| LR | 3e-4, 4% warmup, cosine decay |
| Gradient clipping | max norm 1.0 |
| Precision | FP32 trainable weights, BF16 autocast and frozen shared rows |
| Final elapsed time | 103.36 seconds including evaluation and atomic checkpoint writes |
| Service result | success, inactive/dead; L40S idle after exit |
| Final checkpoint | `/home/ubuntu/dspark2/runs/pilot-2k-468ac177/checkpoint-final.pt` |
| Checkpoint bytes | 3,825,653,493 |
| Checkpoint SHA-256 | `2b110c46c75b7470cac6898fddd5c19684f93c66cc68b2ffdf49cadb530bc620` |

The raw ledger contains 118 train rows, 12 intermediate held-out points, the final held-out
evaluation, source/config hashes, service stdout, and an artifact manifest. The large checkpoint
stays durably on the L40S and is represented in git by its size and SHA-256.

## Held-out q trajectory

Analytical q is the decision metric because it integrates the complete 32,768-row draft
distribution against the stored sparse teacher plus escape. Sampled q uses one fixed seed and
one sparse-rejection draw per position. Greedy is reported separately as required.

| Step | Epoch | temp=.7 analytical q2 | temp=.7 sampled q2 | greedy q2 | analytical slot 1 / 2 | STS ECE mean |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.00 | 0.0000 | 0.0000 | 0.0000 | 0.0002 / 0.0004 | 22.76% |
| 100 | 0.43 | 0.0052 | 0.0071 | 0.0071 | 0.0872 / 0.0601 | 0.38% |
| 200 | 0.85 | 0.0280 | 0.0283 | 0.0448 | 0.1555 / 0.1800 | 1.67% |
| 400 | 1.70 | 0.0522 | 0.0542 | 0.0708 | 0.2129 / 0.2453 | 2.13% |
| 600 | 2.55 | 0.0612 | 0.0684 | 0.0849 | 0.2318 / 0.2640 | 3.41% |
| 800 | 3.40 | 0.0892 | 0.0896 | 0.1061 | 0.2796 / 0.3190 | 3.73% |
| 900 | 3.83 | 0.0975 | 0.0943 | 0.1274 | 0.3024 / 0.3225 | 2.49% |
| 1,000 | 4.26 | 0.1012 | 0.0991 | 0.1321 | 0.3174 / 0.3188 | 7.22% |
| 1,100 | 4.68 | 0.1039 | 0.0967 | 0.1297 | 0.3205 / 0.3242 | 6.71% |
| **1,175** | **5.00** | **0.1043** | **0.0991** | **0.1321** | **0.3209 / 0.3250** | **7.14%** |

The loss learned substantially on training batches, but the held-out first two positions
stopped improving materially in the final epoch. That is the relevant generalization result;
high individual training-batch q values are not substituted for held-out q.

## Final temperature cells

| Position | temp=.7 analytical conditional | temp=.7 sampled conditional | greedy conditional | analytical prefix survival |
|---:|---:|---:|---:|---:|
| 1 | 0.3209 | 0.3255 | 0.3491 | 0.3209 |
| 2 | 0.3250 | 0.3043 | 0.3784 | **0.1043** |
| 3 | 0.4150 | 0.3333 | 0.4286 | 0.0433 |
| 4 | 0.4655 | 0.5000 | 0.5417 | 0.0201 |
| 5 | 0.2905 | 0.1429 | 0.3077 | 0.0059 |

Sampled deeper-position conditionals have small surviving denominators and are not used to
override the all-record analytical kill decision.

## Confidence calibration

Sequential Temperature Scaling fits positions left-to-right to cumulative prefix survival,
keeping earlier fitted temperatures fixed.

| Prefix position | fitted T | calibrated ECE | cumulative AUROC |
|---:|---:|---:|---:|
| 1 | 0.77 | 17.82% | 0.8606 |
| 2 | 0.35 | 9.94% | 0.9265 |
| 3 | 0.36 | 4.85% | 0.9368 |
| 4 | 0.86 | 1.95% | 0.9736 |
| 5 | 0.25 | 1.13% | 0.9976 |
| **mean** | — | **7.14%** | **0.9390** |

STS is fit and reported on the same heldout split, as specified for this bounded probe. That is
an optimistic in-sample calibration check; failing it does not need a second split to establish
the kill.

## Frozen gate accounting

| Gate | Threshold | Result | Verdict |
|---|---:|---:|---|
| Continue q2 | >=0.60 and rising | 0.1043, low plateau | FAIL |
| Continue slot 1 / 2 | >=0.80 / >=0.75 | 0.3209 / 0.3250 | FAIL |
| Kill q2 | plateau <0.55 over final 30% | 0.0975 -> 0.1043 | **TRIGGERED** |
| Kill slot 1 | <0.75 | 0.3209 | **TRIGGERED** |
| Kill slot 2 | <0.65 | 0.3250 | **TRIGGERED** |
| Confidence ranking | AUROC >=0.80 | 0.9390 | PASS |
| Kill confidence calibration | ECE >3% | 7.14% | **TRIGGERED** |
| Dominant temp=.7 acceptance | >=75% | slot 1 =32.09% | FAIL |

The committed current NextN context is q2=0.381 and K=1 acceptance 72.97–73.68%, but this lane
did not replay that head on the identical 106-document sparse corpus. It therefore makes no
protocol-identical baseline claim. The new head is already below the absolute frozen kill bars,
so an engine baseline replay cannot reverse the decision.

## Work deliberately not run

- No Torch-vs-engine DSpark agreement or attach conversion was built after the offline kill.
- No in-engine attach parity, `kernel-check`, `run-gen`, or `run-spec` promotion battery was run.
- No 5090 or PRO 6000 end-to-end performance cell was run.
- No board number, runtime default, release tag, or GGUF delivery surface changed.

Those are integration gates after an offline win. Running them for a q2=0.104 checkpoint would
spend shared-rig time without a promotion path. The orchestrator's 5090 210–1,200 MHz thermal
clock cap remains in place; no capped-clock output is presented as memra performance.

## Limits

- This is a 2K smoke early-stop, not the complete 30K-pair study.
- Sparse target top-64 probabilities are exact at temperature 0.7; all remaining target mass is
  a conservative non-winnable escape because its token-wise distribution was not stored.
- The analytical metric uses teacher-forced target paths. Recursive Markov proposals are used
  in sampled/greedy cells, but teacher distributions after an off-path rejected proposal are
  unavailable by construction; prefix survival stops the decision at rejection.
- The fixed-seed sampled cell is one stochastic draw over 424 anchor blocks, not an N-run
  sampling study. It corroborates, rather than replaces, analytical q.
- The learnable mask latent is a bounded adaptation forced by the absence of a Qwen3.5 mask
  token. It is not a published target-token embedding.

## Evidence index

- environment: `raw/l40s-{host-probe,env-setup,env-smoke,env-freeze,env-artifacts}.*`
- prompt/corpus validation: `raw/prompt-pack-v2-*`, `raw/corpus-pilot-summary-02000.json`
- final 2K ranks: `raw/ranks-pilot-02000.log`
- current external references: `raw/live-reference-20260811.log`
- shared export: `raw/{build,export}-dspark-shared.log`,
  `raw/shared-artifact-{local,l40s}.log`
- model/loss/data gates: `raw/test-dspark-*.log`
- full-size preflight: `raw/preflight-dspark-l40s.log`
- service launch/exit: `raw/train-service-{launch,final}.log`
- full training ledger: `raw/training-pilot-2k/`
- cooperative kill boundary: `raw/kill-stop-receipt.log`,
  `raw/corpus-pilot-summary-02192.json`
