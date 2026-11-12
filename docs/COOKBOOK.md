# Cookbook

Launch commands per model, on the card each was measured on, for an operator who has picked a
model and a GPU. Each block gives the command, the boot line that proves the artifacts attached,
a rollback or alternative, and the model card and receipt behind it.

## Before you paste

- **A block is a measured configuration**: it ran on the named card. Where its receipt used an
  unpublished build, the block names the published equivalent and says so. It is not a support
  state; pack states are in [MODELS.md](MODELS.md), numbers on the model cards and in
  [PERFORMANCE.md](PERFORMANCE.md).
- **Misconfiguration refuses at boot**: a missing or non-file `+draft` path, a drafter that fails
  to load, `MEMRA_DSPARK_DRAFT` without `MEMRA_DSPARK_SPEC=1`, and a retired or unknown `MEMRA_*`
  name under a prefix the binary owns (`[env-audit] REFUSED`). An unknown name outside those
  prefixes only warns, and a crates.io build has no name audit at all
  ([installation](INSTALLATION.md#build-from-source)). The boot line a block names proves which
  artifact attached.
- **`hf:` specs download on first use** into `$MEMRA_MODELS_DIR/hf/` (default
  `~/.cache/memra/models/hf/`). They cannot pin a revision, fetch a `.txt` or load a split GGUF;
  blocks that need one run `hf download` first ([installation](INSTALLATION.md#models)).
- **Responses**: chat completions always use the OpenAI shape. `/v1/completions` uses memra's
  native shape unless `MEMRA_COMPAT=openai` is set, or `MEMRA_COMPAT` is unset and
  `MEMRA_API_KEY` is set. The blocks bind `127.0.0.1:8080` with no key
  ([SERVING.md](SERVING.md)). Flags: [FLAGS.md](FLAGS.md).

## Qwen3.8-27B

Dense Gated DeltaNet hybrid. Trunk, masked heads and ranks: `tiyuvta/Qwen3.8-27B-NVFP4-MTP-GGUF`;
DFlash2 drafter: `tiyuvta/Qwen3.8-27B-DFlash2-memra`. Card: [Qwen3.8-27B](models/qwen38-27b.md).

### RTX 5090 Laptop GPU (24 GB): DFlash2 drafter

The trunk, the DFlash2 drafter (q4, the default) and the agentic 32K ranks, in an 8,192-token,
eight-session envelope measured on the 24 GB laptop part. The desktop card has 32 GB.

```bash
MEMRA_MODELS="q38=hf:tiyuvta/Qwen3.8-27B-NVFP4-MTP-GGUF:Q5K-mtp" \
MEMRA_DSPARK_SPEC=1 \
MEMRA_DSPARK_DRAFT=hf:tiyuvta/Qwen3.8-27B-DFlash2-memra \
MEMRA_FRSPEC_TRIM=hf:tiyuvta/Qwen3.8-27B-NVFP4-MTP-GGUF:q38-ranks-sxc32768.gguf \
MEMRA_CTX=8192 \
MEMRA_MAX_SESSIONS=8 \
MEMRA_PREFIX_CACHE_MB=0 \
memra-server
```

- **Boot proof:** `[worker] q38: DSPARK SPEC route armed (drafter attached (`,
  `[dspark] precision=q4 (MEMRA_DFLASH_PREC unset)` and
  `[dspark] q38: DFlash2 draft head TRIMMED to 32768 rows`.
- **Envelope:** with the prefix cache off, each new turn primes from cold. A DFlash session
  resumes across turns only with the cache on: drop `MEMRA_PREFIX_CACHE_MB=0` and the server sizes
  it from free VRAM ([mechanism](SERVING.md#prompt-caching-cross-request-prefix-cache--2026-08-02)).
- **Rollback:** unset `MEMRA_DSPARK_SPEC` and `MEMRA_DSPARK_DRAFT` together; the embedded MTP
  head then drafts, trimmed by the same ranks.
- **Evidence:** [route and trim on this card](../research/mtp-skip-q38-20260830/VERDICT.md),
  [multi-turn resume](../research/pool-rewind-dspark-20261108/README.md).

### RTX PRO 6000 Blackwell: DFlash2 drafter at full context

The same trunk, drafter and ranks at the checkpoint's own 262,144-token context, with 32 sessions
and a 16 GiB prefix cache.

```bash
MEMRA_MODELS="q38=hf:tiyuvta/Qwen3.8-27B-NVFP4-MTP-GGUF:Q5K-mtp" \
MEMRA_DSPARK_SPEC=1 \
MEMRA_DSPARK_DRAFT=hf:tiyuvta/Qwen3.8-27B-DFlash2-memra \
MEMRA_FRSPEC_TRIM=hf:tiyuvta/Qwen3.8-27B-NVFP4-MTP-GGUF:q38-ranks-sxc32768.gguf \
MEMRA_MAX_SESSIONS=32 \
MEMRA_PREFIX_CACHE_MB=16384 \
memra-server
```

- **Boot proof:** the 5090 block's three lines, and `[dspark] prefix restore switch default OFF`.
  It reads ON on the Max-Q edition, which restores a DFlash session's next turn from the prefix cache
  instead of parking the session.
- **Alternative:** the MTP route with the masked agentic head. Drop the `MEMRA_DSPARK_SPEC`,
  `MEMRA_DSPARK_DRAFT` and `MEMRA_FRSPEC_TRIM` lines and attach the head:
  `MEMRA_MODELS="q38=hf:tiyuvta/Qwen3.8-27B-NVFP4-MTP-GGUF:Q5K-mtp+hf:tiyuvta/Qwen3.8-27B-NVFP4-MTP-GGUF:frspec-sxc32768"`.
  Boot proof: `[worker] q38: regime draft attached (` and `[mtp-draft] external draft head: blk.64`.
- **Evidence:** the `MEMRA_DSPARK_SPEC` and `MEMRA_DFLASH_PREC` rows of [FLAGS.md](FLAGS.md) cite
  the DFlash2 runs on this card; their raw logs are not in this repository. The tag battery runs
  this trunk with its embedded head on every release ([records](../research/release-qualification/)).

### RTX PRO 6000 Blackwell: NVFP4 safetensors checkpoint

The compressed-tensors NVFP4 checkpoint at its measured revision, no GGUF. Its own
`model_mtp.safetensors` head drafts, trimmed at load by the agentic ranks `.txt`.

```bash
hf download unsloth/Qwen3.8-27B-NVFP4 --revision 16b6615af3548b88e2d8e382457bc705b00479cf \
  --local-dir "$HOME/models/Qwen3.8-27B-NVFP4"
hf download tiyuvta/Qwen3.8-27B-NVFP4-MTP-GGUF q38-ranks-sxc32768.gguf.txt \
  --local-dir "$HOME/models/q38-ranks"
MEMRA_MODELS="q38=$HOME/models/Qwen3.8-27B-NVFP4" \
MEMRA_FRSPEC_TRIM="$HOME/models/q38-ranks/q38-ranks-sxc32768.gguf.txt" \
memra-server
```

- **Boot proof:** `[frspec-trim] self-trimmed head: 32768 rows of`.
- **Rollback:** unset `MEMRA_FRSPEC_TRIM` and the full head drafts. The official FP8 checkpoint is
  a different artifact; this block does not cover it.
- **Evidence:** no receipt in this repository covers this exact recipe; the earlier block is in
  [history](history/COOKBOOK.md#nvfp4-safetensors-trunk--ranks-trim).

## Qwen3.5-9B

Dense Gated DeltaNet hybrid, from the community GGUF `michaelw9999/Qwen3.5-9B-NVFP4-MTP-GGUF`
with an embedded MTP head. Card: [Qwen3.5-9B](models/qwen35-9b.md).

### RTX 5090 Laptop GPU (24 GB): embedded MTP head

The GGUF on one card; its embedded head drafts with no flag needed.

```bash
MEMRA_MODELS="q9=hf:michaelw9999/Qwen3.5-9B-NVFP4-MTP-GGUF" \
MEMRA_CTX=8192 \
MEMRA_MAX_SESSIONS=4 \
MEMRA_PREFIX_CACHE_MB=2048 \
memra-server
```

- **Boot proof:** `[worker]   loaded "q9": 33 layers`, then on the first short request a
  `[spec-k] model="q9"` line with `K=3`. Once the head has acceptance history, the
  boot-calibrated cost model picks K per request ([record](decisions/SPEC-K-COLD-SHORT-5090.md)).
- **Alternative:** `MEMRA_SPEC_K=0` pins plain decode. The receipt covers this 8,192-token
  envelope only; size a larger `MEMRA_CTX` before raising it.
- **Evidence:** [five serving cells](../research/modelplan-onboarding-20261002-qwen35-9b/RESULTS.md)
  on this card with this environment (2026-10-02).

## Ornith-1.5-35B-A3B

Mixture of experts on the Qwen3.5 MoE stack: the NVFP4+Q5_K GGUF with a continued-trained MTP head
in `tiyuvta/Ornith-1.5-35B-A3B-NVFP4-MTP-GGUF`. Card:
[Ornith-1.5-35B-A3B](models/ornith15-35b-a3b.md).

### RTX PRO 6000 Blackwell: MTP head with the ranks trim

The trunk at its full 262,144-token context. The own-generated ranks trim the embedded head's
draft vocabulary at load, the adopted default since commit `412c45b0ca`.

```bash
MEMRA_MODELS="ornith=hf:tiyuvta/Ornith-1.5-35B-A3B-NVFP4-MTP-GGUF:Q5K-mtp" \
MEMRA_FRSPEC_TRIM=hf:tiyuvta/Ornith-1.5-35B-A3B-NVFP4-MTP-GGUF:ornith15-ranks-owngen-32768.gguf \
memra-server
```

- **Boot proof:** `[frspec-trim] self-trimmed head: 32768 rows of`.
- **Rollback:** unset `MEMRA_FRSPEC_TRIM`; the untrimmed head drafts, still exact.
- **Alternative:** the standalone masked head. Drop `MEMRA_FRSPEC_TRIM` and append
  `+hf:tiyuvta/Ornith-1.5-35B-A3B-NVFP4-MTP-GGUF:frspec-owngen32768` to the model spec. Boot proof:
  `[worker] ornith: regime draft attached (`.
- **Evidence:** [the trim adoption cells](../research/orndecode-20260822/RESULTS.md); the tag
  battery runs this trunk on every release ([records](../research/release-qualification/)).

## Gemma-4 31B

Dense, with image input. Its assistant drafter is a separate small model that reads the trunk's KV
cache, not an MTP head. Card: [Gemma-4 31B](models/gemma4-31b.md).

### RTX PRO 6000 Blackwell: assistant drafter and image input

A Q6_K trunk of the official weights, the official-lineage assistant drafter (K=5, armed by
`MEMRA_DRAFT` alone) and the Gemma vision tower, at the checkpoint's 262,144-token context.

```bash
M="$HOME/models/gemma-4-31B"
hf download unsloth/gemma-4-31B-it-GGUF gemma-4-31B-it-Q6_K.gguf --local-dir "$M"
hf download tiyuvta/gemma-4-31B-it-assistant-MTP-GGUF gemma-4-31B-it-official-NVFP4-MTP.gguf --local-dir "$M"
hf download google/gemma-4-31B-it-qat-q4_0-gguf gemma-4-31B-it-mmproj.gguf --local-dir "$M"
MEMRA_MODELS="gemma31=$M/gemma-4-31B-it-Q6_K.gguf" \
MEMRA_DRAFT="$M/gemma-4-31B-it-official-NVFP4-MTP.gguf" \
MEMRA_GEMMA_VISION=1 \
MEMRA_GEMMA_MMPROJ="$M/gemma-4-31B-it-mmproj.gguf" \
MEMRA_MAX_SESSIONS=16 \
MEMRA_PREFIX_CACHE_MB=16384 \
NVIDIA_TF32_OVERRIDE=0 \
memra-server
```

- **Boot proof:** `[worker] gemma31: GEMMA SPEC route armed (K=5, assistant drafter attached (`
  and `[gemma-vision] tower loaded from`.
- **TF32 off:** the vision tower's f32 GEMMs go through cuBLASLt with TF32 allowed, and no code
  turns TF32 off for this tower. Its parity gates ran with `NVIDIA_TF32_OVERRIDE=0`; with TF32
  allowed, the tower read min-cos 0.9878 against the reference, because its unscaled attention
  amplifies the rounding over 27 blocks ([receipt](../research/gemma-vision-20260816/REPORT.md)).
  The variable is process-wide. The same receipt found the language model's path dominated by
  quantized kernels, not cuBLASLt f32, so TF32 is not its noise source. A text-only request never
  runs the tower.
- **Lineage:** a drafter must come from its trunk's weights. On the QAT Q4_0 trunk
  (`google/gemma-4-31B-it-qat-q4_0-gguf`) this official-lineage head stays exact but measured about
  half the acceptance; that trunk wants the QAT-lineage assistant, which memra does not publish
  ([receipt](../research/gemma-vision-20260816/ASSISTANT-ARM-RESULTS.md)).
- Do not set `MEMRA_VISION_DIR`: it loads the Qwen tower, and one vision family runs per process.
- **Rollback:** `MEMRA_GEMMA4_SPEC=0` serves plain decode. `MEMRA_DRAFT` also changes three engine
  defaults by its presence ([FLAGS.md](FLAGS.md) row).
- **Evidence:** [the served drafter route](../research/gemma-vision-20260816/SERVED-SPEC.md) and
  the drafter's [Hugging Face card](https://huggingface.co/tiyuvta/gemma-4-31B-it-assistant-MTP-GGUF).
  They ran unpublished builds of the official weights (an NVFP4 mix, a Q6_K mix), and the served
  route also trimmed the drafter head with an unpublished ranks file. unsloth's Q6_K is a published
  build of the same weights ([earlier block](history/COOKBOOK.md#gemma-4-31b)).

## Step-3.7-Flash 196B-A11B

Mixture of experts too large for one card: the IQ4_XS GGUF and the separate Q8_0 MTP head in
`stepfun-ai/Step-3.7-Flash-GGUF`. Card: [Step-3.7-Flash](models/step37-flash.md).

### 2x RTX PRO 6000 Blackwell: PP-2

The three IQ4_XS shards (105 GB) split across two cards with the MTP head attached. PP-2 serves
plain batched decode by default.

```bash
S="$HOME/models/step-3.7-flash"
hf download stepfun-ai/Step-3.7-Flash-GGUF --revision 0b69336d2fd2adfdef9c66e425f7778196c31482 \
  --include "IQ4_XS/*" --include "Step3.7-flash-mtp-Q8_0.gguf" --local-dir "$S"
MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 \
MEMRA_MODELS="step=$S/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf+$S/Step3.7-flash-mtp-Q8_0.gguf" \
memra-server
```

- **Boot proof:** `[pp] cross-device transport: stage0=dev0 stage1=dev1`,
  `[worker] step: regime draft attached (` and
  `[spec-gate] policy placement=pp2-cross-device LOW=0 HIGH=1 source=placement-default spec-admission=off`.
- **Why plain:** speculative decoding over PP-2 is exact but measured slower than plain batching,
  so the placement default admits no request to it; `MEMRA_SPEC_GATE=0` restores always-spec for
  measurement ([PP-2 serving](SERVING.md#pipeline-parallel-pp-2-serving)).
- **Alternative:** drop the `+$S/Step3.7-flash-mtp-Q8_0.gguf` part to save 3.7 GB. The server
  then warns `no MTP drafter attached` and serves the same plain decode.
- **Evidence:** [last gate on this artifact over PP-2](../research/modelplan-onboarding-537-20260919/native-20260920/RESULTS.md)
  (2026-09-20); [the split adds zero deviation](../research/pp2-batch-20260806/RESULTS.md);
  [the head attach](../research/step-draft-20260807/RESULTS.md).

## Everything else

- Models without a block serve through the generic `MEMRA_MODELS` forms
  ([installation](INSTALLATION.md#models)); their cards say what was measured.
- Earlier blocks and their numbers: [history](history/COOKBOOK.md). A new block comes in through a
  [hardware validation report](https://github.com/avifenesh/memra/issues/new?template=hardware-validation.md).
