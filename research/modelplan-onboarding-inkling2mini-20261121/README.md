# Inkling 2 Mini onboarding (opened 2026-11-21)

Status: open. Step 1 is in: the `inkling2` pack parses the config and the quantization
declaration, compiles a typed 48-slot plan and binds every one of the 1,298 header tensors.
Config, TokenizerTemplate and TensorCensus pass at inspect ([Step 1](#step-1-config-plan-and-census)).
Steps 3 to 5 are in: the reference executor runs the plan on CPU and TinyParity passes
([Steps 3 to 5](#steps-3-to-5-reference-executor-and-tinyparity)), so the pack is at the
reference state (record `inkling2-reference`). Every tuned execution surface still refuses the
plan, and CheckpointParity, RewriteParity and Serve are pending. No GPU was used.

## Scope

Onboard `thinkingmachines/Inkling-2-Mini` as a native memra family (`inkling2`, aliases
`inkling2_loop` and `inkling-2-mini`) and tune it on one 24 GB sm_120a card.

- Text only first. Image parts are refused until a vision lane; the vision tower's 422 tensors
  are still owned by the tensor contract.
- The weights stay as shipped: NVFP4 experts on layers 0 to 19, MXFP8 experts on layers 20 to 23,
  BF16 and F32 elsewhere. No requantization.
- The receipts of later steps (inspect, tiny, checkpoint, serve) land in this directory.

## Sources

[`sources.lock.json`](sources.lock.json) pins every source:

| Source | Pin | Role |
|---|---|---|
| Checkpoint | `thinkingmachines/Inkling-2-Mini@ed40e7da625f594fc9c4990f5f66be24873c22d7`, 12 files with sha256 and size | the semantic source |
| llama.cpp | `29b52d398b294509746ec0a3f28948a3107c0213` (`src/models/inkling2.cpp`, `inkling2-loop.cpp`) | second implementation of the float-activation trunk, for triage |
| transformers | PR 51050 at head `6173a2f679ff4ed393afc499aefc569da7e68bc1` (open at capture) | oracle source for the float-activation bring-up class |
| vLLM image | `vllm/vllm-openai:inkling-2-mini`, index digest `sha256:c2dc759f...`, with the 30 files read and their sha256 | only public source of the drafter and of the declared activation-quantized class, the default |

External engines are oracles and baselines. Their results are never memra support evidence.

## Census

[`census.py`](census.py) reads the safetensors headers and `hf_quant_config.json` of a local
checkpoint directory. No weight bytes are read.

```bash
python3 research/modelplan-onboarding-inkling2mini-20261121/census.py <checkpoint-dir> \
  research/modelplan-onboarding-inkling2mini-20261121
# 1298 tensors, 109 classes
```

- [`census.tsv`](census.tsv): one row per tensor (file, name, dtype, shape, bytes, declared
  quant group). Rows sha256 `3752c60c4d5facbc9cbd089e76b710c2f1523c794ffc3776c2628dde9f3aa41c`.
- [`census-classes.tsv`](census-classes.tsv): the same census as 109 name classes. The script
  expands the classes and refuses to write unless they reproduce every row.
- [`census-groups.tsv`](census-groups.tsv): the 6 parameter groups of `hf_quant_config.json`
  with every declared field (`compute_dtype`, `output_dtype`, `fwd_output_dtype`,
  `weight_transform`, `weight_block_shape`, `weight_scale_format`,
  `required_activation_quantization`) beside what the headers store for the group. The script
  refuses a group key it does not know. One row shows a declaration that differs from storage:
  the MXFP8 group declares `[32, 32]` and stores one scale per row per 32 inputs.

| File | Tensors | Data bytes |
|---|---:|---:|
| `model-00001-of-00001.safetensors` (the one indexed shard) | 1,244 | 19,525,870,752 |
| `mtp.safetensors` (drafter, not in the index) | 54 | 635,795,744 |

| Class | Tensors | Bytes | Dtype |
|---|---:|---:|---|
| NVFP4 expert payload, layers 0-19 (`w13 [128,2304,1152]`, `w2 [128,2304,576]`, two codes per byte) | 40 | 10,192,158,720 | U8 |
| NVFP4 expert scales (one e4m3 per 16 inputs per row, `_amax [128]`, `_outer_scale_denominator [1]`) | 120 | 1,274,040,480 | F8_E4M3, F32 |
| MXFP8 expert payload, layers 20-23 (`w13 [128,2304,2304]`, `w2 [128,2304,1152]`) | 8 | 4,076,863,488 | F8_E4M3 |
| MXFP8 expert scales (one u8 exponent per 32 inputs per row) | 8 | 127,401,984 | U8 |
| attention projections (`wq_du`, `wk_dv`, `wv_dv`, `wo_ud`, `gate_attn`, `wr_du`) | 144 | 1,288,175,616 | BF16 |
| embed and unembed `[201024,2304]` | 2 | 1,852,637,184 | BF16 |
| vision tower and projector (owned, not executed) | 422 | 636,512,256 | BF16 |
| loop controllers and the two cross-pass query tensors | 146 | 57,065,472 | BF16, F32 |
| router weight and bias | 48 | 14,168,064 | BF16, F32 |
| norms, clips, sink, relative bias | 154 | 3,873,792 | BF16 |
| short convolutions | 104 | 2,457,600 | BF16 |
| block-attention-residual queries | 48 | 516,096 | F32 |

## Decisions

1. Text only first; image parts fail closed.
2. Weights stay as shipped.
3. The default numeric class, and the only support target, is the program the checkpoint
   declares as required in `hf_quant_config.json`: NVFP4 W4A4 with per-token amax and
   four-over-six scale selection on the layer 0-19 experts, MXFP8 W8A8 on the layer 20-23
   experts. Its only public source is the vLLM image. The `float-activation` class (exact code
   times scale dequantization with f32 or bf16 activations, what the transformers PR and
   llama.cpp run) is a bring-up and oracle step: never the default and never a support claim.
   The two classes are named and never presented as one another. No GGUF-style quant formats
   (Q*_K, IQ*) enter this family's path.
4. The base is `main`. Nothing stacks on unmerged branches.
5. The family is `inkling2`; receipts live here.

## Semantics

[`SEMANTICS.md`](SEMANTICS.md) states the program as read from the config, the two reference
implementations and the bytes, and lists the facts that are still open.

6. No quantization choice is made without a per-layer KL receipt. The plan carries the numeric
   class per expert layer so a measurement can set it layer by layer.
7. The checkpoint defines the program. Reference code (the vendor's serving image, the
   transformers PR, llama.cpp) is something to check against and is copied only where a test
   proves it exact.

## Step 1: config, plan and census

Code: `crates/memra-gguf/src/config/inkling2.rs`, `quant_groups.rs`, `model_plan/inkling2.rs`,
`model_packs/inkling2/`.

- **Config.** `Arch::Inkling2Loop` (`inkling2_loop`, `inkling2_loop_text`). Every key of the
  text, loop, vision, drafter and quantization blocks is required by name, and a topology the
  program cannot run is refused (loop bounds off the residual block, a variable loop, shared
  experts, a second attention geometry for sliding layers, a drafter head other than
  `rnn_hidden`, zero sizes, a hidden or expert width that is not whole 32-input scale blocks).
  The loop is bounded at 1,024 cache slots with checked arithmetic. The drafter's taps must be
  at least two increasing layer visits ending at the visit count, which names the final hidden,
  and its projections must carry no bias.
- **Quantization declaration.** `quant_groups.rs` reads the `parameter_groups` form of
  `hf_quant_config.json`. `required_activation_quantization` is typed and must be present in
  every group (`null` for a float input). A quantized group without a quantizer, an unknown
  format, rule, rounding or key is refused. `weight_transform`, `output_dtype` and
  `weight_block_shape` are typed per compute dtype and anything else is refused. The MXFP8
  group's `[32, 32]` has one explicit reading, decided from the bytes: the stored scales are one
  exponent per row per 32 inputs ([`mxfp8-scale-grid.tsv`](mxfp8-scale-grid.tsv), 0 of 73,728
  aligned 32-row groups uniform; SEMANTICS.md). The plan does not compile until the file is
  attached, and it refuses a declaration that leaves a checkpoint tensor in no group or names a
  tensor the config does not derive.
- **Plan.** 48 layer entries (cache slots) over 24 weight sets, with the slot to physical-layer
  map, the pass and the execution order on `ModelPlan::inkling2.loop_plan`. 8 full-attention
  slots (5, 11, 17, 23, 29, 35, 41, 47) and 40 sliding. New typed parts: relative-bias attention
  with a per-query sink, L2 clips, the log position scale, the separate per-dimension gate,
  static and dynamic short convolutions with per-slot state, the block attention residual, the
  loop and its controllers, softcapped SwiGLU, the embedding norm and logit truncation.
- **Numerics, per layer.** Each physical layer's experts carry their stored format (NVFP4 with a
  per-expert amax on layers 0-19, MXFP8 on 20-23), the activation quantizer their group requires
  and the input mode this plan runs them on (`Declared` or `Float`). The default plan is
  `Declared` on all 24 layers, which is the declared class. `plan_with_expert_activation`
  compiles any other assignment as a measurement plan: one float layer makes the class
  `PerLayerMixed`, with its own plan text and hash, so a per-layer study names exactly what it
  ran. The dense tensor classes (attention projections, router, controllers, convolutions, norms,
  residual queries, embedding, output projection) carry their declared dtype per class; the plan
  refuses a declaration that quantizes any of them or declares one class two ways.
- **Fail closed.** 20 new operations, all with no tuned surface in the registry
  (`docs/EXECUTION-SURFACES.md`). One of them, logit truncation, has a tested implementation in
  the reference executor; the other 19 have none. Every rewrite in `inspect/execution-rewrites.tsv` is
  ineligible and names its blockers. The reference executor refuses the attention, the residual
  and the activation by name. The drafter is declared (`DsparkRnnDrafter`), owned by the contract
  and executed by nothing. No pipeline cut is declared, so the placement files read `failed`.
- **Contract.** The pack derives all 1,298 tensors from config math and the quantization
  declaration; a test compares the derivation with `census-classes.tsv` name by name. The 422
  vision tensors and the 54 drafter tensors are owned like trunk tensors. `mtp.safetensors` is
  outside the shard index, so the pack names it as an auxiliary file and its tensors enter the
  census as `mtp.*`. The safetensors census folds the 128 quantization planes into their 48
  expert banks (`NVFP4_AMAX` block 16, `MXFP8_E4M3` block 1x32), which leaves 1,170 census rows.

### Inspect receipts

```bash
memra model inspect inkling2-mini --against inkling2 --out inspect      # the local checkpoint directory
memra model inspect thinkingmachines/Inkling-2-Mini@ed40e7da625f594fc9c4990f5f66be24873c22d7 \
  --against inkling2 --out inspect-remote
# family=inkling2 tensors=1170 (both)
memra model verify config <either source> --against inkling2
# family=inkling2 verified=Config
```

Both forms write the same lock values:

- `config_sha256=a7a349601f0984c4abf999c480ef775d8567d21f5f84b4e8a3e035b44f288dc2`
- `census_sha256=c6a9b0a9e34961ba754937c6ad40c02e392e9373633b1536bbb4b195c6fec0be`
- `plan_sha256=b66d8356c1cacd7751c14935f6d3fbafca04549df203032c6e177496435daf7a`
- `binding=passed`, `tokenizer=passed`, `dialect=artifact`
- `quant_config_sha256=7c656207d3ce37879c7be209a036b0dde43bf01f78159bff06880b868ed57143`
  (`hf_quant_config.json`; the plan text carries the same hash, so the plan hash moves with it)

[`inspect/`](inspect/) is the full bundle. [`inspect-remote/`](inspect-remote/) keeps its lock
and gates; its other files were byte-identical to the local ones
(`inspect-remote/SAME-AS-LOCAL.sha256`).

What the three passed gates mean here:

- Config: the plan compiles from the config and the quantization declaration.
- TensorCensus: every header tensor binds, with nothing missing, extra, reshaped or in another
  quant layout. A plane-suffixed tensor (`_amax`, `_outer_scale_denominator`) beside a
  `*_weight` tensor that has no `_scale` is not folded away: it stays a census row and the
  contract refuses it.
- TokenizerTemplate: the artifact carries `tokenizer.json` and a chat template, with their
  hashes in the lock. It is not token-id or rendered-byte parity: memra has no pre-tokenizer or
  renderer for this family yet, and that work is its own step.


## Steps 3 to 5: reference executor and TinyParity

Detail, oracles, the table of proven facts and the list of where the references differ:
[`reference/README.md`](reference/README.md). Receipts of the tiny gate: [`tiny/`](tiny/).

- `crates/memra-reference/src/inkling2.rs` and `inkling2/ops.rs` run every operation of the plan
  except the drafter, unfused, in f64, from the stored values. A layer's experts run on float
  inputs or on the declared activation quantizer (`inkling2_actquant.rs`), per physical layer,
  as the plan says.
- 16 operations are each pinned to one module of the float-activation implementation run alone
  in f64, at 1e-12, with a red arm for each fact that was open: the gate/up interleave, the
  window edge, the sink form, both convolution tap orders, the log-scale floor, the message
  query slots, the NVFP4 nibble order.
- `memra model verify tiny --against inkling2` runs a 40-token request on a tiny checkpoint in
  the real layout (hidden 64, 8 layers in 4 blocks, 3 loop passes, 16 slots, window 8, extent
  12, floor 24). The float-activation plan equals the f64 oracle to the bit after rounding to
  f32, as one chunk and under three schedules that split at and around the window, the extent
  and the floor, and the gate requires that bit equality. The oracle is the PR's trunk in f64
  with a lane-written expert module; the PR program with nothing replaced is 4.8e-5 from it in
  f64 with the same argmax at all 40 positions. The default-class plan is within 2.3e-6 of a
  separate simulation of the activation-quant program, under a tiny-gate band of 1e-5 that also
  requires every argmax.
- The pack declares no checkpoint band for any class, so `verify checkpoint` refuses it: a
  float-activation oracle cannot pass for the declared class.
- The tiny checkpoint and its token ids regenerate byte-identically from the tracked generator
  (`reference/ink2_tiny.py`, seed 20261121).
- On that fixture the two classes are 5.4 apart in logits and agree on the argmax at 5 of 40
  positions.

```bash
memra model verify tiny --against inkling2 --out tiny-out
# family=inkling2 verified=Tiny
cargo test -p memra-reference inkling2
cargo test -p memra-cli supported_packs_write_deterministic_native_oracles
```

The inspect bundles were written again on this tree: only `artifact.lock` moved (the support
state).

## Next

The tokenizer and chat dialect; the checkpoint runner for the declared numeric class, with an
oracle of that class from a serving binary; the drafter.
