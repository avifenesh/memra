# HY3 native safetensors and all-expert NVFP4 onboarding

Status: offline integration sealed; RTX PRO 6000 load, checkpoint, rewrite, MTP, and serving gates
pending.

## Decision

- Semantic source: `tencent/Hy3@a960ebc3da325ba167f069f76c41eb62c9280d22` BF16
  safetensors.
- Primary artifact: our streaming NVIDIA ModelOpt 0.46.0 mint from the pinned semantic source,
  using ModelOpt commit `43fd41a58d52c4e6e5dec1d1ff5989ecc737ae1a`. The deterministic W4A16 recipe emits
  46,080 routed-expert NVFP4 weights: 45,504 trunk weights plus all 576 layer-80 MTP expert
  weights. Dense layer 0, attention, router, shared MLP, embeddings, output head, norms, biases,
  and non-expert MTP tensors remain source BF16/F32. Within every expert, gate and up share the
  larger ModelOpt per-tensor `weight_scale_2`, matching the pinned fused-MoE export recipe; all
  15,360 pairs must be bit-equal in the finished artifact.
- Existing all-expert artifacts are comparison controls, not the publication payload:
  - `kodelow/Hy3-NVFP4-W4A16@4f7e1f...` supplies prior qstream MSE-scale quality and 83.4% MTP
    acceptance evidence;
  - `LibertAIDAI/Hy3-NVFP4@8a805d...` is an independent ModelOpt-format control. Its card says
    MTP is BF16, but its pinned index has all 576 MTP expert weights in NVFP4. The index wins.
- There is no MTP-BF16 candidate or fallback. The owner has already measured that NVFP4 MTP loses
  nothing; the `hy3_nvfp4` model pack therefore requires NVFP4 for every routed expert, including
  MTP.
- Primary container: Hugging Face safetensors. The loader accepts both strict encodings without a
  numeric substitution:
  - compressed-tensors: `weight_packed` + E4M3 `weight_scale` + F32
    `weight_global_scale` divisor;
  - ModelOpt: packed U8 `weight` + E4M3 `weight_scale` + F32 `weight_scale_2` multiplier.
  The compressed global scale is validated and inverted exactly once at the loader boundary.
- Publish only our generated payload after its exact Memra load, same-artifact oracle, MTP, sampled
  quality, stress, and serving gates pass. The external controls cannot be substituted for it.
- GGUF remains an independent portability/oracle twin after safetensors qualification. It is not the
  primary artifact and cannot substitute a different quantization program.

## DFlash decision (live check 2026-08-30)

No HY3 DFlash2 checkpoint or Memra HY3 DFlash2 consumer exists today. AngelSlim publishes HY3
DFlash-B8/B16 and DFly checkpoints, but those are different draft architectures, not DFlash2. Memra's
current DFlash2 consumer is not wired to HY3's full-attention target. Therefore the owner's condition
"DFlash2 means no MTP" is not engaged: embedded MTP remains the required speculative path. A future
real HY3 DFlash2 release reopens this decision from a pinned checkpoint and consumer contract; name
similarity does not.

## Exact payload

Header-derived tensor payloads exclude safetensors headers and index metadata:

| component | bytes | decimal GB | GiB |
|---|---:|---:|---:|
| complete all-expert NVFP4 artifact | 180,826,481,152 | 180.826 | 168.408 |
| complete MTP block (593 tensors) | 2,295,901,696 | 2.296 | 2.138 |
| artifact without MTP | 178,530,579,456 | 178.531 | 166.270 |

The complete artifact has 139,298 stored tensors normalized to 47,138 semantic tensors across 99
shards. Removing MTP is not an active profile; the split is recorded only to answer capacity
questions. `size-estimates.json` derives the profile from the pinned official BF16 census.

## Offline evidence complete

- Official BF16 remote header census: 47,138 tensors / 99 shards; config, tokenizer/template,
  tensor contract, and ModelPlan binding pass in `bf16-inspect/`.
- Official FP8 control census: 46,647 per-tensor FP8 weights, 411 BF16 tensors, and 80 F32 tensors.
- Compressed-tensors control header gate passed binding and tokenizer, with census SHA-256
  `b6b96e84d6d99b7b52e396b34bb1b7f21695155d48209bc903f88ed54bc81da4` locked in
  `sources.lock.json`.
- Independent ModelOpt control header gate passed binding and tokenizer, with census SHA-256
  `566db2975edac5cd1a86061ec6943988ef695cc8ae8c6cda050ad0d354ae2600` locked in
  `sources.lock.json`. Our deterministic ModelOpt mint must reproduce this logical census.
- Native tiny plan deterministically covers dense layer 0, QK-normalized attention, sigmoid+bias
  routed MoE, shared MLP, and embedded MTP.
- The loader preserves every deliberately unquantized HY3 BF16 tensor for both ModelOpt and
  compressed-tensors artifacts. It does not silently substitute Q8.
- Strict header and runtime gates reject absent/ambiguous macro scales, malformed per-16 grids,
  BF16 expert substitutions, NVFP4 attention substitutions, and missing MTP experts.
- Negative census receipt: `r0b0tlab/Hy3-295B-NVFP4@63949b...` lacks all 576 MTP expert weights.
  BF16-MTP artifacts are rejected by policy and are not retained as qualification arms.

## RTX PRO 6000 qualification order

1. Use an isolated non-production host with exactly four full-power RTX PRO 6000 Blackwell cards.
   Record GPU UUIDs, power limits, P2P matrix, driver, CUDA, image digest, Memra SHA, artifact
   revision, config/index hashes, and full downloaded byte manifest before model allocation.
2. Storage preflight:
   - existing artifact with the default on-disk stream-repack cache: at least 400 GB genuinely free
     before download, or at least 180 GB free after the 180.9 GB artifact is present; a smaller host
     must explicitly use `MEMRA_ST_REPACK_DISK=0` and the supported in-RAM gather path;
   - streaming mint: at least 850 GB free because the 597,572,342,272-byte BF16 source and
     180,826,481,152-byte output coexist.
3. Run `run-modelopt.sh experts` on all four selected GPUs. It must reproduce the expected config,
   quant-config, index, payload, stored-tensor count, tokenizer/template, and census locks before any
   model allocation. Preserve a full post-mint byte manifest off-box.
4. Run one bounded native forward with PP-4 placement
   (`MEMRA_PP_STAGES=4 MEMRA_PP_DEVICES=0,1,2,3`) and default peer integrity probes. Require finite
   logits, expected source-native BF16/NVFP4 residency announcements, and all four stage owners.
5. Capture an external oracle outside serving and Memra outputs from the same quantized artifact and
   token inputs. Unified ModelOpt exports must use a supported deployment loader (pinned vLLM
   `modelopt_fp4` here), not plain `transformers.AutoModel`. Gate finiteness, argmax, top-k
   stability, and named error bounds. Never compare one artifact's logits to the other as an
   exactness oracle.
6. Run greedy plain-vs-MTP self-consistency at K=1..8 on real bounded prompts. Require token identity,
   nonzero acceptance, and explicit PP verify engagement. Greedy loops are excluded from performance
   rows; they are not model findings.
7. Compare the sealed result against the pinned external controls only after our artifact passes.
   The qstream card's published numbers are prior evidence, not Memra qualification or publication
   identity.
8. Qualify eager, batched, PP-4, MTP verify, cache, and every selected rewrite on the exact
   artifact/plan/binary/hardware tuple. Any required unwired path fails closed.
9. Serving gate: readiness, pinned identity, tokenizer/template, tools, reasoning levels, sampled
   vendor defaults with no sampling parameters, concurrency, context/admission, cache-on eight-turn
   twin, stress succession, rollback, and server-log evidence that the intended NVFP4/PP/MTP paths
   engaged. A 200 response is not proof.
10. Only after qualification may Darklanes upload our sealed artifact and add it to the established
    Memra/Tiyuvta Hugging Face collection with both projects named in the model card. No roster or
    performance claim occurs before those receipts exist.

## Stop conditions

- Any artifact/hash/tensor/scale mismatch, tokenizer/template drift, non-finite logit, wrong router
  program, absent MTP experts, or missing path-engagement line stops the candidate.
- A host with only 200 GB container disk cannot download the artifact safely and cannot mint it.
- Another GPU architecture does not substitute for RTX PRO 6000 qualification.
- A spot host never holds the only copy of code or receipts; sync at each rung and close it when no
  next goal-linked cell can start.
