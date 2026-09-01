# ModelPlan onboarding evidence (2026-08-22)

This directory preserves compiler and rewrite receipts produced by the native Memra gates. No
external engine is used at runtime. Hugging Face Transformers is used only by the offline
checkpoint oracle.

## Qwen3 pinned checkpoint

- Source revision: `tiny-random/qwen3@84ad45b4ecda2d4849aac0b768d520c239ff5875`
- Plan: `c79bf2b99033f5f13c33c6d79c173f428e59a13060a90f67f0d13e2277d344f2`
- Config, tokenizer/template, tensor census, tiny reference, checkpoint parity, and native serve
  gates passed.
- `decode-eager.v1` compared the portable ModelPlan executor with native CUDA over 151,936 logits:
  argmax matched and max absolute error was `0.0016501546`.
- `pipeline.v1` compared 911,616 logits across unsplit and two-stage execution with zero differing
  bits. The two stages used device `0` with dual PP disabled to validate serial
  partition/state-transport semantics; this is not a multi-GPU performance qualification.
- The native serve gate installed this partial bundle. Because `decode-batch.v1` is still
  unqualified for the BF16 artifact, the scheduler served successfully through the receipt-backed
  eager fallback; `serve.log` records that selection.
- The public evidence keeps the serve gate and server log, not the raw HTTP response envelope. The
  raw response contains a live build fingerprint and remains only in the local gate bundle.
- Rewrite parity remains pending because this BF16 artifact has no qualified exact batch or graph
  receipt.

## Qwen3.5-9B NVFP4 GGUF

- Source: `Qwen3.5-9B-NVFP4-MTP-GGUF.gguf` (local artifact)
- Plan: `402f51bbe73b9b5131a397b1db54f8964e0f2f71f6b261068d43f98e569d5741`
- Config, tokenizer/template, tensor census, tiny reference, and every eligible rewrite gate passed.
- `carried-prime.v1`: continuation argmax and four-token streams matched for two uneven sessions.
- `decode-batch.v1`: 1,986,560 logits were bit-identical between isolated and batched execution.
- `decode-graph.v1`: 80 greedy tokens were identical across four observed attention bucket keys and
  two graph captures.
- `mtp-spec.v1`: eight tokens were identical to the plain target with nonzero acceptance (`3/8`).
- Checkpoint and serve qualification remain pending. The matching safetensors artifact is ModelOpt
  NVFP4; forcing the existing f32 oracle class would dequantize into a different, OOM-sized numeric
  configuration.

The TSV receipts bind the artifact lock, model plan, reference/candidate streams, and producing
executable by SHA-256. `gates.txt` is authoritative for what remains pending.
