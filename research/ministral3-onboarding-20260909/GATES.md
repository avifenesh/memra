# Ministral 3 native onboarding gates

State: **NativeReference**. No production admission or NativeTuned claim.
Pinned source: `mistralai/Ministral-3-8B-Instruct-2512@5b26027e7b19eeb4b7352e1fed3926375dd2cb4d`.

All builds, tests and execution gates ran on rented RTX 5090 32 GB hardware with
`MEMRA_CUDA_ARCH=120a` and `MEMRA_GPU_LOCK=/tmp/memra-gpu.lock`. No rig builds,
tests or GPU work. Pushes use `MEMRA_SKIP_PERF_CI=1` under the owner restriction.

| Gate | Result | Evidence |
|---|---|---|
| Pack/config/text tensor census | PASS: 785 physical source tensors, 309 semantic weights, 9,563,579,320 bytes | `receipts/gguf-reference-tests.log`, `receipts/checkpoint-hebrew/` |
| Nonmatching config refusals | PASS: window, rotary amplitude, embedding tie, activation | `receipts/gguf-reference-tests.log` |
| Native tiny executor | PASS, including scale boundary values | `receipts/native-reference-host.log` |
| Tekken IDs | PASS: 2,000/2,000 Hebrew, English and code samples, both special-token modes | `receipts/tokenizer-parity.log`, corpus and reference IDs beside it |
| Canonical template | PASS: default/explicit system, merged users, definitions, multiple calls/results | `receipts/tokenizer-tests.log`, tokenizer fixture directory |
| Streaming parser | PASS: fragmented/multiple calls, malformed preservation and recovery | `receipts/tool-parser-tests.log` |
| Source-FP32 checkpoint | PASS: max_abs=0.00006532669, both argmax=1267 on the same Hebrew input IDs | `receipts/checkpoint-hebrew/checkpoint-parity.tsv` |
| Pinned-source GPU rotary | PASS at 0/16383/16384/32768: Q max_abs=0.004052341, K=0.0017828941 | `receipts/rope-gate.log` |
| Kernel-check | 93 cells PASS, 22 artifact-dependent SKIPS; query-scale boundary max_abs=0 | `receipts/kernel-check.log` |
| NVFP4 mint and census | PASS: 238 quantized projections; 309 semantic tensors | `receipts/mint.log`, `receipts/nvfp4-final/` |
| run-gen | PASS: prefill/decode argmax=6008 and Hebrew output; cross-phase max logit difference 0.4207, not a byte-identity claim | `receipts/run-gen.log` |
| run-spec | REFUSED: checkpoint has no MTP/NextN head | `receipts/run-spec.log` |
| Vendor-shaped sampled HTTP | PASS c=1 and four concurrent clients, each with real tool call/result and eight continuation turns; sampling fields omitted | `receipts/serve/SUMMARY.json` |
| Anthropic and Responses | PASS: real tool round trips and Hebrew final answer | `receipts/wire/`, `receipts/wire-gate.log` |
| Streaming and refusals | PASS: streamed tool call; unknown model, disabled image input, and 262145-token prompt rejected | `receipts/endpoint/` |
| Stop/rollback | PASS: exact server PID stopped and health connection closed | `receipts/rollback-stop.json` |
| Regression/lint | PASS: 654 server tests; memra-gguf/reference/tokenizer suites; Clippy | `receipts/server-unit-2.log`, `receipts/unit.log`, `receipts/clippy-3.log` |
| CONTRIBUTING interleaved model/performance battery | NOT RUN | No PR eligibility claim |
| Sealed tuned rewrite bundle | NOT PRODUCED | NativeReference only |

## Precision and immutable artifacts

The served artifact is **weight-only GPTQ NVFP4**, with 71 kept BF16 embedding,
head and normalization tensors. Its compressed-tensors config has
`input_activations: null`; it is not a static-FP8 or W4A4 checkpoint.
The mint used the DictaLM recipe, 128 x 512 calibration samples, actorder disabled,
and an explicitly recorded BF16 expansion of the pinned FP8 weight codes times
`weight_scale_inv`. Pixtral and the multimodal projector were dropped.

The original static-activation FP8 program is refused by native serving. The
source-FP32 oracle expands the original FP8 codes directly to FP32, with static
activation rounding disabled for the `source-weights-float32-accumulation` class.
That oracle is independent of the minted artifact. The quantized GPU smoke and
serving gates do not assert lossless equality to the source-FP32 checkpoint.

- NVFP4 safetensors: 6,319,377,120 bytes,
  `03d9e693636d57976e56afbe6e494d9978f28c384dc82902a172b863b3a2bef5`.
- Serving config: `8c35abc33b251dd04dcb4c4a97b4360f16d08be35a39af4e122ce7550b08a343`.
- Tokenizer: `99cf274236c60277fcfad861a5a1007518687ad06ba8938760f50b55ffa0b1ef`.
- Template: `74eeb55fd3341286ec3fd44e902b7120721acc81cd394e96b431f85e93a1ea56`.
- Server binary: `a5e4e1b7c337f7cd16c47be68b07342b439e677c7dd6c2b5382358d18ff4ab0d`,
  content ID `memra-0.137.0-88a124f2db78`, Ubuntu 22.04/glibc 2.35 build.

GPU eager and carried-prime paths implement the query-scaled YaRN operation.
Batch, graph, speculative and pipeline rewrites remain refused for this operation;
concurrent HTTP sessions use independent eager execution. No environment flag was added.
`MEMRA_CTX=8192` is an initial allocation setting, not a hard ceiling. The configured
262144 cap is enforced, but this lane does not claim long-context serving qualification.
