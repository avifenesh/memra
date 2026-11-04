# Gemma 4 E4B

| | Recommended use |
|---|---|
| **Status** | Supported on the QAT Q4_0 GGUF |
| **Best starting path** | QAT Q4_0 GGUF; the Gemma assistant drafter runs through `gemma-gate` |
| **Hardware** | RTX 5090-class Blackwell, the card the record below was measured on |
| **Use this when** | You need this exact Gemma 4 checkpoint rather than assuming another Gemma card transfers |

The family name does not transfer tuning or support. Confirm the exact row in
[models and hardware](../MODELS.md).

E4B is the dense Gemma 4 program plus two pieces the other Gemma cards do not have: per-layer
input embeddings and KV sharing (the last 18 layers read the caches of layers 22 and 23). Its
`gemma4_e4b` model pack types both in the plan, and its global layers carry their own V
projection. The artifact gates are pending a tracked receipt.
Pack state: **NativeReference** on the tiny gate. <!-- support: gemma4_e4b-reference -->

The served speculative attach (`MEMRA_DRAFT` on `memra-server`) skips E4B. Its assistant
drafter runs through the `gemma-gate` CLI only.

## Qualification record

RTX 5090 Laptop, 2026-11-04, binaries built from commit `795cf9f824` (sm_120a). Artifact
`gemma-4-E4B_q4_0-it.gguf` (5,154,941,280 bytes, from `google/gemma-4-E4B-it-qat-q4_0-gguf`),
drafter `gemma-4-E4B-it-assistant.Q8_0.gguf`. Each cell is a single run.

| Cell | Command | Result |
|---|---|---|
| Load | `memra model inspect <gguf> --against gemma4_e4b`, then every run below | 666 tensors bound by the pack contract; Config, TokenizerTemplate and TensorCensus pass |
| Tiny plan | `memra model verify tiny --against gemma4_e4b` | pinned oracle matches; an independent NumPy forward agrees to max abs 3.3e-7 |
| run-gen argmax | `MEMRA_NGEN=20 run-gen` on the pinned Gemma chat ids | prefill/decode MATCH, batched prime MATCH, 20 tokens identical to the `e4b` fast-gate golden |
| Calibrated argmax | `tools/argmax-margin-gate.sh <gguf>` (board-2048, window 12, max flips 1) | PASS, flips 0, bad 0, prime MATCH |
| Speculative K=1..8 | `MEMRA_SPEC=<K> MEMRA_DRAFT=<assistant> MEMRA_NGEN=64 gemma-gate` | stream agreement 64/64 at every K; acceptance 0.684 at K=1, 0.407 at K=8 |
| Batched verify | `MEMRA_VERIFY_GATE=7 gemma-gate` on the 1,736-id depth prompt | PASS |
| Graph door | `MEMRA_E4B_GRAPH_GATE=256 gemma-gate` | 256/256 tokens identical to eager |

`run-spec` needs an embedded MTP head, so the Gemma assistant drafter is gated with
`gemma-gate` stream agreement, the same check the 31B runs in `tools/local-ci.sh`.
