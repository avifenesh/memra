# Gemma 4 E4B

| | Recommended use |
|---|---|
| **Status** | Supported on the QAT Q4_0 GGUF through the CLI gates below; `memra-server` serving is not qualified |
| **Best starting path** | QAT Q4_0 GGUF; the Gemma assistant drafter runs through `gemma-gate` |
| **Hardware** | RTX 5090-class Blackwell, the card the record below was measured on |
| **Use this when** | You need this exact Gemma 4 checkpoint rather than assuming another Gemma card transfers |

The family name does not transfer tuning or support. Confirm the exact row in
[models and hardware](../MODELS.md).

E4B is the dense Gemma 4 program plus two pieces the other Gemma cards do not have: per-layer
input embeddings and KV sharing (the last 18 layers read the caches of layers 22 and 23). Its
`gemma4_e4b` model pack types both in the plan, and its global layers carry their own V
projection. The pack matches this geometry only (42 layers, 18 of them KV-shared, per-layer
inputs 256 wide); another checkpoint with per-layer inputs gets no pack until it has its own
record. On the QAT Q4_0 GGUF, Config, TokenizerTemplate and TensorCensus pass on a tracked
receipt; CheckpointParity, RewriteParity and Serve are pending.
Pack state: **NativeReference** on the tiny gate. <!-- support: gemma4_e4b-reference -->

E4B runs its own programs: whole-prompt prime, eager and dc decode, a graph-exec loop, and
`gemma-gate`'s batched verify and assistant-drafter speculative decode. It has no batched
decode and no pipeline split; a load with `MEMRA_PP_STAGES` set refuses by plan operation.

## Serving

`memra-server` loads E4B in its eager-only class (per-session decode, whole-prompt prime), and
the served speculative attach (`MEMRA_DRAFT`) skips it, so the assistant drafter runs through
`gemma-gate` only. Serving is not qualified. In the record below, serve-smoke's plain surface
and a solo-versus-concurrent identity check pass, but its gemma4 chat arm fails: on a
thinking-off chat E4B writes a thought after the template's closed thought channel and ends
it with a stray `<channel|>`, so the thought and the tag reach `content`. The Serve gate stays
pending.

## Qualification record

RTX 5090 Laptop (driver 615.71.09), 2026-11-04, sm_120a builds. The engine cells (load, run-gen,
calibrated argmax, speculative, verify, graph) were re-run on the fix-stack commit `fbdd631927`;
the pipeline-refusal and serve cells come from the lane's first run at `09348940b8`, whose binary
hashes are in `serve-09348940b8/binaries.sha256`. Raw logs for every cell are in
[`research/modelplan-onboarding-e4b-20261104/`](../../research/modelplan-onboarding-e4b-20261104/RESULTS.md),
which also backs the TokenizerTemplate and TensorCensus gates in `docs/support-records.toml`.

Artifact `gemma-4-E4B_q4_0-it.gguf` (5,154,941,280 bytes, from
`google/gemma-4-E4B-it-qat-q4_0-gguf`), drafter `gemma-4-E4B-it-assistant.Q8_0.gguf`
(98,653,280 bytes). Each cell is a single run.

| Cell | Command | Result |
|---|---|---|
| Load | `memra model inspect <gguf> --against gemma4_e4b`, then every run below | 666 tensors bound by the pack contract; Config, TokenizerTemplate and TensorCensus pass; plan_sha256 `5b09a55734a8b55d61d8eceb3d16742fdb1347d0bf462e5bb743326c44589f7a` |
| Tiny plan | `memra model verify tiny --against gemma4_e4b` | pinned oracle matches; an independent NumPy forward (window 2, weighted PLE norms) agrees to max abs 1.8e-7, and its red arms (no PLE tail, no KV sharing, no window, unit PLE norms) all flip argmax |
| run-gen argmax | `MEMRA_NGEN=20 run-gen` on the pinned Gemma chat ids | prefill/decode MATCH, batched prime MATCH, 20 tokens identical to the `e4b` fast-gate golden |
| Calibrated argmax | `tools/argmax-margin-gate.sh <gguf>` (board-2048, window 12, max flips 1) | PASS, flips 0, bad 0, prime MATCH |
| Speculative K=1..8 | `MEMRA_SPEC=<K> MEMRA_DRAFT=<assistant> MEMRA_NGEN=64 gemma-gate` | stream agreement 64/64 at every K; accepted/drafted 26/38 (0.684) at K=1, 33/81 (0.407) at K=8 |
| Batched verify | `MEMRA_VERIFY_GATE=7 gemma-gate` on the 1,736-id depth prompt | PASS |
| Graph door | `MEMRA_E4B_GRAPH_GATE=256 gemma-gate` | 256/256 tokens identical to eager |
| Pipeline refusal | `MEMRA_PP_STAGES=2 run-gen <gguf>` | load refused before any upload; blockers name PerLayerInputEmbedding, PerLayerInputGate and SharedKvState |
| Serve, plain surface | `tools/serve-smoke.sh <gguf>` | 7/7: model list, chat, stream, completions, greedy determinism, 3 concurrent chats, long generation |
| Serve, gemma4 chat arm | `GEMMA_MODEL=<gguf> tools/serve-smoke.sh <gguf>` | FAIL: thinking-off content is `The user is asking ... exactly that.<channel|>ok`; a `reasoning_effort: low` chat returns no reasoning |
| Serve, cache metering | serve-smoke's cache-meter gate | not applicable: the eager-only class takes no prefix snapshot, so 0 prefix-cache hits where the gate expects 4 |
| Serve, identity | one greedy chat solo, then the same chat 3x concurrent (96 tokens) | all 3 identical to solo |

`run-spec` needs an embedded MTP head, so the Gemma assistant drafter is gated with
`gemma-gate` stream agreement, the same check the 31B runs in `tools/local-ci.sh`.
