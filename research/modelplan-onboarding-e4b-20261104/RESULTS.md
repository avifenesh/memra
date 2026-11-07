# Gemma-4 E4B onboarding receipts (2026-11-04)

Verdict: the gemma4_e4b pack binds the QAT Q4_0 GGUF (666 tensors), and Config,
TokenizerTemplate and TensorCensus pass on it. The engine cells pass on the RTX 5090.
CheckpointParity, RewriteParity and Serve stay pending: serving leaks a thought into content
(see Serve).

## Artifact

`gemma-4-E4B_q4_0-it.gguf` (local, 5154941280 bytes), `inspect/artifact.lock`:
`plan_sha256=5b09a55734a8b55d61d8eceb3d16742fdb1347d0bf462e5bb743326c44589f7a`,
`census_sha256=4d13361aa1fe46add50845f686f8d357c39be611a490d469b6063fab46245ae2`.

## Inspect (`memra model inspect <gguf> --against gemma4_e4b`)

`inspect/gates.txt`: Config=passed, TokenizerTemplate=passed, TensorCensus=passed. The rest are
pending. The tensor census, tokenizer contract, compiled plan and capability manifest are beside
it. Run at the fix-stack top `fbdd631927`. The artifact lock is byte-identical to the lane's
first inspect at `09348940b8`.

## Engine cells (RTX 5090, `gpu/`, driver `e4b_gpu_cells.sh`, HEAD `fbdd631927`)

| cell | result |
|---|---|
| fast-gate probe (`MEMRA_NGEN=20 run-gen`) | MATCH, tokens byte-identical to `tools/fast-gate/goldens/e4b.tokens` |
| `tools/argmax-margin-gate.sh` | `SUMMARY flips=0 bad=0`, `PRIME-SUMMARY prime=MATCH`, PASS |
| `gemma-gate` spec K=1..8, assistant MTP Q8_0 drafter, 64 tokens | stream agreement 64/64 at every K; accepted/drafted 26/38 at K=1 (0.684) to 33/81 at K=8 (0.407) |
| `MEMRA_VERIFY_GATE=7` on the 1736-token depth prompt | `VERIFY-GATE K=7: PASS` |
| `MEMRA_E4B_GRAPH_GATE=256` | stream 256/256 IDENTICAL (eager 169.28, graph 185.24 tok/s) |

## Serve

Pending. On a thinking-off chat the rendered prompt ends with a closed thought channel, but E4B
writes a thought anyway and closes it with a stray `<channel|>`, so content reads
`<thought text><channel|>ok`. A raw completion of the same prompt reproduces it, so the model
writes it, not the chat path. The fix needs an owner call on the gemma4 parser rule.
