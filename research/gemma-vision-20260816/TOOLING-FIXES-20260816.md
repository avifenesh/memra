# Tooling-fix lane: the two walls, resolved (2026-08-16)

Fixer lane for the perfection mission's two blockers. Verdicts first:

## Wall 1 — "memra-gguf parser desync": NO PARSER BUG EXISTS

Evidence chain, all on the Japan box against the exact artifact
(`gemma-4-31B-it-NVFP4mix.gguf`, 22,010,961,408 bytes, mtime Aug 16 16:40):

1. `gguf-inspect` (memra's own parser) reads the FULL file: version=3, kv=41,
   tensors=833, arch=gemma4, `tokenizer.ggml.eos_token_id = U32(1)`,
   `bos = U32(2)`, tokens array [262144 entries] — every key lands.
2. A bare `memra-server` boot on the artifact (no draft/vision/ranks env)
   SUCCEEDS: `loaded "g31": 60 layers, eos=1`, template caps read
   (tok="gemma4", ctx=262144), prefix cache initializes.
3. The failing boot log (`evidence/gemma-text/serve-nvfp4.log`) FATALs at
   `tokenizer gemma: missing tokenizer.ggml.eos_token_id` — but the artifact's
   mtime (16:40) POSTDATES those boots: the file the server rejected was the
   pre-injection convert output, and the file gguf-py later verified (and that
   now boots) is the post-injection rebuild. Artifact-state churn during the
   debugging loop, not a parse defect.

FIXTURE DELIBERATELY NOT ADDED: the directive asked for a fixture reproducing
the desync; there is no desync to reproduce, and a synthetic fixture that
passes trivially would document nothing. The receipt is the on-box whole-file
read above. (memra-tokenizer's lookup uses `as_u64()`, which accepts every
integer type variant — a type-flip in a future converter cannot re-create this
symptom either.)

SIDE-FINDING (load-bearing for the aggregate cell): the boot line
`EAGER-ONLY serving (gemma4 class — no batched decode arm): per-session eager
decode, monolithic prefill, no graph promotion, no prime batching` — gemma4
c8 aggregate scaling is bounded by the missing batched decode arm, REGARDLESS
of quantization. The Q4_0-vs-NVFP4 c8 comparison measures kernel speed under
eager round-robin, not batch scaling. The batched decode arm for gemma4 is a
(sized, separate) engine lane if aggregate throughput becomes a pricing basis.

## Wall 2 — gemma-gate dflash verdict never emitted: BRANCH SHADOWING, fixed

`gemma-gate`'s env-selected runs are an ordered if-chain, and the FR-rank
corpus-mint branch (`MEMRA_GEN_CORPUS`/`MEMRA_GEN_OUT`) precedes the dflash
branch (`MEMRA_SPEC_DFLASH`). A shell still exporting the mint envs while
requesting the dflash acceptance cell silently minted instead of measuring —
the "summary never emitted" incident. Fix: the combination now REFUSES loudly
with a contextual error naming both selections. Two defused traps for the
record: the `&[]` argument in the dflash call is `eos`, not ranks (dflash
doesn't consume FR-Spec ranks in this API), and the acceptance stats DO print
inside the round under `MEMRA_SPEC_STATS=1`
(`[dflash] acceptance a/b = r`), so a clean-env run emits everything.

## Courtesy measurements (raw handoff — the gemma lane owns interpretation)

NVFP4mix through memra-server, Japan GPU1 (450W cap), ctx 8192, q38 bench
harness prompts, max_tokens 128, temp harness-default:

```
nvfp4-c1: 24/24 ok | agg 55.2 tok/s | decode p50 57.21 | ttft p50 86ms
nvfp4-c8: 48/48 ok | agg 55.25 tok/s | decode p50 7.15 | ttft p50 601ms
```

(c8 agg == c1 agg to within noise — the flat line the EAGER-ONLY boot notice
predicts. Raw rows in evidence/gemma-text/cells.jsonl on the box.)
