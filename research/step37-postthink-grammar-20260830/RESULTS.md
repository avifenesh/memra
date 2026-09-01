# Post-think constrained decoding (lane/step37-postthink-grammar-20260830)

`response_format` (json_object + json_schema) on models whose chat template force-opens a
think channel with no `enable_thinking` switch. Served model in scope: stepfun/step-3.7-flash
(the one roster model that refused `response_format` under the old honesty gate). Lane commits:
`2fceedf6f` (two-phase decoding) + `45359b4e2` (fail-closed terminal, review fix) on
`lane/step37-postthink-grammar-20260830`, branched from origin/main `189548721`.

Hardware: a 2x RTX PRO 6000 Blackwell Server dev pair, artifact
`/data/models/step37-flash-nvfp4` (stepfun-ai/Step-3.7-Flash-NVFP4, HF rev
`4275532ffd9a9496ff36b7a2dc4a9db1048da438`, shard sha256 verification receipt
`raw/shard-sha-verify.txt`). Serving env: the box's canonical ENVV (TP-2 0-44@0,1) +
`MEMRA_LOAD_MTP=1 MEMRA_MTP_HEADS=3 MEMRA_SPEC_K=3 MEMRA_SPEC_PMIN=0.5 MEMRA_SPEC_PMIN0=1
MEMRA_SERVE_SPEC=1 MEMRA_CTX=262144`. Every cell pins the reasoning-effort law shape: NO
effort field in any request; constrained cells carry NO sampling params (vendor defaults,
temp 0.5 / top_p 0.9 applied by the server).

## Design

**Two phases, one seam.** The phase gate (`PostThinkGate`) lives INSIDE
`SessionConstraint` (`crates/memra-server/src/constrained.rs`), so every existing
mask/consume call site takes the phase behavior with zero new branches: the host masked
sample, the batched device mask staging (`stage_grammar_mask` H2D of the packed SimpleVob
words), and graph promotion + per-step mask re-upload all flow through the one
`compute_mask()` / `consume()` pair.

- Phase 1 (think): `compute_mask()` returns `alloc_ones(n_vocab)` minus the request's
  FINAL end-of-generation set (caller stops + the model's eog union, taken AFTER the
  admission-time eog merge). Generation is unconstrained exactly as the model was
  trained; EOS cannot be sampled, which closes the receipted step37 EOS-inside-think
  quirk (finish=stop with `content: ""`) for constrained requests by construction.
  `consume()` advances the close detector and never touches the llguidance matcher.
- Phase flip: a rolling KMP match over the emitted TOKEN IDS against the template
  contract's close sequence. When the last close token is consumed the gate closes; the
  next `compute_mask()` is the grammar's INITIAL mask (matcher untouched during think,
  proven by unit pin `postthink_phase1_bans_eos_and_grammar_engages_at_close`).
- Phase 2 (content): the existing constrained machinery, unchanged; EOS is legal only
  where the grammar accepts (finished grammar collapses the mask to EOS-only).

**Phase detector: token ids from the template contract, never decoded text.**
`postthink_close_contract` (worker.rs, load-time, pure) derives the close sequence:
template must be think-forced (`<think>` + `add_generation_prompt`, no `enable_thinking`,
not dsv4 - the same substring laws the caps probe uses), close marker `</think>` (the
string parser's THINK_END law), token form from the tokenizer: the atomic vocab entry
when one exists (`id_of`), else the special-aware encoding accepted only on an exact byte
round-trip, else EMPTY = no contract = the loud 400 stays. On step37-NVFP4 the contract
is the single added token id 128799; measured against the artifact's tokenizer.json: no
other vocab entry contains the close-marker bytes, so the atomic id is the only trained
close signal. Known residual (accepted, documented): a multi-token byte alias of
"</think>" (e.g. "</" + "think" + ">") is not a close under the token contract; the
model was trained on the dedicated token and the battery measures reality.

**Think ceiling: `MEMRA_POSTTHINK_CEILING`, default 0 = OFF (new flag, FLAGS.md row in
the lane commit).** Past the ceiling the phase-1 mask collapses to exactly the next
unmatched close token, so the sampler is FORCED to walk the close sequence and the
grammar engages - implemented as mask contents, so it works identically on every path.
Default OFF by design: (a) the request's max_tokens governs the whole completion exactly
as today, and finish=length inside think is an honest, receipted face; (b) no fixed value
fits - step37 thinks routinely exceed 1024 tokens on real agentic prompts
(darklanes corpus QUIRK:step37:think-budget-wall-1024), so a low default clips real
reasoning and a high one never fires; (c) unmeasured behavior does not default ON. Both
arms receipted below (G cell = ceiling 256 forced-close arm).

**Spec interaction: DISENGAGED for post-think requests (chosen arm).**
`spec_eligible &&= !postthink` at admission; the `[spec-k]` admit line receipts
`K=0 source=eligibility-fallback` and the boot + arm receipts say so loudly
(`[postthink] ... spec disengaged`). Sampled constrained requests were already
spec-ineligible before this lane; the conjunction closes the greedy-constrained arm too.
Why not grammar-filtered drafts: the draft-side clone (`SpecGrammar`) would have to be
phase-aware and byte-exact THROUGH the mid-stream flip (clone the detector state, not the
matcher, in phase 1; hand off at the close boundary inside a chain) - ungated, no
receipts, so it stays off by design. The plain sanity cells prove spec still ENGAGES for
unconstrained step37 requests on the same boot.

**Switchable models (qwen path): untouched.** `response_format` still forces the
template's no-think switch and the grammar masks from token 1 - the same branch, pinned
by `response_format_think_table_switch_postthink_refusal` (think resolves to NoThink) and
by the base-vs-lane byte-identity gates below. dsv4 is excluded from the contract
derivation (its renderer honours NoThink through its own chat mode) and keeps its
existing path. The refusal for think-forced templates with NO derivable close contract
stays loud, message updated to name the missing contract.

**Fail-closed terminal (review fix, 2026-08-30, coordinator-endorsed).** A
response_format request whose generation ends INSIDE the think channel (max_tokens /
context bound / stop sequence, with the close never emitted) produced zero
schema-constrained content; the old terminal was 200 + finish=length + content "" -
a success a structured-output client can mistake for the contract being honored. The
terminal is now a NAMED error: 400 invalid_request_error, param naming the field that
ended generation (max_tokens / stop), reasoning-token count in the message; a stream
that already delivered reasoning deltas ends with the same error object (the existing
mid-stream error seam). finish=length remains possible only AFTER the grammar engaged
(non-empty truncated content - the same face every constrained model has; the
switchable-model constrained path is untouched). `postthink_unclosed_error` is pure and
unit-pinned on all four StopReason faces.

**Streaming: no new code.** The reasoning/content split is the existing string-level
parser (toolcall.rs Prethink -> Scan); phase-1 tokens stream as `reasoning` deltas, the
close token is syntax, post-close grammar-clamped tokens stream as `content` deltas.
Gate F pins the ordering, grammar-validity of concatenated content deltas, and the usage
receipt.

## Unit gates (GPU-free, all in the lane commits; memra-server suite 413/413)

| gate | pin |
|---|---|
| phase-1 mask bans the full eos set, allows everything else; grammar untouched during think; phase-2 initial mask == fresh matcher initial mask | `constrained::tests::postthink_phase1_bans_eos_and_grammar_engages_at_close` |
| ceiling collapses masks to the forced close walk, then grammar engages; forced-close receipt | `postthink_ceiling_forces_the_close_sequence` |
| KMP detector catches overlapping-prefix closes | `postthink_close_detector_handles_overlapping_prefixes` |
| arming is fail-closed (empty close / out-of-vocab id = loud error) | `postthink_arming_is_fail_closed` |
| close-contract derivation: atomic token / switch template / no tail / no template, on a real HF-dir tokenizer fixture | `worker::tests::postthink_close_contract_derivation` |
| admit/refuse table: switch -> NoThink grammar-from-token-1; contract -> admitted think-ON; no contract -> loud 400 naming the close contract | `response_format_think_table_switch_postthink_refusal` |
| fail-closed terminal: all four StopReason faces map to 400 invalid_request_error with the right param and the reasoning-token count | `postthink_unclosed_terminal_is_a_named_client_error` |

## Live gate table (dev box, raw receipts in raw/)

Binaries: base = origin/main `189548721` (md5 c97294fb0010ce2ec59ed3219a845c34), lane
FINAL = `45359b4e2` (md5 5d3e9829a41fe70b77fe7f42c6e3abbe; fail-closed terminal), lane
first-cut = `2fceedf6f` (md5 06b885e8693f32d43d6d7547dd745976; two-phase decoding only).
Artifact shards sha256-verified against HF rev `4275532ffd9a` (raw/shard-sha-verify.txt:
14/14 OK). One boot per arm, bin md5 recorded at every boot, no memra-server alive
pre-boot. The box's canonical ENVV minus the two flags REMOVED from the engine on
2026-08-29 (`MEMRA_NVFP4_BANK_V2`, `MEMRA_SEL_DOWN8`; the lane hit the server's own loud
removal refusal first, receipt in raw/battery.txt). Prompt pool: 16 real prompts (8
owner-blessed agentic + 8 banked real request payloads incl. two 30k contexts), pool
sha256 in raw/battery.txt; no synthetic prompts. Cells on the FINAL binary carry the
`2` suffix in raw/ (battery2/battery-lane2 etc.); the 2fceedf6f cells are retained as
the pre-review arm.

| gate | verdict | receipt |
|---|---|---|
| engine-path identity | PASS | run-spec built from base and lane commits is BIT-IDENTICAL (md5 c8f1835e... both): the lane changes memra-server only |
| run-spec K=1..8 greedy self-consistency (sweep + serving-policy twin, heads=3, real prompt) | PASS | raw/runspec-engine-*-summary.txt, SELF-CONSISTENCY PASS every K, illegal=0 sentinel87=0 |
| live text-path byte identity WITHOUT response_format, base vs FINAL lane (greedy seed0 x5 + vendor-sampled seed42 x3, serving spec K=3) | PASS | raw/identity-base.jsonl vs raw/identity-lane2.jsonl: 8/8 rows sha-identical positionally; lane greedy rep-stable (the 2fceedf6f arm also passed, raw/identity-lane.jsonl) |
| B step37+json_object, n=16 real prompts, vendor-default sampling, max_tokens 4096, FINAL binary | 5 valid JSON + 3 honest 200-length (grammar engaged, truncated non-empty content) + 8 named 400 fail-closed; ZERO mistaken-success faces | raw/battery-lane2.jsonl |
| C step37+json_schema strict, same 16 prompts, FINAL binary | 12 schema-valid + 4 named 400 fail-closed; every PASS validates the full strict schema | raw/battery-lane2.jsonl |
| adequate-budget arm (max_tokens 12288, streamed, the 8 length-prone prompts x obj + schema, + bounded-schema twin; binary 2fceedf6f) | json_object 6/8 valid, json_schema 7/8 valid, bounded schema 4/4 valid; residual misses are thinks past 12k tokens (curve-30k-2 twice, agentic8-3 once) - the named 400 under the FINAL binary | raw/budget.jsonl + raw/budget.txt |
| FAIL-CLOSED gate (review fix): never-close prompt class at max_tokens 1024, n=8 non-streaming + 2 streaming | PASS 10/10: named 400 invalid_request_error, message "response_format could not be honored ... N reasoning tokens", param max_tokens; streams end with the same error object and zero content deltas | N cells in raw/battery-lane2.jsonl |
| EOS-inside-think quirk face (finish=stop with content=="") | PASS: ZERO occurrences across every constrained request on both binaries (63 + 57 requests) | raw/*.jsonl: FAIL-stop-with-empty-content never fired |
| finish faces | PASS all three: named 400 for a budget ending inside think (E cell at max_tokens 64 + N cells); honest 200 finish=length only AFTER the grammar engaged (non-empty truncated content); stop after valid JSON | raw/battery-lane2.jsonl |
| streaming twin | PASS: reasoning deltas strictly before content deltas, content deltas concat to (schema-)valid JSON, usage receipt present; a never-close stream ends with the named error object (F3 + N-stream cells) | F/N cells in raw/battery-lane2.jsonl |
| vision x response_format (real image, MEMRA_STEP_VISION_DIR boot, FINAL binary) | PASS: json_object non-streamed + strict schema streamed (think 3540 tokens, close by model, schema-valid content); one non-streamed schema draw hit the PRE-EXISTING 90s non-streaming deadline (finish="error" partial delivery, honest, not a postthink face) - the streamed shape is the documented long-request form | raw/vision-lane2.jsonl + raw/vision2-lane2.jsonl + raw/vision2.txt |
| spec disengagement on constrained + spec ENGAGED on plain, same boot | PASS: [spec-k] K=0 source=eligibility-fallback on every constrained admission; plain sanity cells carry usage.spec | raw/receipts-serve-lane2-main.txt |
| ceiling arm (MEMRA_POSTTHINK_CEILING=256 boot, both binaries) | forced close fired 4/4 at exactly 257 think tokens each run; 3/4 valid JSON; 1/4 = the documented unbounded-grammar whitespace-degeneration face ("{" then whitespace to budget) | raw/ceiling-lane2.jsonl + raw/receipts-serve-lane2-ceiling.txt |
| refusal path (think-forced template with NO derivable close contract) | PASS (unit): loud 400 naming the missing close contract; no served artifact without a contract exists to hit live | `response_format_think_table_switch_postthink_refusal` + `postthink_close_contract_derivation` |
| switchable (qwen) path | PASS (unit + code-identity): think forced to NoThink, grammar from token 1, same branch as before; no qwen artifact on this box for a live twin | unit pins + the base-vs-lane live identity above |
| zero ILLEGAL / #87 / panic | PASS across every boot and cell, both binaries | receipt blocks in raw/battery.txt, raw/battery2.txt, raw/diag.txt |
| [postthink] receipts | boot contract line (ids [128799]), per-request armed lines, per-finish close receipts | raw/receipts-serve-lane2-*.txt |

## Finding: never-closed thinks are the MODEL'S OWN budget wall, not an EOS-ban artifact

The B/C length rows all share one shape: think never closes within max_tokens 4096
(reasoning 10-17k chars, coherent deliberation, no repetition loop - full-tail
inspection). The discriminator (raw/diag.jsonl, one boot, paired plain vs json_object
twins on the same 8 length-prone prompts x2 reps, vendor-default sampling, max_tokens
4096): plain closed 7/16, constrained closed 8/16, pairs (plain,constrained) =
{(T,T): 7, (F,F): 8, (F,T): 1} - ZERO pairs where only the constrained arm failed to
close. The phase-1 EOS ban does not lengthen thinks; this is the known step37
think-budget wall (darklanes corpus QUIRK:step37:think-budget-wall-1024) at 4096.
Measured natural closes on these prompts (main boot): p50 2119 / p90 3554 / max 3900
think tokens - 4096 truncates the upper tail, so the product lever is the client's
max_tokens (streamed for long budgets: the non-streaming deadline gate refuses a
12288-token non-streaming request by design, with its own advice to stream).

## Decisions of record

1. Phase detector = token-id KMP against the load-time close contract (atomic `</think>`
   id 128799 on step37-NVFP4). No string matching on decoded text.
2. `MEMRA_POSTTHINK_CEILING` default OFF (0); both arms receipted.
3. Spec disengages for post-think constrained requests; receipted via [spec-k] K=0 +
   [postthink] lines; grammar-filtered drafts deliberately not attempted (ungated).
4. Refusal stays for think-forced templates without a derivable close contract; dsv4
   excluded from the contract on purpose.
5. Fail-closed terminal (review fix): generation ending inside think on a
   response_format request is a named 400 (or the mid-stream error object), never a
   200 with empty content. Gated live on the never-close prompt class (N cells).
