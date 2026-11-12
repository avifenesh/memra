# Reasoning control: one effort table on every surface, the model's own default, no hidden reasoning (2026-08-23)

**Status:** adopted. The effort surface landed 2026-08-07 (owner directive: every supported model
is a thinking model). One canonical table for the three request dialects landed 2026-08-21 (issue
#31). The schema decisions below were made 2026-08-23 (lane/reasoning-schema-20260823); the
GLM-5.3-Flash `medium` rung was set 2026-09-02 (issue #75). Code: `canonical_effort_for` and
`parse_think` in `crates/memra-server/src/lib.rs`, the `thinking` translation in
`crates/memra-server/src/anthropic.rs`, the template arms in `crates/memra-tokenizer/src/chat.rs`.
Receipts: [`research/effort-surface-parity-20260821/`](../../research/effort-surface-parity-20260821/RECEIPTS.md),
[`research/reasoning-schema-20260823/`](../../research/reasoning-schema-20260823/) (goldens),
[`research/step37-reasoning-effort-20260829/`](../../research/step37-reasoning-effort-20260829/RESULTS.md),
[`research/step-sku-20260807/`](../../research/step-sku-20260807/).

The rationale below moved out of the serving doc on 2026-11-12. The old text, with the per-model
mapping table, is kept in
[docs/archive/SERVING-2026-11-12.md](../archive/SERVING-2026-11-12.md#openai-tools-surface-serve-tools-lane-2026-08-02).
The request fields themselves are documented in [docs/API-SURFACES.md](../API-SURFACES.md).

## Chosen

1. **One table for three dialects.** `none | minimal | low | medium | high`, plus the aliases
   `xhigh | max | ultra`, are accepted on chat (`reasoning_effort`, `reasoning`), `/v1/responses`
   (`reasoning.effort`) and `/v1/messages` (`output_config.effort`), with the same rejection and
   the same effect. Any other value is a 400. An explicit boolean switch (`reasoning.enabled`,
   Anthropic `thinking.type`) wins the on/off decision over the switch an effort implies; the
   effort is still validated and still supplies the level.
2. **Absent means the model's own default.** The server never overrides it. An operator can set
   `default_reasoning_effort` per model in `MEMRA_MODEL_METADATA`, and an explicit client choice
   still wins.
3. **The aliases clamp to `high`**, except on a template that defines a rung above `high`
   (DeepSeek-V4 0731 and GLM-5.3-Flash), where they map to `max`.
4. **On a template without a level input, a named level means reasoning on.** The rendered prompt
   is byte-identical to `reasoning: {"enabled": true}`.
5. **Qwen3.8's ladder renders the vendor's default when unset.** Its template reads an effort
   (`xhigh` default, `medium`, `low`; `high`, `max` and `ultra` fold onto `xhigh` per Qwen's own
   mapping), and the unset request renders `xhigh`. An operator who wants the old bytes sets
   `default_reasoning_effort = "medium"`, the vendor's zero-steering rung. Since v0.109.1 a
   ladder template routes even unset non-tools chats through the tools-capable renderer, because
   only that renderer injects the default; ladder-less templates keep the byte-identical bypass.
6. **`minimal` means off.** On a model that cannot turn reasoning off, `minimal` is the same named
   400 as `none`.
7. **GLM-5.3-Flash `medium` maps up to `High`.** Its template has `Low`, `High` and `Max` (its own
   default) and no `medium`. A sub-max ask never falls through the template's `else` arm to `Max`.
8. **The vLLM spellings are aliases.** Top-level `enable_thinking` and
   `chat_template_kwargs.enable_thinking` mean `reasoning.enabled`, with the same precedence and the
   same rendered prompt. Every other `chat_template_kwargs` key is a 400 naming the key, except
   `preserve_thinking: true` (below). Two explicit switches that disagree are a 400. A
   client-explicit off request on a template whose think tail is unconditional is a 400 naming the
   model (`cannot disable reasoning`); an operator default of `none` never 400s a caller who sent
   nothing.
9. **Reasoning is always delivered.** Every format returns the thinking text (chat
   `message.reasoning` and `reasoning_details`, a Messages `thinking` block, a Responses `reasoning`
   item), and a reasoning-off request carries no thinking field. `include_reasoning: false` and
   `reasoning.exclude: true` stop the model reasoning: they are aliases of
   `reasoning.enabled: false` and inherit its refusal on a template that cannot go off.
10. **`max_tokens` is one output budget for reasoning and content.** Reasoning counts as completion
    tokens. There is no separate reasoning budget and no `reasoning_tokens` field.
    `reasoning.max_tokens` and Anthropic `thinking.budget_tokens` are named 400s.

## Rejected

- **A table per dialect.** Before #31, `/v1/messages` never read `output_config.effort`, so it
  accepted every string with a 200 and no effect, and `/v1/responses` kept its own copy of the
  table. Two tables drifted. After the fix all three surfaces produce the same 244 reasoning
  characters at `high` and the same suppression at `none`
  ([`research/effort-surface-parity-20260821/RECEIPTS.md`](../../research/effort-surface-parity-20260821/RECEIPTS.md)).
- **A distinct `xhigh` tier, or a 400 for it.** No three-rung or binary template has a level above
  `high` for it to ride, and stock Codex (on `/v1/responses`) and Claude Code (on `/v1/messages`)
  send `xhigh` by default. A 400 would break default-config agent sessions.
- **A 400 for a named level on a binary-switch model.** Those same clients send `xhigh` on every
  request; against a model like Ornith-1.5, whose cards define no effort at all, the 400 would
  refuse every default session. The caller asked for reasoning and gets reasoning.
- **Reading the levels as a quality ladder.** Measured on the level-consuming Step-3.7-Flash
  template (NVFP4 safetensors, TP across two RTX PRO 6000 cards, 2026-08-29, three real prompts,
  n=8 vendor-default sampled per level, `max_tokens` 1024; reasoning characters, medians):

  | prompt | absent | low | medium | high | xhigh |
  |---|---:|---:|---:|---:|---:|
  | `17*23?` | 229 | 80 | 83 | 84 | 79 |
  | train arrival | 1531 | 272 | 373 | 175 | 190 |
  | python one-liner | 2723 | 263 | 992 | 529 | 818 |

  Every level is honoured and answers sanely, but depth is not monotone (`high` lands below
  `medium` on both non-trivial prompts) and the absent default is the deepest arm. Naming a level
  spends fewer reasoning tokens; a higher level does not buy more depth. On binary-switch templates
  (Qwen3.5 and Qwen3.6, gemma4) `low`, `medium` and `high` render identical prompts.
- **Keeping Qwen3.8's historical no-instruction bytes for an unset request.** They were the
  defect: the template's own default is `xhigh`. v0.109.0 kept the non-tools bypass and served a
  split surface, where the same unset request rendered the default when a `tools` array rode along
  and the bare bytes without one. Goldens rendered from the vendor's jinja pin every rung:
  [`research/reasoning-schema-20260823/goldens/`](../../research/reasoning-schema-20260823/goldens/).
- **`minimal` as brief reasoning, as Qwen's hosted API maps it (`minimal` to `low`).** "Use the
  lab's model" governs each model's behaviour; `minimal` is a value of this server's schema, whose
  promise is that the no-reasoning side is real.
- **GLM-5.3-Flash `medium` clamped down to `Low`.** The first mapping. The 2026-09-02 ruling put
  the middle ask on the middle rung. Letting `medium` fall through to the template's `else` arm
  (`Max`) was never an option: answering "reason less" with the deepest setting.
- **Accepting unimplemented template kwargs.** The renderer is Rust, not jinja, so a kwarg it does
  not implement changes nothing. Until 2026-08-23 both `enable_thinking` spellings were
  deserialized away, so a thinking-off request got a 200 with reasoning on.
- **A suppression mode (generate, count, withhold).** Reasoning is output; withholding it would
  count output that was never sent. The drop path was deleted from the streaming parser. A client
  that does not want the text discards the field; a client that does not want it generated
  switches it off. Owner ruling, 2026-08-23: "also thinking content should be returned, not only
  the content itself".
- **Honouring `budget_tokens`.** No lever can cap a reasoning segment: `GenParams` carries only
  `max_new`, `max_ctx` and `eos`, and a `</think>` stop string would end the whole request with an
  empty answer. Accepting it would promise a spend cap the server cannot keep.

## Measurement

- The single budget: on `qwen/qwen3.8-27b` at `max_tokens: 3072` on a hard puzzle, 3/3
  reasoning-on runs returned `finish_reason: "length"` with empty `content`, against 0/3 with
  `reasoning_effort: "none"` (2026-08-23; the raw probe is not in this repository). This is the
  budget working as specified. The caller's remedies are `none`, a lower rung or a larger cap.
- The level ladder: the Step-3.7-Flash table above
  ([`research/step37-reasoning-effort-20260829/RESULTS.md`](../../research/step37-reasoning-effort-20260829/RESULTS.md)).
- Template parity: the per-model renders are golden-tested against each shipped template
  (`research/step-sku-20260807/render-thinking-goldens.py`, the Qwen3.8 goldens above), with
  serve-smoke receipts in `research/step-sku-20260807/raw/`.

## Amended since

- **`preserve_thinking`.** The 2026-08-23 text refused it in both directions, because the Qwen arm
  did not replay a prior assistant turn's `<think>` block, so multi-turn Qwen3.8 prompts were off
  the vendor's bytes for turns that carried reasoning. The renderer now implements the vendor
  default (replay, empty when the client sent no reasoning), so `true` is accepted and only
  `false`, the strip arm, refuses (`parse_template_kwargs`, `crates/memra-server/src/lib.rs`).
- **`reasoning.summary`** accepts only `"auto"`: the server returns the reasoning text verbatim and
  does not summarise.

## What would change it

A template that defines its own meaning for an alias or for `minimal` gets its own row, keyed on
the template probe, never on a family name. A lever that can close a reasoning span after N tokens
would make `budget_tokens` honourable.
