# effort-surface-parity — issue #31: reasoning-effort validation and effect diverge across the three request dialects (2026-08-21)

**Lane** `lane/effort-surface-parity-20260821`, branched off `origin/main` @ `e5f8f50ecc`
(v0.100.0 line; the brief named v0.99.1/3ace0df3b9 but main had moved — the delta is
kernel/docs only, `crates/memra-server` byte-identical between the two, verified with
`git diff 3ace0df3b9..e5f8f50ecc -- crates/memra-server/` = empty).

## Root cause (confirmed, exactly as the issue diagnosed)

- `anthropic.rs::translate()` read ONLY `thinking.type`; `output_config.effort` was never
  read, so it was dropped before `parse_think()` — which is why `/v1/messages` accepted
  every string (bogus/banana/"" → 200) and why no value had any effect. The drop was even
  documented: API-SURFACES.md listed `output_config` under "accepted and ignored".
- `parse_think()` (main.rs) owned the strict allowlist `none|minimal|low|medium|high`
  that `/v1/chat/completions` reaches directly.
- `/v1/responses` (responses_api.rs) had its OWN copy of the table with a deliberate,
  documented clamp `xhigh|max|ultra → high` — the xhigh disagreement was two tables
  drifting, the exact disease.

## Canonical value set decision

`none | minimal | low | medium | high` + clamp aliases `xhigh | max | ultra → high`,
accepted on ALL THREE dialects, owned by ONE function (`canonical_effort`, consulted by
`parse_think` and the /v1/responses translator). Rationale — engine truth + real clients:

- No template distinguishes a level above `high` (`ModelCaps::effort_levels` templates:
  step35 `Reasoning: low|medium|high`, hy3 `no_think|low|high`), so `xhigh` as a distinct
  tier has nothing to ride — alias, not new tier.
- Real default-config clients SEND the aliases: codex sends `reasoning.effort: "xhigh"`
  on /v1/responses; Claude Code sends `effort: "xhigh"` by default on current models via
  /v1/messages (`output_config.effort`, forwarded verbatim by vLLM's Anthropic surface
  too). Reject-everywhere breaks stock CLI sessions that work today; accept-on-some was
  the bug. Clamp-everywhere is the only option that is both consistent and compatible.
- Rejecting on `/v1/responses` would also be a behavior regression for clients passing
  `xhigh` since the surface shipped.

## Precedence (thinking.type × output_config.effort)

`thinking.type` (the documented Anthropic lever) wins the on/off switch when both are
present; the effort value is ALWAYS validated (invalid → 400 even next to an explicit
switch) and still supplies the level for level-consuming templates. Implemented once in
`parse_think` as explicit-switch precedence (`reasoning.enabled` — which `thinking.type`
translates onto — beats the switch an effort level implies), so the same rule now also
governs the OpenRouter object form on the chat surface. Anthropic's own API treats
`thinking` as mode and effort as depth (they coexist; effort has no "off" value there),
so switch-owns-on/off + effort-owns-level is the faithful mapping of their semantics
onto our binary-switch templates. Side effects on the chat surface, deliberate:
`{"enabled": true, "effort": "none"}` now thinks (explicit switch wins);
`{"enabled": false, "effort": "banana"}` now 400s (the old `enabled==false` early-return
skipped validation — the same silent-accept class as the /v1/messages hole).

## Changes

| file | change |
|---|---|
| `crates/memra-server/src/main.rs` | `canonical_effort()` (the ONE allowlist), `parse_think()` validate-then-precedence rewrite, worker-truth tap extension (`WorkerSaw`), tests |
| `crates/memra-server/src/anthropic.rs` | `translate()` maps `output_config.effort` → `reasoning.effort` (verbatim; parse_think validates) + `thinking.type` → `reasoning.enabled`; non-string effort is a 400 naming the field; tests |
| `crates/memra-server/src/responses_api.rs` | local effort table replaced by `crate::canonical_effort` (same wire behavior, one owner) |
| `docs/API-SURFACES.md` | `/v1/messages` gains the `output_config.effort` row (+ precedence), `output_config` removed from "accepted and ignored"; `/v1/responses` row points at the shared table |
| `docs/SERVING.md` | canonical set + clamp aliases + precedence + "level knob is connected only on level-consuming templates" caller note |

## Tests with teeth

- `same_effort_value_resolves_identically_on_every_surface` (main.rs): drives the three
  REAL handlers per value; accepted rows compare resolved `(ThinkMode, effort_level)` at
  the WORKER boundary (extends the v0.98-line worker-truth four-surface pattern of
  `same_omitted_request_resolves_identically_on_all_four_surfaces`); rejected rows assert
  the same 400 on all three + each surface's documented error envelope. Re-dropping the
  parameter fails every row with a message naming issue #31.
- `output_config_effort_flows_to_the_one_reasoning_surface` (anthropic.rs): every string
  reaches `reasoning.effort` verbatim (a second table in the translator is how surfaces
  drift), precedence emission, non-string type error, other `output_config` fields still
  ignored.
- `reasoning_effort_maps_to_think_switch` extended: clamp aliases, explicit-switch
  precedence, invalid-next-to-switch rejection rows, `canonical_effort` table pin.
- `reasoning_effort_maps_to_effort_level_on_step35_class_templates` extended: aliases
  render as `high` on level-consuming templates.

## Gates (all on this branch @ c65b34dea5, rig 5090, 2026-08-21)

- `cargo test -p memra-server`: **336 passed, 0 failed** (dev, and release inside local-ci).
- `tools/local-ci.sh --perf-quick`: **GREEN end to end** — unit suite 336/0, drafter-attach
  wiring ALL GREEN, kernel-check ALL GREEN (106 cells, 1 skipped), argmax-margin-gate PASS
  (31B, calibrated), serve-smoke all arms, serve-stress-gate ALL GREEN (c=64), accept-gate
  q27-p1 PASS (accept=0.6825, sha-identical long text), SPEC-ON-CACHE-HIT ALL GREEN
  (mtp + gemma arms), **perf stage 0 fail 0 warn**: 31b-plain-short 42.23 tok/s,
  31b-plain-d1736 39.77, 31b-spec-short 107.96 (accept .798), 31b-spec-d1736 102.22
  (accept .817) — rows banked in research/tune-data/perf-ci.jsonl.
- Battery scheduling notes (surfaced, not this lane's to fix): first attempt OOM'd at
  prime-gate under a co-resident foreign 10GB process — local-ci runs prime-gate with NO
  lock wrapper (main.rs battery line ~187), a fail-open window when another lane boots a
  server outside `/tmp/memra-5090.lock`. Second attempt under a whole-run outer lock broke
  `spec-on-cache-hit-gate`'s internal `flock -w 300` on the same file (self-deadlock,
  empty server log, "server died during boot"). Green run = outer whole-run lock +
  `MEMRA_CI_LOCK_HELD=1` + `MEMRA_GPU_LOCK=<private file>` so the hitgate's internal flock
  is uncontended inside the exclusive window.

## E2E probe — before/after (issue's own table)

Before = the issue's live-prod measurements (2026-08-21). After = local memra-server on
this branch @ c65b34dea5, `Qwen3.8-27B-NVFP4-Q5K-mtp.gguf` (the production artifact), rig
5090, temperature 0, functional/exactness only (no timing claims). Raw:
`/tmp/effort-probe-results.txt` run 2026-08-21T15:14Z, banked below.

### Validation matrix (HTTP status: before → after)

| value | `/v1/chat/completions` | `/v1/responses` | `/v1/messages` |
|---|---|---|---|
| none/minimal/low/medium/high | 200 → 200 | 200 → 200 | 200 → 200 |
| xhigh | **400 → 200 (clamp)** | 200 → 200 | 200 → 200 |
| max / ultra | 400-class → 200 (clamp) | 200 → 200 | 200 → 200 |
| bogus | 400 → 400 | 400 → 400 | **200 → 400** |
| banana | 400 → 400 | 400 → 400 | **200 → 400** |
| "" | 400 → 400 | 400 → 400 | **200 → 400** |

### Effect (after; before, /v1/messages was 269 thinking chars for none/minimal/high alike)

| probe | result |
|---|---|
| /v1/messages effort=none | thinking_chars=0, blocks=[text] |
| /v1/messages effort=minimal | thinking_chars=0, blocks=[text] |
| /v1/messages effort=high | thinking_chars=244, blocks=[thinking, text] |
| /v1/messages thinking=disabled | thinking_chars=0 (lever unchanged) |
| /v1/messages thinking=enabled + effort=none | thinking_chars=244 (thinking.type wins) |
| /v1/messages thinking=enabled + effort=banana | 400 (validated even next to the switch) |
| /v1/chat/completions effort none / high | reasoning_chars 0 / 244 |
| /v1/responses effort none / high / xhigh | reasoning_chars 0 / 244 / 244 |

Same prompt, same artifact: all three surfaces produce the same 244 reasoning chars at
`high` and the same suppression at `none` — the effect half of the identity, live.

### Error shapes (invalid value, after)

- chat: `{"error":{"message":"bad reasoning_effort \"banana\" (none|minimal|low|medium|high; xhigh/max/ultra clamp to high)","type":"invalid_request_error",...}}`
- responses: `{"error":{"message":"reasoning.effort \"banana\" is not supported","type":"invalid_request_error","param":"reasoning",...}}`
- messages: `{"type":"error","error":{"type":"invalid_request_error","message":"bad reasoning_effort \"banana\" (...)"},"request_id":"msg_..."}`
