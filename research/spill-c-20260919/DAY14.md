# Session C day fourteen: the door binds MTP draft-bearing entries (lead ruling 16)

Scope: lead ruling 16 (`research/spill-lead-20260919/INTEGRATION-DAY12.md`): grow
`bind_tier_image` to draft planes before Option B, with the identity and failure gates equal
OFF/ON under the gates' DEFAULT spec environment as the exit criterion. Tree: lane merge of
main `30e433c4c` (`fbe6c1635`), census commit `e82363a8b`, code commit `62f48ecec`. Target-card
cells ran on the `62f48ecec` tree (binary sha256
`7ffa58cb55daaaffba9afe220f59199faa7153bb1d836363a9416b662ab4d184`, release build 3m06s), one
RTX PRO 6000 Blackwell Server Edition at its 600 W limit (`pro-single-day14/card.csv`), through
the canonical collector (`tools/tier-battery.py --rig pro-single`, lock `/tmp/memra-gpu.lock`),
N=1, `executed-not-qualified`. Nothing here is a support state. Door doc: `HOSTPREFIX-DOOR.md`
(the "Draft planes" census, written before code, and the surface table).

## The census, in one paragraph (`HOSTPREFIX-DOOR.md` "Draft planes")

A spec-served prefix entry carries, beyond the plain surface: the MTP draft-scratch K/V rows
`[0..pos)` (`PrefixEntry.draft`, `worker.rs:6467`; bytes owned by `memra_kv::KvLayer`, q8_0 K rows
of 34 B and q5_1 V rows of 24 B per 32 elements, row width from the MTP head's `n_head_kv`;
published by `SpecSession::draft_plane_ref` and `prefix_insert_from_spec_boundary`; copied by the
same `host_plane_from_device` / `plane_up` programs as the trunk planes), the boundary trunk hidden
`last_h` (already `Role::Hidden`), and, only under `MEMRA_DSPARK_SPEC=1`, the DFlash draft KV tail
(`dspark_draft`, f32 planes per draft layer with `base/rows/len/row_bytes/floor`). Two findings
shaped the slice: `MEMRA_KV_HOST_VERIFY`'s digest is blind to the draft plane
(`prefix_entry_state_digest`, `11178-11262`, hashes `kv`, `conv`/`ssm`, `latent` only), and the
draft head's source is knowable at boot for GGUF heads only (embedded, per-model `+draft`, or
`MEMRA_MTP_DRAFT`); the DFlash tail needs an export-directory manifest no GGUF digest provides.

## What changed (`crates/memra-server/src/worker.rs`, commit `62f48ecec`)

| Piece | What it does |
|---|---|
| `HostTierEntryClass`, `host_tier_entry_class(glm, mtp_draft, dflash_tail)` (pure) | `Plain` or `MtpDraft`; refuses BY NAME `entry carries TP or latent (GLM) planes outside the contract-routed surface` and `entry carries a DFlash draft tail outside the contract-routed surface (no drafter artifact identity in this slice)`. The same function at demote, `bind_tier_image`, insert and promote, so one entry names one class. |
| `HostTierDraftSource` (`Embedded`, `External { sha256_hex }`) | The MTP head's provenance, resolved in `host_tier_context` with the loader's precedence (per-model `+draft`, then `MEMRA_MTP_DRAFT`, then embedded); an external file is hashed like the trunk (`sha256_file_hex`). A model without an MTP head gets no draft program. |
| `host_tier_draft_program(plain, artifact_hex, plan_debug, source)` | The plain base with `artifact = digest("artifact-sha256+mtp-draft", framed(trunk hex, source))`, `serialized_plan = digest("plan-debug+mtp-draft", framed(plan, source))`, `numeric = digest("numeric", "<plain class>+mtp-draft-kv-q8_0-34B-q5_1-24B")`; stream, tokenizer, template, adapter, modality, position unchanged; tenant salt zero until `program(key, class)` stamps it. `host_tier_framed_pair` length-frames both halves. |
| `HostTierPrograms { plain, draft: Option, generation }`, `HostTierContext::program(key, class)` | Per model name; typed refusals `tier program identity missing` and `tier draft program identity missing: the model has no MTP head, so a draft-bearing entry cannot name its program`. |
| `bind_tier_image` | Classifies first (refuses by name), binds the draft plane as `Role::Draft` K (`mtp-draft-q8_0`, row bytes `k_tok_bytes`) and `Role::Draft` V (`mtp-draft-q5_1`, `v_tok_bytes`) after `Hidden`, each `checksum(bytes)` in `StateBundle.checksums`; the trunk geometry rule (34/24 multiples, `len == pos`) applies to the draft plane unchanged; shape blob `host-prefix-shape-v2` (`host_tier_shape_metadata`, pure) frames draft presence and `(len, k_tok_bytes, v_tok_bytes)` unconditionally. |
| `tier_charge(key, class, ..)`; demote | The `pinned` sum at demote covers trunk KV planes and the draft plane (both `HostPlane`s); the refusal line names the plane class. |
| insert, promote | Class from the entry's own fields; the promote holds `(lease, program, generation)` and re-`require`s after the H2D with the same program. |
| `host_tier_context` boot line | After the day-13 identity line: `[prefix-host] contracts door: model <name> draft program identity source=embedded numeric=server-prefix-entry-v5-kv-q8_0-34B-q5_1-24B+mtp-draft-kv-q8_0-34B-q5_1-24B (0ms)` (or `... has no MTP head: no draft program, draft-bearing entries are refused by name`). |

Packed bytes stay as they are: the D2H and H2D are the pre-door `dtoh_u8_into_pinned` /
`htod_u8_into` for every plane including the draft; no numeric or copy program was added. OFF is
byte-identical by construction: every changed statement is inside a `tier`-Some block
(`if host.tier.is_some()`, `let Some(tier) = &self.tier`, `if let Some(tier) = &self.tier`) or a
new function only those blocks call (`git diff 4d1ed8d54..62f48ecec -- crates/memra-server`); the
promote's `tier_class` is `None` with the tier absent and every arm that reads it is skipped.

## Target-card receipts (`pro-single-day14/`, replay `verify-day14.py`: `DAY14 REPLAY: PASS`)

Every cell: door OFF then ON, same binary, same prompts, same environment otherwise. `default` =
the gate's own environment (this MTP artifact serves speculatively; every prefix insert is a
`spec-boundary` capture with a draft plane; lines 42, 50, 67 and 80 of the ON server log are
`insert probation (spec-boundary)`, zero plain-published inserts). `plain` = `MEMRA_SERVE_SPEC=0`
added (the day-13 surface, regression). `MEMRA_HOSTGATE_CACHE_MB=256` for the four host-gate
pairs (day-13 finding 1). Verdict lines verbatim.

| Gate | OFF | ON | Equal |
|---|---|---|---|
| `tools/serve-smoke.sh` plain + cache-metering arms | 31 `ok:` lines, `ok: cache-metering accounting exact (per-request + /metrics + economics)`, `serve-smoke: 0 failed` | the same 33 verdict lines | yes |
| `tools/kv-host-spill-identity-gate.sh`, **default (exit criterion)** | `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)`, 12 `ok:` | `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)`, the same 12 `ok:` | yes, 13 lines |
| same, `[prefix-host] demote:` byte counts | `89 tokens, 160.7MB`, `86 tokens, 160.6MB`, `102 tokens, 161.1MB`, `102 tokens, 161.1MB` | identical | yes |
| same, promote and verify | `verify ok ... (89 tokens)`, `promote: 89 tokens, 159.7MB`, `verify ok ... (102 tokens)`, then `[prefix-cache] skip pinned host-promote insert: entry 160.1MB would evict protected bytes below their 215MB share (need 51.4MB, probation/demotable 0.0MB)` | identical: 1 promote, 2 `verify ok`, 1 named device-side skip | yes (finding 2) |
| same, whole `[prefix-cache]`/`[prefix-host]` event sequence, timings stripped, door boot lines excluded | 23 events | the same 23 events | yes |
| same, r1/r3 texts and cached tokens | r3 `cached_tokens=89` of `prompt_tokens=102` | identical texts, `cached_tokens=89` of 102 | yes |
| same, ON refusal lines | none | `0` lines matching `contracts door\): \|REFUSED\|refused` | the day-13 `5 FAILURE(S)` is gone |
| same, OFF twin boot (`MEMRA_KV_HOST_MB=0`) | 0 `[prefix-host]` lines | 0 `[prefix-host]` lines | yes |
| `tools/kv-host-spill-identity-gate.sh`, plain | `ALL GREEN`, demotes `89 tokens, 160.5MB`, `86 tokens, 160.4MB`, 1 promote, 1 `verify ok` | identical | yes (day 13 held) |
| `tools/kv-host-spill-failure-gate.sh`, **default (exit criterion)** | `KV-HOST-SPILL FAILURE GATE: 1 FAILURE(S)` (`FAIL: pool-full refusal is LOUD and named`, finding 3 of day 13, pre-existing) | `KV-HOST-SPILL FAILURE GATE: 1 FAILURE(S)`, the same 15 lines; poolfull (2), digest (5), alloc (2) tier event lines identical with timings stripped; `FAULT: flipped one demoted K byte` and `VERIFY FAILED` in both digest cells; `TIER DISABLED: pinned host alloc` in both alloc cells; 0 refusal lines in the ON cells | yes; the day-13 `6 FAILURE(S)` is gone |
| `tools/kv-host-spill-failure-gate.sh`, plain | `1 FAILURE(S)` (the same line) | identical | yes (day 13 held) |

Collector: every cell `capture-integrity`, `qualification: false`, `gpu_power_limits` 600.00 W
(`validate.log`, `--validate` per cell, `validate.exit` 0; `status=failed` is the collector's word
for a red gate's nonzero exit, and the four failure-gate cells are red on the one pre-existing
line in both arms).

### Findings

1. **Entry size with a draft plane.** Draft-bearing entries are 160.7 MB (89 tokens), 160.6 MB
   (86) and 161.1 MB (102) against the plain 160.5 MB and 160.4 MB: the MTP draft plane adds about
   0.2 MB at these lengths on this artifact. The 256 MB device budget still holds one entry, which
   is the gate's contract.
2. **The second promote is declined by the device cache, in every arm.** Under the default
   environment r3 publishes the 102-token boundary entry, the device evicts and demotes it, and r4
   promotes it back: `verify ok (102 tokens)` then `[prefix-cache] skip pinned host-promote insert:
   entry 160.1MB would evict protected bytes below their 215MB share (need 51.4MB,
   probation/demotable 0.0MB)`, and r4 serves the 89-token device hit. Identical in OFF and ON and
   in day 13's OFF-default cell (`pro-single-day13/hostgate-identity-off-default`, lines 71-72): the
   protected-share rule at `MEMRA_HOSTGATE_CACHE_MB=256`, not a door effect. `verify-day14.py`
   accounts every `verify ok` as a promote or that named skip and requires the counts equal across
   arms (my first replay asserted `verify ok == promotes` and failed on exactly this line; the
   assertion was wrong, the receipts were not).
3. **Hash cost.** 7814 ms to hash the 15.7 GB trunk (day 13: 7837 ms); the embedded draft adds no
   file and no time (`(0ms)` on the draft line).
4. **The verify digest attests the trunk only.** `verify ok` under the default environment proves
   the trunk planes and recurrent state round-tripped; the draft plane's byte receipt is the pair
   of `Role::Draft` checksums in the entry's `StateBundle`, checked in-process by `IdentitySlot`.
   A `memra-prefix-split-state-v3` digest covering the draft plane would change the OFF arm's
   printed digests, so it is a separate slice with its own gate line (census finding 2).

## Local battery (this rig, `systemd-run --user --scope -p CPUQuota=1200% -p MemoryMax=28G`)

On `62f48ecec` (plus the docs in this commit): `cargo fmt --all -- --check` clean; `cargo test -p
memra-server -p memra-kv --offline` 744 + 70 + 2 passed, 0 failed, 6 ignored (GPU); the four new
tests `host_tier_entry_class_admits_plain_and_mtp_draft_and_refuses_glm_and_dflash_by_name`,
`host_tier_draft_program_differs_from_plain_in_exactly_artifact_plan_and_numeric`,
`host_tier_context_program_selects_the_class_and_refuses_a_draft_entry_without_a_head`,
`host_tier_shape_metadata_v2_frames_the_draft_plane_presence_unconditionally` `ok`, the day-13
door tests still `ok`; `cargo clippy -p memra-server -p memra-kv --offline --all-targets -- -D
warnings` clean (one `collapsible_if` fixed before the commit); `tools/check-flags.sh` 867 runtime
names, none uncovered (no new `MEMRA_*` read: `MEMRA_MTP_DRAFT` already has its row);
`tools/docs-registry-census.sh` clean (58 tables, 903 rows; a first pass caught an unescaped `|` in
the FLAGS row, fixed); `git diff --check` clean. No `DOCS_RS=1` command ran.

## What remains

- **Option B** (ruling 14, after these receipts): D2H demotion through `TransferEngine` on the
  pageable path with owned `KvPlane`s, `HostPlane` gaining a `CudaPinnedLease` arm, the
  `StateBundle` checksums moving onto `SegmentExpectation` so `Completion::require` proves the
  bytes; the draft plane rides the same route as two more segments. Its admissibility gate is this
  day's table plus per-segment `Completion` checksums equal to `StateBundle::verify`.
- **DFlash tail slice** (refused by name today): an export-directory byte manifest as the drafter
  artifact, `DflashCfg` as the plan, an f32 tail numeric class, `Role::Tail` segments per draft
  layer, and a target-card gate that boots `MEMRA_DSPARK_SPEC=1` (none does; no DFlash checkpoint
  is staged on the target card).
- **Verify digest v3** covering the draft plane (finding 4), its own slice and gate line.
- **Pool-full failure-gate line** (day-13 finding 3): stale relative to main's tenant share cap,
  red in both arms, not this lane's.

Effort: approximately 4 agent-hours (budget 8).
