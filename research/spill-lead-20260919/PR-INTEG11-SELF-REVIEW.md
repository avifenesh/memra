# integ11 self-review (lead, 2026-09-21)

Read in full: C day 13 `crates/memra-server/src/worker.rs` diff (door parse, arena and vision and directory refusals,
`host_tier_program_base`, `host_tier_governor`, `host_tier_context`, insert identity check placement),
`crates/memra-kv/src/tiered/hostprefix.rs` (`tenant_salt`, `shared_governor`, tests), FLAGS row, `HOSTPREFIX-DOOR.md`,
`verify-day13.py`; spot-checked the receipts.

## Findings
1. **OFF is byte-identical by construction.** The constructor is reached only when the door parses to `true`; every
   OFF-path statement is unchanged (the diff adds calls behind the door and moves one identity check after the
   existing `skip demote` refusal, which C proved keeps the OFF line). serve-smoke plain + cache-metering lines are
   byte-equal OFF/ON on the target card and `serve-smoke: 0 failed` locally on this tree with the door unset.
2. **Fail-closed door.** `1`/`0` only, junk refuses at boot with a message naming the value; the startup arena, a
   vision tower and a directory checkpoint each refuse at boot with a typed sentence naming the reason; no host tier
   means no construction and one printed line. Nothing falls back to OFF silently.
3. **Identity fields are derived, not declared.** Artifact digest streamed from the GGUF; plan, numeric class,
   stream, tokenizer, template, adapter, modality, position each a labelled digest; `tenant_salt` per pool key through
   the one helper (ruling 13). The governor is server-owned with 2x budgets (C's rationale: the ledger tracks
   pinned and pageable images plus device planes that the prefix budget already bounds); a tighter figure is a
   measured decision for Option B.
4. **Findings honestly kept.** Draft-bearing entries refuse by name under ON (identity gate `5 FAILURE(S)` under the
   default spec env): the door is plain-only today, ruling 16 records it. `tenant_salt("")` derives, with the reason
   (ruling 17). The failure gate's pre-existing `1 FAILURE(S)` is the same OFF and ON and is not this lane's.
5. **Nits (not blocking).** The plan identity is the `Debug` form of `ModelPlan`, so a Debug-derive change moves every
   identity; acceptable for a gate-only door, worth a serialized form before promotion. `sha256_file_hex` reads the
   whole GGUF at boot (7.8 s for 15.7 GB on the card); fine behind a door, a boot-time cost to state before promotion.

## Verification this review relied on
integ11 CPU battery (`integration-day12/integ11-cpu-battery/`): fmt, tier+kv+gguf and memra-server suites, clippy
`-D warnings`, censuses, collector pytest, `verify-day13.py`, perf board, `git diff --check`. Local 5090 serve-smoke on
this tree with the door unset (`integ11-serve-smoke-5090/`). C's target-card OFF/ON table. This rig cannot run the
model gates; the GPU evidence is the lane's target-card cells.
