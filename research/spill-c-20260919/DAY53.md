# WP-C day 53 (2026-09-24): verify digest v3, the draft plane and the boundary rows inside `MEMRA_KV_HOST_VERIFY`

`OWED.md` C6. Source: `HOSTPREFIX-DOOR.md` "Draft planes" finding 2 ("The verify arm is blind to the draft plane
... Extending the digest (`memra-prefix-split-state-v3`) would change the digest strings the OFF arm prints in
`VERIFY FAILED` lines, so it is not part of this slice; recorded here as the follow-up it is") and section D item 3.
Written before any code; tree at start: `094c46b24`. No sibling lane touches the functions below
(`git diff origin/main...origin/lane/spill-{a,b}-20260919 -- crates/memra-server/src/worker.rs` names none of them).

## 0. Census, today's tree (`crates/memra-server/src/worker.rs`)

- The verify arm has one digest, `host_roundtrip_digest`: taken on the device entry at demote (before the image is
  built, stored as `verify_digest`, carried by the handoff file) and on the re-materialized device entry in
  `host_promote_finish`; a different string prints `[prefix-host] VERIFY FAILED: promoted digest X != demote digest
  Y`, drops the host entry, counts `digest_mismatches`, serves cold.
- For a GLM entry (`tp` or a latent plane) it is `host_glm::digest` (`memra-host-prefix-state-v1`), which already
  covers the logits, the hidden row, the draft plane and the DFlash tail. For every other entry it is
  `prefix_entry_state_digest(entry, entry.pos)` (`memra-prefix-split-state-v2`): per layer the KV planes at the
  boundary, conv and ssm, the latent snapshot. Nothing else.
- What a non-GLM round trip carries beyond v2, each consumed on restore: the MTP draft plane (`draft`, a spec restore
  re-arms the head from it), the boundary hidden row (`last_h`, the spec session's anchor), the boundary logits
  (`last_logits`, the first token of an empty-suffix resume), the DFlash tail (`dspark_draft`). A corrupted byte in
  any of them promotes with `verify ok`.
- `prefix_entry_state_digest` has two more consumers that must not move: the split-state trace
  (`trace_prefix_entry_state`) and the HIRADIX restore oracle, which compares it with `prefix_cache_state_digest` of
  the restored `Cache`; a `Cache` holds no draft plane, hidden row or tail, so v2 cannot grow.

## 1. Pre-registration

**The design.**

- (a) **v3, composed, v2 untouched.** A new `prefix_entry_roundtrip_digest`: SHA-256 over the domain
  `memra-prefix-split-state-v3`, the v2 digest string of the entry at its boundary (the trunk, by the unchanged v2
  function, so v2's own output cannot move), then in order: the draft plane (a presence byte; `len`, `k_tok_bytes`,
  `v_tok_bytes`; the K bytes `[0..len*k_tok_bytes)` and the V bytes `[0..len*v_tok_bytes)`, the logical window the
  host copy holds, each with its length), `last_h` and `last_logits` (each as `digest_f32_plane`: length, then
  every word's bits), the DFlash tail (a presence byte; `base`, `rows`, `len`, `row_bytes`, `floor`, the layer
  count; per layer the whole K and V buffers as f32 planes, the buffers the round trip copies whole).
- (b) **Tagged strings.** `host_roundtrip_digest` returns `split-state-v3:<hex>` for a non-GLM entry and
  `host-prefix-state-v1:<hex>` for a GLM entry (the GLM digest itself unchanged).
- (c) **A carried digest from another program is typed.** At promote, if the carried string's tag differs from the
  one this binary computes (an untagged v2 string from a handoff file written by an older binary counts as the tag
  `untagged`): `[prefix-host] VERIFY FAILED: carried digest program {carried} != this binary's {ours} (a handoff
  written by another binary); host entry dropped, cold path serves`, the entry dropped and counted exactly as a
  mismatch. Same tag, different hex: today's line, unchanged but for the tagged strings. No wire change: the handoff
  carries the digest as a string today.
- (d) **Three red arms of the check**, new values of the existing fault door `MEMRA_KV_HOST_FAULT`:
  `flip-demote-draft`, `flip-demote-hidden`, `flip-demote-logits`. Each flips the first byte of the host copy's
  draft K plane, hidden row or logits row after the demote digest was recorded, and prints `[prefix-host] FAULT:
  flipped one demoted {draft K|hidden|logits} byte (MEMRA_KV_HOST_FAULT=...)`. They apply where `flip-demote`'s own
  whole-image point is and only on the legacy copy path (no contract route), which is where the verify digest is
  the only byte attestation; under `MEMRA_KV_HOST_CONTRACTS=1` the door's receipts attest those planes and the
  values apply nowhere (no line, so a cell under the door fails, the convention of the other one-shot faults). An
  image with no draft plane or an empty hidden row gets no flip and no line.
- (e) Docs in the same change: `docs/FLAGS.md` rows `MEMRA_KV_HOST_VERIFY` (what v3 covers) and
  `MEMRA_KV_HOST_FAULT` (the three values, gate-set red arms reviewed with the door's decide-by 2026-10-05 like the
  other values); `HOSTPREFIX-DOOR.md` finding 2 and section D item 3 point here.
- (f) **The failure gate grows three cells** (`tools/kv-host-spill-failure-gate.sh`), each its own boot beside
  `digest`: `digest-draft`, `digest-hidden`, `digest-logits`, with verify on and the fault. Spec-served entries
  (the gate's default boot): the FAULT line, a `demote:` line, `VERIFY FAILED`, no `promote:` line, zero promotions,
  r3 byte-equal to the pool-full reference. Under `MEMRA_SERVE_SPEC=0` (plain-published entries: no draft plane, no
  hidden row) the draft and hidden cells assert the opposite: no FAULT line, `verify ok`, a `promote:` line (the
  fault touches nothing else); the logits cell is the same in both modes.

**Correctness and acceptance.**

- CPU (under the 1200% cap): `cargo test -p memra-server --lib` green; unit tests of the tag parse and compare
  (equal, different hex, different tag, untagged); a source census pinning that `host_roundtrip_digest`'s non-GLM
  arm calls `prefix_entry_roundtrip_digest`, that the v2 body is byte-identical to the base tree's, and that the
  three new flips sit only on the legacy copy path; clippy `-D warnings`; `cargo fmt --check`; the flags census.
- GPU unit cell (`worker::tests`, ignored without a device; run on the RTX 5090 inside a lock hold): a synthetic
  entry with trunk planes, a draft plane, a hidden row, logits and a DFlash tail. v3 equal on an identical copy; v3
  different after one byte of each of the five changes (a trunk K byte, a draft K byte, a hidden word, a logits
  word, a tail word); v2 different after the trunk byte and EQUAL after each of the other four (the finding, now a
  test).
- RTX 5090 gates on the 9B, each through the canonical lock (the day-24 driver shape, `day53-cell.sh`): the failure
  gate `ALL GREEN` with the three new cells in the default (spec) mode and in plain mode; the identity gate `ALL
  GREEN` (with `verify ok` and no `VERIFY FAILED`) door OFF and door ON (`MEMRA_KV_HOST_CONTRACTS=1`), the door-ON
  arm being the v3 digest across the contract route's round trip of the draft plane and boundary rows.
- The target card: the same failure and identity cells on the 27B, added to the DAY52 sitting as its own section
  before that sitting runs.

**What each card decides.** Each card's gates pass or fail on that card; no number is compared.

## 1a. Addendum, with the code and before any gate ran

- **The failure gate's door-ON arm.** Section 1 (d) said a cell under the door fails because the values apply
  nowhere there. Other lanes run this gate with `MEMRA_KV_HOST_CONTRACTS=1` (the door-ON arm, `docs/TESTING.md`),
  so that would turn their door-ON runs red for a reason that is not a defect. Settled before any run: with the
  door on, the three cells assert the no-flip outcome (no FAULT line, `verify ok`, a real promote, r3 byte-equal
  to the reference), which is also the v3 digest's check across the door's round trip of the draft plane and the
  boundary rows. No other term moves.
- **The 5090 cells** (`day53-cell.sh`, the day-24 driver shape, the 9B
  `Qwen3.5-9B-NVFP4-MTP-GGUF.gguf`, device prefix budget 64 MB as day 24): `failure-default-off`,
  `failure-plain-off`, `failure-default-on`, `identity-default-off`, `identity-default-on`, and `unit-server` (the
  GPU cell `verify_digest_v3_covers_every_round_tripped_plane_and_v2_stays_trunk_only`). The pool-full cell's line
  is the day-24 known `1 FAILURE(S)` shape only if it recurs; every new check must pass, and each cell's failing
  checks are quoted.

## 2. First attempt on the target card (BOX8, DAY52 section 4; receipts `pro-single-day52/c6-attempt1/`)

The six cells ran 23:32Z to 23:36Z on the box's `memra-server-v3` (`172123b7...`, tree `256c3c640`), the 27B, the
256 MB device prefix budget. Verbatim verdicts (`c6-attempt1/<cell>/verdict.txt`):

- `failure-default-off`: `KV-HOST-SPILL FAILURE GATE: 3 FAILURE(S)`
- `failure-default-on`: `KV-HOST-SPILL FAILURE GATE: 3 FAILURE(S)`
- `failure-plain-off`: `KV-HOST-SPILL FAILURE GATE: ALL GREEN`
- `identity-default-off`: `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)`
- `identity-default-on`: `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)`
- `unit-server`: not run: the cell wraps the test in `systemd-run --user --scope`, and the box has no systemd
  (`Failed to connect to bus: No medium found`).

**Why the two failure runs fail.** In each, the three failing checks are the same one per new cell, `digest-<plane>:
the default boot published spec entries (the plane exists)`, and every other check of the cell is `ok`: the FAULT
line, the demote, `VERIFY FAILED: promoted digest split-state-v3:... != demote digest split-state-v3:...`, no promote,
zero promotions, r3 byte-equal to the reference (door OFF), and the door-ON no-flip checks. The check grepped
`[prefix-cache] insert (spec-snapshot)`, a string I took from a trace role; the spec publisher's insert line reads
`[prefix-cache] insert (spec-boundary): 64 tokens, 158.9MB ...`. A wrong literal in the gate, not a verify defect; the
verdicts stand as recorded. Fixed in the gate (`insert (spec-boundary)`) and in the unit cell (the 1200% cap as a user
scope where systemd runs, else 12 pinned cores), both before any rerun; the three cells rerun on the target card
(attempt-1 receipts kept) and run first time on the RTX 5090 with the fix.

## 3. Results on the RTX 5090 Laptop GPU (`rtx5090-day53/`)

The six cells ran 00:58:43Z to 01:01:42Z on `memra-server-v3` `709f079c...` (tree `256c3c640`'s crates; the cells'
scripts at `422a0d470`, the fixed gate literal and unit cap), the 9B, the 64 MB device prefix budget, each gate
taking `/tmp/memra-5090.lock` itself. Verbatim verdicts (`<cell>/verdict.txt`) and each gate's check counts:

- `unit-server`: `test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 939 filtered out; finished in 0.26s`
  (`test worker::tests::verify_digest_v3_covers_every_round_tripped_plane_and_v2_stays_trunk_only ... ok`)
- `failure-default-off`: `KV-HOST-SPILL FAILURE GATE: ALL GREEN` (36 ok, 0 FAIL)
- `failure-plain-off`: `KV-HOST-SPILL FAILURE GATE: ALL GREEN` (31 ok, 0 FAIL)
- `failure-default-on`: `KV-HOST-SPILL FAILURE GATE: ALL GREEN` (33 ok, 0 FAIL)
- `identity-default-off`: `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` (12 ok)
- `identity-default-on`: `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` (12 ok)

Every term of sections 1 and 1a holds on this card: in the spec mode each new cell's flip is announced and refused
`VERIFY FAILED: promoted digest split-state-v3:... != demote digest split-state-v3:...` with zero promotions and
reference bytes; plain entries flip nothing for the draft and hidden cells and promote with `verify ok`; the door-ON
arm flips nothing and verifies ok across the contract route.

## 4. Results on the target card, the rerun (BOX8; receipts `pro-single-day52/c6/`)

The six cells ran again 01:11:43Z to 01:18:37Z with the fixed gate literal and the unit cell's 12-core cap
(`cap=taskset -c 0-11`), the same `memra-server-v3` `172123b7...`, the scripts at `4417bbd1b`, the 27B, 256 MB.
Verbatim verdicts and check counts:

- `unit-server`: `test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 939 filtered out; finished in 0.26s`
- `failure-default-off`: `KV-HOST-SPILL FAILURE GATE: ALL GREEN` (36 ok, 0 FAIL)
- `failure-plain-off`: `KV-HOST-SPILL FAILURE GATE: ALL GREEN` (31 ok, 0 FAIL)
- `failure-default-on`: `KV-HOST-SPILL FAILURE GATE: ALL GREEN` (33 ok, 0 FAIL)
- `identity-default-off`: `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` (12 ok)
- `identity-default-on`: `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` (12 ok)

C6's acceptance holds on both cards. OWED C6 closes.
