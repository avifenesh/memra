# WP-D day 8 — collector regressions and native legacy gates

Repository: **memra**, branch `lane/spill-d-20260919`. This is a pushed lane
milestone, **not merged, released, deployed, or multi-card qualified**.

## Source and delivery

- First pushed the inherited `417e2e40cb35f87caa0c559969369580e8c9b0ed` G2 N=1
  plumbing receipt, with hooks enabled.
- Merged `origin/lane/spill-integ4-20260920` (`5ffe935c`) with `--no-ff` and
  pushed `ff50258b`. No main checkout or other lane was modified.
- Refusal/storage binding: `a8ce78f2`, compatibility follow-up `b7ca59fc`.
- Runbook/initial checks: `e5fc0fa2`.
- Legacy native receipts: identity `43468e1f`, teeth/build `086b5984`, failure
  cells `23722ec4`; bare-control comparison and native storage wrapper `f385e336`.
  Subsequent commits contain the integrated check receipts and this summary.

## Native legacy gates: both bare and collector outputs

Built `memra-server` in D's own clone from B's applied HostPrefix-v2 source
**`a7358f9acd2be24d3f1443e18209732442024bed`**. Build completed with exit 0.
Binary SHA256:
`c986c6ee9f0fad9ad50a04d6c82f4fc9c80bc77ad9e20ab69153d561cfe7b7c6`.
Model: `Qwen3.8-27B-NVFP4-Q5K-mtp.gguf`; freshly read SHA256:
`1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a`, matching
B's baseline artifact identity. No artifact bytes were written.

All three legacy cells ran through the collector at **`a8ce78f2`**, with an
inherited `/tmp/memra-5090.lock` FD, verified by `tier-lock-proof.py`. The
external-lock fragment was already applied in D; reverse-apply checking verified
that fact before execution. Copied scripts and their hashes are retained. Each
cell had empty before/after compute-app snapshots, no timeout, and exit **0**.

| Cell | B's bare output, verbatim | D collector output, verbatim |
| --- | --- | --- |
| Identity | `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` | `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` |
| Forced-tiny teeth | `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=1)` | `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=1)` |
| Pool-full / corrupt-digest / alloc-fail | `KV-HOST-SPILL FAILURE GATE: ALL GREEN` | `KV-HOST-SPILL FAILURE GATE: ALL GREEN` |

Raw collector logs: `day8/native/legacy-{identity,teeth,failures}/cell/command.log`.
Server logs, requests, metrics, lock proofs and sampled CSVs sit beside them under
`state/` and `cell/`. B's original `legacy-identity-r3`, `legacy-teeth-r2` and
`legacy-failures` driver logs and CELL-NOTEs are copied byte-for-byte under
`day8/bare-controls/`, with exact source paths/revision/hashes in `provenance.json`.
The original bare controls used B source `b569164b` with its scratch HostPrefix-v2
patch; they did not use the collector or 250 ms telemetry. These are outcome
comparisons, **not timing or performance comparisons across builds**.

Diagnostic conditions match B's passing controls: **256 MiB is
`MEMRA_HOSTGATE_CACHE_MB` (device prefix cache)**, not the host-tier budget.
Identity uses tenant percentage 50; teeth/failure use 100. Identity/failure keep
the 8192 MiB host-tier default; teeth forces 1 MiB. The collector retained 116,
121 and 163 GPU samples respectively, requested every 250 ms. Every observed
power-limit pair was **400.00 W / 600.00 W**. No cap was changed.

These are real single-card, restricted-power development legacy whole-prefix
correctness cells. They do not qualify generic active TierManager binding,
NVMe spilling, multi-card transport, a full serving battery, or performance.

## Refusal and storage-root fixes

`CAPTURE-CONTRACT.md` defines the terminal token contract: exit 2, no timeout,
final line `REFUSED: <reason>` or `kv-tier-gate: REFUSED: <reason>`. Both are
retained verbatim. Generic `Error:` + exit 2 is **failed**, not refused. Tests
cover both accepted spellings, generic errors, nonterminal tokens, wrong exit
status and timeout. Historical failed capture statuses are not rewritten.

The collector now resolves the canonical one-command `bash -c` storage wrapper
before applying the root guard; opaque/noncanonical storage shell commands fail
closed. It preserves storage-bench's optional byte/backend arguments. The
backend object directory must be at/beneath the supplied root, on the same stat
device, statfs filesystem id and Linux mount id. Those identities are recorded
in **both CELL rows and the capture**, tied to the original object argument, and
rechecked after execution. Symlink escapes, nested-mount mismatches, changed
bindings and command/binding tampering have CPU regressions.

An actual native 264-byte buffered roundtrip through the canonical `bash -c`
wrapper ran at source **`23722ec4`**. Both `--validate CELL.jsonl` and
`--schema storage-cell` accepted its retained evidence. Raw ancestry and binding
are in `day8/native/storage-wrapper/`; the output join is `storage-join.jsonl`.
It remains **overlay/unproven — not NVMe, not spill speed**. The owned storage
scratch directory was removed after capture. A complete SHA256 manifest seals
98 native receipt files; no instance ids or monetary values are in the journals.

## Runbook and checks

`DAY8-CELLS.md`, linked prominently from `RIG-DAY1.md`, adds exact collector
spellings for A's `tier-transfer-gate conformance|roundtrip`, B's
`kv-tier-gate --case active`, C's `--rows-via-tier`, the prospective combined
`--rows-via-tier --device-publish`, all legacy external-lock cells, and G2 N=1.
Existing B active and C row-tier archives pass current integrity validation;
B's original failed status remains failed, not a fabricated positive gate.

The integrated receipt under `day8/checks-integrated/` runs 25 commands:

- `cargo fmt --all -- --check`;
- macOS and Linux-target offline all-target checks for memra-tier/memra-kv;
- `cargo test -p memra-tier --offline --no-fail-fast` (181 tests across suites);
- memra-kv tests (62), memra-tier clippy with warnings denied;
- Python battery (70 tests), `py_compile`, `bash -n`, bootstrap syntax and
  shellcheck (legacy script shellcheck is also exercised by Python tests);
- archived/current capture validators, fixture pins, campaign integrity,
  metadata targets, native 98-file hash manifest;
- `git diff --check` and `tools/check-flags.sh` (864 literal runtime reads).

Initial and final pre-integration check runs are preserved separately; they are
not native GPU evidence. The native server and storage-bench builds both ran on
Linux. No new dependency, flag, kernel, numerical program or main edit landed.

## Remaining gates and effort

- Full **G2 remains blocked on probe `--copies >1` support**. Current F accepts
  only `--repeats 1 --copies 1`; the inherited N=1 receipt is plumbing, not a
  full 4 KiB–1 GiB bidirectional envelope. No N<5 performance median was emitted.
- A's transfer gate is documented from A's source but not rerun in this lane.
- B's generic active/prefix binding remains unsupported; a refusal is not pass.
- C's inspected tip has no `--device-publish` implementation and can ignore
  unknown arguments. The combined command is explicitly blocked until C adds
  the adapter and strict argument admission; an old green binary cannot prove it.
- NVMe ancestry, multi-card PRO qualification, full model/serving and scored
  performance gates remain pending. Single-card development evidence does not
  waive them.

This relaunch used approximately **0.6 agent-hour**, below its 2.5-hour stop
budget. This is campaign **day 8 of 9**; cumulative prior-session agent-hours
were not reconstructed, so no invented total/remaining-hours estimate is given.
The lane stays open for lead integration; its owned worktree and receipt mirror
are not abandoned scratch. No unrelated worktree changes were absorbed.
