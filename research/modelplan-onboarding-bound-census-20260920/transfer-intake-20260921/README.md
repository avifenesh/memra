# Reviewed transfer/identity stack intake — 2026-09-21

This is an actual issue-local merge of reviewed #542
`68e5e520810fd105bf64f43048a3a88514f6a064` into a separate derivative of frozen #541
`285760cfc141af90183097fc68872bff8fce7b20`. Both older #541 worktrees remain clean/frozen.
No moving upstream tip was independently merged, and no imported native receipt qualifies this
new combined tree.

The incoming lockstep router program, transfer/cancellation source-and-destination retention,
KV plane ownership/reclaim, exact expert-bank budgets, owner proxy Send/Sync checks, and current
qualification-runner contract are retained. The only overlapping engine file, `moe_cache.rs`,
auto-composed its new constructor/budget/ownership guards with #541's opaque disk-reader and
keepalive variants. Source deltas against both parents are retained for review.

Of 1,117 paths changed by the dependency since `222d50405`, 1,114 have identical staged blob IDs
to `68e5e520`. The three composed paths are `moe_cache.rs`, the research index (both lanes kept),
and the synthetic SLRU fixture. Only that fixture's source pin conflicted. Regeneration through
the existing script and an independent comparison proved all 2,013 decisions, 256 serial rows,
and every non-pin field identical to BOTH parents. No oracle decisions were altered.

## Validation

- `cargo test --locked -p memra-gguf -p memra-cli -p memra-tier -p memra-kv`: passed.
  Includes 360 GGUF tests (2 ignored), existing CLI/inspector/Step tests, external authority and
  compile-fail controls, 200 tier checks including all 53 storage tests, and the KV suites.
  Artifact-dependent omissions do not constitute native qualification.
- Same four crates, all-target Clippy with warnings denied: passed.
- Actual-source CPU identity/snapshot/repack harness: 27 passed.
- Actual-source model memory fixture and device-memory harnesses: 10 and 11 passed.
- Incoming #537 qualification controls: 13 passed. #542 caller, build provenance, finalization,
  and signal controls: 17/33/7/3 passed respectively. The environment-controller suite ran 27
  cases with five Linux-only checks skipped on macOS. These commands use CPU fixtures, not a GPU.
- Linux-target engine/server lib/bin/test Clippy with warnings denied: passed under `DOCS_RS=1`
  documentation stubs. Incoming transfer/router code and #541 I/O interfaces typecheck together;
  actual CUDA, O_DIRECT, kernel, serving and performance gates remain pending.
- Formatting and flags census passed. Net staged delta against reviewed `68e5e520` passes
  whitespace checking. First-parent diagnostics concern 15 inherited raw files, each independently
  byte-verified identical to that dependency. They are preserved without trimming or overrides.

All commands' logs, runner statuses, source hashes, parent diffs and preservation comparisons are
losslessly archived here. No skip override, rental, root activation, support promotion or main
merge was performed. Bounded composition review is required on this exact merge before activation.

Remaining implementation still includes scoped CPU expert ABI/cache/direct/mirror/detached I/O,
overlay/pruning composite contracts and identity, standalone MTP/student/trim contracts, private
canonical repack adoption, and root closure. The incoming experts-via-tier gate also accepts a raw
GGUF separately from the model; its source boundary must be included in the final activation audit.
