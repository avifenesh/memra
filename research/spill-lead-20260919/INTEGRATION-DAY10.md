# Integration — day 10 (`lane/spill-integ5-20260920`, replay onto current `main`)

Lead: successor of @agent-c07799 (session moved from the operator Mac to the Linux rig 2026-09-20 ~16:00Z).

## Replay
`origin/main` moved `fdb78136` → `f79b3e57` (#555 GLM TP device ownership, #558 Qwen2 tokenizer, INDEX rows).
`git merge --no-ff origin/main` into integ5 at `5f386c72` → `4f36a0e7`; no conflicts (`git merge-tree`
clean; the only file both sides touched was `crates/memra-engine/src/lib.rs`, auto-merged). INDEX.md rows
are the union.

## CPU battery on the replayed tree (`4f36a0e7`, Linux rig, `--offline`, CPUQuota 1200%)
Raw logs: `integration-day10/cpu-battery/*.log`, summary `SUMMARY.txt`.
`cargo fmt --check` OK · `cargo test -p memra-tier -p memra-kv`: 261 passed / 0 failed (63 kv + 2 + 2 + 58 bank
+ 55 contracts + 18 peer + 6 placement + 53 storage + 4 doctests) · clippy `-D warnings` (engine, server, tier,
kv, gguf; all-targets, `x86_64-unknown-linux-gnu`, `DOCS_RS=1`) clean · `check-flags.sh`: 864 runtime names, no
uncovered · publish census 12/12 · docs-registry census OK · collector Python suite 78 passed (32 subtests) ·
perf board up to date · `git diff --check` clean.

## Review state on #568
CI jobs all green on `5f386c72`. Both `revuto` inline findings (fixed VA field, pooled G1 label) are fixed at
`a799cf5d` / `eb010c87` and verified against the tip (`active.rs`: `not-applicable-pooled` on both fields;
`reclaimed = vmm_granularity != 0 && bounded_no_leak && residual_class != "unclassified"`). `revuto` then hit its
2-round cap ("Revuto did not run a review on this pull request"); Bugbot capped for the day. Per the owner
ruling of 2026-09-16 the author's self-review COMMENT is the review; merged with `gh pr merge 568 --merge`.

## BOX3 state at resync (16:10Z)
One tmux `b-day10-vmm32` (B's original-source 32k VMM cell, collector attempt 3 after three lock refusals)
holds `/tmp/memra-gpu.lock`. B's 8k VMM on the target card: `g1_reclaim_qualified=true residual_bytes=0
residual_class=none vmm_granularity_bytes=2097152` (B pushed `4c295f225`). C's four 8 GiB pressure cells:
`pressure-status.json` `state: complete`. D's smoke exit 0; D's `--validate` over all BOX3 receipts still
`REFUSED: interrupted/invalid CELL journal; not a completed capture` (a peer cell was live). A: storage cells
264/4097/1048576 captured; 4194568 and the pread baseline pending.
