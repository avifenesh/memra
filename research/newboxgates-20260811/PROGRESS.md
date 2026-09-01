# New-box gate progress — 2026-08-11

Lane: `lane/cx-newboxgates`

Rig: Vast `109.231.106.68:45771`, 2x RTX PRO 6000 WS FULL 600W, PCIe 5.0, driver 610.

## Status

- [x] Confirmed the local worktree is on the dedicated lane branch and clean at start.
- [x] Read the project instructions in `CLAUDE.md`.
- [x] Checked the lane inbox; `~/.lanectl/inbox/cx-newboxgates.md` was absent locally and on the rig.
- [x] Captured remote revision, hardware, model, binary, toolchain, and serving receipts.
- [x] Ran b1fix one-hash subset: c=1 x10 fresh boots and c=8 barrier x5 — 50/50 one hash.
- [x] Ran the correctness battery: kernel/decode gates, run-gen step35, run-spec K=1..8,
  and step35 chunk/tick invariance with canaries — all pass.
- [x] Ran N=5 interleaved performance battery: short TTFT, 4k TTFT, decode c=1/4/8.
- [x] Ran the 262k capacity row with `MEMRA_SWA_RING` off and on via the capbase harness.
- [x] Restored serving and soak after every disruptive block; independently rechecked
  `/readyz`, `/v1/models`, a streamed completion, and fresh soak rows.
- [x] Wrote `RESULTS.md`, retained raw evidence, and committed the lane in small commits
  without pushing.

## Notes

- Raw logs will be captured before parsing; failures will be reported only from captured stderr.
- Performance summaries will state N and thermal/interleaving regime.
- The server may be stopped only for bounded gate blocks and must be restored with
  `setsid nohup /root/start-memra.sh` before handoff.
- The final verdict is PASS. The N=5 medians supersede the provisional single-run values.
- No runtime code, generated performance board, runtime default, tag, or remote branch was changed.
