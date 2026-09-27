# memra#659: the speculative context edge, results

Pre-registration: `PREREG.md` (committed `dff1a6220`, before any GPU run). Local RTX 5090 Laptop, the
Qwen3.5-9B NVFP4 MTP GGUF, one hold of `/tmp/memra-5090.lock` at 04:00:57Z (`rtx5090/run.log`). A
foreign process (another project's python, 1390 MiB) was on the card and is recorded in the run log
and in every boot's `compute-apps-before.csv`. The gate reads no timing. Executed on one card.

## Verdicts, verbatim

**Red arm**, the unfixed tree (binary `920eebe9`, built at `aceef3589`: main `9c07b398b` plus A day
31's host-tier code behind the default-OFF contracts door): `SPEC-CTX-EDGE GATE: RED`, 2 PASS, 10 FAIL.

- `SCE (on) r1 open request: http=500 ... -> FAIL`, the same for r2 to r4; the bounded control
  `http=200 completion_tokens=16 expected=16 -> PASS`; census `argmax_sentinel=4 ... -> FAIL`.
- `SCE (plain) r1 open request: http=200 finish_reason=length completion_tokens=72 expected=64 -> FAIL`
  (the old budget, `v + 8`).
- `SCE (off) r1 open request to the cap: http=500 ... -> FAIL`, the same for r2 and r3; census
  `argmax_sentinel=3 ... -> FAIL`.

The red run reproduces the defect through the #87 trap (`argmax sentinel` in the server logs); no
request lived long enough to reach the deferred-fill panic that lane B saw at a larger cap.

**Green arm**, the fix tree (binary `ceb185d1`, built at `dbaf0f341`), twice in a row:
`SPEC-CTX-EDGE GATE: ALL GREEN`, 13 PASS and 0 FAIL each.

- `SCE (on) r1 open request: http=200 finish_reason=length prompt_tokens=31 completion_tokens=64 expected=64 -> PASS`,
  the same for r2 to r4 and the control.
- `SCE (plain) message equals the on arm's r1: plain=341ddf3169a9d21a spec=341ddf3169a9d21a -> PASS`.
- `SCE (off) r1 open request to the cap: http=200 finish_reason=length prompt_tokens=31 completion_tokens=352 room=353 -> PASS`,
  the same for r2 and r3.
- Every boot census `panicked=0 argmax_sentinel=0 worker_fatal=0 respawn=0 verify_refused=0 -> PASS`.

**On main.** Rebased onto main `f69119ae0` (`e23adc70d`, binary `91e56c12`, build line
`memra-0.138.0-8636637e872c (id: source-tree, git: e23adc70d21e)`): `SPEC-CTX-EDGE GATE: ALL GREEN`, 13 PASS
(`rtx5090/rebased/`).
Merged with main `580e8a4a1` (integ49, A day 31's `worker.rs` under its default-OFF door) at `1fc303461`,
binary `395a42b9`, build line `memra-0.138.0-709466af1831 (id: source-tree, git: 1fc3034619f5)`:
`SPEC-CTX-EDGE GATE: ALL GREEN`, 13 PASS (`rtx5090/merged-main/`).

**Target card (the #668 PRO 6000 run, owed at merge).** Lane E ran the gate on BOX3's 27B during its #641
qualification (tree `9e3b7250e`, main `d544c6b82` plus the #641 fix, `SCE_CTX=384`, a 73-token prompt):
`SPEC-CTX-EDGE GATE: ALL GREEN`, 13 PASS and 0 FAIL, with the plain and spec messages equal (`0ba35b0ee89ade1d`) and the
door-OFF runaways ending at 309, 311 and 309 of 311 tokens of room (`research/decode-exact-641-20260923/pro6000/run.log`). #668
merged before this run on 5090 evidence alone. This line closes that gap.

**Also owed by PREREG, done.** `run-spec` K=1..8 on the 9B (single-shot route, `MEMRA_SPEC_TEMP=0`,
32 tokens, the tier-2 probe prompt): `self-consistency: PASS` 8 of 8 and `=== SELF-CONSISTENCY PASS ===`
(`rtx5090/run-spec-9b/`). `run-gen` runs the plain route, which this change does not touch. CPU: the
source census `spec::ctx_edge_659_census` (3 passed) and `request_budget_keeps_the_slack_on_the_open_door_arm`;
clippy `-D warnings` on engine and server, all targets.

## Reading

1. Acceptance 1 holds: the unfixed tree fails 10 verdicts and its logs carry `argmax sentinel`.
2. Acceptance 2 holds: two consecutive ALL GREEN runs on the same binary. The gate is wired into
   `tools/local-ci.sh` (`MEMRA_CI_SPEC_CTX_EDGE=0` skips).
3. Acceptance 3 holds: open requests emit exactly 64 tokens, and the plain message equals the spec
   message byte for byte.
4. Acceptance 4: no token moved. The only client-visible change is where an open request stops: the
   door-ON budget is `v` (64; the unfixed plain arm emitted 72), and a door-OFF speculative runaway at
   `MEMRA_CTX=384` ends at 352 of 353 tokens of room, one short of the cap, where it used to overrun.

Scope: one card, one model, the qwen MTP route. The gemma loops carry the same guard (CPU census
only here; no gemma model on this rig).
