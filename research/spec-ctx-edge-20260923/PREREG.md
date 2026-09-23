# memra#659: the speculative context edge, pre-registration

Written before any GPU run of `tools/spec-ctx-edge-gate.sh`. Nothing below moves after a result.

## The defect
A request whose budget spans its whole session cap (`max_tokens` omitted) commits its last
speculative round past the session cache. Lane B day 31 hit it on the local RTX 5090 with the
Qwen3.5-9B NVFP4 MTP GGUF: under `MEMRA_ADMIT_BY_MEMORY=1` every open request at `P + v + 8`,
with the door OFF a runaway at `MEMRA_CTX` (issue body, receipts under
`research/spill-b-20260919/rtx5090-day31/boots/`).

## The cell
`tools/spec-ctx-edge-gate.sh <model> <binary> <out>`, three boots in sequence under one hold of
`/tmp/memra-5090.lock`, the 9B NVFP4 MTP GGUF, `SCE_OPEN=64`, `SCE_CTX=384`, defaults otherwise:

- `on`: door ON, open output 64, default spec. r1 to r4 open, then a 16-token bounded control.
- `plain`: the same door with `MEMRA_SERVE_SPEC=0`, one open request.
- `off`: door OFF, `MEMRA_CTX=384`, default spec, r1 to r3 open.

Verdicts are the gate's own lines (its header comment lists them). The gate reads no timing.

## Acceptance
1. **Red arm**: the gate on the unfixed tree, the integ49 merge tree `40891cf2d` (main `9c07b398b`
   plus A day 31's host-tier code, which sits behind the default-OFF contracts door), fails at
   least one verdict, and its server logs carry at least one of `mtp_kv_fill: scratch overflow`,
   `argmax sentinel` or `[worker] FATAL`. A red arm that does not reproduce the defect is recorded
   as "red arm did not reproduce", and then the green arm alone proves nothing about the fix.
2. **Green arm**: the gate on the fix tree prints `SPEC-CTX-EDGE GATE: ALL GREEN`, twice in a row
   on the same binary. Then it is wired into `tools/local-ci.sh`.
3. On the green arm the `on` open requests emit exactly 64 tokens (`finish_reason: length`), and
   the `plain` message equals the `on` arm's r1 byte for byte. If that equality fails on both
   trees, the divergence predates this fix: it is filed as its own issue and the gate stays red
   until it is resolved. It is not waived.
4. The fix moves no token: the `on` and `plain` messages are the same program's output up to the
   bound. The only client-visible change is where an open request stops (the door-ON budget is
   `v`, not `v + 8`; a door-OFF speculative runaway ends `ContextFull` up to `k + 3` short of the
   cap, where it used to overrun).

## Also owed before merge
The CPU censuses (`spec::ctx_edge_659_census`, `request_budget_keeps_the_slack_on_the_open_door_arm`),
clippy `-D warnings`, fmt, check-flags. On the 5090, the spec self-consistency cells the change
touches: the `run-spec` K=1..8 self-consistency gate and the `run-gen` argmax gate on the 9B.
