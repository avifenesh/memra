# grouped-serve — PP-2 serving promotion gate

Branch: `lane/cx-grouped`
Base: `2d9359df353b00f196b124aa62e19ce3bfb7789a`
Rig: box1 `<rented-box-ip>`, 2x RTX PRO 6000, PP-2 devices `0,1` under
`/tmp/memra-gpu.lock`.
Model: `~/step37/models/step-3.7-flash`

## Mission and stop line

Decide whether the existing opt-in grouped Step prefill path should be promoted in
the **serving configuration** for the owned PP-2 RTX PRO 6000 deployment shape.
The naked runtime default stays unchanged because the local RTX 5090 transfer gate
rejected it. No code default flip is in scope.

The decision requires same-binary evidence from one bounded lock hold:

1. Exact 4k streaming TTFT, grouped ON versus OFF, interleaved with cold salts,
   N=5 per arm.
2. Prefill throughput at pp512, pp2048, and pp4096.
3. Grouped-arm model-backed `kernel-check`, `run-gen` argmax MATCH, and the
   Lever-C grouped-versus-sequential byte-identity gate.
4. A c=4 streaming burst with grouped ON to expose interaction with batched prime.

Raw logs are part of the deliverable and will be retained under `raw/`. Every
reported median will state N and the observed thermal regime. Errors will be
captured to raw logs before parsing.

## Coordination state

The requested `~/.lanectl/inbox/cx-grouped.md` path was absent at lane start on
2026-08-10. The inbox directory exists and contains no alternate `cx-grouped`
entry. The exact requested path will be checked again at the start of every
bounded work block.

## Pre-registered decision rule

Promote only if the grouped arm is exact, materially improves same-window PP-2
serving TTFT/prefill on box1, and the c=4 burst shows no correctness or serving
failure. Promotion means an explicit `MEMRA_MOE_GROUPED=1` line only in PP-2
RTX PRO 6000 deployment templates/docs. Otherwise hold the opt-in posture.

`RESULTS.md` is the stop artifact. No origin push, release, tag, or runtime-default
change will be made in this lane.

## Increment 1 — non-vacuous box1 harness

The inherited Lever-C gate script was not reusable as-is: it omitted
`MEMRA_MOE_GROUPED=1` because grouped prefill was the default at that historical
commit. On the current default-off tree that would make the grouped oracle and
`run-gen` checks vacuous. This lane's gate driver sets `=1` explicitly and requires
all 210 byte-identity rows, both live Step dispatch classes, zero `MISMATCH` rows,
`kernel-check` `ALL GREEN`, and both `run-gen` argmax `MATCH` lines.

The performance drivers preregister adjacent alternating pairs. Prefill uses five
independent processes per arm and shape, each with one warmup. Streaming TTFT uses
five server boots per arm, one excluded warmup and one measured 4107-token request
per boot, with a unique cache salt. The final grouped-only c=4 cell runs three cold
bursts and requires a live `[step35-batch] ... B=4` record from the server log.

Each driver owns one bounded `/tmp/memra-gpu.lock` hold and records the same binary
hash, thermals, memory state, raw command output, and exit status. No tuning flag
besides the controlled grouped arm is changed.

## Box1 blocks completed

The exactness block passed all three required gates with grouped explicitly on:
210/210 grouped-versus-sequential rows were byte-identical across both live Step
dispatch classes, model-backed `kernel-check` ended `ALL GREEN`, and PP-2
`run-gen` reported prefill and decode argmax `MATCH`.

The interleaved prefill block completed N=5 per arm at all three shapes. The
interleaved streaming block completed N=5 per arm before its separate burst
phase. Those samples remain frozen; a failure in the burst client is not a reason
to rerun favorable TTFT data.

The first c=4 attempt was excluded before any scored burst. Its synthetic
`prompt_ids` warmup completed on the server (4096-token prime, first SSE byte,
eight decode ticks, clean shutdown), but all decoded completion strings were
empty and the old client required visible text. Recovery therefore changes only
the burst probe: it uses the same known-visible chat prompt as the TTFT block,
requires exactly 4107 cold prompt tokens per request, keeps N=3 at c=4, and still
requires a live server-side `B=4` batching record. `BURST_ONLY=1` prevents any
rerun of the completed TTFT arms.
