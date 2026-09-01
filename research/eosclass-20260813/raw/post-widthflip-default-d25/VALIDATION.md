# Validation — repaired default Q27 width transition

- Runner exit: `client.exit=0`; all 105 admitted requests completed. The measurement contains four
  serial one-token seeds, one solo restored-hit control, and 25 target-plus-three-peer cells.
- Every one of the 101 post-seed requests was a full 4,860-token prefix-cache hit. Counters record
  zero cache evictions, admission/session/VRAM defers, step-OOM parks, or protocol failures.
- The target began before three already-restored peers at delays 0 through 600 ms in 25-ms steps.
  All 25 targets completed 60/60 tokens with `finish_reason=length` and matched the solo control's
  sole SHA-256:
  `5790654979cb98bfacf6d3593b6a5d3def7a5f4bd2a1b8b65e4a6fabe1a72f66`.
- The run used the repaired release binary SHA-256
  `17a222026e08b65f9344407ba9108cb554688c0431365932bb4e78de1033597d`, with B1FAST and
  GraphSession at their new defaults (OFF). Diagnostics were disabled, so this is a trace-free
  regression gate rather than a sampling-instrumentation run.
- The paired pre-fix sweep used the exact same workload and delay grid and reproduced the historical
  11-token EOS at 50 and 225 ms. This post-fix sweep removes that load-history dependence by keeping
  solo and concurrent ticks on the generic batched numeric program.
- The sole `failure-signature-scan.log` match is the startup gpu-watch banner listing configured
  fatal Xid numbers. It is not an observed Xid or runtime failure.
