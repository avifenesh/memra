# F8F4 v3 follow-up in the engine: exactness on GB10 unit 1 (2026-11-08)

Binary: `292b21bdf8`, built on unit 1 (`SHA256SUMS`). The v3 copies use the CTA-local destination,
and the fused gate/up silu launch runs v3 again. No timing runs happened in this lease. Script:
`fup-gates.sh`.

- **kernel-check**, default mode (`kernel-check/`): ALL GREEN (124 cells, 5 skipped). Every
  `MMQ-F8F4-V2*` cell runs under both mainloops against the v1 tile, now including GATEUP under v3.
  The result is 0 byte mismatches.
- **run-gen pp3855 logits** (`gates/rollback/`, `logits3.sh`): the logits are identical with v3,
  with `MEMRA_MMQ_F8F4_V3=0` and with `MEMRA_MMQ_F8F4_V2=0`. They are also identical to the landed
  head's logits (`f81e969042a7e155`, `gemm-v3-engine-gates-unit1/`). Prefill medians over 3 reps,
  single runs: 3.258 / 3.395 / 3.899 s. The landed binary measured 3.361 / 3.429 / 3.926 s in its own
  window.
- **16k argmax-margin gate:** stopped at the lease end before its v3 arm finished (`stop-gates.sh`,
  `stop-gates.out`). It ran again on unit 2 with the served campaign and passed, with a probe table
  byte-identical to #1102's (`gemm-v3-cta-served-unit2/argmax16k-v3/`).
