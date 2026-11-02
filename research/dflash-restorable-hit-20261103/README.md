# DFlash restorable-hit preference (2026-11-03)

Change: a DFlash request whose longest prefix hit has no draft tail looks for the longest entry
that does (and is a restorable prefix under `MEMRA_DSPARK_PARTIAL_RESTORE`) and takes it when the
route admits DFlash on that hit and the decode budget repays the extra suffix prefill by the
cold-prime ratio (`[dspark] restorable-preferred` line). Motivation: GB10 GGUF DFlash2 c1 at
`b014549cb9` served plain (28 `cache-preferred` vetoes, 3 DSPARK restores, 12.0 tok/s), see
`research/gb10-tuning-20261102/serving/`.

## RTX 5090 Laptop (24 GB, platform profile `balanced`)

- memra-server lib suite 1137 passed, 0 failed; clippy `-D warnings` clean; the new test
  `a_dflash_request_prefers_a_restorable_hit_over_a_longer_plain_one`.
- `tools/dspark-serve-smoke.sh` on the new binary: ALL GREEN, greedy completions byte-identical
  between the DFlash boot and the spec-off boot on all three prompt classes
  (`5090-smoke/dspark-serve-smoke.log`).
- Agentic c1 A/B, base `cd2e277a7d` against new `8084041d31`, both built with the same profile,
  order base/new/new/base, Qwen3.8-27B NVFP4+Q5_K GGUF + DFlash2, `MEMRA_MTP_SKIP=1`, context
  32,768, one session, the 29 bodies of the agentic workload that fit 24 GB
  (`dflash-ab-5090.sh`): aggregate 34.09 / 32.67 / 33.40 / 33.49 tok/s. On this card the new
  branch never fires (0 `cache-preferred`, 0 `restorable-preferred`, 11 DSPARK restores in every
  arm): the small prefix cache and short sessions keep the drafter-carrying entries the longest
  hits. The spread is T=1 sampling noise on a path both binaries run identically.

## GB10

Pending: GGUF DFlash2 c1 to c32 on the new build, against `research/gb10-tuning-20261102/serving/`.
