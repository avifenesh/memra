# Flags-census self-test pipefail repair

Date: 2026-09-04  
Issue: memra#190  
Source: `378b38910f37d38e09b284950757cc8b2281a39d`

## Verdict

`tools/test_check_flags.sh` now tests membership in its captured live census without a pipe whose
producer can be killed by a successful early `rg -q` exit. The census itself was correct; the
self-test was timing-dependent under `set -o pipefail` and falsely reported two present flags as
absent. No production or runtime behavior changes.

## Gates

- Before: the live census contained `MEMRA_ALLOW_UNKNOWN_PRETOKENIZER` and `MEMRA_FATBIN`, but the
  fixture reported both absent and ended `22 passed, 2 failed`.
- After: `tools/test_check_flags.sh` reports `24 passed, 0 failed`; its chained docs-registry
  fixture reports `9 passed, 0 failed`.
- `tools/test_gate_integrity_r2.sh`: `29 passed, 0 failed`, including the deliberately blind census
  arm and the live-tree control.
- `tools/local-ci.sh --perf`: exit 0 with no flags-census warning; 107 kernel cells green, c=64 serve
  stress green, correctness/cache/spec gates green, and the available perf cell in band.
