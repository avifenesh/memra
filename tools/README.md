# tools/ index

One line per tool, newest first. Started 2026-08-30 (lane/kv-host-spill-20260830); earlier
scripts are documented in their own headers and in docs/FLAGS.md / docs/SERVING.md rows.
When you add a tool, add its line.

- `kv-host-spill-identity-gate.sh`: cached-vs-fresh byte-identity gate with a host-tier arm
  (demote -> promote -> restore must equal the tier-off cold bytes); `MEMRA_HOSTGATE_TEETH=1`
  is the forced-tiny `MEMRA_KV_HOST_MB=1` red arm whose verdict must invert.
- `kv-host-spill-failure-gate.sh`: executes the host tier's failure paths loudly (pool-full
  refusal, `MEMRA_KV_HOST_VERIFY` digest mismatch via the `MEMRA_KV_HOST_FAULT=flip-demote`
  door, pinned-alloc latch-off via `alloc-fail`) and pins byte-identical cold serving under
  each.
