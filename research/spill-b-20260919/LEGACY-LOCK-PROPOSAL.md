# WP-D handoff: collector-compatible legacy prefix gates (proposal only)

Lead decision for this dedicated development session: the existing scripts run under
**their own internal `/tmp/memra-5090.lock`**, not nested beneath the collector. Before
each run verify compute-apps empty and no collector/model process running. Their current
`pkill -x memra-server` is explicitly authorized only for this dedicated non-serving box.
No lock bypass or third lock is introduced. These correctness runs have raw logs, not a
complete 250ms collector telemetry envelope; they are not performance results.

Proposed future changes (WP-D/lead owns implementation, no script edited by B):

1. Add explicit `--external-lock` to the two legacy scripts. It must require a collector
   issued/inherited lock FD for the SAME canonical inode and fail closed if no valid
   proof is supplied. Merely trusting an environment boolean or seeing a lock file
   is insufficient. Do not reacquire that lock in each spawned server.
2. Default stays the current internal canonical lock behavior. Receipt must identify
   owner (`internal-canonical` or `collector`), canonical path and proof mechanism.
3. Hold the collector lock over all boots, probes and teardown, not just startup.
   Track and stop only the spawned process group; replace global pkill in the future
   shared-box interface. Do not run a second lane during gaps between boots.
4. Test internal success, external success, missing/foreign/closed proof refusal,
   concurrent-launch refusal, failure teardown and collector timeout. Verify kernel
   lock retention over subprocess/FD lifetimes, not an assumed shell convention.
5. Wrap the whole script in tier-battery only after these tests pass. Preserve every
   original named-engagement, digest, tiny-pool and fault assertion.

## Observed setup corrections, not runtime changes

On this exact artifact/binary, the default 1024 MiB device budget holds both ~160 MB
seeds and the identity gate correctly fails engagement. A 256 MiB budget holds one,
not both; the engaged identity gate passes. Raw runs are in `legacy-identity-r2/`
and `legacy-identity-r3/` under the rented receipt namespace.

The tiny-pool script expects `[prefix-host] skip demote: entry`, but the current default
50% tenant cap emits `demote evaporated at the tenant share cap before the D2H copy`
first. Its other tiny-pool assertions pass (zero demotions/promotions and identical cold
output). The intended global-pool fault therefore needs the existing diagnostic setting
`MEMRA_KV_HOST_TENANT_PCT=100`. Retain a separate tenant-cap refusal cell; do not weaken
the global-pool grep to an arbitrary refusal or relabel the first failed run green.
