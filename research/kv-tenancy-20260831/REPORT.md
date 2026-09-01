# KV host-tier tenancy + continuation-pool park compaction (lane/kv-tenancy-compaction-20260831)

Executes the tenancy items of the tiering spec's Arc B plus Arc C1 (spec: darklanes
`research/engines-kv-oversubscription-20260830/SPEC-SESSION-TIERING.md`, §0.5 and Arc C).
Base: 52c5f3041 (the merged host-spill tier, lane/kv-host-spill-20260830). CPU-side unit
coverage only in this lane; the GPU cells below are named and PENDING.

## Deliverable 1: tenant lifecycle purge

Key revocation / tenant deletion clears the tenant's parked prompt bytes.

- `HostPrefixCache::purge_tenant(tenant)`: removes every host-tier entry across ALL of the
  tenant's end-user salts. The namespace match is `auth::meter_key(auth::scope_namespace(
  tenant, ""))`, the same two functions that scoped the entries in, so the purge key can
  never drift from the insert key. Raw-salt (no-keyring) namespaces carry no tenant and
  never match. Runs even when the tier is latched off.
- `PrefixCache::purge_tenant(tenant)`: the device half. Unpinned entries in the tenant's
  namespaces drop directly (never through the demote sink). WHY: with the host tier armed,
  a device capacity eviction DEMOTES, so a host-only purge would let a leftover device entry
  re-materialize the purged bytes into host RAM minutes later. Pinned entries (leased by
  in-flight sessions) are counted (`device_pinned_left`) and left: revocation never aborts
  an admitted request; the deployment re-fires after the drain.
- Wiring: `Cmd::PurgeTenantHost { tenant, tx }` parks in `handle_cmd` and executes at the
  tick top beside the trims (the pools live in `run()`'s scope; the sweep must not race a
  demote). Deployment side: `RuntimeHandles.purge` (`PurgeHandle::purge_tenant`), the
  engine half of `/admin/tenants/{tenant}/purge`. The admin contract's parameter is
  `{tenant}`, never `{tenant_id}`. Same handle lifetime contract as `TrimHandle` (drop on
  the shutdown signal).
- Receipts: `prefix_host_purges/purged_entries/purged_bytes` in `/metrics`, per-purge
  `[prefix-host] purge tenant=...` log line with both tiers' counts.
- OUT OF SCOPE, deliberate: the continuation/spec/dspark parked pools hold tenant state in
  VRAM but never demote to host; the spec (§0.5) scopes the retention concern to the COLD
  and FROZEN tiers. Named residual: a parked plain/spec/dspark session's VRAM state
  expires by LRU/park churn, not by this purge.

## Deliverable 2: per-tenant host-pool share cap

- `MEMRA_KV_HOST_TENANT_PCT` (default **50 by design**, integer 1..=100): one tenant's
  maximum share of the `MEMRA_KV_HOST_MB` budget. A demotion that would push the tenant's
  resident bytes past its share EVAPORATES (the entry drops, exactly as with the tier off)
  instead of demoting: the incoming entry is refused, the tenant's own resident entries
  are NOT evicted to make room. An exact-key replace charges only the delta.
- Accounting rides the same meter-key row identity as `ns_tokens` (`t:<tenant>` for
  keyring deployments, the raw salt otherwise), maintained at insert/remove/purge.
- Receipt: `prefix_host_tenant_rejects` in `/metrics` plus the
  `[prefix-host] demote evaporated at the tenant share cap` log line (the pre-copy hook in
  `host_demote_prefix_ref`, the line the production demote path actually emits, verified
  by the battery's T4 cell, 30 occurrences, darklanes
  `research/kv-fastband-20260830/battery-20260831/tenancy-gates/RESULTS.md`). The
  `[prefix-host] skip demote: tenant ... at its share cap` text this report previously
  quoted belongs to the INSERT-path backstop (`HostPrefixCache` insert), which production
  demotes never reach (the hook evaporates before the D2H copy); a gate grepping for it
  would wait for a line that never appears.

## Deliverable 3: continuation-pool park compaction (Arc C1)

- `MEMRA_KV_PARK_COMPACT` (default **0 = OFF by design**): at plain-pool park time,
  compact the retiring session's `Cache` from its ladder cap (`cache.max_ctx`, up to the
  full server context; ~1 GB parked for a 6k-token session on a 262k-ctx deployment) to
  exactly its committed length: allocate a fed-length cache, D2D-copy the live rows
  through the SAME machinery the plain-affinity grow path uses (`Cache::snapshot` +
  `pp::new_cache_planned` + `pp::restore_cache_checkpoint`), free the big allocation.
- Resume growth is the mirror image: the exact-extension probe admits a compacted entry
  (`plain_resume_cap_admits`), and the admit path re-allocates at the REQUEST's own
  charged cap (`ctx_cap`) and restores the parked rows before priming the suffix. Failure
  drops the entry and serves cold (the plain-affinity failure contract).
- Refusals park the ORIGINAL cache unchanged (compaction is an optimization, never a
  gate): SWA-ring caches (ring restore geometry is a NAMED SEAM, not v1; the step37 shape),
  distributed TP KV mirrors, `MEMRA_PP_HOST_BOUNCE=1` (snapshot would retain peer copies),
  `cache.pos != fed` (nothing provably safe to copy), and already-tight caches.
- SPEC/DSPARK reuse entries are OUT OF SCOPE by design: they park LIVE engine sessions
  (device draft scratch, captured CUDA graphs, sampler/Philox state) and are not
  compactable in a small diff (spec Arc C scoping).

## Unit coverage (this lane, CPU-only)

- Host purge: two tenants x two salts + raw-salt pool; purge one tenant; survivors,
  gauges, cumulative counters, LRU exactness, idempotence, raw-salt non-match.
- Device purge: unpinned drop + pinned-lease report + re-fire after unpin; accounting
  invariant asserted.
- Share cap: two-namespace fill where one namespace hits its cap (evaporation + reject
  counter) while the other still demotes; exact-key replace delta; eviction/purge
  accounting.
- Park compaction: `park_compact_rows` refusal matrix; `plain_resume_cap_admits` arms.
- Wiring assertions anchor on invocations in comment-stripped source (purge drain calls
  both tiers, handle sends the command, park site calls the compactor behind the flag,
  probe goes through the cap predicate).

## PENDING GPU gates (battery box; do not flip defaults before these)

1. **Park-compact byte identity**: resume-after-COMPACTED-park vs resume-after-PLAIN-park
   byte identity (greedy oracle), same prompt set, both the exact-extension and the
   affinity-rewind resume shapes; plus a compact-park -> grow-resume -> re-park cycle.
2. **Step-OOM adjacency replay** (spec Arc C1 risk): the multi-active OOM-park incident
   shape replayed with `MEMRA_KV_PARK_COMPACT=1` (incident-fixes-get-a-bench-gate law).
   RAN 2026-08-31: PARTIAL. Pressure survived (8/8 answered 200 under a 1.2 GB external
   squeeze, 9 compactions during the storm, zero declined/failed, no crash markers), but
   the `[admit-oom] step OOM parked` branch itself was UNREACHABLE from outside the
   process: four refusal layers catch external pressure first (darklanes
   `research/kv-fastband-20260830/battery-20260831/tenancy-gates/RESULTS.md`, T2). The
   requested engine fault door now exists: `MEMRA_STEP_OOM_FAULT` (FLAGS.md diagnostics
   table; lane/kv-battery-fixups-20260831) forges a quoted CUDA OOM at the step dispatch
   so the park branch is executable by a gate. The gate re-run WITH the door is what
   turns this row green; the compaction default stays put until then.
3. **Tenant purge under load** (spec B3): purge fired mid-traffic on a two-tenant
   workload; assert the surviving tenant's hit rate and the purged tenant's zero
   residency, plus the `device_pinned_left` re-fire path.
4. **Share-cap contention cell** (spec B3): two-tenant demote pressure at the cap;
   assert the capped tenant evaporates while the other tenant's demotions land, and no
   accounting drift after churn.
5. **Compaction cost receipt**: park-time copy wall-time at the 6k/30k/120k fed shapes
   (the park is retire-time, not per-token, but the receipt belongs in the tick-stall
   ledger like demote_ms).
