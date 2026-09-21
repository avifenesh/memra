# WP-A day 12: memra#384 (tenant-share cap reclaim) and memra#385 (arena startup plan)

Lane `lane/spill-a-20260919`, worktree `wt-spill-a`. Start: tip `432816926` (= origin, merged to
main by #588), merged `origin/main` `be07f2d36` (`e5b1c10d0`). Code commit `405466cf7` (the
reclaim, the gate, the harness), harness sizing `2798e0b2e`; the receipts commit that adds this
file is the branch tip. Owner order: the open pre-DeepSeek issues first; these two host-tier
issues were today's.

## memra#384: at the tenant share cap, evict the tenant's own unleased LRU host entries first

Issue, verbatim: "At the configured host tenant-share cap, demotion currently drops the victim
before D2H even when the arena still has free space... evict that tenant's own unleased LRU host
entries first, then retry admission... Keep the configured tenant cap and lease protections.
Reclaim only this tenant's eligible LRU entries, retry admission, and retain bounded refusal
when there is no eligible space."

### Where it lives

`crates/memra-server/src/worker.rs`:

- `HostPrefixCache::reclaim_tenant_share(key, toks, bytes) -> Result<TenantShareReclaim,
  TenantShareRefusal>`, beside `tenant_cap_would_evaporate`. Two passes over the tier's own
  `lru` map (oldest first): an eligibility pass with no mutation (this tenant's row only, by
  `auth::meter_key`; never the exact-key twin, which the predicate credits and `insert`
  replaces; never a leased entry), then an eviction pass through `remove_at` until the
  predicate is false, re-resolving the victim after every removal because `remove_at`
  swap-removes. The one retry is the final predicate check; `Ok` means it passed. Refusals
  evict nothing: `ImageExceedsShare` (the image alone is above `tenant_budget()`, so no
  eviction can help, into a full row or an empty one), `NoEligibleSpace` (the row's unleased
  bytes cannot cover the shortfall; the refusal names the shortfall, the eligible bytes and the
  leased entries skipped). `RetryStillShort` is unreachable on the single-owner worker and
  fails closed if it ever fires.
- `HostPrefixEntry::leased()` = `IdentitySlot::leased()` (new in
  `crates/memra-kv/src/tiered/hostprefix.rs`: true while an `IdentityLease` handed out by
  `lease` is alive; `Arc::strong_count > 1`, exact because the lease is the only other holder).
  This is the host tier's only lease: device pins never reach the host LRU (they are absent from
  the evictable LRU by construction, `host_demote_prefix_entry`'s doc), and a promote holds the
  identity lease for the length of its synchronous H2D.
- The hook `host_demote_prefix_ref`, as first shipped at `405466cf7`: `if
  tenant_cap_would_evaporate(...) && let Err(refusal) = reclaim_tenant_share(...)` printed
  today's line unchanged plus `; {refusal}` and returned `Evaporated`, counting
  `tenant_rejects` as before; on `Ok` the D2H proceeded. SUPERSEDED by the integ15 review
  (section "Integ15 review fixes" below, commit `8b29b2aa3`): the pre-copy check is now the pure
  `tenant_share_reclaim_plan` and the evictions run after `bind_tier_image`, right before
  `insert`; `reserve_image` (the fixed-arena path) runs the reclaim itself for the demote hook
  only. Every pressure-driven demote site reaches this one hook (the SLRU sink, the pause sweep,
  the admission flush `evict_all_demoting`).
- Per-eviction line: `[prefix-host] evict (tenant share): N tokens, X MB of tenant "t:..."'s
  own entries for its Y MB demotion (row now R MB / C MB share = P% of B MB, model ..., ns
  "t:...")`. Counter `tenant_reclaims` (a subset of `evictions`), published as
  `prefix_host_tenant_reclaims` on `/metrics` (`worker.rs` publish, `lib.rs` render).

Unchanged: `tenant_budget()` and the cap arithmetic, `insert`'s share-cap gate (the pre-existing
test `host_cache_tenant_share_cap_evaporates_one_tenant_and_still_demotes_the_other` still
passes), the D2H payload (the same `host_image_bytes` image copies after the reclaim), the
handoff import path (`insert` direct: an import over the cap still evaporates; out of scope,
stated), no new `MEMRA_*` read, no `.cu`, no engine file, no `unsafe`.

### Unit cells (`cargo test -p memra-server --offline tenant_share`, all `ok`)

| Cell | What it pins |
|---|---|
| `host_cache_tenant_share_reclaim_evicts_the_tenants_own_oldest_entries_only` | Budget 100, share 50. Below the cap the reclaim is `Ok(default)`. At the cap acme's oldest entry goes (its newer one stays, across acme's two salts); beta's entry and row (`t:beta` 20) untouched; the retried `insert` lands with acme exactly at 50; a 50-byte demotion takes the two oldest in LRU order; `evictions == tenant_reclaims == 3`, `tenant_rejects == 0`. |
| `host_cache_tenant_share_reclaim_spares_the_twin_and_refuses_an_image_above_the_share` | Re-demoting the 30-byte key at 31 bytes evicts the older 20-byte entry, never the twin. A 51-byte image refuses `ImageExceedsShare {51, 50}` with nothing evicted, into acme's row and into an empty row (`need 1, eligible 0`); `tenant_pct = 100` disarms the reclaim with the cap. |
| `host_cache_tenant_share_reclaim_skips_leased_entries_and_refuses_when_only_leased_remain` | A real `IdentitySlot` bound through `bind_identity` (a one-segment logits bundle, pure functions, no device planes) and its `IdentityLease` held: unbound `leased() == false`, bound and unleased `false`, leased `true`. The leased oldest entry is skipped and the newer unleased one evicted; with 30 needed and 20 unleased the refusal is `NoEligibleSpace {need 30, eligible 20, 1 leased holding 30}` and nothing is evicted; after `drop(lease)` the same demotion fits by evicting exactly that entry. |
| `tenant_share_reclaim_is_wired_into_the_demote_hook_and_the_metrics` | Source text. At `405466cf7`: the hook called `reclaim_tenant_share(...)` before the evaporation line and both sat before `host_entry_from_device(`. SUPERSEDED at `8b29b2aa3` (integ15 review): the plan sits before the evaporation line and the copy, no `reclaim_tenant_share(` before the copy, the reclaim inside the `Ok` arm after bind and before insert, four `waste_pending_reclaim(` sites; both counters reach the publish and the render. |
| `memra-kv` `unbound_slot_is_never_leased` | The trivial half of `IdentitySlot::leased`. |

First compile of the cells failed twice on the same cause, verbatim: `error[E0433]: cannot find
type `TenantShareReclaim` in this scope` and `error[E0433]: cannot find type
`TenantShareRefusalWhy` in this scope` (the tests module needs `super::`); fixed, rerun `ok`
(`day12/gates/test-tenant-share-first.log` is the rerun; the two error lines above are the
record of the first run).

### Serving-shape gate: `tools/kv-host-tenant-reclaim-gate.sh <base|fix> MODEL BIN EV`

Two keyring tenants (`MEMRA_API_KEYS` inline, acme and beta, one `cache_salt` each), door OFF,
the artifact's default spec environment, `MEMRA_KV_HOST_MB=1024`, `MEMRA_KV_HOST_TENANT_PCT=38`
(share 408 MB: two ~161 MB entries, not three), `MEMRA_PREFIX_CACHE_MB=384` (two device entries,
so every new seed demotes the older one and a promote fits beside one protected entry),
`MEMRA_METRICS_TOKEN` for the scrape. Eight requests: r1 beta P_D; r2 acme P_A; r3 acme P_B (E_D
demotes to beta's row); r4 acme P_C (E_A to acme's row); r5 acme P_E (E_B: acme at its share); r6
acme P_F (E_C: acme's THIRD demotion hits the cap with the pool two-thirds empty: base
evaporates, fix evicts acme's oldest E_A then demotes E_C); r7 beta P_D+EXT (beta's E_D promotes
in both arms; the promote insert evicts E_E and the boundary insert evicts E_F: two more cap
events); r8 acme P_F+EXT (fix promotes the reclaimed E_F, `cached_tokens == r6 prompt_tokens`;
base is cold, the cost #384 removes). Both arms assert eight 200s, a 200 scrape with the
process-wide counters, the cap line, no `evict (LRU)` (the pool never fills), no device-tier
`skip pinned host-promote insert`, no refusal or failure line, beta's demote and promote, no
tenant-share eviction naming beta, every tenant-share eviction naming acme in both the row and
the namespace. `base` asserts the evaporation line naming acme, no reclaim line, r8 cold,
`prefix_host_tenant_rejects >= 1` and no reclaims; `fix` asserts no evaporation line, a reclaim
line, every reclaim followed within eight lines by acme's own `[prefix-host] demote:`, r8
promoted with the promote line naming acme, `prefix_host_tenant_reclaims >= 1` with
`prefix_host_tenant_rejects == 0`. `SUMMARY.json` per arm feeds `verify-day12.py`, which
adds the cross-arm law: equal `[prefix-host] demote:` bytes for every prompt that demoted in both
arms, the fix arm's acme demotes carrying exactly the token counts and bytes the base arm
evaporated (the same D2H payload), every request text byte-identical across arms, beta's
`cached_tokens` and promote lines equal across arms.

### Target card (one RTX PRO 6000 Blackwell, 600 W, N=1, executed-not-qualified)

`/root/wt-a` synced by git bundle (`e5263571..lane/spill-a-20260919`, 2,119,512 bytes, streamed
over the existing ControlMaster socket, deleted after the fetch; two small follow-up bundles for
the harness and gate commits). `build.sh`: base = `origin/main` `be07f2d36` checked out detached,
`cargo build --release -p memra-server --offline -j 16` under `nice -n 19`, exit 0 in 3m04s
(100 crates), binary sha256 `b30ac09d04f4de309f6cb509054a2c87eb2d9252ab540e2c78604a0c802aaa0f`;
fix = `405466cf7`, exit 0 in 36.87s (memra-kv, memra-engine, memra-server), sha256
`d331909a7ef094f1aef88653d77884dc61cad066ab98f796c9f43f720ce9e5ae`; both trees clean at build
(`dirty-*.txt` empty). The Rust sources of `405466cf7` and the branch tip are identical (the later
commits touch `tools/*.py`, `tools/*.sh` and research files), so the fix binary is the lane's
binary. Card `card.csv`: `NVIDIA RTX PRO 6000 Blackwell Server Edition, 600.00 W, 600.00 W,
580.178.04`. Lane B's session `b14-r3` ran on the card during the build and had ended before the
cells; every cell went through the collector (`--rig pro-single`, `/tmp/memra-gpu.lock`,
`lock.json` canonical, bounded retries, no retry needed).

Three attempts, every directory kept:

1. `tenant-base`, `tenant-fix` (07:02Z): all seven completions 200 and the base arm already
   printed the evaporation line three times, then the gate died on its `/metrics` scrape,
   verbatim `urllib.error.HTTPError: HTTP Error 401: Unauthorized`: under a keyring
   `authorize_metrics` requires a bearer and a tenant bearer gets `MetricsScope::Tenant`, which
   withholds the process-wide `prefix_host_*` counters. `arena` exited 2 at once,
   `pinned-host-reserve-bench.py: error: unrecognized arguments: --basis free` (the tree's
   harness predated `2798e0b2e`). Fix: `MEMRA_METRICS_TOKEN` at boot and on the scrape, the
   scrape never uncaught; the tree moved to the sizing commit.
2. `tenant-base-r2`, `tenant-fix-r2`, `arena-r2` (07:12Z): the fix arm's r7 (then `P_C+EXT`)
   did not promote although the host entry was resident: `[prefix-cache] skip pinned
   host-promote insert: entry 159.8MB would evict protected bytes below their 215MB share (need
   50.9MB, probation/demotable 0.0MB)`, the device tier's protected-share rule at a 256 MiB
   budget with beta's entry protected after its hit, not the host tier. Redesigned to a two-entry
   device budget (384 MiB), a sixth prompt and eight requests. The same attempt also showed the
   gate's own matcher fault: `present`/`absent` used basic regex, where `[prefix-host]` is a
   bracket expression, so `FAIL: beta's first demote landed`, `FAIL: the r7 promote line names
   beta`, `FAIL: fix: acme's own oldest entry was evicted at the cap`, `FAIL: fix: the r8 promote
   line names acme` were false negatives and two `absent` checks had passed vacuously. Fix: ERE
   with escaped brackets plus a matcher self-check at start. `arena-r2` ran clean and stands.
3. `tenant-base-r3`, `tenant-fix-r3` (07:16Z): `GATE: kv-host-tenant-reclaim (base arm) PASS`
   (25 `ok`), `GATE: kv-host-tenant-reclaim (fix arm) PASS` (26 `ok`), both cells
   `executed-not-qualified`, exit 0, `600.00 W`.

Base arm, verbatim (three cap events, r6, r7 promote insert, r7 boundary insert):

```text
[prefix-host] demote evaporated at the tenant share cap before the D2H copy: 93 tokens, 160.8MB (38% of 1074MB, MEMRA_KV_HOST_TENANT_PCT; model gate, ns "t:acme\u{1f}s1")
[prefix-host] demote evaporated at the tenant share cap before the D2H copy: 93 tokens, 160.8MB (38% of 1074MB, MEMRA_KV_HOST_TENANT_PCT; model gate, ns "t:acme\u{1f}s1")
[prefix-host] demote evaporated at the tenant share cap before the D2H copy: 95 tokens, 160.9MB (38% of 1074MB, MEMRA_KV_HOST_TENANT_PCT; model gate, ns "t:acme\u{1f}s1")
```

Fix arm, verbatim (each reclaim immediately followed by the demote it made room for; the fourth
is r8's own boundary insert after the promote, which the base arm never reached):

```text
[prefix-host] evict (tenant share): 89 tokens, 160.7MB of tenant "t:acme"'s own entries for its 160.8MB demotion (row now 160.6MB / 408MB share = 38% of 1074MB, model gate, ns "t:acme\u{1f}s1")
[prefix-host] demote: 93 tokens, 160.8MB in 6.3ms (host resident 482.0MB / 1074MB, model gate, ns "t:acme\u{1f}s1")
[prefix-host] evict (tenant share): 86 tokens, 160.6MB of tenant "t:acme"'s own entries for its 160.8MB demotion (row now 160.8MB / 408MB share = 38% of 1074MB, model gate, ns "t:acme\u{1f}s1")
[prefix-host] demote: 93 tokens, 160.8MB in 5.9ms (host resident 482.2MB / 1074MB, model gate, ns "t:acme\u{1f}s1")
[prefix-host] evict (tenant share): 93 tokens, 160.8MB of tenant "t:acme"'s own entries for its 160.9MB demotion (row now 160.8MB / 408MB share = 38% of 1074MB, model gate, ns "t:acme\u{1f}s1")
[prefix-host] demote: 95 tokens, 160.9MB in 5.7ms (host resident 482.3MB / 1074MB, model gate, ns "t:acme\u{1f}s1")
[prefix-host] evict (tenant share): 93 tokens, 160.8MB of tenant "t:acme"'s own entries for its 161.3MB demotion (row now 160.9MB / 408MB share = 38% of 1074MB, model gate, ns "t:acme\u{1f}s1")
[prefix-host] demote: 108 tokens, 161.3MB in 5.8ms (host resident 643.7MB / 1074MB, model gate, ns "t:acme\u{1f}s1")
```

Equal bytes: every image the base arm evaporated (93 tokens 160.8MB, 95 tokens 160.9MB) the fix
arm demoted at the same byte count, and the four prompts that demoted in both arms print equal
`[prefix-host] demote:` bytes (beta 83: 160.5, acme 86: 160.6, acme 89: 160.7, beta 96: 160.9).
Requests: eight 200s in both arms; every text byte-identical across arms (r8 included, where the
fix arm promotes and the base arm primes cold); `prompt_tokens` equal; beta r7 `cached_tokens ==
83 == r1 prompt_tokens` in both arms with equal promote lines; acme r8 `cached_tokens` 95 (fix,
`hit: 95 of 108`) versus 0 (base). `/metrics`: base `prefix_host_tenant_rejects 3`, no reclaims
field, `prefix_host_entries 4`, `demotions 4`, `promotions 1`; fix `prefix_host_tenant_reclaims
4`, `tenant_rejects 0`, `entries 4`, `demotions 8`, `promotions 2`; `rejected_allocs 0` and no
`evict (LRU)` line in either arm: the pool held four entries of 1024 MiB throughout (the "arena
still has free space" condition). No device-tier promote-insert skip in either arm.
`verify-day12.py`: `DAY12 PASS: base evaporates, fix reclaims the tenant's own entry and demotes
the same bytes; texts identical; beta untouched; arena harness ran` (`day12/verify-day12.log`);
`tier-battery.py --validate` exit 0 on `tenant-base-r3`, `tenant-fix-r3`, `arena-r2`
(`day12/collector-validate.log`). Mirror `pro-single-day12/` (184 files, 896K, binaries excluded),
no host name, address or provider id in it (scanned).

## memra#385: pinned host arena startup, plan and harness

`HOST-ARENA-STARTUP.md`: the measurement law (same box, one window, interleaved A/B/A/B, N >= 5
pairs per order, both orders, free between measurements, measure the driver call, record the
regime, correctness beside speed, decision on the target), the candidate mechanisms with code
sites (`pinned_host.rs:95-117` `reserve`, `:9-64` `Extents`, `:79-84` `Drop`, `:145-147`
`reserve_timings_ms`; `worker.rs:14779-14811` the startup reserve; `host_memory.rs:4` the 32 GiB
margin): chunked parallel `cudaHostAlloc` (measured as a harness check), hugepage-backed reserve
(`mmap` + `cuMemHostRegister`; not measurable on BOX3, `HugePages_Total: 0`, THP `madvise`),
lazy commit (rejected on the issue's "zero request-path pin/free" law, not measured). The
harness `tools/pinned-host-reserve-bench.py` measures `cuMemHostAlloc(PORTABLE)` through ctypes
on `libcuda`, arm `single` versus arm `chunked` (N threads joined), AB then BA, in one process.
It sizes on `MemFree` by default (the task's "free host RAM minus 25 percent" read literally:
a pin sized on `MemAvailable` evicts the page cache, which on BOX3 today holds another lane's
artifact); `--basis available` exists for a box that runs nothing else. BOX3 is one card with
88.4 GiB of host RAM and one NUMA node; the issue's box is a 2x B200 pair with a 288 GiB arena.
Nothing measured here transfers; the decision cell is that pair and is out of scope.

### BOX3 harness receipt (`pro-single-day12/arena-r2/`, N=5 pairs per order, executed-not-qualified)

23,486,005,248 bytes (75 percent of `MemFree`, 21.87 GiB), 8 chunks, 20 rows all `ok`, exit 0.
`single` median 3598.8 ms (AB 3578.2, BA 3609.7; min 3569.9, max 3626.5), `chunked` median
3583.8 ms (AB 3582.8, BA 3584.8; min 3571.0, max 3595.1); both 6.1 GiB/s; free 1389.7 versus
1421.0 ms. GPU idle, 40 C / 95.96 W at start and 36 C / 89.19 W at end at the 600 W limit; host
`loadavg` 0.33 to 0.88. The two medians differ by 0.4 percent, inside the single arm's own
AB/BA spread; the per-chunk completions inside one chunked row step by ~450 ms (451, 898, 1345,
1790, 2238, 2687, 3132, 3152 ms), so the eight concurrent `cuMemHostAlloc` calls complete one
after another: the driver serializes pinned host allocation on this box, and eight callers do not
use eight cores. A harness receipt for this box; not a #385 decision, and the 2x B200 pair at
288 GiB is a different question (`HOST-ARENA-STARTUP.md`, last section). A first arena cell
(`arena/`, attempt 1) exited 2 in under a second: `pinned-host-reserve-bench.py: error:
unrecognized arguments: --basis free`, the tree's harness predating the sizing commit; the tree
was moved to `2798e0b2e` and the cell rerun as `arena-r2`.

## CPU gates (all under `systemd-run --user --scope -p CPUQuota=1200% -p MemoryMax=28G`, runner `day12/run-gates.sh`, logs `day12/gates/`)

| gate | exit | verbatim tail |
|---|---|---|
| `cargo fmt --all -- --check` | 0 | (no output) |
| `cargo test -p memra-server --offline` | 0 | 3 suites, `passed` sums to 748, `failed` to 0 (day-12 tree); 749 on the review tree `8b29b2aa3` (the new plan/waste cell) |
| `cargo test -p memra-kv --offline` | 0 | 3 suites, `passed` sums to 73, `failed` to 0 |
| `cargo clippy -p memra-server --offline --all-targets -- -D warnings` | 0 | ``Finished `dev` profile [unoptimized + debuginfo] target(s) in 2m 22s``; review tree ``in 7.78s`` |
| `cargo clippy -p memra-kv --offline --all-targets -- -D warnings` | 0 | ``Finished `dev` profile [unoptimized + debuginfo] target(s) in 1.34s`` |
| `bash tools/check-flags.sh` (final tree) | 0 | `check-flags: every runtime MEMRA_* name resolves against 'docs/FLAGS.md' (no grandfather list)` |
| `bash tools/docs-registry-census.sh` (final tree) | 0 | `docs-registry-census: flags-table-census: docs/FLAGS.md tables=58 rows=903, every row matches its header` |
| `git diff --check` | 0 | (no output) |
| `git diff --cached --check` (the staged receipts; my addition) | 0 after one fix | first run exit 2: `research/spill-a-20260919/day12/gates/test-server.log:777: new blank line at EOF.` and the same in `test-kv.log`, cargo's trailing blank line in my own local logs; the one blank line was removed from each (no measurement touched), rerun 0 |
| `bash -n` on the gate and the nine cell scripts; `python3 -m py_compile` on the harness and the verifier | 0 | (no output) |

The Rust gates ran on the code tree at `405466cf7` and again on the review tree at `8b29b2aa3`
(fmt 0, `memra-server` 749 passed, `memra-kv` 73 passed, clippy server and kv 0; logs
`day12/gates/`, the review run overwrote the day-12 logs with the same names). No `MEMRA_*` read was
added in Rust; the gate's `MEMRA_TENANTGATE_*` knobs and `MEMRA_METRICS_TOKEN` are shell-side
(the census scans `crates/*/src`; `MEMRA_METRICS_TOKEN` is an existing server read with its row).

## Push

Two attempts of `git push origin lane/spill-a-20260919` at `a0ff733ba`, both logged
(`day12/push-attempt1.log`, `day12/push-attempt2.log`), no `--no-verify`, no skip variable.

1. With the branch's upstream set to `origin/lane/spill-a-20260919` (`432816926`, the day-11
   tip), the hook refused, verbatim: `pre-push: engine files touched after the last perf-ci
   battery.` with `base (merge-base with refs/remotes/origin/lane/spill-a-20260919):
   43281692609fd476f96625ba39fb40a79f3b714a` and six engine files
   (`crates/memra-engine/src/bin/kv_tier_gate/active.rs`, `fault.rs`, `fault_contract.rs`,
   `bin/run_lockstep.rs`, `hybrid_forward.rs`, `moe_cache.rs`), every one arriving through the
   merge of `origin/main` `be07f2d36` (`e5b1c10d0`); this lane changes no engine file.
2. `tools/push-range.sh` takes the merge-base with `origin/main` when the branch has no
   upstream (its documented second arm, the fork point of a branch that has merged main), so the
   upstream was unset (`git branch --unset-upstream`), the push retried and the upstream set
   back (`git branch -u origin/lane/spill-a-20260919`). The hook then ran its whole battery
   with base `be07f2d36b66c89071433fb6d7d5a1fed3954f8b`: `perf board is up to date`, `pre-push:
   flags census OK — runtime literal reads=867`, `pre-push: releasability censuses OK — publish
   list, stub ABI, arch matrix`, `pre-push: docs-registry census OK`, `public-boundary: 2 matches
   (2 grandfathered, 0 new).`, and `432816926..a0ff733ba  lane/spill-a-20260919 ->
   lane/spill-a-20260919`. The receipts commit that adds this file is pushed the same way with
   the upstream set (its range is then this commit alone).

This is a range correction, not a bypass: the gate evaluated the lane's real fork point and
found no engine file. The lead's day-11 push of `432816926` was the last time the lane's remote
tip moved, which is why the upstream-based range indicted main's own files.

## Integ15 review fixes (PR #597, revuto findings, lead decision)

Finding 1, docs contradicted the shipped behaviour. `docs/FLAGS.md` `MEMRA_KV_HOST_TENANT_PCT`
row rewritten (reclaim the tenant's own LRU first, then the bounded refusal; the Arc D
within-share displacement follow-up landed as memra#384; receipts now `prefix_host_tenant_reclaims`,
`prefix_host_tenant_reclaims_wasted`, `prefix_host_tenant_rejects`, the `evict (tenant share)` line
and the evaporation line with its suffix). `docs/SERVING.md` share-cap paragraph rewritten the same
way with its receipts, and the `/metrics` table gained the `prefix_host_tenant_reclaims` row (a
subset of evictions; reclaims moving while rejects stay flat means the cap is being served, not
refused) and the `prefix_host_tenant_reclaims_wasted` row (nonzero is a regression to read).

Finding 2, the reclaim's evictions were committed before the image was built, so five later
failure paths (`tier_charge` Err, the two `checked_*` bails, `host_roundtrip_digest` under
`MEMRA_KV_HOST_VERIFY`, `host_entry_from_device` Err, `bind_tier_image` Err or `insert` false)
could abandon the demotion after the row had paid `need` bytes, with `tenant_reclaims` counting it
as a success. Lead decision applied: the hook now runs `tenant_share_reclaim_plan` (pure, the
eligibility half; refuses before the copy exactly as before) and defers `reclaim_tenant_share` to
the `Ok(mut e)` arm after `bind_tier_image`, right before `insert`, so the only refusal left after
an eviction is `insert`'s own. The evicted count parks in `reclaim_pending`; a successful `insert`
consumes it; `waste_pending_reclaim` at every later exit (bind refused, reclaim refused, insert
refused, copy failed) books it into `tenant_reclaims_wasted` (published as
`prefix_host_tenant_reclaims_wasted`) with one `[prefix-host] tenant share reclaim WASTED: N own
entries evicted for a demotion that did not insert (...)` line. The fixed-arena path
(`MEMRA_GLM5_TP_KV_HOST=1`) reserves the planes' backing before the copy inside `reserve_image`,
so the reclaim cannot wait for the image there: `reserve_image(.., reclaim: true)` runs it from
the demote hook only (a handoff import passes `false` and keeps the old refusal), and a copy that
then fails is booked wasted by the hook's `Err` arm. Unit cells:
`host_cache_tenant_share_plan_is_pure_and_a_reclaim_is_consumed_by_insert_or_booked_wasted`
(the plan at the cap leaves the row whole and counts nothing, which is the state a failing copy,
digest or charge leaves behind; below the cap the plan is the default; a refused plan evicts
nothing; the happy path reclaims exactly once and the insert consumes it; a reclaim followed by a
refused insert books exactly one wasted entry; a waste call with nothing pending is a silent
zero), and the rewired source test (plan before the evaporation line before the copy; no
`reclaim_tenant_share(` before the copy; `tier_charge(` and `host_roundtrip_digest(` before the
copy; the reclaim inside the `Ok` arm after bind and before insert; four `waste_pending_reclaim(`
sites; both counters published and rendered). The three day-12 reclaim cells are unchanged and
still pass (same method contract).

Target card rerun (`pro-single-day12/tenant-fix-r4/`, fix arm only, N=1, executed-not-qualified,
600 W): the base binary is `origin/main` `be07f2d36` and did not change, so the attempt-3 base
receipts (`tenant-base-r3/`) stand and only the fix arm was rerun. `build-fix2.sh` at `8b29b2aa3`
(the review commit): `cargo build --release -p memra-server --offline -j 16` exit 0 in 15.74s,
binary sha256 `d47caec8d445d5fa96c512bd61dba463d609d9ecc2600777dbb184c840076ef4`, tree clean.
`GATE: kv-host-tenant-reclaim (fix arm) PASS` (28 `ok`, two new: `fix: /metrics has no wasted
reclaim (every reclaim's demotion inserted)` and `fix: no WASTED line`). The four reclaims and
their demotes, verbatim (the eviction now happens after the copy and bind, right before the
insert, and still prints immediately before the demote line):

```text
[prefix-host] evict (tenant share): 89 tokens, 160.7MB of tenant "t:acme"'s own entries for its 160.8MB demotion (row now 160.6MB / 408MB share = 38% of 1074MB, model gate, ns "t:acme\u{1f}s1")
[prefix-host] demote: 93 tokens, 160.8MB in 41.8ms (host resident 482.0MB / 1074MB, model gate, ns "t:acme\u{1f}s1")
[prefix-host] evict (tenant share): 86 tokens, 160.6MB of tenant "t:acme"'s own entries for its 160.8MB demotion (row now 160.8MB / 408MB share = 38% of 1074MB, model gate, ns "t:acme\u{1f}s1")
[prefix-host] demote: 93 tokens, 160.8MB in 6.2ms (host resident 482.2MB / 1074MB, model gate, ns "t:acme\u{1f}s1")
[prefix-host] evict (tenant share): 93 tokens, 160.8MB of tenant "t:acme"'s own entries for its 160.9MB demotion (row now 160.8MB / 408MB share = 38% of 1074MB, model gate, ns "t:acme\u{1f}s1")
[prefix-host] demote: 95 tokens, 160.9MB in 7.3ms (host resident 482.3MB / 1074MB, model gate, ns "t:acme\u{1f}s1")
[prefix-host] evict (tenant share): 93 tokens, 160.8MB of tenant "t:acme"'s own entries for its 161.3MB demotion (row now 160.9MB / 408MB share = 38% of 1074MB, model gate, ns "t:acme\u{1f}s1")
[prefix-host] demote: 108 tokens, 161.3MB in 41.4ms (host resident 643.7MB / 1074MB, model gate, ns "t:acme\u{1f}s1")
```

`/metrics`: `prefix_host_tenant_reclaims 4`, `prefix_host_tenant_reclaims_wasted 0`,
`prefix_host_tenant_rejects 0`, `entries 4`, `demotions 8`, `promotions 2`; zero `WASTED` lines;
beta r7 `cached_tokens 83`, acme r8 `cached_tokens 95`; every text byte-identical to the base arm
(the verifier's cross-arm law holds against `tenant-base-r3`): `verify-day12.py` `DAY12 PASS`
with `fix (fix2): source 8b29b2aa3… binary sha256 d47caec8…` (`day12/verify-day12.log`),
`tier-battery.py --validate` exit 0 on `tenant-fix-r4` (`day12/collector-validate-r4.log`).
The attempt-3 fix cell (`tenant-fix-r3/`, the pre-review build) stays as the record.

## Scope

Done: #384 code, unit cells, serving-shape gate on base and fix, `/metrics` counter; #385 plan
and one scaled harness cell. Not done, stated: the handoff import path still evaporates at the cap
(`insert` direct, no reclaim; a follow-up if an import ever hits it); a hugepage arm (no pool on
BOX3); the #385 decision cell (2x B200 pair, the issue's box); any 5090 run (no 5090-facing
default changed). Neither issue is closed; both carry a comment pointing here.
