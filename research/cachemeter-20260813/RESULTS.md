# cx-cachemeter — baseline-first result

## Independent verification after `96361c531`

**PASS: the prefix-snapshot fix restores the money-path cache behavior under both policies, and
neither held descendant reintroduces the zero-insert regression.** All four requested focused
verification cells passed without changing either gate, increasing a budget, or enabling partial
restore.

The original regression boundary remains `c3976d488`: its all-allocated-layers
`l.len == cache.pos` check rejected the allocated-but-unexecuted NextN slot before
`PrefixCache::insert`. Fix `96361c531` records that zero-length slot as absent and accepts the
symmetric live-destination/absent-source restore pair. Runtime verification now shows publication
and attribution actually occurring; this is no longer source-only reasoning.

### Actual counters

Every row observed the same exact cache-accounting values:

- Cache-metering arm: prompt/cached/computed tokens `1632/1024/608`, hit-token ratio
  `0.6274509803921569`, hits/misses/inserts `4/2/2`, hit tokens `1024`, cold/shared LCP buckets
  `2/4`, and revenue multiplier `2.6842`.
- Q35 mixed-c=4 arm: `20/20` exact responses, hits/misses/inserts `18/2/2`, prompt/cached/computed
  tokens `97200/87480/9720`, hit-token ratio `0.9`, retained entries `8 -> 10`, zero prompt,
  cached-token, or hit-token drift, no seed failures, and no short/non-length response.

| Engine tree | Policy | Server SHA-256 | Cache-meter | Q35 hot-seed/c=4 |
|---|---|---|---|---|
| fix `96361c531` + research-only lane commits | shipped SLRU | `59889be44b48c3c9b3ed19a229d265ab8fbcdf98777400cc266a7a1e5626b628` | PASS | PASS |
| same engine and binary | explicit plain LRU | same | PASS | PASS |
| fix + `keep/cx-budgetsize-merged` | shipped SLRU | `c7dd155100bab78cdf7925abc7e5adb2a36ad370563824f45b718a65af70d84f` | PASS | PASS |
| fix + `keep/cx-shmconflict-merged` | shipped SLRU | `d18408cde91374094eebe374366df36d334feaee2687081b2fafe83da0e2ef68` | PASS | PASS |

The budget-size row exercised its derived 9B budget of `348651520` bytes; Q35 still exercised the
explicit `4294967296`-byte serve-smoke budget. The shared-memory row includes the actual
`crates/memra-engine/cu/flash_attn.cu` change. Neither descendant produced a snapshot refusal,
`prefix not cached`, or gate failure line.

### Tree provenance

The evidence lane was rebased onto exact fix
`96361c531d26bdd95f4330617b872b2fa7d96f3d`; outside
`research/cachemeter-20260813/`, its engine tree is identical to that commit. The two held branches
were tested through clean detached virtual merges without moving either preserved branch:

- fix + budget-size: commit `5c602b340bf119e1b8159e6e195359d4cef7a4e8`, tree
  `2d55b47d278319363e8fa3d957185841c2348b66`;
- fix + shared-memory-conflict: commit `50d3350be148af1be9c0a7820c0adfa171a3f873`, tree
  `9310d8008700aca49de0add89a877ba799cc71ad`.

The live fetched `origin/main` remained `56f7ac0d8a20c367d6dba25cc03427098bf7f248`; none of the
local fix/repair/held-merge state was pushed by this lane.

### Broader local battery receipt

The separately launched owner battery was copied byte-for-byte to
`raw/verification/owner-battery-cachefix.log`. It records `kernel-check: ALL GREEN (106 cells, 1
skipped)`, Q35 `run-spec K=1..8` PASS, 31B and 12B `run-gen` MATCH, and `serve-smoke: 0 failed`,
including both cache gates. It also records four cross-day performance tripwire FAIL rows and
`perf stage: 4 fail, 0 warn`; therefore this report does **not** relabel that full battery as wholly
green based on its trailing `BATTERY_EXIT=0` line. The focused cache verdict above is independently
green and unaffected by those throughput comparisons.

An independent `TMPDIR=/home/avifenesh/tmp-lanes cargo test -p memra-server` run on this fixed lane
also passed: **248 passed, 0 failed**. It includes
`zero_length_nextn_layer_is_absent_not_corrupt`.

No merge, tag, push, generated-board edit, threshold relaxation, `cargo fmt`, or hook bypass was
performed.

## Historical baseline-first verdict

**ALREADY BROKEN AT THE PUSHED BASE `56f7ac0d8`.** The cache-metering and Q35 mixed-c=4 arms both
fail before either preserved merge is present. Consequently, neither `keep/cx-budgetsize-merged`
nor `keep/cx-shmconflict-merged` introduced the observed zero-insert regression.

Steering 2 made local `keep/cx-eosclass-merged` (`43caa7e12`, SLRU plus B1FAST) the primary target.
It also fails both focused arms with the same signature under both the shipped SLRU policy and
explicit plain LRU. B1FAST does not repair cache publication, and the LRU rollback does not reach
an admission: snapshot construction fails first.

This conclusion is limited to the reported zero-insert symptom. It does not claim that the two
untested descendant branches have no independent defects.

## Source boundary

The branch began at exact pushed tip
`56f7ac0d8a20c367d6dba25cc03427098bf7f248`. The original focused runs recorded commit
`8d9041b97cf08deac9f440126ebe9674e5c71cc5`, whose only delta from that base is this lane's
research ledger, initiating log copy, and focused-arm driver. This command exits zero:

```text
git diff --quiet 56f7ac0d8 8d9041b97 -- . ':(exclude)research/cachemeter-20260813'
```

After steering 2, the evidence commits were rebased onto exact local target
`43caa7e1213167e685012b368479ead4e1dc9850`. The current focused runs recorded commit
`56610d0f513ad4e41b36f9ad7de2e792dbd1d7ee`; this equivalent check also exits zero with
`43caa7e12` as its left side. Thus each run exercised the named engine/server/gate tree exactly,
plus research-only evidence files. Ancestry checks also exit zero for both held-merge edges:

```text
56f7ac0d8 -> keep/cx-budgetsize-merged -> keep/cx-shmconflict-merged
```

The two original-base runs built the same release server binary, SHA-256
`d908cec3d5c12209150c40f8479860a763e056c1677cf9fe605b4205e6f7895f`.

## Runs

`run-focused-arms.sh` copies the two relevant `tools/serve-smoke.sh` arms without changing their
commands or assertions:

- cache metering: `MEMRA_SERVE_SPEC=0`, then `tools/cache-meter-gate.py ... --n 5 --k 256
  --suffix 16`;
- Q35: the same environment block as `tools/serve-smoke.sh`, including
  `MEMRA_PREFIX_CACHE_MB=4096`, then `tools/q35-cold-mixed-gate.py ... --timeout 600`.

Each invocation was serialized under `flock /tmp/memra-5090.lock` and captured with `tee` before
interpretation. The original-base runs left the GPU at 49 MiB with no compute application listed;
the same no-compute-app state was recorded after the current-tree runs.

| Engine tree | Policy | Cache-metering arm | Q35 mixed c=4 arm | Direct result |
|---|---|---|---|---|
| pushed `56f7ac0d8` | shipped SLRU 80/20 | FAIL, 13 assertions | FAIL | 0 inserts, 0 hits, 6 misses; Q35 retained 0 entries after 8 hot seeds |
| pushed `56f7ac0d8` | explicit LRU | FAIL, 13 assertions | FAIL | same zero-insert result |
| local `43caa7e12` | shipped SLRU 80/20 | FAIL, 13 assertions | FAIL | same zero-insert result; Q35 cell had 0 inserts, 0 hits, 20 misses |
| local `43caa7e12` | explicit LRU | FAIL, 13 assertions | FAIL | same zero-insert result; Q35 cell had 0 inserts, 0 hits, 20 misses |

The default-SLRU cache-metering counter assertions were:

```text
FAIL: prefix_cache_hits == N-1 — got 0
FAIL: prefix_cache_misses == 2 (one A leader + cross-salt B) — got 6
FAIL: prefix_cache_inserts == 2 (shared A prefix + B seed) — got 0
```

All explicit-LRU runs emitted the same lines and an identical `cache-meter-gate: 13 failed`
summary. This directly refutes the proposed SLRU non-promotion explanation: selecting plain LRU
does not make one entry publish on either engine tree.

Q35 completed all responses at exactly 60 tokens, but its required cache attribution was absent.
Both policies ended with this captured gate failure:

```text
"seed_failures": ["q35: cache retains 0 entries after 8 hot seeds"]
```

Every Q35 policy/tree cell recorded 0 inserts and 0 hits. On `43caa7e12`, both policies recorded
20 misses. The separate carried-prime assertion passed in all four Q35 runs.

## What the server itself reported

The cache was neither disabled nor silently using a zero budget. On the default-policy run, boot
reported:

```text
[prefix-cache] on: budget 268MB (MEMRA_PREFIX_CACHE_MB), policy byte-SLRU protected/probation 80%/20% (MEMRA_PREFIX_CACHE_PROTECTED_PCT), min prefix 64 tokens, immediate partial restore=off (rollback) (transformer-only; hybrid mid-entry + routed-MoE N/A)
[prefix-cache] on: budget 4295MB (MEMRA_PREFIX_CACHE_MB), policy byte-SLRU protected/probation 80%/20% (MEMRA_PREFIX_CACHE_PROTECTED_PCT), min prefix 64 tokens, immediate partial restore=off (rollback) (transformer-only; hybrid mid-entry + routed-MoE N/A)
```

The second line is Q35 recognizing serve-smoke's explicit 4096 MiB configuration. In the LRU run,
the same two budgets were reported with `policy plain-LRU`.

The captured refusal is a snapshot validation failure, not a budget or eviction message. The 9B
gate repeatedly logged:

```text
[prefix-dedup] snapshot failed (prefix snapshot layer 32 len 0 != cache pos 256); siblings prime cold
[prefix-cache] snapshot failed (prefix snapshot layer 32 len 0 != cache pos 272); prefix not cached
```

Q35 repeatedly logged:

```text
[prefix-cache] snapshot failed (prefix snapshot layer 40 len 0 != cache pos 4860); prefix not cached
```

Those verbatim messages identify the publication refusal. `prefix_snapshot` returns `Err` when any
allocated KV slot has `l.len != cache.pos`; its caller's `Err` arm prints `prefix not cached` and
does not call `PrefixCache::insert`. The insert counter therefore cannot move: `self.inserts += 1`
lives inside the insert path after its identity and budget preflights. It counts successful cache
admissions, not SLRU promotions or snapshot attempts.

## Exact refusing change and mechanism

The hard error was added by exact commit
`c3976d48855e0f87ad5155692dec323e3e3c6462` (`feat: restore transformer prefixes at LCP splits`):

```rust
if l.len != cache.pos {
    return Err(format!(
        "prefix snapshot layer {il} len {} != cache pos {}",
        l.len, cache.pos,
    ).into());
}
```

That commit is an ancestor of pushed `56f7ac0d8`. The mismatch is structural for these embedded-MTP
artifacts, not an eviction-policy outcome:

- the 9B artifact declares 33 layers with `nextn=1`; its executed trunk has 32 layers, and the
  refusal is at slot 32;
- Q35 declares 41 layers with `nextn=1`; its executed trunk has 40 layers, and the refusal is at
  slot 40;
- `Cache::new_inner` allocates slots across `cfg.n_layer`, while target serving executes
  `n_trunk = n_layer - nextn_predict_layers`. The trailing NextN KV slot therefore remains length
  zero while `cache.pos` advances with the trunk.

Before `c3976d488`, `prefix_snapshot` copied each slot at its own recorded length and did not reject
this trailing zero-length NextN slot. The new all-slots-equal invariant now returns before either
SLRU or LRU sees the entry. This is the exact hunk that produces the observed zero-insert refusal.
A pre/post GPU run at `c3976d488^` and `c3976d488` was not launched after the task's pushed-base
stop condition, so this report does not relabel that source/history isolation as a completed
runtime first-bad bisect.

## Historical stop decision and then-unrun gates

The task explicitly defined a pushed-base failure as a complete result and a priority change.
Steering 2 authorized the bounded repeat on `43caa7e12`; both policy arms failed there too. At that
checkpoint, this lane therefore did not test the budget-size/shared-memory descendant tips, modify
engine code, weaken a gate, raise a budget, run `cargo test -p memra-server`, or launch the full
`tools/local-ci.sh --perf-quick` battery. Running the full battery after both of its focused arms
are already proven red would not qualify a fix and would contradict the baseline-first stop.

No merge, tag, push, generated-board edit, `cargo fmt`, or hook bypass was performed.

## Evidence map

- `raw/battery-perfci-pre-reset-20260813.log`: byte-identical initiating log; SHA-256
  `52b8a528a8b329a07013baade93393df343febf7c3828c812c3ef62ddc25f2de`.
- `raw/base-default-slru/`: complete default-policy build, binary hash, driver, gate, and server
  logs.
- `raw/base-explicit-lru/`: complete explicit-LRU build, binary hash, driver, gate, and server
  logs.
- `raw/eosclass-default-slru/`: complete `43caa7e12`-engine default-policy build, binary hash,
  driver, gate, and server logs.
- `raw/eosclass-explicit-lru/`: complete `43caa7e12`-engine explicit-LRU build, binary hash, exit
  receipt, driver, gate, and server logs.
- `raw/source-isolation.log`: exact commit diff, current counter/caller source, cache-allocation and
  trunk-boundary source, and the two models' recorded NextN geometry.
- `raw/fix-default-slru/` and `raw/fix-explicit-lru/`: independent fixed-base build, binary,
  counter scrape, gate, server, and exit receipts under both policies.
- `raw/fix-budgetsize-default-slru/` and `raw/fix-shmconflict-default-slru/`: the same receipts for
  both held descendants on top of the fix.
- `raw/verification/merge-tree-provenance.log`: exact virtual-merge commits, trees, parents, and
  retained-fix checks.
- `raw/verification/focused-counter-summary.log`: parsed actual counters and clean failure scans for
  all four verification cells.
- `raw/verification/cargo-test-memra-server.log`: independent 248/0 server test receipt.
- `raw/verification/owner-battery-cachefix.log`: byte-identical broader owner battery receipt,
  including green correctness/serve-smoke and the four red perf tripwires.
- `raw/MANIFEST.sha256`: checksums for every raw receipt.
