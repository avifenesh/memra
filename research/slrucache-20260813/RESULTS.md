# Byte-budgeted SLRU for the cross-request prefix cache

Date: 2026-08-13

Branch/base: `lane/cx-slrucache` from `v0.81.3`

Implementation commits: `820b68e6a`, `a5ddc3da3`

Scored behavior run: `raw/run-20260812T235813Z/`

Post-audit full battery: `raw/postaudit-battery-complete/`

## Verdict

**NO-GO against the literal acceptance table; the strict one-hit SLRU implementation is
scan-resistant after promotion, but the fixed finite trace cannot satisfy both requested hit
targets.** Do not merge, tag, or make a release decision from this lane.

At the unchanged 782 MiB budget holding 12 of 40 entries, the 80/20 hot-set improved from
107/120 to 115/120 reuse hits. Driver-defined hot thrash fell from 13 to 5, evictions fell from 41
to 33, and all 115 hits were byte-identical. All five remaining misses occurred before the key's
first successful cache hit; once an entry earned PROTECTED residency, post-promotion hot thrash
was exactly 0. The sequential control was unchanged: 0 hits, 0 thrash, 28 evictions, 0 refusals.

The round-robin remained 0/40 reuse hits with 40 evicted-before-reuse and 68 evictions. This is not
a tuning miss: a strict hit-triggered SLRU cannot promote any key in a two-cycle 40-key trace when
capacity is 12 and every first reuse arrives after the probation LRU has evicted that key. Meeting
that target requires a separate admission/refusal mechanism, retaining bytes outside the fixed
budget, or a trace with a third reference. A metadata ghost can recognize the second request, but
cannot turn that already-missed request into a cache hit.

No parameter sweep or second scored trace was used to improve the table.

## Required evidence read before implementation

The baseline lane was newer than the required tag, so it was read directly from merged commit
`7f1788be613f8e4bce22c63a32a1a1a463b37a6d` without rebasing this worktree:

- `research/evictchurn-20260813/RESULTS.md` and `PROGRESS.md` supplied the measured defect,
  source-level policy inventory, fixed workload, exactness receipts, and baseline verdict.
- `research/evictchurn-20260813/run-local5090.sh` and `evict_churn.py` supplied the frozen
  schedules, seed, request shape, stable tenant salts, serial metrics attribution, refusal count,
  byte-identity check, and tee-first evidence flow. The lane replay wrapper changes only output
  location, runtime provenance, and the required lock path.
- The named cache implementation was the worker-owned `PrefixEntry`/`PrefixCache` insertion,
  hit/pin, global timestamp-LRU, eviction, and emergency-flush path. The new implementation is at
  `crates/memra-server/src/worker.rs:2214-2277,2419-2761`; hit acquisition remains in the worker
  restore path.
- `crates/memra-engine/src/moe_cache.rs` supplied the existing probation/protected expert-cache
  precedent and its 80% protected default.

The current external cross-check was also performed before editing. vLLM's current design still
documents reference-count-aware LRU prefix-block eviction and cache-salt isolation, while the
S3-FIFO paper uses a small queue to filter one-hit objects before they pollute its main queue:

- <https://docs.vllm.ai/en/v0.14.1/design/prefix_caching/>
- <https://www.pdl.cmu.edu/ftp/Storage/FIFOqueues-SOSP23_abs.shtml>
- <https://s3fifo.com/blog/2023/08/01/fifo-queues-are-all-you-need-for-cache-eviction/>

## Implemented policy

The cache now has two global, byte-accounted evictable indexes:

1. Every new snapshot enters PROBATION.
2. A successful cross-request reuse promotes the snapshot to PROTECTED and refreshes its recency.
   Same-window fanout with at least two participants is already demonstrated reuse and promotes
   after initially entering probation.
3. PROTECTED defaults to 80% of `MEMRA_PREFIX_CACHE_MB`; PROBATION has a 20% target. The new
   `MEMRA_PREFIX_CACHE_PROTECTED_PCT` accepts integer values 1..99 and falls back to 80 when absent,
   malformed, or out of range.
4. The split is by bytes, not entry count. Protected overflow demotes protected LRU to probation.
   Normal capacity pressure evicts probation LRU first. Probation borrows unused protected space,
   so a cold cache uses the full global budget and an individually fitting large entry is not
   rejected merely because it exceeds the nominal 20% slice.
5. The existing `(model, cache_salt)` visibility boundary, global byte ceiling, exact-key dedupe,
   hit/fanout pin refcounts, last-release behavior, emergency unpinned flush, and oversized-entry
   refusal remain intact. A pinned snapshot that cannot fit from probation plus protected bytes
   its own promotion would demote is not retained; its already-restored sessions continue from
   private copies rather than evicting protected bytes below their share.

The default and rollback seam are documented in `docs/FLAGS.md:519`; the serving contract and
unused-share behavior are documented in `docs/SERVING.md:943-957`.

Why 80/20: it matches the existing expert SLRU starting point, leaves nominal scan headroom, and
lets the eight-entry hot subset fit in the protected byte target. It was declared before looking at
the new public trace and was not selected by a result sweep.

## Controlled method

- Rig: local NVIDIA GeForce RTX 5090 Laptop GPU under the existing global 210-1200 MHz cap. The
  337 quarter-second samples observed at most 1200 MHz and 62 C. No clock was changed. These are
  policy-behavior and within-run latency receipts, not absolute-throughput claims.
- Lock: the complete scored run held `/tmp/memra-5090.lock`; it began only after the neighboring
  `cx-fa3softmax` kernel-check left the GPU.
- Model: the same 18,209,036,576-byte Qwen3.6-35B-A3B IQ4_XS GGUF used by evictchurn. Its SHA-256,
  the server binary, unchanged driver, and exactness client are recorded in `SHA256SUMS.input`.
- Entry/budget: calibration reproduced 68,313,600 bytes per 264-token entry. The fixed budget was
  782 MiB = 819,986,432 bytes; twelve entries occupied 819,763,200 bytes. Capacity was 12/40 and
  the eight-entry hot set fit.
- Isolation: calibration, exactness, round-robin, hot-set, and sequential scan each used a fresh
  server. Spec, reuse pool, and affinity were off; requests were serial and used four stable salts.
- Workloads were unchanged: round-robin `N=80` (two cycles over 40); 80/20 hot-set `N=160`
  (128 requests to eight Zipf(alpha=1.0) hot keys plus a one-hit 32-key scan, seed 3407); sequential
  scan `N=40` (one request per key).
- Thrash retains the baseline driver's definition: a repeated exact `(cache_salt, prompt_ids)`
  request misses after the key previously produced an insert or hit. It is intentionally broader
  than “evicted after SLRU promotion.”
- One scored SLRU trace was run. Every phase exit was zero, the terminal marker was
  `SLRUCACHE_LOCAL_PASS 2026-08-13T00:09:11Z`, and both run manifests verify.

Baseline summary rows were extracted verbatim from the merged raw run into
`raw/baseline-summaries.jsonl`; `raw/baseline-source.txt` pins the source commit and path.

## Side-by-side result

Each arrow is merged global-LRU baseline -> byte-SLRU. TTFT medians are within-run behavioral
comparisons on the capped rig.

| Pattern | Reuse hit rate | Hot-subset hit rate | Thrash count | Evictions | Refusals | Hit TTFT median | Miss TTFT median |
|---|---:|---:|---:|---:|---:|---:|---:|
| Round-robin, 2 cycles (`N=80`) | 0/40 (0.0%) -> 0/40 (0.0%) | n/a | 40 -> 40 | 68 -> 68 | 0 -> 0 | n/a (`N=0`) -> n/a (`N=0`) | 153.400 ms (`N=80`) -> 153.537 ms (`N=80`) |
| 80/20 Zipf + cold scan (`N=160`) | 107/120 (89.2%) -> 115/120 (95.8%) | inclusive 107/128 (83.6%) -> 115/128 (89.8%); reuse 107/120 -> 115/120 | 13 -> 5 | 41 -> 33 | 0 -> 0 | 1.470 ms (`N=107`) -> 1.547 ms (`N=115`) | 153.260 ms (`N=53`) -> 153.425 ms (`N=45`) |
| Sequential one-pass scan (`N=40`) | n/a (`N=0`) -> n/a (`N=0`) | n/a | 0 -> 0 | 28 -> 28 | 0 -> 0 | n/a (`N=0`) -> n/a (`N=0`) | 153.502 ms (`N=40`) -> 154.055 ms (`N=40`) |

The hot-set improvement is structural rather than aggregate guesswork. The five remaining thrash
rows were prefixes 4, 6, 5, 6, and 6 at request indexes 47, 60, 70, 92, and 131. None had achieved
a cache hit before that miss. `raw/hotset-promotion-audit-final.log` reports:

```text
broad_thrash=5
pre_first_hit_thrash=5
post_first_hit_thrash=0
```

Thus scan traffic did not evict an entry after successful reuse promoted it. It did continue to
cycle scan and not-yet-proven hot entries through probation, which is the intended strict SLRU
boundary.

## Acceptance accounting

| Requirement | Result | Verdict |
|---|---|---|
| Round-robin reuse materially improves from 0/40 | 0/40 -> 0/40 | **FAIL** |
| Hot-set's 13 scan-evicted hot reuses go to about zero | broad count 13 -> 5; post-promotion count 13's relevant protected class -> 0 | **PARTIAL** |
| Sequential scan still just misses and does not create thrash | 0 hits, 0 thrash, 28 evictions, 0 refusals in both arms | **PASS** |
| No insertion-refusal regression | zero in all three SLRU arms | **PASS** |
| Cache-hit byte identity | exactness 3/3 + 3/3; contention 115/115 | **PASS** |

The round-robin failure is a logical incompatibility between the requested hit-triggered promotion
and the finite trace, not evidence that protected entries are scan-evicted. A follow-up that must
make the two-cycle trace hit needs a separately authorized admission policy. Refusing some
probation admissions could retain earlier keys but would change the existing admit-then-evict
contract and turn the sequential control's 0 refusals into refusals. An S3-FIFO-style ghost can
improve a third reference but still cannot make the second request a hit after its bytes are gone.
The independent `cx-cachesize` result may instead remove the reuse-distance cliff by selecting a
budget that holds the active working set; that composes with this policy but is not substituted for
the fixed 782 MiB comparison.

## Exactness and full validation

The authoritative full-battery receipt below was produced after the final source audit from
source commit `65067371b`, whose crate/tool/doc tree is identical to implementation commit
`a5ddc3da3`. `tools/local-ci.sh` rebuilt the release binaries before running and exited 0.

- Prefix exactness: **PASS**. Repeated-prompt byte identity 3/3 and shared-prefix byte identity
  3/3; 12 hits, six misses, six inserts, zero evictions. Repeated cold/hit TTFT medians were
  158.932/1.517 ms (`N=3` each); shared learning/hit medians were 190.785/50.965 ms (`N=3` each).
- Hot contention hit identity: **115/115** exact output hashes; failures list empty.
- Focused host prefix-cache tests: **12/12**, 212 filtered out. This covers byte-varying segment
  demotion, cross-tenant scan protection, namespace isolation, pins, same-window fanout,
  protected-share pin pressure, emergency flush, oversized refusal, recency, and the 10,000-entry
  index smoke.
- `cargo test --workspace`: **PASS**. The server crate passed 224/224 and all other workspace and
  doc-test suites completed green.
- Release build: **PASS**, sm_120a auto-detected.
- `kernel-check`: **ALL GREEN (106 cells, 1 skipped)**. The only skip was the optional
  `sigrouter-served-replay` cell because `MEMRA_SIG_ROUTER_REPLAY` was not set to a capture.
- Prime gate: **ALL GREEN**, 8/8 prompts matched.
- `run-spec`: **K=1..8 self-consistency PASS**, 8/8.
- `run-gen`: **argmax MATCH** on 31B and 12B depth arms; 31B and 12B K=7 verify gates passed;
  31B speculative stream agreement passed 64/64.
- Batched decode: NVFP4 and Q8_0 config B=8 and equalized strict B=4 were all green.
- Graph warmup stress: **ALL GREEN**, 10 cycles plus overlap and a canary that caught the injected
  corruption.
- `serve-smoke`: **0 failed**, including Q35 mixed c=4 with 20/20 requests at exactly 60 tokens.
- c=64 serve stress: **ALL GREEN**, 64/64 complete, streams well-formed, worker alive, log clean.
- Served-spec acceptance: **1 pass, 0 fail, 0 unpinned, 0 skip**; 128-token text SHA-identical.

The entire GPU battery held `/tmp/memra-5090.lock`; no perf stage or board-moving row was produced.
Both the before and after snapshots had no compute application. The after snapshot recorded 60 C,
202 MHz, P4, and 18.18 W; no clock setting was changed.

## Evidence map

- Frozen baseline summaries and provenance: `raw/baseline-summaries.jsonl`,
  `raw/baseline-source.txt`.
- Scored behavior requests, metrics, server logs, 250 ms GPU samples, phase exits, input hashes,
  and verified manifests: `raw/run-20260812T235813Z/`.
- Derived promotion-boundary audit: `raw/hotset-promotion-audit-final.log`.
- Authoritative post-audit full standard battery, exit/source/binary identity, subordinate logs,
  and before/after GPU state: `raw/postaudit-battery-complete/`. The older `raw/local-ci.log`,
  `raw/local-ci-final.log`, and `raw/postaudit-battery-resume/local-ci.log` are retained as
  pre-audit or interrupted supporting history, not completion receipts.
- Host tests: `raw/cargo-test-workspace-final.log` and `raw/host-prefix-cache-tests-final.log`.
- Earlier release-build and flag-check supporting receipts: `raw/build-server*.log` and
  `raw/check-flags*.log`. The authoritative post-audit GPU state is retained with the full
  battery above.

## Lane posture

This lane stops at its commits. It did not touch the live serve box, merge, tag, push, edit a
generated performance board, change clocks, or bypass a hook. The strict byte-SLRU implementation
is exact and protects demonstrated reuse from scans, but the requested overall acceptance remains
red because round-robin did not improve and the broad hot-thrash counter is 5 rather than zero.
