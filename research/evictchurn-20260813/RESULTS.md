# Prefix-cache eviction churn under a multi-tenant working set

Date: 2026-08-13

Branch/base: `lane/cx-evictchurn` from `v0.81.2`

Scored run: `raw/run-20260812T221018Z/`

## Verdict

**NEEDS-POLICY-FIX. The current global timestamp-LRU policy thrashes under cyclic contention and
measurably lets scan traffic flush proven-hot prefixes.** It is not insertion order and it does
not normally refuse an entry that fits: it inserts first, then evicts the globally least-recently
used unpinned entries until the global byte budget is met.

At a fixed 782 MiB budget holding 12 of 40 equal-size entries (30% of the exercised working set),
round-robin produced zero hits and evicted every entry before its one reuse. In the 80/20 hot-set
trace, the cache retained the hottest mass well enough for 107/120 hot reuses to hit, but the 32
one-hit cold prefixes still displaced 13 previously reusable hot entries. Thus this is not the
desired graceful behavior, “the hot set stays resident and the cold scan just misses.” Revenue
does not collapse on this trace, but 10.8% of hot reuse opportunities are lost to eviction churn.

No policy change is shipped in this lane. The minimal follow-up is a byte-budgeted
probation/protected SLRU replay, using the already-established expert-bank policy shape, followed
by the full engine-change exactness battery before any default change.

## Required source inventory

The starting report identifies cache sizing as unresolved and explicitly calls for the actual
working-set bytes, active tenant prefixes, and measured churn
(`research/prefixmoney-20260812/REPORT.md:134-150`). The new harness keeps that work's external
shape rather than inventing a new API: direct `prompt_ids`, greedy seeded streaming completions,
`cache_salt`, usage-based cached-token truth, metrics snapshots, and raw JSONL receipts
(`research/prefixmoney-20260812/prefix_gate.py:18-94,115-133`;
`research/prefixmoney-20260812/cache_concurrency.py:37-40,43-97`).

The public serving contract says entries are compact device snapshots keyed by exact token
prefix, per model, reusable through a deep copy, and “LRU under the byte budget”
(`docs/SERVING.md:925-933`). It also says sessions win over unpinned residency and live hit/fanout
leases remain pinned (`docs/SERVING.md:943-948`). `cache_salt` is an exact namespace boundary:
only the same salt shares history, no salt means namespace `""`, and the prefix byte budget remains
global across namespaces (`docs/SERVING.md:962-982`).

### What the code actually does

The decisive code quotations are:

> “POLICY IDENTICAL: both pick the global minimum `last_use` (timestamp-LRU)”
> — `crates/memra-server/src/worker.rs:2489-2497`

> `if e.bytes > budget { ... "skip {why} insert: entry ... > budget ..." ... return None; }`
> — `crates/memra-server/src/worker.rs:2530-2541`

> `while self.total_bytes > budget { ... self.lru.values().next() ... self.evictions += 1; }`
> — `crates/memra-server/src/worker.rs:2549-2575`

Together these answer the motivating ambiguity: the policy is LRU eviction for normal fitting
entries; refusal is a separate oversized-entry or pinned-capacity path.

1. **The prefix policy lives in the worker, not in `memra-engine`.** `memra-engine` re-exports the
   shared dual KV/recurrent-cache structure from `memra-kv`
   (`crates/memra-engine/src/lib.rs:29-33`; `crates/memra-kv/src/lib.rs:1-8`). The reusable
   cross-request `PrefixEntry` and eviction controller are worker-owned
   (`crates/memra-server/src/worker.rs:2186-2237`).
2. **Visibility is tenant-scoped; capacity is not.** Entries are partitioned by `(model,
   namespace)` and lookup scans only that pool (`worker.rs:2212-2216,2359-2374`). The insert
   comment and code make the byte budget global across namespaces (`worker.rs:2489-2497`). A busy
   tenant can therefore evict an idle tenant's entry even though it cannot read or hit it.
3. **Victim policy is exact timestamp LRU across all namespaces.** Every entry carries
   `last_use`, a unique tie-breaking id, bytes, and a pin count (`worker.rs:2186-2203`). The
   recency `BTreeMap` orders all evictable entries by `(last_use,id)`; the comments state the first
   key is the same global minimum chosen by the old scan and that insertion-order tie breaking is
   only determinization, “never a different policy” (`worker.rs:2217-2231`). A successful hit pins
   the entry and refreshes recency (`worker.rs:2423-2459,6163-6173`).
4. **A normal fitting insert evicts; it is not admission-controlled.** Exact duplicate keys are a
   no-op (or gain leases). An individual entry larger than the entire budget is refused with
   `skip {why} insert: entry ... > budget ...`; an in-flight pinned insert is also refused when it
   cannot fit beside already pinned bytes (`worker.rs:2530-2547`). Otherwise the new entry is
   counted and installed, then global LRU victims are removed until `total_bytes <= budget`
   (`worker.rs:2549-2578`). This explains the earlier `343.0MB > budget 268MB` log: refusal is the
   oversized-entry branch, not the ordinary contention policy.
5. **Pins and allocation pressure are exceptions to ordinary LRU.** Pinned entries are absent
   from the evictable index until their last lease retires (`worker.rs:2423-2459`). If allocating
   a fresh session cache fails during restore, every unpinned prefix entry is evicted and the cold
   path serves (`worker.rs:6175-6183`; the flush loop is `worker.rs:2581-2591`).
6. **Stored state is a deep device copy.** Snapshot copies the per-layer KV bytes and recurrent
   conv/SSM state and accounts their bytes; restore copies those bytes into a fresh session cache
   (`worker.rs:2603-2657,2660-2708`). Cold prompts seed at prefill completion and LCP learning
   inserts at the exact boundary (`worker.rs:2711-2724,2770-2784,7186-7197`).
7. **The source tests encode these semantics.** A touched old entry survives in favor of the
   untouched second-oldest; pin leases block eviction; emergency flush preserves pins; and a
   10,000-entry flush removes the oldest half (`worker.rs:9961-10063`). The targeted host receipt
   passed all eight prefix-cache tests.

Policy inventory in one sentence: **global, entry-granular, byte-budgeted timestamp LRU across
tenant namespaces; hit/fanout leases are temporarily unevictable; ordinary fitting inserts always
admit and evict as needed; only exact duplicates, individually oversized entries, and pinned-capacity
conflicts skip insertion; session-allocation pressure flushes all unpinned entries.** Victim choice
is recency-only, not entry-size-aware or frequency-aware.

## Method

- Rig: local NVIDIA GeForce RTX 5090 Laptop GPU, global 210–1200 MHz cap untouched. The 355
  quarter-second samples observed a maximum 1200 MHz SM clock and 64 C. This is a behavior lane;
  none of the timing values below is an absolute-throughput claim.
- Model: requested 18,209,036,576-byte
  `/data/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf` (SHA-256
  `df27a780435b7b45c2597536112ea3cb091f8544c3d0c3318d9f4258b31f7adf`). It fit, so the smaller
  9B fallback was not used.
- Runtime: release `memra-server` built from the lane's `v0.81.2` runtime source, sm_120a,
  SHA-256 `ded52bfb8d2238eea45cec5262f1d2f25447cc427746984152ea5a6c8a1f8a60`.
- Serving shape: single GPU, plain batched serving, spec OFF, reuse/affinity pools OFF, context
  512, four sessions. Every pattern starts a fresh server so state never leaks between arms.
- Workload: `W=40` distinct exact 264-token prompts across `T=4` stable salts, ten prefixes per
  tenant (tenant is `prefix_id mod 4`), generated greedily for up to eight tokens. Requests are serial so each
  metrics delta belongs to exactly one request; this measures policy behavior, not scheduler
  throughput.
- Budget: one cold calibration entry consumed 68,313,600 bytes. The scored budget was then fixed
  once at 782 MiB = 819,986,432 bytes for every arm. Twelve equal entries occupy 819,763,200 bytes,
  so capacity was 12/40 = 30%, while the eight-entry hot subset could fit.
- Patterns: round-robin is two complete cycles over all 40 keys (`N=80`); hot-set is 128 hot
  requests to eight keys using fixed-seed Zipf(alpha=1.0) plus a one-hit scan of all 32 cold keys,
  shuffled together (`N=160`); sequential scan visits 40 new keys once (`N=40`). Seed is 3407.
- Thrash signal: a repeated exact `(tenant cache_salt, prompt_ids)` request misses after that key
  previously produced an insert or hit. It would have hit if its retained entry remained. This is
  counted only on real reuse opportunities; the one-pass scan consequently has zero thrash even
  though it produces evictions.
- Evictions/inserts/hits/misses come from per-request `/metrics` deltas. Refusals count the
  worker's existing `[prefix-cache] skip ... insert:` lines. Hit output hashes are compared with
  that exact key's first cold output.
- TTFT is first visible UTF-8. A valid terminal EOS with no visible bytes is retained and explicitly
  labeled `terminal_eos_event` (8/80 round-robin, 4/160 hot-set misses, 4/40 scan); it is never
  silently discarded. Every hot-set hit had visible output.

## Churn results

Every value in this table is from the thermally capped local 5090 (210–1200 MHz, no clock change),
and each pattern is one isolated scored trace. Timing is included only to show the within-run
behavioral hit/miss separation.

| Pattern | N | Hits / requests | Reuse hits | Hot hits | Evictions (per 100 req) | Refusals | Evicted before reuse (per 100 req) | Hit TTFT median | Miss TTFT median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Round-robin, 2 cycles | 80 | 0/80 (0.0%) | 0/40 (0.0%) | n/a | 68 (85.000) | 0 | 40 (50.000; 100% of reuse) | n/a (N=0) | 153.400 ms (N=80) |
| 80/20 Zipf + cold scan | 160 | 107/160 (66.9%) | 107/120 (89.2%) | 107/128 inclusive (83.6%); 107/120 hot reuse (89.2%) | 41 (25.625) | 0 | 13 (8.125; 10.8% of reuse) | 1.470 ms (N=107) | 153.260 ms (N=53) |
| Sequential one-pass scan | 40 | 0/40 (0.0%) | n/a (N=0) | n/a | 28 (70.000) | 0 | 0 (0.000; no reuse) | n/a (N=0) | 153.502 ms (N=40) |

The hot-set loss is directly attributable, not an aggregate guess: all 13
`evicted_before_reuse` rows are hot prefixes. Prefix 0, the most frequent, missed only on its
first request and then hit 42 times; the colder end of the hot set was polluted repeatedly
(prefixes 3, 4, and 6 each suffered three premature evictions). This is exactly the frequency
blindness expected from recency-only admission under an interleaved scan.

The sequential control is important: 28 evictions alone are not called thrash because no key was
requested twice. The failure appears when scan admissions displace something that later returns.
Round-robin makes that boundary unambiguous: capacity 12 is below reuse distance 40, so all 40
second-cycle requests miss and all 40 are counted as evicted-before-reuse.

No insertion refusal occurred in any scored pattern. Every 68,313,600-byte entry fit individually;
the server took the normal admit-then-evict branch.

## Exactness and validation

- Existing `research/prefixmoney-20260812/prefix_gate.py`: **PASS**. Three independent namespaces,
  repeated-prompt byte identity 3/3, shared-prefix byte identity 3/3, 12 hits, six misses, six
  inserts, zero evictions. On the same capped run, repeated cold/hit TTFT medians were 159.892 ms
  (`N=3`) and 1.540 ms (`N=3`); shared learning/hit medians were 191.211 ms (`N=3`) and 51.097 ms
  (`N=3`). These are behavioral within-run comparisons only.
- Contention trace cache hits: **107/107 output-byte hashes matched** the same key's cold output.
- `cargo test -p memra-server prefix_cache -- --nocapture`: **PASS, 8/8**, 212 filtered out.
- Release build: **PASS**, sm_120a auto-detected.
- Harness validation: Python AST parse, essential Ruff error/F checks, and `bash -n` all passed.
- Scored campaign: all five phase exits were zero; final marker
  `EVICTCHURN_LOCAL_PASS 2026-08-12T22:12:15Z`; all hashes in `SHA256SUMS` verify.
- No engine/runtime code changed. Therefore the conditional engine-change battery
  (`kernel-check`, `run-gen`, `run-spec`, `serve-smoke`) was not triggered and was not run.

The first attempt under `raw/run-20260812T220712Z/` is deliberately retained as non-scored. Its
exact failure was `RuntimeError: stream completed without a visible text token`: the imported
exactness client rejected eight valid EOS-only outputs. The worker still completed all 80
requests. The harness was corrected only to retain the terminal event and label its timing basis,
then the complete workload was rerun from fresh servers. See that directory's `ATTEMPT.md`.

## Minimal policy follow-up

The smallest credible policy experiment is to transfer the expert bank's existing SLRU shape:
new entries enter a probation segment, a real hit promotes to protected MRU, protected LRU demotes
when its cap is exceeded, and eviction consumes probation LRU before protected LRU. The expert
implementation states this is specifically so “a one-off cold expert can never evict a genuinely
hot one” and implements probation-hit promotion plus probation-first victim selection
(`crates/memra-engine/src/moe_cache.rs:1-13,159-193`). Its current protected cap is 80% of slots
(`moe_cache.rs:522-538`).

For prefix entries the segment caps must be **bytes**, not slots, because entries vary with prefix
length and model geometry. Preserve the existing `(model, cache_salt)` visibility boundary, global
byte ceiling, exact-key dedupe, pin semantics, and emergency session-first flush. Start with an
80% protected / 20% probation byte split as an experiment, not a default. Replay this exact trace
against LRU and SLRU and require: hot reuse loss falls from 13/120 to zero or a predeclared bound;
cold scan still serves as misses; no refusal explosion; accounting stays under budget; and byte
identity plus the full GPU exactness battery remains green.

“Refuse a new entry if it would evict a more-recently-used entry” is not sufficient: a just-created
entry is necessarily most recent, so recency alone provides no admission signal. Probation is the
minimal mechanism that distinguishes a one-hit insertion from an entry that has demonstrated
reuse. A frequency sketch or more elaborate admission filter can be considered only if SLRU's
replay remains inadequate.

This recommendation also matches current external practice rather than relying on a stale policy
memory: current vLLM documents touch-on-hit and LRU eviction for reusable prefix blocks, while the
S3-FIFO paper's small FIFO queue is explicitly intended to filter one-hit objects before they can
pollute the main cache. Sources consulted 2026-08-13: vLLM Automatic Prefix Caching design,
<https://docs.vllm.ai/en/stable/design/prefix_caching/>; S3-FIFO SOSP 2023 paper,
<https://www.cs.cmu.edu/~rvinayak/papers/s3-fifo-sosp-2023-fifo-queues-are-all-you-need-for-cache-eviction.pdf>.

## Production posture, composed with `cx-cachesize`

1. **Do not treat a green one-prefix exactness gate as a working-set revenue guarantee.** Keep
   exactness green, but call the current partial-budget policy scan-vulnerable until an admission
   or protected-segment replay closes the 13/120 hot-loss result.
2. **Let `cx-cachesize` determine the byte budget, then apply this policy result.** If its measured
   budget holds the complete active tenant working set, ordinary LRU churn is dormant. If budget
   holds only a fraction, it must at least hold the measured hot bytes plus bounded probation
   headroom; merely raising a still-partial LRU budget moves the reuse-distance cliff without
   removing scan pollution.
3. **Keep stable tenant salts and sticky routing.** `cache_salt` prevents cross-tenant hits, not
   cross-tenant eviction: the budget is global. Monitor per-namespace working-set bytes and the
   process-wide hit, eviction, refusal, and evicted-before-reuse rates under production replay.
4. **Retain the current oversized-entry refusal.** An entry larger than the whole budget cannot be
   made resident by eviction and should continue to log a refusal. This is separate from admission
   of individually fitting one-hit entries.
5. **No production default change from this rig.** Implement an instrumented SLRU A/B lane, run
   the full 5090 correctness battery during development, and promote only after the designated
   2x RTX PRO 6000 pre-release battery. This local thermally capped run establishes behavior, not
   transferable throughput.

## Evidence map

- Scored provenance, budget, requests, metrics, server logs, 250 ms GPU samples, before/after GPU
  state, phase exits, and manifest: `raw/run-20260812T221018Z/`.
- Full release build receipt: `raw/build-server.log`.
- Targeted cache tests: `raw/host-prefix-cache-tests.log`.
- Harness/source checks: `raw/harness-python-ast.log`, `raw/harness-ruff-essential.log`, and
  `raw/harness-bash-n.log`.
- Retained non-scored failure: `raw/run-20260812T220712Z/`.

No board, merge, tag, push, or live serve box was touched.
