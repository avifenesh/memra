# SLRU acceptance target from serving-shaped traffic

Date: 2026-08-13

Branch/base: `lane/cx-slrutarget` from `v0.82.0`
(`7624b4f5fd914f4909056ae794f5229fbd14b21b`)

Traffic-model lock: `traffic_model.lock.json`

Raw policy replay: `raw/simulation.jsonl` (`N=26,332` JSONL rows)

Reduced result: `analysis.json`

## Decision: NEEDS-REAL-TRAFFIC

**Do not endorse SLRU as the default from synthetic traffic alone.** The old two-cycle 40-key
round-robin target is rejected: it is not a defensible router-traffic model, and both LRU and a
strict hit-triggered SLRU must score 0/40 when 40 entries cycle at a 12-entry capacity. The new
primary estimate is an eight-session Zipf hot set interleaved with unique one-hit traffic. Under
that estimate, SLRU is better at the production 4,096 MiB budget and equal when the modeled
working set fits. It provides exactly the protection the mechanism claims: zero misses after a hot
entry's first successful hit in every primary arm.

That is not enough to make a default decision because there is no external traffic today, the
cache-size lane had not committed its N=5 recommendation when this lane started, and a real losing
shape exists. If an old promoted cohort goes idle and a disjoint new cohort cycles just beyond the
remaining probation capacity, LRU learns the new cohort while SLRU retains the stale protected
cohort. In the controlled four-cycle cases here, LRU hits 75% and SLRU hits 0% at both evaluated
budgets and for Q27-only, Q35-only, and worker-global paired entries.

There is also a configuration fact that matters to the recommendation: v0.82.0 already runs SLRU
whenever the prefix cache is enabled. `MEMRA_PREFIX_CACHE_PROTECTED_PCT` accepts only 1..99 and
defaults or falls back to 80; it changes the protected share, not the policy. Setting
`MEMRA_PREFIX_CACHE_MB=0` disables the whole cache. There is no plain-LRU rollback seam. Thus the
merge description's “behind its knob” is not an LRU/SLRU selection boundary. Before treating SLRU
as an accepted production default, add an explicit `lru|slru` policy seam and collect the live
trace described below. This lane deliberately changes neither.

## Evidence read first

The shipped work was used as evidence and was not rerun:

- [`../slrucache-20260813/RESULTS.md`](../slrucache-20260813/RESULTS.md) and
  [`PROGRESS.md`](../slrucache-20260813/PROGRESS.md) establish the exact v0.82.0 byte-SLRU
  semantics and the 782 MiB campaign: hot reuse 107/120 -> 115/120, broad thrash 13 -> 5,
  evictions 41 -> 33, 115/115 hit outputs byte-identical, zero post-promotion hot misses, and
  unchanged round-robin and sequential controls.
- [`../evictchurn-20260813/RESULTS.md`](../evictchurn-20260813/RESULTS.md), its progress ledger,
  frozen Python driver, and local runner establish the timestamp-LRU defect and the exact request
  schedules reproduced by this lane's simulator self-check.
- `research/cachesize-20260813/` was absent from the frozen v0.82.0 tree. At lane start, the clean
  sibling worktree had commit `f5657142ee586cb1a6ea23c857265d9e597a67e6`. Its `RESULTS.md`
  explicitly said the capacity curve, active concurrency, and production recommendation were
  pending the completed N=5 reduction. This lane therefore uses its committed exact entry-byte
  receipts and budget arithmetic, but does not invent a “recommended budget.”
- [`../coldfix-20260812/RESULTS.md`](../coldfix-20260812/RESULTS.md) supplies the existing N=5
  sold-shape latency and throughput evidence. It is not relabeled as a cache-budget capacity run.

The simulator first reproduced the committed policy summaries exactly before emitting any study
row: round-robin 0 hits / 68 evictions in both arms; hot-set LRU 107 hits / 13 broad thrash / 41
evictions versus SLRU 115 / 5 / 33 with zero post-first-hit misses; and sequential scan 0 hits / 28
evictions in both arms. `analysis.json.validation.verdict` is `PASS`.

## What “realistic” means here

It does **not** mean that this lane observed production traffic. It means the estimate is tied to
measured entry sizes and a stated request mechanism, and every unobserved parameter is labeled as
an estimate.

### Measured storage shape

The cache-size commit measured the exact 4,860-token sold prompt:

| Entry | Bytes | MiB |
|---|---:|---:|
| Q27 | 301,215,744 | 287.2617 |
| Q35 | 110,964,480 | 105.8240 |
| One Q27 + Q35 logical-session pair | 412,180,224 | 393.0857 |

The worker has one global byte budget across models. Exact floor capacities are:

| Budget | Q27 entries | Q35 entries | Q27+Q35 pairs | 80%-protected Q27 / Q35 / pairs |
|---:|---:|---:|---:|---:|
| 4,096 MiB (production) | 14 | 38 | 10 | 11 / 30 / 8 |
| 49,152 MiB (highest tested cache-size arm) | 171 | 464 | 125 | 136 / 371 / 100 |

The second row is **not** called the cache-size recommendation. No recommendation existed in the
commit frozen at start. It is reported because it was the highest tested arm and it can hold the
cache-size protocol's full 96-session paired working set.

### Frozen primary estimate

Before simulation, `traffic_model.lock.json` fixed:

- 1,000 requests per trace and 30 deterministic seeds (3407..3436);
- eight returning logical sessions, an estimate chosen as two waves of the sold c=4 shape;
- 90% returning requests and 10% unique one-hit requests, preserving the committed 9:1
  working/cold role mix;
- Zipf alpha 1.0 over the returning sessions;
- Q27-only, Q35-only, and a conservative worker-global paired variant. A paired logical session
  retains one entry for each model, while requests split equally across the two entries;
- exact byte admission and eviction at 4,096 and 49,152 MiB under the v0.82.0 policy semantics.

The eight-session count, Zipf exponent, 90/10 mix, and equal model split are estimates, not live
observations. Each hot entry appears once before weighted sampling so that the finite trace does
not accidentally omit a declared session. The induced primary reuse distance is short rather than
round-robin: per-model p50 averages 4.27 intervening requests, p90 21.47, and p99 59.57; the paired
variant averages 8.67 / 42.93 / 120.73 because each logical session owns two independent model
entries. Unique scan keys never return.

This mechanism is supported, but not numerically calibrated, by current primary studies:

- [TraceLab](https://arxiv.org/html/2606.30560v2) analyzes 4,265 real coding-agent sessions and
  357,161 LLM steps. It reports high prefix reuse within sessions, closely spaced tool-driven
  steps, and sharply lower reuse after human-scale idle gaps. That supports near-term returns plus
  later cohort expiry; it does not prove this lane's eight sessions, 90/10 split, or Zipf alpha.
- The [SGLang paper](https://papers.nips.cc/paper_files/paper/2024/file/724be4472168f31ba1c9ac630f15dec8-Paper-Conference.pdf)
  identifies multi-turn chats and agent programs as prefix-sharing workloads and reports
  production Chatbot Arena cache evidence. It supports session-local reuse rather than a uniform
  exact two-cycle scan.
- [Mooncake](https://arxiv.org/abs/2407.00079) reports temporal locality and highly skewed block
  reuse in a real LLM-serving trace. Again, it supports a hot/cold distribution, not the exact
  parameters selected here.

### Acceptance target

On the frozen primary estimate, SLRU must:

1. lose neither total nor returning-request hit rate versus plain LRU;
2. produce zero misses after a hot key's first successful hit and zero scan hits;
3. produce no refusal or byte-accounting overflow; and
4. report Q27, Q35, and the worker-global paired case separately at both budgets.

Commercial acceptance is a separate gate: the largest clean number of simultaneous cache-hit
sessions per card whose hit TTFT p95 remains at or below 22 ms for Q27 and 11 ms for Q35. A policy
replay cannot derive this number or dollars/day.

## Primary result

Each number below is the mean of 30 deterministic 1,000-request traces. Hit rate includes cold
first references and unique scan misses; returning hit rate excludes scans.

| Budget | Shape | LRU -> SLRU total hit rate | Delta | LRU -> SLRU returning hit rate | Delta | LRU -> SLRU post-first-hit misses |
|---:|---|---:|---:|---:|---:|---:|
| 4,096 MiB | Q27 | 88.170% -> 89.193% | **+1.023 pp** | 97.967% -> 99.104% | **+1.137 pp** | 10.23 -> **0** |
| 4,096 MiB | Q35 | 89.200% -> 89.200% | 0.000 pp | 99.111% -> 99.111% | 0.000 pp | 0 -> 0 |
| 4,096 MiB | paired | 83.493% -> 88.153% | **+4.660 pp** | 92.770% -> 97.948% | **+5.178 pp** | 46.97 -> **0** |
| 49,152 MiB | Q27 | 89.200% -> 89.200% | 0.000 pp | 99.111% -> 99.111% | 0.000 pp | 0 -> 0 |
| 49,152 MiB | Q35 | 89.200% -> 89.200% | 0.000 pp | 99.111% -> 99.111% | 0.000 pp | 0 -> 0 |
| 49,152 MiB | paired | 88.400% -> 88.400% | 0.000 pp | 98.222% -> 98.222% | 0.000 pp | 0 -> 0 |

At production size, the paired case is the policy-relevant pressure test: LRU evicts an average
48.53 entries that had demonstrated reuse and later incurs 46.97 post-first-hit misses. SLRU has
zero for both. Q27 also benefits because its larger entries leave less scan slack. Q35's eight-key
hot set plus the induced reuse distances fit the 38-entry budget under both policies, so SLRU has
nothing to improve. At 49,152 MiB every primary working set fits and the policies are equal.

The acceptance checks all pass: no primary hit-rate loss, zero SLRU post-first-hit misses, zero
scan hits, zero refusals, and no budget overflow. This is a model result, not a live default gate.

## Sensitivity and the workload where SLRU loses

The stationary grid spans 4/8/16/32/64/96 logical sessions, Zipf alpha 0.8/1.0/1.2, scan shares
0/10/25/50%, all three model variants, both budgets, and 30 seeds. Of 432 aggregated scenarios,
SLRU is better in 239, equal in 193, and worse in zero. The largest mean gain is +15.943 pp total
and +31.887 pp returning-hit rate: paired entries, 4,096 MiB, eight sessions, 50% unique scan,
alpha 0.8. These are robustness cells, not alternate targets selected after seeing the result.

Stationary exact cycles are also symmetric. At or below the byte-floor capacity, both policies
miss the first cycle and hit the next two (66.667% total); one logical session beyond capacity,
both policies hit 0%. This is why the old 40-at-capacity-12 two-cycle target says nothing about
which policy is better.

SLRU **is worse** under phased hot-set turnover:

| Budget | Shape | Idle demonstrated-reuse cohort | New cyclic cohort | LRU -> SLRU hit rate |
|---:|---|---:|---:|---:|
| 4,096 MiB | Q27 | 11 | 4 | 75% -> **0%** |
| 4,096 MiB | Q35 | 30 | 9 | 75% -> **0%** |
| 4,096 MiB | paired sessions | 8 | 3 | 75% -> **0%** |
| 49,152 MiB | Q27 | 136 | 36 | 75% -> **0%** |
| 49,152 MiB | Q35 | 371 | 94 | 75% -> **0%** |
| 49,152 MiB | paired sessions | 100 | 26 | 75% -> **0%** |

The setup first references every old entry twice so it genuinely earns protected residency. The
old cohort then goes completely idle. A disjoint new cohort, exactly one logical session beyond
the residual byte-floor capacity, cycles four times. LRU evicts stale old entries, misses the first
new cycle, then hits three cycles. SLRU cannot promote any new entry before probation evicts it, so
no operation demotes the stale protected cohort and all four cycles miss.

This is a deliberately sharp boundary control, not a claim about the frequency of real cohort
turnover. But it is a valid cyclic workload and directly answers whether SLRU can be worse. Live
reuse-age and cohort-turnover evidence are therefore required before choosing the policy globally.

## Commercial capacity: what is known and what is not

The number that matters is simultaneous cache-hit sessions per card inside the sold hit-latency
envelope, followed by completed requests or tokens per day. The current evidence bounds but does
not finish that answer:

- The production 4,096 MiB budget physically holds 14 Q27 entries, 38 Q35 entries, or 10 paired
  logical sessions. SLRU's 80% protected target holds 11 / 30 / 8. These are residency counts, not
  a serving-concurrency claim.
- The prior N=5 target-rig campaign proves a lower bound at sold c=4: Q27 mixed-hit TTFT p95 is
  19.820 ms (within its 21.565 ms sold envelope and the cache-size lane's 22 ms gate); Q35 is
  10.260 ms (within the 11 ms gate). Its clean mixed-throughput knees are c=16 and c=40,
  respectively. Those knees do not prove that every distinct session is a resident cache hit at
  each cache budget.
- The cache-size commit frozen at lane start explicitly leaves “Active concurrency at the sold
  hit-latency envelope” and “Production recommendation” pending. Therefore there is no honest
  recommended-budget comparison or $/day number to publish here. The 49,152 MiB row above is only
  the highest tested arm.

Once the cache-size N=5 reduction lands, substitute its actual recommended budget into the same
policy replay. Dollars/day still additionally needs the observed request/output-token mix,
completion rate, price receipt, and uptime; pricing decisions belong in the product repository.

## Exact live log needed to settle the default

Collect a privacy-safe event stream on the serving worker. Do not log prompts or token text.

For every cache-eligible request, log:

1. monotonic request sequence, wall-clock timestamp, worker/card/replica, model and artifact hash,
   runtime commit, cache budget, protected percentage, context, and policy;
2. a stable keyed HMAC of the exact cache identity (model, namespace/cache salt, token ids through
   the cached boundary, and layout version), plus prefix tokens and exact entry bytes;
3. hit/miss, cached tokens, insert/skip/refusal, and segment before/after;
4. promotion, demotion, and eviction events with victim HMAC, bytes, last request sequence/time,
   last successful hit sequence/time, segment, trigger HMAC/class, and pin/lease count;
5. a bounded ghost record for an evicted identity so a later request records reuse distance in
   intervening requests, intervening admitted bytes, and wall-clock time, including whether the
   entry had ever hit before eviction;
6. emergency flushes, session-allocation reclaim, oversized admission skips, dedup/fanout, route
   changes, and worker misses, so routing loss is not mislabeled as eviction-policy loss;
7. active and queued sessions, admission/VRAM defers, OOM parks, hit/miss TTFT, output length,
   completion status, and output tokens/s.

The HMAC key must remain private and rotate by analysis epoch. The event schema should make the
same request/byte/pin sequence replayable offline under both byte-LRU and byte-SLRU at 4,096 MiB
and at the cache-size lane's eventual recommended budget. This mirrors the prefix-fingerprint
approach used for cache-aware routing in current [NVIDIA Dynamo documentation](https://docs.nvidia.com/dynamo/latest/user-guides/kv-cache-aware-routing)
while preserving memra's tenant namespace boundary; current [vLLM prefix-cache documentation](https://docs.vllm.ai/en/stable/design/prefix_caching/)
also reinforces that exact hashed prefix identity and routing are part of observed cache behavior.

Predeclare a live decision window rather than stopping at a favorable hour: as an initial estimate,
require both seven consecutive days and 10,000 cache-eligible requests, at least 1,000 natural
reuse opportunities, and coverage of three peak windows. Reduce the whole stream and each
day/peak/cohort boundary separately. A future SLRU-default recommendation requires:

- nonnegative shadow-replay total and returning hit-rate deltas at both budgets, with the lower
  endpoint of a day-block bootstrap interval at least zero;
- zero scan-triggered misses after prior successful reuse, or an explicitly reviewed exception;
- no day, peak window, or cohort turnover bucket with a material SLRU loss;
- a target-rig canary proving the cache-size lane's maximum simultaneous hit sessions stay within
  Q27 22 ms / Q35 11 ms p95 and retain output exactness; and
- an explicit plain-LRU rollback seam tested on the same trace and canary.

The seven-day/10,000-request thresholds are proposed estimates, not facts learned from traffic.
If volume is too low to reach them, the decision remains `NEEDS-REAL-TRAFFIC` rather than treating
absence of misses as proof.

## Evidence integrity and lane stop

- The deterministic run used only CPU policy analysis. No local GPU, box1 GPU, or live serve box
  was touched. No absolute-throughput claim was produced.
- No engine, generated board, README, or product file changed, so no GPU exactness battery or perf
  regeneration was triggered.
- `raw/provenance.log` freezes the branch/head, v0.82.0 base, cache-size source commit, and script
  hashes. `raw/simulation.jsonl` was teed before reduction. `raw/SHA256SUMS` verifies every raw
  artifact.
- `analysis.json` records all primary, sensitivity, stationary-cycle, and turnover reductions.
- This lane stops after its commits. It does not merge, tag, push, bypass a hook, or flip a default.
