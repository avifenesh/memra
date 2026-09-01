# Prefix-cache extension feasibility: HiRadix and CacheBlend

Date: 2026-08-11
Lane: `lane/cx-prefixdoors`
Posture: read-only research; no implementation or performance claim

## Verdict

| Door | Verdict | Decisive reason |
| --- | --- | --- |
| HiRadix | **DOOR-OPEN**, narrowly: start with an exact whole-`PrefixEntry` GPU/CPU tier under the existing `PoolKey`; page-granular partial-node reuse remains conditional | HiRadix remains continuous-prefix reuse and can move the same cached state across tiers; memra already deep-copies every stored plane/state on snapshot and restore (`crates/memra-server/src/worker.rs:2217-2322`; [HR-DOC](#hr-doc); [HR-BLOG](#hr-blog)). |
| CacheBlend | **DOOR-CLOSED** | It reuses non-prefix chunks by selectively recomputing only part of their state, and the paper reports a nonzero quality delta from full recompute; that is an approximate distribution, not memra's lossless cache contract ([CB-LMCACHE](#cb-lmcache); [CB-PAPER](#cb-paper); `crates/memra-server/src/worker.rs:1135-1139`). |

The open HiRadix door is contingent on carrying the complete `(model, cache_ns)`
identity into every metadata lookup and storage object; a shared lower tier keyed
only by tokens or a content hash would be an isolation regression
(`crates/memra-server/src/worker.rs:786-794,1825-1830`).

## External source quote ledger

All external material below was refreshed on 2026-08-11. The SGLang and LMCache
links are pinned to the repository revisions inspected; the CacheBlend link is the
paper's latest arXiv revision.

<a id="hr-doc"></a>**HR-DOC** — [SGLang, *HiCache System Design and
Optimization*, revision `546965fc`](https://github.com/sgl-project/sglang/blob/546965fc72a73b147323fc839ad0aa955ef64aff/docs/docs/advanced_features/hicache_design.mdx):
“distributed storage as L3”; “the node is automatically split to create an exact
boundary”; “returns a continuous prefix of the request.”

<a id="hr-blog"></a>**HR-BLOG** — [SGLang, *SGLang HiCache: Fast Hierarchical KV
Caching with Your Favorite Storage Backends*](https://www.lmsys.org/blog/2025-09-10-sglang-hicache/):
“acts as a page table for referencing KV caches residing locally in GPU and CPU
memory”; backend operations are “`get(key)`, `exist(key)`, `set(key, value)`.”

<a id="hr-code"></a>**HR-CODE** — [SGLang `RadixKey`, revision
`546965fc`](https://github.com/sgl-project/sglang/blob/546965fc72a73b147323fc839ad0aa955ef64aff/python/sglang/srt/mem_cache/radix_cache.py):
“extra key (e.g. lora_id, cache_salt)” and “namespaced by `extra_key`.”

<a id="cb-paper"></a>**CB-PAPER** — [Yao et al., *CacheBlend*, arXiv
2405.16444v3](https://arxiv.org/html/2405.16444v3): non-prefix caches “ignore the
text's cross-attention with the preceding texts”; CacheBlend can only “partially
update each reused KV cache”; evaluation reports “0.01-0.03 quality drop.”

<a id="cb-lmcache"></a>**CB-LMCACHE** — [LMCache, *CacheBlend*, revision
`a976ce09`](https://github.com/LMCache/LMCache/blob/a976ce09dd9896edde79f3e81a820a4d3a9ce03f/docs/source/kv_cache_optimizations/cacheblend.rst):
“reuse the KV cache of any repeated text chunk -- not only a shared prefix -- by
selectively recomputing a small fraction.”

## The shipped delta surface

This is the minimum baseline needed to judge the two doors; it does not repeat the
existing survey's general prefix-cache description.

- **Isolation key.** Every cross-request reuse pool uses `PoolKey = (String,
  String)`, meaning `(model, cache namespace)`, and `PrefixCache.entries` is keyed
  by that tuple (`crates/memra-server/src/worker.rs:786-794,1825-1830`). With a
  keyring, the HTTP layer derives `t:<tenant>\x1f<cache_salt>`; tenant ids cannot
  forge the separator (`crates/memra-server/src/auth.rs:16-20,456-462`;
  `crates/memra-server/src/main.rs:2835-2848`).
- **Match shape.** A hit is the longest stored entry whose complete token vector
  exactly prefixes the request; `best_lcp` separately computes the maximum token
  LCP against entries in the same `PoolKey`
  (`crates/memra-server/src/worker.rs:1973-1995`). A miss can arm a boundary
  snapshot at that LCP, but the current request still computes through the
  boundary before inserting it
  (`crates/memra-server/src/worker.rs:5232-5244,6208-6213`).
- **Storage shape, not a page table.** `PrefixEntry` owns the token vector, one
  optional K/V plane per layer, recurrent conv/SSM state, boundary position and
  logits (`crates/memra-server/src/worker.rs:1793-1816`). The underlying
  full-attention `KvLayer` is a pair of context-linear GPU `CudaSlice<u8>` planes
  with per-token quantized-block strides; it is not a page-id/block-table
  allocator (`crates/memra-kv/src/lib.rs:213-245,265-283`). Snapshot and restore
  copy the whole retained prefix plane-by-plane
  (`crates/memra-server/src/worker.rs:2217-2322`).
- **Eligibility boundary.** Recurrent state cannot be truncated to an arbitrary
  shorter prefix, which is why memra captures state at explicit boundaries
  (`crates/memra-server/src/worker.rs:1126-1133`). SWA-ring caches reject flat
  prefix snapshots/restores and are excluded at admission
  (`crates/memra-server/src/worker.rs:2226-2228,2282-2284,5137-5144`).
- **Eviction and observability.** Visibility is per `PoolKey`, while the byte
  budget and timestamp-LRU victim order are global across namespaces
  (`crates/memra-server/src/worker.rs:2103-2113,2182-2189`). `/metrics` publishes
  hits, misses, inserts, evictions, hit tokens, token-weighted hit ratio and the LCP
  histogram (`crates/memra-server/src/main.rs:1930-1979`); entry/byte capacity
  gauges are operator-only (`crates/memra-server/src/main.rs:1981-1999`).

## Door 1: HiRadix

### What is actually new

Here **HiRadix** means SGLang HiCache's `HiRadixTree` metadata/control-plane
component, not a non-prefix cache algorithm: it is a page table over prefix spans
resident in GPU, host memory, or an external tier ([HR-DOC](#hr-doc);
[HR-BLOG](#hr-blog)). Its result remains one continuous request prefix, so it does
not make a middle chunk reusable under a different preceding context
([HR-DOC](#hr-doc)).

It adds two reuse cases over memra's shipped flat entry pool:

1. **Lower-tier survival.** An exact prefix that is no longer GPU-resident can
   remain addressable in host/external storage and be restored as the same prefix
   state ([HR-DOC](#hr-doc); [HR-BLOG](#hr-blog)). Memra currently removes the
   victim entry and its owned device state when global LRU eviction fires
   (`crates/memra-server/src/worker.rs:2080-2101,2182-2189`).
2. **Immediate partial-node reuse.** HiRadix can split a cached token-span node at
   the exact point where a request diverges ([HR-DOC](#hr-doc)). Memra can measure
   that same LCP, but `lookup` cannot restore it unless an entry already ends
   there; a miss first computes to the boundary and captures it for subsequent
   traffic (`crates/memra-server/src/worker.rs:1973-1995,5232-5244,6208-6213`).

The second case is not expressible for every current memra cache shape. A longer
hybrid entry contains conv/SSM state only at its endpoint, and memra explicitly
forbids truncating that state to an arbitrary shorter prefix
(`crates/memra-server/src/worker.rs:1126-1133,1800-1808`). Therefore immediate
partial-node reuse is **N/A for hybrid entries with the current format** unless
the exact recurrent state is also captured at each exposed split boundary; whole
existing entry boundaries remain expressible
(`crates/memra-server/src/worker.rs:2217-2271`).

### Losslessness

Tiering an existing whole `PrefixEntry` can preserve the shipped lossless
contract: the lower tier must store every K/V byte, recurrent-state byte,
`pos`, and boundary logits, and restoration must reproduce the same fields
(`crates/memra-server/src/worker.rs:1793-1816,2217-2322`). This changes residency,
not the cached state consumed by decode; memra's existing contract defines a hit
as bit-identical to the prime configuration that produced the snapshot
(`crates/memra-server/src/worker.rs:1135-1139`).

Page-granular transformer-only slicing can also be lossless in principle because
the full-attention planes are position-addressed, context-linear token rows
(`crates/memra-kv/src/lib.rs:213-245`). It is nevertheless a new storage ABI:
`PrefixEntry` has no page identifiers, host representation, or partial restore
interface today (`crates/memra-server/src/worker.rs:1793-1816,2217-2322`). Full
HiRadix page-table semantics are therefore
**conditional**, while whole-entry L1/L2 tiering is the admissible first arm.

SWA-ring sessions are **N/A** for either arm until memra defines an exact ring
snapshot/restore contract; the current code rejects them
(`crates/memra-server/src/worker.rs:2226-2228,2282-2284,5137-5144`).

### Tenant isolation

The tree and every backing-store operation must take the complete memra
`PoolKey`; token equality or a content hash is not an authorization boundary
(`crates/memra-server/src/worker.rs:786-794,1825-1830`). SGLang's radix key has an
`extra_key` namespace dimension that explicitly includes `cache_salt`, so the
metadata shape can carry this boundary ([HR-CODE](#hr-code)). For memra, that
dimension must be the already-derived `cache_ns`, including its unforgeable
tenant prefix when a keyring is active
(`crates/memra-server/src/auth.rs:16-20,456-462`).

Sharing the capacity/LRU controller across tenants is admissible only while
lookup visibility remains per `PoolKey`, matching the shipped global-budget,
namespaced-visibility rule (`crates/memra-server/src/worker.rs:2103-2105`). A
host or external object addressed without `(model, cache_ns)` would let one
namespace load state created by another and is a security regression
(`crates/memra-server/src/worker.rs:1973-1977,8243-8262`).

A durable or cross-process tier also needs a format/version discriminator in
addition to `PoolKey`: the current key contains only model text plus namespace,
while K/V encodings are selected independently by runtime configuration
(`crates/memra-server/src/worker.rs:786-794`; `crates/memra-kv/src/lib.rs:13-41`).
The safe object identity is therefore at least `(model artifact/revision,
cache_ns, KV layout/version, exact prefix/page key)`; an identity mismatch must
fail closed rather than return bytes. The current restore path's explicit
layer/count/kind bounds are the minimum validation pattern, not proof that those
new identity fields already exist
(`crates/memra-server/src/worker.rs:2285-2319`).

### Gate and verdict

**Verdict: DOOR-OPEN** for a default-off, whole-entry GPU/CPU hierarchy. Name the
gate **`HIRADIX-EXACT-ISO`**. It must establish all of the following before a
page-table or external-store extension is considered
(`crates/memra-server/src/worker.rs:1135-1139,1793-1816,2217-2322`;
[HR-DOC](#hr-doc); [HR-BLOG](#hr-blog)):

1. Snapshot -> host encode -> host decode -> restore reproduces every K/V plane,
   recurrent plane, `pos`, and boundary-logit bit, then a warm request matches the
   same-prime cold request's logits under the shipped exactness class
   (`crates/memra-server/src/worker.rs:1135-1139,1793-1816,2217-2322`).
2. Same model/prefix in namespace A hits through L1 and L2; namespaces B and the
   default namespace miss in both directions, including after eviction/reload
   (`crates/memra-server/src/worker.rs:8243-8262`).
3. A new lower-tier decoder rejects corrupt, truncated, wrong-model,
   wrong-KV-layout and wrong-version objects before entering the existing
   bounded/kind-checked device-copy path
   (`crates/memra-server/src/worker.rs:2285-2319`).
4. The initial support matrix says **N/A** for SWA ring and for mid-entry hybrid
   splits rather than silently falling back to an unsafe truncation
   (`crates/memra-server/src/worker.rs:1126-1133,2226-2228,2282-2284`).

The later page-granular arm may open only after it passes the same gate at every
split boundary; it must not borrow correctness from the whole-entry result
(`crates/memra-server/src/worker.rs:1126-1139`).

## Door 2: CacheBlend

### What it adds

CacheBlend addresses a different reuse class: independently cached text chunks
can be reused when they occur after other chunks, and only a selected fraction of
tokens is recomputed at chunk boundaries ([CB-LMCACHE](#cb-lmcache)). This is
strictly beyond memra's current matcher, which admits only a stored token vector
that is an exact prefix of the request
(`crates/memra-server/src/worker.rs:1973-1988`).

The need for repair is causal: a chunk's precomputed K/V did not attend to the
new preceding chunks, so direct transplantation omits cross-chunk attention
([CB-PAPER](#cb-paper)). CacheBlend partially updates the reused cache rather than
recomputing every affected token ([CB-PAPER](#cb-paper);
[CB-LMCACHE](#cb-lmcache)).

### Losslessness is the closing condition

Selective repair is an approximation to full prefill, not an identity transform.
The paper calls the update partial and reports a nonzero quality drop against
full recompute ([CB-PAPER](#cb-paper)). Consequently it supplies empirical
task-quality proximity, not equality of the K/V tensors, logits, or next-token
distribution required by memra's cache contract
(`crates/memra-server/src/worker.rs:1135-1139`).

Recomputing 100% of context-dependent state would be cold full prefill rather
than the method's defining “small fraction” selective recomputation
([CB-LMCACHE](#cb-lmcache)); the cited sub-100% method supplies empirical quality
results, not a general lossless equality proof ([CB-PAPER](#cb-paper)).

### Format and isolation

CacheBlend is **N/A in the current `PrefixEntry` format**: the format indexes one
complete token prefix and one boundary state, with no independently addressable
chunks and no layer-wise selected-token recompute interface
(`crates/memra-server/src/worker.rs:1793-1816,2217-2322`). It is additionally
**N/A for memra's recurrent layers**: the current format exposes stored boundary
state, not an operation to reconstruct conv/SSM state from an arbitrary chunk,
and it explicitly rejects arbitrary shorter-prefix truncation
(`crates/memra-server/src/worker.rs:1126-1133,1800-1808`).

PoolKey isolation could scope a hypothetical chunk store, but the shipped
isolation does not automatically protect a new map or external index: every
chunk lookup, selective-recompute input, and stored object would have to use the
same `(model, cache_ns)` boundary as `PrefixCache.entries`
(`crates/memra-server/src/worker.rs:786-794,1825-1830`). Reusing or reconstructing
a chunk across two cache namespaces would violate the existing both-direction
isolation contract (`crates/memra-server/src/worker.rs:8243-8262`). Isolation is
therefore implementable in principle, but it cannot rescue the failed lossless
condition.

### Verdict

**Verdict: DOOR-CLOSED.** CacheBlend's defining sub-full recomputation is
approximate across blend boundaries ([CB-PAPER](#cb-paper);
[CB-LMCACHE](#cb-lmcache)). There is no gated experiment in memra's lossless
serving lane: reopening would require a method that proves identical target
logits while retaining non-prefix reuse, not a task-score tolerance
(`crates/memra-server/src/worker.rs:1135-1139`). Any future approximate-reuse
study belongs outside this shipped PrefixCache contract and must still carry
`PoolKey` through every cache tier
(`crates/memra-server/src/worker.rs:786-794,1825-1830`).

## Orchestrator handoff

- The only admissible next implementation lane is the **whole-entry**
  `HIRADIX-EXACT-ISO` experiment; it should not claim upstream page-granular
  partial-node coverage
  (`crates/memra-server/src/worker.rs:1793-1816,2217-2322`).
- Keep page-granular transformer splitting conditional, and mark hybrid mid-entry
  splits plus SWA ring N/A until their exact state formats exist
  (`crates/memra-server/src/worker.rs:1126-1133,2226-2228,2282-2284`).
- Do not open a CacheBlend implementation lane under the current lossless policy
  ([CB-PAPER](#cb-paper); `crates/memra-server/src/worker.rs:1135-1139`).
