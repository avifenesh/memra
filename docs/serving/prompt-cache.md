# Prompt caching

How `memra-server` reuses prompt state across requests: the cache tiers, the budget and its
per-device defaults, eviction and reclaim, tenant isolation, `cached_tokens`, session affinity,
the host tier and speculation on cache hits. It is for an operator sizing a deployment or reading
its cache counters. The short form is in [SERVING.md](../SERVING.md); flag cells are in
[FLAGS.md](../FLAGS.md); counter definitions are in [metrics.md](metrics.md); the exactness law
behind capture boundaries is in [exactness.md](exactness.md).

## Tiers

| Tier | What it holds | Hit rule | Switch |
|---|---|---|---|
| Prefix cache | Snapshots of primed state at token boundaries: KV and recurrent state, plus the MTP draft plane, boundary hidden state and logits when a spec session published it | The prompt starts with the entry's exact token ids; the whole entry restores | `MEMRA_PREFIX_CACHE_MB` (`0` off) |
| Continuation pool | A retired plain session's whole state, prompt and generation | The new prompt exactly extends it; single use | `MEMRA_KV_REUSE` (`0` off) |
| MTP spec pool | Parked speculative sessions | Exact token extension, text extension or session affinity, with an equivalent sampler | `MEMRA_KV_REUSE` |
| DSpark pool | Parked DSpark (DFlash2) sessions, rewound to their own grid capture | Exact token extension only | `MEMRA_KV_REUSE` |
| Host tier | Prefix entries demoted from the device cache | A device miss that exactly matches a host entry | `MEMRA_KV_HOST_MB` (`0` off, the default) |

Session affinity is a third probe over the continuation and spec pools (see
[Session affinity](#session-affinity)). Each whole-session pool keeps `MEMRA_REUSE_POOL` entries
(default 2) per model and cache namespace.

Requests that use no reuse tier: image inputs (pad runs match across different images), the
embeddings and rerank capture surfaces (a hit would skip the forward they read), choices after
the first of an `n > 1` request, and every request while `MEMRA_CONFIDENCE_TRACE` is set.
Sessions on the Step-3.7 sliding-window ring (`MEMRA_SWA_RING`, armed by default for that family)
and GLM-5 latent-plane models without `MEMRA_PREFIX_LATENT=1` take no prefix snapshot. A PP host
bounce sets the prefix budget to 0 at runtime.

## How the prefix cache learns

Entries are keyed by the exact token-id prefix within one `(model, cache namespace)`. A restore
deep-copies the entry into the new session, so one entry serves any number of sessions. Prefixes
under 64 tokens are not cached.

- **Seed.** A cold session publishes its prompt at prefill-done, cut at the capture boundary
  (see [Accounting](#accounting-cached_tokens-and-capture-length)).
- **LCP split.** A cold miss that shares 64 or more tokens with an existing entry splits its own
  prime at the longest common prefix, snapshots there, then continues. For a shared system
  prompt: request 1 seeds, request 2 publishes the shared boundary, request 3 and later hit.
- **First message.** A plain chat request whose LCP taught nothing also publishes at the end of
  its first message.
- **Same-window fanout.** Cold requests admitted in the same window with an exact common prefix
  inside one `(model, cache namespace)` prime it once. Each sibling copies the snapshot, gets
  exact `cached_tokens` credit and holds one lease. Requests from different tenants or salts are
  never compared. Receipt on 2x RTX PRO 6000 Blackwell Server Edition, PP-2, Step-3.7-Flash
  (2026-08-08): eight requests sharing 1,024 tokens, same-burst TTFT p50 3.852 s against 22.263 s
  for eight cold primes ([`research/prefixdedup-20260808/`](../../research/prefixdedup-20260808/)).
- **Restores are whole-entry only.** A partial restore of a longer entry fed its suffix through
  another numeric program and was deleted
  ([`research/lcprestore-20260813/`](../../research/lcprestore-20260813/)).

An entry stores the bytes of whichever prime ran (single, chunked or concat batch prime). Decode
from those bytes is deterministic, so a hit is bit-identical to the run that computed the prefix:
16/16 partial-prefix and 16/16 full-prefix cached-versus-fresh greedy cells
([`research/prompt-cache-20260802/`](../../research/prompt-cache-20260802/)). A hit compared with a
different prime configuration's fresh stream carries the batched-prime near-tie class
([exactness.md](exactness.md#first-token-cross-config-drift-batched-prime)).

## Budget

One worker-global byte budget covers every model and namespace. A valid `MEMRA_PREFIX_CACHE_MB`
is authoritative and unclamped, `0` included. A present but invalid value falls back to 256 MiB.
Unset, the budget is derived at boot:

- **Floor:** two entries at the served context of the largest loaded model (served context:
  `MEMRA_CTX`, else the checkpoint's declared context). Entry bytes are trunk full-attention K+V
  bytes per token times the context, plus the recurrent state; MTP and NextN blocks are excluded.
- **Clamp:** boot free memory after load, less the 1.5 GiB serving-transient reserve.
- **Request:** the larger of the floor and the device term below, then clamped.

| Device | Device term | Example |
|---|---|---|
| Discrete card | Boot free less the reserve, `MEMRA_ADMIT_DRIVER_HEADROOM_MB` (2 GiB) and the session peak: every lane's resident-session cap times one served-context session | One RTX PRO 6000 Max-Q, Qwen3.8-27B at 32,768 tokens, `MEMRA_MAX_SESSIONS=12`: 54,277,046,272 B. The default caps and the 24 GB RTX 5090 Laptop stay on the floor |
| Unified memory (GB10) | Half of (allocatable free less the reserve, the driver headroom and the host tier) | 34,048 MiB at 70 GiB free ([dgx-spark.md](../rigs/dgx-spark.md)) |

The session peak is an estimate, not a bound. Sessions still win memory: the admission reclaim,
the evict-and-retry on a failed session allocation, the post-insert reserve yield before decode
and the step-OOM park all take cache bytes first. The boot line states every term:
`[prefix-cache] on: budget ...MB (...), policy plain-LRU ...`. Records:
[PREFIX-CACHE-DISCRETE.md](../decisions/PREFIX-CACHE-DISCRETE.md),
[C8-SERVING-24GB.md](../decisions/C8-SERVING-24GB.md) (the reserve yield).

Entry size is linear in prefix length. Measured with `prefix_cache_bytes` on one RTX PRO 6000
Blackwell Server Edition, v0.81.2, 2026-08-13
([`research/cachesize-20260813/`](../../research/cachesize-20260813/)):

| Model | Fixed bytes | Bytes per token | 4,096 tokens | 8,192 tokens |
|---|---:|---:|---:|---:|
| Qwen3.6-27B NVFP4 + MTP | 156,893,184 | 29,696 | 265.6 MiB | 381.6 MiB |
| Qwen3.6-35B-A3B IQ4_XS | 65,863,680 | 9,280 | 99.1 MiB | 135.3 MiB |

The fixed part is the recurrent state at the entry's end, so many short entries cost more than
their token count suggests.

## Eviction (plain global LRU) and reclaim

Capacity pressure evicts the oldest unleased entry first, whatever its history. A hit refreshes
recency. Every unleased byte is reclaimable for a publication that fits the budget, so the newest
turn of a growing conversation always fits beside other tenants' entries: an entry is never its
own victim. A publication is refused in two cases, each printed in bytes:

```text
[prefix-cache] insert refused: entry N exceeds budget M (...)
[prefix-cache] insert refused: entry N cannot fit beside L leased bytes (budget M, ...)
```

`/metrics` counts the first as `prefix_cache_skips_budget` (the first one also logs a warning) and
the second as `prefix_cache_skips_pinned`. A fanout snapshot that cannot fit beside the current
leases is not kept; its participants continue from their own copies. The segmented policy and its
two flags are gone ([PREFIX-CACHE-POLICY.md](../decisions/PREFIX-CACHE-POLICY.md)).

When an arrival does not fit in memory, the admission reclaim frees cache bytes before it queues
the arrival. The incoming prompt's own entry and other sessions' leases stay protected.

| Where | What the reclaim frees |
|---|---|
| Discrete card, derived budget above its two-entry floor, single primary reading | Only the arrival's shortfall, the entry it will publish and what the priming sessions still owe, oldest first ([`research/partial-reclaim-20261109/`](../../research/partial-reclaim-20261109/)) |
| Full-power RTX PRO 6000 Workstation Edition, one card (memory-shaped admission armed) | Every unleased entry, demoting into the host tier up to the arrival's shortfall first; with the host tier off they all drop ([ADMIT-BY-MEMORY-DEFAULT.md](../decisions/ADMIT-BY-MEMORY-DEFAULT.md)) |
| Everywhere else (unified memory, the floor, an explicit budget, PP or multi-device readings) | Every unleased entry |

A failed session-cache allocation evicts every unleased entry and retries before it errors.

## Plain prefix-hit leases

Plain prefix-hit leases end after final carrier selection and the restore stream (root and TP
peers) completes, before the request publishes a deeper entry. The worker fences the restore,
releases the lease and logs `[prefix-cache] source lease released after restore fence`. If the
fence fails, the source stays leased and the request skips publication. Other consumers keep
their leases until their last participant retires: spec-conversion and drafter restores, and
fanout siblings. Snapshot preflight reclaims only eligible (unleased) entries before it allocates.
This is what lets the previous turn's entry be evicted when the next turn of the same
conversation publishes.

<a id="cache-salt-isolation"></a>

## Per-tenant isolation (`cache_salt`)

Every reuse tier keys on `(model, cache namespace)`. Requests share cached state only with
requests in the same namespace, in either direction, so `cached_tokens` can only reveal the
caller's own namespace's history. This is the cross-tenant hit-oracle mitigation in
[`research/cache-tools-20260802/REPORT.md`](../../research/cache-tools-20260802/REPORT.md).

- **Source.** The optional `cache_salt` body field on `/v1/completions` and
  `/v1/chat/completions`. No salt is the default `""` namespace. `/v1/responses` and
  `/v1/messages` carry no salt: their identity fields (`prompt_cache_key`, `metadata.user_id`)
  nominate session affinity and leave the namespace alone.
- **Validation.** At most 64 bytes of `A-Z a-z 0-9 - _ . + / =`. The `t:` prefix is reserved
  when no keyring is configured. A bad salt is a 400.
- **Keyring scope.** With `MEMRA_API_KEYS` set, the namespace becomes `t:<tenant>\x1f<salt>`, so
  two tenants never share entries even with the same salt.
- **Budget.** The prefix-cache byte budget stays global across namespaces. The whole-session
  pools are bounded process-wide by `MEMRA_REUSE_POOL_GLOBAL_CAP` (default 16) across every model,
  namespace and pool; a park past it evicts the globally oldest entry. On 2x RTX PRO 6000 WS
  Max-Q (2026-08-10) the two-namespace Qwen3.6-27B shape held 16 spec entries with zero OOM parks,
  and a client that minted a new salt per turn reached 76 entries and a CUDA OOM before the cap
  existed ([`research/27bab-20260810/RESULTS.md`](../../research/27bab-20260810/RESULTS.md#vram-ledger)).
- **Gateways.** A gateway that serves many end users through one key sets one salt per end user
  or conversation and bounds its fanout, or every user shares one namespace.
- **Purge.** A deployment binary wires `RuntimeHandles.purge` into key revocation and tenant
  deletion. It removes the tenant's host-tier entries across all its salts and its unleased device
  entries; leased device entries are reported as `device_pinned_left`, never dropped. Raw-salt
  namespaces (no keyring) carry no tenant and never match.

Gate: [`research/pc-iso-20260802/`](../../research/pc-iso-20260802/) (same-salt hit, cross-salt
miss both ways, default-namespace blindness).

## Accounting: `cached_tokens` and capture length

Every response shape reports `usage.prompt_tokens_details.cached_tokens`: prompt tokens resumed
from any tier (prefix cache, continuation pool, spec pool, DSpark pool, host tier). `/metrics`
carries the same split as `prompt_tokens_in` and `cached_tokens_in`
([cache metering](metrics.md#cache-metering)).

Captures land on the 32-token prime grid (`gdn_chunk_size()`), so a published entry's length is
`capture_len(P)` for a prompt of `P` tokens:

- `P` when `P` is a multiple of 32;
- otherwise the largest multiple of 32 below `P` that leaves at least 16 tokens after it (a
  remainder of 1 to 15 steps one more grid unit down).

What an operator sees:

- A grid-entry hit reports `cached_tokens == capture_len(P_previous)`, never `P_previous`.
- An exact re-send of an off-grid prompt restores all `P` tokens and reports `cached_tokens == P`
  while the entry's prompt-end tail is resident ([below](#prompt-end-tails)). Without the tail it
  restores `capture_len(P)` and primes 16 to 47 tokens from a grid start.
- A prompt from 65 to 79 tokens off the grid has no grid-aligned entry at or above the 64-token
  floor and publishes nothing: `[prefix-cache] seed REFUSED (grid): ...`, counted as
  `prefix_cache_seed_grid_refusals`.
- A hit site skips an entry that carries recurrent state unless it covers the prompt whole, or
  ends on the grid with at least 16 tokens left to prime. The longest entry that qualifies serves.
  The refusal logs `[prefix-cache] restore grid law: ...` and counts in
  `prefix_cache_restore_grid_refusals`. Attention-only entries keep every whole-entry hit.

Why the grid matters, and what it does not promise, is
[the prime-grid law](exactness.md#the-prime-grid-law).

### Prompt-end tails

A plain session can end its prompt off the grid after it published a grid entry, or after it
restored one and primed past it. At prefill-done it then attaches the rest of the state its prime
ended in to that entry: the KV rows past the entry, the recurrent (GDN) state and the logits at the
prompt end (`PromptEndTail`, worker.rs). A prompt equal to the entry's key plus the tail's tokens
restores both in one launch. It decodes its first token from the restored logits, the state the
cold prime itself ended in. Every longer prompt still restores the grid entry, so the grid law above
is unchanged.

A tail attaches to plain entries only (no draft plane, latent planes or TP shards). An exact re-send
on a speculative route therefore still primes its remainder.

The recurrent state dominates a tail: 157.6 MB at 24 tokens on Qwen3.8-27B (an 11,000-token entry
is about 483 MB), 66.1 MB on Ornith-1.5-35B-A3B. A tail therefore never costs an entry that served
a hit:

- it attaches only into free budget room, else `[prefix-cache] tail skipped: ...`;
- an unproven tail is dropped before any entry. A tail is unproven when a cold prime attached it
  and no restore has gone through it yet; one a covered hit attaches is proven from the start;
- a proven tail is dropped before any entry that was itself restored since its insert;
- it outlives only an oldest entry that never served a hit, which plain LRU takes anyway
  (`[prefix-cache] evict tail ...`).

Tails never spill to the host tier. `/metrics` counts `prefix_cache_tail_attaches`,
`prefix_cache_tail_strips`, `prefix_cache_tail_hits` and `prefix_cache_tail_skips`. The choice and
the rejected arms (LRU-item tails, strict tails-first) are in
[PREFIX-PROMPT-END-TAIL](../decisions/PREFIX-PROMPT-END-TAIL.md).

## Session affinity

The prefix probes need the new prompt to extend what is cached. Agent clients often rewrite
history instead, for example by stripping `<think>` blocks from earlier assistant turns, so
turn N is not an extension of turn N-1 and every turn re-primes. Affinity asks whether this is the
same conversation and resumes it at a retained boundary.

**Identity nominates.** Two tiers:

- **Explicit.** The `session_id` body field, then `user`, then the `x-session-id` header. The body
  wins over the header. `/v1/responses` maps `prompt_cache_key` to `session_id`; `/v1/messages`
  maps `metadata.user_id` to `user`. Blank values count as absent; a value over 256 bytes or with
  control characters is a 400. An id on one side only never matches.
- **Implicit.** With no id, the prompt is split at the chat template's control tokens and each
  segment contributes a hash of its first and last 8 tokens, so a rewritten segment body does not
  change its hash. Three shared segments are needed before a fingerprint names a conversation.

**Bytes decide.** A nominated session resumes only if the new prompt reproduces its committed
tokens exactly up to its turn checkpoint. A collision costs one comparison, never a wrong resume.
A rewrite below the checkpoint declines, logged with offsets (`history diverged at N of
checkpoint M`).

**The checkpoint.** Full-attention KV truncates by length, so a checkpoint copies only the
recurrent state. Both tiers place it at the last turn-marker control token before the live
generation header, rounded down to the prime grid. Prompt end is not used: the template
re-renders the generation header on the next turn, so a prompt-end checkpoint sits past the
divergence ([`research/cachespec-20260809/RESULTS.md`](../../research/cachespec-20260809/RESULTS.md)).
A recurrent cache resumes at its checkpoint only with at least 16 rows left to prime; otherwise the
turn primes cold ([`research/affinity-floor-20261109/`](../../research/affinity-floor-20261109/)).

**Scope.** Affinity is stored per `(model, cache namespace)`, so a `cache_salt` is an affinity
boundary too. Grammar-constrained requests never resume a spec session. Resumed sessions go
through the same evict-first and right-size ladder as new ones. `MEMRA_AFFINITY=0` turns it off.
Receipts: [`research/session-affinity-20260805/`](../../research/session-affinity-20260805/).

## Pool resumes of recurrent caches

A recurrent (GDN) cache never resumes on a parked end that decode or verify wrote. Each pool
rewinds to a grid checkpoint and primes the rest, or primes cold
([the pool grid law](exactness.md#the-pool-grid-law-pool-rewinds)):

| Pool | Resume point | Price |
|---|---|---|
| Continuation | The session's grid checkpoint, with at least 16 rows to prime | About 0.1 s TTFT on a 3k-token turn, RTX 5090 ([`research/pool-resume-cost-20261107/`](../../research/pool-resume-cost-20261107/)) |
| MTP spec | The on-grid turn checkpoint; a session with none is freed at retire | About +0.1 s on turns that used to resume at the parked end ([`research/spec-pool-grid-20261109/`](../../research/spec-pool-grid-20261109/)) |
| DSpark | Parks rewound to its own last grid capture and grows to the next turn's cap | Turn 3 TTFT 2.462 s to 0.275 s against a cold re-prime, RTX 5090 ([`research/pool-rewind-dspark-20261108/`](../../research/pool-rewind-dspark-20261108/), [`research/pool-grow-dspark-20261108/`](../../research/pool-grow-dspark-20261108/)) |

Every pool resume that primes a suffix starts the request's Philox counters at (0, 0), as a cold
session does. Attention-only caches resume where they stand. Sessions with no plain checkpoint
(no turn markers and no session id, `MEMRA_AFFINITY=0`, PP host bounce, glm5_next hyper trunks,
DFlash sessions demoted to plain) prime such a turn cold. Counters: `spec_pool_grid_rewinds`,
`spec_pool_grid_declines`, `spec_pool_grid_park_skips`, `dspark_pool_grid_rewind_resumes`,
`dspark_pool_grid_rewind_grows`.

## Host tier

`MEMRA_KV_HOST_MB` (MiB, default 0) adds a pinned host-memory tier behind the device prefix
cache. A device capacity eviction demotes the entry's bytes verbatim (K stays q8_0, V q5_1); a
later device miss that exactly matches a host entry re-materializes a device entry and restores
through the normal path, byte-lossless by construction. At 0 nothing ever reaches the tier.

- The budget is clamped at boot to MemAvailable x 0.6 with a warning. A pinned-allocation failure
  latches the tier off for the process; there is no pageable fallback. On a host shared by
  several stacks, set an explicit per-stack budget.
- On unified memory (GB10) host and device are the same DRAM: the tier adds copies, not capacity,
  and its budget is subtracted from the derived device share.
- `MEMRA_KV_HOST_TENANT_PCT` (default 50) caps one tenant's share of the tier. A demotion past the
  share first evicts that tenant's own oldest host entries; only an image larger than the share,
  or a row too small to cover it, evaporates. `100` disarms the cap.
- `MEMRA_KV_HOST_VERIFY=1` digests each entry at demote and checks it at promote (diagnostic).
- `MEMRA_KV_PAUSE_DEMOTE=1` (off by default) demotes a session that ended in a completed tool
  call after `MEMRA_KV_PAUSE_DEMOTE_MS` (5,000) untouched.
- `MEMRA_KV_HOST_HANDOFF` names a file for the deployment binary's export and import handles,
  which carry the tier across a restart; unset, they do nothing.
- Entries that cannot snapshot (the Step-3.7 SWA ring) never reach the tier. GLM-5 TP latent
  images reach it only with `MEMRA_GLM5_TP_KV_HOST=1`, which reserves the whole budget as one
  pinned arena at startup.

Every copy logs a `[prefix-host]` line with its duration. Counters (`prefix_host_*`) are in
[metrics.md](metrics.md#cache-metering).

## Continuation-pool park compaction

On the RTX 5090 class and the full-power RTX PRO 6000 Blackwell Workstation Edition, a parked
plain session compacts its cache from the ladder cap to its committed length, and the resume grows
it back to the request's own cap before any suffix primes. Without it a 6k-token session can park
about 1 GB on a 262k-context deployment. Other cards, the Max-Q and Server Edition included, park
whole. A failure in either direction falls back to the full-size park or a cold serve. Spec and
DSpark pools park live engine sessions and are not compacted. There is no flag
([KV-PARK-COMPACT-DEFAULT.md](../decisions/KV-PARK-COMPACT-DEFAULT.md)).

## Speculation on cache hits

MTP spec sessions probe and publish the prefix cache like plain ones. A spec session's boundary
capture adds the MTP draft plane, boundary hidden state and logits to the entry, counted in the
entry's bytes. On an unconstrained, spec-eligible whole-entry hit the worker re-arms a spec
session from the restored state instead of serving plain. Every shape that cannot convert serves
plain, and the downgrade line names why:

- `entry carries no draft plane (plain-published)`
- `partial (mid-entry) restore`
- `constrained request (grammar owns generation)`
- `full-cover hit without the entry's boundary hidden + logits`
- a sampled restore refused by the load rule below, or with `MEMRA_SPEC_RESTORE_SAMPLED=0`

**Sampled hits.** A sampled hit draws its first token by the same rule, from the same row, as the
cold spec session, so per seed a full-cover sampled hit is byte-identical to its cold twin.
Admission at load: where the cost-model gate owns the decision
([speculative.md](speculative.md#admission-gate)), a sampled restore is admitted exactly when the
model would speculate at the current demand. Without a verdict the SOLO watermark stands: demand
`max(worker-visible, HTTP in-flight)` must be 1, read after the restore's own copy, else
`[spec-restore-guard] sampled restore REFUSED: demand N > SOLO watermark 1 (...)`. The watermark
dates from before sampled sessions could demote: on RTX PRO 6000 Blackwell (96 GB), a 27B NVFP4 +
MTP model and a 4,860-token shared prefix (2026-08-19, N=5 interleaved), a sampled restore was
1.623x solo and 0.669x at c=2. An operator `MEMRA_SPEC_K` pin, `MEMRA_SPEC_GATE=0` or
`MEMRA_SPEC_RESTORE_LOAD_GUARD=0` bypasses the rule. The hit path's gates and receipts are in
[`research/spec-cache-20260818/`](../../research/spec-cache-20260818/).

**Republication.** A restored spec session publishes its own boundary after the suffix feed, so
each turn of a growing conversation can hit a longer prefix than the last
(`MEMRA_SPEC_RESTORE_REPUBLISH`, default on). It republishes at the render-stable boundary ahead of
what it restored, or at the grid boundary.

**Spec-pool sampler check.** A parked spec session's state depends on its sampler (penalty window,
draft plane). A resume whose sampler differs is refused on all three probe tiers and on the DSpark
pool, and the request serves cold: `[worker] spec-reuse REFUSED: parked session's sampler differs
on <field>; serving COLD spec ...` (`dspark-reuse REFUSED` on the DSpark pool). `seed` is not
compared. `/metrics` counts `spec_pool_sampler_refusals`, a subset of `spec_pool_misses`.

**DSpark (DFlash2).** A DSpark drafter's KV cannot come back from a trunk-only entry. With
`MEMRA_DSPARK_PREFIX_RESTORE` on, entries also carry the drafter's readable tail (about 85 MB at
30k tokens, charged to the budget) and a hit re-arms the DSpark session. The device default is on
for GB10 and the RTX PRO 6000 Max-Q and off elsewhere, every RTX 5090 included. With it off, a
hit carries no drafter tail, and the route is decided before the probe: a low-load request whose
decode budget repays the prompt's prefill bypasses the hit and primes the full prompt to
speculate, and any other request takes the hit on the plain path. The full rule is the flag's row
in [FLAGS.md](../FLAGS.md).

**Other drafters.** Gemma assistant-drafter sessions re-arm on a hit with a non-empty suffix;
full-cover gemma hits stay plain. GLM-5 spec sessions re-arm when the entry carries a drafter
tail.

Gate: `tools/spec-on-cache-hit-gate.sh` (tier 2; the tag battery does not run it): hit engagement
and byte identity against the spec-off replay, the sampled cells, the refusal lines, and a teeth
arm (`MEMRA_HITGATE_TEETH=1`) that shuts every door and asserts the plain posture.

## Per-device defaults

| Default | RTX 5090 class | RTX PRO 6000 WS (full power) | RTX PRO 6000 Max-Q, Server Edition | GB10 |
|---|---|---|---|---|
| Derived prefix budget | discrete rule | discrete rule | discrete rule | unified share |
| Admission reclaim | partial above the floor, else flush | flush with host-tier demotion (one card), flush (several cards) | partial above the floor, else flush | flush |
| Park compaction | on | on | off | off |
| DSpark prefix restore (with DSpark armed) | off | off | on for the Max-Q, off for Server Edition | on |
| Host tier | off | off | off | off (adds no capacity) |

## Records and receipts

Decisions: [PREFIX-CACHE-POLICY](../decisions/PREFIX-CACHE-POLICY.md),
[PREFIX-CACHE-DISCRETE](../decisions/PREFIX-CACHE-DISCRETE.md),
[KV-PARK-COMPACT-DEFAULT](../decisions/KV-PARK-COMPACT-DEFAULT.md),
[ADMIT-BY-MEMORY-DEFAULT](../decisions/ADMIT-BY-MEMORY-DEFAULT.md),
[C8-SERVING-24GB](../decisions/C8-SERVING-24GB.md). Gates:
`tools/prefix-newest-turn-fits-gate.py`, `tools/prefix-restore-identity-gate.py`,
`tools/cache-meter-gate.py`, `tools/kv-host-tenant-reclaim-gate.sh` (see
[TESTING.md](../TESTING.md)). The dated history of each mechanism, with its measurements, is in the
[archived serving reference](../archive/SERVING-2026-11-12.md#prompt-caching-cross-request-prefix-cache--2026-08-02).
