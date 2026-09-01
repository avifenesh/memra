# Prefix cache over latent planes (glm5_next): entries carry what restore must return

Lane: `lane/glm5-prefix-latent`, 2026-08-30. Parent: `lane/prefix-restore-toolcall`
(research/prefix-restore-toolcall-20260828/FINDINGS.txt, the guard and the measured
empty-history restore) and the engine survey C4
(research/glm53-flash-bringup-20260827/engine-survey-20260829/ENGINE-SURVEY.md on
`lane/glm5-engine-survey`: SGLang ships prefix caching over KDA+MLA state, ON by default,
"Keep the prefix cache enabled for every strategy").

The defect this lane closes, in one line: `PrefixEntry` carries `kv` and `recur`, glm5_next
keeps its entire retrievable history in a third plane (`cache.latent`, 11 MLA/DSA layers),
so a restore handed the model its KDA gist and an EMPTY attention history while
`cached_tokens` reported N of N. The guard from the parent lane refuses capture, restore,
and plain-affinity arming on any latent-bearing cache. This lane makes entries CARRY the
latent planes so the restore-side guard can ADMIT entries that carry them; entries that do
not still refuse.

Why it is worth an entry that is no longer free: every repeated-prefix glm5 request today
pays the full prime. On the bench box that was ~13.4 s of TTFT per hit at the shapes the
parent lane measured, and multi-turn agents (the ICP traffic) repeat their prefix on every
turn.

ACCEPTANCE BAR (owner-set, adopted by the parent lane coordinator): restored-versus-cold
BYTE IDENTITY on the glm5 raw `/v1/completions` greedy probes, p5/p7, within one boot,
rep index the only variable (the honest cold bank from the cache-off re-baseline: p5
`0e6d22ebc3787fc9`-class within-cell identity, REBASELINE.txt R3). Not "the tool call came
back". Plus the unguessable-tool round trip (`zqx_fetch_glimb_status`, latentprobe.py
pattern): cold and restored must BOTH emit it.

---

## 1. What a latent-plane snapshot contains

A `PrefixEntry` gains one plane vector, `latent: Vec<Option<LatentPlaneSnapshot>>`, one
slot per layer, `None` on every non-MLA layer (and on an allocated-but-unexecuted MTP/NextN
latent layer, the same absent-at-capture convention `kv` already uses). Each snapshot
carries exactly what `mla_attn_cached` + `mla_kpool_indices` need to continue as if the
session had primed the prefix itself:

| piece | contents | why this form |
|---|---|---|
| `rows` | `len * width` f32, rows `[0..len)` of `LatentKvLayer::rows` | The plane is flat, per token, appended by absolute row, `len` written in exactly one place (`mla_attn_cached`) and only ever growing. Copy out, copy back, set `len`/`len_d`. Deliberately UNQUANTIZED f32: the maxdiff oracle against `memra_engine::mla` depends on the f32 plane, and this lane's own acceptance bar is byte identity, so the snapshot must not introduce a numeric program. |
| `index_pool_keys` + `index_pools_ready` | `pools_ready * d` f32 (d = index_width/2), plus the count | THE KEYS TRAVEL, NOT A REBUILD. See below. |
| `index_tail` | the `len % pool` live rows of the `index_rows` tail ring | See §3. |
| `index_pool` | the indexer's pool size at capture | The StatePlan does not carry `pool` (the same gap that makes `index_pool_keys` a lazy allocation). Restore needs it to compute ring addressing and key-plane sizing on the destination, so it is captured. Its resident source is a new engine-written `LatentKvLayer::index_pool` field, set by `mla_attn_cached` exactly like the lazy key-plane allocation, and guarded: a nonzero resident value that disagrees with the loaded geometry is an `Err`, not an overwrite. |

KDA recurrent state (34 layers): UNCHANGED, whole at the entry endpoint, the existing
`conv`/`ssm` planes (the measured 152.6 MB). The design question "whole, or on the SGLang
page grid?" resolves to WHOLE, for three reasons:

1. Grid checkpoints exist to serve PARTIAL hits (truncate a miss to the deepest
   checkpointed boundary). This lane keeps the whole-entry-only posture (§4), so interior
   checkpoints are dead weight here.
2. At f32 a 64-token grid on an 8k entry is 128 checkpoints x 152.6 MB. SGLang affords its
   grid through an int8 checkpoint pool with per-slot scales; int8 changes the state bytes,
   and this lane's acceptance bar is BYTE identity against the cold program. A quantized
   checkpoint store cannot pass this lane's own gate.
3. The endpoint snapshot is what the entry already carries today; no new capture program.

The SGLang template (64-grid, int8 store, snapshot writes on decode AND extend, hit
truncation to checkpointed depth) is adopted as the named LATER increment that unlocks
partial hits, with its own acceptance stance (it cannot be byte identity); this lane adopts
the template's load-bearing idea in its memra form: THE STATE TRAVELS WITH THE ENTRY, and
snapshot writes happen at every publish boundary the worker already has (seed at
prefill-done, retire, fanout leader, learned mid-boundaries).

### The kpool derived state: keys travel, rebuild is impossible

`index_pool_keys` is derived state with an append-only finality invariant: a pool's key is
final the instant its last row lands, `index_pools_ready` counts final keys, and the
incremental build is "bit-identical to a rebuild" (mla_kpool_indices, step 2). The brief's
alternative, snapshot `index_pools_ready` + rows and rebuild keys on restore, is REFUSED on
a structural ground, not a latency preference:

* Under the shipped default (`MEMRA_DSA_INDEX_RING`, ON, 5120 working rows) the rows a
  rebuild would read ARE GONE. A row of `index_rows` is read exactly once, by the pool-key
  build of its own pool, and the ring overwrites read rows by design; that is the whole
  8.7x memory cut of the ring lane. Only the flat-plane fallback (`MEMRA_DSA_INDEX_RING=0`,
  or max_ctx <= 5120 where the ring collapses to flat) retains the rows, and a design that
  only restores in the fallback configuration is not a design.
* The rig-timing half of the question is moot twice over: rebuild has no source data under
  the ring, and the rig carries no timing authority anyway (exactness only).

Cost of carrying them: d f32 per pool = 128 B/token/layer at glm5 geometry, 6.3% of the
rows plane. Correctness: the carried keys are the final keys by the invariant, so restore
re-establishes exactly the resident state the donor session had, and the existing residency
tripwire (`pools_ready > slot/pool` refuses) plus this lane's red mutations guard the
boundary.

Capture-side invariant, asserted not assumed: at every call boundary the drain loop leaves
`index_pools_ready == len / pool` (`debug_assert!(t == 0 || *pools_ready == n_pools)`).
`snapshot_plane` turns that into an `Err` on violation, so a capture can never publish a
key plane that is behind or ahead of its rows.

### What stays out, stated

`CacheSnapshot` / `Cache::rollback` stay two-plane and stay guarded (the plain-affinity
checkpoint keeps refusing to arm on latent caches, loudly-once). Growing latent awareness
there is the spec-arm prerequisite the struct doc already names; it has different rollback
semantics (truncate-in-place vs restore-into-fresh) and no reachable caller today. Not this
lane, on purpose.

---

## 2. Memory budget: what an entry really costs

glm5_next geometry (banked glm-config.json): 34 KDA layers (heads 64 x head_dim 128,
conv_kernel 4), 11 trunk MLA/DSA layers + 1 MTP MLA layer (absent from entries, len 0),
kv_lora_rank 512, NoPE (rope 0, so latent width = 512), index_head_dim 128 (index_width
256), index_kpool 4, vocab 154,880.

Fixed per entry (any depth):

* KDA conv: 24,576 x 3 f32 = 294,912 B/layer
* KDA ssm: 64 x 128 x 128 f32 = 4,194,304 B/layer
* 34 layers: **152,633,344 B = 152.6 MB** (exactly the constant the serving log printed;
  it was the whole entry, now it is the floor)
* boundary logits 154,880 f32 = 619,520 B; toks 4 B/token; index tails <= 3 rows x 1,024 B
  x 11 = 33,792 B (noise)

Per token (the part that was ZERO in the defective entry):

* latent rows: 512 f32 = 2,048 B x 11 layers = 22,528 B/token
* pool keys: 128 f32 / 4 tokens = 128 B/token x 11 layers = 1,408 B/token
* **total 23,936 B/token (23.4 KiB)**

Honest entry size at 8k context: 152.6 MB + 8,192 x 23,936 B = 152.6 + 196.1 =
**~349 MB per warm session** (348,717,056 B + logits/toks ~= 349.4 MB). At 32k:
152.6 MB + 784.3 MB ~= **937 MB**.

What `MEMRA_PREFIX_CACHE_MB` buys, in warm sessions:

| budget | 8k warm sessions | 32k warm sessions |
|---|---|---|
| derived default (2 entries) | 2 (~699 MB requested, boot-clamped) | 2 (~1.9 GB requested, boot-clamped) |
| 4096 MiB | ~12 | ~4 |
| 8192 MiB | ~24 | ~8 |

The derived-budget arithmetic (`model_prefix_entry_bytes`) currently books glm5 at
0 B/token (only quantized full-attn planes count) and would derive a budget that holds a
fraction of one real entry. With the flag on, the derivation adds the latent per-token term
(22,528 + 1,408 B) from the plan, so "2 entries" stays true in bytes.

For scale: SGLang's KDA-state arithmetic in the survey (136 MiB/sequence fp32) is the same
136 MiB our 34 ssm planes cost; upstream bounds concurrency on it
(`--max-mamba-cache-size`). Our budget bound is the same fact through the entry ledger.

---

## 3. The tail ring: restore the rows, because nothing can re-derive them

A restore lands mid-ring-history by construction (the entry endpoint is rarely
pool-aligned). The two legal designs from the brief resolve decisively:

* RE-DERIVE: impossible. `index_rows` rows are `[k_norm(wk(h)) | gate(h))]`, functions of
  the layer's HIDDEN STATES at those positions. The entry carries tokens and cache planes,
  not hidden states; re-deriving would mean re-running the trunk forward over the tail
  tokens, which is a partial re-prime, which is the one-numeric-program violation the
  partial-restore lane already banked as a NO-GO mechanism.
* RESTORE: bounded and exact. At any call boundary the drain has built every complete
  pool (`pools_ready == len / pool`), so the LIVE rows are exactly the incomplete tail
  pool: `len % pool < 4` rows per layer, pool-aligned start, contiguous mod the effective
  ring (the ring is a whole number of pools). One `copy_range_into` out, one in. The rows
  are dead below `pools_ready * pool` in BOTH the ring and the flat plane (single-reader
  contract), so restoring only the live tail is complete for either destination layout,
  including a flat-source-to-ring-destination restore across differing max_ctx.

What `index_ring_take` changed about "restorable", honored here: the ring made row
liveness a function of `index_pools_ready`, not of `len`. So the snapshot does not copy
"the index plane"; it copies the derived keys (final, resident) plus the sub-pool tail
(live), and the restore re-establishes `index_pools_ready` so the very next
`mla_kpool_indices` call's tripwire and drain arithmetic see a state indistinguishable
from a session that primed the prefix itself. A restore that forgot the clamp-or-carry
rule would either trip `pools_ready > slot/pool` (refuses) or lap the ring
(`index_ring_take` returns None, refuses); both red arms are in the gate.

---

## 4. Partial (mid-entry) restores stay refused

Whole-entry only, the existing posture, now stated for the latent plane class in its own
right (not just via glm5's recurrent layers): `partial_prefix_decision` gains a
latent-plane arm so a hypothetical all-MLA pack (glm_dsa) cannot slide through the
`has_recurrent == false` gap when `MEMRA_PREFIX_PARTIAL_RESTORE=1` is armed. Grounds, so
nobody relitigates from the rows plane alone: latent `rows` COULD truncate at position,
but (a) the index tail for an interior boundary is unrecoverable under the ring (the rows
that would form the new incomplete pool were read and overwritten), and (b) the suffix
would re-prime through a different numeric program than the cold monolithic prime, the
exact mechanism `research/lcprestore-20260813` byte-diverged on. Whole-entry restores
reuse the donor's own boundary; no new program.

---

## 5. The gate, first, red-proven

Unit/GPU (rig 5090, exactness only, `NVIDIA_TF32_OVERRIDE=0 flock /tmp/memra-5090.lock`,
no timing):

1. `gpu_latent_plane_snapshot_restores_byte_identically`: fill a ring'd kpool fixture
   (the glm5_kpool_indexer_gpu fixture geometry) by priming a live layer, snapshot, restore
   into a fresh layer (both same-layout and flat-to-ring), compare rows/keys/tail/len/
   pools_ready as f32 BITS, then run a decode-shaped `mla_kpool_indices` step from the
   restored state and require the selected index set byte-equal to the donor's.
2. RED mutations, each asserting a LOUD refusal or a detected divergence, banked red
   before the fix lands and green after:
   * latent rows truncated one short (snapshot `rows` shrunk / `len` decremented):
     restore must Err on shape, never restore a shorter history under a longer `pos`;
   * KDA state restored from the wrong snapshot column: conv/ssm cross-wiring is caught
     by shape (73,728 vs 1,048,576 f32); the shape-identical wrong-ENTRY ssm is
     undetectable by any local check and is exactly what the box byte-identity probes
     exist to catch, plus the unit arm: restore entry A's planes, compare against donor
     B's bytes, require the comparison itself to bite;
   * pool keys stale after restore (finality violated): `index_pools_ready` inflated
     beyond `len / pool` must refuse at capture (`snapshot_plane` Err) and, if forged
     into a restored layer, must trip the resident-keys tripwire on the next call.
3. Worker predicates device-free: flag OFF keeps today's refusal string on capture and
   restore (byte-for-byte the parent lane's guard); flag ON + entry WITHOUT latent planes
   still refuses restore into a latent cache (older/foreign entries); flag ON + entry
   with planes admits; `maybe_plain_checkpoint` refuses regardless of the flag.
4. The existing 34 `prefix*` worker tests stay green (capture/restore symmetry on
   non-latent models is untouched).

Box battery (owner window required, the box is busy; plan in §7): the acceptance bar
itself, raw p5/p7 greedy byte identity cold-vs-restored, the zqx unguessable-tool round
trip on both arms, decode byte identity with the flag OFF, the 8-turn multiturn cache twin
per the owner law, TTFT-at-depth receipts.

---

## 6. The flag

`MEMRA_PREFIX_LATENT`, default OFF, by design: the mechanism's whole failure class is
"fluent wrong answers behind a truthful-looking counter", so it ships dark until the box
battery on real hardware banks the byte-identity receipts. OFF is bit-exact today's
posture (capture refuses, restore refuses, checkpoint declines). ON arms capture and
restore of latent planes together; restore admits ONLY entries that carry planes, so a
mixed population (entries minted before the flip, or by a publish site that cannot
capture) keeps refusing per entry, loudly. Rollback seam: unset the flag; entries carrying
planes then refuse restore wholesale and the cache serves cold paths, i.e. rollback is the
parent lane's guard, unchanged. FLAGS.md row lands in the same commit as the flag read
(pre-push census enforces).

Flip condition, pre-stated: restore-vs-cold byte identity on real hardware (p5/p7 raw
probes), the multiturn 8-turn larger-prompt cache twin with per-turn TTFT + cache
engagement receipts, TTFT-at-depth, and the refusal-to-admit transition verified in the
server log (refusal lines gone for carrying entries, still present for non-carrying).
Sampled rows at vendor defaults per the serving law; greedy is the instrument.

---

## 7. Box battery plan (needs an owner window)

One box, one card, the parent lane's serve env, binary from this lane's head, pinned
commit in every receipt. Arms:

* B-off: `MEMRA_PREFIX_LATENT` unset, `MEMRA_PREFIX_CACHE_MB=0` and cache-on twins:
  p5/p7 greedy x4, ONE sha per prompt per arm (the honest cold bank class), server log
  shows the refusal lines. This is the no-regression arm.
* B-on: `MEMRA_PREFIX_LATENT=1`, budget sized for >= 4 entries (e.g. 2048 MiB): rep 0
  cold + reps 1-3 restored, byte-identical request bodies within one boot, rep index the
  only variable (the latentprobe arm design). PASS = one sha per prompt across all 4 reps,
  `cached_tokens = N of N` on restored reps, `[prefix-cache]` hit lines present, entry
  size line shows real per-token bytes (not 152.6 MB constant).
* zqx round trip: tool/recall/bare probes, cold and restored, all must pass both arms
  (0/18 restored became 18/18 in the parent lane's B5 by refusing; here it must be 18/18
  by RESTORING CORRECTLY).
* Owner-law twin: 8-turn larger-prompt multiturn, cache on vs off, vendor-default
  sampling, reasoning_effort pinned, per-turn TTFT + accept + cached_tokens, loop-scored
  rows excluded and reported separately.
* TTFT-at-depth: hit vs cold TTFT at 2k/4k/8k prefix depths, sampled shape.

Receipts bank under this directory; lane closes into research/INDEX.md with the corpus
gotchas (ring liveness vs len, keys-travel law, pool-not-in-plan gap).

TURNKEY, so the window spends zero minutes on plumbing (`battery.py` in this directory;
asserts its arm's cache-engagement contract per row and exits 2 with named violations):

    # OFF arm boot (no flag; guard active), then:
    PROMPTS_JSON=... python3 battery.py out-off off
    grep -c 'snapshot failed (latent' server-off.log          # refusals present

    # ON arm boot: MEMRA_PREFIX_LATENT=1 MEMRA_PREFIX_CACHE_MB=4096, then:
    PROMPTS_JSON=... python3 battery.py out-on on
    python3 ../prefix-restore-toolcall-20260828/latentprobe.py out-zqx-on   # tool/recall/bare
    grep -c 'snapshot failed (latent' server-on.log           # must be 0
    grep 'insert probation' server-on.log                     # per-token bytes, not 152.6MB flat

battery.py cells: C1 = the acceptance bar (p5/p7 raw greedy, rep 0 cold + restored reps,
ONE sha per prompt AND cached_tokens == prompt_tokens on every restored rep); C2 = the
owner-law 8-turn larger-prompt twin (per-turn TTFT + engagement, loop-scored rows flagged
and excluded); C3 = TTFT-at-depth (~8k/16k/32k chars, cold vs repeat).
