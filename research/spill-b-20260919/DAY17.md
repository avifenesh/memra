# WP-B day 17: the day-16 digest divergence, reproduced and classified on the local RTX 5090 (#523 thread)

Repository: **avifenesh/memra**, branch `lane/spill-b-20260919`, merged with `origin/main` `9ef2f04d6` at
`ee208a4df`. Day 16 stopped on its precondition: one restored-suffix request per arm lineage produced different
bytes from the cache-off boot on this card, deterministically, while the target card held 28/28. The law that
binds this day (memra CLAUDE.md, "One numeric program per request"; the #379 class fixed by #578): a restored
suffix must produce the same bytes as the cold prime, on every card class; a divergence is a correctness
defect, never a data point. This day reproduces the divergence with the smallest request, keeps both token
streams, and names the two numeric programs from the server's own receipts and two same-binary controls. It
changes no engine or server code, no gate, no default and no byte; every cell is N=1, `executed-not-qualified`
in the collector's vocabulary; no timing is compared with any other card.

## The binary, and why it is the lane tip rather than byte-for-byte `main`

`origin/main` `9ef2f04d6` still boots the segmented default with the `MEMRA_PREFIX_CACHE_POLICY` door; the lane
carries `87d9e5963` (plain LRU as the only policy, victim selection and door deletion, no numeric program moved,
open as #598) and its twin gate refuses a boot that does not report `plain-LRU`. The cold prime, the prefix
snapshot, the restore and the suffix prime are `main`'s code in both trees (`git diff origin/main -- crates`
touches `worker.rs` victim selection, `host_glm.rs` five lines, and two gate scripts). Build receipt
`rtx5090-day17/build-tip/`: `memra-server` release at `ee208a4df`, `dirty.txt` empty, exit 0 in 2 min 52 s
under `systemd-run --user --scope -p CPUQuota=1200% -p MemoryMax=28G` (`build-day17.sh`), SHA-256
`5191591ba8b9f494d28b6dd0efcf9d859d0bc6fc6822917b4ac9e538d76bb6a6`, cargo 1.97.1, CUDA 13.1. Day 16's binary
was `4a55ad8e...` at `9466b8912`: two binaries, the same restored digests (below).

## The card and its regime

NVIDIA GeForce RTX 5090 Laptop GPU, 24,463 MiB, driver 595.84, `power.limit` `[N/A]`, `power.max_limit` 175 W,
clocks not pinned (a laptop part; the collector's 250 ms telemetry is the record). The card entered the first
cell idle at 53 C and 15 MiB and ran between 53 and 89 C across the day's six cells, peak draw 178.9 to
185.7 W; every cell's first telemetry sample read 15 MiB (the card was alone). Every GPU command went through
`tools/tier-battery.py --rig rtx5090 --external-lock`, the canonical `/tmp/memra-5090.lock` inherited as an FD
and proven by `tier-lock-proof.py` before any port bound (`lock.json` and `cell/LOCK.json` in every cell).
Artifact `Qwen3.8-27B-NVFP4-Q5K-mtp.gguf`, the one every B cell has used.

| cell | what | collector status | wall s | telemetry samples | temp C | peak `memory.used` MiB | peak draw W |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: |
| A `gate-a-default` | twin gate, default shape | `executed-not-qualified` (exit 0) | 126.1 | 501 | 53..87 | 21,657 | 182.3 |
| B `gate-b-day16-lru-shape` | twin gate, the day-16 lru loop shape | `executed-not-qualified` (exit 0) | 177.6 | 706 | 65..88 | 21,785 | 178.9 |
| C `probe-c-batched` | probe, texts and prime receipts, the day-16 chain | `failed` (exit 1 = DIVERGENT by the probe's design) | 157.8 | 627 | 62..89 | 23,033 | 179.7 |
| E `probe-e-restore-points` | probe, the 12,350 prompt restored from five entries | `failed` (exit 1, DIVERGENT) | 76.0 | 302 | 56..88 | 22,169 | 185.7 |
| F `probe-f-gdn-sequential` | probe, the chain under `MEMRA_GDN_CHUNKED=0` | `executed-not-qualified` (exit 0, IDENTICAL) | 168.8 | 671 | 71..88 | 22,233 | 179.9 |
| D `probe-d-tokenwise` | probe, every prompt token through `decode_step`, cold turn 10 only | `executed-not-qualified` (exit 0, IDENTICAL) | 563.7 | 2,230 | 66..87 | 17,497 | 172.9 |

## Task 1: the request that diverged on day 16, and its reproduction

From `rtx5090-day16/ab-smoke/cell/` (`REQUESTS.md`, `runs/*/run.json`, `cold/rows.json`), the two mismatches:

| arm | idx | loop turn | prompt ids | restored | suffix | generation | restored digest | cold digest | finish (restored / cold) |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| slru (runs 1, 4) | 14 | 6 | 11,750 | 11,600 | 150 | greedy, `max_tokens=8` | `70efa8d085a5531a` | `24a97a2867768b2d` | `length` 8 / `length` 8 |
| lru (runs 2, 3) | 20 | 10 | 12,350 | 12,200 | 150 | greedy, `max_tokens=8` | `22f023976ebc22fd` | `4415b7e361fc6f6b` | `length` 8 / `stop` 3 |

Both are the same shape: a restore of the previous turn's prompt-end entry plus a queued suffix of 150 ids, then a
greedy generation, on the plain path (`MEMRA_SERVE_SPEC=0`). At idx 14 BOTH arms restored 11,600 and prefilled
150, and only slru diverged: the two arms' 11,600 entries had different lineages (slru: 11,450 = r(11,150) + 300
after the return evicted 11,300; lru: 11,450 = r(11,300) + 150). At idx 20 the same 12,350 prompt matched the
cold digest under slru (r(12,050) + 300) and diverged under lru (r(12,200) + 150). Same prompt, same card,
two restore lineages, two outputs: the restored program is not the cold program, and which near-tie flips
depends on where the restore starts. Under lru the cohort returns always evicted the entry the loop had
already consumed, so the lru lineage is a straight chain (turn k restores turn k-1) and a plain 12-turn
`start 11,000, grow 150` twin reproduces it exactly. The slru lineage cannot be replayed on this binary (the
arm is deleted). `tools/prefix-newest-turn-fits-gate.py`'s `twin_ids(11000, 150, 12)` reproduce day 16's
twelve loop prompts byte for byte (`plan.json` `ids_sha256`, 12/12).

### Cell A: the twin gate on its default shape (cohort 2,800/3,000/3,200; 8 turns 9,200 + 300), verbatim

```text
PREFIX-NEWEST-TURN-FITS: budget_bytes=1073741824 cohort_bytes=737943552 turns=8 cold_turns_after_1=0 cached_ok=7/7 lines_ok=8/8 evictions=9 cohort_evictions=3 self_evictions=0 refused_or_skipped=0 effective_free_ok=8/8 V1=ok V2=ok V3=ok V4=ok -> PASS
```

Identity column `== cold`: **8/8 yes** (`254a65a01730e58b`, `24a97a2867768b2d`, `65eeb1ef8f716af1`,
`64a99158cb668b0c`, `c6b9d167a76a942e`, `8e4798b352770d9a`, `f34ee12b3db5cb9c`, `ee53848835a29d90`: the target
card's day-14 digests, to the byte). No `[admit-oom]` line in either boot. The gate passed on its default shape,
so the day-16 shape was added as the second cell.

### Cell B: the twin gate on the day-16 lru shape (cohort 1,250/1,350/1,450/1,550; 12 turns 11,000 + 150), verbatim

```text
PREFIX-NEWEST-TURN-FITS: budget_bytes=1073741824 cohort_bytes=793870336 turns=12 cold_turns_after_1=0 cached_ok=11/11 lines_ok=12/12 evictions=10 cohort_evictions=4 self_evictions=0 refused_or_skipped=0 effective_free_ok=12/12 V1=ok V2=ok V3=ok V4=ok -> PASS
```

The mechanics pass (V1 to V4 are the cache's bookkeeping); the identity column, which the gate records and does
not judge, reads **11/12**: turn 10, prompt 12,350 = 12,200 restored (`hit: 12200 of 12350`, `cached=12200
lcp=12200`) + 150, restored `22f023976ebc22fd`, cold `4415b7e361fc6f6b`, `finish_reason` `length` versus
`stop`. **All twelve loop digests are identical to day 16's lru run** (`verify-day17.py` checks each row against
`rtx5090-day16/.../runs/02-AB-0-lru/run.json`), the divergent turn included: the divergence is deterministic
across days and across two binaries built from different trees. The measured boot logged 8 `[admit-oom]
reclaim-on-defer` lines (the day-16 confound, present again), the calibration boot 0.

## Task 2: classification

### Probe C: both streams, the first differing position, the server's own prime receipts

`day17-probe.py` (this lane's instrument, imports the gate's id generators so the prompts are the gate's) replays
the day-16 chain and keeps the completion text, the `[primeseg]` prime-call receipts (`MEMRA_DEBUG_PRIMESEG=1`,
an existing documented diagnostic) and the `[prefix-cache]`, `[admit-oom]`, `[spec-k]` and `[ttft]` lines per
request. Verbatim:

```text
RESTORE-VS-COLD: arm=batched turns=12 compared=12 identical=11/12 first_divergent_turn=10 prompt=12350 restored=12200 suffix=150 first_diff_char=1 -> DIVERGENT
```

| side | text (JSON) | completion tokens | finish |
| --- | --- | ---: | --- |
| restored (12,200 + 150) | `"_\t\t\"\t\t\"\t"` | 8 | `length` |
| cold (12,350 in one boot) | `"_\n"` | 3 | `stop` |

The first generated token (`_`, the prime's own argmax) agrees; **generated token 2 differs** (`\t` versus `\n`,
first differing character 1), and the cold stream ends at token 3 while the restored stream runs to
`max_tokens`. The prime receipts for the same prompt, verbatim:

```text
cold, turn 10:
[primeseg] call start=0 take=8192 grid_off=0 bound_rem=None ckpt_at=None snapshot_at=None q=12350 budget=8192
[primeseg] call start=8192 take=1024 grid_off=0 bound_rem=None ckpt_at=None snapshot_at=None q=4158 budget=1024
[primeseg] call start=9216 take=1024 grid_off=0 bound_rem=None ckpt_at=None snapshot_at=None q=3134 budget=1024
[primeseg] call start=10240 take=1024 grid_off=0 bound_rem=None ckpt_at=None snapshot_at=None q=2110 budget=1024
[primeseg] call start=11264 take=1024 grid_off=0 bound_rem=None ckpt_at=None snapshot_at=None q=1086 budget=1024
[primeseg] call start=12288 take=62 grid_off=0 bound_rem=None ckpt_at=None snapshot_at=None q=62 budget=1024
restored, turn 10:
[admit-oom] reclaim spared the prompt's prefix entry (12200 tokens, model gate)
[admit-oom] reclaim-on-defer: evicted 1 prefix entries + 0 plain + 0 spec + 0 dspark parked sessions (global LRU); effective free 4046MB -> 4560MB
[prefix-cache] hit: 12200 of 12350 prompt tokens from cache (model gate)
[spec-k] model="gate" tenant="grow" K=0 source=eligibility-fallback prompt=12350 cached=12200 lcp=12200 active=1 wave=1 placement=single-or-non-pp2
[prefix-cache] source lease released after restore fence (entry 12200 tokens, model gate)
[primeseg] call start=12200 take=150 grid_off=8 bound_rem=None ckpt_at=None snapshot_at=None q=150 budget=1024
[prefix-cache] insert (seed): 12350 tokens, 523.6MB (resident 1042.8MB / 1074MB, model gate, ns "grow")
```

`grid_off` is `fed_len % Engine::gdn_chunk_size()` (32 by default, `MEMRA_GDN_CHUNK`). Every cold call starts on
the grid; the restored suffix call starts at the restored entry's end, 12,200, eight tokens past a grid line.
Turn 9's restored call: `start=12050 take=150 grid_off=18`; turn 2's: `start=11000 ... grid_off=24`.

### Not the admission reclaim

The reclaim ladder ran before every chain turn on this card in every cell (8 to 22 `reclaim-on-defer` lines per
cache-on boot, 0 in every cache-off boot), in the eleven identical turns as in the divergent one. At the
divergent request it ran BEFORE the hit and spared the entry that was then restored under its lease:
`[admit-oom] reclaim spared the prompt's prefix entry (12200 tokens, model gate)`, then `[prefix-cache] hit:
12200 of 12350`, then `[prefix-cache] source lease released after restore fence (entry 12200 tokens, model
gate)` (cell B and probe C, the same three lines in the same order; day 16's run 2 at lines 200 to 206 of its
server log likewise). No restore raced an eviction. Probe E's 12,288 point (below) saw no reclaim at all and its
hit was identical, while its 12,250 and 12,300 points saw the reclaim spare their entries and diverged; the
divergent and the identical rows share the reclaim, so the reclaim does not separate them.

### The two programs, named

The engine states the rule in `crates/memra-engine/src/hybrid_forward.rs` (`align_prime_ranges_to_gdn`,
verbatim): "under the chunked WY scan a prompt primed as two calls split at L is bit-identical to the monolithic
prime iff `L % gdn_chunk_size() == 0`; an off-grid call start shifts the fold grid and materializes recurrent
state at a point the monolithic program never computes." The worker applies it to the LCP-split and
message-boundary captures (`worker.rs`, "Captures land ON the prime grid (`grid_align_boundary`): the entry then
restores into a suffix prime whose call start reproduces the monolithic fold, so a hit is byte-identical to the
cold render of the same bytes"), but the **prompt-end seed** (`insert (seed): N tokens` at prefill-done,
`maybe_prefix_seed`) is captured wherever the prompt ends, and a `prompt_ids` conversation's prompt ends
anywhere: 11,000 % 32 = 24, 12,200 % 32 = 8. So:

- **Program A, the cold prime:** `prime_cache` calls whose starts all lie on the 32-token GDN WY-chunk grid
  (`grid_off=0` on every receipt above), the chunked WY scan folding the recurrent state on the grid the
  monolithic prime uses.
- **Program B, the restored render:** a KV and recurrent-state restore at the prompt-end seed boundary, an
  arbitrary position, followed by one `prime_cache` call starting there, off the grid (`grid_off=8`, 18, 24, 26,
  12 in this day's receipts). Under the chunked scan that call start shifts the WY fold grid, so the low bits of
  the recurrent state and every later logit differ from program A's, and a greedy near-tie flips.

Two controls on the same binary and card separate the fold grid from every other candidate:

**Probe E: the same 12,350 prompt restored from five entries, one `cache_salt` per entry, one boot.** For each
entry length p the probe primes `prompt[:p]` cold (the seed entry at p) and then sends the whole prompt (a restore
of p plus a suffix of 12,350 - p, every suffix at or above `PRIME_MIN_T=16` so it takes the batched prime
branch), and compares with the cache-off boot's completion of the same prompt. Verbatim:

```text
RESTORE-POINTS: arm=restore-points target=12350 points=5 identical=3/5 12288(off0,suffix62):yes 12320(off0,suffix30):yes 12200(off8,suffix150):yes 12250(off26,suffix100):NO 12300(off12,suffix50):NO -> DIVERGENT
```

| entry p | p % 32 | restored suffix call (`[primeseg]`) | text | == cold `"_\n"` |
| ---: | ---: | --- | --- | --- |
| 12,288 | 0 | `start=12288 take=62 grid_off=0` (byte for byte the cold prime's own last call) | `"_\n"` (3, `stop`) | yes |
| 12,320 | 0 | `start=12320 take=30 grid_off=0` | `"_\n"` (3, `stop`) | yes |
| 12,200 | 8 | `start=12200 take=150 grid_off=8` | `"_\n"` (3, `stop`) | yes (single restore from a cold seed; the chain's 12,200, eight restores deep, flips) |
| 12,250 | 26 | `start=12250 take=100 grid_off=26` | `"_\t\t\"\t\t\"\t"` (8, `length`) | NO, first diff at generated token 2 |
| 12,300 | 12 | `start=12300 take=50 grid_off=12` | `"_\t\t\"\t\t\"\t"` (8, `length`) | NO, first diff at generated token 2 |

Both on-grid restores reproduce the cold bytes; two of three off-grid restores flip to the same alternative
stream the chain produced; the off-grid restore whose entry had a single cold lineage did not flip while the
same entry length eight restores deep did. Suffix length does not separate the rows (62 and 30 identical,
150 identical, 100 and 50 divergent): the restore point's grid offset and the entry's lineage do.

**Probe F: the whole day-16 chain under `MEMRA_GDN_CHUNKED=0` on both boots** (the documented sequential GDN
prefill scan; the engine's `gdn_prime_grid_on` doc: "the sequential scan (`MEMRA_GDN_CHUNKED=0`) [is]
split-invariant, so the grid is a no-op contract there"). Same prompts, same cohort, same restore starts, the
same `[primeseg]` geometry (cold calls `grid_off=0`, the turn-10 restored call `start=12200 take=150
grid_off=8`). Verbatim:

```text
RESTORE-VS-COLD: arm=gdn-sequential turns=12 compared=12 identical=12/12 first_divergent_turn=none -> IDENTICAL
```

Under a split-invariant scan the restored render equals the cold render on all twelve turns: the restore
itself (the copied KV bytes, the materialized recurrent state, the lease and fence) is exact, and the only thing
the chunked arm adds is the fold-grid shift at the off-grid call start. Two more facts from F's rows: the
sequential scan's COLD completions equal the chunked scan's on 10 of 12 turns and differ at exactly turns 6
(11,750) and 10 (12,350), the day-16 slru and lru flip prompts, so those two prompts are greedy near-ties that
any low-bit change resolves either way; and the sequential scan's cold 12,350 completion is
`"_\t\t\"\t\t\"\t"`, the very stream the chunked arm's off-grid restores produce. The sequential scan is a third
numeric program, not a fix: a single-arm serving default is out of this day's scope and `MEMRA_GDN_CHUNKED`
stays a rollback seam.

### Probe D: one program on both sides (the bisect the brief named, on the door the server actually has)

The brief's seam, `MEMRA_PRIME_TOKENWISE`, is read by the engine's CLI generate paths (`decode.rs`), the
spec prime (`spec.rs`, `spec/prime.rs`) and `run_gen`/`run_spec`; the server's `prefill_tick` never reads it,
so on the serving path it selects nothing. The server's own one-program door is `prefill_tick`'s `else` arm
(the "W1 two-programs door" its comments name): with the tick budget below `PRIME_MIN_T=16` the prime branch
is never taken and every prompt token, cold or suffix, rides `decode_step`. `MEMRA_PREFILL_TICK` is a documented
per-tick budget (`memra-lanes`, `u("MEMRA_PREFILL_TICK", 1024)`, authoritative when set, no floor), so probe D
runs the chain to turn 10 with `MEMRA_PREFILL_TICK=8` on both boots, cold turn 10 only, no cohort, request
ceiling 900 s (the walk runs at roughly 28 prompt tokens per second on this card: about 7 minutes for the cold
12,350 and for the chain's cold 11,000). Verbatim:

```text
RESTORE-VS-COLD: arm=tokenwise turns=10 compared=1 identical=1/1 first_divergent_turn=none -> IDENTICAL
```

Both boots walked every prompt token through `decode_step`: the cold log holds 12,350 `[primeseg] TOKENWISE
prompt token at fed=... (bound_rem=None q=... budget=8)` receipts and zero `call start=` receipts, the chain log
11,000 + 9 x 150 = 12,350 and zero. Turn 1's seed published from the walked state (`insert (seed): 11000 tokens,
483.5MB`), every turn restored the previous one (`hit: 12200 of 12350` at turn 10, `source lease released after
restore fence`), and the restored turn 10 IS the cold turn 10: `"_\n"`, 3 tokens, `stop`,
`4415b7e361fc6f6b` on both sides. Nine restores deep, under a single numeric program the restore is exact at the
very request that flips under the batched arm. The walk ran at about 45 prompt tokens per second (cold 12,350 in
272.6 s, chain turn 1 in 239.2 s), no `[admit-oom]` line in either boot (peak `memory.used` 17,497 MiB: no
prefill workspace), collector `executed-not-qualified`. The tokenwise program's own chain outputs differ from the
batched program's at turn 2 (`10d20bd924836441` versus `66d394ced7203380`, another near-tie prompt), which is two
programs disagreeing with each other as the W1 comments say they legally do, not a restore question. D's two
server logs (one line per walked token) are kept gzipped; the probe's `summary.json` carries the collapsed
receipts the replay reads.

### What this shape has that the #379 gate's cells do not

`tools/spec-on-cache-hit-gate.sh qwen` is ALL GREEN on this card in the local battery, and it does not ask this
question. Its identity cells compare a full-cover hit (a restore with an EMPTY suffix, so no suffix prime call
and no fold-grid shift) with its cold leader (`r2 text == r1 text`, `g4 reproduces its publisher's continuation`),
a spec-on hit with a spec-off boot serving the same hit (two restores), or two sampled suffix hits with each
other at one seed (`sx sampled suffix hit reproduces byte-for-byte at one seed`); the suffix-fed cells (`r3`,
`sx`, `g2`, `g3`) assert `cached_tokens > 0` and spec engagement, never bytes against the cold render of the
same prompt. Its prompts are one chat-rendered paragraph of a few hundred tokens with a one-sentence extension,
spec on (MTP head), sampled and greedy cells, `max_tokens` up to a few dozen. The day-16 shape: `prompt_ids`
prompts of 11,000 to 12,650 ids, the restore taken from the prompt-end seed at an arbitrary position with a
50 to 150 id suffix that takes the batched prime branch, spec OFF (`MEMRA_SERVE_SPEC=0`, no MTP head), one
request at a time, greedy `max_tokens=8`, and a greedy near-tie at generated token 2 of two of the twelve prompts.
The gate that covers this class is the twin gate's identity column, which today is recorded and not judged.

## Verdict

**A correctness defect, its two programs named.** On this card the restored render of a growing conversation
(program B: restore at the prompt-end seed boundary, then one off-grid `prime_cache` call under the chunked GDN
WY scan) is not the cold render (program A: on-grid prime calls), and two greedy near-ties in a twelve-turn
chain flipped, deterministically, on two binaries. The restore is exact (probes F and D, two different single programs); the seed boundary is what sits off the
grid the engine's own law requires (probes C and E). The class is card-independent (it is the
fold grid, not a kernel); which request flips is card-dependent (the target card's 28/28 on day 15 and 8/8 on
day 14 and today's cell A are near-ties that did not flip there, not identity), so the day-15 A/B's digest
precondition held on the target card by chance and the decision record now says so. The fix is a lane day of
its own and is not started here; the receipts point at the prompt-end seed's capture position (the LCP and
message-boundary captures already align down to the grid through `grid_align_boundary_within`, the seed does
not), and at the twin gate's identity column becoming a verdict. Day 16's slru request (11,750 = 11,600 + 150)
is the same class: probe F shows 11,750 is the chain's other near-tie prompt.

Reported as **memra#602** (labelled bug) with the exact repro (shape, card, binary, both digests, both texts, the
first differing token, the receipts, the controls), and linked from the #523 thread by a comment (#523 stays
open for its own items).

## Checks actually run

| Check | Result |
| --- | --- |
| `git merge --no-ff origin/main` (`9ef2f04d6`) at `ee208a4df`; push through the pre-push hook (perf board, flags census, releasability and docs-registry censuses, public boundary) | allowed, `0fae7e715..ee208a4df` |
| Native release build, lane tip, own target dir, CPU quota (`build-tip/`) | exit 0, `dirty.txt` empty, 2 min 52 s |
| Cell A, `tools/prefix-newest-turn-fits-gate.py` default shape through the collector | `-> PASS`, identity 8/8, exit 0, `executed-not-qualified` |
| Cell B, the same gate on the day-16 lru shape | `-> PASS` (mechanics), identity 11/12 (turn 10), exit 0, `executed-not-qualified`; twelve digests equal day 16's lru run |
| Probe C, `day17-probe.py` batched arm, texts and receipts | `-> DIVERGENT` (turn 10, generated token 2), exit 1, collector `failed` by the probe's exit code |
| Probe E, restore points 12,288 / 12,320 / 12,200 / 12,250 / 12,300 | `identical=3/5`: on-grid 2/2 identical, off-grid 1/3 identical, exit 1 |
| Probe F, the chain under `MEMRA_GDN_CHUNKED=0` | `identical=12/12 -> IDENTICAL`, exit 0, `executed-not-qualified` |
| Probe D, `MEMRA_PREFILL_TICK=8` (every prompt token through `decode_step`, both boots) | `compared=1 identical=1/1 -> IDENTICAL`, exit 0, `executed-not-qualified` |
| `research/spill-b-20260919/verify-day17.py` (offline replay of every receipt above) | `DAY17 REPLAY OK` (`rtx5090-day17/verify-day17.log`) |
| `cargo fmt --all -- --check` (CPU quota), `bash tools/docs-registry-census.sh`, `git diff --check` | PASS (`fmt-check.log`, `docs-registry-census.log`) |
| `tools/spec-on-cache-hit-gate.sh qwen` | NOT RUN today: ALL GREEN on this card in the local battery; its cells do not ask restored-versus-cold on a suffix-fed hit (named above) |
| Full GPU exactness battery | NOT RUN (no code change) |

## Boundaries and record

- No engine or server code changed; no gate relaxed; no new `MEMRA_*` name (the probe passes only existing
  documented reads through `--env`); nothing of V4.1; no external dependency; no captured, restored or served
  byte changed. `--no-verify` not used, no skip variable, no third lock name, no bare GPU run, no other lane's
  worktree touched, `main` untouched. Build and every cell under `systemd-run --user --scope -p CPUQuota=1200%
  -p MemoryMax=28G`.
- Receipts under `rtx5090-day17/`: `build-tip/`, one collector directory per cell (`CELL.jsonl`, `lock.json`,
  `command.log`, `command.gpu.csv`, `command.capture.json`, `gate-source.txt`, and `cell/` with `LOCK.json`,
  `plan.json` or `rig.json`, `summary.json`, `TURNS.md`, `VERDICT.txt`, per-boot `server.log`), the driver logs,
  exit files, `chain*.sh` and `card-before-*.csv`, `fmt-check.log`, `docs-registry-census.log`,
  `verify-day17.log`. Drivers: `build-day17.sh`, `run-day17-gate.sh`, `run-day17-probe.sh`; instrument
  `day17-probe.py`; replay `verify-day17.py`.
- No timing is compared with the target card or any other box; the `prime_ms` and wall figures are this card's
  own record of its own single runs.
