# 262k on 2 cards, pinned-recipe arm: NOT a 262k SKU (wall at ~8k tokens)

Cell date: 2026-08-30. Owner question: does the 2-card PINNED-RECIPE arm hold 262,144
context, i.e. is a 2-card box a viable glm5 serving SKU? The old 2-card OOM wall was
measured on the f32-trunk arm (QUIRK:glm53:no-resident-pp2-headroom-f32arm); the recipe
arm (BF16 trunk) showed ~9 GiB/card free in the L2 receipts and had never been
deep-prefill-tested. This cell is that test, through the serving surface, real text only.

## Verdict

NOT VIABLE. The 2-card pinned-recipe arm is not a 262k SKU, and not a 16k SKU either.
The wall is the monolithic per-request PREFILL WORKSPACE, not session state and not
admission: it grows with prompt depth (about 0.8 MiB per token per card observed at 8k)
and exceeds the ~9 GiB/card post-boot headroom just above 8k prompt tokens.

One-line answer: a 2-card box serves GLM-5.3-Flash only to ~7-8k prompt tokens, with
single-prefill concurrency; 262k is off by a factor of ~30. Fixing it is engine work
(prefill chunking or workspace bounding on the grouped-prefill arm), not a placement or
split change. Same failure family as the 3-card 1M cell of the same night (dev2 layer-31
DSA k-pool OOM mid-prime, prefix-latent window done-line 2026-08-30T04:55Z).

## Shape under test

- Engine: cc718b988 plus cherry-pick 7cc36698c only (the 1m-demo lane's
  MEMRA_TIMEOUT_MS_MAX measurement-cell override; named deviation below).
- Artifact: glm53-nvfp4 mint, 20 shards, local copy on the box.
- Cards: 2x RTX PRO 6000 Blackwell 97,887 MiB, 600 W, CVD=0,1.
- Recipe env (owner-accepted, verbatim): MEMRA_PP_STAGES=2 MEMRA_PP_SPLITS=24
  MEMRA_PP_DEVICES=0,1 MEMRA_BF16_MMV=1 MEMRA_PP_BF16=1 MEMRA_MOE_GROUPED_PREFILL=1
  MEMRA_MOE_RESIDENT_GB=98 MEMRA_MOE_SLOTS=16 MEMRA_CTX=262144 MEMRA_MAX_SESSIONS=4
  MEMRA_PREFIX_CACHE_MB=0 NVIDIA_TF32_OVERRIDE=0 MEMRA_COMPAT=openai.
- Corpus: real Gutenberg prose per the 1m-demo corpus-build books (2600, 1184, 2701,
  145), concatenated, 9,288,681 bytes, sha256 in corpus-1m.sha256. Never synthetic.
- Probe: primeprobe262.py = the 1m-demo lane's primeprobe.py with EP/MODEL
  env-parametrized. TTFD counts the reasoning channel; token counts are the server's
  usage fields, never estimated; reasoning_effort pinned low on every arm.

## Boot gate (bar 1): PASS, twice

Both boots identical (receipts/bootgate-lines.txt, receipts/boot2/):

```
[moe] resident-experts decision (PP dev0): experts 97.84GB + trunk 0.00GB vs free 100.04GB (expert budget 98.00GB) -> RESIDENT
[moe] resident-experts decision (PP dev1): experts 89.69GB + trunk 0.00GB vs free 99.70GB (expert budget 98.00GB) -> RESIDENT
```

VRAM at ready: dev0 88,660 MiB, dev1 88,950 MiB of 97,887 (about 9.0 and 8.7 GiB free
per card), matching the L2 receipts. No SLRU decision, no FATAL.

Output-sample gate (bar 2, vendor default, fresh boot): PASS both boots. 541-token
prompt, fluent on-topic answer naming War and Peace, 0.949 s TTFD, decode 34.8 tok/s
(receipts/sample-gate.json, receipts/boot2/sample-gate-b2.json).

## Depth ladder (bar 3): the wall is BELOW the first rung

Pre-registered rungs 16k / 45k / 130k / 250k. The 16k rung died instantly; 45k/130k/250k
were never reachable (ladder charter: the OOM depth is the wall).

```
[engine-error] class=Overloaded prefill error: DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")
```

The 16,447-token prime failed 0.17 s after send, on the first workspace allocation
(receipts/ladder/rung-16000.*). The server survives the error (engine-error, not a
panic; class=Overloaded; no restart).

Downward bisect (added diagnostic, same greedy max_tokens=32 shape):

| prompt tokens (usage) | pool state              | result |
|---|---|---|
| 541    | fresh boot                     | PASS, TTFD 0.949 s |
| 7,108  | steady (post-bars, worst seen) | PASS, TTFD 9.65 s, 736.7 tok/s prefill |
| 8,072  | steady, 3 runs                 | PASS, TTFD 10.9 s, 738-740 tok/s, decode 30.1 tok/s |
| ~8,250 target | steady                  | OOM at 1.0 s |
| ~8,500 target | steady                  | OOM at 2.0 s |
| 9,202 target  | VIRGIN (fresh boot #2, only the 541-tok sample before it) | OOM at 1.0 s |
| ~10k / ~12k targets | steady            | OOM at 1.2 s / 0.1 s |
| 16,447 | first deep request after boot  | OOM at 0.17 s |

Failed rungs report no usage; their targets are chars / 4.09 (the measured chars-per-
token of this corpus at 8k). The wall is NOT pool-state dependent: a virgin-pool
9,202-token prime fails the same way.

Workspace attribution from the VRAM watch (receipts/vramwatch.csv, 1 s cadence): the
8,072-token prime raised dev1 from 89,656 to 96,152 MiB, a +6.3 GiB prefill workspace at
8k, retained by the allocator pool afterwards. The 16k attempt grew dev0 from 88,660 to
97,238 MiB and then failed its next allocation: per-card workspace demand at 16k exceeds
the whole ~9.2 GiB headroom. Scaling is roughly linear (~0.8 MiB/token/card). Progressive
death times (0.17 s at 16k, 1.0-2.0 s at 8.5-10k) match per-layer workspace growth.
Consistent with TRAP:glm53:sigmoid-router-denied-every-batched-arm: prefill runs the
grouped arm with a monolithic per-request footprint.

Admission never defends: every request was admitted ("request cost ... = 0 B/token x ctx
+ 155MB fixed"), so the context-cost model the admission gate uses does not see the
prefill workspace at all. 402/429 never appeared; the failure surface is always the
mid-stream engine-error.

## Session bars at the deepest surviving rung (bar 4): single-prefill concurrency

Deep session = 8,086-token vendor-default prime, long-form ask, live through all bars
(receipts/bars/sessionbars.json). MEMRA_MAX_SESSIONS=4.

| bar | shape | result |
|---|---|---|
| +1 | deep decoding + one 8k short | BOTH SERVED. Short: 7,880 tok, prefill 10.7 s, decode 14.7 tok/s. Deep decode degraded 30 -> 6 tok/s while contended |
| +2 | deep + two concurrent 8k shorts | BOTH shorts OOM (0.9 s / 1.8 s) |
| +3 | deep + three concurrent | ALL three OOM |
| +4 | deep + four concurrent | three OOM; the last one served (prefill 13.3 s) after the others failed and freed the pool |

Admission (MEMRA_MAX_SESSIONS=4) never engaged; no 429 was ever returned. What stops
concurrency is the same prefill-workspace OOM: TWO simultaneous 8k-class prefills do
not fit, one at a time does.

After the bars, the previously-3x-passing 8,086-token prime OOMs PERSISTENTLY (retry
too; receipts/deep-greedy.json, deep-greedy-retry.json): at 8k the box sits exactly ON
the wall and real traffic history (failed prefills, concurrency) pushes it over. The
stable serving depth is about 7k.

## Greedy + vendor rows at the stable depth (bar 5)

At 7,108 prompt tokens (receipts/deep-greedy-7k.json, deep-vendor-7k.json), both fluent
and on-topic; the vendor row correctly observes the excerpt contains one work:

| arm | TTFD | prefill tok/s | decode tok/s |
|---|---|---|---|
| greedy (temperature 0) | 9.648 s | 736.7 | 30.5 |
| vendor default (no sampling params) | 9.636 s | 737.7 | 29.4 |

## TTFD ladder vs the platform deadline

Prefill throughput is flat at ~737-740 tok/s across 0.5k-8k. Even if the memory wall
did not exist, the product route's 90 s first-token ceiling would cap this arm near
~66k tokens, and a 262k prime would take ~355 s. The 262k SKU on 2 cards fails on both
axes independently.

## Named deviations from the brief

1. Binary = cc718b988 + cherry-pick 7cc36698c (MEMRA_TIMEOUT_MS_MAX), with
   MEMRA_TIMEOUT_MS_MAX=64800000 armed from boot (the 1m-demo serve pin). Armed
   pre-emptively rather than after a deadline hit: at the pre-registered depths the 90 s
   ceiling binds unless prefill exceeds 2.9k tok/s, and a deadline cancel would be
   instrument failure, not the wall under test. With the env unset the function is
   byte-identical to stock; it changes only request-deadline validation, never timing.
2. Port 18600 (coordinator slot on a shared box; 18400/18500 were other lanes').
3. serve wrapper stop() is pidfile+exe-scoped, never a basename sweep (shared box; the
   basename-sweep incident of the same night).
4. The pre-registered 45k/130k/250k rungs were not run: the wall is below the first
   rung; a downward bisect (8 extra primes) replaced them.
5. Session bars extended past the brief's +1/+2 with +3/+4 probes; the deep ask is
   long-form so the deep session stays live through the bars.
6. Greedy/vendor rows moved from 8,072 to 7,108 tokens after the post-bars pool state
   killed 8k persistently; the three banked 8,072 passes carry the 8k numbers.
7. bootgate FATAL grep is case-sensitive: the benign gpu-watch config line
   ("fatal Xid [48, ...]") is lowercase and is not a failure.

## What goes back to the engine (memra lane, not product code)

The blocker is the grouped-prefill arm's monolithic per-request workspace
(~0.8 MiB/token/card). A 2-card 262k SKU needs either prefill chunking through the
grouped arm, a bounded/streamed workspace, or an admission cost model that sees the
workspace (so deep requests get a clean 4xx instead of a mid-stream engine error).
Until an engine lane lands that, 2-card GLM-5.3-Flash is a ~7k-context shape with
single-prefill concurrency, and the 262k SKU question is answered NO on this arm.

Receipts: receipts/ (boot gates, sample gates, ladder rungs + server-log slices, bars,
7k rows, vramwatch 1 s CSV, both serve logs); cell-scripts/ (the exact probe, serve,
gate, ladder, bars scripts); corpus-1m.sha256.
