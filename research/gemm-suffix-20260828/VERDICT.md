# VERDICT: lane/gemm-suffix, the continuation prime rides the batched GEMM path

Lane branch `lane/step37-mtp-masked-vocab-20260825`, commits `553a072471` (fix + flag),
`a0142b30d6` / `3f89dd062b` (pre-registered predictions and flip gate, before the data),
`0cce9704c6` (battery 1 receipts + the pre-registered battery 2 reading), `a33e76aade`
(reconciliation). One binary (`/root/memra-server.gsuffix`, md5 `0216d7011fb3`, tree
`e3faf5a17c` + `gemmsuffix.patch` sha256 `8d1ca90e...`), one GPU lock hold per battery,
every door named explicitly. Raw receipts: `raw/gs-battery.txt` (battery 1),
`raw/gs-battery2-attempt1.txt` + `raw/gs-battery2.txt` (battery 2), per-arm receipt
extracts `raw/gs-{off,on,canary}-receipts.log` and
`raw/gs-walkonly{-attempt1,}-receipts.log`, build receipt `raw/gs-build.txt`,
`raw/binary-fingerprint.txt`.

## 1. Perf: the goal is met, and beaten

The walk suffix cost line reproduced in-battery (`0.2490 s + 5.8067 ms/suffix-token`,
R^2 0.9999, within 3.7% of the coordinator's independent sweep, flip-gate condition (c)
SATISFIED). On the batched entry it collapses:

| quantity | off (walk suffix) | on (batched suffix) |
|---|---|---|
| suffix cost line | `0.2490 s + 5.8067 ms/tok` | `0.3060 s + 0.7286 ms/tok` |
| LEG G growing-conversation pairwise TTFT ratio (n=5, interleaved) | median 1.088x | median **3.388x** (min 3.058, max 4.171) |
| ARM VALIDITY | VALID (eng_suffix=0, walk_suffix=41) | VALID (eng_suffix=41, walk_suffix=0) |

A **7.97x slope collapse** (5.8067 -> 0.7286 ms/suffix-token), beating the predicted
0.99 ms/tok. The session-affinity wash (1.012x-1.088x) becomes a 3.388x median TTFT win on
the growing agentic conversation shape. Per-row agreement with the pre-registered
prediction is tight (s0250: predicted 0.539 s, measured 0.557 s). The ~0.25 s intercept is
session restore, out of this lane's scope as pre-registered. Predictions 1, 2 and 3 hold.

## 2. The seq_end live bug, demonstrated clean

`step35_prime_batch_layers` passed the CHUNK-LOCAL length as `seq_end`, the value step35's
SWA arm keys on (`seq_end > win`, win=512). A sub-512 trailing chunk at a nonzero base
therefore took the UNWINDOWED arm over a view still holding ~win-1+t rows: attention
outside the sliding window, on a FRESH prompt, no continuation needed. The fix
(request-absolute `seq_end = cache.pos + t + queued_after`, computed once in
`prime_cache_overlaid`, threaded through the batched entry) is unconditional, behind no
flag.

LEG F, a fresh 4338-token prompt (4096 + 242 trailing chunk):

| arm | LEG F sha |
|---|---|
| on (fixed) | `84ddc832a39c92ff` |
| canary (`MEMRA_STEP35_PRIME_BATCH_TSEND=1`, pre-fix seq_end restored) | `41f9b3df5e88909b` |

canary != on with IDENTICAL chunk decomposition, doors, and implementation isolates the
defect to the one value the canary changes. The pre-registered window-straddle signature
also appeared: canary flips every sub-512-chunk warm row and MATCHES s0700
(`fe32811b97ed12c7` in both arms), exactly where chunk-local and request-absolute both
exceed win=512. The seam is provably read; the pre-registered "canary MATCH ambiguity"
worry is retired.

## 3. Identity: the narrowed claim, and the mechanism

Pre-registered prediction 4 (ON-arm byte identity 6/6) is FALSIFIED: LEG S is **0/6 in
BOTH the ON arm and the OFF (incumbent) arm**. This is not a regression introduced by the
hoist; it is a property the incumbent already has, and it PREDATES this lane. Battery 2
(section 5) then showed it is not specific to the batched prime either. `docs/FLAGS.md`
already declines byte-identity claims for the batched prime ("admission = prefill-KV
acceptance gate + ship-shape tape + interleaved wall, never byte identity").

The narrowed claim, with the affinity lane's correction folded in:

* WRONG, too broad: "a rewound session answers the same bytes differently from a cold
  session."
* RIGHT: **for a GROWING conversation, where the reused prefix was primed as part of a
  shorter prompt than the cold twin's, the reused answer differs from the cold answer.**
  Measured 0/6 across suffixes of 80 to 4440 tokens in the ON and OFF arms, and 0/2 in the
  walk-only arm at a smaller geometry. Where the identical prompt is resent (same session,
  same split), bytes match 5/5, receipted by the affinity lane (`578598fb4d`).

The discriminating variable is the prime decomposition: the warm session's reused prefix
rows were primed inside a shorter span (battery 1: m=1440 by turn 1) than the span the
cold twin primes the same rows in (m=1696). Under a decomposition-dependent prime those
rows differ before a single suffix token exists, and every later row inherits it. The
engagement receipts pin the comparison: `seq_end` agrees pair-by-pair on every pair (so
the prompts are identical and the SWA arm choice is identical), the trailing chunks are
literally identical on both sides, and the divergence survives arm changes.

What battery 1 could NOT decide is whether that decomposition-dependence is specifically
the batched prime's (its grouped NVFP4 MoE builds CSR and per-expert GEMM shapes from the
chunk contents) or shared more widely. That is what battery 2 asked.

## 4. Reconciliation with the affinity lane's 5/5

Both results are correct; they answer different questions. The affinity lane resends the
SAME prompt on the SAME session: its reused rows were primed by the cold send itself at the
same m, so byte identity there is the arithmetic being literally repeated, and it proves
session RESTORE is faithful. This lane's LEG S is turn 2 of a GROWING conversation: the
reused prefix was primed at a different m than the cold twin's, which is what every growing
conversation does by construction. Discriminating variable: m equal -> MATCH (5/5); m
different -> DIFFER (0/6, and 0/2 walk-only). Full table in `RECONCILIATION.md`
(`a33e76aade`).

## 5. Battery 2, the walk-only attribution arm: the walk fails identity too

Pre-registered reading (committed in `0cce9704c6`, before any battery 2 data): "If it
comes back 2/2 MATCH, the identity failure is the batched prime's m-dependence and nothing
to do with the hoist. If it fails too, the cause is upstream of the prime entirely and
this lane's reading is wrong."

**Attempt 1 (banked, `raw/gs-battery2-attempt1.txt`) was ARM INVALID**, and the failure
mode is itself a finding. With `MEMRA_STEP_GEMM_PRIME=0` the full-prompt walk prime
measured ~57.5 ms/token (84.5 s for 1480 tokens, 133.6 s for 2292, linear across ten
observations), so every request at battery 1's geometry blew the server's 90 s first-token
deadline (`TIMEOUT_MS_MAX = 90_000`, a platform cap that streaming does not lift for the
first token) and 0/17 rows were valid. Two consequences worth keeping:

* **The batched prime is load-bearing, not just faster: with it off, the engine cannot
  answer any fresh prompt past ~1500 tokens inside its own deadline.** The
  `MEMRA_STEP_GEMM_PRIME=0` rollback seam is not a viable serving fallback for long
  prompts.
* The timed-out primes ran to completion anyway (the abort is detected at first write), so
  each dead row still held the GPU for its full prime; the whole invalid arm cost ~33 min.

The relaunch (one instrument change, banked in `gs-battery2.sh` / `gs-drive.py`, both
env-gated so the battery 1 arms replay unchanged) ran the SAME shapes at a geometry the
deadline admits: turn 1 truncated to ~704 tokens (`GS_U1_WORDS=480`), sweep s0250 + s0450
(largest cold twin 1176 tokens, ~69 s), LEG F/G skipped (`GS_LEGS=S`, unreachable at walk
speed and not this arm's question). The m-fork the prediction needs is preserved: the warm
prefix rows 0..703 are primed by turn 1 as `WALK t=704 base=0 seq_end=734`; the cold twin
primes the same rows inside `WALK t=960 base=0 seq_end=976` (s0250) / `WALK t=1152
seq_end=1176` (s0450).

**Result: ARM VALID, and LEG S DIFFERs 0/2.**

```
eng_fresh=0 eng_suffix=0 walk_fresh=4 walk_suffix=8   (batched prime fully off)
s0250 suffix=272 | warm fc1f99701c7183e6 | cold d7ac1c76b94b032b -> DIFFER
s0450 suffix=472 | warm 381e9fa6669ba0eb | cold 03f2a02fdccefbbb -> DIFFER
```

Warm rows genuinely rewound (`plain-affinity rewound = 2`), cold rows did not, `seq_end`
agrees within each pair, and the trailing chunk is literally identical on both sides
(`WALK t=16 base=960 seq_end=976` appears in warm and cold s0250 alike).

**The walk path is NOT m-invariant in this cell.** The answer to the attribution question,
stated against the pre-registered bar and not past it: the growing-conversation identity
failure is NOT specific to the batched prime. Every measured prime path fails
cold-vs-rewound byte identity under a decomposition fork; either the walk shares the
decomposition-dependence or the cause sits upstream of the prime dispatch entirely. There
is NO measured m-invariant reference path, so the follow-up idea of pinning m for the
batched prime alone is not shown to be actionable: pinning would have to make warm and
cold decompositions equal, which session reuse on a growing conversation cannot do by
construction. Caveats, stated rather than hidden: n=2, one geometry (smaller than battery
1's, forced by the platform deadline), and `MEMRA_STEP_GEMM_PRIME=0` is the whole-path
seam, so this arm speaks for the walk THAT SEAM selects. It prints the same
`[gemm-prime] WALK` receipt line battery 1's off-arm suffixes printed, but it runs ~10x
slower per token (~57.5 vs ~5.8 ms/token, suffix walks included), so whether it is the
byte-identical code path of the off-arm suffix walk is NOT established here; this lane
reports the discrepancy and does not explain it.

What survives untouched: the hoist is not the cause (it is absent from every failing
comparison), the failure predates this lane, and the narrowed claim of section 3 stands.

## 6. Flip decision: the default stays OFF

Pre-registered gate (`a0142b30d6`, before the data), all three required:

* (a) ON-arm LEG S MATCH on every valid row: **FAILED** (0/6).
* (b) ON-arm validity: passed (eng_suffix=41, walk_suffix=0).
* (c) OFF-arm slope replication: passed (within 3.7%).

Condition (a) fails, so **`MEMRA_STEP_GEMM_PRIME_SUFFIX` stays default OFF**. The gate was
fixed before the data and is not renegotiated now that the perf numbers are good.
Independently, every battery row here is greedy, and a default flip is never justified by
greedy-only rows (owner law); a future flip lane needs vendor-default sampled rows plus the
8-turn cache-on twin.

What the data supports instead, as a recommendation for the owner rather than a unilateral
flip: admit the suffix arm under the SAME standard the batched prime itself is admitted
under (prefill-KV acceptance gate, ship-shape tape, interleaved wall), because
cold-vs-rewound byte identity is a bar NO measured prime path meets under a decomposition
fork, and session reuse on a growing conversation forks the decomposition by construction.
The bar as written is unreachable, not merely unmet.

## 7. What this settles, and what it does not

This lane settles BYTES, not QUALITY. Nothing here says which of two differing completions
is better; that is the owner's product call and must not be inferred from a sha. The
instrument caveat stands: LEG S prompts differ only in filler, so cross-row sha equality is
uninformative and only within-pair comparisons (verified by identical seq_end) are used.
The finding is sharpened, not weakened, by that: output is insensitive to hundreds of
filler tokens, yet the same prompt primed at two decompositions moves the answer.
