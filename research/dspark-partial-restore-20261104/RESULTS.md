# DSpark strict-prefix restore pair: restored against cold on the RTX 5090 (2026-11-04)

Verdict: PASS. With `MEMRA_DSPARK_PREFIX_RESTORE`, `MEMRA_DSPARK_PARTIAL_RESTORE` and
`MEMRA_DSPARK_BOUNDARY_CAPTURE` armed, turns 2 and 3 of a three-turn chat restored a strict
prefix of their prompt from the previous turn's boundary-cut entry and primed only the suffix.
Each answer was byte-identical to a fresh cold boot that primed the same full prompt in one call.

## Question

The owner's ruling (2026-11-04): "the DSPARK partial-restore pair gets one battery cell or is
deleted". The cell asks one thing: is a strict-prefix restore, from an entry cut at the
render-stable turn boundary, the same numeric program as a cold monolithic prime of the whole
prompt?

## Setup

- Card: RTX 5090 Laptop GPU (82 SMs, sm_120a), one card. One run per arm.
- Binary: `memra-server` built at fe262e905b, the tree that still had the pair
  (`[server] build: memra-0.138.0-ad9b992b8875 (id: source-tree, git: fe262e905bf4)`; binary
  sha256 prefix in `raw/binary.sha`).
- Model: `Qwen3.8-27B-NVFP4-Q5K-mtp.gguf` as the trunk. Drafter: the
  `Avifenesh/Qwen3.8-27B-DFlash2-memra` export at revision
  `8116c2a38d2dfe59787e2f9b29ce3bae0dc6d264`. `MEMRA_CTX=8192`, `MEMRA_MAX_SESSIONS=2`,
  `MEMRA_DSPARK_SPEC=1`.
- Warm arm: one boot with the three switches. Turns 1, 2 and 3 in order on
  `/v1/chat/completions`, temperature 0, `max_tokens` 96, thinking off. Turn N+1 sends turn N's
  messages, the warm answer and one new user message, so each prompt strictly extends the last.
- Cold arm: one fresh boot per turn (2 and 3) with the pair and the prefix restore unset, sending
  the same messages as the warm turn.
- Pass rule (`dspark-pair-client.py compare`): the warm answer of turns 2 and 3 equals its cold
  twin byte for byte (reasoning and content), and the warm log shows at least two strict-prefix
  `DSPARK restore` lines.
- Driver: `dspark-pair-cell.sh`. Raw output: `raw/`.

## Results

| turn | prompt tokens | warm: restored + primed | answer | warm == cold | DSpark rounds / drafted / accepted (warm, cold) | elapsed s (warm, cold) |
|---|---|---|---|---|---|---|
| 1 | 2,992 | cold prime, entry cut at 2,976 | 512 chars, 90 tokens | (no cold twin) | 34 / 104 / 57 | 3.594 |
| 2 | 3,109 | 2,976 + 133 | 4 chars, 5 tokens | yes | 1 / 7 / 4, 1 / 7 / 4 | 0.261, 2.404 |
| 3 | 3,144 | 3,072 + 72 | 371 chars, 75 tokens | yes | 36 / 93 / 39, 36 / 93 / 39 | 1.538, 3.660 |

`run.txt`: `turn 2: warm==cold True (len 5 vs 5)`, `turn 3: warm==cold True (len 372 vs 372)`,
`DSPARK restores [(2976, 3109), (3072, 3144)]; strict-prefix restores 2`,
`DSPARK-PAIR CELL PASS`. The compared length counts a one-byte separator between reasoning and
content; no turn produced reasoning text.

The warm boot's entries sit on the 32-row prime grid and stop short of the live turn: turn 1
inserted 2,976 tokens for a 2,992-token prompt, turn 2 (a restored session republishing at its new
boundary) 3,072 for 3,109, turn 3 3,104 for 3,144. The cold boots inserted at prompt end (3,109
and 3,144). `insert (dspark-boundary)` is the name of the insert site in both arms, not the cut.

Speculation ran after the restore. The `[spec-k] K=0 source=eligibility-fallback` line on every
turn is the MTP route's K: DSpark owns the speculative program on this boot, so no MTP session
exists. The DSpark drafter ran on every turn in both arms (the `[dspark-acc]` lines and the
response's `usage.spec`), and turns 2 and 3 drafted and accepted the same totals in the restored
arm as in the cold one.

On turn 3 admission reclaimed memory before serving: `[admit-oom] reclaim-on-defer: evicted 1
prefix entries + 0 plain + 0 spec + 1 dspark parked sessions`, and the reclaim spared the
prompt's own 3,072-token entry, which the turn then restored.

## Limits

- One card (the RTX 5090 Laptop GPU), one model and drafter, one chat shape near 3k tokens, greedy
  only. Sampled DSpark, long contexts and other qwen35-class checkpoints are not covered.
- One run per arm. Turn 2's answer is 4 characters in one round, so turn 3 carries most of the
  identity evidence.
- The elapsed times are single runs, not an interleaved A/B, and the cold arm pays a full prefill
  on a fresh boot. They are not a performance claim.
- Not exercised: an exact re-send (with the cut, it restores at the boundary and re-primes the live
  turn), a suffix under `PRIME_MIN_T` or an off-grid carrier (both refused by the restorable
  predicate, unit-tested), the memory-pressure retained admission plan, concurrent sessions and the
  host tier.
- The binary is fe262e905b. The default this cell decides is a later binary and needs this cell
  rerun on it with only `MEMRA_DSPARK_PREFIX_RESTORE=1` set (`WARM_ENV`).

Files: the logs and responses in `raw/` are the cell's output with two local paths replaced by
`<MODEL>` and `<DRAFT>`. `raw/SHA256SUMS.orig` holds the hashes of the files as the run wrote
them. The driver's paths are parameters here (`EV`, `BIN`, `MODEL`, `DRAFT`, `PORT`, `WARM_ENV`);
with `WARM_ENV` unset it repeats this run.
