# DFlash restorable-hit preference (2026-11-03)

Change: a DFlash request whose longest prefix hit has no draft tail looks for the longest entry
that does (and is a restorable prefix under `MEMRA_DSPARK_PARTIAL_RESTORE`) and takes it when the
route admits DFlash on that hit and the decode budget repays the extra suffix prefill by the
cold-prime ratio (`[dspark] restorable-preferred` line). Motivation: GB10 GGUF DFlash2 c1 at
`b014549cb9` served plain (28 `cache-preferred` vetoes, 3 DSPARK restores, 12.0 tok/s), see
`research/gb10-tuning-20261102/serving/`.

## RTX 5090 Laptop (24 GB, platform profile `balanced`)

- memra-server lib suite 1137 passed, 0 failed; clippy `-D warnings` clean; the new test
  `a_dflash_request_prefers_a_restorable_hit_over_a_longer_plain_one`.
- `tools/dspark-serve-smoke.sh` on the new binary: ALL GREEN, greedy completions byte-identical
  between the DFlash boot and the spec-off boot on all three prompt classes
  (`5090-smoke/dspark-serve-smoke.log`).
- Agentic c1 A/B, base `cd2e277a7d` against new `8084041d31`, both built with the same profile,
  order base/new/new/base, Qwen3.8-27B NVFP4+Q5_K GGUF + DFlash2, `MEMRA_MTP_SKIP=1`, context
  32,768, one session, the 29 bodies of the agentic workload that fit 24 GB
  (`dflash-ab-5090.sh`): aggregate 34.09 / 32.67 / 33.40 / 33.49 tok/s. On this card the new
  branch never fires (0 `cache-preferred`, 0 `restorable-preferred`, 11 DSPARK restores in every
  arm): the small prefix cache and short sessions keep the drafter-carrying entries the longest
  hits. The spread is T=1 sampling noise on a path both binaries run identically.

## GB10 (2026-11-03, `gb10/`, build `3718972d38`)

GGUF DFlash2 c1 to c32: 12.01 / 39.72 / 59.60 / 66.55 / 67.26 tok/s, the same as the base build
(12.02 / 39.77 / 59.70 / 66.61 / 68.02). The new branch fired 0 times. The reason is upstream of
it: there is no draft-tail entry for it to prefer. The bench primes every body with
`max_tokens=1`, those primes are served plain (a 1-token decode never repays a cold DFlash
prime), and they publish plain `lcp-split` entries; the measured turns then hit those and the
cache-preferred veto serves them plain too, publishing more plain entries. `dspark-boundary`
(tail-carrying) entries exist only for the few short prompts that cold-primed DFlash.

Verdict: the preference is correct but inert on this traffic shape. The fix that matters is the
other half: a plain-served prime of a DFlash-armed model has to publish a draft tail (keep the
drafter's tap features during the plain prime, `DflashDraft::ingest_ctx`, then `export_tail`),
which costs tap state of about 100 KB per prompt token on the 27B and needs its own exactness
gate. Not done in this PR.

## Second half: a vetoed plain hit re-arms a cold drafter (2026-11-05)

Measured under a temporary door, `MEMRA_DSPARK_COLD_DRAFTER=1`; positive with nothing against it, so the door is deleted and the conversion is on wherever `MEMRA_DSPARK_PREFIX_RESTORE=1` and `MEMRA_DSPARK_PARTIAL_RESTORE=1` are (FLAGS.md "Removed doors, 2026-11-05"). The tables below are the door off/on measurements.


The traffic shape above never publishes a draft tail because the cache-preferred veto serves a long prompt's plain hit plain. With the door, when the suffix that hit leaves is at least `sliding_window + block` rows (2,056 on the 27B DFlash2), the request goes to DFlash anyway:
- the drafter is born empty at the restored boundary (`DflashKv::new_cold_at`);
- it is filled from the suffix prime's taps;
- the session speculates and publishes at its boundary capture.

Every row a round at the prompt end reads lies in the suffix, so the draft state equals a cold DFlash prime's. Shorter suffixes keep the plain hit. Needs `MEMRA_DSPARK_PREFIX_RESTORE=1` and `MEMRA_DSPARK_PARTIAL_RESTORE=1`. Commits:
- 0fda98ddc3: the conversion;
- 720d7b86f0: a fix the gate forced.
- After the rebase onto batch-spec 2239ab9d45 these are a7b5face11 and 3de636cab4; ea5ef13e8b deletes the door. The measurement SHAs below are the pre-rebase heads.

**Exactness gate** (`tools/dspark-cold-drafter-gate.py`, `cold-drafter-gate.sh`; receipts `cold-drafter-gate/`). The arm has the prefix cache and doors on; the reference has the prefix cache off, so it cold-primes DFlash every time. The sequence:
1. a plain greedy-penalized prime of P;
2. P + S (S longer than the drafter window), sampled T=1 seed 1234 and greedy;
3. a follow-up turn.

The gate needs every completion byte-identical.

| run | a greedy | a sampled | b greedy | b sampled | conversions |
|---|---|---|---|---|---|
| GB10, 0fda98ddc3 | identical | differs from token 1 | identical | differs | 4 |
| GB10, 720d7b86f0 | identical | identical | identical | identical | 4 |
| RTX 5090, 720d7b86f0 (smaller P/S) | identical | identical | identical | identical | 4 |
| RTX 5090, ea5ef13e8b (rebased on batch-spec, door deleted; smaller P/S) | identical | identical | identical | identical | 4 |
| GB10, ea5ef13e8b (`gb10-ea5e/`) | identical | identical | identical | identical | 4 |
| RTX 5090, 2a71203eb7 (memra#1069 head with the grid guard, doors on by device default; `5090-guard-2a71/`) | identical | identical | identical | identical | 4 (0 grid refusals) |
| GB10, 2a71203eb7 (same source-tree id c49c9b3691f0, doors on by device default; `gb10-guard-2a71/`) | identical | identical | identical | identical | 4 (0 grid refusals) |

Rebased head ea5ef13e8b (RTX 5090): memra-server lib 1151 passed / 0 failed, memra-engine `dspark` 23 / 0, `tools/dspark-serve-smoke.sh` ALL GREEN (`5090-smoke-ea5e/`; a first attempt died before the server wrote a line, cause not captured; the rerun with the serve lock on disk passed).

The first GB10 run had identical drafts (greedy acceptance trajectories matched round for round). The difference was the sampling stream:
- `dspark_spec_session_from_restored` drew a provisional boundary token from the entry's logits, which advanced the Philox counter;
- the real boundary draw after the suffix prime then took event 1, where a cold prime takes event 0.

This bug predates the door and shifted every sampled partial restore. 720d7b86f0 rewinds the counter on the first resume of a restored session, before any round. Tests: memra-server lib 1137 passed / 0 failed, memra-engine `dspark` 23 / 0.

**GB10 GGUF DFlash2 agentic cell** (`gb10-cold-drafter/`, spark-b23d, 720d7b86f0, one binary with the door off/on, fresh boot each, one run each):

| c | door off | door on | change |
|---:|---:|---:|---:|
| 1 | 12.04 | **14.47** | **+20.2%** |
| 2 | 22.52 | 22.47 | flat |
| 4 | 39.83 | 40.03 | flat |
| 8 | 60.11 | 56.73 | -5.6% |
| 16 | 63.43 | 66.85 | +5.4% |
| 32 | 67.67 | 67.57 | flat |

The door converted 15 of the 56 cache-preferred vetoes. The rest left suffixes shorter than the drafter window: measured turns whose hit is their own prime's entry, 19-30 tokens short of the prompt. Those stay plain, and a sub-`PRIME_MIN_T` suffix cannot take the restore path either. So c1 goes from 12.0 to 14.5 tok/s, still below the MTP route's 21.4. The c8/c16 moves are inside the single-run T=1 spread of this cell.

**RTX 5090 agentic c1 A/B** (`5090-cold-drafter-ab/`, order off/on/on/off): 33.73 / 32.75 / 32.60 / 32.75 tok/s. The door never fires on this card (0 vetoes in every arm, as in the first half), so the arms run the same path.

## Rebased head on GB10 (2026-11-05, `gb10-served/`)

Verdict: on GB10 the cold-drafter conversion lifts GGUF DFlash2 c1 by 17-20% and leaves c2..c32 flat. It acts only with the restore doors armed.

The binary is ea5ef13e8b, built on spark-cb33 (sm_121a) and copied to spark-b23d. The branch was later rebased onto batch-spec e34d55815a and then main c0f8499d4d. `crates/memra-server` did not change in either rebase. The one engine change, the `mmq_f8f4_route` lookup in `mmq_ffi.rs`, moves the same F8F4 route decision into a method.

The GB10 gate passed 4/4 with 4 conversions (`cold-drafter-gate/gb10-ea5e/`).

Agentic cell, GGUF DFlash2, the doors armed in the launcher (`MEMRA_DSPARK_PREFIX_RESTORE=1`, `MEMRA_DSPARK_PARTIAL_RESTORE=1`, `MEMRA_DSPARK_BOUNDARY_CAPTURE=1`, `MEMRA_PREFIX_STABLE_BOUNDARY=1`), fresh boot, one run each, bench client off the unit through an ssh tunnel, aggregate tok/s:

| c | batch-spec 655a07cb82 (spark-b23d) | #1031 ea5ef13e8b (spark-cb33) | change |
|---:|---:|---:|---:|
| 1 | 12.32 | **14.39** | **+16.8%** |
| 2 | 23.58 | 23.30 | -1.2% |
| 4 | 40.66 | 40.27 | -1.0% |
| 8 | 62.82 | 62.41 | -0.7% |
| 16 | 77.79 | 77.54 | -0.3% |
| 32 | 84.38 | 83.25 | -1.3% |

Both runs logged 56 `cache-preferred` vetoes. The head converted 15 of them to a cold-drafter DFlash prime, the base none. These are single runs on two units, so I read only c1. The cb33 launcher also sets `MEMRA_VISION_DIR`, so that boot loaded the vision tower; the b23d boot did not. Text-only requests never reach it.

Without the doors the conversion is off, so this PR changes nothing for a boot that does not arm them. The follow-up memra#1069 makes the doors a device default on GB10 and the RTX PRO 6000 Max-Q (the RTX 5090 stays off, owner ruling), with the doors-on vs doors-off A/Bs in `gb10-flip/` and `rig5090-flip/`.

## A converted session captures its tail entry (2026-11-06)

Verdict: on main 3bbdf42ff9 a converted session never published its draft tail, so every later turn was vetoed and converted again. c28fb5ee0e fixes it: on GB10 the third turn of a session now restores the second turn's tail, and its TTFT drops from 8.3 s to 1.4 s.

The bug: `dspark_on` was derived before the conversion's session (`dspark_prefix_restored`) was folded into `dspark_resume`, so it was false for every converted session.
- The resume ran with `DsparkCapture::Off`, so no tail-carrying boundary entry was published. The next turn hit the same plain entry, was vetoed and converted again.
- The plain-cache arm (`(None, None)`) allocated a `ctx_cap` session cache that the DFlash session never used.
- Output stayed correct: dispatch already read `dspark_on || dspark_session_installed`.

`dspark_session_armed` now also counts a prefix-cache restore. The unit test `a_cold_drafter_conversion_runs_as_a_dflash_session` fails on the old derivation. With `dspark_on` true the session-cache match takes its first arm (`None`), so no plain cache is allocated.

**Serving shape** (`tools/dspark-cold-drafter-gate.py --check-log`, now run by `cold-drafter-gate.sh`). The gate's arm log must show one `cache-preferred` veto, one `cold-drafter restore`, then at least one `DSPARK restore` and one `insert (dspark-boundary)`. Every gate receipt banked before this fix fails it with 4 / 4 / 0 / 0 (`cold-drafter-gate/shape-banked-receipts.txt`: GB10 and 5090 at 6561499516 and 2a71203eb7, PRO 6000 Max-Q at c76b8c4678).

| run | completions vs cold reference | veto / conversion / tail restore / boundary insert | shape |
|---|---|---|---|
| RTX 5090, main 3bbdf42ff9, doors armed (`5090-capture-main-3bbd/`) | 4/4 identical | 4 / 4 / 0 / 0 | FAIL |
| RTX 5090, c28fb5ee0e, doors armed (`5090-capture-fix-c28f/`) | 4/4 identical | 1 / 1 / 3 / 3 | PASS |
| GB10 (spark-b23d), c28fb5ee0e, doors on by device default (`gb10-capture-fix-c28f/`) | 4/4 identical | 1 / 1 / 3 / 3 | PASS |

The two 5090 arms produced completions of the same lengths (428 / 518 / 436 / 469), each identical to its reference. The 5090 runs used `GATE_P_PARAS=10 GATE_S_PARAS=12` (x0 / a / b prompts 1,745 / 3,857 / 3,961 tokens); the GB10 run used 70 / 80 (12,243 / 26,239 / 26,343).

**Gate exit status.** Until this PR `cold-drafter-gate.sh` exited 0 whatever happened. It had no `pipefail`, the compare and the shape check piped into `tee`, and a failed `boot && run` was never recorded. At the old default sizes (70 / 80, about 26k tokens on b) a 24 GB 5090 refuses every request with HTTP 400. arm.jsonl is then never written, the compare raises, and the script still exits 0. Now:
- the default sizes follow the card's memory (`nvidia-smi` memory.total of the first visible device): under 32 GB, 10 / 14 (about 4.4k tokens on b, S above the drafter window); GB10 (unified memory, `[N/A]`), an RTX PRO 6000 and larger cards keep 70 / 80. 30 / 30 (about 10.8k on b) is refused on a 24 GB 5090 under the arm's 4,000 MB prefix cache;
- `compare.txt` starts with the sizes, the memory reading, context and doors used;
- a refused request prints its HTTP status and body;
- a failed arm run, reference run, compare or shape check sets the exit status, and the last line is `COLD-DRAFTER GATE PASS` or `FAIL`.

`tools/test_dspark_cold_drafter_gate.py` runs the script against a stub server, no GPU. It has four cases: agreeing (24 GB reading, expects 10 / 14), refused, divergent (`[N/A]` reading, expects 70 / 80), and re-vetoed (4 / 4 / 0 / 0). Only the agreeing case may exit 0. Against main's script all four cases fail, because it exits 0 in every one. Every receipt above has its `arm.jsonl` and `ref.jsonl` and a compare that ran.

**GB10 multi-turn persistence** (`gb10-capture/multi-turn/`, spark-b23d, GGUF DFlash2, doors on by device default, `MEMRA_PREFIX_STABLE_BOUNDARY=0`, client on the unit). Order main/fix/fix/main, fresh boot each. Each run has 4 sessions of three turns:
1. x0: a plain greedy-penalized prime of P (12.2k tokens);
2. t2: P + S + a question (17.5k tokens), T=1 seed 1234, 256 tokens; the veto fires and the conversion runs;
3. t3: t2's prompt + t2's text + a follow-up (18.3k tokens).

| run | t2 TTFT s | t3 TTFT s | t3 total s | t3 cached tokens | veto / conversion / tail restore / boundary insert |
|---|---:|---:|---:|---:|---|
| 1 main 3bbdf42ff9 | 7.32 | 8.29 | 12.39 | 12,232 | 8 / 8 / 0 / 0 |
| 2 fix c28fb5ee0e | 7.33 | **1.45** | 7.19 | 17,472 | 4 / 4 / 4 / 8 |
| 3 fix c28fb5ee0e | 7.33 | **1.35** | 7.62 | 17,472 | 4 / 4 / 4 / 8 |
| 4 main 3bbdf42ff9 | 7.32 | 8.34 | 12.42 | 12,232 | 8 / 8 / 0 / 0 |

On main, t3 hits P's plain entry again, is vetoed and converted again, and prefills ~6.0k tokens. On the fix, t3 restores t2's tail entry and prefills ~0.8k. t3 decode rate does not move (47.5 / 42.6 main, 47.3 / 43.0 fix tok/s). The prompts carry the run tag, so texts and completion lengths differ between runs; TTFT and cached tokens are the clean signal. Per-turn records: `turns.jsonl`; table: `summary.md`.

**GB10 agentic c1** (`gb10-capture/c1-main-a/`, `c1-fix-a/`, the batch-spec agentic cell, one run each): main 14.26, fix 14.25 tok/s, 48 turns and 18,847 tokens in both. Both logged 28 vetoes and 15 conversions. Boundary inserts went from 13 to 28, but no measured turn requests them: the harness primes every body plain before measuring, so each measured turn's longest hit is its own prime's plain entry. This cell cannot show the persistence, so the ABBA stopped after one pair and the multi-turn shape above replaced it.

Every lever of this lane, with SHAs and receipts: `LEVER-LEDGER.md`.
