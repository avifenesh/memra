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

## Second half: a vetoed plain hit re-arms a cold drafter (2026-11-05, `MEMRA_DSPARK_COLD_DRAFTER=1`)

The traffic shape above never publishes a draft tail because the cache-preferred veto serves a long prompt's plain hit plain. With the door, when the suffix that hit leaves is at least `sliding_window + block` rows (2,056 on the 27B DFlash2), the request goes to DFlash anyway:
- the drafter is born empty at the restored boundary (`DflashKv::new_cold_at`);
- it is filled from the suffix prime's taps;
- the session speculates and publishes at its boundary capture.

Every row a round at the prompt end reads lies in the suffix, so the draft state equals a cold DFlash prime's. Shorter suffixes keep the plain hit. Needs `MEMRA_DSPARK_PREFIX_RESTORE=1` and `MEMRA_DSPARK_PARTIAL_RESTORE=1`. Commits:
- 0fda98ddc3: the conversion;
- 720d7b86f0: a fix the gate forced.

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
