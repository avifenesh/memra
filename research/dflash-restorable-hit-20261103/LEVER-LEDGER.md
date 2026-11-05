# Lever ledger: multi-turn DFlash after a prefix hit (memra#1031), 2026-11-03 to 2026-11-05

One row per lever tried in this lane, adopted, rejected or broken: config, hardware, n, baseline, candidate, commit and receipt. Model: owner GGUF Qwen3.8-27B (NVFP4 + Q5_K trunk) with the DFlash2 drafter, FR-Spec trim, `MEMRA_MTP_SKIP=1`. Workload: the agentic cell of `research/batch-spec-20261101/GB10-AGENTIC.md` (48 turns, `max_tokens` 768, T=1 seed 1234, thinking on), aggregate generated tok/s per concurrency.

Hardware:

- **GB10:** one DGX Spark, sm_121a. Two units, named per row.
- **5090:** RTX 5090 Laptop (24 GB), sm_120a, context 32,768, the 29 bodies that fit.

Exactness gate: `tools/dspark-cold-drafter-gate.py` (plain greedy-penalized prime of P, then P + S sampled T=1 seed 1234 and greedy, then a follow-up turn; every completion byte-identical to a prefix-cache-off reference that cold-primes DFlash every time). "4/4" = a greedy, a sampled, b greedy, b sampled all identical.

Pre-rebase SHAs stay reachable on `research/dflash-restorable-pref-8084041d31`, `research/dflash-cold-drafter-720d7b86f0` and `research/dspark-restore-flip-6561499516` (which holds ea5ef13e8b, 6561499516 and batch-spec 655a07cb82).

| # | lever | config / flags | hw | n | baseline | candidate | sha | receipt | verdict |
|---:|---|---|---|---|---|---|---|---|---|
| 1 | restorable-hit preference: a DFlash request whose longest hit has no draft tail takes the longest one that does | default with `MEMRA_DSPARK_PARTIAL_RESTORE=1` | 5090 | 2+2, order base/new/new/base, c1 | cd2e277a7d 34.09 / 33.49 | 32.67 / 33.40 | 8084041d31 | 5090-ab/, README.md | neutral: never fires on this card (0 `cache-preferred`, 0 `restorable-preferred`); spread is T=1 noise |
| 2 | same | same | GB10 (b23d) | 1 run, c1/c4/c8/c16/c32 | base 12.02 / 39.77 / 59.70 / 66.61 / 68.02 | 12.01 / 39.72 / 59.60 / 66.55 / 67.26 | 3718972d38 | gb10/, README.md | neutral: inert on this shape (fired 0 times; no draft-tail entry exists to prefer) |
| 3 | cold-drafter conversion: a vetoed plain hit whose suffix covers the drafter window goes to DFlash with a drafter born empty at the restored boundary | `MEMRA_DSPARK_COLD_DRAFTER=1` + the restore doors | GB10 | gate | cold reference | greedy identical, sampled differs from token 1 | 0fda98ddc3 | cold-drafter-gate/gb10-0fda-shift/ | breaks: the restored session drew its boundary token at Philox event 1 |
| 4 | Philox rewind on the first resume of a restored session | default | GB10, 5090 | gate | cold reference | 4/4 on both | 720d7b86f0 | cold-drafter-gate/gb10-720d/, 5090-720d/ | adopted (fixes every sampled partial restore, door or not) |
| 5 | cold-drafter door on vs off | `MEMRA_DSPARK_COLD_DRAFTER=1` vs unset, doors on | GB10 (b23d) | 1 run each, fresh boot | off c1..c32 12.04 / 22.52 / 39.83 / 60.11 / 63.43 / 67.67 | 14.47 / 22.47 / 40.03 / 56.73 / 66.85 / 67.57 | 720d7b86f0 | gb10-cold-drafter/, README.md | **adopted**: c1 +20.2% (15 of 56 vetoes converted); c8 -5.6% and c16 +5.4% inside single-run T=1 spread |
| 6 | same | same | 5090 | 2+2, order off/on/on/off, c1 | 33.73 / 32.75 | 32.75 / 32.60 | 720d7b86f0 | 5090-cold-drafter-ab/ | neutral: never fires on this card (0 vetoes) |
| 7 | door deleted, conversion on wherever `MEMRA_DSPARK_PREFIX_RESTORE=1` and `MEMRA_DSPARK_PARTIAL_RESTORE=1`; rebased onto batch-spec 2239ab9d45 | doors armed explicitly | 5090; GB10 (cb33) | gate | cold reference | 4/4; 4/4 | ea5ef13e8b | cold-drafter-gate/5090-ea5e/, gb10-ea5e/ | adopted (server lib 1151/0, engine `dspark` 23/0, serve smoke green) |
| 8 | #1031 head against its batch-spec base, both with the launcher doors on | `MEMRA_DSPARK_PREFIX_RESTORE=1 MEMRA_DSPARK_PARTIAL_RESTORE=1 MEMRA_DSPARK_BOUNDARY_CAPTURE=1 MEMRA_PREFIX_STABLE_BOUNDARY=1` | GB10, base on b23d, head on cb33 | 1 run each, fresh boot | 655a07cb82: 12.32 / 23.58 / 40.66 / 62.82 / 77.79 / 84.38 | 14.39 / 23.30 / 40.27 / 62.41 / 77.54 / 83.25 | ea5ef13e8b | gb10-served/arm2-batchspec-doors-on-b23d/, arm1-1031-doors-on-cb33/ | adopted: c1 +16.8% from 15 cold-drafter conversions (0 on the base); c2..c32 within 1.4%, single runs on two units |
| 9 | restore doors default ON with DFlash armed, keyed by device (GB10, RTX 5090), OFF elsewhere | doors unset (`DOORS=default`) | GB10 (cb33); 5090 | gate | cold reference | 4/4; 4/4 | 6561499516 | cold-drafter-gate/gb10-flip-6561/, 5090-flip-6561/ | exact on both devices |
| 10 | same, served: doors on by default against all doors off | arm 3 doors unset, arm 4 all four doors `0`; both `MEMRA_PREFIX_STABLE_BOUNDARY=0` | GB10, arm 3 on b23d, arm 4 on cb33, client on the unit | 1 run each, fresh boot, same window | off (ea5ef13e8b) 12.16 / 22.23 / 40.50 / 62.95 / 78.35 / 84.98 | 14.28 / 23.55 / 40.43 / 62.66 / 77.77 / 84.19 | 6561499516 | gb10-flip/ (first arm-3 attempt void: tunnel died, `gb10-flip/void-arm3-1256-tunnel/`) | **adopted** for GB10: c1 +17.4%, c2 +5.9%, c4..c32 within 0.9% |
