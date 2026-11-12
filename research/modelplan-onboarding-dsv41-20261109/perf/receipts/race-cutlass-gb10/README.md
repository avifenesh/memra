# The peer-pull race of the CUTLASS GEMMs on GB10

One GB10 card, `dsv41-gate tp` with 2 ranks on the card (`--devices 0,0`), `MEMRA_TP_TRANSPORT=peer-pull`,
`MEMRA_FULL_PREC=1`, layers 2 and 24 of the needle capture. Runs were interleaved in each lease (`scripts/plan*.txt`,
driven by `scripts/race_run.sh`); every `runs.txt` line is one process: return code, checks, failed checks, FAIL
lines (two a corrupt pass: one a rank, the ranks always agreed), seconds. The logs are gzipped.

## Debug builds (`debug-arms-d6.patch`, against the branch head before the fix)

Arms are environment variables outside the `MEMRA_` namespace, read only by the debug builds:

| arm | what it does |
|---|---|
| `V41DBG_REPS=N` | the gate repeats its 12/16/32-row passes N times; round r > 0 takes the prompt rows from 7r (so a stale read returns other bytes) |
| `V41DBG_ASFKEEP` | the split MoE's activation scale buffer (1.97 MB at 16 rows) is a per-stream buffer that is never freed |
| `V41DBG_POOLBIG` | no temporary ever goes back to the device memory pool (the per-stream free list keeps every size) |
| `V41DBG_NOREUSE` | the device pool's three reuse attributes set to 0 |
| `V41DBG_FULLWAIT` | every TMA store wait and descriptor wait in the CUTLASS unit waits for the writes (`cp.async.bulk.wait_group`, not `.read`) |
| `V41DBG_NOP` | an empty kernel after each CUTLASS launch |
| `V41DBG_SYNC` | a stream sync after each CUTLASS launch |
| `V41DBG_GSYNC` | a sync of every rank before and after each gather |
| `V41DBG_CPKERN` | every TP copy as a kernel on the stream instead of the copy engine |
| `V41DBG_SPLITPROJ` | w1 and w3 as two grouped launches |
| `V41DBG_GNORMAL` / `V41DBG_GSWAP8` | the grouped calls forced to the normal form / to swapped tileN 8 |
| `V41DBG_XSTREAM` | logs every buffer pointer taken for a stream other than the one that allocated it |
| `V41DBG_KEEP` | per-rank attention and MoE outputs against the single card's, plus copies of the split MoE's stages compared between the gate's two passes over the same rows (`DBGSTAGE`, `DBGDET`, `DBGMAP` lines) |

## Results (failed checks a run)

| lease | build, rounds | arm: runs |
|---|---|---|
| race1 | d1, REPS 10 | base 2, 6; ASFKEEP 4, 6; POOLBIG 0, 4; NOREUSE 6, 8; FULLWAIT 10, 0; NOP 6, 4; SYNC 0, 0; unchanged c2 (plain gate) 0, 0 |
| race3 | d3 | XSTREAM (REPS 10) 8: only the side stream's own buffers off their stream; KEEP (REPS 40) 44: attn_out never differs, moe_out always |
| race4 | d4, REPS 40 | KEEP 68, 68, 36 (every event starts at the split MoE's hidden rows on one rank); base 28, 22; SYNC 0, 0; GSYNC 0, 0 |
| race5 | d5, REPS 40 | KEEP 88, 62, 88: codes, scale rows and plan equal between passes; the grouped w1 + w3 output differs (w3 50, w1 10 events) in one warp's 2 x 16 columns of one pair |
| race6 | d6, REPS 40 | base 22, 28, 26; GNORMAL 0, 0, 0; CPKERN 10, 18; SPLITPROJ 18, 16; GSWAP8 194, 174 (32-row passes too) |
| race7 | the gate as it failed (decode 2, one round) | unchanged c2: 4 of 13 runs failed; fix f1: 0 of 13 |

`sfb-probe.txt` (`sfb_probe_main.inc` appended to the CUTLASS unit, host only): the mainloop's scale-factor smem a
stage, the SFB TMA box and the transaction bytes per form; they agree (no overflow).

Bins of race7: `race7/bins.sha256` (c2 = the branch head's code, f1 = the fix).
