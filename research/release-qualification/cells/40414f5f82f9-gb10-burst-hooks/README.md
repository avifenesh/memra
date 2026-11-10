# The burst-hook fixes together on GB10, at 40414f5f82f9

Two scoped extras for the v0.141.1 landing of memra#1210, #1214 and #1213 (PR memra#1218). They are not a cell record; the gate does not read them.

## Why

memra#1210 (stop strings, memra#1207) and memra#1213 (the entry-yield bound, memra#1201) both change the MTP burst's emission state and round hook in `crates/memra-server/src/worker.rs`. Each lane ran its device gate on its own branch. This landing's tree holds both changes, merged by hand (PR memra#1218 names the merge), a combination no lane ran. These two extras run it, each with the gate of one of the two lanes.

## What ran

On one GB10, inside the landing lease, right after the device record `40414f5f82f9-gb10-battery-perf-serving` and with that record's own `memra-server` (`memra-server.sha256` equals the record's `build.binaries` digest). Qwen3.8-27B NVFP4 + Q5_K with its MTP head. The commands are in the `*.cmd` files and the timeline is in `driver.log`.

### 1. memra#1210's stop-string gate

`tools/spec-stop-string-gate.py`: a plain boot (`ref`, `MEMRA_SPEC_K=0`), then the MTP boot at the defaults (`spec`, burst 32), then the judge. Every request is streamed alone, greedy and at temperature 1 (seeded).

| boot | sampler | stop rows equal to plain decode | close ms, non-stop median / stop rows max (limit) |
|---|---|---|---|
| spec (mtp) | t1 | 13/13 | 76 / 180.1 (226) |
| spec (mtp) | greedy | 13/13 | 0 / 204.1 |

Coverage, per sampler: 5 (t1) and 3 (greedy) stops that began in an earlier commit slice than the one they completed in; 3 and 3 requests with two stop strings in one burst; and 4 and 4 confirmed stop strings that only a rejected draft would have completed. `STOP-STRING GATE: PASS` (`stop-judge.log`). The stop rows' completion tokens equal the lane's GB10 rows on its final head (`research/spec-stop-string-20261110/gb10/fix-e3f4c1a478/judge.log`): t1 `[19, 36, 50, 68, 72, 27, 54, 78, 27, 54, 78, 36, 29]` and greedy `[16, 32, 45, 58, 73, 25, 53, 79, 27, 53, 79, 32, 30]`.

### 2. memra#1213's stall cell

`research/spec-entry-yield-bound-20261110/tools/stall_cell.py`, greedy, at the default gate. One long streamed request runs alone for 8 s. Then N closed-loop clients send one-token requests with a think time for 30 s. Then the long request runs alone for 12 s more. The lane's columns come from its GB10 sweeps on the fix and on main v0.141.0 (`research/spec-entry-yield-bound-20261110/gb10/sweep-fix/`, `sweep-main141/`).

| clients | think ms | long tokens in the 30 s window (lane fix / main v0.141.0) | longest gap s (lane fix) | short requests | short p50 / p95 s | demotions | long text |
|---:|---:|---|---|---:|---|---:|---|
| 2 | 0 | 74 (74 / 1) | 0.96 (0.94) | 130 | 0.472 / 0.519 | 0 | prefix-equal to both |
| 4 | 0 | 57 (56 / 1) | 1.79 (1.77) | 160 | 0.835 / 0.938 | 1 | prefix-equal to both |
| 4 | 200 | 50 (50 / 1) | 1.56 (1.56) | 136 | 0.731 / 0.747 | 0 | prefix-equal to both |

In every cell the long request advances, its longest gap stays under 2 s, and its text is prefix-equal to the lane's fix text and to main v0.141.0's text. A greedy stream is cut at a time, not a token count, so texts are compared by prefix. The four-client, 0 ms cell demotes once, as the lane's own fix row of that cell does. `stall-compare.md` is the table that `compare_stall.py` made from the rows.

## Files

- `stop-string/`: the gate's plan, boots, per-row `runs.jsonl`, both server logs and the reject candidates.
- `stall/`: `stall.jsonl` (one row per cell) and the server log.
- `stop-*.cmd`, `stop-*.log`, `stop-*.rc`, `stop-judge.log`, `stall.cmd`, `stall.log`, `stall.rc`: the commands, their output and exit codes.
- `texts.jsonl` in both run directories: the raw answer texts, one JSON line each (`file`, `sha256`, `text`). The bytes are exact; each `sha` in `runs.jsonl` and `stall.jsonl` is the first 16 hex digits of the text's sha256. They are kept as JSON so that their trailing spaces stay byte-exact without a tracked-attribute change after the record.
- `memra-server.sha256`: the digest of the binary that ran.
