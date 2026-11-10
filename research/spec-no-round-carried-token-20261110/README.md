# A continuation burst that runs no round and its carried token (memra#1196, 2026-11-10)

Verdict: fixed. A greedy speculative burst that entered on a carried pending token and ran no
round pushed that token into the session's committed list although no round had committed its
row. `session-gate` with a no-round burst fails on main (`committed=326 cache rows=325`) and
passes with the fix, on one RTX 5090 Laptop GPU and on GB10. The tokens a request writes never
moved. What the extra token costs the next turn in serving is not measured on the
defect's own path: on this recurrent trunk the next turn is served from the prefix cache or a
rewind below the duplicate, and the serving runs kept here never put a no-round burst on a
carried token.

Model: Qwen3.8-27B NVFP4 + Q5_K GGUF with its MTP head. Arms are two builds per device: `main`,
commit 2628ece940 with the gate's new check and nothing else, and `fix`, this change. Each
device built its own two binaries.

## The defect

The MTP route decodes in bursts. A greedy burst ends on a pending token: emitted to the client,
not yet in the trunk cache. It is stashed on the session and the next burst verifies it as the
first column of its first round (the pending carry, 2026-08-01). `SpecSession::committed` is the
list of tokens whose rows the cache holds, and must have the cache's length.

The worker's commit hook can say stop at a burst's entry: a request is waiting for admission
(the admission yield, 2026-08-06). The burst then runs no round. Its tail re-stashed the
carried token, correctly, and then pushed it into `committed` as if round 0 had verified it
(`spec_burst_end`). From then on the list is one token longer than the cache for every such
burst, and it holds the token twice once its round runs.

The tokens a request writes do not change: the verify reads the cache, and the engine
positions from the cache's length. What reads the list:

| reader | what the extra token does |
|---|---|
| the context-full check (`committed + K + 3` against the cache size) | the request stops one token early per no-round burst |
| the park at the end of a request | the parked session's token list, its text and its fingerprint carry the token twice |
| the pool resume, exact and text tiers | the next turn's prompt does not extend the list: a miss. A recurrent trunk rewinds to its turn checkpoint, below the duplicate, and loses nothing. An attention-only trunk loses the resume at the parked end |
| a debug build | two assertions in the tail fail |
| a sampled penalty window (seeded from the list) | nothing on main, where a sampled burst never carries; the sampled pending carry (memra#1186) needs the list right |

The fix: the carried token joins `committed` unless the tail re-stashed it without a round.

## The gate

`session-gate` (K=3, 24 tokens per turn) ends with a continuation burst whose first commit
hook says stop, then a normal burst. The carried token must stay pending, `committed` must stay
the cache rows and the true history less that token, and the next burst must emit the oracle's
tokens.

| device | arm | the no-round check | verdict |
|---|---|---|---|
| RTX 5090 | main + the check | `committed=326 cache rows=325; next burst 24 tok, committed=350 cache rows=349; committed is the history false/false -> MISMATCH` | `SESSION GATE: FAIL` |
| RTX 5090 | fix | `committed=325 cache rows=325; next burst 24 tok, committed=349 cache rows=349; committed is the history true/true -> MATCH` | `SESSION GATE: ALL TURNS MATCH` |
| GB10 | main + the check | `committed=326 cache rows=325; next burst 24 tok, committed=350 cache rows=349; committed is the history false/false -> MISMATCH` | `SESSION GATE: FAIL` |
| GB10 | fix | `committed=325 cache rows=325; next burst 24 tok, committed=349 cache rows=349; committed is the history true/true -> MATCH` | `SESSION GATE: ALL TURNS MATCH` |

On every arm the four turns before the check match the oracle and the step identity line reads
`0 of 1986560 logits differ`: the defect moves the list, not a token. The no-round burst
itself emits nothing and leaves the carried token pending on both arms; what differs is the
list. Logs: `rtx5090/session-gate-*.log`, `gb10/session-gate-*.log`.

## In serving

Not measured on the defect's own path. What was run, on one RTX 5090 Laptop GPU, greedy, with
the server binary of main 031f146690 (the commit before tonight's landing; `spec.rs`'s tail and
the worker's readers are the same there), `MEMRA_SPEC_GATE=0`, eight sessions:

| first request | arrivals beside it | its continuation bursts | of them no-round | next request: prompt / cached tokens |
|---|---|---:|---:|---|
| chat turn, 1,286-token prompt, 160 tokens | none | 4 | 0 | 1,478 / 1,248 |
| the same shape | four closed-loop one-token clients | 97 | 97 | 1,414 / 1,248 |
| text completion, 1,083-token prompt, 160 tokens | none | 4 | 0 | 1,243 / 1,056 |
| the same shape | four closed-loop one-token clients | 97 | 97 | 1,219 / 1,088 |

(`rtx5090/flood/resume.jsonl`; the server log beside it.)

- With nothing else arriving, the next request restores its prefix from the prefix cache at
  the 32-token grid (`[prefix-cache] hit: 1248 of 1478 prompt tokens`). The parked session is
  not what serves it: the worker frees it at the park (`spec-park SKIPPED (grid law)`), as it
  does for every session of a recurrent trunk whose checkpoint is off the grid.
- With four clients sending one-token requests back to back, every burst of the long request
  began while a request waited for admission. It never ran a round, so it never held a carried
  token, and each burst emitted its one boundary token: 97 tokens in 90 s, where the server's
  non-streaming deadline cut it (`[deadline] ... delivered PARTIAL: 97 tokens in 90.0s of a
  90000 ms deadline`, finish reason `error`). That is another finding (memra#1201), and these
  rows are its receipts. The request after it still restored its prefix.

So this cell shows no cost on Qwen3.8-27B, and it does not show the defect: no burst in it ran
no round on a carried token. By the code (the table above) a recurrent trunk has no reader of
the duplicate on the resume path, and an attention-only trunk loses the resume at the parked
end. No attention-only model with an MTP head was on either device, so that miss is read from
the code, not measured.

## Files

- `rtx5090/session-gate-main.log`, `rtx5090/session-gate-fix.log`, `gb10/session-gate-main.log`,
  `gb10/session-gate-fix.log`: the gate on the two arms.
- `rtx5090/flood/`: `resume.jsonl`, `ctx.jsonl` and the two server logs of the serving cell in
  its four-client form (main 031f146690's server). `ctx.jsonl` is the same pressure on an open
  request in a 640-token window: 18 tokens in 90 s.
- `rtx5090/binaries.sha256`, `gb10/guard-*.log`: the gate binaries.
- `tools/noround_cell.py`: the serving cell, in its last form (the long request runs alone
  first, then two clients for a window). The rows above come from its four-client form:
  the same requests, with four clients from 0.5 s on and no window.
