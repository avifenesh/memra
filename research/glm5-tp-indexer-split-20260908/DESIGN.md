# GLM-5.3-Flash TP-2 pool-split indexer

Implementation baseline: memra `15f4bf96f` (origin/main, 2026-09-08; advanced from `001c09e5d` before remote build).
Branch: `lane/glm5-tp-indexer-split-20260908`.
Status: paired f32 model identity PASS; prime wins at 128k/1M, decode NEGATIVE at 128k and FLAT at 1M. Merge cost is the decode blocker. Owner retains the code with the door OFF. Complete rows and kernel tallies: [RESULTS.md](RESULTS.md).
No cargo, CPU/GPU test, benchmark or smoke server was run on the rig.

## Selection and cost

Choose local top-k candidate exchange over raw score exchange. For P pools and k selected
pools, each rank sends k * 8 bytes/query versus ceil(P/2) * 4 raw-score bytes/query. GLM's
index_topk=2048 is a RAW TOKEN budget: pool=4 makes k=512. At P=250,482 (the 1M trace),
that is 4,096 bytes versus 500,964 bytes, 122.3x less per rank/query/layer. At the conservative
k=2048 ceiling it is 16,384 bytes, 30.6x less. At 128k (P=32,212), the GLM payload ratio
is 15.7x. These are byte counts, not measured latency or bandwidth claims.

Each rank computes all query rows over its contiguous pool half; rank 0 owns ceil(P/2).
Pool-key pointer offset and subtracting offset*pool from first_pos reuse the same f32 scorer,
including the head-blocked decode RP arm and tiled prime arm. The score arithmetic is unchanged;
negative relative positions mask the peer's not-yet-visible pools during early prime chunks.
State and complete-pool key appends stay replicated. TC levels 1 and 2 passed the scorer range
bit-identity gate on the target pair, including early-prime causal boundaries; both are now admitted
by the experimental door. Full-model TC composition remains pending.

The existing local selector is invoked with pool=1 and always_tail=true: its tail is then empty,
and its output is ascending local pool ids. Adding a constant offset preserves its tie order.
Pack original score bits and global ids. The two-device gather reads both rank buffers after
MemraArSignal start barriers and protects their lifetime with exit barriers. A local bitonic
merge over at most 2k exact keys finds global top-k, then sorts selected pool ids ascending.
The extra work is pack + exchange + merge; nsys must price it against the saved scorer/selector
work. At small P the payload argument alone does not predict a win; this remains default OFF.

## Exactness contract

The existing selector uses descending finite score, then ascending GLOBAL pool id. Signed zeros
compare equal (canonical +0 in the order key). NaN and both infinities are skipped. The 64-bit
key is desc32(score) in the high word and the global pool id in the low word. Each id is unique.
A candidate absent from its rank's local top-k already has at least k candidates ahead of it
in that same global total order, so it cannot belong to global top-k.

Output is selected pools ASCENDING by id, each expanded to its raw token rows, followed by the
query's incomplete causal tail, then -1 padding to the replicated width. Preserving membership
alone is insufficient because attention walks the emitted order. CHECK compares the entire
plane on BOTH ranks, with host readback only in diagnostic cells.

## Runtime and capture boundary

The same existing door now selects pool splitting instead of the older prime-only query split.
It reaches `mla_tp_attn_cached`, `mla_tp_attn_cached_sym`, and `sym_mla_mid_eager`. The latter
consumes the PRE graph's existing stable workspace and returns that SAME workspace before the
next PRE replay. It writes attention output to the existing FFN graph handoff buffers.

The middle stays eager. Score lengths, pool halves, select-k, and tail width derive from the
host cache length; baking them into a captured piece would replay stale geometry. This lane
adds no live-position middle twin. The exchange itself is device-signalled and contains no
host synchronization or cross-stream CUDA events. Full capture has not been attempted or
claimed; PRE/FFN and KDA graph replay must remain engaged in the on-box receipt.

All admission checks run before either cache is moved or a barrier is queued. Unsupported
shapes stay replicated: missing/mismatched indexers, invalid bounds, fewer than two pools,
k outside 1..2048, non-peer-access/two-device groups and rows-exact verification.
Odd P is supported. Once work starts, allocation/CUDA failures remain errors; a barrier timeout
traps to prevent unwritten candidate memory from reaching attention. No retry of an already
mutated cache is attempted. Both placeholders are allocated before moving either cache; PRE
workspaces and resident planes are restored after a fallible middle.

No change to Glm5TpGlue replication or the by-name refused-door list is required: the indexer
weights and state remain replicated; only score/select work is partitioned.

## Original gate plan and historical checkpoints

The target was verified non-production and cudaMalloc succeeded on both B200s before staging.
The shared instance was subsequently stopped while this lane waited for the GPU lock. The
orchestrator owns resumption/replacement. Do not use the serving pair or the local GPU.
Record target/source SHAs, binary hash, model artifact/prompt hashes and env per run.

Build with `MEMRA_CUDA_ARCH=100a`, nice 19, cargo -j 16. Run cargo fmt --all -- --check,
clippy with -D warnings, relevant engine unit tests, and the new test target's CPU tests.
Run `glm5_tp_indexer_split` ignored GPU tests with --test-threads=1 under
`MEMRA_GPU_LOCK=/tmp/memra-gpu.lock`, including scorer RP=1 and RP=2 in separate processes.
Run the existing dsa-select-gate and docs registry/flag census. Keep compilation out of timing.

Then run the tpprime5 probe shape at 128k and 1M using staged prompts. Run 160-token decode
OFF/ON in interleaved x3 at each context and retain every id, tape, raw output and log. First
use CHECK for exact full-index diagnostics, then unset CHECK for timing. Pin an unmodified
baseline build as well as the split build's OFF arm; compare all 160 ids/tape to that baseline.
Retain nsys per-rank score/select/pack/exchange/merge totals and graph engagement. Greedy is
the bounded identity instrument only; repeat vendor-default sampled shape and the eight-turn
cache-on continuation for serving decisions, with spec-engagement receipts where applicable.

| Context | Prime OFF/ON | Decode OFF/ON x3 | IDs vs baseline | Indexer nsys totals |
|---|---|---|---|---|
| 128k | pending | pending | pending | pending |
| 1M | pending | pending | pending | pending |

Commit with the required Claude-Session trailer after target-box checks. Push only after the
box build/tests pass using MEMRA_SKIP_PERF_CI=1. The local pre-commit invokes cargo fmt, so
committing is deferred to the remote validation step instead of invoking cargo on the rig.
Draft PR body is staged in PR.md; its last line is the required session URL. No push or PR yet.

## Validation observed before tune-host shutdown

- Release probe and dsa-select-gate built with CUDA 13.1.115, sm_100a, nice 19, -j 16.
- Remote cargo fmt check and scoped clippy -D warnings passed, including the profiling helper.
- New CPU tests: 3 passed. Existing glm5_tp preflight CPU tests: 4 passed.
- First GPU pass: both merge/select and range-score tests passed with RP=1 and RP=2.
  The range-score test also passed with TC=1 and TC=2, six successful test executions total.
- Flag table census: 39 tables, 862 rows, all rows match their headers. Runtime census:
  838 literal reads, no uncovered runtime names.
- Unmodified-main control probe built at 15f4bf96f; SHA256
  36ec68d96411269a84e281a9a510310a6f959cf2fe0cfe7cdf3160ba71869aa4.
- Local custody: receipts/build-cpu-flags.log. Per-test GPU logs and the final validation log
  remain on the stopped tune host; recovery is pending. No full-model CHECK or timing row ran.
- No commit, push or PR yet. Resume the queued final gates and model CHECK first; then run
  the rig-side cell controller. Do not interpret the pending matrix table as a result.

## Resume after container restart

Both artifacts survived; remote cargo build reported up to date. A fresh cudaMalloc on each
rank succeeded and all six GPU gate executions passed again. Their raw logs are now in local
custody under receipts/gpu-rp1.log, gpu-rp2.log, gpu-tc1.log and gpu-tc2.log.
Lane probe SHA256: f525178a6cdf3e38070ffc095f310d97ded17fd3dc98ffad375bcf18ee2fd8eb.
The orchestrator requested deletion of the old unlocked lock file once; this was done only after
lslocks showed no holders. All subsequent GPU execution uses flock on /tmp/memra-gpu.lock.

Cell output now lands at the remote worktree's receipts/<cell>/ directory. A controller on the
rig retrieves each completed cell immediately before dispatching the next, so an interruptible
host cannot strand a whole campaign's completed rows. CHECK cells may run alongside compilation
because their timing is not scored; timing cells wait for compiler processes to finish. Each
cell includes its own env, binary/prompt hashes, exit, UTC bounds, GPU metadata, rows and ids.

## Second interruption

The 4k TC=1 CHECK completed: 3,766 prompt tokens, 160 generated tokens, full merged index
rows byte-identical in prime/decode, 12 graph pieces / 56 segments, 11 eager MLA middles and
zero whole eager layers. The raw cell was retrieved immediately into
receipts/check4k-tc1-r1/. Its timing is diagnostic (CHECK enabled), not a performance row.

The 128k CHECK was queued on the shared lock when the instance stopped again. It has no result.
A restart request was made; the provider returned that resources were unavailable and queued
the state change. Long-context cells and both PRs remain pending. The controller detects a dead
cell PID after restart, archives any partial receipt, and retries only that interrupted cell.

## 128k resume checkpoint

The 128k CHECK completed and its raw files were copied to the rig: 128,847 prompt tokens,
160 output tokens, exact merged index rows, 12 graph pieces / 56 segments, 11 eager MLA
middles and zero whole eager layers. The unmodified-main control completed at 36.9209 s
prime and 67.457 decode tok/s (single greedy instrument row). Its 160-id file is byte-identical
to the split CHECK file; SHA256 308075f01e83ff9fce016c5ff5976e9b0578c287de7419edca941852edf7c087.
Receipt: receipts/check128-vs-main.json; raw cells: check128-f32/ and main-128k/.
CHECK timings are not included in performance comparisons. Timed OFF/ON pairs remain pending.

The instance stopped again during pair1-128k-off after the main-128k control was safely
retrieved. No completed lane OFF row was retrieved. The last observed phase was model loading.
Per the latest owner instruction, completed rows are preserved and the stop is reported;
no restart is attempted in this window. Resume the rig-side controller after the target is
available: it will keep the completed CHECK/control rows and retry the interrupted OFF cell.
