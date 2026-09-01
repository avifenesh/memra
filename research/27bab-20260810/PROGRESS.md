# Qwen3.6-27B beside Step-3.7 progress — 2026-08-10

Lane: `lane/cx-27bab`

Base: `dc77de733c1615da8a3c93788ee221032ec3fd2d`

Rig: Vast `<vast-box-ip-2>:39411`, 2x RTX PRO 6000 WS Max-Q, CUDA 13.1.

Status: complete. Both servers and the Step soak are live on the Vast host.

## Block checklist

- [x] Setup: verify remote tip, build, exact model/draft paths, launch contracts, and content-sanity prompt.
- [x] A: Step alone with host bounce — short TTFT, 4k TTFT, c=1 and c=4 decode.
- [x] B: Qwen3.6-27B alone on card 0 with its MTP + own-trim drafter — short TTFT,
      c=1 decode and acceptance, c=4 aggregate decode.
- [x] C: both resident — Step under steady c=2 27B chat load, then 27B under a Step 4k prime.
- [x] D: both-resident per-card VRAM and KV/cache headroom ledger.
- [x] E: greedy known-prompt content sanity for every scored server state; stop and restart on
      BOS-garbage.
- [x] Write `RESULTS.md`, verify raw receipts, and leave Step `:8002` plus 27B `:8003` and the
      Step soak running.

## Fixed constraints

- Every Step PP-2 launch sets `MEMRA_PP_HOST_BOUNCE=1`; this Vast host's peer-copy path is unsafe.
- Step launch starts from `/root/serve-env.sh`: PP-2 on devices 0,1, context 262144,
  grouped MoE, prefill tick 2048.
- Qwen3.6-27B is single-card on physical card 0, port 8003, context 32768, speculative serving
  with the staged MTP trunk and own-trim drafter.
- Scored medians are interleaved N=5 unless a cell is explicitly labeled N=1.
- Raw stdout/stderr is tee'd before parsing; failure causes are quoted, never inferred.
- No push, tag, `rustup`, `nsys`, or `cargo fmt`.

## Inbox checks

- Start / progress bootstrap: `~/.lanectl/inbox/cx-27bab.md` was absent.
- Before remote setup: the same inbox path was absent.
- Before each Block A launch/retry: the same inbox path was absent.
- Before Block B: the same inbox path was absent.
- Before each Block C retry and the final-live block: the same inbox path was absent.

## Notes

- The lane began from a clean, dedicated worktree at the exact requested base.
- The personal Qwen3.6 serving notes were consulted only for regime context. Remote artifact names,
  flags, and performance will be taken from this checkout and the Vast box, not reconstructed from
  the older local wrapper.
- The Vast checkout was clean but initially at the host-bounce implementation commit `650a2aec`.
  It fetched only `origin/restructure/public-split`, switched to a dedicated remote
  `lane/cx-27bab` at `dc77de73`, and built `memra-server` with CUDA 13.1. The release binary
  SHA-256 is `de9c201d27993275b1448f778e6942bbaca5e902864e867090366c5fe13087e8`.
- Exact staged artifacts and fresh SHA-256 values are retained in
  `raw/setup/artifact-manifest.log`. The 27B serve command uses
  `Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf` plus the requested
  `draft-daily-owntrim-nvfp4head-q4blk.gguf`; the separately staged full NVFP4 MTP head is
  inventoried but is replaced by the own-trim regime draft on the serve path.
- `measure.py` streams every request, records server-authoritative token/spec counters, and rejects
  empty or special-token-only output before a timing point can pass. `run-vast-block.sh` alternates
  scored cell order, tees raw logs, and fails on a content-hash mismatch or captured CUDA/OOM/panic
  signature.
- The first Block A launch stopped before its first request: the environment receipt's grep pattern
  captured `MEMRA_PP_` literally instead of the required `MEMRA_PP_*` names, so the harness's own
  host-bounce assertion failed. `raw/A-aborted-env-audit/` retains the clean rc=1 startup/shutdown;
  no timing row was emitted. The audit pattern was corrected before retrying from 0 MiB on both
  cards.
- The second Block A launch stopped at its first long-prompt shape check: 136 deterministic filler
  repetitions tokenized to 3,561 Step prompt tokens, below the harness's 3,800-token lower bound.
  `raw/A-aborted-prompt-class/` retains that rc=1 receipt plus one preceding coherent short sanity
  row; neither is scored. The prompt was resized to 157 repetitions (about 4.1k from the observed
  tokenizer slope) before another clean retry.
- The third Block A launch completed all 20 requested cells cleanly, then the post-run failure scan
  matched the normal lowercase startup phrase `fatal Xid [48, ...]` because grep was
  case-insensitive. `raw/A-excluded-scan-false-positive/` retains all N=5 rows and the sole matched
  line, but the rc=1 set is excluded rather than repaired in place. The scanner now matches uppercase
  `FATAL` while retaining the exact lowercase CUDA/OOM/panic/illegal-address patterns; A was then
  re-receipted from a cold server state for rc=0.
- The final Block A receipt passed with rc=0 and coherent known-prompt output. Across N=5, medians
  were 188.4 ms short TTFT, 6.791 s 4k TTFT (4,107 prompt tokens each), 73.74 tok/s c=1 decode,
  and 107.67 aggregate tok/s c=4. Exact per-run summaries, server logs, the environment receipt,
  GPU samples, and raw request JSONL are in `raw/A/`.
- The first Block B pass produced coherent output and timings, but its per-cell speculative deltas
  were all zero while cumulative counters advanced on the following request. The stream closes just
  before request-finalization counters are published. `raw/B-excluded-metrics-lag/` preserves that
  unscored rc=0 receipt; `measure.py` now waits, for at most 10 seconds, until `completed` includes
  the current invocation before sampling speculative counters or passing the cell.
- The corrected cold Block B receipt passed with rc=0, stable known-prompt hashes, and settled
  counters in every cell. Across N=5, medians were 173.5 ms short TTFT (361.3 ms first-load outlier),
  169.21 tok/s c=1 decode at 72.43% weighted speculative acceptance (880/1,215), and 151.49
  aggregate tok/s c=4 at 72.33% weighted acceptance (3,515/4,860). Exact per-run receipts are in
  `raw/B/`.
- Block C's harness runs an untimed known-prompt hash check against both servers after every
  forward and reverse co-serve arm, in addition to the timed outputs' non-garbage checks and the
  initial/boundary/final checks. Any mismatch excludes the complete block.
- The first Block C launch stopped at the start barrier for forward arm r2. Its c=2 client actually
  completed 16/16 coherent requests over 14.61 seconds, eight per worker with overlapping request
  intervals, but the single-card scheduler never happened to publish `active_sessions=2` during a
  50 ms poll. `raw/C-aborted-active-snapshot-gate/` retains the rc=1 receipt; it contains no CUDA,
  OOM, panic, or illegal-address signature. The c=2 client contract is unchanged. The overlap
  barrier now requires one observed in-flight request while both continuously replenished client
  workers remain live, preventing a transient server snapshot from consuming the whole load window.
- The second Block C launch completed all five forward pairs and their hash gates, then stopped on a
  real Q27 allocation failure before any reverse-prime point: `DriverError(CUDA_ERROR_OUT_OF_MEMORY,
  "out of memory")`. The failing receipt had 76 speculative-pool entries, 249,888,768 bytes of
  driver-free memory before the request, three recorded OOM parks, and N=1 request failure. GPU
  samples and both resident process identities are retained in `raw/C-aborted-ephemeral-session-oom/`.
  The duration client had generated a new session id for every turn (76 live entries, zero hits),
  which does not model steady c=2 chat. The scored retry keeps exactly two persistent background
  session ids while retaining unique cache salts, and the after-arm known prompts reuse one sanity
  session. The high-fanout OOM remains a reportable capacity boundary rather than being discarded.
- Cross-process request rows now include host-monotonic start/end timestamps. Forward Step probes
  must overlap the c=2 Q27 intervals, and reverse Q27 probes must overlap the Step 4k interval; each
  overlap is asserted and printed into the raw driver receipt.
- The third Block C launch proved that stabilizing only the request body's `session_id` does not
  provide speculative-pool affinity on this server surface: the two background session ids were
  stable, but unique `cache_salt` values again produced 76 entries, zero hits, and the same quoted
  OOM at the same post-forward gate. `raw/C-aborted-session-id-not-affinity/` retains that rc=1
  receipt and all five successful forward overlap assertions. The steady-c=2 retry stabilizes both
  `cache_salt` and `session_id` per worker. Q27 is launched with `MEMRA_PREFIX_CACHE_MB=0`, so this
  cannot turn the scored load into a prefix-cache benchmark.
- The final Block C passed rc=0. All 15 host-monotonic overlap assertions passed; all 86 C summary
  files had settled counters, zero request errors, and zero BOS-garbage, including 56 exact
  known-output hashes. The c=2 Q27 load used two namespaces and reported zero cached prompt tokens.
  Step short-TTFT median was 734.9 ms active versus 188.4 ms alone (+290.2%), far outside the
  suggested 5–8% bound; Step c=1 decode fell from 73.74 to 23.12 tok/s (-68.7%). Resident-idle Step
  was neutral (184.8 ms, -1.9%). Q27 delivered 117.58 aggregate tok/s at c=2 while contending, but a
  Step 4k prime raised Q27 TTFT 112.3% and cut Q27 decode 56.1%. Both-resident after-run free VRAM
  was 26,077 MiB on card 0 and 39,628 MiB on card 1. Exact derived values are in `summary.json`.
- `RESULTS.md` records the resident-standby-only verdict, standalone Q27 listing values, both-way
  interference, VRAM ledger, bounded-session capacity warning, recommended launch/admission shape,
  and the current flash-class output-token economy reference.
- Final-live receipt `raw/final/` passed rc=0. At 2026-08-10T17:39:50Z, Step PID 69696 was ready on
  `:8002` with PP-2 host bounce, Q27 PID 69935 was ready on `:8003` with card-0 K=3 speculation,
  and soak PID 70049 had completed nine new Step iterations with 97 chunks each and empty errors.
  Both live server logs had zero captured failure signatures and both metrics surfaces reported zero
  OOM parks.
