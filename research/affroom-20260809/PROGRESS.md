# plain-affinity parked-cap growth — progress

## 2026-08-09 intake

- Branch/worktree: `lane/cx-affroom` at base `d2d6e6d1` after the request-owned context-charge
  merge (`02d67866`).
- Steering check: `~/.lanectl/inbox/cx-affroom.md` is absent at intake; re-check every work
  block.
- Live failure: nine growing pi chat requests produced zero plain-affinity rewinds because each
  parked cache was sized for the preceding request (`cap < next need`). Near-turn growth missed
  by tens to hundreds of rows; a large paste missed by about 8k rows.
- Scope: make an otherwise-valid nominated plain checkpoint grow to the incoming request's
  required capacity before rollback/resume. Preserve byte-exact checkpoint state, request-owned
  admission accounting, existing cold fallbacks, and unrelated paths.
- No origin push, merge, tag, `rustup`, `nsys`, perf-board edit, or runtime-default expansion.

## Planned proof

1. Trace the plain checkpoint park/nominate/decide path, request-owned charge interaction, and
   the val256 harness shape that already resumed.
2. Add a focused server unit test for the pi shape: park at need `N`, request `N + delta`, and
   require resume rather than a no-room decline.
3. Implement the smallest grow-on-resume path, copying the parked cache byte-exactly and keeping
   allocation/reclaim behavior honest.
4. Run `cargo test -p memra-server`, cache-meter 23/23, serve-smoke, and the affinity 5090 replay
   under `flock /tmp/memra-gpu.lock`; retain raw logs under `raw/`.
5. Record the audited verdict in `RESULTS.md` and stop without pushing.

## Status

Complete: implementation, CPU tests, focused before/after replay, full 5090 affinity replay,
17/17 true-cold teeth, cache-meter, serve-smoke, and the clean owner-pod PP-2 pi shape are
recorded. No push, merge, or tag.

## Owner rig follow-up

- New inbox steering makes the RunPod Step-3.7 box an additional required verification target.
  Keep the owner's port 8002 process untouched; use an isolated worktree, port 8010+, and
  `/tmp/pod-gpu.lock` for every work block.
- Reproduce the live 262,144-context PP-2 arithmetic rather than report another small-context
  proxy: park around 12.6k prompt tokens with a 32,768-token completion allowance, grow by a few
  hundred tokens on the next turn, then grow to about 20.7k after a large paste.
- Acceptance receipt: the test server log must show `plain-affinity` growth and rewinds where the
  owner's pre-fix server logged capacity declines. Co-residency, if any, is recorded as
  `window_clean=false`; this is a correctness receipt, not a performance row.
- Pod access initially lacked a published SSH mapping. It recovered only after an external pod
  restart; both 96 GB GPUs were then empty and no process listened on 8002. No live server was
  stopped or modified. The lane bundle is deployed at commit `875a27f3` in the separate
  `/workspace/wt-cx-affroom` worktree, and a release build is in progress.
- Owner update 2 confirms the serving shutdown was intentional and all pod work is now development
  work. Both GPUs are free under the shared lock, so the receipt uses the full PP-2 shape and marks
  the window clean rather than taking the single-card/co-resident fallback.
- Exploratory full-PP pass retained: Step tokenized the initial deterministic payload at 10,153
  tokens and the paste turn at 16,770, below the owner's target windows, so the shape gate failed.
  The runtime behavior itself was green (three grows, three rewinds, no decline); tune only the
  deterministic note counts to 206 base / 129 paste and rerun against a fresh server.
- Final clean PP-2 pass: prompts 12,654 -> 12,692 -> 12,733 -> 20,759; all three larger requests
  grew and rewound, including the 8,026-token paste. Final metrics report three affinity rewinds;
  server receipts contain no decline or resume failure. The isolated port-8010 server was stopped
  and both cards returned to 2 MiB idle before releasing the lock.

## Diagnosis and implementation

- Box1 val256 resumed because it ran before the ctxcharge repair: every request inherited the
  262,144-token server floor, so its 37,823 -> 47,175-token conversation always fit the parked
  cache. The fixed request-owned sizing exposed the real sequential shape: each finite pi request
  parks at its own `prompt + max_tokens + 8` cap, then the next larger prompt exceeds it.
- Identity and exact bytes now decide before capacity. A valid but short parked cache grows to the
  incoming request's bounded `need` (not the 262,144 server fallback); admission charges the same
  request-shaped maximum of `ctx_cap` and `need`. A right-sized F5 session already covering
  `need` is retained without inflation.
- The restore is shared and PP-aware: each layer's full-attention KV rows and recurrent checkpoint
  state copy through its owning stage engine, then open PP contexts synchronize before publication.
  Spec-affinity uses the same trunk restore and additionally copies checkpoint-valid MTP scratch
  rows; pointer-baking draft graphs are dropped for recapture.
- Commits: `ed1d5a07` introduced plain growth; `aeda0041` corrected the target to honestly charged
  `need`, made restore PP-aware, and fixed the identical spec-affinity capacity mismatch exposed by
  the required serve-smoke arm.
- Focused pi-shape test: previous cap 45,064, incoming need 45,522, exact checkpoint -> grow/resume.
  Full `cargo test -p memra-server`: **157 passed**, 0 failed (156 baseline + 1 regression).

## Final gates

- Unchanged full affinity wrapper, N=3: 11 rewinds in every ON arm, zero named failures.
- Short-window true-cold teeth: 17/17 exact, 11 rewinds, budget green.
- Cache-meter: 23/23 assertions, terminal `0 failed`.
- Serve-smoke: terminal `0 failed`; both fresh-server spec-affinity arms rewound 3 times.
- Focused spec before/after: 0 rewinds + 3 capacity declines -> 3 grows + 3 rewinds + 0
  failures on the same four-turn shape.
- Owner pod, Step-3.7, 262k, PP-2, clean window: exact live-sized shape PASS with 3/3 growth
  rewinds and the cross-device transport banner present.
- An exploratory 256-token affinity-vs-every-tier-cold heuristic remains red on 2/17 outputs at a
  15-character shared prefix. This is outside the requested decisive gate and is not hidden:
  short-window state exactness is 17/17, while no long-window resumed-equals-cold claim is made.
  See `RESULTS.md` for the bounded interpretation and raw paths.
