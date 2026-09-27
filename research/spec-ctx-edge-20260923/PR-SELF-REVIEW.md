# Self-review: memra#659 (no speculative round writes past the session cache)

Author's review of the full diff `main..lane/spec-ctx-edge-20260923`, posted as a PR comment per the owner rule.

## What the diff is
- `crates/memra-engine/src/spec.rs`: the qwen round loop's first statement ends the burst when
  `cache.pos + k_this + 2 > cache.max_ctx` (the pending token and up to `k_this` drafts, plus the sampled tail's
  bonus row; glm5's guard form). The round-stream arm is chosen only with room for its M rounds. The verify funnel
  `decode_step_t_core_stream` refuses a host-positioned window past the cache with a typed error before the PP
  dispatch. A CPU source census, `ctx_edge_659_census`.
- `crates/memra-engine/src/gemma_spec.rs`: the same guard, `cache.pos + kr + 1 > cache.max_ctx`, at the top of both
  burst loops (greedy and sampled; neither has a tail write).
- `crates/memra-server/src/worker.rs`: `request_budget` is pulled out of `prepare_request`. On the open-output door
  arm the budget is the charged output `v`, not the whole cap `v + 8`, so the 8 slack rows stay free as on a bounded
  request. Every other arm is unchanged, with a unit test per arm.
- `tools/spec-ctx-edge-gate.sh` (new), wired into `tools/local-ci.sh` behind `MEMRA_CI_SPEC_CTX_EDGE`; FLAGS rows (the
  new CI switch, and a sentence on the door row), a TESTING paragraph, research receipts and the INDEX row.

## What I checked
- The worker's between-burst guard fires whenever the engine guard ends a burst, so the new exit never spins.
  qwen: the engine stops at `pos + k_this + 2 > cap` with `k_this <= k`, and the worker stops at
  `committed + k + 3 >= cap` with `committed == cache.pos`. gemma: `pos + kr + 1 > cap` against
  `committed + k + 2 >= cap`. In both, the engine condition implies the worker's.
- No bounded request stops earlier than before for `k <= 7`. A bounded session's cap is `P + max_tokens + 8` and
  `pos <= P + budget - 1` inside a burst, so the qwen guard trips only for `k > 7`. At `k = 8` the existing worker guard
  already ends the request first. The gemma guard needs `kr > 8`.
- The break sits before any state of the round is touched (snapshot, scratch `set_len`, draft), which is the same
  point where the loop condition exits for the admission yield. The session tail after the loop is unchanged. The
  sampled tail's bonus write at `cache.pos` is covered by the guard's extra row.
- The round-stream arm is opt-in (`MEMRA_SPEC_STREAM=1`) and non-session. Its bound only moves an otherwise
  overflowing round to the eager arm.
- The verify refusal checks `cache.pos`, which is where host-len appends land. It never reads positions in stream mode.
- One numeric program per request: no token moves. The gate's plain message equals the spec message
  (`341ddf3169a9d21a` on both). `run-spec` K=1..8 on the 9B is 8 of 8 PASS.
- Red and green: the unfixed tree fails 10 verdicts through the #87 NaN trap. The fix is ALL GREEN twice on one
  binary. Receipts are in `research/spec-ctx-edge-20260923/`, pre-registered before any run.
- CPU: the census (3), the budget test, clippy `-D warnings` on engine and server with all targets, fmt,
  check-flags and the docs census.

## Limits
- One card and one model (the qwen MTP route on the local 5090). The gemma guards are checked by the CPU census
  only; this rig has no gemma model.
- The door-ON change is client-visible: an open request emits exactly `MEMRA_ADMIT_OPEN_OUTPUT_TOKENS` tokens where it
  used to emit `v + 8` (and then crash at the cap). A door-OFF speculative runaway ends `ContextFull` one to `k + 2`
  tokens short of `MEMRA_CTX`; the plain route still ends at the cap.

## Push regime
The branch went up with `MEMRA_RELEASE_QUALIFICATION_MODE=development` (announced, logged) because the release
qualification gate refuses engine-source ranges on a topic branch as `UNQUALIFIED`. Every other hook ran and
passed. Revuto: if capped or unavailable, this comment is the review.
