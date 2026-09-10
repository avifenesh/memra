# GLM shared-expert dual, 2026-09-09

Plan reading before implementation:
  The F16 RESULTS next-exact plan means pairing the shared expert gate/up
  projections through existing matmul_decode_exact_dual at t=2..4, then the
  unchanged activation, down projection and routed-output add. It does not mean
  batching a new down kernel or reviving PR221 side-stream overlap (killed at
  c1, -1.07%). Only the unsharded GLM verify-rows call is eligible; t7 stays
  current as a control. Door MEMRA_GLM5_SHEXP_DUAL default OFF, decide-by
  2026-09-23. Exact projection and composed-chain bytes, finite values, band and
  row argmax are mandatory on all42 layers at t2/4/7 before any timing.
  Replay the archived F16 t2/t4 input and route bytes; capture t7 with the same
  mint, p4k prefix and native short-prime recipe. These are component inputs,
  not production DFlash2-round captures. Warmed ABBAx5 complete-chain timing;
  48/162,103/162,11/162 weights, all42 layers once. Owner's >=0.5ms weighted
  saving supersedes the earlier >=0.3ms eligible-round target. No rig cargo;
  only assigned qualification B200, one GPU phase behind the common lock.
  rev: 2026-09-23

Concluded NEGATIVE against the owner's fixed weighted threshold.
See RESULTS.md. The door and its executable code are removed.
rev: 2026-09-23
