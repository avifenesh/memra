# Qwen3.6 mixed-cache c=4 regression

Tracking #777. The harness correction is already in merged PR #892.
This receipt reruns its owed native workload on source
`2873dd4ca37faa15cd4261e7b398926cced30106`.

The artifact is `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` at revision
`5bc3e238d916f48a861bac2f8a1990a0e9b7e98d`, file
`Qwen3.6-35B-A3B-UD-IQ4_XS.gguf`, SHA-256
`df27a780435b7b45c2597536112ea3cb091f8544c3d0c3318d9f4258b31f7adf`.
The actual local file was hashed before the native build.

The cell retains serve-smoke section 12's shape: c=4, 20 requests,
18 cache hits and two cold misses, 4860 prompt tokens and exactly 60 output
tokens each. The server context ceiling is 8192; the frozen harness passes
`max_ctx=4860+60+8=4928` on every request. GDN grid 32 derives a
4832-token cache entry independently of
the reported usage. `run-live.py` preserves the invocation, launch settings,
raw response/usage records, binary identity, and 250 ms GPU observations.
The source has no harness or runtime modifications for this run.

The 13 CPU tests pass, including the stale-expectation red control and the
separate token, usage, seed, and accounting failure classes. Their output is
in `cpu-tests.log`; CPU success does not establish the live result.

Hardware preflight: one RTX 5090 Laptop GPU, 24463 MiB total VRAM. The
16.96 GiB artifact size is not a peak-memory estimate. The native cell has
an exclusive GPU lease, a 1200-second limit, a 4-core CPU quota, two compiler
threads, a 24 GiB memory-high limit and a 32 GiB memory-max limit.

Live result: **PASS**, one native run. All 20 requests returned exactly 60 tokens
with `finish_reason=length`; 18 hits restored 4832 tokens and both cold misses
restored zero. The live summary reports no failure classes, seed failures,
golden mismatches, accounting drift, or carried-prime-batch violation.

`live/gate.jsonl` contains the raw receipt. Replaying it under the current law
passes. The offline red control restores the old 4860-token expectation and
fails as `usage_mismatch` only, with no token regression. See
`live/replay-green.json` and `live/replay-stale-expectation.json`.

At 250 ms sampling, GPU memory peaked at 20249 MiB, minimum free memory was
3736 MiB, and maximum observed GPU temperature was 81 C (197 samples).
`live/gpu-after-cleanup.txt` shows no remaining inference process. These are
one-run resource observations on an RTX 5090 Laptop GPU, not a throughput
comparison or another hardware target's qualification.

The original acceptance shape was not reduced. This completes the named #777
local regression. Other harnesses changed by #892 retain their separate live
qualification requirements; no broad model or target support state changes.
