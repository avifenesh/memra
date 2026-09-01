# FA3 softmax/PV pipeline progress

Date: 2026-08-13
Branch: `lane/cx-fa3softmax`
Base: `v0.81.2` (`18885ec479d897a3e8c42b0d408a71fa3edaa708`)
Rig: local RTX 5090 Laptop GPU, owner-imposed 210--1200 MHz thermal cap

## Objective

Profile the cached-prefill `fa_prefill_qw_db` serving path at the frozen 4,860-token shape for both served models. Only if the profile shows a material serialized QK -> softmax/P-restage -> PV bubble, test a minimal FA3 Algorithm 2-style software pipeline that preserves the existing numeric class.

## Non-negotiable boundaries

- Profile first; a small or absent bubble is a complete `NO-GO` result.
- Timed and profiler runs take `/tmp/memra-5090.lock` and start only after an idle `nvidia-smi --query-compute-apps` check.
- All local numbers are relative-only under the 210--1200 MHz cap; no absolute-throughput claim.
- Do not touch decode T=1, the live serve box, generated perf boards, tags, merges, or remotes.
- Do not commit `.ncu-rep`, `.nsys-rep`, `.qdrep`, or SQLite profiler captures. Retain only raw console logs and extracted tables.
- Any bit difference stops the lane. A scheduling change that alters accumulation order is a new numeric class and is not silently accepted.

## Starting state

- Worktree was clean and isolated on the requested branch.
- `HEAD` exactly matched the `v0.81.2` tag.
- CUDA 13.1 `ncu` and `nsys` are installed.
- Initial GPU check showed no compute applications.

## Planned evidence sequence

1. Resolve the two served manifests, frozen prompt, and exact cached-prefill invocation.
2. Capture locked `nsys` timelines and targeted `ncu` metrics for both models; store profiler reports outside the repository and commit only sanitized console logs/extracted metrics.
3. Decide whether P restaging / softmax serialization is material enough to justify code.
4. If justified, implement the smallest pipeline, build separate baseline/candidate targets, and run every exactness cell before timing.
5. Record interleaved N>=5 prefill and cold-TTFT evidence, or stop with a profile-backed `NO-GO` / `NEEDS-PRO-RIG`.

## Status

`COMPLETE; NO-GO`

- Built the `v0.81.2` baseline in the isolated target directory
  `/home/avifenesh/.cache/memra-targets/cx-fa3softmax-base-v0812`.
- Captured locked Nsys and NCU evidence for Q27 and Q35 at the exact cold 4,860-token request.
- Profiler reports remain outside the repository under `/tmp/memra-fa3softmax-20260813`; only
  hashes, logs, and extracted CSV tables are retained here.
- The profile found 8.33% occupancy, 0.12 eligible warps/scheduler, 29--34% tensor-pipe activity,
  and 3.84--3.94 short-scoreboard cycles per issued instruction. QK and PV account for roughly
  three quarters of sampled cycles; shared wavefront traffic is 6.42--6.45x ideal.
- Decision: implement the minimal two-cohort, double-P-stage candidate while preserving every
  warp's tile traversal and accumulation order. See `PROFILE.md` for the full gate evidence.
- Implemented and separately built that candidate. It staggered two warp cohorts by one KV tile,
  double-buffered P, and used generation-tagged named-barrier handshakes. The exact patch is
  retained as `candidate.patch`; the production source was restored after the negative result.
- Exactness stayed green: both required manifests, both full model manifests, both run-gen
  checks, both K=1..8 run-spec sweeps, Q27/Q35 chunk invariance, and the frozen actual-shape
  output hashes all passed without a differing bit in the bit-identity cells.
- Completed N=5 per arm/model in one locked, strictly interleaved thermal window. Under the
  210--1200 MHz cap, paired median prefill rate changed -0.904% on Q27 and -1.518% on Q35;
  paired median cold TTFT changed +0.899% and +1.544%. Both losses exceed per-arm spread.
- Verdict: `NO-GO`. Do not promote or send this formulation to box1. `RESULTS.md` contains the
  tables, raw-log map, numeric boundary, and final disposition.
