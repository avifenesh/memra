Produce and qualify our own calibrated FP4-activation artifact for Qwen3.8-27B prefill on sm_120a. Refs #400, #408, #409 and the closed exact INT8 study #411.

Phase 1 reports the existing mint tensor program, native loader input-scale support, external-checkpoint provenance, calibration corpus and hardware decision, phase/restore consistency, and qualification matrix. Stop for owner steering before minting.

After steering: mint a separately named and hash-locked artifact; add artifact-driven native prefill FP4 dispatch while decode and speculative verification retain the existing q8 activation program; qualify exactness within the new program, paired quality, DFlash2 acceptance, cold prefill, decode, cache and admission. External implementations are offline oracle/calibration tools only. No runtime dependency, release, fleet change or deploy.

Acceptance requires all same-artifact exactness gates, quality within measured noise of the served artifact, and sampled DFlash2 acceptance no worse than 5% relative.

Branch: lane/qwen-fp4-activation-mint-20260909. Worktree: wt-qwen-fp4-mint. Receipts: research/qwen-fp4-activation-mint-20260909/. No local rig gates. GPU jobs use the shared canonical lock and owned PIDs. Pushes use MEMRA_SKIP_PERF_CI=1; hosted CI remains required.
