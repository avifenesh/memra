# v0.137.0 owner-go qualification

Owner go includes #378, merged at `ff0937dd09b808cd97dde174de1b6aedff63a0b8`. Release branch rebased onto that exact main. No other main change entered this rebase.

GLM TP-2 gains opt-in GPU sampling (#378, merge `ff0937dd09b808cd97dde174de1b6aedff63a0b8`, `MEMRA_GLM5_TP_DEVICE_SAMPLE` default OFF, decide-by 2026-09-22). Five p32k OFF/ON pairs across two windows measured median wall throughput 81.32 -> 91.25 tok/s and server decode 11.89 -> 10.61 ms/token; 160-token greedy/top-k-one twins were byte-identical. Unset/0 with a restart restores host sampling. [Sampler receipts](../glm5-tp2-gpu-sampler-20260908/RESULTS.md).

The full fixed-roster build and battery run on the assigned non-production single B200 with MEMRA_CUDA_ARCH=100a and MEMRA_GPU_LOCK=/tmp/memra-gpu.lock. Exact head, raw logs and tails are attached to PR #391. A final battery and release guard also run on the resulting squash before tagging. No Cargo or gates run on the rig. Pushes use MEMRA_SKIP_PERF_CI=1.

The release does not deploy a fleet or claim released-composition TP-2 host-cache qualification. Those remain the separate deployment step.
