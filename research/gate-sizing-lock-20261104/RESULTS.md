# Host-spill gate sizing and gate locks under gpulock (2026-11-04)

Two gate defects found while running the wave-4 stack-top cells on the RTX 5090 Laptop.

1. `kv-host-spill-identity-gate.sh` and `kv-host-spill-failure-gate.sh` set the device prefix
   budget to a fixed 1024 MB, meant to hold one seed entry and not two. Their prompts tokenize
   to about 102 tokens and the server captures them at the 64-token spec boundary, so on
   Qwen3.5-9B an entry is 53.8 MB and 1024 MB holds both. No eviction fired, nothing reached
   the host tier, and the identity gate failed 6 checks. The pre-stack binary (fe262e905b)
   fails the same 6 (`raw/identity-fixed-1024-pre-stack.log`), so the stack did not cause it.
   With the variable unset, both gates now run one sizing boot, read P_A's entry size, and set
   the budget to 1.5x that entry: 81 MB on Qwen3.5-9B and 243 MB on Qwen3.8-27B with its
   DFlash2 drafter (161.4 MB entries).
2. `dspark-serve-smoke.sh` and `spec-cache-mixed-gate.sh` defaulted to a third lock name
   (`/tmp/memra-serve.lock`) and wrapped every boot in a fresh `flock` on `MEMRA_GPU_LOCK`.
   Under `gpulock run`, which holds the 5090 lock on FD 9 and exports that path, the boot
   waited 300 s on the caller's own lock and the server never started
   (`raw/dspark-smoke-held-lock.log`). Both now default to `/tmp/memra-5090.lock` and boot
   under the inherited lock when FD 9 owns it.

| cell | before | after |
|---|---|---|
| identity gate, Qwen3.5-9B | 6 FAILURE(S) | ALL GREEN, budget 81 MB |
| identity gate, drafter arm, Qwen3.8-27B + DFlash2 | 6 FAILURE(S) | ALL GREEN, budget 243 MB |
| failure gate, Qwen3.5-9B | not run | rc=0, budget 81 MB |
| DSpark serve smoke under gpulock | server died (lock wait) | PASS |

Binary: memra-server from the wave-4 stack top (code-identical to the branch with these
script changes). Single runs. Raw logs in `raw/`.
