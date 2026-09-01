# Toolchain A/B: does the container build cost step37 decode? NO. (And where the gap actually lives.)

**Perf-chain cell one.** Question: does the CONTAINER build toolchain (the release-workflow
glibc-floor recipe: ubuntu22.04 + cuda-nvcc-13-1 + rustup stable, which builds every fleet
binary) cost step37 serving throughput versus an ON-BOX native build of the SAME memra
commit? Prime suspect for the ~25% decode gap between the 2026-08-29 box-built seal
(140.6 tok/s median, fingerprint memra-c9a617ca994b) and every container-built binary
since (87-106 on the same protocol).

**Verdict 1 (the assigned axis): the toolchain is NOT the gap.** Same commit, two
toolchains, one box, interleaved fresh-boot x5 per arm: container 116.89 vs native
116.45 wall tok/s (container +0.37%, far inside the 2.2-2.4% within-arm spreads).
A 25%-class effect is excluded by two orders of magnitude.

**Verdict 2 (the pre-authorized alternative axis): the gap is the engine + serving-env
delta between the 140-era stack and the current stack, and it reproduces
box-independently.** On this same bench box, same sealed protocol, interleaved x3:
the 140-era stack (memra c9a617ca994b + its serving env) does **140.95** median,
reproducing the 140.6 seal almost exactly on different hardware and vantage, while
the current stack (memra 3999a92a6e18 + current deploy env) does **115.74** (**-17.9%**).
Decomposition below says the regression is per-verify-round wall time, not acceptance,
and TTFT actually improved 2.3x.

Everything here: pinned commit(s), one bench box, PID-verified arm identity per boot,
`git log -1` in every build receipt, spec-engagement receipt on every measured stream.

## Design

- One memra commit both arms: `3999a92a6e18a231ce8e18fb2b6f37997b00e882` (the
  fleet-convergence pin; public build, no accounting env, the supported shape).
  `system_fingerprint: memra-3999a92a6e18` verified in every streamed response of both arms.
- Arm C (container): built on the build host inside the cached fleet image (the exact
  image generation that builds deployment binaries), `cargo build --release -p memra-server`,
  `MEMRA_NVCC=/usr/local/cuda-13.1/bin/nvcc`, `MEMRA_CUDA_ARCH=120a`, `CARGO_BUILD_JOBS=10`.
- Arm N (native): built on the bench box with its own toolchain,
  `MEMRA_NVCC=/usr/local/cuda-13.2/bin/nvcc`, `MEMRA_CUDA_ARCH=120a`.
- Bench box: 2x RTX PRO 6000 Blackwell Server Edition 96 GB (600 W, driver 595.91.07),
  GPUs verified 0 MiB and no server before every boot.
- Artifact: step37-flash-nvfp4, all 14 LFS files sha256-verified against HF revision
  `4275532ffd9a9496ff36b7a2dc4a9db1048da438` (`receipts/model-sha256-actual.txt`).
- Launcher: byte-faithful port of the production step37 serve launcher minus the metering
  seam (admin/keys/ledger/budgets omitted), loopback bind, prod registry copy for
  vendor sampling defaults (temp 0.5 / top_p 0.9), `MEMRA_STEP_TP=0-44@0,1`,
  MTP heads 3, K=3, spec serving on (`harness/launch.sh`).
- Protocol: the sealed digits protocol, 512-token streamed completions, vendor-default
  sampling (NO sampling params in any payload), banked digits prompt + fresh salt per rep,
  wall clock including TTFT (`wall_tok_s = completion_tokens / wall`), token counts from
  the stream's own usage block, spec receipts from `usage.spec`. Per boot: spec-engagement
  smoke gate, 1 discarded warmup, 8 measured reps. Loopback vantage (both arms identical).
- A/B law: interleaved fresh boots C1 N1 C2 N2 C3 N3; **escalated to x5** because both
  amendment rules fired: (1) within-arm spread of the decision median >0.5% (C 2.19%,
  N 2.37%), (2) verdict delta within 2x pooled spread (0.37% < ~4.5%). Escalation added
  C4 N4 C5 N5.
- Arm identity per boot: fresh `BOOT_NONCE` injected into the server env and read back
  from `/proc/<pid>/environ`, `readlink /proc/<pid>/exe` vs the arm binary, binary md5,
  baked fingerprint (`receipts/boot-*.receipt`); stop = anchored-pattern pkill on this
  lane's binary path only + wait for 0 MiB on both cards.

## Toolchain census

Same commit, same `Cargo.lock`, same 137 crates both arms. See `toolchain-census.md`
for the full table. The axis that actually differs:

| axis | C (container / fleet path) | N (on-box native) |
|---|---|---|
| nvcc / ptxas | CUDA 13.1, V13.1.115 | CUDA 13.2, V13.2.51 |
| host gcc | 11.4.0 (u22.04) | 13.3.0 (u24.04) |
| glibc (link) | 2.35 | 2.39 |
| rustc / cargo | 1.98.0 / 1.98.0 | 1.98.0 / 1.98.0, **identical** |
| binary md5 | dc58e8c52f8d3bce20941fb69736579b | 93db82e0599933ff1af05a201ae3a5c3 |

The census did NOT kill the hypothesis (nvcc differs by a minor CUDA release, and the
140.6 seal's build log records "nvcc auto-detected CUDA 13.2", the same class as arm N),
so the GPU cell ran. At run time both binaries resolve libcudart/libcublas to the box's
CUDA 13.2 lib64 (ldd receipts in the lane log), so the cell isolates compiled-in device
code + host codegen, the axis a fleet-build change could move.

## A/B result (assigned axis)

Decision metric: median `wall_tok_s` of the 8 guard-clean reps per boot; arm value =
median of boot medians. Guards green on all 80 reps: completion 512/512,
finish_reason=length, spec engaged, single fingerprint per arm.

| boot | C (cuda 13.1 container) | N (cuda 13.2 native) |
|---|---|---|
| 1 | 117.44 | 116.48 |
| 2 | 116.89 | 114.20 |
| 3 | 115.06 | 115.25 |
| 4 | 117.61 | 116.95 |
| 5 | 116.57 | 116.45 |
| **arm median** | **116.89** | **116.45** |
| boot-median spread | 2.19% | 2.37% |
| pooled (40 reps) median / mean / sd | 116.89 / 116.16 / 3.00 | 115.68 / 115.54 / 2.88 |
| TTFT median | 0.1784 s | 0.1767 s |
| decode tok/s median | 121.62 | 120.30 |
| spec acceptance median | 0.928 | 0.929 |

Delta C-N: **+0.44 tok/s = +0.37%** (nominally in the CONTAINER's favor). TTFT and
acceptance are indistinguishable. The container toolchain costs nothing on this
protocol; **no fleet-build change is recommended on this axis**, keep the container
(it is also the reproducible, glibc-floor path).

Protocol incident, logged: one N4 boot attempt failed before serving because the boot
script was overwritten by scp while the runner was live (partial-file read, phantom
bash syntax error). No measurement was taken on that boot; the cycle re-ran cleanly.
Lesson: never ship harness edits onto a box while its runner is mid-flight, stage to a
temp name and `mv`, or wait for the cycle boundary.

## Alternative-axis cell (pre-authorized on toolchain death): the era A/B

Arms, interleaved fresh-boot x3 on the same box, same protocol, same artifact:

- **O** = memra `c9a617ca994b` (the 140.6-seal commit) built on-box, + the exact 140-era
  serving env (including the `MEMRA_NVFP4_BANK_V2=1` + `MEMRA_SEL_DOWN8=1` doors the
  2026-08-29 incident later removed; no vision; era registry copy, sampling defaults
  identical to current). `harness/launch-140era.sh`. Fingerprint memra-c9a617ca994b in
  every response.
- **P** = memra `3999a92a6e18` built on-box (arm N's binary), + the current deploy-shape
  env (doors removed, vision armed, current defaults). Fingerprint memra-3999a92a6e18.

Guard-clean result (2 of 24 O reps finished early on eos <512 and are excluded by the
sealed guard; excluded rows are in the raw receipts, flagged):

| boot | O (140-era stack) | P (current stack) |
|---|---|---|
| 1 | 141.40 | 117.00 |
| 2 | 138.14 | 115.47 |
| 3 | 140.95 | 115.74 |
| **arm median** | **140.95** | **115.74** |
| boot-median spread | 2.31% | 1.33% |
| TTFT median | 0.409 s | 0.177 s |
| decode tok/s median | 156.81 | 120.72 |
| spec acceptance median | 0.976 | 0.929 |
| spec rounds median (512 tok) | ~149 | ~148.5 |

Reading:

1. **The 140.6 seal reproduces**: 140.95 on this box, loopback, from the era stack. The
   original seal was through the production edge from a remote vantage; the era stack is
   simply that fast. The gap is therefore NOT the serving box, NOT the vantage, NOT the
   toolchain, it is the engine+env delta, **-17.9%** on identical hardware.
2. **WHERE it lives: per-round wall time, not acceptance.** Both arms complete 512 tokens
   in ~149 verify rounds (~3.44 tokens/round, realized speculation depth is unchanged).
   O spends ~25.0 ms wall per round, P ~29.8 ms (**+19% per round**). The regression is
   the execution cost of the verify/decode walk, not draft quality.
3. **TTFT improved 2.3x** in the current stack (0.409 → 0.177 s). The delta is
   decode-side only.
4. Caveat on O: the era env carries the v2-bank door, which the incident lane showed
   corrupts output text (it was removed for correctness, priced at ~1 tok/s at removal
   time). O's rows measure the speed of the era shape, not a shippable configuration.
   Its 0.976 acceptance (vs 0.929) may be partly corruption-inflated; but even granting
   acceptance parity, the per-round wall gap stands on its own.

## What the perf chain should do next (data-pointed)

- Keep building fleet binaries in the container. This axis is closed.
- The hunt moves to the **engine-code delta c9a617ca994b -> 3999a92a6e18 under a fixed
  env**, and the **env delta under a fixed binary**, in that order:
  1. O-binary + doors-OFF era env (prices the two removed doors directly, the ~1 tok/s
     removal pricing looks too small against this cell's +19% per-round gap and deserves
     re-verification);
  2. P-binary + vision-OFF current env (vision's ~8 GB f32 residency is in the P arm and
     in prod, but was absent from the 140-era arm);
  3. then a commit bisect of the remainder on this protocol (the harness in `harness/`
     boots any binary in ~3.5 min and prices an arm in ~7 min).
- Absolute levels for calibration: current stack ~116 loopback on this 600 W bench box
  vs the production re-seal 99.5-106 through the edge, consistent with vantage cost;
  no residual box anomaly is indicated.

## Receipts

- `receipts/rows.jsonl`, 80 toolchain reps + smokes/warmups, one JSON row per stream
  (tokens from usage, spec acc/rounds, fingerprint, bin md5, boot nonce).
- `receipts/rows-era.jsonl`, 48 era reps + smokes/warmups (2 guard-violating rows flagged
  by `full_tokens:false`).
- `receipts/progress.txt`, the full interleave timeline, boot receipts inline, the
  escalation declaration naming the fired rules, the N4 incident.
- `receipts/boot-*.receipt`, per-boot arm identity: nonce, md5, `/proc` exe + environ
  nonce, `git log -1` of the build checkout, GPU snapshot at ready.
- `receipts/model-sha256-actual.txt`, 14/14 artifact shards vs the pinned HF revision.
- `logs/container-build.log`, `logs/native-build.log`, `logs/old-build.log`, the three
  builds (each 137 crates; nvcc provenance lines; `git log -1` receipts).
- `logs/server-*.log`, every boot: model load, admission calibration, spec engagement.
- `harness/`, launcher (prod-shape minus metering), 140-era launcher, boot/stop with
  PID-verified identity, the digits client, the interleave runners, the digits prompt.
- `toolchain-census.md`, the full toolchain table and its provenance.
