# MEMRA_SERVE_DEVPENALTY qualification on RTX 5090 / Qwen3.5-9B (memra #429)

VERDICT: WIN, exactness intact. The door's default flipped ON for the qualified pair
(`Arch::Qwen35` on a `120a` build) in this lane; every other (model, board) pair keeps the
OFF default. `MEMRA_SERVE_DEVPENALTY=0` is the rollback seam and overrides the default
either way.

## Question

Issue #429: Qwen3.5-9B NVFP4+MTP on one RTX 5090 does not scale past c=1 on the vendor
non-thinking shape (`temperature 0.7, top_p 0.8, top_k 20, presence_penalty 1.5`), exactly
the signature `MEMRA_SERVE_DEVPENALTY` exists to fix: concurrency sheds spec rows
(`K=0 source=concurrency`, the #266 guard, working as decided) onto the plain batched path,
where the host penalty sampler's O(n_vocab) hash-and-sort runs per token. Every prior
receipt for the flag was Qwen3.8-27B on RTX PRO 6000; the 5090 and the qwen35 class were
both unqualified.

## Pins

- Box: vast instance 50430170, label `hebrew-agentic-devpenalty-20260909`,
  RTX 5090 32 GB, power limit 575 W, `$0.649/h`. CUDA acceptance (cudaGetDeviceCount + 1 MiB
  cudaMalloc + memset, sm_120a) ran before staging (`cuda-accept.sh`, output in
  `receipts/cuda-accept.txt`).
- Cell binary: `target/release/memra-server` sha256
  `02316c7a8181563e1c190f3f6457cd7da9ac39a3dce9d0300fbc7e051b7f998b`, built from memra
  `4e107318e` (main tip at lane start) with the lane checkout, plain `cargo build
  --release` (build.rs default arch 120a). All six boots of the cell ran this one binary;
  the boot nonce receipt hashes the exe per boot.
- Flip-smoke binary: same tree rebuilt after the cell with the default-flip diff, sha256
  `981a69b89dce547d90f817d631a8616bdc4ef0dd6e8559cf1d7bd1476295568c`. Only memra-engine and
  memra-server recompiled. The off-b4 smoke proves the arch const resolves `120a` in this
  build behaviorally: the class default can fire only when `BUILT_CUDA_ARCH == "120a"`.
- Model: `Qwen3.5-9B-NVFP4-MTP-GGUF.gguf` sha256
  `52c9cceb190055e0591a9a30c21f7200572eaf3ff1c59f6e9a1eda838a8f39de`
  (`Qwen/Qwen3.5-9B` revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`), metadata
  `scripts/qwen35-9b.models.toml` (vendor recommendation resolved server-side).
- Serving env: `MEMRA_COMPAT=openai`, `MEMRA_PORT=8080`,
  `MEMRA_NONSTREAM_DEADLINE_GATE=0`, `MEMRA_TIMEOUT_MS_MAX=1800000`. The ONLY difference
  between arms is `MEMRA_SERVE_DEVPENALTY`: absent in the off arm, `1` in the on arm.

## Method

- Interleaved counterbalanced A/B: three boots per arm, order off/on, on/off, off/on, five
  repetitions per cell per boot (`scripts/run-cell.sh`, `scripts/boot-block.sh`). Same
  binary for both arms; the arms differ only by the env var.
- Arm identity is read off `/proc`, not asserted: per-boot nonce records pid, start ticks,
  exe path, binary sha256, and the process's `MEMRA_*` environment; the block script
  asserts `devpenalty_env` matches the arm (`<unset>` vs `1`) before any probe runs
  (`receipts/raw/*-boot-nonce.json`).
- Workload: the office page-task from darklanes
  `research/hebrew-agentic-base-20260909/perf-page/`: turn 1 prefills a 2-page Hebrew
  document inside an English JSON tool-result envelope (2,465 prompt tokens) and decodes a
  200-token Hebrew summary; turn 2 reuses the cached prefix and decodes 60. Client sends no
  sampling field; `enable_thinking=false`; the vendor non-thinking recommendation is
  resolved server-side. Think-off, vendor-default sampled, per the serving laws.
- Controls: the same c-sweep with explicit `presence_penalty: 0.0` (separates the penalty
  cost), and the pp512/tg128 twin at c=1 with explicit `top_p 1.0, top_k 0` (the
  truncation-filter control; pure-temperature regime).
- Exactness instrument: greedy byte-identity per boot in two penalty shapes (pp15 = the
  vendor's 1.5, pp00 = explicit 0.0). Greedy penalties stay host-side by design in both
  arms, so identity here proves the door changes no greedy token; the sampled path's
  contract is the distributional oracle (`sample-check`) plus the serving gates.
- No cross-host timing: every number below is from this one box.

## Results

Aggregate = measured inside the fully-overlapped turn-1 decode window (c x per-stream),
medians over 15 rows per (arm, c) across the three boots.

### Vendor non-thinking shape (the production shape)

| c | per-stream OFF | per-stream ON | aggregate OFF | aggregate ON | agg delta | TTFT t1 p50 OFF | ON |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 233.5 | 242.2 | 233.5 | 242.2 | +3.7% | 0.261 s | 0.263 s |
| 4 | 27.3 | 138.2 | 109.6 | 527.2 | +381.0% | 0.962 s | 0.946 s |
| 8 | 14.5 | 99.0 | 115.5 | 758.0 | +556.3% | 1.880 s | 1.876 s |

### presence_penalty 0.0 control (penalty-free rows never enter the door)

| c | aggregate OFF | aggregate ON | delta |
|---:|---:|---:|---:|
| 1 | 259.3 | 248.8 | -4.1% |
| 4 | 524.6 | 514.1 | -2.0% |
| 8 | 742.8 | 739.4 | -0.5% |

The penalty-free rows scale in BOTH arms (524.6 at c=4 with the door OFF): the non-scaling
was the host penalty path and nothing else. The flat control also bounds the door's cost on
rows it never touches at noise level.

### How much of the collapse the door removes

| c | vendor OFF agg | vendor ON agg | pp0 ceiling agg | ON as % of ceiling |
|---:|---:|---:|---:|---:|
| 1 | 233.5 | 242.2 | 259.3 | 93.4% |
| 4 | 109.6 | 527.2 | 524.6 | 100.5% |
| 8 | 115.5 | 758.0 | 742.8 | 102.0% |

At c>=4 the penalized ON rows run at or slightly above the penalty-free ceiling (the
device penalty epilogue is not on the critical path once the host sort is gone).

### pp512/tg128 twin, c=1, explicit top_p 1.0 / top_k 0

| arm | prompt tok | pp tok/s | tg tok/s |
|---|---:|---:|---:|
| off | 501 | 5,813 | 210.2 |
| on | 501 | 5,753 | 215.1 |

pp -1.0%, tg +2.4%: the truncation-filter arm is neutral; the win is not a top_k/top_p
artifact.

### Spec engagement (turn 1, per the serving laws)

c=1: engaged on 15/15 requests in both arms (K>0; median rounds 61/60, acceptance
0.646/0.688). c=4 and c=8: `K=0 source=concurrency` on every request in both arms, exactly
as #266 decided; the door does not re-arm spec, it fixes the plain batched path the shed
lands on. Raw `usage.spec` blocks are in every receipt row (`receipts/raw/`).

### Greedy exactness

- `greedy-pp15`: BYTE-IDENTICAL across all 6 boots, both arms (sha
  `bb40202d3e37eceac745334394be2a1f1c4e9ef0d850d1c96fc8bce896fa98fc`, 185 tokens).
- `greedy-pp00`: BYTE-IDENTICAL across all 6 boots (sha
  `4139d34e4470ff074ae44e39ad2de02aef38f4ff903e0158734f46b0369ffae0`, 200 tokens).

No red arm. The door did not change one greedy token.

### Default-flip smoke (post-cell rebuild, binary 981a69b8)

`MEMRA_SERVE_DEVPENALTY` unset on the flipped binary takes the device path, verified by
behavior, boot nonce sha, and /proc env (`receipts/raw/off-b4-*`): vendor shape aggregate
526.2 tok/s at c=4 and 750.7 at c=8 (the ON rows), greedy hashes unchanged. The flipped
default is live end to end, and `=0` remains the rollback seam (unit-tested in
`serve_devpenalty_default_tests`).

## The flip

`serve_devpenalty_from(env, built_arch, arch)`: explicit `1`/`0` always win on every pair
(rollback seam); absent or any other spelling follows the qualified-pair default, which is
ON only for `Arch::Qwen35` on a `120a` build. Resolved once per session at admit
(`Session.devpenalty`), never per token; the value cannot drift mid-request.
`BUILT_CUDA_ARCH` is exported by memra-engine's build script and read as a const, so the
default keys on the build's compiled arch, not a runtime probe.

Not flipped, deliberately: Qwen3.8/Ornith on RTX PRO 6000 keep the OFF engine default even
though they carry the larger historical win, because that fleet pins the flag explicitly;
their default change is a separate lane with its own launcher smoke.

## Gates

(to be appended: local-ci correctness stage output from the box)

## Cost

Box 50430170 at $0.649/h from first stage to destroy; cell ~40 min GPU-idle wall, flip
rebuild 2m19s, smoke ~4 min, gates battery (see above). Total lane box time is printed in
the destroy receipt; roughly 24 h held including the overnight the dead worker left it
running.
