# MTP verify capture under VRAM pressure

The 10 GiB squeeze reproduced #263 on main `dd8cc9c74`. The fix is
`6adf6bbc9`; all after-fix GPU runs use that commit on one RTX 5090, sm_120a,
with CUDA 13.0. These are correctness cells, not performance measurements.

## Root cause

The loader makes one expert-residency decision per device, so the 10 GiB holder
sends both Ornith trunk and MTP experts to the SLRU path. Draft admission already
rejects the non-resident MTP head, but MTP verify-pool admission did not consult
that refusal. During `DsparkVerifyGraphs::run_full` or `run_segment`, the captured
`qwen35_tparallel_linear_layer` calls `moe_ffn_il_zq8`, whose softmax host router
calls `Engine::moe_router_topk_host`: two pinned DtoH copies followed by
`cuStreamSynchronize`, which CUDA refuses during capture.

The `il=65535 capture_open=false` diagnostic is emitted during eager MTP drafting,
before the failing captured trunk. Its `capture_open` field is the GLM-specific
capture marker, not a query of the CUDA stream's state, so it does not establish
that the subsequent MTP verify capture was closed.

## Fix

Draft capture and MTP verify-pool admission share `mtp_capture_refusal`.
A non-resident or host-routed MoE MTP head prevents pool creation and reuse,
even with the existing verify-graph flag forced on, and logs the eager fallback.
No flag or numerical kernel was added or changed.

The MoE cache destructor is unchanged. Its error does not prove that outstanding
copy and compute work has drained; freeing slots on that error alone is unsafe.
Refusing before capture avoids the invalidated stream and leak warning in this
reproduction without weakening that ownership rule.

## Reproduction

Build on the test host with `CARGO_TARGET_DIR=/root/target MEMRA_CUDA_ARCH=120a`
and `MEMRA_GPU_LOCK=/tmp/memra-gpu.lock`, using `cargo build --release --bins`.
The holder uses `cudaMalloc` for exactly 10240 MiB and sleeps until killed.
The held process occupies 10738 MiB including its CUDA context.
With the holder live, run `MEMRA_GPU_LOCK= /root/target/release/run-spec`
against `Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp.gguf`; kill and wait for the holder
at the end of the cell. The archived holder source and runner retain exact commands.

| Held allocation | Main observation |
| --- | --- |
| 8192 MiB | Resident, verify pool engaged, K=1..8 PASS |
| 9216 MiB | Resident, verify pool engaged, K=1..8 PASS; draft capture refused for headroom |
| 10240 MiB | Non-resident, verify pool engaged, CUDA capture unsupported, exit 1 |

The 10 GiB cell reproduced the issue, so no larger squeeze was needed.

## Validation

All GPU and Rust checks below ran on code commit `6adf6bbc9` on the test host.
Only formatting ran on the workstation. The later receipt commit changes no engine code.

- 10240 MiB held: the named eager verify fallback, eight self-consistency PASS rows,
  final `=== SELF-CONSISTENCY PASS ===`, exit 0, no capture error or slot-leak warning.
- Empty card: verify pool ENGAGED, eight self-consistency PASS rows, final PASS.
- Qwen3.8-27B: eight self-consistency PASS rows, final PASS.
- Plain token vectors match across before-squeeze, after-squeeze, and resident logs.
- Full release battery: PASS, kernel-check 95 cells and 21 skips; Ornith margin
  `flips=1 bad=0`, Qwen margin `flips=0 bad=0`, both models K=1..8 PASS.
- `cargo clippy --release --all-targets -- -D warnings`: exit 0.
- `cargo test --release -p memra-engine --lib`: 446 passed, 0 failed, 18 ignored.
  All three `mtp_capture_admission_tests` passed.
- `cargo fmt --check`: exit 0 on the workstation.

Raw logs are archived losslessly as `*.log.gz` beside this file. `before-8g`,
`before-9g`, and `before-10g` record the main sweep; `after-10g`, `after-resident`,
and `after-qwen` record the fixed cells. `battery`, `clippy`, `unit`, `build-main`,
and `build-fixed` retain the other receipts. Each run log prints its box SHA and
status; `?? target` is the test host's pre-existing target-directory symlink shape,
not an engine source change. The failed first bundle import was detected from this
readback and its build was stopped before any GPU cell; it supplies no qualification.

The holder was killed and waited for after each squeeze. The final device readback
and restoration of the preserved unrelated checkout are reported in the PR handoff.

