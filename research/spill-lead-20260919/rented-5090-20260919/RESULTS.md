# Rented RTX 5090 — first hour (2026-09-19)

**Box:** Vast.ai interruptible instance (Japan host; RTX 5090 32 GB, driver 595.84, power 600/600 W,
PCIe 4.0 x16 max, 128 vCPU, 220 GB host RAM visible, CUDA image `nvidia/cuda:13.1.2-devel-ubuntu24.04`,
nvcc 13.1 at `/usr/local/cuda`). Container root is **overlay**: the host's NVMe is not visible as a
block device, so NVMe ancestry is **unproven** and no storage number here is a spill-speed claim.
Owner-approved rental (Vast/RunPod, spot preferred, development only). One earlier Puerto Rico offer
failed to start (host CDI GPU injection error: `unresolvable CDI devices … gpu=2`) and was destroyed.
Provider ids / cost are private receipt metadata, not published here.

**Source:** `avifenesh/memra` `lane/spill-integ2-20260919` @ `01e7b77f` (contains main `61be8b0d` = PR #518).

## Bootstrap (`tools/tier-rig-bootstrap.sh`, receipt `receipts/*run3/BOOTSTRAP.json`)
status `bootstrap-complete-not-tier-qualified`; `cuda_acceptance = two-full-readbacks` (8 GiB
cudaMalloc + memset + readback, twice, 60 s gap); power 600 W confirmed on-box; rustup stable;
clone + minimum-source check; **native release build of memra-engine / memra-server / storage-bench /
pp-transport-smoke / qwen4exp_gpu_gate / run-gen / run-spec succeeded with nvcc 13.1 / sm_120a** —
first native compile of the tier lanes (B's `worker.rs` field, engine→memra-tier edge) on hardware.
Two bootstrap runs before it were BLOCKED and are kept as receipts: NVMe-ancestry refusal on the
overlay root (correct fail-closed), and a false-negative arch check (`--list-gpu-arch` never lists
`compute_120a`; fixed in `01e7b77f`).

## First-hour cells (`receipts/first-hour-20260919T130057Z/`, collector `tools/tier-battery.py`)
| Cell | Result | Note |
|---|---|---|
| A storage-bench roundtrip/restore × {264, 1048576, 4194568} B, buffered | 6/6 `byte-exact` | overlay filesystem exactness baseline; NOT NVMe, NOT O_DIRECT, no H2D |
| D1 `pp-transport-smoke` (single device, PP boundary slots) | PASS, 4 roundtrips `bytediff=0` | no peer pair on a single GPU |
| C `qwen4exp_gpu_gate` tiny PLE (run 1) | FAIL: goldens missing | RIG-DAY1 omitted copying `research/qwen4exp-bringup-20260829/gpu-eager/bank-bytes-goldens.tsv` beside the receipt — runbook bug |
| C `qwen4exp_gpu_gate` tiny PLE (run 2, goldens beside receipt) | **PASS, failures=0** (310-line receipt) | incl. plecache oracle EXACT (69,635 comparisons), qsa-index top-k exact, mtp-spec-ring byte-identity |
| B Qwen3.8-27B `run-gen` / `run-spec` (pinned HF `tiyuvta/Qwen3.8-27B-NVFP4-MTP-GGUF@0f82b27d`, sha256-verified) | see `driver-b.log` when synced | fitting baseline only; no active-8k demote/reload (gate not built yet) |

Every cell exit 0 is `executed-not-qualified` by the collector's own status. **No G0–G7 gate is
advanced by this hour.** What it does establish: the integrated tip builds and runs natively on
sm_120; PLE n-gram baseline exists on the rented box; storage/PP primitives are byte-exact.

## Box 2 (Czechia; PCIe 5.0 x16, 192 vCPU, 503 GB host RAM, overlay root, CUDA 13.1.2 image) — after box 1 was preempted
Box 1 (Japan) was preempted mid-download (`exited/stopped`; `start` → "Required resources are currently
unavailable"); A/D1/C receipts had already been synced; B's partial artifact and scratch worktree were
lost; box 1 destroyed. Lesson folded into WP-D: sync after every cell.

Bootstrap (`receipts-box2/*run3/`): acceptance ×2 OK, native release build OK in ~9 min (tip `9612aa34`).

| Cell | Result |
|---|---|
| Qwen3.8-27B artifact `tiyuvta/Qwen3.8-27B-NVFP4-MTP-GGUF@0f82b27d` `Qwen3.8-27B-NVFP4-Q5K-mtp.gguf` | sha256 **matches** HF-published LFS oid (`qwen-artifact.json`) |
| B `run-gen` argmax | **MATCH** (prefill argmax = decode argmax = 258; logit maxdiff 3.9e-1 within gate); 88.6 tok/s gen-only — development number, not a board claim |
| B `run-spec` K=1..8 | **SELF-CONSISTENCY PASS** at every K (identical to plain target) |
| A2 pinned-worker exact-byte test (`spill_pread::tests::worker_positioned_reads_preserve_exact_bytes_and_reuse_after_short_read`) | **ok** (run 2; run 1 exit 2 was the runbook passing a literal `--` through the collector's REMAINDER parser — runbook fix for WP-D) |

Still `executed-not-qualified`: these are the existing gates on a fresh box + the new substrate compiled
natively. No active-KV demote/reload cell exists yet (B's `kv_tier_gate` is being built); NVMe ancestry
remains unproven on the overlay root. G0–G7 pending.
