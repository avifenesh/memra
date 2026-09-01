# B200 block progress log

- 03:16Z block active, box b200-1 launched (MarketType=capacity-block required).
- 03:44Z first regen attempt: flashinfer refused on Blackwell GDN (allowlist trtllm_mha/fa4/triton) — trtllm_mha adopted.
- 04:0xZ reasoning-parser missing → 100% error pass; fixed (servers need --reasoning-parser qwen3 for --reasoning save).
- 04:20-04:42Z cascade: prepare_data silent arg failure (pb empty), mooncake_master absent, port collisions across 3 arms, mid-run script edit crash. All root-caused.
- 04:5xZ regen3 running clean: pb-30k staged (30,000 rows), 8 replicas at ~100% GPU, pb-think 4.58 it/s ETA ~1.5h; then nothink + own 16K retries + train-file rebuild + port-offset arm launch, fully scripted (tools/regen3.sh, tools/launch_arms_inner.sh).
- memra sm_100: v0.83.0 builds CLEAN with MEMRA_CUDA_ARCH=100a (upstream lane already existed). kernel-check queued for arms phase (cards 6-7).
- Phase-3 script staged: tools/export-eval-phase.sh (export → normalize → DSPARK serve → serving gate → own-sessions + gsm8k eval per arm).
- 06:42Z regen3 complete: pb 26.5K think + ~29K nothink rows; own retries recovered 0 (resume treats
  errored ids as attempted — known limitation, own stays 1307+636 think rows). Train files rebuilt:
  arm-a/c 80,965 rows, arm-b 61,645.
- 06:4x-07:1xZ arm capture-server crash chain: sglang 0.5.14 (SpecForge capture pin) ICEs in
  nvidia-cutlass-dsl on B200+CUDA 13.2. 4.5.2 = MLIR ICE ('llvm.mlir.global_dtors requires data');
  4.6.0 = API break (make_kwargs_wrapper map_dataclass_to_tuple); **4.5.3 = works**. Also: fresh
  control_dir required per relaunch; capture backend forced to triton via
  model.sglang_attention_backend (0.5.14 predates the Blackwell GDN allowlist).
- 07:2xZ ALL THREE ARMS UP: capture servers on cards 0/2/4 (130.6 GiB each), trainers on 1/3/5.
- 07:3x-07:5xZ two more capture-stack layers: producer watermark must be >= consumer quantum
  (runtime.in_flight_high_watermark=1024/low=512 vs batch1 x accum512), then mooncake C++ SIGABRT
  at transfer-engine init — CUDA-12 wheel on a CUDA-13.2 box; fix = mooncake-transfer-engine-cuda13
  (SpecForge docs line 313).
- 08:0xZ **TRAINING LIVE, all 3 arms**: captures 0/2/4 at 100%/137GB producing (rollout batches
  publishing), trainers 1/3/5 loaded (41/42/48GB; arm-C block-16 largest), producer-timing flowing.
- memra sm_100: workspace build lacks engine bins; building -p memra-engine --bins for
  kernel-check/run-gen recon on cards 6-7.
- 08:2xZ memra sm_100a recon FINDING (the block's engine deliverable): the workspace "build ok"
  was a tee-masked false green. Real state: memra-engine FAILS ptxas on sm_100a in
  cu/mmq_nvfp4_w4a8.cu — `Instruction 'mma with block scale' not supported on .target sm_100a`
  (x5 sites + fatal). Datacenter Blackwell (sm_100) has NO mma.block_scale form; block-scaled
  FP4/FP8 rides tcgen05.mma there. Consequence: memra's NVFP4 W4A8 path is sm_120a-only until a
  tcgen05 twin exists — a real (week-class) B200 port lane, NOT a flag fix. Raw: memra-bins.log
  (pulled to raw/b200/). CPU-side crates + non-NVFP4 kernels compile clean.
- Training arms unaffected: 6 cards hot (captures 100%, trainers 73-100%).
- 09:5xZ **memra native sm_100a kernel-check ALL GREEN** — 85 cells OK, 21 skipped (NVFP4
  fail-closed stubs), card 6, Q38 NVFP4+Q5K mint GGUF as the tensor source. First memra
  correctness receipt on B200 ever. raw: raw/b200/memra-sm100/kernel-check-100a-q38mint.log.
- Owner call: vLLM v0.27.1 (doctrine pin) single-GPU best-settings baseline on card 7 —
  market-lead reference numbers for Q38-FP8 on B200. Not a target (we race ourselves);
  context only. Install in flight.
- 10:0x-10:2xZ vLLM v0.27.1 market-lead baseline, 1x B200 (card 7), Q38-FP8 ST, best-practice
  defaults (FLASHINFER/trtllm-gen, block-FP8 GEMM, DeepGEMM-E8M0, async sched, FULL_AND_PIECEWISE
  graphs), --max-model-len 262144 --kv-cache-dtype fp8, random 8k-in/1k-out, ignore-eos:
    c=1:  103.43 tok/s   TPOT  9.25ms  TTFT  442ms   (n=10)
    c=8:  612.82 tok/s   TPOT 11.17ms  TTFT 1933ms   (n=64)
    c=32: 1269.06 tok/s  TPOT 22.20ms  TTFT 3089ms   (n=256)
  Single runs, co-tenant regime (6 cards training concurrently — thermals shared; context
  numbers, not publishable denominators). Raw JSON+logs: raw/b200/vllm-baseline/.
  Context: memra single-stream on RTX PRO 6000 publishes 127 tok/s (main fbd8bfd4 docs).
- 10:3xZ **memra FIRST LIGHT on B200 (sm_100a): Qwen3.8-27B-FP8 SAFETENSORS serving end-to-end,
  spec decode ON.** MEMRA_MODELS=q38fp8=<st-dir>, card 6, 100a build (plain-f8f6f4 arms).
  Smoke: coherent output; spec rounds live off the ST MTP head (accept 0.556 short / 0.486 on an
  800-token explain), 67.2 tok/s single-stream UNTIMED (co-tenant: 6 cards training + vLLM on
  card 7 — thermals/PCIe shared; NOT a publishable number, first-light smoke only).
  Chain: ST loader worked unmodified; only build.rs needed the three 100a deltas
  (branch lane/sm100a-fp8-bringup, commit e91dbd779 in the memra worktree).
  vs vLLM same box same shape class: 103.4 tok/s c=1 (also co-tenant). Gap analysis + clean-box
  measurement = post-block work on a dedicated rig; plain-MMA arm (2x slower than block_scale)
  and unported GDN fast paths are the known headroom.
- 11:0xZ **B200 tuning battery 1** (memra Q38-FP8 ST, card 6, single-stream 800-tok explain,
  N=3/cell, co-tenant direction cells — variance <1% within cells):
    baseline auto-K3: 66.5 tok/s (accept 0.486) | spec-OFF: 81.0 | **K=1: 82.5 (accept 0.756)**
    K=2: 76.4 (0.61) | HPOST@K3: 71.8 (0.549, +6.3pt acceptance vs baseline)
  Verdicts: default spec regime (auto K=3) SUBTRACTS ~17% on B200 in this shape — per-hardware
  arm doctrine case in point; K=1 is the leading default candidate; HPOST is a real acceptance
  lever here. Battery 2 in flight: K1+HPOST, K2+HPOST, K1+PMIN0.3, MEMRA_KV_FP8.

## ~12:30-15:40Z — tcgen05 prototype ladder closed; FP8-blk twin v2 shipped on lane/sm100a-fp8-bringup

- Rungs 2-6 all EXACT on hardware: real ue8m0 scales (SF tmem atom = 32x16B warpx4 tile, byte
  l*16+q*4 — two refuted layouts pinned it), K-loop enable-input-d chaining, 128x256 full tile,
  4104 TFLOP/s @ 91% of dense-fp8 peak (plain arm 1061 = 23.6%), TMA staging via the k-core-outer
  layout (no swizzle needed). Multi-warp-per-quarter tcgen05.ld verified legal; N-sweep: N=128 is
  the twin's operating point (3991 TF), N<=32 = plain-arm rate.
- Engine twin v1: fp8-mmq-check caught a token-tile offset bug (the m=512 cell), then ALL GREEN —
  but perf-flat (~0.9x floor). Loop-shape probes: the commit/drain round-trip is the wall, tcgen05.ld
  does not overlap MMAs, and synchronous staging dominates end-to-end.
- Twin v2 (ITER_K=384, 3 tmem D slices per commit, TMA double-buffered staging): fp8-mmq-check ALL
  GREEN, kernel-check 85 cells ALL GREEN, prime-path argmax MATCH (drift = accepted cross-arm class;
  NOTE run-gen's generate path is decode-parity MMVQ — prefill gates must use MEMRA_PP_ONLY),
  run-spec K=1..8 PASS. fp8_mmq_bench: 1.34-1.50x the Q8_0 floor on all 27B shapes (plain arm
  0.94-1.09x). Serving TTFT on 802-token prompt: 0.3275 s vs 0.4282 s = -23.6%.
- Merged origin/main (fused3 QKV) into the lane; box rebuilt. All receipts in
  /scratch/receipts/memra-sm100/ + memra-b200-tune/cells.jsonl, synced to raw/b200/.
- Ops: transient WARP flap killed the pull-loop (box was fine) — restarted with 3-strike tolerance.
  The provider CLI credentials now return InvalidClientTokenId (key rotated/revoked?) — OWNER INPUT NEEDED;
  SSH to the box is unaffected.

## ~15:40-20:30Z — decode wall mapped; NVFP4 exact + 8.5 PF; arms step-250 evals; eval automation

- Decode: ncu pinned qmatvec_e4m3_blk_mmvq at 72% of decode; 7-variant harness ablation → the
  kernel is I2F-throughput-bound on B200 (memory-bound on the 5090; owner: no 5090 re-check
  needed absent an affecting change). No shared fix exists inside bit-identity; escapes priced
  and banked. No default changed.
- Twin v2 serving: TTFT -23.6% on 800-token prefill; decode leader regression-free on the
  merged build (85.0 tok/s, acceptance byte-stable).
- NVFP4 (owner: "if b200 is bw nvfp is plausible"): rung 7 EXACT (mxf4nvf4 4X; bit23=UE8M0-when-
  set, atype code 1 = e2m1, SF-4X quad atom) after a poison-probe ladder; rung 8 rate 8.5 PF/s
  (94% of dense-FP4 peak) — NVFP4 is the fastest quant on the card. Twin = week-class, all
  ingredients proven.
- Arms: step-250 evals — arm-a own 1.307/1.370, gsm8k 1.721 vs arm-b 1.253/1.212/1.696: the
  own-corpus lead WIDENS (+13% agentic). arm-c@125 inversion (trainer acc 0.436, serving accept
  1.05); blk8 window control flat — points at warm-transfer failure or a deeper DFLASH tap issue;
  next read at its step-250. DFLASH serve path itself now works (fixed projector relabel + algo
  flag + mid-write race).
- Ops: ckpt-watch-v2 deployed (fires eval for EVERY new checkpoint, serialized, mid-write guard,
  per-step receipt archiving). Training ETA: arms will NOT finish 10 epochs before the block ends
  2026-08-16T11:30Z — a/b land ~step 860, c ~step 580; ship best checkpoint per curves at block
  end. Provider CLI creds invalid (InvalidClientTokenId) — owner input needed; SSH unaffected.

## BLOCK CLOSED (2026-08-16 ~09:05Z termination)

Final state of the three workstreams this block carried:

1. **DSpark training (the lane's goal)**: G3 verdict IN — own-corpus training beats the
   PerfectBlend control on own traffic (+20-35% accept at equal step) and the cum-750 own-mix
   drafter EXCEEDS the RadixArk control bar on own-sessions (2.65/2.96 vs 2.607) while tying it
   on gsm8k trend. Clean license end to end. Best resumable weights saved locally at cum-625
   (with optimizer state); full scoreboard + receipts in ARMS.md. Ops learned: managed_local
   cannot resume (weights-only warm seam works and is proven lossless at the quality level);
   ckpt-watch must guard mid-write checkpoints; one un-pinned GPU process is a fleet hazard.
2. **memra sm_100a (B200) lane**: tcgen05 prototype ladder rungs 1-9 complete and banked
   (DESIGN-tcgen05-sm100.md). FP8-blk tcgen05 twin v2 SHIPPED on lane/sm100a-fp8-bringup —
   full gate battery green, 1.34-1.50x the Q8_0 floor at every prefill m, serving TTFT -23.6%.
   NVFP4 proven exact AND fastest quant on the card (8.5 PF/s, 94% of peak). W4A8 container
   semantics pinned. Decode e4m3-blk MMVQ wall mapped (I2F-bound; no bit-identity fix exists).
3. **Baselines**: vLLM 0.27.1 B200 best-practice numbers banked pre-block (103/613/1269 tok/s).

Owner items outstanding: provider credentials invalid (InvalidClientTokenId) — cannot query/launch
until rotated; DP2-trainer batch question moot with box gone; lane/sm100a-fp8-bringup merge is
an owner call (pre-merge battery on a PRO 6000 verifier per doctrine); next training block
decision (cum-625 -> bar-plus weights ≈ 3-4h of B200-class time using the saved checkpoint).
