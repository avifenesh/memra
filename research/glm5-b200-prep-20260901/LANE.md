# glm5 B200 PREP lane (lane/glm5-b200-prep-20260901)

> Current update: `research/b200-kernel-twins-dry-20260901/README.md` supersedes this lane's
> "no Memra tcgen05 kernel" state for NVFP4 only. The default W4A8 int8 base and an opt-in W4A4
> tcgen05/TMEM twin now dry-build. Block-FP8 also has a dense tcgen05/TMEM twin plus its legal
> grouped plain-MMA fallback. They have no current real-B200 runtime qualification.

Owner pivot 2026-09-01: serving launches on the current Workstation cards; the main
engineering effort moves to 2x B200 + 1M context for GLM-5.3-Flash. NO BOX EXISTS YET
(the rental waits on the owner's provider ruling; the quote sheet is business content and
lives in the private repo, `research/b200-provider-quotes-20260901/QUOTES.md` there). This
lane is everything that needs NO GPU, so the box window starts hot.

Base: `origin/main` @ `f98cfbf17`. Worktree `wt-b200-prep`, branch
`lane/glm5-b200-prep-20260901`. Local toolchain: nvcc 13.1 (`cuda_13.1.r13.1`), compile-only,
zero GPU work.

Contents:

1. [The sm_100 compile census](#1-the-sm_100-compile-census) (+ the three guard fixes this
   lane lands, and the CI cell that keeps them fixed)
2. [Card-class requalification checklist](#2-card-class-requalification-checklist)
3. [TP transport on NVLink: design note](#3-tp-transport-on-nvlink-design-note)
4. [1M on 2x192GB: posture and bring-up plan](#4-1m-on-2x192gb-posture-and-bring-up-plan)

---

## 1. THE sm_100 COMPILE CENSUS

Method: per-TU nvcc invocations mirroring `crates/memra-engine/build.rs` argument-for-argument
(`compile-census.sh`, banked; results `receipts/census.tsv`), so one failure cannot hide the
others the way build.rs's first-failure assert does. Three arms:

* **A** = build.rs-faithful `MEMRA_CUDA_ARCH=100a` (stub substitutions honored,
  `-DMEMRA_FA3_STUB`, `-DMEMRA_DISABLE_NATIVE_FP4=1` on qmatvec_gemm, `-fmad=false` on dsv4).
  This is what a naked 100a cargo build attempts.
* **B** = the REAL files wherever arm A substituted or stubbed. This is the needs-port
  catalog, not a build config.
* **C** = `compute_100` (non-a family arch), informational.

**"100 vs 100a": the build plumbing does NOT distinguish.** `build.rs` asserts
`120a | 100a | 90a | 89`; there is no plain-`100` opt-in. Arm C shows all 13 fatbin cells
compile identically for `compute_100`, so nothing in the fatbin set depends on an `a`-suffix
feature today; a family-arch build would only matter for a future multi-SKU Blackwell-DC
fatbin and is not worth plumbing now.

### Headline

**Before this lane, `MEMRA_CUDA_ARCH=100a cargo build -p memra-engine` failed on exactly ONE
translation unit of 29 census cells (13 fatbin + 16 static-lib)** (cargo receipt: `panicked at build.rs:635: nvcc static-lib build
failed for cu/mmq_q8_0_f32acc.cu`, 256 ptxas error sites, `receipts/`). The two 2026-08-23
stub-polarity fixes (mmq_nvfp4_w4a8, mmq_fp8_blk) had already landed; the third predicted
defect was the whole remaining wall. **This lane fixes it, plus one runtime door and the
census exceptions, and adds a CI compile cell so main stays sm_100a-green from here on.**

### Census table (arch = sm_100a; full TSV in `receipts/census.tsv`)

| TU | arm A (build.rs-faithful) | arm B (real file) | class |
|---|---|---|---|
| cu/kernels.cu, hybrid.cu, qmatvec.cu, flash_attn.cu (+5 KV variants), qmatvec_gemm.cu, moe_router.cu, spec_sample.cu, kda.cu (all 13 fatbins) | **OK** | n/a (A is the real file) | compiles-clean |
| cu/mmq_q45k.cu, mmq_iq_experts.cu, mmq_q8_0.cu, mmq_q4_0.cu, fp8_prefill.cu, f16_prefill.cu, mmq_nvfp4_f8f4.cu, moe_f16_grouped.cu, fp8_blk_dequant.cu, mla_attn.cu, dsv4_gpu.cu | **OK** | n/a | compiles-clean |
| cu/mmq_q8_0_f32acc.cu | **FAIL** ptxas `'mma with block scale' not supported on .target `sm_100a`` (256 error sites) | same | **needs-guard, FIXED this lane**: in-TU `__CUDA_ARCH__ >= 1000` admitted the one arch that rejects the instruction; now `>= 1200`, 100a takes the fail-closed `__trap()` arm like every other non-120a arch. Post-fix: compiles on 100a/120a/90a/89 (verified all four) |
| cu/mmq_fp4.cu (W4A4 mxf4nvf4) | OK (stub) | FAIL, block-scale MMA | needs-port (sm_120a encoding) |
| cu/mmq_nvfp4_w4a8.cu (NVFP4 W4A8 MMQ prefill) | OK (stub) | FAIL, block-scale MMA | needs-port |
| cu/mmq_fp8_blk.cu (per-block FP8 MMQ prefill) | OK (stub) | FAIL, block-scale MMA | needs-port |
| cu/qmatvec_gemm.cu native-FP4 kernel | OK (`-DMEMRA_DISABLE_NATIVE_FP4=1`) | FAIL, block-scale MMA | needs-port |
| cu/fa3_prefill.cu | OK (`-DMEMRA_FA3_STUB`) | FAIL in the C++ FRONTEND (`k45_fence` undefined): the wgmma path is 90a-only by construction, it does not even reach ptxas | needs-port (sm_100 dropped wgmma; the native path is tcgen05/tmem, a different programming model, not a retarget) |

Post-fix receipts: full `MEMRA_CUDA_ARCH=100a cargo build -p memra-engine` green
(`receipts/cargo-100a-postfix-tail.log`), and
`tools/fatbin-lookup-census.py --arch 100a` = `13 fatbins, 718 kernels present, 412 looked
up, 2 excused, 0 unexcused — OK`.

### Runtime-lookup gaps (the panic class compile gates cannot see)

The 100a census initially reported **2 unexcused looked-up kernels absent from every 100a
fatbin**, i.e. `Engine::func` runtime panics:

1. `qmatvec_gemm_nvfp4_fp4` — TWO reachable chains on a 100a build, both fixed this lane
   on the 120a PROPERTY. (a) `MEMRA_FP4=1`: the existing refusal (`refuse_portable_force`)
   only covered portable builds (an ENUMERATION, 89|90a); a 100a build is not portable, so
   the force sailed into the panic — the door now also asserts
   `MEMRA_BUILT_CUDA_ARCH == "120a"` and refuses by name. (b) kernel_check's Stage-C FP4 arm
   gated on `!cfg!(memra_portable_cuda)` — TRUE on 100a — so kernel-check on any NVFP4 GGUF
   reached the lookup with NO env force at all (peer review catch); `nvfp4_checks` now keys
   on the same 120a property and 100a records the skip cell. Same enumeration-vs-property
   class as the 2026-08-23 stub-polarity bugs, twice more.
2. `qmatvec_gemm_q8_0_wgmma` — all three call sites (mmq_ffi.rs, lib.rs, kernel_check's
   wgmma-mirror arm) compiled out under `cfg!(memra_hopper_mma)` (90a-only), the strongest
   guard kind. Declared in `tools/fatbin-lookup-exceptions.txt` with the 120a entry's
   reasoning.

### Arch-conditional catalog (what a B200 runtime inherits)

* `Engine::new`'s arch guard (`lib.rs`, `MEMRA_BUILT_CUDA_ARCH` vs device compute cap)
  already accepts `("100a", 10, 0)`; nothing to change.
* `detect_arch()` still refuses to auto-select 100a, deliberately, and its comment now states
  the current truth: compiles + CI-covered, zero runtime receipts, so the opt-in stays
  explicit until the box window banks gates. Same update applied to `docs/RELEASING.md`.
* `gdn_mma_default_on()` (`lib.rs:388`) is `hopper_mma || 120a`: on a 100a build the GDN MMA
  pre-work chain defaults OFF while its kernels ARE present (hybrid.fatbin compiles clean:
  the sm_80-class `mma.sync`/`ldmatrix` PTX is legal on sm_100). Conservative and correct for
  bring-up; re-pricing it is a box question and only matters if a GDN family (q38/hy3) ever
  serves on B200.
* `MEMRA_CUTLASS` build.rs-asserts 120a-only; unchanged.
* The Hopper warp-spec contracts (setmaxnreg, mbarrier noinc, wgmma C7515/C7517/C7519, the
  darklanes `hopper-warpspec-compiler-findings` memory) live in `cu/wgmma_common.cuh` +
  `fa3_prefill.cu` and are 90a-gated everywhere; **none of them transfer to sm_100a**, whose
  native tensor path is tcgen05 + tmem (new contracts, none probed by us yet). Nothing else
  in the tree issues warp-spec PTX.

### The SM100 native-FP4 question, settled (cross-lane conflict, census-authoritative)

Two circulating claims collide: "B200 is Blackwell, so NVFP4 is native" (and B200-TRANSFER
point 10's "SM100 carries native FP4 tensor cores") vs a banked
`-DMEMRA_DISABLE_NATIVE_FP4=1` build. Both halves are real; they are about different layers:

* **The silicon: YES.** B200 has FP4 tensor cores — reached through the tcgen05 (5th-gen
  tensor core) instruction family with tmem. No memra kernel targets tcgen05 today.
* **memra's kernels: NO.** Every "native FP4" kernel in the tree is hand-written in the
  sm_120a warp-level MMA encodings (`kind::mxf4nvf4.block_scale`, `kind::f8f6f4`), and
  ptxas REJECTS those on sm_100a ("Instruction 'mma with block scale' not supported",
  census arm-B receipts). Consequently a `MEMRA_CUDA_ARCH=100a` build compiles OUT the
  entire family, per kernel/TU: `cu/mmq_fp4.cu` (W4A4), `cu/mmq_nvfp4_w4a8.cu` (the tuned
  W4A8 int8 prefill — its block-scale int8 form is the same sm_120a-only kind),
  `cu/mmq_fp8_blk.cu` -> fail-closed stubs; `cu/qmatvec_gemm.cu` built with
  `-DMEMRA_DISABLE_NATIVE_FP4=1` (drops `qmatvec_gemm_nvfp4_fp4`); the `MEMRA_FP4` door and
  kernel_check's Stage-C arm refuse/skip on the 120a property (this lane).
* **What NVFP4 artifacts actually run through on an sm_100a build:** the dp4a decode matvec
  family (`cu/qmatvec.cu`, `qmatvec_nvfp4_dp4a_sel*` — plain int8 dp4a, compiles clean,
  421 dp4a sites) plus the fused MoE epilogue kernels, and cuBLAS/cuBLASLt-class prefill.
  Functional, not FP4-TC-accelerated.
* **The route to real native FP4 on B200** is a port lane: tcgen05 kernels (new contracts,
  none probed by us) and/or cuBLASLt's block-scaled FP4 matmul on sm_100 — plus the SM100
  re-gate law (every decode-exact / rows-exact class contract re-proven, never assumed).

So: "B200 = NVFP4 native" is TRUE of the hardware ceiling and FALSE of memra's current
sm_100a binary. B200-TRANSFER point 10 is right about the silicon AND right that the class
contracts must be re-gated; any owner-facing claim that a B200 box serves NVFP4 on FP4
tensor cores TODAY is wrong until the port lane lands. The stale build.rs line that said
B200 "uses the accuracy-safe W4A8 int8 path instead" (that path is a stub on 100a) is
corrected in this lane.

### What GLM-5.3-Flash actually needs on sm_100a (the census mapped onto the serving path)

The needs-port family (block-scale MMQ prefill tiles, FA3) has **zero call sites in the glm5
walk**: `rg w4a8` over `hybrid_forward.rs`/`glm5*` is empty. The glm5 path is: KDA/MLA/DSA
kernels (kda.cu, mla_attn.cu, hybrid.cu, all compile), NVFP4 decode matvec + fused MoE
epilogue (qmatvec.cu/hybrid.cu, compile), MLA-TC prefill (cuBLASLt bf16, host API, native on
sm_100), grouped MoE prefill (`cublasGemmGroupedBatchedEx`, host API), spec sampling
(spec_sample.cu, compiles). **So GLM5 on B200 is a bring-up-and-tune problem, not a port
problem.** The port problem (block-scale MMA prefill tiles at tcgen05) belongs to the wider
catalog (qwen/step MMQ prefill) and is a separate named lane if those families ever serve on
this card class.

### Landed in this PR

* `cu/mmq_q8_0_f32acc.cu` guard `>= 1000` -> `>= 1200` (the one compile blocker).
* `MEMRA_FP4` door: sm_100a builds refuse by name instead of panicking at lookup (portable
  builds keep their pre-existing early-return path).
* kernel_check `nvfp4_checks` keyed on the 120a property (the Stage-C FP4 arm reached the
  missing kernel on 100a with no env force; 100a now records the documented skip cell).
* `tools/fatbin-lookup-exceptions.txt`: the two declared 100a rows.
* `ci.yml` `release-arch-mirror` matrix: `["100a", "90a", "89"]` — every merge now compiles
  the 100a source set and runs its fatbin census. 100a stays OUT of `release.yml` (advisory
  invariant untouched: compile-covered OR shipped, never an unexercised ship).
* Comment/doc truth: `build.rs` detect_arch block + warning, `docs/RELEASING.md`.

---

## 2. CARD-CLASS REQUALIFICATION CHECKLIST

Every default below was priced on GDDR7 (1.79 TB/s, 96GB) + PCIe (53-56 GB/s copy-engine,
atomics-banned) and re-prices on HBM3e (8 TB/s, 192GB) + NVLink 5 (1.8 TB/s bidi/GPU).
Governing laws: `card-keyed-full-pins` (a conditional default is pinned on every engaging
card class and every phase), `capacity-keyed-defaults-need-big-card-gates`,
`pin-against-truth-not-siblings` (fresh-boot output-sample gate for every new default),
`interleaved-ab-protocol-law`. Ordered by serving impact:

| # | default / family | today (WS cards) | why it re-prices on 2xB200 | box action |
|---|---|---|---|---|
| 1 | **Expert residency posture**: SLRU arena (`MEMRA_MOE_VRAM_FRAC`=0.85 grow-on-demand, `MEMRA_MOE_HARD_VRAM_FRAC`, `MEMRA_MOE_SLOTS` floor 3*n_used, `MEMRA_MOE_RESIDENT_HEADROOM_GB`, `MEMRA_ST_PINNED` host staging) | host-pinned staging + capped arena is the ONLY demonstrated 1M config (PP4); arena calibration is 96GB-capacity arithmetic; first-request arena growth silently ate 1M admission (1m-battery finding 1) | the whole posture is a capacity artifact: on 384GB the experts are RESIDENT and the arena should not exist. Grouped MoE prefill (`MEMRA_MOE_GROUPED_PREFILL`, default ON, executes 0 times on the 1M posture today) engages only on a resident slab | boot resident, assert `[moe-grouped-prefill] execute` lines > 0 and `provenance=resident-slab`, pin the posture EXPLICITLY (never let a growth fraction decide capacity) |
| 2 | **`MEMRA_TP_TRANSPORT`** (host-canonical default; peer-pull arm) + the atomics ban + pull-only law | default OFF by written decision, zero receipts on real peer fabric; every transport law is PCIe-scoped (see §3) | fabric term collapses ~13-17x at prime; NativeAtomic ban and ForceP2P note are PCIe-scoped laws that LIFT on NVLink; the default flip requires the on-box interleaved re-price receipt per its own FLAGS row | §3's gate ladder; the flip lands in the same commit as the receipt |
| 3 | **bf16 decode mirror (`MEMRA_BF16_MMV`) x 1M capacity trade** | 37.68 vs 24.01 tok/s at shallow depth, mutually exclusive with 1M on 4x96GB (1m-battery finding 3: "the fleet cannot have both") | on 384GB both fit simultaneously; the exclusivity law dissolves, but the mirror's win (halved decode read traffic) re-prices at 8 TB/s where decode may stop being DRAM-bound | A/B mirror on/off at the 1M resident posture; expect ON but measure, do not inherit |
| 4 | **Spec policy**: `MEMRA_SPEC_K` auto table (K=3 default, K=2 cached-long, kpolicy-20260808 measured on 5090/RTX6000), `MEMRA_SPEC_PMIN` 0.7, `MEMRA_GLM5_VERIFY_BATCH` arms | K knee priced on GDDR7 draft-vs-verify cost ratio; verify-rows kernels tuned against WS DRAM peak | verify cost falls faster than draft cost at 8 TB/s, so the K knee likely moves UP; acceptance is model-side and carries over | re-run the K=1..8 battery + PMIN sweep on-box before any spec claim; keep the placement gate receipts in scope (§4 defect 1) |
| 5 | **Prefix cache + admission at 1M**: `MEMRA_PREFIX_CACHE_MB` (derived 2-entry boot clamp), `MEMRA_ADMIT_PREFILL_WORKSPACE` formula (~1.02 MiB/chunk-token est vs 0.8 measured on WS), the 23,936 B/token admission cost row | prefix cache pinned 0 in every 1M cell (no VRAM for it); workspace formula constant measured on one card class | at 1M a prefix snapshot is ~25GB and 384GB can hold it: prefix cache flips from impossible to the single largest repeat-TTFT lever (an 88-minute prime amortized to a restore); admission constants must be re-measured, not assumed | re-derive the clamp at ctx=1M, re-measure workspace bytes/chunk-token on B200, then a warm-prefix 1M TTFT row |
| 6 | **Matvec door family** (door X `MEMRA_BF16_TCOLS_X1` ON, door R `MEMRA_BF16_TCOLS_RED_FUSED` OFF, door M `MEMRA_MOE_VROWS_PACK` OFF, `MEMRA_NVFP4_BANK_SM`/`MEMRA_NVFP4_SEL_DOWN8` ON since 2026-09-01) | grid/occupancy arithmetic keyed to WS SM count and one-resident-wave shapes; down8 pins are decode-only on this card class | B200: different SM count, 228KB smem/SM, HBM latency; "about one resident wave" arithmetic is false by construction; per card-keyed-full-pins these doors carry NO receipt on B200 in either phase | kernel-check + per-door A/B on-box; defaults stay as-shipped until then (they are correctness-safe, only the speed pick is stale) |
| 7 | **Capacity-keyed literals**: `MEMRA_KQRP` (free-VRAM covers mirror + 8GiB headroom; 24GB refuses), `MEMRA_Q8RP` (the card-keyed-defaults law's origin), `MEMRA_CPU_EXPERT_CACHE_GB`=16, `MEMRA_SPILL_PREAD_DEPTH` | thresholds and headrooms sized against 24-96GB cards | all trivially satisfied at 192GB (the danger is silent over-provisioning, not refusal); host-tier spill paths should be DEAD on a resident posture | assert the spill/host-tier counters read 0 on the 1M posture; if not, the posture is wrong |
| 8 | **Placement**: `MEMRA_PP_SPLITS` (13,26,39 on PP4; 15,30 PP3), `MEMRA_PRIME_CHUNK` 4096, pipeline shapes | splits balance 96GB cards; chunk size tuned against WS prefill workspace | PP2 split re-derivation on 45 layers x 192GB (the last-stage +MTP+hiddens tax moves); chunk knee re-prices against B200 workspace | placement-arith rerun + a chunk-size sweep inside cell 3 (§4) |
| 9 | `MEMRA_FA_PP_MINBLOCKS`, `MEMRA_MMQ_X/Y_*` tile seams | ncu-derived on H100/5090 (smem-per-CTA occupancy knees) | different SM geometry; only matters for the families that use them (not glm5) | out of scope until a block-scale-MMA port lane exists |
| 10 | `gdn_mma_default_on()` = OFF on 100a builds | const arch enumeration | kernels present and legal on sm_100; default conservative | only if q38/hy3 serve on B200: A/B then extend the const |

Top-5 for the owner's eye: rows 1-5.

---

## 3. TP TRANSPORT ON NVLINK: DESIGN NOTE

Base facts from `tp-transport-20260901/LANE.md` + `tp_transport.rs` (now general:
`MEMRA_TP_TRANSPORT`, rank-widened, glm5 walk is the only consumer): the seam has two arms,
`host-canonical` (dtoh->host->htod, a full stream drain per leg; the 13-18 ms/token TP-2 join
tax = 847 host legs / 446 draining syncs per decode token, reconstructed from counts with
zero residual) and `peer-pull` (consumer-issued `cuMemcpyDtoDAsync` + publication/release
events, 4 event primitives per read, atomics-free, gate-proven byte-identical and
transport-vs-transport bit-identical on the rig).

### The core design claim: the NVLink arm IS the peer-pull arm

Nothing structural replaces the PCIe peer read. `cuMemcpyDtoDAsync` on an NVLink-connected
pair rides NVLink automatically once peer access is granted (`tp::grant_peer_access` does
both halves already: `cuCtxEnablePeerAccess` + `cuMemPoolSetAccess`). The transport module
does not know the fabric, and that is correct; what changes is the ARITHMETIC and which
banked LAWS still bind. No new transport arm is needed for day 1 on B200. What is needed is
fabric identity in the receipts (below), because a peer-pull number with the fabric unnamed
would repeat the exact receipt-scope failure the tp-transport lane spent stage 1 correcting.

### Expected join-cost change (predictions to price, not claims)

Decode (t=1), TP-2, EP diet ON: peer-pull moves 6.09 MiB/token over 275 reads + 1100 event
primitives. The exposed cost is LAUNCH-bound (~2.0-3.5 ms/token predicted on PCIe; fabric
time 0.115 ms is 1% of the v1 tax).

* **NVLink decode: essentially unchanged.** Fabric term 0.115 ms -> ~7-9 us (6.09 MiB at
  ~700-900 GB/s effective). The launch/event term does not move: the per-hop cost is CPU
  issue, not wire. Predicted decode join ~2.0-3.4 ms/token, i.e. NVLink alone does NOT beat
  the PCIe peer-pull prediction at decode. The decode unlock stays CUDA-graph capture of the
  copies+events (the capturable set by construction), on both fabrics alike.
* **Prime (t>1): this is where NVLink pays.** At prime the same hop count carries t-wide
  payloads, so the join is BYTE-bound: ~14.6 MiB/token crossed (host-canonical census) means
  a 4096-token chunk crosses ~58 GiB. At PCIe 53 GB/s that is ~1.1 s/chunk of pure fabric
  (~4.8 GPU-minutes over a 253-chunk 1M prime); at NVLink ~0.07 s/chunk (~17 s over 1M) —
  a ~13-17x cut of the fabric term, which is what makes TP-2 prime worth re-testing at all
  on this fabric (on PCIe it was never viable).
* **TP-2 vs PP-2 on 2 cards is a fresh question.** PP forecloses nothing here (spec+PP works
  on our stack, stages 2-3 gated), but TP adds the second memory system at DECODE (the
  RESEARCH.md §1.5c mechanism) while PP alternates. The banked TP-4-vs-TP-2 priors are
  4-card PCIe numbers; do not transfer them. Cell 5 (§4) prices TP-2-NVLink against PP2
  directly.

### Which PCIe-scoped laws LIFT on NVLink (scope lines are part of the law)

1. **The atomics ban lifts.** `NativeAtomicSupported=0` and "CAS silently loses barrier
   tokens under PCIe load" are SM120-PCIe-pair findings. NVLink pairs report native atomic
   support, so a future fused collective may use device-scope flags/atomics. The ban stays
   in force for any PCIe deployment; the transport doc's scoping already says so.
2. **The push-collapse (2.6 GB/s SM peer writes) is PCIe-scoped.** On NVLink, SM-issued peer
   loads AND stores run at fabric rate. Pull stays the shape (the ordering argument is
   fabric-independent), but the 20x asymmetry is no longer a design constraint.
3. **The ForceP2P/SysMem-staging trap does not exist on NVLink.** Kernel-dereferenced peer
   pointers are served by the fabric, not staged through SysMem, so the fused-collective arm
   is not blocked on a modprobe override. `peer-read-probe.cu` (banked, tools/) becomes a
   FABRIC-IDENTITY instrument on B200: if it measures PCIe-class bandwidth on a "NVLink"
   pair, the box is mis-provisioned; refuse the rental (this is the acceptance gate the
   provider quote sheet's "NVLink-pair confirmed" column needs).
4. **What does NOT lift:** the pull-ordering design, the event (not host-fence) contract,
   the write-after-read release event (gate-caught race), the census counters, and the
   `tensor.copy_()` latency-instrument trap (stream-launch overhead still dominates any
   host-timed hop; latency comes from a GPU-resident ping-pong kernel or end-to-end tok/s
   only).

### The byte-integrity ladder on NVLink

Keep the ladder exactly as is (16 KiB / 64 KiB / 1 MiB / 64 MiB, both directions, poisoned
destination, `f32::to_bits` compare, census-excluded arm-time traffic), and add:

1. **A fabric-identity receipt in the armed announce**: `fabric=nvlink|pcie|unknown` derived
   from the P2P performance-rank attribute + a measured ladder bandwidth (a 64 MiB rung at
   >200 GB/s cannot be PCIe 5.0 x16). Design only, lands with the box window so the receipt
   is born with hardware to verify it. Every `[glm5-tp-*]` boot line then names the fabric
   the number was measured on.
2. **The kernel peer-read probe as a boot gate** (peer-read-probe.cu via box HEALTH), pass
   REQUIRED on B200 (unlike PCIe NODE hosts where its failure is expected and only blocks
   the fused arm).
3. **NVSwitch/link-state receipt**: `nvidia-smi nvlink -s` capture per boot, so a degraded
   link (fewer lanes lit) cannot masquerade as an engine regression.

### Gate list for the box window (transport slice)

| gate | bar |
|---|---|
| ladder x2 directions + peer kernel probe + link-state | mismatches=0, NVLink-class bandwidth, all links up |
| `glm5-tp-gate` full matrix, both transports, on-box | all arms pass; decode BYTE identity; prime band; XT transport-vs-transport BYTE identity |
| undieted sequential EP walk arm | REQUIRED (the highest-hop-count walk is the transport instrument; the dieted arm was a vacuous green once already) |
| census non-vacuity | `peer_pulls>0, host_legs=0, host_syncs=0`, pinned-`=0` arm flat |
| interleaved x5 fresh-boot re-price vs host-canonical AND vs PP2 | the default flips only in the same commit as this receipt |

No code lands from this section now: the only compile-clean-without-hardware piece (the
fabric announce field) is worthless without a fabric to name, and a receipt field that has
never printed truthfully is the DE-DFlash2 shape. Everything above is design + gates.

---

## 4. 1M ON 2x192GB: POSTURE AND BRING-UP PLAN

Banked inputs (`1m-demo-20260829`, `1m-battery-20260901`, `gpf-workspace-20260830`,
`prefix-latent` window): 1M prime on 4x PRO 6000 PP4 = **88.3 min TTFD / 195.33 tok/s
prefill / 16.6 tok/s decode @1.035M**, MoE-DISPATCH-bound because `MEMRA_MOE_GROUPED_PREFILL`
(default ON, the attributed dominant lever) **executes 0 times** on the only posture that
fits 1M (host-pinned staging + capped SLRU arena has no resident slab). Spec and 1M are
mutually exclusive today: spec admits at PP stages 2-3 only, 1M needs PP4, and the refusal
is SILENT (`serve route ARMED` then plain forever). Decode is depth-flat by architecture
(KDA linear trunk + DSA topk+tail: 21.07 -> 20.37 plain across 16k -> 131k; spec uplift
1.305x @16k -> 1.259x @131k).

### The fit (why 2 cards dissolve the 4-card tensions)

Per-card 192 GB HBM3e (~178.9 GiB; hyperscaler catalogs report ~179 GiB usable per GPU, so
treat ~176 GiB/card allocatable). Budget on 2 cards, resident posture, PP2:

| item | size | note |
|---|---|---|
| NVFP4 weights, ALL resident (incl. all 288 experts x 42 layers) | **190.75 GB** = 171.2 GB routed bank + ~19.5 GB everything else (B200-TRANSFER.md point 11) | ~95 GB/card at an even split; no SLRU arena, no host staging, no arena-fraction calibration. Single-card full residency does NOT fit (~1 GB left of 192) — 2-card is the floor |
| 1M KV + DSA planes | ~24.8 GB (the banked admission row: 23,936 B/token x 1.0357M; MLA/DSA layers only, KDA state is O(1)) | per-stage share on PP2 |
| prefill workspace @1M | ~25.6 GB (banked admission print) + hiddens ~16 GiB last stage (gpf-workspace) | admission formula re-measured on-box (checklist row 5) |
| bf16 decode mirror | tens of GB | now AFFORDABLE alongside 1M (finding-3 trade dissolves; still A/B it) |
| prefix-cache snapshot @1M | ~25 GB/entry | flips from impossible to the top repeat-TTFT lever |

Total core posture ~258 GB of ~352 GiB usable: **fits resident with >90 GB headroom.** The
3-card DSA-kpool OOM class (97,242 of 97,887 MiB peak on the last-stage card) has 80+ GB
of slack on a 192 GB card.

### Where the prime bottleneck moves

Today's 88.3 min is per-token MoE dispatch on a non-resident posture. On B200 resident:

1. Grouped MoE prefill EXECUTES (resident slab exists). Its banked engagement multiplier on
   the plain walk is 85 -> 616-639 tok/s shallow; even discounted for depth, prime moves
   from dispatch-bound to **grouped-GEMM + MLA-TC bound** (both cuBLAS/cuBLASLt bf16, native
   and tuned on sm_100, 2,250 dense bf16 TF/card).
2. Then the DSA indexer scan + kpool selection at depth and the PP2 boundary/hiddens staging
   become the next terms; at 8 TB/s the flat-plane-vs-ring question (QUIRK
   dsa-ring-off-slow-scorer) re-prices too.
3. Honest target class: 195 tok/s -> thousands of tok/s prefill is PLAUSIBLE (grouped
   multiplier x TC prefill on 3.6x per-card bandwidth and 2xTF), i.e. 88.3 min -> the
   ~5-15 min class. The 90 s fronted deadline stays a PRODUCT wall regardless (the banked gaps: 71.3x
   at the demo's 161.28 tok/s, ~58.9x at this head's 195.33 / 88.3 min); the 1M SKU ships behind streaming + prefix-restore economics, not
   behind a cold 90 s prime. No number above is a claim; cells 3-5 price them.

### Defects that MUST close before 1M serves (ordered)

1. **The silent spec ARMED-then-plain refusal** (`glm5_sharded_placement_admits` declines
   with no log line; 1m-battery finding 4). Fix: one loud refusal line naming stage count +
   admitted range, AND the deploy battery asserts the `usage.spec` engagement receipt at the
   serving depth (the never-serve-greedy / spec-engagement owner law). Engine-side, small,
   unit-testable without GPU; named follow-up lane, not this PR.
2. **Spec x placement receipts on the B200 shape.** PP2 is inside the gated stages 2-3 set,
   so no stages=4 gate extension is needed on 2 cards, but per
   capacity-keyed-defaults-need-big-card-gates the stage-2 receipts must be RE-RUN on the
   new card class before spec serves there (cell 4).
3. **sm_100a runtime exactness from zero** (this lane's compile fixes are necessary, not
   sufficient): kernel-check, oracle parity, the glm5 gate suites, fresh-boot output-sample
   gate (pin-against-truth law) before any perf row.
4. **Posture pinning**: the serving env pins residency EXPLICITLY and asserts the grouped
   arm's execute lines > 0 at boot; an arena growth fraction must never decide capacity
   implicitly again (finding 1's lesson, restated for the new box).
5. **Edge-timeout product seam**: `MEMRA_TIMEOUT_MS_MAX` is a measurement pin and never
   reaches a fronted route; 1M serving needs the streaming/phase-aware edge design before
   any public claim (edge-timeouts-phase-aware).

### The first 5 box cells

Cross-checked against the composition lane's handoff
(`composition-20260901/B200-TRANSFER.md`, PR #89): its transfer laws are folded in below —
the launch-diet census as an early instrument (every later receipt divides by the per-box
us/launch constant), FRESH prime-band calibration (near-tie bands are per (rank count,
shard shape, KERNEL ARCH); SM100 selects different K-reduction splits, so the banked 2e-4 /
4e-3 bands are calibration rows for the WS arch, never inherited), and the acceptance-parity
gate on any sharded spec arm. Its point 11 also pins the fit arithmetic: 190.75 GB total =
171.2 GB routed bank + ~19.5 GB everything else, so FULL single-card residency leaves ~1 GB
on a 192 GB card — the 2-card postures below are the only resident shapes, and the PP2-split
vs TP-2-NVLink question is measured (cell 5), never inherited from the PCIe conclusion.
One DELIBERATE deviation from its first-actions order, stated rather than silent: the
handoff decides PP-2-vs-TP-2 at 262k BEFORE composing spec; this plan runs spec x 1M
(cell 4) on the PP2 posture first because the owner target IS 1M and PP2's spec receipts
already exist at stages 2-3 — the topology re-price (cell 5) then decides the serving
shape. Its points 5-6 also bind cell 5: the EP-aware vrows arm is the biggest composed-shape
term and lands BEFORE TP hours are burned, and the `[glm5-tp-ep] verify rows ride the
SEQUENTIAL EP walk` announce is the receipt that the unsharded vrows fit does not transfer.

| cell | arm | gate | expected receipt |
|---|---|---|---|
| **1. box acceptance + arch bring-up** | provider box, 2x B200; `MEMRA_CUDA_ARCH=100a` build on-box; HEALTH: peer-read-probe + ladder + `nvidia-smi nvlink -s`; launch-diet census (per-box us/launch constant) | NVLink-pair CONFIRMED (kernel peer read at NVLink-class GB/s, all links up); build attribution (git log -1 + binary strings census); kernel-check ALL GREEN on sm_100a | fabric receipt + first-ever sm_100a kernel-check log + the launch constant every later receipt divides by; refuse the rental if the pair is not NVLinked |
| **2. glm5 exactness ladder on sm_100a** | resident 2-card load, oracle-bank greedy parity + glm5 gate suites (tp-gate both transports, hyper-ppn, spec-ppn stages 2-3) + fresh-boot output-sample | decode BYTE identity stays the bar at t=1 regardless of arch; prime bands RECALIBRATED fresh on SM100 (the banked 10x-over-worst procedure, never the WS-arch bands); reds bite | the sm_100a exactness bank + the SM100 prime-band calibration rows; nothing downstream runs without them |
| **3. 1M prime, resident PP2, plain** | the demo corpus (sha a07d4fcd...), greedy + vendor-default sampled twin, `MEMRA_MOE_GROUPED_PREFILL` ON, prefix cache 0, chunk-size sweep piggybacked | `[moe-grouped-prefill] execute > 0` at 1M (the 4-card 0-execute finding INVERTED); admission green with NO arena pins; error census 0; per-card VRAM peaks banked | TTFD + prefill tok/s + decode@1.035M vs the 88.3 min / 195.33 / 16.6 baseline; the grouped-at-depth multiplier, measured for the first time |
| **4. spec x 1M composition on PP2** | ship spec env (DFlash2, PMIN 0.7, auto-K) on the cell-3 posture; plain twin same boot recipe | `usage.spec` block PRESENT at 1M (rounds>0, acceptance logged); the loud-refusal line proven on a deliberately inadmissible arm; loop-law screen; acceptance-parity vs a single-shard control before any spec row is read time-only (B200-TRANSFER law 7) | the first spec-at-1M row: acceptance + uplift at depth (banked curve predicts ~1.2x-class if acceptance holds past 131k), or a named refusal |
| **5. TP-2-NVLink re-price vs PP2** | `MEMRA_TP_TRANSPORT` 0 vs peer-pull, interleaved x5 fresh boots, boot-nonce identity, decode + prime rungs, census deltas; EP-aware vrows arm landed first (B200-TRANSFER laws 5-6, the sequential-EP-walk announce is the tell) | §3 gate list (ladder, XT byte identity on-box, undieted-walk arm, census non-vacuity) | join tax by fabric: decode ~launch-bound (prediction 2.0-3.4 ms/token), prime fabric term ~13-17x down; feeds the transport default decision + the topology pick for the serving posture |

Then (not in the first five): bf16-mirror A/B at 1M, prefix-restore 1M TTFT, K/PMIN battery,
the multiturn 8-turn cache-on twin per the standing measurement laws, and the matvec door
re-pins (checklist rows 4-6).

---

## Lane state

* Compile census + fixes: DONE (this PR; receipts in `receipts/`).
* Requal checklist: DONE (§2).
* NVLink transport note + gates: DONE, design-only by rule (§3).
* 1M posture + first cells: DONE (§4).
* Provider quote sheet: in the private repo (business content), one table, dated 2026-09-01.
* No GPU was used; no box rented; no default changed; no product fact touched.
