// Compile engine .cu kernels to the selected CUDA fatbin (same pattern as memra-probe).
use std::path::{Path, PathBuf};
use std::process::Command;

/// `nvcc --version`'s reported release, as `(major, minor)`. `None` when the binary cannot
/// be run or the line cannot be parsed — such a candidate is unusable and gets skipped.
/// Parses the canonical last line: `Cuda compilation tools, release 13.2, V13.2.68`.
fn nvcc_version(p: &std::path::Path) -> Option<(u32, u32)> {
    let out = Command::new(p).arg("--version").output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout);
    let rel = s.split("release ").nth(1)?;
    let rel = rel.split(',').next()?.trim();
    let mut it = rel.split('.');
    let maj = it.next()?.parse::<u32>().ok()?;
    let min = it.next().and_then(|m| m.parse::<u32>().ok()).unwrap_or(0);
    Some((maj, min))
}

/// Resolve the `nvcc` to build with.
///
/// EXPLICIT INTENT WINS, unvalidated: `MEMRA_NVCC` first, then
/// `CUDA_HOME`/`CUDA_PATH`/`CUDA_ROOT` + `/bin/nvcc`. Whoever sets those has chosen a
/// toolkit (CI pins 13.1 through `MEMRA_NVCC`; the portable `89`/`90a` arches build fine on
/// older releases), so this function must not second-guess them.
///
/// Otherwise: enumerate every candidate — `PATH` entries, `/usr/local/cuda/bin/nvcc`, and
/// each `/usr/local/cuda-<x.y>/bin/nvcc` — ask each one its release, and take the
/// **NEWEST**, not the first found.
///
/// WHY NEWEST-WINS RATHER THAN FIRST-ON-PATH (measured on the 5090 dev rig, 2026-08-06):
/// this rig has BOTH a distro `/usr/bin/nvcc` (**CUDA 12.4**) and `/usr/local/cuda-13.1`.
/// `/usr/bin` precedes the toolkit on the default `PATH`, and 12.4 predates sm_120 entirely
/// — a first-hit resolver picks it and the build dies at
/// `nvcc fatal : Unsupported gpu architecture 'compute_120a'`, which is a WORSE failure
/// than the hardcoded pin it replaced (it breaks a machine that used to work). Ranking by
/// reported release makes the pick order-independent and monotone: the rig picks 13.1, the
/// PRO 6000 box picks 13.2, and a stale distro nvcc can never shadow a real toolkit.
///
/// WHY AT ALL: the old code defaulted to a bare `/usr/local/cuda-13.1/bin/nvcc`. The
/// 2x RTX PRO 6000 box ships 12.8/12.9/13.0/**13.2** and no 13.1, so a naked `cargo build`
/// died at `panicked at build.rs: spawn nvcc: Os { code: 2, kind: NotFound }` — a message
/// naming neither CUDA nor the missing path. It cost the pp2-hardening lane its first build
/// (`research/pp2-hardening-20260806/PROGRESS.md` Phase 0) and the vast 2x5090 bring-up
/// before that (`research/vast2x5090-bringup-20260803/SUMMARY.md`). The resolved path and
/// release are always echoed via `cargo:warning`, so every build log records which nvcc
/// produced its fatbins.
fn resolve_nvcc() -> String {
    println!("cargo:rerun-if-env-changed=MEMRA_NVCC");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_ROOT");
    if let Ok(p) = std::env::var("MEMRA_NVCC") {
        println!("cargo:warning=nvcc from MEMRA_NVCC: {p}");
        return p;
    }
    for var in ["CUDA_HOME", "CUDA_PATH", "CUDA_ROOT"] {
        if let Ok(root) = std::env::var(var) {
            let cand = PathBuf::from(&root).join("bin/nvcc");
            if cand.is_file() {
                println!("cargo:warning=nvcc from ${var}: {}", cand.display());
                return cand.to_string_lossy().into_owned();
            }
        }
    }
    // Candidate set. Canonicalized + deduped so `/usr/local/cuda -> cuda-13.2` and a PATH
    // entry pointing at the same tree are not probed (or reported) twice, and so the
    // `<nvcc>/../../lib64` derivations below land in the real toolkit tree.
    let mut cands: Vec<PathBuf> = Vec::new();
    if let Ok(path) = std::env::var("PATH") {
        cands.extend(
            path.split(':')
                .filter(|d| !d.is_empty())
                .map(|d| PathBuf::from(d).join("nvcc")),
        );
    }
    cands.push(PathBuf::from("/usr/local/cuda/bin/nvcc"));
    if let Ok(rd) = std::fs::read_dir("/usr/local") {
        for ent in rd.flatten() {
            if ent.file_name().to_string_lossy().starts_with("cuda-") {
                cands.push(ent.path().join("bin/nvcc"));
            }
        }
    }
    let mut seen: Vec<PathBuf> = Vec::new();
    let mut ranked: Vec<((u32, u32), PathBuf)> = Vec::new();
    for c in cands {
        if !c.is_file() {
            continue;
        }
        let abs = c.canonicalize().unwrap_or(c);
        if seen.contains(&abs) {
            continue;
        }
        seen.push(abs.clone());
        if let Some(v) = nvcc_version(&abs) {
            ranked.push((v, abs));
        }
    }
    // Newest release wins; ties (same release reachable by two paths) keep discovery order,
    // which puts PATH ahead of /usr/local — the sort is stable.
    ranked.sort_by_key(|r| std::cmp::Reverse(r.0));
    if let Some(((maj, min), p)) = ranked.first() {
        let others: Vec<String> = ranked[1..]
            .iter()
            .map(|((a, b), q)| format!("{a}.{b} @ {}", q.display()))
            .collect();
        println!(
            "cargo:warning=nvcc auto-detected CUDA {maj}.{min} at {}{}",
            p.display(),
            if others.is_empty() {
                String::new()
            } else {
                format!(
                    " (newest of {}; also saw {})",
                    ranked.len(),
                    others.join(", ")
                )
            }
        );
        return p.to_string_lossy().into_owned();
    }
    // Nothing runnable found: keep the historic pin so the failure text names a familiar
    // path, and say out loud what to set. The spawn below turns this into a build error.
    let pin = "/usr/local/cuda-13.1/bin/nvcc";
    println!(
        "cargo:warning=no runnable nvcc found via MEMRA_NVCC / CUDA_HOME / PATH / \
         /usr/local/cuda*; falling back to {pin} — set MEMRA_NVCC=<path/to/nvcc>"
    );
    pin.to_string()
}

/// Arch auto-detection (MEMRA_CUDA_ARCH unset): probe the first GPU's compute capability
/// via nvidia-smi and pick the matching build arch. GPU-less machines (CI compile gate)
/// and unrecognized caps fall back to 120a — the naked build stays the sm_120a build.
/// An explicit MEMRA_CUDA_ARCH always wins (this fn is not called then).
fn detect_arch() -> String {
    let cap = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.lines().next().map(|l| l.trim().to_string()));
    // HISTORY (2026-08-23): it did not build at all —
    //   ptxas ... error : Instruction 'mma with block scale' not supported on .target 'sm_100a'
    // Three translation units, two distinct causes, found by fixing each and watching the
    // failure move:
    //   1. cu/mmq_nvfp4_w4a8.cu (~40 sites) — stub-gate polarity. FIXED below.
    //   2. cu/mmq_fp8_blk.cu   (~400 sites) — same polarity bug. FIXED below.
    //   3. cu/mmq_q8_0_f32acc.cu — NOT polarity: its f8f6f4 arm was guarded by
    //      `__CUDA_ARCH__ >= 1000`, and 1000 IS sm_100a, so the guard admitted the very arch
    //      that rejects the instruction. FIXED lane/glm5-b200-prep-20260901 (>= 1200; the TU's
    //      fail-closed __trap() arm now covers sm_100a like every other non-120a arch).
    //
    // CURRENT STATE (2026-09-01, lane/glm5-b200-prep-20260901): MEMRA_CUDA_ARCH=100a COMPILES —
    // per-TU census 29/29 arm-A cells green, and ci.yml carries a compile-only sm_100a matrix cell so
    // it stays that way. tools/fatbin-lookup-census.py --arch 100a passes with two DECLARED
    // exceptions (qmatvec_gemm_nvfp4_fp4 — the MEMRA_FP4 door refuses on non-120a builds;
    // qmatvec_gemm_q8_0_wgmma — call sites compiled out, cfg!(memra_hopper_mma)).
    //
    // HARDWARE CLOSURE (2026-09-01): sm_100a is now auto-selected. The exact 100a binary at
    // 69a2eb3684e1 passed the synthetic NVFP4 and block-FP8 batteries on one NVIDIA B200, then a
    // pinned Qwen3.5-9B NVFP4 artifact passed model-backed manifests, K=1..8, vendor-default
    // sampled serving, concurrency, context refusal, and rollback. The safe W4A8 path remains the
    // NVFP4 default. True raw-layout W4A4 (`MEMRA_RP=0 MEMRA_MMQ=1`) is correct but measured only
    // 0.521x raw W4A8 prefill, so it stays explicit. The block-FP8 twin is also correct and serves
    // a pinned official Qwen3.8-27B-FP8 checkpoint, but measured 0.173x its established fallback
    // with worse teacher-forced NLL, so `MEMRA_FP8_MMQ=1` stays explicit on this architecture.
    // Receipts: research/b200-kernel-twins-dry-20260901/receipts/.
    let arch = match cap.as_deref() {
        Some("12.0") | Some("12.1") => "120a",
        Some("10.0") => "100a",
        Some("9.0") => "90a",
        Some("8.9") => "89",
        _ => "120a",
    };
    match cap.as_deref() {
        Some(c) => println!(
            "cargo:warning=MEMRA_CUDA_ARCH auto-detected {arch} (compute_cap {c}); set MEMRA_CUDA_ARCH to override"
        ),
        None => {
            println!("cargo:warning=no GPU visible; defaulting MEMRA_CUDA_ARCH=120a (compile-only)")
        }
    }
    arch.to_string()
}

fn main() {
    // DOCS_RS flips the whole build between the CUDA build and the docs stub (below). Declare the
    // dependency before the docs branch returns: a `DOCS_RS=1` pass must invalidate the next real
    // build, or it links stub artifacts (2026-09-21: `undefined symbol: memra_dsv4_c4_recent_write`
    // in a server test run after a `DOCS_RS=1` clippy pass). memra #608: the earlier declaration
    // sat in resolve_nvcc, which the docs branch never reaches.
    println!("cargo:rerun-if-env-changed=DOCS_RS");
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    // Boot-time MEMRA_* environment audit (memra #483): the legal name set and the retired-door
    // set are generated from docs/FLAGS.md, the one registry the flags census already enforces,
    // so a deleted door gets a real ending (a refusal that names its ledger) instead of parsing
    // into nothing. Generated before the docs branch: the audit module is part of the rlib.
    generate_env_registry(&out);

    // docs.rs builders have no nvcc and no CUDA libs: emit empty placeholder fatbins so
    // the include_bytes!/env! consts compile, skip every nvcc/ar/link step. The resulting
    // rlib is documentation-only — Engine::new would fail to load a zero-byte module, but
    // docs.rs never runs code. Normal builds (CI included) never set DOCS_RS.
    if std::env::var_os("DOCS_RS").is_some() {
        for stem in [
            "kernels",
            "hybrid",
            "kda",
            "qmatvec",
            "flash_attn",
            "qmatvec_gemm",
            "moe_router",
            "spec_sample",
            "flash_attn_kf8vf8",
            "tier_receipt",
        ] {
            std::fs::write(out.join(format!("{stem}.fatbin")), []).unwrap();
        }
        for (env, stem) in [
            ("MEMRA_ENGINE_FATBIN", "kernels"),
            ("MEMRA_HYBRID_FATBIN", "hybrid"),
            ("MEMRA_KDA_FATBIN", "kda"),
            ("MEMRA_QMATVEC_FATBIN", "qmatvec"),
            ("MEMRA_FLASH_FATBIN", "flash_attn"),
            ("MEMRA_GEMM_FATBIN", "qmatvec_gemm"),
            ("MEMRA_ROUTER_FATBIN", "moe_router"),
            ("MEMRA_SAMPLE_FATBIN", "spec_sample"),
            ("MEMRA_FLASH_FATBIN_KF8VF8", "flash_attn_kf8vf8"),
            ("MEMRA_TIER_RECEIPT_FATBIN", "tier_receipt"),
        ] {
            println!(
                "cargo:rustc-env={env}={}",
                out.join(format!("{stem}.fatbin")).display()
            );
        }
        println!("cargo:rustc-check-cfg=cfg(memra_portable_cuda)");
        println!("cargo:rustc-check-cfg=cfg(memra_hopper_mma)");
        println!("cargo:rustc-check-cfg=cfg(memra_sm100_tcgen05)");
        println!("cargo:rustc-check-cfg=cfg(memra_cutlass)");
        println!("cargo:rustc-env=MEMRA_BUILT_CUDA_ARCH=120a");
        // `MMQ_ARCHIVE_HASH` is `env!`-bound in lib.rs; the docs-only build has no archive.
        println!("cargo:rustc-env=MEMRA_MMQ_ARCHIVE_HASH=docs-rs-stub");
        return;
    }

    let nvcc = resolve_nvcc();
    println!("cargo:rerun-if-env-changed=MEMRA_CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=MEMRA_CUTLASS");
    println!("cargo:rustc-check-cfg=cfg(memra_portable_cuda)");
    println!("cargo:rustc-check-cfg=cfg(memra_hopper_mma)");
    println!("cargo:rustc-check-cfg=cfg(memra_sm100_tcgen05)");
    println!("cargo:rustc-check-cfg=cfg(memra_cutlass)");
    let cuda_arch = std::env::var("MEMRA_CUDA_ARCH").unwrap_or_else(|_| detect_arch());
    assert!(
        matches!(cuda_arch.as_str(), "120a" | "100a" | "90a" | "89"),
        "MEMRA_CUDA_ARCH must be 120a (default), 100a (B200), 90a (Hopper), or 89 (portable eval)"
    );
    // Hopper boot arch rides the portable-CUDA correctness path: sm_90a SASS, no
    // sm_120a/sm_100a MMA kinds. Tuned wgmma paths are a separate later lane.
    // Phase A (ARCHITECTURE-H100.md): 90a additionally re-enables the portable-PTX
    // tensor-core paths the boot lane gated off — int8 mma.m16n8k32/k16.s8, bf16
    // m16n8k16, ldmatrix, cp.async are sm_80-class and run natively on Hopper.
    // The sm_120a/sm_100a-only MMA kinds (mxf4nvf4, kind::f8f6f4) stay dead: their
    // launchers remain fail-closed stubs on this arch.
    let portable = matches!(cuda_arch.as_str(), "89" | "90a");
    let hopper_mma = cuda_arch == "90a";
    assert!(
        !(cuda_arch != "120a" && std::env::var_os("MEMRA_CUTLASS").is_some()),
        "MEMRA_CUTLASS is sm_120a-only and cannot be enabled for this CUDA architecture"
    );
    let gencode = format!("arch=compute_{cuda_arch},code=sm_{cuda_arch}");
    if portable {
        println!("cargo:rustc-cfg=memra_portable_cuda");
    }
    if hopper_mma {
        println!("cargo:rustc-cfg=memra_hopper_mma");
    }
    if cuda_arch == "100a" {
        println!("cargo:rustc-cfg=memra_sm100_tcgen05");
    }
    // Runtime arch guard reads this (Engine::new): fatbins are single-arch SASS, so the
    // engine verifies the device's compute capability matches the built arch at init.
    println!("cargo:rustc-env=MEMRA_BUILT_CUDA_ARCH={cuda_arch}");

    for (src, env) in [
        ("cu/kernels.cu", "MEMRA_ENGINE_FATBIN"),
        ("cu/hybrid.cu", "MEMRA_HYBRID_FATBIN"),
        ("cu/kda.cu", "MEMRA_KDA_FATBIN"),
        ("cu/qmatvec.cu", "MEMRA_QMATVEC_FATBIN"),
        ("cu/flash_attn.cu", "MEMRA_FLASH_FATBIN"),
        ("cu/qmatvec_gemm.cu", "MEMRA_GEMM_FATBIN"),
        ("cu/moe_router.cu", "MEMRA_ROUTER_FATBIN"),
        ("cu/spec_sample.cu", "MEMRA_SAMPLE_FATBIN"),
        // WP-A day 22 (memra#536 Move 2 slice 3): the D2D receipt digest and the fault delay,
        // loaded by `tier_transfer::CudaTransfers::new_with_copy_stream` only.
        ("cu/tier_receipt.cu", "MEMRA_TIER_RECEIPT_FATBIN"),
    ] {
        println!("cargo:rerun-if-changed={src}");
        println!("cargo:rerun-if-changed=cu/wgmma_common.cuh");
        let stem = src.split('/').next_back().unwrap().trim_end_matches(".cu");
        let fatbin = out.join(format!("{stem}.fatbin"));
        let mut args = vec!["-gencode", &gencode, "-O3", "--fatbin"];
        if portable {
            args.push("-DMEMRA_PORTABLE_CUDA=1");
        }
        if hopper_mma {
            args.push("-DMEMRA_HOPPER_MMA=1");
        }
        // The hand-written mxf4nvf4 block-scale MMA in qmatvec_gemm.cu is an sm_120a
        // instruction encoding, so omit that opt-in kernel from the sm_100a fatbin.
        // qmatvec_gemm's optional native-FP4 fatbin kernel is still the sm_120a encoding and stays
        // omitted. B200 NVFP4 prefill now lives in the static MMQ archive instead: default W4A8
        // uses the valid int8 MMA base, while MEMRA_MMQ=1 selects the new tcgen05/TMEM W4A4 twin.
        if cuda_arch == "100a" && src == "cu/qmatvec_gemm.cu" {
            args.push("-DMEMRA_DISABLE_NATIVE_FP4=1");
        }
        // TUNE SEAM: MEMRA_FA_PP_MINBLOCKS sweeps fa_prefill_f32_pp's __launch_bounds__
        // min-blocks (occupancy vs register-spill tradeoff; H100 ncu 2026-07-26).
        println!("cargo:rerun-if-env-changed=MEMRA_FA_PP_MINBLOCKS");
        let fa_mb = std::env::var("MEMRA_FA_PP_MINBLOCKS").ok();
        let fa_mb_arg;
        if let (Some(mb), "cu/flash_attn.cu") = (&fa_mb, src) {
            fa_mb_arg = format!("-DFA_PP_MINBLOCKS={mb}");
            args.push(&fa_mb_arg);
        }
        args.extend(["-o", fatbin.to_str().unwrap(), src]);
        let status = Command::new(&nvcc).args(args).status().expect("spawn nvcc");
        assert!(status.success(), "nvcc fatbin build failed for {src}");
        println!("cargo:rustc-env={env}={}", fatbin.display());
    }

    // ---- The e4m3 KV fatbin variant of flash_attn.cu (kvbytes lane, 2026-07-08) ----
    // Same kernels/entry names, compile-time K/V cache format via -D: kf8vf8 is the module
    // gemma's GKV/WKV e4m3 layers load alongside the default (lib.rs `func_g`). The env-selected
    // trunk variants (VQ4, VF8, KF8, KF8VQ4) were removed 2026-09-05 (door sweep).
    {
        let fatbin = out.join("flash_attn_kf8vf8.fatbin");
        let mut args = vec![
            "-gencode".to_string(),
            gencode.clone(),
            "-O3".to_string(),
            "--fatbin".to_string(),
        ];
        if portable {
            args.push("-DMEMRA_PORTABLE_CUDA=1".to_string());
        }
        if hopper_mma {
            args.push("-DMEMRA_HOPPER_MMA=1".to_string());
        }
        args.extend([
            "-DMEMRA_KV_KFMT=1".to_string(),
            "-DMEMRA_KV_VFMT=2".to_string(),
            "-o".to_string(),
            fatbin.to_string_lossy().into_owned(),
            "cu/flash_attn.cu".to_string(),
        ]);
        let status = Command::new(&nvcc)
            .args(args)
            .status()
            .expect("spawn nvcc (flash_attn kf8vf8 variant)");
        assert!(
            status.success(),
            "nvcc fatbin build failed for the flash_attn kf8vf8 variant"
        );
        println!(
            "cargo:rustc-env=MEMRA_FLASH_FATBIN_KF8VF8={}",
            fatbin.display()
        );
    }

    // ---- Vendored llama MMQ GEMMs: a STATIC LIB with C-ABI host launchers (extern "C"). ----
    // Same kind as the CUTLASS artifact (a host-side launcher cannot go through the device-only fatbin
    // path), but ALWAYS built (no external header deps — fully ggml-decoupled). The launchers do
    // cudaFuncSetAttribute (>48KB dynamic smem) + the mul_mat_q kernel launch internally.
    // Called from Rust via FFI (mmq_ffi.rs), dispatched behind MEMRA_MMQ=1.
    // Two translation units: mmq_fp4.cu (Blackwell mxf4nvf4 W4A4) and mmq_q45k.cu
    // (Q4_K/Q5_K int8-MMA W4A8, sm_75+ portable). Both archived into one libmemra_mmq.a.
    {
        let mut objs: Vec<PathBuf> = Vec::new();
        // TUNE SEAM: MEMRA_MMQ_X_Q45K=64 rebuilds the k-quant MMQ with a 64-token tile
        // (47KB smem -> 2 CTA/SM vs 57KB/1; the q45k occupancy ceiling found by ncu).
        println!("cargo:rerun-if-env-changed=MEMRA_MMQ_X_Q45K");
        let q45k_x = std::env::var("MEMRA_MMQ_X_Q45K").ok();
        // TUNE SEAM: MEMRA_MMQ_X_Q4=64|96 shrinks the q4_0 MMQ token-tile (same axis as the
        // q45k/w4a8 seams; build-time — the tile is a template constant).
        println!("cargo:rerun-if-env-changed=MEMRA_MMQ_X_Q4");
        let q4_x = std::env::var("MEMRA_MMQ_X_Q4").ok();
        // TUNE SEAM: MEMRA_MMQ_X_W4A8=<n> rebuilds the NVFP4 W4A8 MMQ with an n-token tile.
        // ncu 2026-07-06 (27B pp6257): default 128x128 tile = 61KB smem = 1 CTA/SM ->
        // warps_active 16.7%, tensor pipe 53% — the same occupancy ceiling q45k hit.
        println!("cargo:rerun-if-env-changed=MEMRA_MMQ_X_W4A8");
        let w4a8_x = std::env::var("MEMRA_MMQ_X_W4A8").ok();
        // TUNE SEAM: MEMRA_MMQ_X_IQEXP=<n> rebuilds the expert-segmented MMQ with an n-token
        // tile (the round-45 kernel-rate dig: 128 costs 64 accumulator regs/thread at
        // occupancy 12.5%; smaller tiles trade MMA j-reuse for CTAs/SM).
        println!("cargo:rerun-if-env-changed=MEMRA_MMQ_X_IQEXP");
        let iqexp_x = std::env::var("MEMRA_MMQ_X_IQEXP").ok();
        // ROLLBACK SEAM: MEMRA_IQEXP_K16=1 rebuilds the IQ-experts + IQ4_XS-dense MMQ tiles with
        // the ORIGINAL m16n8k16.s8 MMA form instead of the m16n8k32.s8 default (lane/iq-experts-
        // k32, 2026-08-07). The int8 pipe is K-FREE on sm_120a (both forms 16.06 cyc/warp-MMA),
        // so k32 does 2x the K-work per instruction and halves the f32 fold arity; the merge is
        // legal because both per-16 scale slots of a 32-block are equal by loader construction.
        // Receipts: research/iq-k32-20260807/.
        println!("cargo:rerun-if-env-changed=MEMRA_IQEXP_K16");
        let iqexp_k16 = std::env::var("MEMRA_IQEXP_K16").ok();
        // TUNE SEAM: MEMRA_MMQ_Y_W4A8=64 halves the row tile AND warp count together (mmq_y =
        // nwarps*16) — 42KB->21KB tile_x, 2 CTA/SM. Unlike MMQ_X, this axis doesn't duplicate
        // weight reads, so it attacks the 16.7%-warps occupancy ceiling for free.
        println!("cargo:rerun-if-env-changed=MEMRA_MMQ_Y_W4A8");
        let w4a8_y = std::env::var("MEMRA_MMQ_Y_W4A8").ok();
        // CEILING PROBE (not a tune seam, NOT shippable): MEMRA_MMQ_FOLD_CEILING=1 rebuilds
        // the NVFP4 W4A8 MMQ with the per-k01 f32 scale-fold collapsed to an s32 add, to
        // measure the upper bound of every fold-removal lever. Output is NUMERICALLY WRONG
        // by construction — argmax/exactness gates WILL fail under it, by design. Receipts:
        // research/prefill-gemm-20260806/.
        println!("cargo:rerun-if-env-changed=MEMRA_MMQ_FOLD_CEILING");
        let w4a8_fold_ceiling = std::env::var("MEMRA_MMQ_FOLD_CEILING").ok();
        // ROLLBACK SEAM: MEMRA_MMQ_F8F4_PLAIN=1 rebuilds the f8f4 tile with the ORIGINAL plain
        // kind::f8f6f4 MMA instead of the block_scale form at the ue8m0 identity scale. The two
        // are bit-identical (0/128 elements differ, live-operand controls at 2^1/2^-1 exact) but
        // the plain form issues at 32.02 cyc/warp-MMA against the block_scale form's 16.06 — a
        // 1.994x MMA-rate difference. Receipts: research/w4a8-prefill-20260806/ slices 3-4.
        println!("cargo:rerun-if-env-changed=MEMRA_MMQ_F8F4_PLAIN");
        let f8f4_plain = std::env::var("MEMRA_MMQ_F8F4_PLAIN").ok();
        // TUNE SEAM: MEMRA_MMQ_X_FP8=<n> rebuilds the per-block FP8 prefill tile with an n-token
        // tile (it sets the WIDE candidate; the launcher picks between it and FP8_MMQ_X_SMALL per
        // call by wave fill). v1 needed this seam because its 128-token default sat below the Q8_0
        // floor at every 27B shape; v2's restructure made X=256 affordable and it is now the
        // default. Neither arm is MMA-bound at any width measured (110-130 TF) — though note the
        // denominator that judgement used was WRONG: the tile issued the PLAIN kind::f8f6f4 form,
        // a 155-TF ceiling, not the 381-TF block_scale class (corrected 2026-08-06; see the FORM
        // CHOICE block in cu/mmq_fp8_blk.cu).
        //
        // MEMRA_MMQ_Y_FP8 / MEMRA_MMQ_OCC_FP8 / MEMRA_MMQ_PIPE_FP8 (v2 slice-2/slice-3 seams)
        // CONCLUDED NEGATIVE and were DELETED per the flags doctrine (v0.69.0): halving Y splits
        // the same 8 warps across two CTAs, Y=128 with OCC=2 spills the accumulator, and cp.async
        // on the weight tile cannot pay while the activation tile stays a synchronous copy. The
        // record is research/fp8st-20260804/mmq-v2/RESULTS.jsonl slices 2-3 and experiment B.
        println!("cargo:rerun-if-env-changed=MEMRA_MMQ_X_FP8");
        let fp8_x = std::env::var("MEMRA_MMQ_X_FP8").ok();
        // ROLLBACK SEAM: MEMRA_MMQ_FP8BLK_PLAIN=1 rebuilds the per-block FP8 prefill tile with the
        // ORIGINAL plain kind::f8f6f4 MMA instead of the block_scale form at the ue8m0 identity
        // scale. The two are bit-identical (0/128 accumulator elements differ, live-operand controls
        // at 2^1/2^-1 exact) but the plain form issues at 32.02 cyc/warp-MMA against the block_scale
        // form's 16.06 — a 1.994x MMA-rate difference. Same seam the W4A8 tile carries as
        // MEMRA_MMQ_F8F4_PLAIN. Receipts: research/w4a8-prefill-20260806/ slices 3-4,
        // research/rp-on-st-20260806/.
        println!("cargo:rerun-if-env-changed=MEMRA_MMQ_FP8BLK_PLAIN");
        let fp8blk_plain = std::env::var("MEMRA_MMQ_FP8BLK_PLAIN").ok();
        // RESEARCH-INSTRUMENT ARM (not a runtime flag): ACCPROBE_F32_PLAIN=1 — see the
        // mmq_q8_0_f32acc.cu branch below.
        println!("cargo:rerun-if-env-changed=ACCPROBE_F32_PLAIN");
        let accprobe_plain = std::env::var("ACCPROBE_F32_PLAIN").ok();
        // RESEARCH-INSTRUMENT ARM (not a runtime flag, never a shipping default):
        // MEMRA_DSV4_FMAD=1 drops the blanket `-fmad=false` from cu/dsv4_gpu.cu so nvcc may
        // contract mul+add into FMA. It exists to PRICE the oracle-parity law's cost: a PTX
        // census of the default build (iteration 4, 2026-08-19) found ZERO fma.rn.f32 in the
        // hot decode kernels -- dsv4_fp4_gemm_sel (168 mul + 149 add), dsv4_gemv_bf16_m<T>
        // (64 + 72), dsv4_dots_f32acc_mrow<T> (128 + 136), dsv4_rmsnorm/rowsq_f32acc -- so
        // every hot inner loop pays two dependent instructions and two roundings where one
        // FMA would do. That is a candidate explanation for lane 9's banked 'sel is not
        // bandwidth-bound' and for the ~13-16% roofline position.
        // FORKS NUMERICS: contraction removes an intermediate rounding, so an FMA build is
        // NOT bit-identical to the CPU oracle's separate mul+add fixed tree. It must never
        // become a default without the standard fork discipline (derived thresholds banked
        // before rerun, CPU-oracle teacher-forcing, every disagreement an in-band near-tie).
        // It does NOT threaten the batched-vs-sequential bit-exactness property, because a
        // TU-wide compile flag lands identically on the M=1 and M=T twins.
        println!("cargo:rerun-if-env-changed=MEMRA_DSV4_FMAD");
        let dsv4_fmad = std::env::var("MEMRA_DSV4_FMAD").ok();
        // fp8_prefill.cu rides the same static-lib kind: a cuBLASLt host launcher + quantize
        // kernels for the MEMRA_PP_FP8 prefill path (runtime-gated; always built — no external
        // header deps beyond the CUDA toolkit, which ships cublasLt).
        for mmq_src in [
            "cu/mmq_fp4.cu",
            "cu/mmq_q45k.cu",
            "cu/mmq_nvfp4_w4a8.cu",
            "cu/mmq_iq_experts.cu",
            "cu/mmq_q8_0.cu",
            "cu/mmq_q4_0.cu",
            "cu/fp8_prefill.cu",
            "cu/f16_prefill.cu",
            "cu/mmq_nvfp4_f8f4.cu",
            "cu/fa3_prefill.cu",
            "cu/moe_f16_grouped.cu",
            "cu/fp8_blk_dequant.cu",
            "cu/mmq_fp8_blk.cu",
            // Q1 instrument of the FP8-ST v3 gate lane: the Q8_0 MMQ floor with the
            // accumulator (s32 vs f32) as its ONE free variable, to price v2's
            // "what is left is the f32 accumulator" ceiling claim before anyone builds
            // a v3. Research-only: no dispatch seam, never linked into a serving path.
            // Its f8f6f4 arm is guarded by __CUDA_ARCH__ >= 1200 inside the TU (>= 1000
            // wrongly admitted sm_100a, the one arch that rejects the instruction — fixed
            // lane/glm5-b200-prep-20260901), so it needs no portable stub (the s32 arm
            // builds everywhere; every non-120a arch takes the fail-closed __trap() arm).
            "cu/mmq_q8_0_f32acc.cu",
            // MLA (multi-head latent attention) forward: the glm-dsa / glm5_next attention
            // core. Portable CUDA C (no tensor-core intrinsics, no arch stub needed).
            // Deliberately NOT compiled -fmad=false like dsv4_gpu.cu below: that flag serves
            // that lane's bit-parity-with-oracle contract, while the MLA gates are maxdiff
            // bounds (the online-softmax tiling already reorders accumulation vs the CPU
            // oracle), so forbidding contraction would cost speed and buy nothing.
            "cu/mla_attn.cu",
            // DeepSeek-V4-Flash trunk bring-up kernels + bf16 cuBLASLt GEMM (lane 4).
            // Portable CUDA C (no tensor-core intrinsics) — no arch stub needed. Compiled
            // with -fmad=false below: the kernels mirror the lane-3 CPU oracle's separate
            // mul+add rounding; default FMA contraction would silently fork the f32-island
            // arithmetic from the oracle contract.
            "cu/dsv4_gpu.cu",
            "cu/dsv4_sampler.cu",
            // Pinned MiMo source QKV shard outputs gathered on-device.
            "cu/mimo_qkv_gather.cu",
            // Bounded f32 MiMo attention reference component with learned sink.
            "cu/mimo_sink_attn.cu",
            // Source-inspection split attention over q8_0 K and NVFP4 V.
            "cu/mimo_mixed_attn.cu",
            // Source-inspection NVFP4 cache row codec.
            "cu/mimo_kv_nvfp4.cu",
            // Small-message cross-rank all-reduce for TP decode (lane/tp-allreduce-20260906).
            // Portable CUDA C; peer stores plus a bounded flag wait, no oracle to match, so it
            // takes the default flags rather than dsv4_gpu.cu's -fmad=false.
            "cu/tp_ar.cu",
        ] {
            println!("cargo:rerun-if-changed={mmq_src}");
            println!("cargo:rerun-if-changed=cu/mmq_common.cuh");
            println!("cargo:rerun-if-changed=cu/dsv4_dense_m1_exact_tail.cuh");
            println!("cargo:rerun-if-changed=cu/dsv4_dense_tile.cuh");
            println!("cargo:rerun-if-changed=cu/mmq_mma_i8.cuh");
            println!("cargo:rerun-if-changed=cu/sm100_blockscale_layout.cuh");
            // fa3_prefill.cu includes the shared wgmma header (dedup 2026-08-21).
            println!("cargo:rerun-if-changed=cu/wgmma_common.cuh");
            let compile_src =
                if !matches!(cuda_arch.as_str(), "120a" | "100a") && mmq_src == "cu/mmq_fp4.cu" {
                    // NVFP4 W4A4 has two native Blackwell programs: sm_120a warp MMA and the
                    // sm_100a tcgen05/TMEM twin. Other architectures retain the fail-closed ABI.
                    "cu/mmq_fp4_stub.cu"
                } else if !matches!(cuda_arch.as_str(), "120a" | "100a")
                    && mmq_src == "cu/mmq_nvfp4_w4a8.cu"
                {
                    // POLARITY FIXED 2026-08-23: this tested `portable` (89|90a), so sm_100a got the
                    // REAL file — and ptxas refuses it there:
                    //   error : Instruction 'mma with block scale' not supported on .target 'sm_100a'
                    // at ~40 sites. So `MEMRA_CUDA_ARCH=100a` could not build AT ALL, and nothing
                    // noticed because no workflow compiled that arch. `cu/mmq_fp4.cu` one branch up
                    // already had the correct `cuda_arch != "120a"` test; this is the same class of
                    // kernel (block-scale MMA, sm_120a-only in practice) and now shares the test.
                    //
                    // The bug's real cost was not sm_100a specifically: `portable` is a two-arch
                    // ENUMERATION, so every future non-120a arch inherited the breakage silently,
                    // while `!= 120a` is a property. Prefer the property.
                    "cu/mmq_nvfp4_w4a8_stub.cu"
                } else if !matches!(cuda_arch.as_str(), "120a" | "100a")
                    && mmq_src == "cu/mmq_fp8_blk.cu"
                {
                    // Per-block FP8 MMQ: same .kind::f8f6f4 gate as the W4A8/F8F4 launchers, and
                    // therefore the SAME polarity fix (2026-08-23). Testing `portable` sent sm_100a
                    // to the real file, which ptxas rejects at ~400 sites. Found by fixing the
                    // mmq_nvfp4_w4a8 branch above and watching the failure MOVE here — the two are
                    // one class, not two bugs.
                    "cu/mmq_fp8_blk_stub.cu"
                } else {
                    mmq_src
                };
            println!("cargo:rerun-if-changed={compile_src}");
            let stem = mmq_src
                .split('/')
                .next_back()
                .unwrap()
                .trim_end_matches(".cu");
            let obj = out.join(format!("{stem}.o"));
            let mut args: Vec<String> = vec![
                "-gencode".into(),
                gencode.clone(),
                "-O3".into(),
                "-std=c++17".into(),
                "--expt-relaxed-constexpr".into(),
            ];
            if mmq_src.ends_with("mmq_q45k.cu")
                && let Some(x) = &q45k_x
            {
                args.push(format!("-DMMQ_X={x}"));
            }
            if mmq_src.ends_with("mmq_q4_0.cu")
                && let Some(x) = &q4_x
            {
                args.push(format!("-DMMQ_X={x}"));
            }
            if mmq_src.ends_with("mmq_nvfp4_w4a8.cu") {
                if let Some(x) = &w4a8_x {
                    args.push(format!("-DMMQ_X={x}"));
                }
                if let Some(y) = &w4a8_y {
                    args.push(format!("-DMMQ_Y={y}"));
                }
                if let Some(v) = &w4a8_fold_ceiling {
                    args.push(format!("-DMEMRA_MMQ_FOLD_CEILING={v}"));
                }
                if f8f4_plain.as_deref() == Some("1") || cuda_arch == "100a" {
                    args.push("-DMEMRA_F8F4_PLAIN_MMA".into());
                }
            }
            if mmq_src.ends_with("mmq_fp4.cu") && cuda_arch == "100a" {
                args.push("-DMEMRA_SM100_TCGEN05=1".into());
            }
            if mmq_src.ends_with("mmq_iq_experts.cu") {
                if let Some(x) = &iqexp_x {
                    args.push(format!("-DMMQ_X={x}"));
                }
                if iqexp_k16.as_deref() == Some("1") {
                    args.push("-DMEMRA_IQEXP_K16_MMA".into());
                }
            }
            // Per-block FP8 MMQ token-tile geometry (X only — the Y/OCC/PIPE seams concluded
            // negative and were deleted; see the TUNE SEAM note above).
            if mmq_src.ends_with("mmq_fp8_blk.cu") {
                if let Some(x) = &fp8_x {
                    args.push(format!("-DFP8_MMQ_X={x}"));
                }
                if fp8blk_plain.as_deref() == Some("1") || cuda_arch == "100a" {
                    args.push("-DMEMRA_FP8BLK_PLAIN_MMA".into());
                }
                if cuda_arch == "100a" {
                    args.push("-DMEMRA_SM100_TCGEN05=1".into());
                }
            }
            // RESEARCH INSTRUMENT ARM: ACCPROBE_F32_PLAIN=1 rebuilds the fp8-v3-gate Q1 instrument's
            // F32 arm with the ORIGINAL plain kind::f8f6f4 MMA. Needed to REPRODUCE the published
            // +19.8/+20.2 delta_pp (research/fp8v3-gate-20260805/), which was measured with that
            // form — and which is confounded, because the plain form issues at 32.02 cyc/warp-MMA
            // against the S32 arm's 16.06, so "f32 vs s32 accumulate" moved the MMA interval too.
            // The default arm now uses the block_scale form at the ue8m0 identity scale (bit-identical
            // product, 16.06 cyc), which is what makes the instrument single-variable.
            if mmq_src.ends_with("mmq_q8_0_f32acc.cu") && accprobe_plain.as_deref() == Some("1") {
                args.push("-DMEMRA_ACCPROBE_PLAIN_MMA".into());
            }
            if mmq_src.ends_with("fa3_prefill.cu") && cuda_arch != "90a" {
                args.push("-DMEMRA_FA3_STUB".into());
            }
            if mmq_src.ends_with("dsv4_gpu.cu") {
                // Oracle-parity law (see the TU header): no FMA contraction in the dsv4
                // f32-island kernels. MEMRA_DSV4_FMAD=1 lifts it for the pricing instrument
                // documented above -- research only, and it is echoed so no build log is
                // ambiguous about which arm produced a binary.
                if dsv4_fmad.as_deref() == Some("1") {
                    println!(
                        "cargo:warning=dsv4_gpu.cu built with FMA CONTRACTION ENABLED \
                         (MEMRA_DSV4_FMAD=1) -- RESEARCH INSTRUMENT, forks the CPU oracle"
                    );
                } else {
                    args.push("-fmad=false".into());
                }
            }
            args.extend([
                "-c".into(),
                compile_src.into(),
                "-o".into(),
                obj.to_str().unwrap().into(),
            ]);
            let status = Command::new(&nvcc)
                .args(&args)
                .status()
                .expect("spawn nvcc (mmq)");
            assert!(
                status.success(),
                "nvcc static-lib build failed for {mmq_src}"
            );
            objs.push(obj);
        }
        let lib = out.join("libmemra_mmq.a");
        let _ = std::fs::remove_file(&lib);
        let mut ar_args = vec!["crus".to_string(), lib.to_str().unwrap().to_string()];
        ar_args.extend(objs.iter().map(|o| o.to_str().unwrap().to_string()));
        let status = Command::new("ar")
            .args(&ar_args)
            .status()
            .expect("spawn ar (mmq)");
        assert!(status.success(), "ar failed for {}", lib.display());
        // The archive is BUNDLED into the rlib, and a .cu-only edit changes no rustc input: with
        // RUSTC_WRAPPER=sccache (the rig default) the cached rlib came back carrying the previous
        // archive while every timestamp under target/ looked fresh (2026-09-06: three rig timing
        // rounds and one gate ran a kernel two edits old). Hashing the archive into a rustc env
        // makes the kernel bytes a rustc input, so any kernel change forces a real compile.
        {
            use std::hash::Hasher;
            let bytes = std::fs::read(&lib).expect("read libmemra_mmq.a for the archive hash");
            let mut h = std::collections::hash_map::DefaultHasher::new();
            h.write(&bytes);
            println!("cargo:rustc-env=MEMRA_MMQ_ARCHIVE_HASH={:016x}", h.finish());
        }
        // rustc-link-lib (NOT rustc-link-arg): link-arg applies only to THIS package's own
        // binaries, so downstream crates (memra-server) failed to link the MMQ symbols. link-lib
        // metadata propagates through the dependency graph; +whole-archive keeps the CUDART
        // fatbin-registration global ctor alive (same MANDATORY reasoning as the CUTLASS link).
        println!("cargo:rustc-link-search=native={}", out.display());
        println!("cargo:rustc-link-lib=static:+whole-archive=memra_mmq");
        let cuda_lib = std::path::Path::new(&nvcc)
            .parent()
            .and_then(|p| p.parent())
            .map(|p| p.join("lib64"))
            .unwrap_or_else(|| std::path::PathBuf::from("/usr/local/cuda-13.1/lib64"));
        println!("cargo:rustc-link-search=native={}", cuda_lib.display());
        // dylib link-lib (not link-arg) so cudart/stdc++ propagate to downstream binaries too.
        println!("cargo:rustc-link-lib=dylib=cudart");
        println!("cargo:rustc-link-lib=dylib=stdc++");
        // fp8_prefill.cu calls the cuBLASLt host API directly (same lib64 search path as cudart).
        println!("cargo:rustc-link-lib=dylib=cublasLt");
        // plain cublas: the MoE grouped f16 GEMM (cublasGemmGroupedBatchedEx, CUDA >= 12.5)
        println!("cargo:rustc-link-lib=dylib=cublas");
        // fa3_prefill.cu calls the driver API (cuTensorMapEncodeTiled). GPU-less build
        // machines (CI compile gate, release matrix) have no driver libcuda.so — the
        // toolkit's stubs dir satisfies the link; at runtime ld.so resolves the real
        // libcuda.so.1 from the driver (the stubs dir is not on any runtime path).
        println!(
            "cargo:rustc-link-search=native={}/stubs",
            cuda_lib.display()
        );
        println!("cargo:rustc-link-lib=dylib=cuda");
    }

    // DSV4 dense tensor-core path (memra #472). Deliberately INDEPENDENT of the MEMRA_CUTLASS
    // block below: that one carries the fp4 experiment, sets the memra_cutlass cfg and pulls the
    // fp4 Rust call sites in with it, and its archive hit an lld symbol-ordering failure on this
    // box. This path needs none of that. It has its own switch, its own archive, and NO cfg,
    // because cu/dsv4_gpu.cu declares memra_dsv4_dense_cutlass_fp8 WEAK: when this block does not
    // run the symbol is null and the scalar kernel runs untouched.
    //
    // -fmad=false to match cu/dsv4_gpu.cu, because this family's numerics are the point.
    if std::env::var("MEMRA_DSV4_CUTLASS").is_ok() {
        println!("cargo:rerun-if-env-changed=MEMRA_DSV4_CUTLASS");
        let dense_src = "cu/dsv4_dense_cutlass.cu";
        println!("cargo:rerun-if-changed={dense_src}");
        let root = std::env::var("MEMRA_CUTLASS_ROOT")
            .expect("MEMRA_DSV4_CUTLASS requires MEMRA_CUTLASS_ROOT");
        let obj = out.join("dsv4_dense_cutlass.o");
        let lib = out.join("libmemra_dsv4_cutlass.a");
        let status = Command::new(&nvcc)
            .args([
                "-gencode",
                "arch=compute_120a,code=sm_120a",
                "-O3",
                "-std=c++17",
                "-fmad=false",
                "--expt-relaxed-constexpr",
                "-I",
                &format!("{root}/include"),
                "-I",
                &format!("{root}/tools/util/include"),
                "-c",
                dense_src,
                "-o",
                obj.to_str().unwrap(),
            ])
            .status()
            .expect("spawn nvcc (dsv4 dense cutlass)");
        assert!(status.success(), "nvcc build failed for {dense_src}");
        let _ = std::fs::remove_file(&lib);
        let status = Command::new("ar")
            .args(["crus", lib.to_str().unwrap(), obj.to_str().unwrap()])
            .status()
            .expect("spawn ar (dsv4 dense cutlass)");
        assert!(status.success(), "ar failed for {}", lib.display());
        // link-LIB, not link-ARG, and this is the whole reason the first attempt produced a
        // binary byte-identical to the baseline with the symbol still weak-undefined:
        // `cargo:rustc-link-arg` applies only to the crate whose build script emitted it and to
        // binaries in the SAME package, so it never reached memra-server in another package.
        // `cargo:rustc-link-lib` does propagate to downstream crates, and the
        // `static:+whole-archive` modifier carries the whole-archive that the CUDART
        // fatbin-registration ctor needs (without it the device kernels silently never register).
        println!("cargo:rustc-link-search=native={}", out.display());
        println!("cargo:rustc-link-lib=static:+whole-archive=memra_dsv4_cutlass");
        println!("cargo:rustc-link-lib=dylib=cudart");
        println!("cargo:rustc-link-lib=dylib=stdc++");
        let _ = &lib;
    }

    // ---- CUTLASS sm_120a NVFP4 GEMM: a STATIC LIB (7th artifact, different kind), NOT a fatbin ----
    // CUTLASS needs its host-side GemmUniversalAdapter::run() (host C++), so it cannot go through the
    // fatbin/load_module path above. It is compiled to an object, archived, and whole-archived at link.
    // Additive: the 6-fatbin loop above is byte-for-byte unchanged (the parallel flash_attn.cu FA build
    // is untouched). Guarded by MEMRA_CUTLASS so the default build is unaffected until Phase 0 lands.
    if std::env::var("MEMRA_CUTLASS").is_ok() {
        let cutlass_src = "cu/cutlass_fp4_sm120.cu";
        println!("cargo:rerun-if-changed={cutlass_src}");
        // CUTLASS 4.x header tree (on-box, probe-verified). TODO Phase 1: vendor a pinned tree into the
        // repo for reproducibility rather than pointing at the venv install.
        let cutlass_root = std::env::var("MEMRA_CUTLASS_ROOT").unwrap_or_else(|_| {
            "/home/avifenesh/.venvs/torch/lib/python3.12/site-packages/flashinfer/data/cutlass"
                .into()
        });
        let cutlass_inc = format!("{cutlass_root}/include");
        let cutlass_util = format!("{cutlass_root}/tools/util/include");
        let obj = out.join("cutlass_fp4_sm120.o");
        let lib = out.join("libmemra_cutlass.a");
        let status = Command::new(&nvcc)
            .args([
                "-gencode",
                "arch=compute_120a,code=sm_120a",
                "-O3",
                "-std=c++17",
                "--expt-relaxed-constexpr",
                "-DENABLE_BF16",
                "-DENABLE_FP4",
                "-DCUTLASS_ENABLE_GDC_FOR_SM100=1",
                "-I",
                &cutlass_inc,
                "-I",
                &cutlass_util,
                "-c",
                cutlass_src,
                "-o",
                obj.to_str().unwrap(),
            ])
            .status()
            .expect("spawn nvcc (cutlass)");
        assert!(
            status.success(),
            "nvcc static-lib build failed for {cutlass_src}"
        );
        let _ = std::fs::remove_file(&lib);
        let status = Command::new("ar")
            .args(["crus", lib.to_str().unwrap(), obj.to_str().unwrap()])
            .status()
            .expect("spawn ar");
        assert!(status.success(), "ar failed for {}", lib.display());
        // --whole-archive is MANDATORY: a plain static link drops the CUDART fatbin-registration global
        // ctor (_ZL24__sti____cudaRegisterAllv in .init_array) -> the device kernel silently never
        // registers -> no-kernel launch failure. Verified on-box (plan §2.2).
        println!("cargo:rustc-link-search=native={}", out.display());
        println!("cargo:rustc-link-arg=-Wl,--whole-archive");
        println!("cargo:rustc-link-arg={}", lib.display());
        println!("cargo:rustc-link-arg=-Wl,--no-whole-archive");
        // libstdc++ AFTER the archive (link-arg, not link-lib) so the function-local-static guard
        // symbols (__cxa_guard_acquire/release, from CUTLASS's tile_atom_to_shape statics) resolve.
        // A plain `link-lib=stdc++` can be ordered before the archive under -nodefaultlibs/lld and
        // leave them undefined for bins other than cutlass-smoke (whole-archive applies to ALL bins).
        // cudart (CUTLASS host adapter uses the runtime API) and stdc++ BOTH as trailing link-args so
        // they sit AFTER the whole-archive; the cudart fatbin-registration ctors + the C++ static
        // guards in cutlass_fp4_sm120.o resolve against them. The CUDA lib dir is needed for -lcudart.
        let cuda_lib = std::path::Path::new(&nvcc)
            .parent()
            .and_then(|p| p.parent())
            .map(|p| p.join("lib64"))
            .unwrap_or_else(|| std::path::PathBuf::from("/usr/local/cuda-13.1/lib64"));
        println!("cargo:rustc-link-search=native={}", cuda_lib.display());
        // dylib link-lib (not link-arg) so cudart/stdc++ propagate to downstream binaries too.
        println!("cargo:rustc-link-lib=dylib=cudart");
        println!("cargo:rustc-link-lib=dylib=stdc++");
        // Let the smoke-test bin gate compile out cleanly when CUTLASS is not built.
        println!("cargo:rustc-cfg=memra_cutlass");
    }
}

/// Parse docs/FLAGS.md into the boot audit's registry (`env_audit.rs` includes the output).
///
/// Mirrors the flags census (`tools/check-flags.sh`): a `MEMRA_...` token ANYWHERE outside the
/// "Removed" ledgers documents a name (a trailing `*` documents a prefix family), because that
/// is what the census accepts today and the audit must not refuse a name the census passed.
/// Definitions are stricter: the first cell of a live table row (`| \`MEMRA_X\` | ...`).
/// A ledger is any heading, at any level, whose text starts with `Removed`; it ends at the next
/// heading of the same or a shallower level. Inside it, a definition block is a `- ` bullet or
/// a `| ` row together with its indented continuation lines, and EVERY non-prefix name in the
/// block is retired (ledgers delete several doors in one bullet), minus names that still have a
/// live DEFINITION (prose mentions in live rows do not un-retire a door). A retired name that a
/// prose wildcard would also cover stays retired: the audit checks the retired set first.
///
/// A packaged crate has no docs/ tree: the registry is then empty, announced with a cargo
/// warning, and the audit is inert (`memra_env_registry_present` is not set).
fn generate_env_registry(out: &Path) {
    println!("cargo:rustc-check-cfg=cfg(memra_env_registry_present)");
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let flags = manifest.join("../../docs/FLAGS.md");
    println!("cargo:rerun-if-changed={}", flags.display());
    let text = match std::fs::read_to_string(&flags) {
        Ok(t) => t,
        Err(e) => {
            println!(
                "cargo:warning=env audit registry: {} not readable ({e}); packaged build, the \
                 MEMRA_* boot audit is inert in this binary",
                flags.display()
            );
            std::fs::write(
                out.join("memra_env_registry.rs"),
                "// generated by build.rs: docs/FLAGS.md absent, registry empty\n\
                 pub const LEGAL_NAMES: &[&str] = &[];\n\
                 pub const LEGAL_PREFIXES: &[&str] = &[];\n\
                 pub const OWNED_FAMILIES: &[&str] = &[];\n\
                 pub const RETIRED: &[(&str, &str)] = &[];\n",
            )
            .expect("write env registry");
            return;
        }
    };
    println!("cargo:rustc-cfg=memra_env_registry_present");
    fn tokens(cell: &str) -> Vec<(String, bool)> {
        let bytes = cell.as_bytes();
        let mut out = Vec::new();
        let mut i = 0;
        while let Some(rel) = cell[i..].find("MEMRA_") {
            let start = i + rel;
            let mut end = start;
            while end < bytes.len()
                && (bytes[end].is_ascii_uppercase()
                    || bytes[end].is_ascii_digit()
                    || bytes[end] == b'_')
            {
                end += 1;
            }
            let prefix = end < bytes.len() && bytes[end] == b'*';
            let stem = cell[start..end].trim_end_matches('_');
            if stem.len() > "MEMRA".len() {
                out.push((
                    if prefix {
                        format!("{stem}_")
                    } else {
                        stem.to_string()
                    },
                    prefix,
                ));
            }
            i = end.max(start + 1);
        }
        out
    }
    fn heading(line: &str) -> Option<(usize, &str)> {
        let level = line.bytes().take_while(|b| *b == b'#').count();
        if level == 0 || level > 6 || !line[level..].starts_with(' ') {
            return None;
        }
        Some((level, line[level..].trim()))
    }
    let mut defs = std::collections::BTreeSet::new();
    let mut legal = std::collections::BTreeSet::new();
    let mut prefixes = std::collections::BTreeSet::new();
    let mut ledger: std::collections::BTreeMap<String, String> = Default::default();
    let mut removed: Option<(usize, String)> = None; // (heading level, heading text)
    let mut in_block = false;
    for line in text.lines() {
        if let Some((level, text)) = heading(line) {
            in_block = false;
            removed = match &removed {
                Some((lvl, _)) if level > *lvl && !text.starts_with("Removed") => removed.clone(),
                _ if text.starts_with("Removed") => Some((level, text.to_string())),
                _ => None,
            };
            continue;
        }
        if let Some((_, h)) = &removed {
            // a bullet, a table row, or a line that opens with a backticked name (the lane-local
            // `### Removed experiments` ledger writes `\`MEMRA_X\`: removed before merge`)
            let starts =
                line.starts_with("- ") || line.starts_with("| ") || line.starts_with("`MEMRA_");
            let continues = in_block && line.starts_with(' ') && !line.trim().is_empty();
            if starts || continues {
                in_block = true;
                for (name, prefix) in tokens(line) {
                    if !prefix {
                        ledger.entry(name).or_insert_with(|| h.clone());
                    }
                }
            } else {
                in_block = false;
            }
            continue;
        }
        for (name, prefix) in tokens(line) {
            if prefix {
                prefixes.insert(name);
            } else {
                legal.insert(name);
            }
        }
        if let Some(rest) = line.strip_prefix("| ") {
            let mut cells = rest.split('|');
            let first_cell = cells.next().unwrap_or("");
            // A live row whose default cell opens with REMOVED documents a door that is gone
            // (`| \`MEMRA_SEL_DOWN8\` | **REMOVED 2026-08-29** | ...`): the name stays documented
            // for the census, and retired for the audit, with the row as its ledger.
            let second_cell = cells.next().unwrap_or("").trim().trim_start_matches('*');
            let removed_row = second_cell.starts_with("REMOVED");
            for (name, prefix) in tokens(first_cell) {
                if prefix {
                    continue;
                }
                if removed_row {
                    ledger.entry(name).or_insert_with(|| {
                        format!("row marked {}", second_cell.trim_end_matches('*'))
                    });
                } else {
                    defs.insert(name);
                }
            }
        }
    }
    // Owned families come from the code, not from the catalog: a `MEMRA_<FAMILY>_` prefix that
    // any crate in the workspace READS under three or more distinct names (every `crates/*/src`,
    // the census's discovery rule; memra-kv, memra-gguf, memra-lanes and memra-tokenizer all read
    // doors). The catalog also documents shell-side knobs (`MEMRA_CI_*`, `MEMRA_GATE_*`, ...) that
    // no Rust reads, and a typo among those is the launcher's business, not a boot refusal.
    // Sources are scanned once per engine build (no rerun-if-changed on them: a sibling-crate
    // edit must not recompile the CUDA crate; the families move rarely and the FLAGS.md rerun
    // catches a catalog change).
    let mut read_names = std::collections::BTreeSet::new();
    let mut roots = Vec::new();
    if let Ok(rd) = std::fs::read_dir(manifest.join("..")) {
        for entry in rd.flatten() {
            let src = entry.path().join("src");
            if src.is_dir() {
                roots.push(src);
            }
        }
    }
    roots.sort();
    for root in roots {
        let mut stack = vec![root];
        while let Some(dir) = stack.pop() {
            let Ok(rd) = std::fs::read_dir(&dir) else {
                continue;
            };
            for entry in rd.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                } else if path.extension().is_some_and(|x| x == "rs") {
                    let Ok(text) = std::fs::read_to_string(&path) else {
                        continue;
                    };
                    for line in text.lines() {
                        if !(line.contains("env::var") || line.contains("env!(")) {
                            continue;
                        }
                        for (name, prefix) in tokens(line) {
                            if !prefix {
                                read_names.insert(name);
                            }
                        }
                    }
                }
            }
        }
    }
    let mut family_counts: std::collections::BTreeMap<String, usize> = Default::default();
    for n in &read_names {
        if let Some(rest) = n.strip_prefix("MEMRA_")
            && let Some(cut) = rest.find('_')
        {
            *family_counts
                .entry(n[..("MEMRA_".len() + cut + 1)].to_string())
                .or_default() += 1;
        }
    }
    let owned: Vec<String> = family_counts
        .into_iter()
        .filter(|(_, c)| *c >= 3)
        .map(|(f, _)| f)
        .collect();
    assert!(
        owned.len() > 5,
        "env audit registry: only {} owned families derived from source",
        owned.len()
    );
    let retired: Vec<(String, String)> = ledger
        .into_iter()
        .filter(|(n, _)| !defs.contains(n))
        .collect();
    for (n, _) in &retired {
        legal.remove(n);
    }
    assert!(
        legal.len() > 200,
        "env audit registry: only {} legal names parsed from FLAGS.md",
        legal.len()
    );
    assert!(
        !retired.is_empty(),
        "env audit registry: no retired doors parsed from FLAGS.md"
    );
    let mut src = String::new();
    src.push_str("// generated by build.rs from docs/FLAGS.md; do not edit\n");
    src.push_str("pub const LEGAL_NAMES: &[&str] = &[\n");
    for n in &legal {
        src.push_str(&format!("    {n:?},\n"));
    }
    src.push_str("];\npub const LEGAL_PREFIXES: &[&str] = &[\n");
    for p in &prefixes {
        src.push_str(&format!("    {p:?},\n"));
    }
    src.push_str("];\npub const OWNED_FAMILIES: &[&str] = &[\n");
    for f in &owned {
        src.push_str(&format!("    {f:?},\n"));
    }
    src.push_str("];\npub const RETIRED: &[(&str, &str)] = &[\n");
    for (n, h) in &retired {
        src.push_str(&format!("    ({n:?}, {h:?}),\n"));
    }
    src.push_str("];\n");
    std::fs::write(out.join("memra_env_registry.rs"), src).expect("write env registry");
}
