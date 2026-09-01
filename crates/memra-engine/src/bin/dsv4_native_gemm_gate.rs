//! dsv4-native-gemm-gate: lane-7 gate (a) — kernel-level paired gate for the native
//! quantized expert GEMM arms (NVFP4 trunk / MXFP4 MTP). Protocol + derivations banked
//! in wt-dsv4-loader research/dsv4-flash-loader-20260818/RECEIPTS.md "Lane 7" BEFORE
//! this binary existed.
//!
//! For a sample of real expert weights (both recipes, incl. the lane-1 pin tensors
//! layers.20.experts.7 and mtp.0.experts.7) and REAL activation rows captured from the
//! fixture-prompt GPU forward (moe_x capture; the w2 input h is derived from those rows
//! through the quantized CPU chain — the reference Expert.forward order, weight inside
//! the quantization), three comparisons per (expert, projection):
//!   1. GPU native kernel vs the CPU BIT-EXACT MIRROR (same act_quant grid ops, same
//!      thread-strided group partials, same halving tree, all f32): bit-exact REQUIRED.
//!   2. GPU native vs an f64-ordered reference of the same quantized arithmetic:
//!      |err| <= n_ops · 2^-24 · Σ|p_scaled|  (the stated f32-accumulation error model;
//!      n_ops = gs-1 sequential + per-thread group adds + 7 tree levels).
//!   3. native vs the lane-4 bf16-dequant rung on the SAME inputs: the numeric-class
//!      shift, measured against the banked prediction |Δ| ≈ (u_q/√3)·‖x⊙w‖₂ per output
//!      (informational characterization, ratio distribution printed).
//!
//! Usage: MEMRA_DSV4_EXPERT_ARM=native dsv4-native-gemm-gate <model-dir> <fixtures.json>
//!        [dev0,dev1]                                          exit 0 = PASS

use cudarc::driver::{DevicePtr, DevicePtrMut};
use memra_engine::dsv4_ffi as k;
use memra_engine::dsv4_gpu::{Dsv4Gpu, ExpertArm, ExpertKind, GpuCapture};
use memra_gguf::dsv4::{E2M1, e8m0_to_f32};
use memra_gguf::dsv4_forward::{FixtureSpec, U_FP8, pow2_ceil};
use memra_gguf::nvfp4_repack::{f32_to_fp8_e4m3, fp8_e4m3_to_f32};
use std::os::raw::c_void;
use std::path::Path;

const THREADS: usize = 128; // must equal the kernel launch (fixed-tree width)

/// CPU mirror of memra_dsv4_act_quant_fp8: per-row-per-128 codes + pow2 scales.
/// Identical ops to the GPU kernel (amax is order-free; the rest is elementwise).
fn quant_rows(x: &[f32], rows: usize, kdim: usize) -> (Vec<u8>, Vec<f32>) {
    let kq = kdim / 128;
    let inv = (1.0f64 / 448.0) as f32;
    let mut codes = vec![0u8; rows * kdim];
    let mut scales = vec![0f32; rows * kq];
    for r in 0..rows {
        for g in 0..kq {
            let grp = &x[r * kdim + g * 128..r * kdim + g * 128 + 128];
            let mut amax = 0f32;
            for v in grp {
                amax = amax.max(v.abs());
            }
            amax = amax.max(1e-4);
            let s = pow2_ceil(amax * inv);
            scales[r * kq + g] = s;
            for (i, v) in grp.iter().enumerate() {
                codes[r * kdim + g * 128 + i] = f32_to_fp8_e4m3((v / s).clamp(-448.0, 448.0));
            }
        }
    }
    (codes, scales)
}

struct ExpertRaw {
    kind: ExpertKind,
    w: Vec<u8>,  // nibble pairs [n, k/2]
    sc: Vec<u8>, // e4m3 per-16 (nvfp4) or e8m0 per-32 (mxfp4)
    scale2: f32, // nvfp4 only
    n: usize,
    kdim: usize,
}

/// CPU BIT-EXACT mirror of dsv4_fp4_gemm_kernel: thread-strided groups + halving tree.
fn mirror_gemm(codes: &[u8], scales: &[f32], ex: &ExpertRaw, g: usize) -> Vec<f32> {
    let gs = match ex.kind {
        ExpertKind::Nvfp4 => 16usize,
        ExpertKind::Mxfp4 => 32,
    };
    let ngroups = ex.kdim / gs;
    let kq = ex.kdim / 128;
    let mut out = vec![0f32; g * ex.n];
    for row in 0..g {
        let arow = &codes[row * ex.kdim..(row + 1) * ex.kdim];
        let asrow = &scales[row * kq..(row + 1) * kq];
        for col in 0..ex.n {
            let wrow = &ex.w[col * ex.kdim / 2..(col + 1) * ex.kdim / 2];
            let srow = &ex.sc[col * ngroups..(col + 1) * ngroups];
            let mut partials = [0f32; THREADS];
            for (tid, part) in partials.iter_mut().enumerate() {
                let mut j = tid;
                while j < ngroups {
                    let k0 = j * gs;
                    let mut sub = 0f32;
                    for i in 0..gs {
                        let kk = k0 + i;
                        let byte = wrow[kk >> 1];
                        let code = if kk & 1 == 1 { byte >> 4 } else { byte & 0x0F };
                        sub += fp8_e4m3_to_f32(arow[kk]) * E2M1[code as usize];
                    }
                    let ws = match ex.kind {
                        ExpertKind::Nvfp4 => fp8_e4m3_to_f32(srow[j]) * ex.scale2,
                        ExpertKind::Mxfp4 => e8m0_to_f32(srow[j]),
                    };
                    let sc = ws * asrow[k0 / 128];
                    *part += sub * sc;
                    j += THREADS;
                }
            }
            // fixed halving tree (kernel order)
            let mut off = THREADS >> 1;
            while off > 0 {
                for t in 0..off {
                    partials[t] += partials[t + off];
                }
                off >>= 1;
            }
            out[row * ex.n + col] = partials[0];
        }
    }
    out
}

/// f64-ordered reference of the same quantized arithmetic + Σ|p_scaled| per output.
fn f64_ref(codes: &[u8], scales: &[f32], ex: &ExpertRaw, g: usize) -> (Vec<f64>, Vec<f64>) {
    let gs = match ex.kind {
        ExpertKind::Nvfp4 => 16usize,
        ExpertKind::Mxfp4 => 32,
    };
    let ngroups = ex.kdim / gs;
    let kq = ex.kdim / 128;
    let mut out = vec![0f64; g * ex.n];
    let mut absp = vec![0f64; g * ex.n];
    for row in 0..g {
        let arow = &codes[row * ex.kdim..(row + 1) * ex.kdim];
        let asrow = &scales[row * kq..(row + 1) * kq];
        for col in 0..ex.n {
            let wrow = &ex.w[col * ex.kdim / 2..(col + 1) * ex.kdim / 2];
            let srow = &ex.sc[col * ngroups..(col + 1) * ngroups];
            let mut acc = 0f64;
            let mut aabs = 0f64;
            for j in 0..ngroups {
                let ws = match ex.kind {
                    ExpertKind::Nvfp4 => fp8_e4m3_to_f32(srow[j]) * ex.scale2,
                    ExpertKind::Mxfp4 => e8m0_to_f32(srow[j]),
                } as f64;
                let sc = ws * asrow[j * gs / 128] as f64;
                for i in 0..gs {
                    let kk = j * gs + i;
                    let byte = wrow[kk >> 1];
                    let code = if kk & 1 == 1 { byte >> 4 } else { byte & 0x0F };
                    let p = fp8_e4m3_to_f32(arow[kk]) as f64 * E2M1[code as usize] as f64 * sc;
                    acc += p;
                    aabs += p.abs();
                }
            }
            out[row * ex.n + col] = acc;
            absp[row * ex.n + col] = aabs;
        }
    }
    (out, absp)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: dsv4-native-gemm-gate <model-dir> <fixtures.json> [dev0,dev1]");
        std::process::exit(2);
    }
    assert!(
        memra_gguf::dsv4_forward::expert_arm_native(),
        "this gate REQUIRES MEMRA_DSV4_EXPERT_ARM=native (the class seam)"
    );
    let t0 = std::time::Instant::now();
    let dir = Path::new(&args[1]);
    let spec = FixtureSpec::load(Path::new(&args[2]));
    let devices: Vec<usize> = args
        .get(3)
        .map(|s| s.split(',').map(|x| x.parse().expect("device")).collect())
        .unwrap_or_else(|| vec![0, 1]);
    println!(
        "dsv4-native-gemm-gate | model {} | fixtures {} | variant {} | devices {devices:?}",
        dir.display(),
        args[2],
        spec.variant_tag
    );

    let gpu = Dsv4Gpu::load(dir, &devices, spec.variant, 512).expect("load");
    assert_eq!(gpu.expert_arm, ExpertArm::Native, "arm seam mismatch");
    let n_trunk = gpu.model.mc.n_layer - gpu.model.mc.nextn_predict_layers;

    // ---- capture REAL moe-input rows from the fixture-prompt forward
    let want_trunk: std::collections::BTreeSet<u32> = [0u32, 2, 20, 22, 42].into();
    let mut cap = GpuCapture {
        want: want_trunk.clone(),
        ..Default::default()
    };
    let ids = &spec.tokens_32;
    let fwd = gpu
        .forward(ids, Some(&mut cap), None)
        .expect("forward")
        .expect("logits");
    let mut cap_mtp = GpuCapture {
        want: [n_trunk].into(),
        ..Default::default()
    };
    let _ = gpu
        .mtp_logits_last_cap(&fwd.h_last, ids, Some(&mut cap_mtp))
        .expect("mtp forward");
    cap.moe_x.append(&mut cap_mtp.moe_x);
    println!(
        "captured moe_x at layers {:?} (t={:.0}s)",
        cap.moe_x.keys().collect::<Vec<_>>(),
        t0.elapsed().as_secs_f64()
    );

    let hidden = gpu.model.mc.n_embd as usize;
    let inter = gpu.model.mc.moe.as_ref().expect("moe").expert_ff_length as usize;
    let limit = gpu.model.cfg().swiglu_limit;
    let m_rows = 8usize.min(ids.len());

    let samples: Vec<(u32, &str, usize)> = vec![
        (0, "layers.0", 0),
        (2, "layers.2", 100),
        (20, "layers.20", 7), // lane-1 NVFP4 oracle pin tensor
        (gpu.split_at, "SPLIT", 31),
        (n_trunk - 1, "LAST", 255),
        (n_trunk, "mtp.0", 7), // lane-1 MXFP4 oracle pin tensor
        (n_trunk, "mtp.0", 200),
    ];

    let mut failures = 0usize;
    println!(
        "\n| tensor | m x n x k | mirror | f64 max-err / bound | rung shift meas (pred med ratio) |"
    );
    println!("|---|---|---|---|---|");
    for &(lid, pfx, exi) in &samples {
        let prefix = match pfx {
            "SPLIT" => format!("layers.{}", gpu.split_at),
            "LAST" => format!("layers.{}", n_trunk - 1),
            other => other.to_string(),
        };
        let x_full = cap
            .moe_x
            .get(&lid)
            .unwrap_or_else(|| panic!("moe_x missing for layer {lid}"));
        let x: Vec<f32> = x_full[..m_rows * hidden].to_vec();

        // locate the GPU-resident layer + stage
        let (stage, layer) = if prefix == "mtp.0" {
            let m = gpu.mtp.as_ref().expect("mtp");
            (&gpu.stages[gpu.stages.len() - 1], &m.layer)
        } else {
            let il: u32 = prefix.strip_prefix("layers.").unwrap().parse().unwrap();
            let st = &gpu.stages[gpu.layer_stage[il as usize]];
            (st, st.layers.iter().find(|l| l.il == il).expect("layer"))
        };
        stage.gpu.ctx.bind_to_thread().expect("bind ctx");
        let stream = stage.gpu.stream();
        let kind_i = match layer.expert_kind {
            ExpertKind::Nvfp4 => 0i32,
            ExpertKind::Mxfp4 => 1,
        };

        // raw slabs from the artifact (host side, per projection)
        let raw = |proj: &str| -> (Vec<u8>, Vec<u8>, f32, usize, usize) {
            let name = format!("{prefix}.ffn.experts.{exi}.{proj}");
            let (wi, wb) = gpu
                .model
                .st
                .raw(&format!("{name}.weight"))
                .unwrap_or_else(|| panic!("{name}.weight"));
            let n = wi.shape[0] as usize;
            let kdim = wi.shape[1] as usize * 2;
            match layer.expert_kind {
                ExpertKind::Nvfp4 => {
                    let (_, sb) = gpu.model.st.raw(&format!("{name}.weight_scale")).unwrap();
                    let (_, s2b) = gpu.model.st.raw(&format!("{name}.weight_scale_2")).unwrap();
                    let s2 = f32::from_le_bytes(s2b.try_into().unwrap());
                    (wb.to_vec(), sb.to_vec(), s2, n, kdim)
                }
                ExpertKind::Mxfp4 => {
                    let (_, sb) = gpu.model.st.raw(&format!("{name}.scale")).unwrap();
                    (wb.to_vec(), sb.to_vec(), 0.0, n, kdim)
                }
            }
        };

        // CPU quantized chain to derive the REAL w2 input h from the captured x
        // (Expert.forward order: silu(clamp) * clamp(up), weight ~ 1.0 here — the gate
        // exercises the GEMM arithmetic; routing weights are a row scalar upstream of
        // the quantizer and are covered by the (b)/(c)/(d) full-forward gates)
        let (w1w, w1s, w1s2, _, _) = raw("w1");
        let (w3w, w3s, w3s2, _, _) = raw("w3");
        let (xcodes, xscales) = quant_rows(&x, m_rows, hidden);
        let kind = layer.expert_kind;
        let ex1 = ExpertRaw {
            kind,
            w: w1w,
            sc: w1s,
            scale2: w1s2,
            n: inter,
            kdim: hidden,
        };
        let ex3 = ExpertRaw {
            kind,
            w: w3w,
            sc: w3s,
            scale2: w3s2,
            n: inter,
            kdim: hidden,
        };
        let (g1_64, _) = f64_ref(&xcodes, &xscales, &ex1, m_rows);
        let (g3_64, _) = f64_ref(&xcodes, &xscales, &ex3, m_rows);
        let mut h = vec![0f32; m_rows * inter];
        for i in 0..m_rows * inter {
            let u = (g3_64[i] as f32).clamp(-limit, limit);
            let gt = (g1_64[i] as f32).min(limit);
            h[i] = gt * (1.0 / (1.0 + (-gt).exp())) * u;
        }

        for (proj, pi, inp, m, n, kdim) in [
            ("w1", 0usize, &x, m_rows, inter, hidden),
            ("w3", 2, &x, m_rows, inter, hidden),
            ("w2", 1, &h, m_rows, hidden, inter),
        ] {
            let (ww, wsc, s2, nn, kk) = raw(proj);
            assert_eq!((nn, kk), (n, kdim), "{prefix} {proj} shape");
            let exr = ExpertRaw {
                kind,
                w: ww,
                sc: wsc,
                scale2: s2,
                n,
                kdim,
            };
            let (codes, scales) = quant_rows(inp, m, kdim);

            // ---- GPU native kernel on the SAME inputs (quantized on-device from inp)
            let sv = stream.cu_stream() as *mut c_void;
            let mut inp_d = stream.alloc_zeros::<f32>(m * kdim).expect("inp_d");
            stream.memcpy_htod(&inp[..], &mut inp_d).expect("htod inp");
            let mut cod_d = stream.alloc_zeros::<u8>(m * kdim).expect("cod");
            let mut scl_d = stream.alloc_zeros::<f32>(m * (kdim / 128)).expect("scl");
            let mut out_d = stream.alloc_zeros::<f32>(m * n).expect("out");
            let wbytes = inter * hidden / 2;
            let sbytes = match layer.expert_kind {
                ExpertKind::Nvfp4 => inter * hidden / 16,
                ExpertKind::Mxfp4 => inter * hidden / 32,
            };
            let wp = (layer.experts_w.device_ptr(&stream).0 as usize + (exi * 3 + pi) * wbytes)
                as *const c_void;
            let scp = (layer.experts_sc.device_ptr(&stream).0 as usize + (exi * 3 + pi) * sbytes)
                as *const c_void;
            unsafe {
                let rc = k::memra_dsv4_act_quant_fp8(
                    inp_d.device_ptr(&stream).0 as *const f32,
                    cod_d.device_ptr_mut(&stream).0 as *mut c_void,
                    scl_d.device_ptr_mut(&stream).0 as *mut f32,
                    m as i32,
                    kdim as i32,
                    sv,
                );
                assert_eq!(rc, 0, "act_quant_fp8 rc {rc}");
                let rc = k::memra_dsv4_fp4_gemm(
                    cod_d.device_ptr(&stream).0 as *const c_void,
                    scl_d.device_ptr(&stream).0 as *const f32,
                    wp,
                    scp,
                    if kind_i == 0 {
                        layer.experts_s2[exi * 3 + pi]
                    } else {
                        0.0
                    },
                    kind_i,
                    out_d.device_ptr_mut(&stream).0 as *mut f32,
                    m as i32,
                    n as i32,
                    kdim as i32,
                    sv,
                );
                assert_eq!(rc, 0, "fp4_gemm rc {rc}");
            }
            // codes/scales must match the CPU mirror BIT-EXACTLY too (same grid ops)
            let mut cod_h = vec![0u8; m * kdim];
            stream.memcpy_dtoh(&cod_d, &mut cod_h[..]).expect("dtoh c");
            let mut scl_h = vec![0f32; m * (kdim / 128)];
            stream.memcpy_dtoh(&scl_d, &mut scl_h[..]).expect("dtoh s");
            let mut got = vec![0f32; m * n];
            stream.memcpy_dtoh(&out_d, &mut got[..]).expect("dtoh o");
            stream.synchronize().expect("sync");
            let n_code_diff = cod_h.iter().zip(&codes).filter(|(a, b)| a != b).count();
            let n_scale_diff = scl_h
                .iter()
                .zip(&scales)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            let codes_ok = n_code_diff == 0 && n_scale_diff == 0;
            if !codes_ok {
                // name the first mismatch precisely (encoder-parity instrument)
                if let Some(i) = cod_h.iter().zip(&codes).position(|(a, b)| a != b) {
                    let (r, c2) = (i / kdim, i % kdim);
                    let scale = scales[r * (kdim / 128) + c2 / 128];
                    println!(
                        "  [quant-diff] {} code diffs, {} scale diffs; first at [{r},{c2}]: input {:.6e} (bits {:08x}), scale {:.3e}, gpu code {:#04x}, cpu code {:#04x}",
                        n_code_diff,
                        n_scale_diff,
                        inp[i],
                        inp[i].to_bits(),
                        scale,
                        cod_h[i],
                        codes[i]
                    );
                }
                if let Some(g2) = scl_h
                    .iter()
                    .zip(&scales)
                    .position(|(a, b)| a.to_bits() != b.to_bits())
                {
                    println!(
                        "  [scale-diff] first at group {g2}: gpu {:.6e} vs cpu {:.6e}",
                        scl_h[g2], scales[g2]
                    );
                }
            }

            // ---- 1. bit-exact mirror
            let mir = mirror_gemm(&codes, &scales, &exr, m);
            let gemm_bits_ok = got
                .iter()
                .zip(&mir)
                .all(|(a, b)| a.to_bits() == b.to_bits());
            if !gemm_bits_ok && codes_ok {
                if let Some(i) = got
                    .iter()
                    .zip(&mir)
                    .position(|(a, b)| a.to_bits() != b.to_bits())
                {
                    println!(
                        "  [gemm-diff] first at flat {i}: gpu {:.6e} vs mirror {:.6e}",
                        got[i], mir[i]
                    );
                }
            }
            let bit_ok = codes_ok && gemm_bits_ok;

            // ---- 2. f64-ordered reference bound
            let (r64, absp) = f64_ref(&codes, &scales, &exr, m);
            let gs = if kind_i == 0 { 16f64 } else { 32.0 };
            let ngroups = kdim as f64 / gs;
            let n_ops = (gs - 1.0) + (ngroups / THREADS as f64).ceil() + 7.0;
            let u32r = (2f64).powi(-24);
            let mut worst_err = 0f64;
            let mut worst_bound = f64::MAX;
            let mut f64_ok = true;
            for i in 0..m * n {
                let err = (got[i] as f64 - r64[i]).abs();
                let bound = n_ops * u32r * absp[i];
                if err > bound {
                    f64_ok = false;
                }
                if err > worst_err {
                    worst_err = err;
                    worst_bound = bound;
                }
            }

            // ---- 3. class shift vs the bf16-dequant rung on the SAME inputs
            let mut xb = stream.alloc_zeros::<u8>(m * kdim * 2).expect("xb");
            let mut rung = stream.alloc_zeros::<f32>(m * n).expect("rung");
            unsafe {
                let rc = k::memra_dsv4_cvt_bf16(
                    inp_d.device_ptr(&stream).0 as *const f32,
                    xb.device_ptr_mut(&stream).0 as *mut c_void,
                    (m * kdim) as i64,
                    sv,
                );
                assert_eq!(rc, 0);
                let dst = stage.deq[pi].device_ptr(&stream).0 as *mut c_void;
                let rc = match layer.expert_kind {
                    ExpertKind::Nvfp4 => k::memra_dsv4_nvfp4_deq_bf16(
                        wp,
                        scp,
                        layer.experts_s2[exi * 3 + pi],
                        n as i32,
                        kdim as i32,
                        dst,
                        sv,
                    ),
                    ExpertKind::Mxfp4 => {
                        k::memra_dsv4_mxfp4_deq_bf16(wp, scp, n as i32, kdim as i32, dst, sv)
                    }
                };
                assert_eq!(rc, 0);
                let rc = k::memra_dsv4_gemm_bf16(
                    stage.deq[pi].device_ptr(&stream).0 as *const c_void,
                    xb.device_ptr(&stream).0 as *const c_void,
                    rung.device_ptr_mut(&stream).0 as *mut f32,
                    m as i32,
                    n as i32,
                    kdim as i32,
                    stage.dev as i32,
                    stage.ws.device_ptr(&stream).0 as *mut c_void,
                    stage.ws.len(),
                    sv,
                );
                assert_eq!(rc, 0);
            }
            let mut rung_h = vec![0f32; m * n];
            stream.memcpy_dtoh(&rung, &mut rung_h[..]).expect("dtoh r");
            stream.synchronize().expect("sync");
            // prediction per output: (u_q/sqrt(3)) * ||x .* w||_2, on the DEQUANT weights
            let (_, wf) = gpu
                .model
                .tensor_f32(&format!("{prefix}.ffn.experts.{exi}.{proj}"));
            let mut ratios: Vec<f64> = Vec::with_capacity(m * n);
            let mut shift_max = 0f64;
            for row in 0..m {
                for col in 0..n {
                    let mut l2 = 0f64;
                    for kk2 in 0..kdim {
                        let p = inp[row * kdim + kk2] as f64 * wf[col * kdim + kk2] as f64;
                        l2 += p * p;
                    }
                    let pred = U_FP8 / 3f64.sqrt() * l2.sqrt();
                    let meas = (got[row * n + col] as f64 - rung_h[row * n + col] as f64).abs();
                    shift_max = shift_max.max(meas);
                    if pred > 0.0 {
                        ratios.push(meas / pred);
                    }
                }
            }
            ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med = ratios.get(ratios.len() / 2).cloned().unwrap_or(0.0);

            let pass = bit_ok && f64_ok;
            if !pass {
                failures += 1;
            }
            println!(
                "| {prefix}.experts.{exi}.{proj} | {m}x{n}x{kdim} | {} | {:.2e} / {:.2e} {} | {:.3e} (med ratio {:.2}) |",
                if bit_ok {
                    "BIT-EXACT"
                } else if codes_ok {
                    "FAIL(gemm)"
                } else if gemm_bits_ok {
                    "FAIL(quant,gemm-bits-EQUAL)"
                } else {
                    "FAIL(quant+gemm)"
                },
                worst_err,
                worst_bound,
                if f64_ok { "PASS" } else { "FAIL" },
                shift_max,
                med
            );
        }
    }
    // ---- swiglu_limit SATURATION cell (recon intel: vLLM's sm12x cutedsl fork LOST
    // this clamp — the cell proves ours engages between the quantized GEMMs, both
    // recipes). A REAL captured x row scaled ×50 (declared) drives w1/w3 outputs past
    // ±swiglu_limit; the clamp per model.py:600-602 (up two-sided, gate one-sided,
    // BEFORE silu) must produce h == CPU clamp math within expf-ULP class, and the
    // whole chain (quant→gemm→clamp→quant→gemm) must stay mirror-bit-exact on the
    // GEMM legs.
    println!("\n== swiglu_limit saturation cell (clamp fused with the quantized path) ==");
    for &(lid, pfx, exi) in &[(20u32, "layers.20", 7usize), (n_trunk, "mtp.0", 7)] {
        let prefix = pfx.to_string();
        let (stage, layer) = if prefix == "mtp.0" {
            let m = gpu.mtp.as_ref().expect("mtp");
            (&gpu.stages[gpu.stages.len() - 1], &m.layer)
        } else {
            let il: u32 = prefix.strip_prefix("layers.").unwrap().parse().unwrap();
            let st = &gpu.stages[gpu.layer_stage[il as usize]];
            (st, st.layers.iter().find(|l| l.il == il).expect("layer"))
        };
        stage.gpu.ctx.bind_to_thread().expect("bind ctx");
        let stream = stage.gpu.stream();
        let sv = stream.cu_stream() as *mut c_void;
        let kind_i = match layer.expert_kind {
            ExpertKind::Nvfp4 => 0i32,
            ExpertKind::Mxfp4 => 1,
        };
        let kind = layer.expert_kind;
        let x_full = cap.moe_x.get(&lid).expect("moe_x");
        let xs50: Vec<f32> = x_full[..hidden].iter().map(|v| v * 50.0).collect();
        let raw = |proj: &str| -> (Vec<u8>, Vec<u8>, f32, usize, usize) {
            let name = format!("{prefix}.ffn.experts.{exi}.{proj}");
            let (wi, wb) = gpu.model.st.raw(&format!("{name}.weight")).unwrap();
            let n = wi.shape[0] as usize;
            let kdim = wi.shape[1] as usize * 2;
            match kind {
                ExpertKind::Nvfp4 => {
                    let (_, sb) = gpu.model.st.raw(&format!("{name}.weight_scale")).unwrap();
                    let (_, s2b) = gpu.model.st.raw(&format!("{name}.weight_scale_2")).unwrap();
                    (
                        wb.to_vec(),
                        sb.to_vec(),
                        f32::from_le_bytes(s2b.try_into().unwrap()),
                        n,
                        kdim,
                    )
                }
                ExpertKind::Mxfp4 => {
                    let (_, sb) = gpu.model.st.raw(&format!("{name}.scale")).unwrap();
                    (wb.to_vec(), sb.to_vec(), 0.0, n, kdim)
                }
            }
        };
        let wbytes = inter * hidden / 2;
        let sbytes = match kind {
            ExpertKind::Nvfp4 => inter * hidden / 16,
            ExpertKind::Mxfp4 => inter * hidden / 32,
        };
        let gemm_gpu = |inp_dev: &cudarc::driver::CudaSlice<f32>,
                        m: usize,
                        n: usize,
                        kdim: usize,
                        pi: usize|
         -> (Vec<u8>, Vec<f32>, Vec<f32>) {
            let mut cod = stream.alloc_zeros::<u8>(m * kdim).expect("cod");
            let mut scl = stream.alloc_zeros::<f32>(m * (kdim / 128)).expect("scl");
            let mut out = stream.alloc_zeros::<f32>(m * n).expect("out");
            let wp = (layer.experts_w.device_ptr(&stream).0 as usize + (exi * 3 + pi) * wbytes)
                as *const c_void;
            let scp = (layer.experts_sc.device_ptr(&stream).0 as usize + (exi * 3 + pi) * sbytes)
                as *const c_void;
            unsafe {
                assert_eq!(
                    k::memra_dsv4_act_quant_fp8(
                        inp_dev.device_ptr(&stream).0 as *const f32,
                        cod.device_ptr_mut(&stream).0 as *mut c_void,
                        scl.device_ptr_mut(&stream).0 as *mut f32,
                        m as i32,
                        kdim as i32,
                        sv,
                    ),
                    0
                );
                assert_eq!(
                    k::memra_dsv4_fp4_gemm(
                        cod.device_ptr(&stream).0 as *const c_void,
                        scl.device_ptr(&stream).0 as *const f32,
                        wp,
                        scp,
                        if kind_i == 0 {
                            layer.experts_s2[exi * 3 + pi]
                        } else {
                            0.0
                        },
                        kind_i,
                        out.device_ptr_mut(&stream).0 as *mut f32,
                        m as i32,
                        n as i32,
                        kdim as i32,
                        sv,
                    ),
                    0
                );
            }
            let mut cod_h = vec![0u8; m * kdim];
            stream.memcpy_dtoh(&cod, &mut cod_h[..]).expect("dtoh");
            let mut scl_h = vec![0f32; m * (kdim / 128)];
            stream.memcpy_dtoh(&scl, &mut scl_h[..]).expect("dtoh");
            let mut out_h = vec![0f32; m * n];
            stream.memcpy_dtoh(&out, &mut out_h[..]).expect("dtoh");
            stream.synchronize().expect("sync");
            (cod_h, scl_h, out_h)
        };
        // GPU: w1, w3, swiglu, w2 — the exact moe_forward native chain at g=1
        let mut x_dev = stream.alloc_zeros::<f32>(hidden).expect("x_dev");
        stream.memcpy_htod(&xs50[..], &mut x_dev).expect("htod");
        let (_, _, g1) = gemm_gpu(&x_dev, 1, inter, hidden, 0);
        let (_, _, g3) = gemm_gpu(&x_dev, 1, inter, hidden, 2);
        let mut g1_dev = stream.alloc_zeros::<f32>(inter).expect("g1d");
        stream.memcpy_htod(&g1[..], &mut g1_dev).expect("htod g1");
        let mut g3_dev = stream.alloc_zeros::<f32>(inter).expect("g3d");
        stream.memcpy_htod(&g3[..], &mut g3_dev).expect("htod g3");
        let mut h_dev = stream.alloc_zeros::<f32>(inter).expect("hd");
        unsafe {
            assert_eq!(
                k::memra_dsv4_swiglu(
                    g1_dev.device_ptr(&stream).0 as *const f32,
                    g3_dev.device_ptr(&stream).0 as *const f32,
                    h_dev.device_ptr_mut(&stream).0 as *mut f32,
                    1,
                    inter as i32,
                    limit,
                    std::ptr::null(),
                    sv,
                ),
                0
            );
        }
        let mut h_gpu = vec![0f32; inter];
        stream.memcpy_dtoh(&h_dev, &mut h_gpu[..]).expect("dtoh h");
        stream.synchronize().expect("sync");
        let (_, _, w2_gpu) = gemm_gpu(&h_dev, 1, hidden, inter, 1);

        // CPU: mirrors on the same inputs
        let (w1w, w1s, w1s2, _, _) = raw("w1");
        let (w3w, w3s, w3s2, _, _) = raw("w3");
        let (w2w, w2s, w2s2, _, _) = raw("w2");
        let ex1 = ExpertRaw {
            kind,
            w: w1w,
            sc: w1s,
            scale2: w1s2,
            n: inter,
            kdim: hidden,
        };
        let ex3 = ExpertRaw {
            kind,
            w: w3w,
            sc: w3s,
            scale2: w3s2,
            n: inter,
            kdim: hidden,
        };
        let ex2 = ExpertRaw {
            kind,
            w: w2w,
            sc: w2s,
            scale2: w2s2,
            n: hidden,
            kdim: inter,
        };
        let (xc, xs) = quant_rows(&xs50, 1, hidden);
        let g1_cpu = mirror_gemm(&xc, &xs, &ex1, 1);
        let g3_cpu = mirror_gemm(&xc, &xs, &ex3, 1);
        let w13_bits_ok = g1
            .iter()
            .zip(&g1_cpu)
            .chain(g3.iter().zip(&g3_cpu))
            .all(|(a, b)| a.to_bits() == b.to_bits());
        // clamp math (model.py:600-602): up two-sided, gate one-sided, BEFORE silu
        let mut n_sat_g = 0usize;
        let mut n_sat_u = 0usize;
        let mut h_cpu = vec![0f32; inter];
        for i in 0..inter {
            if g1_cpu[i] > limit {
                n_sat_g += 1;
            }
            if g3_cpu[i].abs() > limit {
                n_sat_u += 1;
            }
            let u = g3_cpu[i].clamp(-limit, limit);
            let g = g1_cpu[i].min(limit);
            h_cpu[i] = g * (1.0 / (1.0 + (-g).exp())) * u;
        }
        let habs = h_cpu.iter().fold(0f32, |a, &v| a.max(v.abs()));
        let mut h_diff = 0f64;
        for (a, b) in h_gpu.iter().zip(&h_cpu) {
            h_diff = h_diff.max((*a as f64 - *b as f64).abs());
        }
        let h_thr = 2e-6 * habs as f64; // expf-vs-exp ULP class (the only differing op)
        // w2 leg: quantize the GPU's h (same bytes both sides) and mirror
        let (hc, hsc) = quant_rows(&h_gpu, 1, inter);
        let w2_cpu = mirror_gemm(&hc, &hsc, &ex2, 1);
        let w2_bits_ok = w2_gpu
            .iter()
            .zip(&w2_cpu)
            .all(|(a, b)| a.to_bits() == b.to_bits());
        let cell_ok = w13_bits_ok && w2_bits_ok && n_sat_g > 0 && n_sat_u > 0 && h_diff <= h_thr;
        if !cell_ok {
            failures += 1;
        }
        println!(
            "  [{}] {prefix}.experts.{exi} x50-row: gate-clamp sat {n_sat_g}/{inter}, up-clamp sat {n_sat_u}/{inter} (both must be >0); w1/w3 {} | h max-diff {h_diff:.3e} vs thr {h_thr:.3e} | w2 {}",
            if cell_ok { "PASS" } else { "FAIL" },
            if w13_bits_ok { "BIT-EXACT" } else { "FAIL" },
            if w2_bits_ok { "BIT-EXACT" } else { "FAIL" },
        );
    }

    println!(
        "\nDSV4 NATIVE GEMM GATE: {} ({} projections x {} samples + 2 saturation cells, {} failures, {:.0}s)",
        if failures == 0 { "PASS" } else { "FAIL" },
        3,
        samples.len(),
        failures,
        t0.elapsed().as_secs_f64()
    );
    std::process::exit(if failures == 0 { 0 } else { 1 });
}
