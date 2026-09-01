//! dsv4-decode-probe: lane-6 DIAGNOSTIC (not a gate). Localizes decode-vs-reprefill
//! drift: prefill 32 with cache population, ONE decode step (probe dump of per-layer
//! intermediates), then a lane-4 re-prefill of the same 33 tokens with fixture-style
//! capture, and per-layer max-abs compares of the last position. Also compares the
//! prefill-populated cache blocks against the re-prefill's freshly computed compressed
//! kv (cache-content instrument).
//!
//! Usage: dsv4-decode-probe <model-dir> <fixtures.json> [dev0,dev1]

use cudarc::driver::{DevicePtr, DevicePtrMut};
use memra_engine::dsv4_ffi as k;
use memra_engine::dsv4_gpu::{Dsv4Gpu, GpuCapture};
use memra_gguf::dsv4_forward::FixtureSpec;
use std::os::raw::c_void;
use std::path::Path;

fn max_abs(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (x as f64 - y as f64).abs())
        .fold(0.0, f64::max)
}

fn max_rel(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (x as f64 - y as f64).abs() / (y.abs() as f64).max(1e-6))
        .fold(0.0, f64::max)
}

/// Pure cuBLASLt m-dependence instrument: the SAME x row replicated m times through
/// the SAME bf16 weight — per-row results must agree across m if the plan is
/// row-stable; any difference is the measured per-hop reorder bound.
fn gemm_m_ab(gpu: &Dsv4Gpu, x_row: &[f32], n: usize, kdim: usize, w_name: &str) {
    let st = &gpu.stages[0];
    st.gpu.ctx.bind_to_thread().expect("bind");
    let stream = st.gpu.stream();
    let layer = &st.layers[0];
    let w = match w_name {
        "wq_a" => layer.wq_a.dev(),
        "wkv" => layer.wkv.dev(),
        other => panic!("unknown w {other}"),
    };
    let run = |m: usize| -> Vec<f32> {
        let mut xh: Vec<f32> = Vec::with_capacity(m * kdim);
        for _ in 0..m {
            xh.extend_from_slice(&x_row[..kdim]);
        }
        let mut xd = stream.alloc_zeros::<f32>(m * kdim).expect("xd");
        stream.memcpy_htod(&xh, &mut xd).expect("htod");
        let mut xb = stream.alloc_zeros::<u8>(m * kdim * 2).expect("xb");
        let mut y = stream.alloc_zeros::<f32>(m * n).expect("y");
        unsafe {
            k::ck(
                "cvt",
                k::memra_dsv4_cvt_bf16(
                    xd.device_ptr(&stream).0 as *const f32,
                    xb.device_ptr_mut(&stream).0 as *mut c_void,
                    (m * kdim) as i64,
                    stream.cu_stream() as *mut c_void,
                ),
            )
            .unwrap();
            k::ck(
                "gemm",
                k::memra_dsv4_gemm_bf16(
                    w.device_ptr(&stream).0 as *const c_void,
                    xb.device_ptr(&stream).0 as *const c_void,
                    y.device_ptr_mut(&stream).0 as *mut f32,
                    m as i32,
                    n as i32,
                    kdim as i32,
                    st.dev as i32,
                    st.ws.device_ptr(&stream).0 as *mut c_void,
                    st.ws.len(),
                    stream.cu_stream() as *mut c_void,
                ),
            )
            .unwrap();
        }
        let mut out = vec![0f32; m * n];
        stream.memcpy_dtoh(&y, &mut out[..]).expect("dtoh");
        stream.synchronize().expect("sync");
        out
    };
    let y1 = run(1);
    let y32 = run(32);
    let y33 = run(33);
    // row 0 of each vs m=1
    println!(
        "GEMM m-A/B {w_name} [n={n},k={kdim}]: m=1 vs m=32 row0 max-abs {:.3e} (rel {:.3e}); m=1 vs m=33 row0 max-abs {:.3e}; m=32 row0 vs m=33 row0 max-abs {:.3e}; m=33 row0 vs row32 max-abs {:.3e}",
        max_abs(&y1, &y32[0..n]),
        max_rel(&y1, &y32[0..n]),
        max_abs(&y1, &y33[0..n]),
        max_abs(&y32[0..n], &y33[0..n]),
        max_abs(&y33[0..n], &y33[32 * n..33 * n]),
    );
}

fn main() {
    // item 3 boot refusal (hermes a4e3d9a8eab4cf17 shape): this probe's instrument
    // reads the RESIDENT bf16 slabs, which are host-staged under the fp8 dense arm —
    // one-line Err at boot, never a post-load abort. Keyed on the RESOLVED arm, not the
    // literal env: the 2026-08-20 ratification made fp8 the DEVICE-decode default, so an
    // env-keyed guard silently admits the exact configuration it exists to refuse (the
    // v0.98 box8 flip cells caught this — the probe booted under device+unset).
    let probe_on_device = matches!(
        std::env::var("MEMRA_DSV4_DECODE_PATH").ok().as_deref(),
        Some("device") | Some("device-hostmath")
    );
    if memra_engine::dsv4_gpu::resolve_dense_arm(
        std::env::var("MEMRA_DSV4_DENSE_ARM").ok().as_deref(),
        probe_on_device,
    ) == Ok(true)
    {
        eprintln!(
            "dsv4-decode-probe: the fp8 dense arm is unsupported here (it RESOLVES fp8 — \
             default on the device decode path since 2026-08-20, or explicit env) — the \
             m-dependence instrument reads the resident bf16 slabs (host-staged under the \
             fp8 arm); run with MEMRA_DSV4_DENSE_ARM=bf16"
        );
        std::process::exit(101);
    }
    let args: Vec<String> = std::env::args().collect();
    let dir = Path::new(&args[1]);
    let spec = FixtureSpec::load(Path::new(&args[2]));
    let devices: Vec<usize> = args
        .get(3)
        .map(|s| s.split(',').map(|x| x.parse().expect("device")).collect())
        .unwrap_or_else(|| vec![0, 1]);
    let prompt = spec.tokens_32.clone();
    let gpu = Dsv4Gpu::load(dir, &devices, spec.variant, 256).expect("load");
    println!("loaded, split at {}", gpu.split_at);

    let mut state = gpu.alloc_decode_state().expect("alloc");
    let pre = gpu
        .prefill_with_cache(&prompt, &mut state)
        .expect("prefill");
    let t32 = {
        let mut best = 0usize;
        for i in 1..pre.logits.len() {
            if pre.logits[i] > pre.logits[best] {
                best = i;
            }
        }
        best as u32
    };
    println!("prefill done, t32 = {t32}");

    let (dec_logits, dump) = gpu.decode_step_probe(t32, &mut state).expect("decode");

    // re-prefill 33 tokens with capture at the probed layers
    let mut ids = prompt.clone();
    ids.push(t32);
    let mut cap = GpuCapture::default();
    for l in [0u32, 1, 2, 3, 4, 21, 22, 42] {
        cap.want.insert(l);
    }
    let ref_out = gpu
        .forward(&ids, Some(&mut cap), None)
        .expect("re-prefill")
        .expect("logits");
    let s = ids.len();
    let d = gpu.model.cfg().clone();
    let mc = gpu.model.mc.clone();
    let hidden = mc.n_embd as usize;
    let hc = d.hc_mult as usize;
    let hd = d.head_dim as usize;
    let heads = mc.n_head as usize;
    let win = d.sliding_window as usize;

    println!(
        "logits max-abs decode-vs-reprefill: {:.3e}",
        max_abs(&dec_logits, &ref_out.logits)
    );

    let dget =
        |name: &str| -> Option<&Vec<f32>> { dump.iter().find(|(n, _)| n == name).map(|(_, v)| v) };
    let cmp_last =
        |m: &std::collections::BTreeMap<u32, Vec<f32>>, l: u32, width: usize, dn: &str| -> String {
            m.get(&l)
                .and_then(|v| {
                    let row = &v[(s - 1) * width..s * width];
                    dget(dn).map(|dv| format!("{:.3e}", max_abs(dv, row)))
                })
                .unwrap_or("-".into())
        };
    println!("| layer | x | q | kv | o | attn_out | h3 |");
    for l in [0u32, 1, 2, 3, 4, 21, 22, 42] {
        println!(
            "| {l} | {} | {} | {} | {} | {} | {} |",
            cmp_last(&cap.x_dbg, l, hidden, &format!("layer{l}.x")),
            cmp_last(&cap.q_dbg, l, heads * hd, &format!("layer{l}.q")),
            cmp_last(&cap.kv_dbg, l, hd, &format!("layer{l}.kv")),
            cmp_last(&cap.o_dbg, l, heads * hd, &format!("layer{l}.o")),
            cmp_last(&cap.attn_out, l, hidden, &format!("layer{l}.attn_out")),
            cmp_last(&cap.layer_out, l, hc * hidden, &format!("layer{l}.h3")),
        );
    }

    // cache-content instrument: prefill-populated blocks vs re-prefill's fresh pooled kv
    for l in [2u32, 3, 4] {
        if let Some((ckv, nb)) = cap.compressor_kv.get(&l) {
            let stage = gpu.layer_stage[l as usize];
            let st = &gpu.stages[stage];
            st.gpu.ctx.bind_to_thread().expect("bind");
            let stream = st.gpu.stream();
            let cache = &state.caches[l as usize];
            let n = (*nb).min(cache.n_blocks);
            if n > 0 {
                let mut got = vec![0f32; n * hd];
                stream
                    .memcpy_dtoh(&cache.kvc.slice(win * hd..(win + n) * hd), &mut got[..])
                    .expect("dtoh kvc");
                stream.synchronize().expect("sync");
                println!(
                    "layer {l} cache blocks vs re-prefill pooled ({} vs {} blocks): max-abs {:.3e}",
                    cache.n_blocks,
                    nb,
                    max_abs(&got, &ckv[..n * hd])
                );
            } else {
                println!(
                    "layer {l}: no comparable blocks (cache {} / ref {nb})",
                    cache.n_blocks
                );
            }
        }
        if let Some((ikv, nb)) = cap.indexer_kv.get(&l) {
            let stage = gpu.layer_stage[l as usize];
            let st = &gpu.stages[stage];
            let stream = st.gpu.stream();
            let cache = &state.caches[l as usize];
            let ihd = d.index_head_dim as usize;
            let n = (*nb).min(cache.i_blocks);
            if n > 0 {
                let mut got = vec![0f32; n * ihd];
                stream
                    .memcpy_dtoh(
                        &cache.ikvc.as_ref().expect("ikvc").slice(0..n * ihd),
                        &mut got[..],
                    )
                    .expect("dtoh ikvc");
                stream.synchronize().expect("sync");
                println!(
                    "layer {l} indexer cache vs re-prefill ({} vs {} blocks): max-abs {:.3e}",
                    cache.i_blocks,
                    nb,
                    max_abs(&got, &ikv[..n * ihd])
                );
            }
        }
    }
    // pure GEMM m-dependence: same row, replicated
    if let Some(x0) = dget("layer0.x") {
        gemm_m_ab(&gpu, x0, 1024, hidden, "wq_a");
        gemm_m_ab(&gpu, x0, hd, hidden, "wkv");
    }

    // CONTROL: the untouched lane-4 path against itself, prefill@32 vs prefill@33,
    // compared at the SHARED positions (zero lane-6 code involved). If this shows the
    // same drift profile, the phenomenon is the prefill path's own m-sensitivity.
    let mut cap32 = GpuCapture::default();
    for l in [0u32, 1, 2, 3, 4, 21, 22, 42] {
        cap32.want.insert(l);
    }
    let _ = gpu
        .forward(&prompt, Some(&mut cap32), None)
        .expect("re-prefill 32");
    println!("CONTROL prefill@32 vs prefill@33, same positions (pure lane-4 path):");
    println!("| layer | x@31 | q@31 | kv@31 | o@31 | attn_out@31 | h3@31 | h3@16 |");
    let s32 = prompt.len();
    for l in [0u32, 1, 2, 3, 4, 21, 22, 42] {
        let cmp_at = |a: &std::collections::BTreeMap<u32, Vec<f32>>,
                      b: &std::collections::BTreeMap<u32, Vec<f32>>,
                      width: usize,
                      p: usize|
         -> String {
            match (a.get(&l), b.get(&l)) {
                (Some(va), Some(vb)) => {
                    let ra = &va[p * width..(p + 1) * width];
                    let rb = &vb[p * width..(p + 1) * width];
                    format!("{:.3e}", max_abs(ra, rb))
                }
                _ => "-".into(),
            }
        };
        let _ = s32;
        println!(
            "| {l} | {} | {} | {} | {} | {} | {} | {} |",
            cmp_at(&cap32.x_dbg, &cap.x_dbg, hidden, 31),
            cmp_at(&cap32.q_dbg, &cap.q_dbg, heads * hd, 31),
            cmp_at(&cap32.kv_dbg, &cap.kv_dbg, hd, 31),
            cmp_at(&cap32.o_dbg, &cap.o_dbg, heads * hd, 31),
            cmp_at(&cap32.attn_out, &cap.attn_out, hidden, 31),
            cmp_at(&cap32.layer_out, &cap.layer_out, hc * hidden, 31),
            cmp_at(&cap32.layer_out, &cap.layer_out, hc * hidden, 16),
        );
    }
    // LOGITS-LEVEL m-sensitivity floor of the pure lane-4 path: row p under m=p+1
    // (forward(ids[0..p+1]).logits) vs the SAME row under m=p+2
    // (trunk_logits_row(forward(ids[0..p+2]).h_last, row=p)). Decode drift at the same
    // rows is the smoke-gate table; if the floor is the same magnitude, decode sits at
    // the reference's own realization-noise floor.
    println!("LOGITS m-floor (pure lane-4, row p: m=p+1 vs m=p+2):");
    let mut ids_ext = prompt.clone();
    ids_ext.push(t32);
    // extend with the decode run's next tokens so prefixes exist (greedy from dump not
    // available here — reuse re-prefill greedy: argmax chain)
    for _ in 0..18 {
        let o = gpu
            .forward(&ids_ext, None, None)
            .expect("fwd")
            .expect("logits");
        let mut best = 0usize;
        for i in 1..o.logits.len() {
            if o.logits[i] > o.logits[best] {
                best = i;
            }
        }
        ids_ext.push(best as u32);
    }
    let mut prev: Option<(usize, Vec<f32>)> = None; // (m, logits at row m-1)
    for m in 33..=48usize {
        let o = gpu
            .forward(&ids_ext[0..m], None, None)
            .expect("fwd")
            .expect("logits");
        if let Some((pm, pl)) = &prev {
            // same row pm-1 under m: from o.h_last
            let again = gpu
                .trunk_logits_row(&o.h_last, m, pm - 1)
                .expect("logits row");
            println!(
                "  row {} : m={} vs m={} max-abs {:.3e}",
                pm - 1,
                pm,
                m,
                max_abs(pl, &again)
            );
        }
        prev = Some((m, o.logits));
    }
    println!("probe complete");
}
