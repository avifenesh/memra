//! DeepSeek-V4.1 single-card gates: every `cu/dsv41_gpu.cu` kernel against the vendor program on
//! random inputs, and trunk layers on the real checkpoint against per-layer captures of the vendor
//! program (stream in, stream out, CSA2 state, per-op intermediates).
//!
//! Fixtures are directories of raw tensors plus `index.json` written by the lane's torch scripts
//! (`research/modelplan-onboarding-dsv41-20261109/gpu/`): `make_fixtures.py` (random inputs through
//! the vendor modules) and `capture_layers.py` (hooks on the vendor model running the checkpoint).
//!
//! usage:
//!   dsv41-gate kernels FIXTURE_ROOT [--out RECEIPT.json]
//!   dsv41-gate layer CKPT_DIR CAPTURE_DIR --layers 1,2,20 [--mint] [--chain N]
//!                    [--token-map FILE] [--floor CPU_FLOOR_DIR] [--out RECEIPT.json]
//!   dsv41-gate decode CKPT_DIR CAPTURE_DIR --layers 1,2,20 --token-map FILE [--out RECEIPT.json]
//!                    (the decode program's integer outcomes byte for byte, with a re-prefill red arm)
//!   any of them with --dense-form F: every dense FP8/FP4 GEMM in form F (`dsv41_gpu::GEMM_FORM`)

use memra_engine::dsv41_gpu::{
    self as v, Buf, Ctx, Dp, Geo, LayerState, LayerW, Ops, QFmt, QW, Shared, Trace, bf16_to_f32,
    bytes_f32, bytes_i32, ck, ffi,
};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

// ------------------------------------------------------------------------------------------------
// fixtures

struct Fx {
    root: PathBuf,
    /// the checkpoint the capture was made from (set by the layer gate; the Engram constants
    /// derive from its config)
    ckpt: PathBuf,
    t: BTreeMap<String, (String, Vec<usize>, String)>,
    meta: serde_json::Value,
}

impl Fx {
    fn open(root: &Path) -> Result<Self, String> {
        let s = std::fs::read_to_string(root.join("index.json"))
            .map_err(|e| format!("{}: {e}", root.display()))?;
        let v: serde_json::Value = serde_json::from_str(&s).map_err(|e| e.to_string())?;
        let mut t = BTreeMap::new();
        for (k, e) in v["tensors"].as_object().ok_or("index.json: no tensors")? {
            let shape = e["shape"]
                .as_array()
                .ok_or("shape")?
                .iter()
                .map(|x| x.as_u64().unwrap_or(0) as usize)
                .collect();
            t.insert(
                k.clone(),
                (
                    e["dtype"].as_str().unwrap_or("").to_string(),
                    shape,
                    e["file"].as_str().unwrap_or("").to_string(),
                ),
            );
        }
        Ok(Self {
            root: root.to_path_buf(),
            ckpt: PathBuf::new(),
            t,
            meta: v["meta"].clone(),
        })
    }
    fn has(&self, k: &str) -> bool {
        self.t.contains_key(k)
    }
    fn bytes(&self, k: &str) -> Result<Vec<u8>, String> {
        let (_, _, f) = self
            .t
            .get(k)
            .ok_or_else(|| format!("{}: no tensor {k}", self.root.display()))?;
        std::fs::read(self.root.join(f)).map_err(|e| format!("{k}: {e}"))
    }
    fn shape(&self, k: &str) -> Vec<usize> {
        self.t.get(k).map(|e| e.1.clone()).unwrap_or_default()
    }
    fn dtype(&self, k: &str) -> String {
        self.t.get(k).map(|e| e.0.clone()).unwrap_or_default()
    }
    /// bf16 or f32 tensor as f32 values
    fn f32s(&self, k: &str) -> Result<Vec<f32>, String> {
        let b = self.bytes(k)?;
        match self.dtype(k).as_str() {
            "bf16" => Ok(bf16_to_f32(&b)),
            "f32" => Ok(bytes_f32(&b)),
            d => Err(format!("{k}: dtype {d} is not a float")),
        }
    }
    fn i64s(&self, k: &str) -> Result<Vec<i64>, String> {
        let b = self.bytes(k)?;
        match self.dtype(k).as_str() {
            "i64" => Ok(b
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                .collect()),
            "i32" => Ok(bytes_i32(&b).into_iter().map(|x| x as i64).collect()),
            "u8" => Ok(b.iter().map(|&x| x as i64).collect()),
            d => Err(format!("{k}: dtype {d} is not an integer")),
        }
    }
    fn dev(&self, ops: &Ops, k: &str) -> Result<Buf, String> {
        Buf::from_bytes(&ops.st, &self.bytes(k)?)
    }
    fn meta_f(&self, k: &str) -> f64 {
        self.meta[k].as_f64().unwrap_or(f64::NAN)
    }
    fn meta_u(&self, k: &str) -> usize {
        self.meta[k].as_u64().unwrap_or(0) as usize
    }
}

// ------------------------------------------------------------------------------------------------
// comparison

#[derive(Debug, Clone, Default)]
struct Stat {
    n: usize,
    exact: usize,
    max_abs: f64,
    rms_ref: f64,
    rel_l2: f64,
    max_ulp: u32,
    over_1ulp: usize,
    nonfinite: usize,
}

fn bf16_bits(x: f32) -> u16 {
    (x.to_bits() >> 16) as u16
}
fn ord16(b: u16) -> i32 {
    if b & 0x8000 != 0 {
        -((b & 0x7FFF) as i32)
    } else {
        b as i32
    }
}

/// Compare float outputs; `bf16` counts ULPs on the bf16 grid (inputs are exact bf16 values then).
fn stat(got: &[f32], want: &[f32], bf16: bool) -> Stat {
    let mut s = Stat {
        n: want.len().min(got.len()),
        ..Default::default()
    };
    let (mut se, mut sr) = (0f64, 0f64);
    for (&a, &b) in got.iter().zip(want) {
        if !a.is_finite() || !b.is_finite() {
            if a.to_bits() == b.to_bits() || (a == b) {
                s.exact += 1;
            } else {
                s.nonfinite += 1;
            }
            continue;
        }
        if a.to_bits() == b.to_bits() || (a == 0.0 && b == 0.0) {
            s.exact += 1;
        }
        let d = (a as f64 - b as f64).abs();
        s.max_abs = s.max_abs.max(d);
        se += d * d;
        sr += (b as f64) * (b as f64);
        if bf16 {
            let u = (ord16(bf16_bits(a)) - ord16(bf16_bits(b))).unsigned_abs();
            s.max_ulp = s.max_ulp.max(u);
            if u > 1 {
                s.over_1ulp += 1;
            }
        }
    }
    s.rms_ref = (sr / s.n.max(1) as f64).sqrt();
    s.rel_l2 = if sr > 0.0 {
        (se / sr).sqrt()
    } else {
        se.sqrt()
    };
    if got.len() != want.len() {
        s.nonfinite += got.len().abs_diff(want.len());
    }
    s
}

struct Check {
    name: String,
    stat: Stat,
    rule: String,
    pass: bool,
    extra: String,
}

#[derive(Default)]
struct Report {
    checks: Vec<Check>,
}

impl Report {
    fn add(&mut self, name: &str, stat: Stat, rule: &str, pass: bool, extra: String) {
        println!(
            "{} {name}: n={} exact={} ({:.4}%) max_abs={:.3e} rel_l2={:.3e} max_ulp={} >1ulp={} nonfinite={} {extra} [{rule}]",
            if pass { "PASS" } else { "FAIL" },
            stat.n,
            stat.exact,
            100.0 * stat.exact as f64 / stat.n.max(1) as f64,
            stat.max_abs,
            stat.rel_l2,
            stat.max_ulp,
            stat.over_1ulp,
            stat.nonfinite
        );
        self.checks.push(Check {
            name: name.into(),
            stat,
            rule: rule.into(),
            pass,
            extra,
        });
    }
    /// bit-exact rule
    fn exact(&mut self, name: &str, got: &[f32], want: &[f32], bf16: bool) {
        let s = stat(got, want, bf16);
        let pass = s.exact == s.n && s.nonfinite == 0;
        self.add(name, s, "bit-exact", pass, String::new());
    }
    /// bf16 outputs whose only difference from the reference is fp32 summation order (and, for
    /// attention, the bf16 rounding of P against a different running max): relative L2 error at
    /// most `rel`, max abs error at most `abs_rms` x the reference RMS. The exact share and the
    /// ULP counts are reported, not gated: a near-zero result of a long cancelling sum can sit many
    /// bf16 ULPs off while its absolute error stays at the fp32 rounding of the sum.
    fn close(&mut self, name: &str, got: &[f32], want: &[f32], rel: f64, abs_rms: f64) {
        let s = stat(got, want, true);
        let pass =
            s.nonfinite == 0 && s.rel_l2 <= rel && s.max_abs <= abs_rms * s.rms_ref.max(1e-30);
        self.add(
            name,
            s,
            &format!("rel_l2 <= {rel:e}, max_abs <= {abs_rms:e} x rms"),
            pass,
            String::new(),
        );
    }
    /// fp32 outputs: relative L2 error and max abs relative to the reference RMS
    fn f32rel(&mut self, name: &str, got: &[f32], want: &[f32], rel: f64) {
        let s = stat(got, want, false);
        let pass =
            s.nonfinite == 0 && s.rel_l2 <= rel && s.max_abs <= rel * 100.0 * s.rms_ref.max(1e-30);
        self.add(
            name,
            s,
            &format!("rel_l2 <= {rel:e}, max_abs <= {:e} x rms", rel * 100.0),
            pass,
            String::new(),
        );
    }
    fn raw(&mut self, name: &str, pass: bool, rule: &str, extra: String) {
        println!(
            "{} {name}: {extra} [{rule}]",
            if pass { "PASS" } else { "FAIL" }
        );
        self.checks.push(Check {
            name: name.into(),
            stat: Stat::default(),
            rule: rule.into(),
            pass,
            extra,
        });
    }
    fn json(&self, header: &str) -> String {
        let mut s = format!("{{\n {header},\n \"checks\": [\n");
        for (i, c) in self.checks.iter().enumerate() {
            let st = &c.stat;
            s += &format!(
                "  {{\"name\": {:?}, \"pass\": {}, \"rule\": {:?}, \"n\": {}, \"exact\": {}, \"max_abs\": {:e}, \"rms_ref\": {:e}, \"rel_l2\": {:e}, \"max_ulp\": {}, \"over_1ulp\": {}, \"nonfinite\": {}, \"extra\": {:?}}}{}\n",
                c.name,
                c.pass,
                c.rule,
                st.n,
                st.exact,
                st.max_abs,
                st.rms_ref,
                st.rel_l2,
                st.max_ulp,
                st.over_1ulp,
                st.nonfinite,
                c.extra,
                if i + 1 < self.checks.len() { "," } else { "" }
            );
        }
        let fails = self.checks.iter().filter(|c| !c.pass).count();
        s += &format!(
            " ],\n \"total\": {}, \"failed\": {fails}\n}}\n",
            self.checks.len()
        );
        s
    }
}

fn bf(ops: &Ops, b: &Buf) -> Result<Vec<f32>, String> {
    Ok(bf16_to_f32(&b.host(&ops.st)?))
}
fn f32h(ops: &Ops, b: &Buf) -> Result<Vec<f32>, String> {
    Ok(bytes_f32(&b.host(&ops.st)?))
}

/// Tie-aware top-k check: per row, the reference and the kernel must agree on every position
/// scoring above the k-th score and take the same number at the k-th score (any of the tied ones).
fn topk_agree(
    score: &[f32],
    width: usize,
    rows: usize,
    want: &[i64],
    got: &[i64],
    k: usize,
    offset: i64,
) -> (usize, usize, String) {
    let mut bad = 0usize;
    let mut first = String::new();
    for r in 0..rows {
        let sc = &score[r * width..(r + 1) * width];
        let wset: Vec<i64> = want[r * k..(r + 1) * k]
            .iter()
            .filter(|&&x| x >= 0)
            .map(|x| x - offset)
            .collect();
        let gset: Vec<i64> = got[r * k..(r + 1) * k]
            .iter()
            .filter(|&&x| x >= 0)
            .map(|x| x - offset)
            .collect();
        let ok = (|| {
            if wset.len() != gset.len() {
                return false;
            }
            if wset.is_empty() {
                return true;
            }
            let thr = wset
                .iter()
                .map(|&i| sc[i as usize])
                .fold(f32::INFINITY, f32::min);
            let above = |s: &Vec<i64>| -> Vec<i64> {
                let mut v: Vec<i64> = s
                    .iter()
                    .copied()
                    .filter(|&i| sc[i as usize] > thr)
                    .collect();
                v.sort();
                v
            };
            let at = |s: &Vec<i64>| s.iter().filter(|&&i| sc[i as usize] == thr).count();
            gset.iter().all(|&i| sc[i as usize] >= thr)
                && above(&wset) == above(&gset)
                && at(&wset) == at(&gset)
        })();
        if !ok {
            bad += 1;
            if first.is_empty() {
                first = format!(
                    "row {r}: want {:?} got {:?}",
                    &want[r * k..(r * k + k.min(8))],
                    &got[r * k..(r * k + k.min(8))]
                );
            }
        }
    }
    (rows - bad, bad, first)
}

// ------------------------------------------------------------------------------------------------
// kernel gates on random inputs

fn kernels(root: &Path, rep: &mut Report) -> Result<(), String> {
    let e = memra_engine::Engine::new(0).map_err(|e| e.to_string())?;
    let ops = Ops::new(e.stream());
    let st = ops.st.clone();
    let sv = st.cu_stream() as *mut std::os::raw::c_void;

    // act_quant
    {
        let f = Fx::open(&root.join("actq"))?;
        let sh = f.shape("x");
        let (m, k) = (sh[0], sh[1]);
        let x = f.dev(&ops, "x")?;
        let q = ops.buf(m * k)?;
        // zero-filled: the layout's padding stays zero
        let s = ops.buf(v::sf_bytes(m, k))?;
        ops.act_quant(x.p(&st), k, Dp::NULL, m, k, q.p(&st), s.p(&st))?;
        let y = ops.buf(m * k * 2)?;
        v::copy_d2d(&ops, &x, 0, &y, 0, m * k * 2)?;
        ops.fp8_qdq(y.p(&st), k, m, k)?;
        let qh = q.host(&st)?;
        let sh_ = s.host(&st)?;
        let wq = f.bytes("q")?;
        // the reference's row-major scales in the GEMMs' layout
        let ws = {
            let raw = f.bytes("s")?;
            v::sf_pack(m, k, |r, b| raw[r * (k / 32) + b])
        };
        let qe = qh.iter().zip(&wq).filter(|(a, b)| a == b).count();
        let se = sh_.iter().zip(&ws).filter(|(a, b)| a == b).count();
        rep.raw(
            "act_quant/codes",
            qe == wq.len(),
            "bit-exact e4m3 codes",
            format!("{qe}/{}", wq.len()),
        );
        rep.raw(
            "act_quant/scales",
            se == ws.len(),
            "bit-exact ue8m0 scales, in the GEMMs' layout",
            format!("{se}/{}", ws.len()),
        );
        rep.exact("act_quant/qdq", &bf(&ops, &y)?, &f.f32s("qdq")?, true);
    }
    // fp8 32x32 block GEMM (linear(): act_quant + fp8_gemm), bf16 out
    for dir in sorted_dirs(root, "fp8gemm_")? {
        let f = Fx::open(&dir)?;
        let xs = f.shape("x");
        let ws = f.shape("w");
        let (m, k, n) = (xs[0], xs[1], ws[0]);
        let x = f.dev(&ops, "x")?;
        let w = qw_fx(&f, &ops, "w", "ws", n, k, QFmt::Fp8Block32)?;
        let y = ops.buf(m * n * 2)?;
        let mut scr = v::Scratch::default();
        ops.linear_q(x.p(&st), k, m, &w, y.p(&st), n, false, &mut scr)?;
        // fp32 accumulation order differs from the reference GEMM: bf16 rounding of an fp32 sum
        rep.close(
            &format!("fp8_gemm/{}", dir.file_name().unwrap().to_string_lossy()),
            &bf(&ops, &y)?,
            &f.f32s("y")?,
            2e-4,
            0.03,
        );
    }
    // fp4 quant-dequant
    {
        let f = Fx::open(&root.join("fp4qdq"))?;
        for (g, xn, yn) in [(32usize, "x32", "y32"), (16, "x16", "y16")] {
            let sh = f.shape(xn);
            let x = f.dev(&ops, xn)?;
            ops.fp4_qdq(x.p(&st), sh[1], sh[0], sh[1], g)?;
            rep.exact(&format!("fp4_qdq/g{g}"), &bf(&ops, &x)?, &f.f32s(yn)?, true);
        }
    }
    // RMSNorm, RoPE tables and rotation
    {
        let f = Fx::open(&root.join("norm_rope"))?;
        for d in [5120usize, 1280, 512, 128] {
            let x = f.dev(&ops, &format!("norm{d}/x"))?;
            let w = f.dev(&ops, &format!("norm{d}/w"))?;
            let rows = f.shape(&format!("norm{d}/x"))[0];
            let y = ops.buf(rows * d * 2)?;
            ops.rmsnorm(x.p(&st), d, &w, y.p(&st), d, rows, d, 1e-20)?;
            rep.close(
                &format!("rmsnorm/{d}"),
                &bf(&ops, &y)?,
                &f.f32s(&format!("norm{d}/y"))?,
                2e-4,
                0.03,
            );
        }
        let start = f.meta_u("rope_start");
        for (ratio, orig, theta) in [(0usize, 0usize, 10000f32), (1, 65536, 160000.0)] {
            let (c, s) = v::rope_tables_host(64, 1024, orig, theta, 16.0, 32.0, 1.0);
            rep.f32rel(
                &format!("rope{ratio}/cos_table"),
                &c,
                &f.f32s(&format!("freqs{ratio}/cos"))?,
                1e-6,
            );
            rep.f32rel(
                &format!("rope{ratio}/sin_table"),
                &s,
                &f.f32s(&format!("freqs{ratio}/sin"))?,
                1e-6,
            );
            let t = v::RopeTable::new(&ops, 1024, orig, theta, 16.0, 32.0, 1.0)?;
            let sh = f.shape(&format!("rope{ratio}/x")); // [1, s, h, 64]
            let (sq, h) = (sh[1], sh[2]);
            let x = f.dev(&ops, &format!("rope{ratio}/x"))?;
            ops.rope(x.p(&st), h * 64, sq, h, 64, &t, start, 1, false)?;
            rep.close(
                &format!("rope{ratio}/forward"),
                &bf(&ops, &x)?,
                &f.f32s(&format!("rope{ratio}/y"))?,
                2e-4,
                0.03,
            );
            ops.rope(x.p(&st), h * 64, sq, h, 64, &t, start, 1, true)?;
            rep.close(
                &format!("rope{ratio}/inverse"),
                &bf(&ops, &x)?,
                &f.f32s(&format!("rope{ratio}/yinv"))?,
                2e-4,
                0.03,
            );
        }
    }
    // sink sparse attention
    // (rows, heads, keys, q, kv, idx, sink, scale, output) of each fixture, for the two-thread run
    #[allow(clippy::type_complexity)]
    let mut sattn: Vec<(usize, usize, usize, [Vec<u8>; 4], f32, Vec<u8>)> = Vec::new();
    for dir in sorted_dirs(root, "sattn_")? {
        let f = Fx::open(&dir)?;
        let qs = f.shape("q"); // [1, m, 64, 512]
        let is = f.shape("idx"); // [1, m, k]
        let (m, h, k) = (qs[1], qs[2], is[2]);
        let q = f.dev(&ops, "q")?;
        let kv = f.dev(&ops, "kv")?;
        let idx = f.dev(&ops, "idx")?;
        let sink = f.dev(&ops, "sink")?;
        let o = ops.buf(m * h * 512 * 2)?;
        ops.sparse_attn(
            q.p(&st),
            h * 512,
            kv.p(&st),
            idx.p(&st),
            k,
            k,
            sink.p(&st),
            f.meta_f("scale") as f32,
            o.p(&st),
            h * 512,
            m,
            h,
        )?;
        sattn.push((
            m,
            h,
            k,
            [q.host(&st)?, kv.host(&st)?, idx.host(&st)?, sink.host(&st)?],
            f.meta_f("scale") as f32,
            o.host(&st)?,
        ));
        // the keyed entry on two separate buffers (the first half of the key rows as the window,
        // the rest as the compressed rows) against the one-buffer entry: byte for byte
        {
            let kvh = kv.host(&st)?;
            let nwin = kvh.len() / 1024 / 2;
            let win = Buf::from_bytes(&st, &kvh[..nwin * 1024])?;
            let comp = Buf::from_bytes(&st, &kvh[nwin * 1024..])?;
            let tab =
                Buf::from_bytes(&st, &v::key_refs(&vec![(win.p(&st), comp.p(&st), nwin); m]))?;
            let o2 = ops.buf(m * h * 512 * 2)?;
            ops.sparse_attn_keys(
                q.p(&st),
                h * 512,
                tab.p(&st),
                idx.p(&st),
                k,
                k,
                sink.p(&st),
                f.meta_f("scale") as f32,
                o2.p(&st),
                h * 512,
                m,
                h,
            )?;
            let name = dir.file_name().unwrap().to_string_lossy().to_string();
            let (got, want) = (o2.host(&st)?, o.host(&st)?);
            let diff = got.iter().zip(&want).filter(|(a, b)| a != b).count()
                + got.len().abs_diff(want.len());
            rep.raw(
                &format!("sparse_attn/{name}/keyed_equals_one_buffer"),
                diff == 0,
                "byte-identical to the one-buffer entry",
                format!("{} bytes, {diff} differ", want.len()),
            );
        }
        // Against the tilelang schedule (key blocks of 64, online max, bf16 P) the only difference
        // is fp32 order: tight. Against the torch stand-in, which rounds P to bf16 against one global
        // max, every probability can sit one bf16 rounding (2^-9 relative) apart: the stand-in is
        // itself 2.3e-3 rel_l2 from an fp32-P attention on this fixture, so 3e-3 is that band.
        let name = dir.file_name().unwrap().to_string_lossy().to_string();
        let got = bf(&ops, &o)?;
        rep.close(
            &format!("sparse_attn/{name}/vs_schedule"),
            &got,
            &f.f32s("o_sched")?,
            5e-4,
            0.03,
        );
        rep.close(
            &format!("sparse_attn/{name}/vs_torch_standin"),
            &got,
            &f.f32s("o")?,
            3e-3,
            0.25,
        );
    }
    // Two threads on the card, each on its own stream, launching the attention with different
    // index widths at once: a kernel's shared-memory limit is one setting a function a context, so
    // a launcher that set it to the call's own size let one thread launch above the limit the
    // other had just set (CUDA error 1 or 701). Every launch must succeed and write the bytes of
    // the one-thread run.
    if sattn.len() >= 2 {
        let fails = std::sync::Mutex::new(Vec::<String>::new());
        std::thread::scope(|sc| {
            for (ti, (m, h, k, bufs, scale, want)) in sattn.iter().take(2).enumerate() {
                let (st, fails) = (&st, &fails);
                sc.spawn(move || {
                    let run = || -> Result<(), String> {
                        let s2 = st
                            .context()
                            .new_stream()
                            .map_err(|e| format!("stream: {e:?}"))?;
                        let o2 = Ops::new(s2.clone());
                        let q = Buf::from_bytes(&s2, &bufs[0])?;
                        let kv = Buf::from_bytes(&s2, &bufs[1])?;
                        let idx = Buf::from_bytes(&s2, &bufs[2])?;
                        let sink = Buf::from_bytes(&s2, &bufs[3])?;
                        let o = o2.buf(m * h * 512 * 2)?;
                        for it in 0..400 {
                            o2.sparse_attn(
                                q.p(&s2),
                                h * 512,
                                kv.p(&s2),
                                idx.p(&s2),
                                *k,
                                *k,
                                sink.p(&s2),
                                *scale,
                                o.p(&s2),
                                h * 512,
                                *m,
                                *h,
                            )?;
                            if it % 16 == 15 {
                                o2.sync()?;
                            }
                        }
                        if &o.host(&s2)? != want {
                            return Err("bytes differ from the one-thread run".into());
                        }
                        Ok(())
                    };
                    if let Err(e) = run() {
                        fails.lock().unwrap().push(format!("thread {ti}: {e}"));
                    }
                });
            }
        });
        let fails = fails.into_inner().unwrap();
        rep.raw(
            "stress/sparse_attn_two_threads_two_widths",
            fails.is_empty(),
            "800 launches from two threads succeed and match the one-thread bytes",
            if fails.is_empty() {
                "ok".into()
            } else {
                fails.join("; ")
            },
        );
    }
    // mHC
    {
        let f = Fx::open(&root.join("hc"))?;
        let xs = f.shape("x"); // [1, s, 4, 5120]
        let (s, hc, d) = (xs[1], xs[2], xs[3]);
        let x = f.dev(&ops, "x")?;
        let fn_ = f.dev(&ops, "fn")?;
        let sc = f.dev(&ops, "scale")?;
        let base = f.dev(&ops, "base")?;
        let nm = (2 + hc) * hc;
        let mixes = ops.buf(s * nm * 4)?;
        let pre = ops.buf(s * hc * 4)?;
        let post = ops.buf(s * hc * 4)?;
        let comb = ops.buf(s * hc * hc * 4)?;
        let parts = ops.buf(s * 8 * (nm + 1) * 4)?;
        ck("hc_mixes", unsafe {
            ffi::memra_v41_hc_mixes(
                x.p(&st).m(),
                fn_.p(&st).f(),
                sc.p(&st).f(),
                base.p(&st).f(),
                s as i32,
                hc as i32,
                d as i32,
                20,
                1e-20,
                1e-6,
                parts.p(&st).fm(),
                mixes.p(&st).fm(),
                pre.p(&st).fm(),
                post.p(&st).fm(),
                comb.p(&st).fm(),
                sv,
            )
        })?;
        rep.f32rel("hc/mixes", &f32h(&ops, &mixes)?, &f.f32s("mixes")?, 1e-5);
        rep.f32rel("hc/pre", &f32h(&ops, &pre)?, &f.f32s("pre")?, 1e-5);
        rep.f32rel("hc/post", &f32h(&ops, &post)?, &f.f32s("post")?, 1e-5);
        rep.f32rel("hc/comb", &f32h(&ops, &comb)?, &f.f32s("comb")?, 1e-5);
        // the one-token schedule (a decode step; up to 8 rows of a verification or a batch) against
        // the first rows of the many-token run, byte for byte
        for r1 in [1usize, s.min(8)] {
            let mixes1 = ops.buf(r1 * nm * 4)?;
            let pre1 = ops.buf(r1 * hc * 4)?;
            let post1 = ops.buf(r1 * hc * 4)?;
            let comb1 = ops.buf(r1 * hc * hc * 4)?;
            let parts1 = ops.buf(r1 * 8 * (nm + 1) * 4)?;
            ck("hc_mixes", unsafe {
                ffi::memra_v41_hc_mixes(
                    x.p(&st).m(),
                    fn_.p(&st).f(),
                    sc.p(&st).f(),
                    base.p(&st).f(),
                    r1 as i32,
                    hc as i32,
                    d as i32,
                    20,
                    1e-20,
                    1e-6,
                    parts1.p(&st).fm(),
                    mixes1.p(&st).fm(),
                    pre1.p(&st).fm(),
                    post1.p(&st).fm(),
                    comb1.p(&st).fm(),
                    sv,
                )
            })?;
            for (nm_, one, all, n1) in [
                ("mixes", &mixes1, &mixes, r1 * nm * 4),
                ("pre", &pre1, &pre, r1 * hc * 4),
                ("post", &post1, &post, r1 * hc * 4),
                ("comb", &comb1, &comb, r1 * hc * hc * 4),
            ] {
                let a = one.host(&st)?;
                let b = all.host(&st)?;
                let same = a[..n1] == b[..n1];
                rep.raw(
                    &format!("schedule/hc_mixes/rows{r1}/{nm_}"),
                    same,
                    "the one-token schedule writes the many-token run's first rows",
                    format!("{n1} bytes"),
                );
            }
        }
        let carried = f.dev(&ops, "carried")?;
        let y = ops.buf(s * d * 2)?;
        ck("hc_pre", unsafe {
            ffi::memra_v41_hc_pre(
                x.p(&st).m(),
                carried.p(&st).f(),
                s as i32,
                hc as i32,
                d as i32,
                y.p(&st).m(),
                d as i64,
                sv,
            )
        })?;
        rep.close("hc/pre_collapse", &bf(&ops, &y)?, &f.f32s("y")?, 2e-4, 0.03);
        // hc_post from the reference post/comb, so this isolates the expand kernel
        let rpost = Buf::from_f32(&st, &f.f32s("post")?)?;
        let rcomb = Buf::from_f32(&st, &f.f32s("comb")?)?;
        let sub = f.dev(&ops, "sub")?;
        let z = ops.buf(s * hc * d * 2)?;
        ck("hc_post", unsafe {
            ffi::memra_v41_hc_post(
                sub.p(&st).m(),
                d as i64,
                x.p(&st).m(),
                rpost.p(&st).f(),
                rcomb.p(&st).f(),
                s as i32,
                hc as i32,
                d as i32,
                z.p(&st).m(),
                sv,
            )
        })?;
        rep.close("hc/post_expand", &bf(&ops, &z)?, &f.f32s("z")?, 2e-4, 0.03);
    }
    // compressor
    for (case, ratio) in [("comp_r1", 1usize), ("comp_r2", 2)] {
        let f = Fx::open(&root.join(case))?;
        let s = f.meta_u("seqlen");
        let x = f.dev(&ops, "x")?;
        let wkv = v::BW {
            w: f.dev(&ops, "wkv")?,
            n: 512,
            k: 5120,
        };
        let norm = f.dev(&ops, "norm")?;
        let n_c = s / ratio;
        let lat = ops.buf(n_c * 512 * 2)?;
        if ratio == 1 {
            ops.linear_bf16(x.p(&st), 5120, s, &wkv, lat.p(&st), 512, false)?;
        } else {
            let wg = v::BW {
                w: f.dev(&ops, "wgate")?,
                n: 512,
                k: 5120,
            };
            let kvf = ops.buf(s * 512 * 4)?;
            let scf = ops.buf(s * 512 * 4)?;
            ops.linear_bf16(x.p(&st), 5120, s, &wkv, kvf.p(&st), 512, true)?;
            ops.linear_bf16(x.p(&st), 5120, s, &wg, scf.p(&st), 512, true)?;
            ck("comp_pool2", unsafe {
                ffi::memra_v41_comp_pool2(
                    kvf.p(&st).f(),
                    scf.p(&st).f(),
                    512,
                    lat.p(&st).m(),
                    512,
                    n_c as i32,
                    512,
                    sv,
                )
            })?;
        }
        ops.rmsnorm(lat.p(&st), 512, &norm, lat.p(&st), 512, n_c, 512, 1e-20)?;
        rep.close(
            &format!("compressor/r{ratio}/latent"),
            &bf(&ops, &lat)?,
            &f.f32s("prefill")?,
            1e-3,
            0.05,
        );
        let t = v::RopeTable::new(&ops, 1024, 65536, 160000.0, 16.0, 32.0, 1.0)?;
        ops.rope(lat.p(&st), 512, n_c, 1, 512, &t, 0, ratio, false)?;
        ops.fp4_qdq(lat.p(&st), 512, n_c, 512, 16)?;
        // fp4 g16 rounding can move a value one grid step when its input moved one bf16 ulp
        let got = bf(&ops, &lat)?;
        let want = f.f32s("cache")?;
        let s_ = stat(&got, &want, true);
        let frac = 1.0 - s_.exact as f64 / s_.n as f64;
        rep.add(
            &format!("compressor/r{ratio}/cache_fp4"),
            s_.clone(),
            "<= 1% of elements off the reference fp4 grid point, rel_l2 <= 5e-2",
            frac <= 0.01 && s_.rel_l2 <= 5e-2,
            format!("off_grid_frac={frac:.4}"),
        );
    }
    // indexer pieces: q path, score, candidates, top-k (from the reference q/k/w, so each kernel is
    // isolated), plus the indexer's own q path from qr
    {
        let f = Fx::open(&root.join("indexer"))?;
        let s_ = f.meta_u("seqlen");
        let topk = f.meta_u("index_topk");
        let cbk = f.meta_u("candidate_topk_blocks");
        let cb = f.meta_u("candidate_block_size");
        let comp = v::RopeTable::new(&ops, 1024, 65536, 160000.0, 16.0, 32.0, 1.0)?;
        for (step, sp, ln) in [(0usize, 0usize, s_), (1, s_, 1), (2, s_ + 1, 1)] {
            let n_c = sp + ln;
            for lid in [20usize, 24] {
                let p = format!("s{step}/L{lid}");
                // q path: wq_b (fp8) on qr rows, RoPE at the query positions, fp4 g32
                let qr_all = f.f32s("qr")?;
                let qr_rows: Vec<u8> = qr_all[sp * 1280..(sp + ln) * 1280]
                    .iter()
                    .flat_map(|&x| v::f32_to_bf16_bits(x).to_le_bytes())
                    .collect();
                let qr = Buf::from_bytes(&st, &qr_rows)?;
                let w = qw_fx(
                    &f,
                    &ops,
                    &format!("L{lid}/wq_b"),
                    &format!("L{lid}/wq_b_s"),
                    4096,
                    1280,
                    QFmt::Fp8Block32,
                )?;
                let q = ops.buf(ln * 4096 * 2)?;
                let mut scr = v::Scratch::default();
                ops.linear_q(qr.p(&st), 1280, ln, &w, q.p(&st), 4096, false, &mut scr)?;
                rep.close(
                    &format!("indexer/{p}/q_lin"),
                    &bf(&ops, &q)?,
                    &f.f32s(&format!("{p}/q_lin"))?,
                    2e-4,
                    0.03,
                );
                ops.rope(q.p(&st), 4096, ln, 32, 128, &comp, sp, 1, false)?;
                ops.fp4_qdq(q.p(&st), 4096, ln, 4096, 32)?;
                let qs = stat(&bf(&ops, &q)?, &f.f32s(&format!("{p}/q"))?, true);
                let off = 1.0 - qs.exact as f64 / qs.n as f64;
                rep.add(
                    &format!("indexer/{p}/q_fp4"),
                    qs.clone(),
                    "<= 1% off the reference fp4 grid point",
                    off <= 0.01,
                    format!("off_grid_frac={off:.4}"),
                );
                // score from the reference q, keys and weights
                let rq = f.dev(&ops, &format!("{p}/q"))?;
                let rk = f.dev(&ops, &format!("s{step}/index_k"))?;
                let rw = f.dev(&ops, &format!("{p}/w"))?;
                let sc = ops.buf(ln * n_c * 2)?;
                ck("index_score", unsafe {
                    ffi::memra_v41_index_score(
                        rq.p(&st).m(),
                        4096,
                        rk.p(&st).m(),
                        n_c as i32,
                        rw.p(&st).m(),
                        32,
                        1.0,
                        sp as i32,
                        1,
                        std::ptr::null(),
                        0,
                        cb as i32,
                        sc.p(&st).m(),
                        n_c as i64,
                        ln as i32,
                        sv,
                    )
                })?;
                let mut want = f.f32s(&format!("{p}/score_raw"))?;
                if sp == 0 {
                    for i in 0..ln {
                        for t in (i + 1)..n_c {
                            want[i * n_c + t] = f32::NEG_INFINITY;
                        }
                    }
                }
                let got_sc = bf(&ops, &sc)?;
                rep.close(&format!("indexer/{p}/score"), &got_sc, &want, 1e-3, 0.03);
                // the candidate pool (source) and the rescored layer's mask use the reference score
                let wsc = Buf::from_bytes(
                    &st,
                    &want
                        .iter()
                        .flat_map(|&x| v::f32_to_bf16_bits(x).to_le_bytes())
                        .collect::<Vec<u8>>(),
                )?;
                let mut score_for_topk = want.clone();
                if lid == 20 {
                    let nb = n_c.div_ceil(cb);
                    let keep = ops.buf(ln * nb)?;
                    ck("candidates", unsafe {
                        ffi::memra_v41_candidates(
                            wsc.p(&st).m(),
                            n_c as i64,
                            n_c as i32,
                            cb as i32,
                            cbk as i32,
                            sp as i32,
                            1,
                            keep.p(&st).u8m(),
                            nb as i64,
                            ln as i32,
                            sv,
                        )
                    })?;
                    let kh: Vec<i64> = keep.host(&st)?.iter().map(|&b| b as i64).collect();
                    let wc = f.i64s(&format!("s{step}/cand"))?; // [1, ln, n_c] bool per position
                    let mut wblk = vec![0i64; ln * nb];
                    for i in 0..ln {
                        for b in 0..nb {
                            wblk[i * nb + b] = wc[i * n_c + b * cb];
                        }
                    }
                    // tie-aware on block maxima
                    let mut bmax = vec![f32::NEG_INFINITY; ln * nb];
                    for i in 0..ln {
                        let vis = (sp + i + 1).min(n_c);
                        for b in 0..nb {
                            let mut m = f32::NEG_INFINITY;
                            for t in b * cb..((b + 1) * cb).min(n_c) {
                                m = m.max(want[i * n_c + t]);
                            }
                            bmax[i * nb + b] = if vis > 0 && b == (vis - 1) / cb {
                                f32::INFINITY
                            } else {
                                m
                            };
                        }
                    }
                    let to_list = |mask: &[i64]| -> Vec<i64> {
                        let mut out = vec![-1i64; ln * nb];
                        for i in 0..ln {
                            let mut j = 0;
                            for b in 0..nb {
                                if mask[i * nb + b] != 0 {
                                    out[i * nb + j] = b as i64;
                                    j += 1;
                                }
                            }
                        }
                        out
                    };
                    let (ok, bad, first) =
                        topk_agree(&bmax, nb, ln, &to_list(&wblk), &to_list(&kh), nb, 0);
                    rep.raw(
                        &format!("indexer/{p}/candidates"),
                        bad == 0,
                        "tie-aware equal kept-block sets per row",
                        format!("{ok} rows agree, {bad} differ {first}"),
                    );
                } else {
                    // rescoring layer: mask by the reference candidates
                    let wc = f.i64s(&format!("s{step}/cand"))?;
                    for i in 0..ln {
                        for t in 0..n_c {
                            if wc[i * n_c + t] == 0 {
                                score_for_topk[i * n_c + t] = f32::NEG_INFINITY;
                            }
                        }
                    }
                }
                let msc = Buf::from_bytes(
                    &st,
                    &score_for_topk
                        .iter()
                        .flat_map(|&x| v::f32_to_bf16_bits(x).to_le_bytes())
                        .collect::<Vec<u8>>(),
                )?;
                let tw = topk.min(n_c);
                let offset = if sp == 0 { ln } else { 128 };
                let out = ops.buf(ln * tw * 4)?;
                ck("topk", unsafe {
                    ffi::memra_v41_topk(
                        msc.p(&st).m(),
                        n_c as i64,
                        n_c as i32,
                        tw as i32,
                        tw as i32,
                        offset as i32,
                        out.p(&st).im(),
                        tw as i64,
                        ln as i32,
                        sv,
                    )
                })?;
                let got: Vec<i64> = bytes_i32(&out.host(&st)?)
                    .into_iter()
                    .map(|x| x as i64)
                    .collect();
                let wantidx = f.i64s(&format!("{p}/idx"))?;
                let (ok, bad, first) =
                    topk_agree(&score_for_topk, n_c, ln, &wantidx, &got, tw, offset as i64);
                rep.raw(
                    &format!("indexer/{p}/topk"),
                    bad == 0,
                    "tie-aware equal selections per row",
                    format!("{ok} rows agree, {bad} differ {first}"),
                );
            }
        }
    }
    // MoE: routing, one expert with clamps engaged, the shared expert, the full layer
    {
        let f = Fx::open(&root.join("moe"))?;
        let xs = f.shape("x");
        let s = xs[1];
        let ne = f.meta_u("n_experts");
        let tk = f.meta_u("topk");
        let rs = f.meta_f("route_scale") as f32;
        let lim = f.meta_f("swiglu_limit") as f32;
        let x = f.dev(&ops, "x")?;
        let gw = v::BW {
            w: f.dev(&ops, "gate_w")?,
            n: ne,
            k: 5120,
        };
        let gb = f.dev(&ops, "gate_b")?;
        let raw = ops.buf(s * ne * 4)?;
        ops.linear_bf16(x.p(&st), 5120, s, &gw, raw.p(&st), ne, true)?;
        let sel = ops.buf(s * tk * 4)?;
        let selw = ops.buf(s * tk * 4)?;
        ck("route", unsafe {
            ffi::memra_v41_route(
                raw.p(&st).f(),
                ne as i64,
                gb.p(&st).f(),
                ne as i32,
                tk as i32,
                1,
                rs,
                sel.p(&st).im(),
                selw.p(&st).fm(),
                s as i32,
                sv,
            )
        })?;
        let gi: Vec<i64> = bytes_i32(&sel.host(&st)?)
            .into_iter()
            .map(|x| x as i64)
            .collect();
        let wi = f.i64s("route_i")?;
        let same = gi.iter().zip(&wi).filter(|(a, b)| a == b).count();
        rep.raw(
            "moe/route_ids",
            same == wi.len(),
            "identical expert ids and order",
            format!("{same}/{}", wi.len()),
        );
        // a router row with NaN raw values: a NaN score never compares, so past the row's
        // orderable experts the selection names none (-1, weight 0) instead of an expert
        // already taken
        {
            let orderable = 2usize;
            let mut row = vec![f32::NAN; ne];
            for (i, v) in row.iter_mut().take(orderable).enumerate() {
                *v = 1.0 + i as f32;
            }
            let draw = Buf::from_f32(&st, &row)?;
            let nsel = ops.buf(tk * 4)?;
            let nselw = ops.buf(tk * 4)?;
            ck("route", unsafe {
                ffi::memra_v41_route(
                    draw.p(&st).f(),
                    ne as i64,
                    gb.p(&st).f(),
                    ne as i32,
                    tk as i32,
                    1,
                    rs,
                    nsel.p(&st).im(),
                    nselw.p(&st).fm(),
                    1,
                    sv,
                )
            })?;
            let ids = bytes_i32(&nsel.host(&st)?);
            let w = bytes_f32(&nselw.host(&st)?);
            let named: Vec<i32> = ids.iter().copied().filter(|&e| e >= 0).collect();
            let mut uniq = named.clone();
            uniq.sort_unstable();
            uniq.dedup();
            let ok = named.len() == orderable.min(tk)
                && uniq.len() == named.len()
                && named.iter().all(|&e| (e as usize) < orderable)
                && ids.iter().zip(&w).all(|(&e, &x)| e >= 0 || x == 0.0);
            rep.raw(
                "moe/route_nan_row",
                ok,
                "a row of NaN raw values past 2 experts: those 2 named once each, every other slot -1 with weight 0",
                format!("ids {ids:?}"),
            );
        }
        rep.f32rel(
            "moe/route_w",
            &f32h(&ops, &selw)?,
            &f.f32s("route_w")?,
            // the fp32 router GEMM's summation order over 5120 inputs: ~1e-6 relative
            1e-5,
        );
        let ex = |tag: &str, i: usize, n: usize, k: usize| -> Result<QW, String> {
            let fmt = if f.dtype(&format!("{tag}/w{i}")) == "f8e4m3" {
                QFmt::Fp8Block32
            } else {
                QFmt::Mxfp4
            };
            qw_fx(
                &f,
                &ops,
                &format!("{tag}/w{i}"),
                &format!("{tag}/w{i}_s"),
                n,
                k,
                fmt,
            )
        };
        let run_expert =
            |w: &[QW; 3], xin: &Buf, rows: usize, wrow: Option<&Buf>| -> Result<Vec<f32>, String> {
                let mut scr = v::Scratch::default();
                let g = ops.buf(rows * 2304 * 2)?;
                let u = ops.buf(rows * 2304 * 2)?;
                let h = ops.buf(rows * 2304 * 2)?;
                let d = ops.buf(rows * 5120 * 2)?;
                ops.linear_q(
                    xin.p(&st),
                    5120,
                    rows,
                    &w[0],
                    g.p(&st),
                    2304,
                    false,
                    &mut scr,
                )?;
                ops.linear_q(
                    xin.p(&st),
                    5120,
                    rows,
                    &w[2],
                    u.p(&st),
                    2304,
                    false,
                    &mut scr,
                )?;
                ck("swiglu", unsafe {
                    ffi::memra_v41_swiglu(
                        g.p(&st).m(),
                        u.p(&st).m(),
                        2304,
                        wrow.map_or(std::ptr::null(), |b| b.p(&st).f()),
                        lim,
                        h.p(&st).m(),
                        2304,
                        rows as i32,
                        2304,
                        sv,
                    )
                })?;
                ops.linear_q(h.p(&st), 2304, rows, &w[1], d.p(&st), 5120, false, &mut scr)?;
                bf(&ops, &d)
            };
        let e0 = [
            ex("e0", 1, 2304, 5120)?,
            ex("e0", 2, 5120, 2304)?,
            ex("e0", 3, 2304, 5120)?,
        ];
        let xe = f.dev(&ops, "solo/x")?;
        let we = f.dev(&ops, "solo/wt")?;
        let rows = f.shape("solo/x")[0];
        rep.close(
            "moe/expert_mxfp4_clamped",
            &run_expert(&e0, &xe, rows, Some(&we))?,
            &f.f32s("solo/y")?,
            2e-4,
            0.03,
        );
        // the minted NVFP4 form of the same expert must give the same bits (the mint is checked and
        // the device holds the ue8m0 codes it came from)
        let mint = |w: &QW| -> Result<QW, String> {
            let sc = w.s.host(&st)?;
            let (_, s2) = v::mint_mxfp4_scales(&sc)?;
            Ok(QW {
                w: Buf::from_bytes(&st, &w.w.host(&st)?)?,
                s: Buf::from_bytes(&st, &sc)?,
                n: w.n,
                k: w.k,
                fmt: QFmt::Nvfp4Minted { scale2: s2 },
            })
        };
        let e0m = [mint(&e0[0])?, mint(&e0[1])?, mint(&e0[2])?];
        rep.exact(
            "moe/expert_minted_nvfp4_equals_mxfp4",
            &run_expert(&e0m, &xe, rows, Some(&we))?,
            &run_expert(&e0, &xe, rows, Some(&we))?,
            true,
        );
        let sh = [
            ex("shared", 1, 2304, 5120)?,
            ex("shared", 2, 5120, 2304)?,
            ex("shared", 3, 2304, 5120)?,
        ];
        rep.close(
            "moe/shared_fp8_clamped",
            &run_expert(&sh, &xe, rows, None)?,
            &f.f32s("solo_shared/y")?,
            2e-4,
            0.03,
        );
        // the full MoE through the engine's layer code path
        let experts: Vec<[QW; 3]> = (0..ne)
            .map(|i| -> Result<[QW; 3], String> {
                let t = format!("e{i}");
                Ok([
                    ex(&t, 1, 2304, 5120)?,
                    ex(&t, 2, 5120, 2304)?,
                    ex(&t, 3, 2304, 5120)?,
                ])
            })
            .collect::<Result<_, _>>()?;
        let y = moe_only(&ops, &x, s, &gw, &gb, &experts, &sh, tk, rs, lim)?;
        rep.close("moe/full", &y, &f.f32s("y")?, 2e-3, 0.05);
    }
    // Engram (GPU side) from the reference hash ids
    {
        let f = Fx::open(&root.join("engram"))?;
        let hs = f.shape("h"); // [1, s, 4, 5120]
        let s = hs[1];
        let hash = f.i64s("hash")?;
        let table = f.bytes("table")?;
        let tsc = f.bytes("table_s")?;
        let mut rows = Vec::new();
        let mut scs = Vec::new();
        for &hh in &hash {
            let r = hh as usize;
            rows.extend_from_slice(&table[r * 256..(r + 1) * 256]);
            scs.extend_from_slice(&tsc[r * 8..(r + 1) * 8]);
        }
        let dr = Buf::from_bytes(&st, &rows)?;
        let ds = Buf::from_bytes(&st, &scs)?;
        let emb = ops.buf(s * 6144 * 2)?;
        ck("engram_dequant", unsafe {
            ffi::memra_v41_engram_dequant(
                dr.p(&st).m(),
                ds.p(&st).m(),
                (s * 24) as i64,
                256,
                emb.p(&st).m(),
                sv,
            )
        })?;
        rep.exact("engram/dequant", &bf(&ops, &emb)?, &f.f32s("emb")?, true);
        let wkv = qw_fx(&f, &ops, "wkv", "wkv_s", 25600, 6144, QFmt::Fp8Block32)?;
        let kv = ops.buf(s * 25600 * 2)?;
        let mut scr = v::Scratch::default();
        ops.linear_q(emb.p(&st), 6144, s, &wkv, kv.p(&st), 25600, false, &mut scr)?;
        rep.close("engram/wkv", &bf(&ops, &kv)?, &f.f32s("kv")?, 2e-4, 0.03);
        let rkv = f.dev(&ops, "kv")?;
        let h = f.dev(&ops, "h")?;
        let qw = f.dev(&ops, "qw")?;
        let kw = f.dev(&ops, "kw")?;
        ck("engram_gate", unsafe {
            ffi::memra_v41_engram_gate(
                h.p(&st).m(),
                rkv.p(&st).m(),
                qw.p(&st).m(),
                kw.p(&st).m(),
                s as i32,
                4,
                5120,
                1e-20,
                (5120f32).powf(-0.5),
                h.p(&st).m(),
                sv,
            )
        })?;
        rep.close("engram/gate", &bf(&ops, &h)?, &f.f32s("y")?, 2e-4, 0.03);
    }
    schedules(&ops, rep)?;
    Ok(())
}

/// Deterministic test bytes for the schedule checks (no fixture: the reference is in this binary).
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn e4m3(&mut self, n: usize) -> Vec<u8> {
        (0..n)
            .map(|_| {
                let b = (self.next() & 0xFF) as u8;
                if b & 0x7F == 0x7F { b ^ 1 } else { b }
            })
            .collect()
    }
    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| (self.next() & 0xFF) as u8).collect()
    }
    fn range(&mut self, n: usize, lo: u8, hi: u8) -> Vec<u8> {
        (0..n)
            .map(|_| lo + (self.next() % (hi - lo + 1) as u32) as u8)
            .collect()
    }
    fn bf16(&mut self, n: usize) -> Vec<u8> {
        // bf16 values of magnitude 2^-8 .. 2^4 with random sign and mantissa
        let mut v = Vec::with_capacity(2 * n);
        for _ in 0..n {
            let r = self.next();
            let e = 119 + (r % 13);
            let b = ((r >> 8) & 0x8000) as u16 | ((e as u16) << 7) | ((r >> 16) as u16 & 0x7F);
            v.extend_from_slice(&b.to_le_bytes());
        }
        v
    }
}

/// The dispatched schedules against the reference schedule, byte for byte: the block-scaled GEMM
/// (small-row and pipelined, every weight format, scale codes across the e8m0 range), the
/// tile-list dispatch over a device plan, the device plan against the host `MoePlan`, and the
/// small-M bf16 and fp32-FMA kernels against their tile kernels on the same rows. Each check is a
/// byte comparison: these kernels change the schedule, never the program.
/// A fixture's block-quantized weight: codes `wn` and the checkpoint-form scales `sn` (FP8: one
/// per 32 x 32 block; MXFP4: one per row and 32 inputs) repacked into the GEMMs' scale layout, as
/// the loader holds them.
fn qw_fx(
    f: &Fx,
    ops: &Ops,
    wn: &str,
    sn: &str,
    n: usize,
    k: usize,
    fmt: QFmt,
) -> Result<QW, String> {
    let raw = f.bytes(sn)?;
    let kb = k / 32;
    let s = match fmt {
        QFmt::Fp8Block32 => v::sf_pack(n, k, |r, b| raw[r / 32 * kb + b]),
        _ => v::sf_pack(n, k, |r, b| raw[r * kb + b]),
    };
    Ok(QW {
        w: f.dev(ops, wn)?,
        s: Buf::from_bytes(&ops.st, &s)?,
        n,
        k,
        fmt,
    })
}

/// A grouped call's two outputs and each pair's scale row, read back.
type Grouped = (Vec<u8>, Vec<u8>, Vec<i32>);

fn schedules(ops: &Ops, rep: &mut Report) -> Result<(), String> {
    use memra_engine::dsv41_gpu::tp::MoePlan;
    let st = ops.st.clone();
    let sv = st.cu_stream() as *mut std::os::raw::c_void;
    let mut rng = Lcg(0xd541_5c4e);
    let same = |rep: &mut Report, name: &str, a: &[u8], b: &[u8]| {
        let eq = a.len() == b.len() && a == b;
        let diff = a.iter().zip(b).filter(|(x, y)| x != y).count();
        rep.raw(
            name,
            eq,
            "byte-identical to the reference schedule",
            format!("{} bytes, {diff} differ", a.len()),
        );
    };
    // the scale layout of `rows` rows from row-major codes `[rows][k / 32]`
    let sfp = |rows: usize, k: usize, raw: &[u8]| v::sf_pack(rows, k, |r, b| raw[r * (k / 32) + b]);
    // One dense CUTLASS call in a given form (1 normal, 2 swapped tileN 8, 3 swapped tileN 16).
    #[allow(clippy::too_many_arguments)]
    let dense = |a: Dp,
                 asf: Dp,
                 w: Dp,
                 ldw: usize,
                 wsf: Dp,
                 wf: i32,
                 c: Dp,
                 ldc: usize,
                 m: usize,
                 n: usize,
                 k: usize,
                 form: i32| {
        ck("cutlass_gemm", unsafe {
            ffi::memra_v41_cutlass_gemm(
                a.c(),
                k as i64,
                asf.c(),
                w.c(),
                ldw as i64,
                wsf.c(),
                wf,
                c.m(),
                ldc as i64,
                m as i32,
                n as i32,
                k as i32,
                form,
                sv,
            )
        })
    };
    // Every form of the dense GEMM against the normal one (rows on A, 128 x 128), byte for byte:
    // (m, n, k, weight format, activation scale code range)
    type GemmCase = (usize, usize, usize, i32, (u8, u8));
    let cases: [GemmCase; 22] = [
        // a last 128-column tile half full (a rank's share of an expert's hidden columns at 4 ranks: 576)
        (200, 576, 5120, 1, (118, 130)),
        (300, 576, 5120, 0, (118, 130)),
        (130, 1344, 2304, 1, (118, 130)),
        (6, 576, 5120, 1, (118, 130)),
        (1, 1280, 5120, 0, (118, 130)),
        (2, 1280, 5120, 0, (118, 130)),
        (3, 4096, 1280, 0, (118, 130)),
        (4, 2304, 5120, 1, (118, 130)),
        (5, 512, 5120, 0, (118, 130)),
        (8, 512, 8192, 0, (118, 130)),
        (12, 1280, 5120, 0, (118, 130)),
        (16, 2304, 5120, 1, (118, 130)),
        (24, 5120, 2304, 1, (118, 130)),
        (32, 512, 8192, 0, (118, 130)),
        (64, 2304, 5120, 1, (118, 130)),
        (77, 1280, 5120, 0, (118, 130)),
        (200, 5120, 2304, 1, (118, 130)),
        (300, 1280, 5120, 0, (118, 130)),
        (130, 2304, 5120, 2, (118, 130)),
        (2, 2304, 5120, 2, (118, 130)),
        // scale codes across the whole e8m0 range
        (96, 1280, 5120, 0, (0, 254)),
        (160, 2304, 5120, 1, (0, 254)),
    ];
    for (ci, &(m, n, k, wf, (slo, shi))) in cases.iter().enumerate() {
        let a = Buf::from_bytes(&st, &rng.e4m3(m * k))?;
        let asf = Buf::from_bytes(&st, &sfp(m, k, &rng.range(m * (k / 32), slo, shi)))?;
        let kb = k / 32;
        let (w, ws, ldw) = match wf {
            0 => {
                let blk = rng.range(n.div_ceil(32) * kb, 100, 127);
                (
                    rng.e4m3(n * k),
                    v::sf_pack(n, k, |r, b| blk[r / 32 * kb + b]),
                    k,
                )
            }
            // MXFP4, and minted NVFP4 (its ue8m0 codes on the device, as the loader holds them)
            _ => (
                rng.bytes(n * k / 2),
                sfp(n, k, &rng.range(n * kb, 112, 124)),
                k / 2,
            ),
        };
        let w = Buf::from_bytes(&st, &w)?;
        let ws = Buf::from_bytes(&st, &ws)?;
        let want = ops.buf(m * n * 2)?;
        dense(
            a.p(&st),
            asf.p(&st),
            w.p(&st),
            ldw,
            ws.p(&st),
            wf,
            want.p(&st),
            n,
            m,
            n,
            k,
            1,
        )?;
        let want = want.host(&st)?;
        let forms: &[i32] = if m <= 16 { &[0, 2, 3, 4] } else { &[0, 2, 3] };
        for &form in forms {
            let got = ops.buf(m * n * 2)?;
            dense(
                a.p(&st),
                asf.p(&st),
                w.p(&st),
                ldw,
                ws.p(&st),
                wf,
                got.p(&st),
                n,
                m,
                n,
                k,
                form,
            )?;
            same(
                rep,
                &format!("schedule/cutlass_gemm/{ci}_m{m}_n{n}_k{k}_wf{wf}_form{form}"),
                &got.host(&st)?,
                &want,
            );
        }
    }

    // The dense small-row body (form 4) and the by-rows choice (form 0) against every CUTLASS form at
    // the decode shapes in play, 1 to 16 rows: a rank's share at four ranks (wq_a, wkv, wq_a + wkv
    // fused, the indexer's wq_b, wq_b, wo_b, the shared expert's w1, w1 + w3 fused, w2) and one
    // card's (wq_b, wo_b, the shared w1, w1 + w3, w2), FP8, and two FP4 weights (MXFP4, minted).
    // (n, k, weight format); each shape's weights are drawn once and serve every row count.
    let decode: [(usize, usize, i32); 16] = [
        (1280, 5120, 0),
        (512, 5120, 0),
        (1792, 5120, 0),
        (4096, 1280, 0),
        (8192, 1280, 0),
        (1280, 8192, 0),
        (576, 5120, 0),
        (1152, 5120, 0),
        (1280, 2304, 0),
        (32768, 1280, 0),
        (5120, 8192, 0),
        (2304, 5120, 0),
        (4608, 5120, 0),
        (5120, 2304, 0),
        (2304, 5120, 1),
        (576, 5120, 2),
    ];
    for &(n, k, wf) in &decode {
        let kb = k / 32;
        let (w, ws, ldw) = if wf == 0 {
            let blk = rng.range(n.div_ceil(32) * kb, 100, 127);
            (
                rng.e4m3(n * k),
                v::sf_pack(n, k, |r, b| blk[r / 32 * kb + b]),
                k,
            )
        } else {
            (
                rng.bytes(n * k / 2),
                sfp(n, k, &rng.range(n * kb, 112, 124)),
                k / 2,
            )
        };
        let w = Buf::from_bytes(&st, &w)?;
        let ws = Buf::from_bytes(&st, &ws)?;
        for m in [1usize, 2, 3, 4, 5, 8, 9, 12, 16] {
            let a = Buf::from_bytes(&st, &rng.e4m3(m * k))?;
            let asf = Buf::from_bytes(&st, &sfp(m, k, &rng.range(m * kb, 118, 130)))?;
            let run = |form: i32| -> Result<Vec<u8>, String> {
                let c = ops.buf(m * n * 2)?;
                dense(
                    a.p(&st),
                    asf.p(&st),
                    w.p(&st),
                    ldw,
                    ws.p(&st),
                    wf,
                    c.p(&st),
                    n,
                    m,
                    n,
                    k,
                    form,
                )?;
                c.host(&st)
            };
            let want = run(1)?;
            for form in [0, 2, 3, 4] {
                same(
                    rep,
                    &format!("schedule/dense_small/m{m}_n{n}_k{k}_wf{wf}_form{form}"),
                    &run(form)?,
                    &want,
                );
            }
        }
    }

    // Weights concatenated by rows at load (`load_fp8_cat`: wq_a + wkv, the shared expert's w1 + w3 at
    // one, two and four ranks): one call over the concatenation against the two calls over the parts,
    // every element byte for byte, at decode and prefill widths, in every form a width takes.
    // (rows of the first part, of the second, k)
    for &(n1, n2, k) in &[
        (1280usize, 512usize, 5120usize),
        (576, 576, 5120),
        (1152, 1152, 5120),
        (2304, 2304, 5120),
    ] {
        let kb = k / 32;
        let n = n1 + n2;
        let (w, blk) = (
            rng.e4m3(n * k),
            rng.range(n1.div_ceil(32) * kb + n2.div_ceil(32) * kb, 100, 127),
        );
        // each part's checkpoint codes: one per 32 x 32 block of its own rows
        let code = |r: usize, b: usize| {
            if r < n1 {
                blk[r / 32 * kb + b]
            } else {
                blk[n1.div_ceil(32) * kb + (r - n1) / 32 * kb + b]
            }
        };
        let wc = Buf::from_bytes(&st, &w)?;
        let sc = Buf::from_bytes(&st, &v::sf_pack(n, k, code))?;
        let w1 = Buf::from_bytes(&st, &w[..n1 * k])?;
        let s1 = Buf::from_bytes(&st, &v::sf_pack(n1, k, code))?;
        let w2 = Buf::from_bytes(&st, &w[n1 * k..])?;
        let s2 = Buf::from_bytes(&st, &v::sf_pack(n2, k, |r, b| code(n1 + r, b)))?;
        for m in [1usize, 3, 8, 16, 64, 300] {
            let a = Buf::from_bytes(&st, &rng.e4m3(m * k))?;
            let asf = Buf::from_bytes(&st, &sfp(m, k, &rng.range(m * kb, 118, 130)))?;
            let forms: &[i32] = if m <= 16 { &[0, 1, 2, 3, 4] } else { &[0, 1] };
            for &form in forms {
                let run = |wd: &Buf, sd: &Buf, nn: usize| -> Result<Vec<u8>, String> {
                    let c = ops.buf(m * nn * 2)?;
                    dense(
                        a.p(&st),
                        asf.p(&st),
                        wd.p(&st),
                        k,
                        sd.p(&st),
                        0,
                        c.p(&st),
                        nn,
                        m,
                        nn,
                        k,
                        form,
                    )?;
                    c.host(&st)
                };
                let cat = run(&wc, &sc, n)?;
                let (p1, p2) = (run(&w1, &s1, n1)?, run(&w2, &s2, n2)?);
                let mut parts = Vec::with_capacity(cat.len());
                for r in 0..m {
                    parts.extend_from_slice(&p1[r * n1 * 2..(r + 1) * n1 * 2]);
                    parts.extend_from_slice(&p2[r * n2 * 2..(r + 1) * n2 * 2]);
                }
                same(
                    rep,
                    &format!("schedule/fused_rows/{n1}+{n2}_k{k}_m{m}_form{form}"),
                    &cat,
                    &parts,
                );
            }
        }
    }

    // The split MoE's small-row quantization: `np + rows` hidden rows into two scale layouts in one
    // launch (the pairs' rows a layout of np rows, the shared expert's rows their own at the next
    // 128-row block) == act_quant's row-major codes and scales placed through sf_place twice.
    for &(np, rows, k) in &[
        (6usize, 1usize, 2304usize),
        (24, 4, 2304),
        (48, 8, 2304),
        (128, 3, 2304),
    ] {
        let n = np + rows;
        let kb = k / 32;
        let mut xb = Vec::with_capacity(n * k * 2);
        for _ in 0..n * k {
            // bf16 near 1 with random signs and exponents a few steps apart
            let sign = if rng.next() & 1 == 1 { 0x8000u16 } else { 0 };
            let v = (0x3F00u16 + (rng.next() % 0x180) as u16) | sign;
            xb.extend_from_slice(&v.to_le_bytes());
        }
        let x = Buf::from_bytes(&st, &xb)?;
        let (q1, s1) = (ops.buf(n * k)?, ops.buf(n * kb)?);
        ck("act_quant", unsafe {
            ffi::memra_v41_act_quant(
                x.p(&st).c(),
                k as i64,
                std::ptr::null(),
                q1.p(&st).m(),
                k as i64,
                s1.p(&st).m(),
                kb as i64,
                std::ptr::null_mut(),
                0,
                n as i32,
                k as i32,
                sv,
            )
        })?;
        let s_np = v::sf_bytes(np, k);
        let lay = ops.buf(s_np + v::sf_bytes(rows, k))?;
        for (src, dst, r) in [
            (s1.p(&st), lay.p(&st), np),
            (s1.at(&st, np * kb), lay.at(&st, s_np), rows),
        ] {
            ck("sf_place", unsafe {
                ffi::memra_v41_sf_place(
                    src.c(),
                    kb as i64,
                    std::ptr::null(),
                    dst.m(),
                    r as i32,
                    k as i32,
                    sv,
                )
            })?;
        }
        let (q2, lay2) = (ops.buf(n * k)?, ops.buf(s_np + v::sf_bytes(rows, k))?);
        ck("act_quant_sf_split", unsafe {
            ffi::memra_v41_act_quant_sf_split(
                x.p(&st).c(),
                k as i64,
                q2.p(&st).m(),
                k as i64,
                lay2.p(&st).m(),
                n as i32,
                np as i32,
                k as i32,
                sv,
            )
        })?;
        same(
            rep,
            &format!("schedule/act_quant_sf_split/np{np}_rows{rows}_codes"),
            &q2.host(&st)?,
            &q1.host(&st)?,
        );
        same(
            rep,
            &format!("schedule/act_quant_sf_split/np{np}_rows{rows}_scales"),
            &lay2.host(&st)?,
            &lay.host(&st)?,
        );
    }

    // the device plan against the host plan, then the grouped GEMM in every form against the dense
    // normal form run per expert, on a 16-expert slice
    // (rows, experts, top-k, tile rows, output columns): 576 is a rank's share of the hidden columns at 4
    // ranks, whose last 128-column tile is half full
    for &(rows, ne, topk, bm, n) in &[
        (1usize, 16usize, 6usize, 1usize, 2304usize),
        (1, 16, 6, 4, 2304),
        (3, 16, 6, 4, 2304),
        (6, 16, 6, 8, 2304),
        (12, 16, 6, 8, 2304),
        (40, 16, 6, 64, 2304),
        (700, 16, 6, 64, 2304),
        (900, 16, 6, 128, 2304),
        (6, 16, 6, 8, 576),
        (700, 16, 6, 64, 576),
        (900, 16, 6, 128, 576),
    ] {
        let nt = if n == 2304 {
            String::new()
        } else {
            format!("_n{n}")
        };
        let mut sel = Vec::with_capacity(rows * topk);
        let mut selw = Vec::with_capacity(rows * topk);
        for _ in 0..rows {
            let mut picked: Vec<i32> = Vec::new();
            while picked.len() < topk {
                let e = (rng.next() % ne as u32) as i32;
                if !picked.contains(&e) {
                    picked.push(e);
                }
            }
            for e in picked {
                sel.push(e);
                selw.push((rng.next() % 1000) as f32 / 997.0);
            }
        }
        let host = MoePlan::new(&sel, &selw, rows, topk, ne)?;
        let np = rows * topk;
        let max_tiles = np.div_ceil(bm) + ne.min(np);
        let dsel = Buf::from_i32(&st, &sel)?;
        let dselw = Buf::from_f32(&st, &selw)?;
        let counts = ops.buf(rows.div_ceil(256) * ne * 4)?;
        let off = ops.buf((ne + 1) * 4)?;
        let ptok = ops.buf(np * 4)?;
        let pw = ops.buf(np * 4)?;
        let order = ops.buf(np * 4)?;
        let tiles = ops.buf(max_tiles * 16)?;
        let ntiles = ops.buf(4)?;
        let err = ops.buf(4)?;
        let flag = ops.buf(4)?;
        ck("moe_plan", unsafe {
            ffi::memra_v41_moe_plan(
                dsel.p(&st).i(),
                dselw.p(&st).f(),
                rows as i32,
                topk as i32,
                ne as i32,
                bm as i32,
                0,
                ne as i32,
                counts.p(&st).im(),
                off.p(&st).im(),
                ptok.p(&st).im(),
                pw.p(&st).fm(),
                order.p(&st).im(),
                tiles.p(&st).m(),
                ntiles.p(&st).im(),
                err.p(&st).im(),
                flag.p(&st).im(),
                sv,
            )
        })?;
        let hoff: Vec<i32> = host.off.iter().map(|&x| x as i32).collect();
        let ok = bytes_i32(&off.host(&st)?) == hoff
            && bytes_i32(&ptok.host(&st)?) == host.ptok
            && bytes_f32(&pw.host(&st)?) == host.pw
            && bytes_i32(&order.host(&st)?) == host.order
            && bytes_i32(&err.host(&st)?)[0] == 0
            && bytes_i32(&flag.host(&st)?)[0] == 0;
        rep.raw(
            &format!("schedule/moe_plan/rows{rows}_bm{bm}{nt}"),
            ok,
            "device plan == host MoePlan (offsets, pair tokens, weights, per-token order)",
            format!("{np} pairs, {} tiles", bytes_i32(&ntiles.host(&st)?)[0]),
        );
        if rows == 3 {
            // a selection naming no expert (the router's -1 for a NaN row): the plan skips the
            // pair, counts it and sets bit 0 of the rank's call flag
            let mut bad_sel = sel.clone();
            bad_sel[topk] = -1;
            let dbad = Buf::from_i32(&st, &bad_sel)?;
            // its own outputs: the tile-list check below reads the plan above
            let (off2, ptok2, pw2, order2) = (
                ops.buf((ne + 1) * 4)?,
                ops.buf(np * 4)?,
                ops.buf(np * 4)?,
                ops.buf(np * 4)?,
            );
            let (tiles2, ntiles2, err2, flag2) = (
                ops.buf(max_tiles * 16)?,
                ops.buf(4)?,
                ops.buf(4)?,
                ops.buf(4)?,
            );
            ck("moe_plan", unsafe {
                ffi::memra_v41_moe_plan(
                    dbad.p(&st).i(),
                    dselw.p(&st).f(),
                    rows as i32,
                    topk as i32,
                    ne as i32,
                    bm as i32,
                    0,
                    ne as i32,
                    counts.p(&st).im(),
                    off2.p(&st).im(),
                    ptok2.p(&st).im(),
                    pw2.p(&st).fm(),
                    order2.p(&st).im(),
                    tiles2.p(&st).m(),
                    ntiles2.p(&st).im(),
                    err2.p(&st).im(),
                    flag2.p(&st).im(),
                    sv,
                )
            })?;
            let ord = bytes_i32(&order2.host(&st)?);
            let (e2, f2) = (
                bytes_i32(&err2.host(&st)?)[0],
                bytes_i32(&flag2.host(&st)?)[0],
            );
            let skipped = ord[topk..2 * topk].iter().filter(|&&o| o < 0).count();
            let placed = bytes_i32(&off2.host(&st)?)[ne];
            rep.raw(
                "schedule/moe_plan/out_of_range_id",
                e2 == 1 && f2 == 1 && skipped == 1 && placed as usize == np - 1,
                "an id outside [0, ne): counted, call flag bit 0 set, its pair skipped (order -1, one pair fewer placed)",
                format!("err {e2}, flag {f2}, token 1 skips {skipped}, {placed} of {np} placed"),
            );
        }
        // w1-shaped MXFP4 experts (k 5120): the pairs' codes, row-major scale codes placed through
        // each pair's scale row
        let k = 5120usize;
        let kb = k / 32;
        let a = Buf::from_bytes(&st, &rng.e4m3(np * k))?;
        let a_rows = rng.range(np * kb, 118, 130);
        let a_rm = Buf::from_bytes(&st, &a_rows)?;
        let mut wts = Vec::with_capacity(ne);
        let mut tab = Vec::with_capacity(ne * 64);
        for _ in 0..ne {
            let w = Buf::from_bytes(&st, &rng.bytes(n * k / 2))?;
            let s = Buf::from_bytes(&st, &sfp(n, k, &rng.range(n * kb, 112, 124)))?;
            for _ in 0..3 {
                tab.extend_from_slice(&w.p(&st).addr().to_le_bytes());
            }
            for _ in 0..3 {
                tab.extend_from_slice(&s.p(&st).addr().to_le_bytes());
            }
            for _ in 0..3 {
                tab.extend_from_slice(&1.0f32.to_le_bytes());
            }
            tab.extend_from_slice(&0i32.to_le_bytes());
            wts.push((w, s));
        }
        let tabd = Buf::from_bytes(&st, &tab)?;
        // the reference: each expert's rows as one dense call in the normal form
        let want = ops.buf(np * n * 2)?;
        for (e, wt) in wts.iter().enumerate().take(ne) {
            let (p0, p1) = (host.off[e], host.off[e + 1]);
            if p1 == p0 {
                continue;
            }
            let esf = Buf::from_bytes(&st, &sfp(p1 - p0, k, &a_rows[p0 * kb..p1 * kb]))?;
            dense(
                a.at(&st, p0 * k),
                esf.p(&st),
                wt.0.p(&st),
                k / 2,
                wt.1.p(&st),
                1,
                want.at(&st, p0 * n * 2),
                n,
                p1 - p0,
                n,
                k,
                1,
            )?;
        }
        let want = want.host(&st)?;
        // one grouped call over local experts [e0, e0 + ng) of the plan, in `form`, into a fresh
        // buffer of 0xA5 bytes (rows of other experts must stay untouched)
        // `small`: the small-row body over the device plan's tile list (experts [0, ne), tiles of
        // `bm` rows) instead of CUTLASS's grouped kernel
        let grouped = |e0: usize,
                       ng: usize,
                       form: i32,
                       tab: &Buf,
                       projs: (i32, i32),
                       small: bool|
         -> Result<Grouped, String> {
            let sfrow = ops.buf(np * 4)?;
            ck("cutlass_sfrow", unsafe {
                ffi::memra_v41_cutlass_sfrow(
                    off.p(&st).i(),
                    e0 as i32,
                    ng as i32,
                    np as i32,
                    sfrow.p(&st).im(),
                    sv,
                )
            })?;
            let asf = ops.buf(v::sf_bytes(np + 127 * ng.min(np), k))?;
            ck("sf_place", unsafe {
                ffi::memra_v41_sf_place(
                    a_rm.p(&st).c(),
                    kb as i64,
                    sfrow.p(&st).i(),
                    asf.p(&st).m(),
                    np as i32,
                    k as i32,
                    sv,
                )
            })?;
            let fill = vec![0xA5u8; np * n * 2];
            let (d0, d1) = (Buf::from_bytes(&st, &fill)?, Buf::from_bytes(&st, &fill)?);
            let nproj = if projs.0 == projs.1 { 1 } else { 2 };
            if small {
                ck("moe_small", unsafe {
                    ffi::memra_v41_moe_small(
                        a.p(&st).c(),
                        k as i64,
                        asf.p(&st).c(),
                        sfrow.p(&st).i(),
                        tiles.p(&st).c(),
                        ntiles.p(&st).i(),
                        max_tiles as i32,
                        tab.p(&st).c(),
                        nproj as i32,
                        projs.0,
                        projs.1,
                        (k / 2) as i64,
                        d0.p(&st).m(),
                        d1.p(&st).m(),
                        n as i64,
                        n as i32,
                        k as i32,
                        bm as i32,
                        sv,
                    )
                })?;
                return Ok((d0.host(&st)?, d1.host(&st)?, bytes_i32(&sfrow.host(&st)?)));
            }
            let bytes = unsafe { ffi::memra_v41_cutlass_moe_bytes((ng * nproj) as i32) };
            let blk = ops.buf(bytes.max(1) as usize)?;
            ck("cutlass_moe", unsafe {
                ffi::memra_v41_cutlass_moe(
                    off.p(&st).i(),
                    e0 as i32,
                    ng as i32,
                    nproj as i32,
                    projs.0,
                    projs.1,
                    // a rank's table: record g is local expert g (expert e0 + g)
                    tab.at(&st, e0 * 64).c(),
                    a.p(&st).c(),
                    asf.p(&st).c(),
                    d0.p(&st).m(),
                    d1.p(&st).m(),
                    n as i64,
                    n as i32,
                    k as i32,
                    (k / 2) as i64,
                    rows as i32,
                    form,
                    blk.p(&st).m(),
                    bytes,
                    sv,
                )
            })?;
            Ok((d0.host(&st)?, d1.host(&st)?, bytes_i32(&sfrow.host(&st)?)))
        };
        // each pair's scale row: the padded prefix of the experts' rows
        let (mut prow, mut want_sfrow) = (0usize, vec![-1i32; np]);
        for e in 0..ne {
            for (i, p) in (host.off[e]..host.off[e + 1]).enumerate() {
                want_sfrow[p] = (prow + i) as i32;
            }
            prow += (host.off[e + 1] - host.off[e]).next_multiple_of(128);
        }
        // the grouped calls take the normal form only (by rows: 0, and 1)
        for form in [1, 0] {
            let (got, _, sfrow) = grouped(0, ne, form, &tabd, (0, 0), false)?;
            if form == 1 {
                rep.raw(
                    &format!("schedule/cutlass_sfrow/rows{rows}{nt}"),
                    sfrow == want_sfrow,
                    "each pair's scale row == the host's padded prefix",
                    format!("{np} pairs"),
                );
            }
            same(
                rep,
                &format!("schedule/cutlass_moe/rows{rows}_form{form}{nt}"),
                &got,
                &want,
            );
        }
        // the small-row body over the tile list (plans of tiles of at most 8 rows): the same bytes
        if bm <= 8 {
            let (got, _, _) = grouped(0, ne, 0, &tabd, (0, 0), true)?;
            same(
                rep,
                &format!("schedule/moe_small/rows{rows}_bm{bm}{nt}"),
                &got,
                &want,
            );
        }
        // a rank's range of experts [4, 12): its pairs' rows as above, every other row untouched
        {
            let (got, _, _) = grouped(4, 8, 0, &tabd, (0, 0), false)?;
            let (p0, p1) = (host.off[4] * n * 2, host.off[12] * n * 2);
            let outside = got[..p0].iter().chain(&got[p1..]).all(|&b| b == 0xA5);
            let inside = got[p0..p1] == want[p0..p1];
            rep.raw(
                &format!("schedule/cutlass_moe_range/rows{rows}{nt}"),
                inside && outside,
                "experts [4, 12) of 16: their pairs' rows == the full call's, other rows untouched",
                format!("inside {inside}, outside untouched {outside}"),
            );
        }
        // w1 and w3 in one launch (2 x 16 groups) == two launches; the table's third projection is
        // the next expert's weight, so the two outputs differ
        {
            let mut tab2 = Vec::with_capacity(ne * 64);
            for e in 0..ne {
                let (w1, w3) = (&wts[e], &wts[(e + 1) % ne]);
                for x in [&w1.0, &w1.0, &w3.0, &w1.1, &w1.1, &w3.1] {
                    tab2.extend_from_slice(&x.p(&st).addr().to_le_bytes());
                }
                for _ in 0..3 {
                    tab2.extend_from_slice(&1.0f32.to_le_bytes());
                }
                tab2.extend_from_slice(&0i32.to_le_bytes());
            }
            let tab2 = Buf::from_bytes(&st, &tab2)?;
            let (g2, u2, _) = grouped(0, ne, 0, &tab2, (0, 2), false)?;
            let (g1, _, _) = grouped(0, ne, 0, &tab2, (0, 0), false)?;
            let (u1, _, _) = grouped(0, ne, 0, &tab2, (2, 2), false)?;
            if bm <= 8 {
                // the small-row body's two projections in one launch == CUTLASS's
                let (gs, us, _) = grouped(0, ne, 0, &tab2, (0, 2), true)?;
                same(
                    rep,
                    &format!("schedule/moe_small_pair/rows{rows}_w1{nt}"),
                    &gs,
                    &g2,
                );
                same(
                    rep,
                    &format!("schedule/moe_small_pair/rows{rows}_w3{nt}"),
                    &us,
                    &u2,
                );
                // the split MoE's small-row path: the pairs' scales in a layout of their own (pair p
                // at scale row p, no scale-row kernel)
                let asf_np = Buf::from_bytes(&st, &sfp(np, k, &a_rows))?;
                let fill = vec![0xA5u8; np * n * 2];
                let (d0, d1) = (Buf::from_bytes(&st, &fill)?, Buf::from_bytes(&st, &fill)?);
                ck("moe_small", unsafe {
                    ffi::memra_v41_moe_small(
                        a.p(&st).c(),
                        k as i64,
                        asf_np.p(&st).c(),
                        std::ptr::null(),
                        tiles.p(&st).c(),
                        ntiles.p(&st).i(),
                        max_tiles as i32,
                        tab2.p(&st).c(),
                        2,
                        0,
                        2,
                        (k / 2) as i64,
                        d0.p(&st).m(),
                        d1.p(&st).m(),
                        n as i64,
                        n as i32,
                        k as i32,
                        bm as i32,
                        sv,
                    )
                })?;
                same(
                    rep,
                    &format!("schedule/moe_small_own_layout/rows{rows}_w1{nt}"),
                    &d0.host(&st)?,
                    &gs,
                );
                same(
                    rep,
                    &format!("schedule/moe_small_own_layout/rows{rows}_w3{nt}"),
                    &d1.host(&st)?,
                    &us,
                );
            }
            same(
                rep,
                &format!("schedule/cutlass_moe_pair/rows{rows}_w1{nt}"),
                &g2,
                &g1,
            );
            same(
                rep,
                &format!("schedule/cutlass_moe_pair/rows{rows}_w3{nt}"),
                &u2,
                &u1,
            );
        }
    }

    // small-M bf16 and fp32-FMA kernels: rows 0..m of an 8-row tile-kernel run
    for &(m, n, k, batch) in &[
        (1usize, 1024usize, 4096usize, 8usize),
        (3, 32, 5120, 1),
        (4, 512, 5120, 1),
        (1, 384, 5120, 1),
        (1, 5000, 1280, 1),
    ] {
        // 24 rows a plane: the first 8 are the small-row kernel's, all 24 the tile kernel's
        let a = Buf::from_bytes(&st, &rng.bf16(24 * k * batch))?;
        let b = Buf::from_bytes(&st, &rng.bf16(n * k * batch))?;
        for out_f32 in [0, 1] {
            let eb = if out_f32 == 1 { 4 } else { 2 };
            let small = ops.buf(m * n * eb * batch)?;
            let full = ops.buf(8 * n * eb * batch)?;
            let tile = ops.buf(24 * n * eb * batch)?;
            for (c, rows) in [(&small, m), (&full, 8), (&tile, 24)] {
                let ldc = n;
                let s_cz = rows * n;
                // 24 rows with the small-row limit at 0: the tile kernel
                if rows == 24 {
                    v::set_wide_rows(0, 0, 0, 0, 0);
                }
                ck("gemm_bf16", unsafe {
                    ffi::memra_v41_gemm_bf16(
                        a.p(&st).c(),
                        k as i64,
                        (24 * k) as i64,
                        b.p(&st).c(),
                        k as i64,
                        (n * k) as i64,
                        c.p(&st).m(),
                        ldc as i64,
                        s_cz as i64,
                        out_f32,
                        rows as i32,
                        n as i32,
                        k as i32,
                        batch as i32,
                        sv,
                    )
                })?;
            }
            v::set_wide_rows_default(1);
            let sh = small.host(&st)?;
            let fh = full.host(&st)?;
            let mut want = Vec::with_capacity(sh.len());
            for z in 0..batch {
                let base = z * 8 * n * eb;
                want.extend_from_slice(&fh[base..base + m * n * eb]);
            }
            same(
                rep,
                &format!("schedule/gemm_bf16/m{m}_n{n}_k{k}_b{batch}_f32{out_f32}"),
                &sh,
                &want,
            );
            // the two kernels cut K alike: the tile kernel's first 8 rows are the small-row
            // kernel's 8 rows (a bf16 output; an fp32 output is the FMA kernels' on both)
            if out_f32 == 0 {
                let th = tile.host(&st)?;
                let mut first = Vec::with_capacity(fh.len());
                for z in 0..batch {
                    let base = z * 24 * n * eb;
                    first.extend_from_slice(&th[base..base + 8 * n * eb]);
                }
                same(
                    rep,
                    &format!("schedule/gemm_bf16_tile/n{n}_k{k}_b{batch}"),
                    &fh,
                    &first,
                );
            }
        }
        if batch == 1 {
            // the fp32-FMA family is one chain an output over K in K order, so every form writes the same bytes
            // for a row: up to 8 rows the small-row kernels (the ring kernel on a narrow output), 24 the narrow
            // kernel with K in parts, 40 the groups of 8, 64 the tile kernel
            let a64 = Buf::from_bytes(&st, &rng.bf16(64 * k))?;
            let small = ops.buf(m * n * 4)?;
            let full = ops.buf(8 * n * 4)?;
            let mid = ops.buf(24 * n * 4)?;
            let tall = ops.buf(40 * n * 4)?;
            let tile = ops.buf(64 * n * 4)?;
            for (c, rows) in [
                (&small, m),
                (&full, 8),
                (&mid, 24),
                (&tall, 40),
                (&tile, 64),
            ] {
                ck("gemm_ffma", unsafe {
                    ffi::memra_v41_gemm_ffma(
                        a64.p(&st).c(),
                        k as i64,
                        b.p(&st).c(),
                        k as i64,
                        c.p(&st).fm(),
                        n as i64,
                        rows as i32,
                        n as i32,
                        k as i32,
                        sv,
                    )
                })?;
            }
            let fh = full.host(&st)?;
            same(
                rep,
                &format!("schedule/gemm_ffma/m{m}_n{n}_k{k}"),
                &small.host(&st)?,
                &fh[..m * n * 4],
            );
            same(
                rep,
                &format!("schedule/gemm_ffma_24/n{n}_k{k}"),
                &mid.host(&st)?[..8 * n * 4],
                &fh[..],
            );
            same(
                rep,
                &format!("schedule/gemm_ffma_40/n{n}_k{k}"),
                &tall.host(&st)?[..8 * n * 4],
                &fh[..],
            );
            same(
                rep,
                &format!("schedule/gemm_ffma_tile/n{n}_k{k}"),
                &tile.host(&st)?[..8 * n * 4],
                &fh[..],
            );
            // the compressor's pair: one call over two weights writes each output's single-call bytes
            let b1 = Buf::from_bytes(&st, &rng.bf16(n * k))?;
            for rows in [m, 8] {
                let (p0, p1) = (ops.buf(rows * n * 4)?, ops.buf(rows * n * 4)?);
                let one = ops.buf(rows * n * 4)?;
                ck("gemm_ffma2", unsafe {
                    ffi::memra_v41_gemm_ffma2(
                        a64.p(&st).c(),
                        k as i64,
                        b.p(&st).c(),
                        k as i64,
                        p0.p(&st).fm(),
                        n as i64,
                        n as i32,
                        b1.p(&st).c(),
                        k as i64,
                        p1.p(&st).fm(),
                        n as i64,
                        n as i32,
                        rows as i32,
                        k as i32,
                        sv,
                    )
                })?;
                ck("gemm_ffma", unsafe {
                    ffi::memra_v41_gemm_ffma(
                        a64.p(&st).c(),
                        k as i64,
                        b1.p(&st).c(),
                        k as i64,
                        one.p(&st).fm(),
                        n as i64,
                        rows as i32,
                        n as i32,
                        k as i32,
                        sv,
                    )
                })?;
                same(
                    rep,
                    &format!("schedule/gemm_ffma2/m{rows}_n{n}_k{k}/first"),
                    &p0.host(&st)?,
                    &fh[..rows * n * 4],
                );
                same(
                    rep,
                    &format!("schedule/gemm_ffma2/m{rows}_n{n}_k{k}/second"),
                    &p1.host(&st)?,
                    &one.host(&st)?,
                );
            }
        }
    }
    // the ring kernel on a narrow output whose width is not a multiple of its 4-column blocks (the
    // last block partial), against a form that is not the ring: a 24-row call takes the narrow kernel
    // with K in parts, so its first rows hold the same chains from another kernel. The pair's two
    // outputs against the same 24-row calls.
    {
        let (m, n, k) = (2usize, 30usize, 5120usize);
        let a24 = Buf::from_bytes(&st, &rng.bf16(24 * k))?;
        let b0 = Buf::from_bytes(&st, &rng.bf16(n * k))?;
        let b1 = Buf::from_bytes(&st, &rng.bf16(n * k))?;
        let ffma = |b: &Buf, c: &Buf, rows: usize| {
            ck("gemm_ffma", unsafe {
                ffi::memra_v41_gemm_ffma(
                    a24.p(&st).c(),
                    k as i64,
                    b.p(&st).c(),
                    k as i64,
                    c.p(&st).fm(),
                    n as i64,
                    rows as i32,
                    n as i32,
                    k as i32,
                    sv,
                )
            })
        };
        let (ring, mid0, mid1) = (
            ops.buf(m * n * 4)?,
            ops.buf(24 * n * 4)?,
            ops.buf(24 * n * 4)?,
        );
        ffma(&b0, &ring, m)?;
        ffma(&b0, &mid0, 24)?;
        ffma(&b1, &mid1, 24)?;
        let (h0, h1) = (mid0.host(&st)?, mid1.host(&st)?);
        same(
            rep,
            &format!("schedule/gemm_ffma_ring_partial/m{m}_n{n}_k{k}"),
            &ring.host(&st)?,
            &h0[..m * n * 4],
        );
        let (p0, p1) = (ops.buf(m * n * 4)?, ops.buf(m * n * 4)?);
        ck("gemm_ffma2", unsafe {
            ffi::memra_v41_gemm_ffma2(
                a24.p(&st).c(),
                k as i64,
                b0.p(&st).c(),
                k as i64,
                p0.p(&st).fm(),
                n as i64,
                n as i32,
                b1.p(&st).c(),
                k as i64,
                p1.p(&st).fm(),
                n as i64,
                n as i32,
                m as i32,
                k as i32,
                sv,
            )
        })?;
        same(
            rep,
            &format!("schedule/gemm_ffma2_partial/m{m}_n{n}_k{k}/first"),
            &p0.host(&st)?,
            &h0[..m * n * 4],
        );
        same(
            rep,
            &format!("schedule/gemm_ffma2_partial/m{m}_n{n}_k{k}/second"),
            &p1.host(&st)?,
            &h1[..m * n * 4],
        );
    }
    // the embedding gather on the device against the host's rows (`Runner::embed` and its hc
    // copies): a table of 1,000 random bf16 rows of 5,120, ids with both ends and repeats, into 4
    // stream slots, and the one-hot pre-mix
    {
        let (vocab, d, hc, s) = (1000usize, 5120usize, 4usize, 37usize);
        let table = rng.bf16(vocab * d);
        let ids: Vec<i32> = (0..s)
            .map(|i| match i {
                0 | 5 => 0,
                1 | 6 => vocab as i32 - 1,
                _ => (rng.next() as usize % vocab) as i32,
            })
            .collect();
        let tb = Buf::from_bytes(&st, &table)?;
        let ib = Buf::from_i32(&st, &ids)?;
        let (h, pm) = (ops.buf(s * hc * d * 2)?, ops.buf(s * hc * 4)?);
        ck("embed rows", unsafe {
            ffi::memra_v41_embed_rows(
                tb.p(&st).c(),
                vocab as i64,
                ib.p(&st).i(),
                s as i32,
                d as i32,
                hc as i32,
                h.p(&st).m(),
                pm.p(&st).fm(),
                sv,
            )
        })?;
        let (mut wh, mut wp) = (vec![0u8; s * hc * d * 2], vec![0u8; s * hc * 4]);
        for (t, &id) in ids.iter().enumerate() {
            let row = &table[id as usize * d * 2..(id as usize + 1) * d * 2];
            for k in 0..hc {
                wh[(t * hc + k) * d * 2..(t * hc + k + 1) * d * 2].copy_from_slice(row);
            }
            wp[t * hc * 4..t * hc * 4 + 4].copy_from_slice(&1f32.to_le_bytes());
        }
        same(rep, "schedule/embed_rows/stream", &h.host(&st)?, &wh);
        same(rep, "schedule/embed_rows/premix", &pm.host(&st)?, &wp);
    }
    // the one-row selects over a long row cut into blocks against the one-block forms: a row of few
    // distinct scores (so the threshold has many ties) with some -inf
    for &t in &[20_000usize, 131_072, 262_144] {
        let mut row = Vec::with_capacity(t * 2);
        for _ in 0..t {
            let r = rng.next();
            let bits: u16 = if r.is_multiple_of(23) {
                0xFF80
            } else {
                v::f32_to_bf16_bits((r % 37) as f32 * 0.25 - 3.0)
            };
            row.extend_from_slice(&bits.to_le_bytes());
        }
        let sc = Buf::from_bytes(&st, &row)?;
        let (k, offset, cb, cbk) = (512usize, 128usize, 8usize, 2048usize);
        let nb = t.div_ceil(cb);
        let (o1, o2) = (ops.buf(k * 4)?, ops.buf(k * 4)?);
        let (k1, k2) = (ops.buf(nb)?, ops.buf(nb)?);
        ck("topk", unsafe {
            ffi::memra_v41_topk(
                sc.p(&st).m(),
                t as i64,
                t as i32,
                k as i32,
                k as i32,
                offset as i32,
                o1.p(&st).im(),
                k as i64,
                1,
                sv,
            )
        })?;
        ops.topk_long(sc.p(&st), t, k, k, offset, o2.p(&st))?;
        same(
            rep,
            &format!("schedule/select_long/topk_t{t}"),
            &o2.host(&st)?,
            &o1.host(&st)?,
        );
        ck("candidates", unsafe {
            ffi::memra_v41_candidates(
                sc.p(&st).m(),
                t as i64,
                t as i32,
                cb as i32,
                cbk as i32,
                (t - 1) as i32,
                1,
                k1.p(&st).u8m(),
                nb as i64,
                1,
                sv,
            )
        })?;
        ops.candidates_long(sc.p(&st), t, cb, cbk, t - 1, 1, k2.p(&st))?;
        same(
            rep,
            &format!("schedule/select_long/candidates_t{t}"),
            &k2.host(&st)?,
            &k1.host(&st)?,
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn moe_only(
    ops: &Ops,
    x: &Buf,
    s: usize,
    gw: &v::BW,
    gb: &Buf,
    experts: &[[QW; 3]],
    sh: &[QW; 3],
    tk: usize,
    rs: f32,
    lim: f32,
) -> Result<Vec<f32>, String> {
    // the same orchestration as Ctx::moe, on a small expert count
    let st = ops.st.clone();
    let sv = st.cu_stream() as *mut std::os::raw::c_void;
    let ne = experts.len();
    let raw = ops.buf(s * ne * 4)?;
    ops.linear_bf16(x.p(&st), 5120, s, gw, raw.p(&st), ne, true)?;
    let sel = ops.buf(s * tk * 4)?;
    let selw = ops.buf(s * tk * 4)?;
    ck("route", unsafe {
        ffi::memra_v41_route(
            raw.p(&st).f(),
            ne as i64,
            gb.p(&st).f(),
            ne as i32,
            tk as i32,
            1,
            rs,
            sel.p(&st).im(),
            selw.p(&st).fm(),
            s as i32,
            sv,
        )
    })?;
    let selh = bytes_i32(&sel.host(&st)?);
    let selwh = bytes_f32(&selw.host(&st)?);
    let y = ops.buf(s * 5120 * 4)?;
    let mut scr = v::Scratch::default();
    for (e, w) in experts.iter().enumerate() {
        let mut ri = Vec::new();
        let mut rw = Vec::new();
        for t in 0..s {
            for j in 0..tk {
                if selh[t * tk + j] as usize == e {
                    ri.push(t as i32);
                    rw.push(selwh[t * tk + j]);
                }
            }
        }
        if ri.is_empty() {
            continue;
        }
        let n = ri.len();
        let dri = Buf::from_i32(&st, &ri)?;
        let drw = Buf::from_f32(&st, &rw)?;
        let xg = ops.buf(n * 5120 * 2)?;
        ck("gather_rows", unsafe {
            ffi::memra_v41_gather_rows(
                x.p(&st).m(),
                5120,
                dri.p(&st).i(),
                xg.p(&st).m(),
                5120,
                n as i32,
                5120,
                sv,
            )
        })?;
        let g = ops.buf(n * 2304 * 2)?;
        let u = ops.buf(n * 2304 * 2)?;
        let h = ops.buf(n * 2304 * 2)?;
        let d = ops.buf(n * 5120 * 2)?;
        ops.linear_q(xg.p(&st), 5120, n, &w[0], g.p(&st), 2304, false, &mut scr)?;
        ops.linear_q(xg.p(&st), 5120, n, &w[2], u.p(&st), 2304, false, &mut scr)?;
        ck("swiglu", unsafe {
            ffi::memra_v41_swiglu(
                g.p(&st).m(),
                u.p(&st).m(),
                2304,
                drw.p(&st).f(),
                lim,
                h.p(&st).m(),
                2304,
                n as i32,
                2304,
                sv,
            )
        })?;
        ops.linear_q(h.p(&st), 2304, n, &w[1], d.p(&st), 5120, false, &mut scr)?;
        ck("scatter_add", unsafe {
            ffi::memra_v41_scatter_add(
                d.p(&st).m(),
                5120,
                dri.p(&st).i(),
                y.p(&st).fm(),
                5120,
                n as i32,
                5120,
                sv,
            )
        })?;
        ops.sync()?;
    }
    let g = ops.buf(s * 2304 * 2)?;
    let u = ops.buf(s * 2304 * 2)?;
    let h = ops.buf(s * 2304 * 2)?;
    let d = ops.buf(s * 5120 * 2)?;
    ops.linear_q(x.p(&st), 5120, s, &sh[0], g.p(&st), 2304, false, &mut scr)?;
    ops.linear_q(x.p(&st), 5120, s, &sh[2], u.p(&st), 2304, false, &mut scr)?;
    ck("swiglu", unsafe {
        ffi::memra_v41_swiglu(
            g.p(&st).m(),
            u.p(&st).m(),
            2304,
            std::ptr::null(),
            lim,
            h.p(&st).m(),
            2304,
            s as i32,
            2304,
            sv,
        )
    })?;
    ops.linear_q(h.p(&st), 2304, s, &sh[1], d.p(&st), 5120, false, &mut scr)?;
    let out = ops.buf(s * 5120 * 2)?;
    ck("add_cast", unsafe {
        ffi::memra_v41_add_cast(
            y.p(&st).f(),
            d.p(&st).m(),
            out.p(&st).m(),
            (s * 5120) as i64,
            sv,
        )
    })?;
    bf(ops, &out)
}

fn sorted_dirs(root: &Path, prefix: &str) -> Result<Vec<PathBuf>, String> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(root)
        .map_err(|e| format!("{}: {e}", root.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.file_name()
                .is_some_and(|n| n.to_string_lossy().starts_with(prefix))
        })
        .collect();
    v.sort();
    Ok(v)
}

// ------------------------------------------------------------------------------------------------
// layer gates on the real checkpoint

/// Distribution metrics of one layer's stream against the reference: rel L2, max abs relative to
/// the stream's own max, per-token worst relative error, cosine.
fn stream_metrics(got: &[f32], want: &[f32], s: usize) -> (f64, f64, f64, f64) {
    let (mut se, mut sr, mut dot, mut sg) = (0f64, 0f64, 0f64, 0f64);
    let mut maxd = 0f64;
    let mut maxr = 0f64;
    for (&a, &b) in got.iter().zip(want) {
        let d = a as f64 - b as f64;
        se += d * d;
        sr += (b as f64).powi(2);
        sg += (a as f64).powi(2);
        dot += a as f64 * b as f64;
        maxd = maxd.max(d.abs());
        maxr = maxr.max((b as f64).abs());
    }
    let per = want.len() / s.max(1);
    let mut worst_tok = 0f64;
    for t in 0..s {
        let (mut e, mut r) = (0f64, 0f64);
        for i in t * per..(t + 1) * per {
            e += (got[i] as f64 - want[i] as f64).powi(2);
            r += (want[i] as f64).powi(2);
        }
        if r > 0.0 {
            worst_tok = worst_tok.max((e / r).sqrt());
        }
    }
    (
        (se / sr.max(1e-300)).sqrt(),
        maxd / maxr.max(1e-300),
        worst_tok,
        dot / (sg.sqrt() * sr.sqrt()).max(1e-300),
    )
}

/// Optional dump of the rewrite's own tensors under the capture's key names (`--save DIR`), in the
/// fixture format, for offline comparison with captures and CPU floors.
#[derive(Default)]
struct Saver {
    dir: Option<PathBuf>,
    idx: serde_json::Map<String, serde_json::Value>,
}

impl Saver {
    fn put(&mut self, key: &str, dtype: &str, shape: &[usize], bytes: &[u8]) -> Result<(), String> {
        let Some(d) = &self.dir else {
            return Ok(());
        };
        let file = format!("{}.bin", key.replace('/', "__"));
        std::fs::write(d.join(&file), bytes).map_err(|e| format!("save {key}: {e}"))?;
        self.idx.insert(
            key.to_string(),
            serde_json::json!({"dtype": dtype, "shape": shape, "file": file}),
        );
        Ok(())
    }
    fn close(&self) -> Result<(), String> {
        let Some(d) = &self.dir else {
            return Ok(());
        };
        let v = serde_json::json!({"tensors": self.idx, "meta": {}});
        std::fs::write(
            d.join("index.json"),
            serde_json::to_string_pretty(&v).unwrap(),
        )
        .map_err(|e| e.to_string())
    }
}

/// Pass rule of a layer-level comparison. With a floor (the same vendor layer rerun on the CPU
/// from the same captured inputs), the GPU rewrite may sit at most twice as far from the GPU
/// vendor output as the CPU vendor does, plus 1e-4 for the near-exact ops; without one, the
/// fixed `tol` band applies.
fn band(r: f64, floor: Option<f64>, tol: f64) -> (bool, String) {
    match floor {
        Some(fr) => (
            r <= 2.0 * fr + 1e-4,
            format!("rel_l2 <= 2 x floor ({fr:.3e}) + 1e-4"),
        ),
        None => (r <= tol, format!("rel_l2 <= {tol:e} (no floor)")),
    }
}

/// The 99th percentile over prefill tokens of the CPU floor's per-token relative error against the
/// capture, for `key` (an `s0/...` stream `[1][s][...]`). A decode token is one token, so its
/// band is where the floor's own tokens sit, not a whole-sequence rel L2.
fn floor_tok_p99(fl: Option<&Fx>, f: &Fx, key: &str, s: usize) -> Result<Option<f64>, String> {
    let Some(x) = fl else {
        return Ok(None);
    };
    if !x.has(key) || !f.has(key) {
        return Ok(None);
    }
    let a = x.f32s(key)?;
    let b = f.f32s(key)?;
    let per = b.len() / s.max(1);
    let mut v: Vec<f64> = (0..s)
        .map(|t| {
            let (mut e, mut r) = (0f64, 0f64);
            for i in t * per..(t + 1) * per {
                e += (a[i] as f64 - b[i] as f64).powi(2);
                r += (b[i] as f64).powi(2);
            }
            if r > 0.0 { (e / r).sqrt() } else { 0.0 }
        })
        .collect();
    v.sort_by(|p, q| p.partial_cmp(q).unwrap());
    Ok(v.get(((s as f64) * 0.99) as usize).copied())
}

/// Share of tokens whose top-k expert set matches between two routings `[s][k]`.
fn route_agree(a: &[i64], b: &[i64], s: usize, k: usize) -> usize {
    (0..s)
        .filter(|&t| {
            let mut x = a[t * k..(t + 1) * k].to_vec();
            let mut y = b[t * k..(t + 1) * k].to_vec();
            x.sort();
            y.sort();
            x == y
        })
        .count()
}

/// The CSA2 state a layer reads from earlier layers at one step, from the capture: the group's
/// compressed KV (Reuse and Reindex layers), the source's index keys (Reindex), the last selection
/// (Reuse) and the candidate pool (rescoring layers). Sources make their own.
fn load_shared(
    f: &Fx,
    ps: &str,
    l: usize,
    g: &Geo,
    rows: usize,
    sh: &mut Shared,
    ops: &Ops,
) -> Result<(), String> {
    let ratio = g.compress_ratios[l];
    if ratio == 0 {
        return Ok(());
    }
    let kv_src = g.kv_sources.contains(&l);
    let idx_src = g.index_sources.contains(&l);
    let k = |n: &str| format!("{ps}/in/{n}");
    if !kv_src && f.has(&k("compress_kv")) {
        let n = f.shape(&k("compress_kv"))[1];
        sh.compress_kv = Some(f.dev(ops, &k("compress_kv"))?);
        sh.n_c = n;
        sh.compress_cap = n;
    }
    if !kv_src && idx_src && f.has(&k("index_k")) {
        sh.index_k = Some(f.dev(ops, &k("index_k"))?);
        sh.index_k_cap = f.shape(&k("index_k"))[1];
    }
    if !idx_src && f.has(&k("topk_idxs")) {
        sh.topk = Some(f.dev(ops, &k("topk_idxs"))?);
        sh.topk_w = f.shape(&k("topk_idxs"))[2];
    }
    // the hook sees the shared slot at every layer entry; only a rescoring layer reads it
    if l > g.cand_source
        && f.has(&k("candidates_blocks"))
        && f.shape(&k("candidates_blocks"))[1] == rows
    {
        let nb = f.shape(&k("candidates_blocks"))[2];
        sh.cand = Some(f.dev(ops, &k("candidates_blocks"))?);
        sh.cand_ld = nb;
        sh.cand_row0 = 0;
    } else if l > g.cand_source && f.has(&k("candidates")) && f.shape(&k("candidates"))[1] == rows {
        let c = f.i64s(&k("candidates"))?; // [1, rows, n_c] bool per position -> per block
        let n_c = f.shape(&k("candidates"))[2];
        let nb = n_c.div_ceil(g.cand_block);
        let mut kb = vec![0u8; rows * nb];
        for i in 0..rows {
            for b in 0..nb {
                kb[i * nb + b] = c[i * n_c + b * g.cand_block] as u8;
            }
        }
        sh.cand = Some(Buf::from_bytes(&ops.st, &kb)?);
        sh.cand_ld = nb;
        sh.cand_row0 = 0;
    }
    Ok(())
}

/// Engram hash ids `[tokens][cols]` of layer `li` for `ids`: the native host hash when the vendor
/// constants are given (checked against the capture's ids at `want_key`), else the capture's.
fn engram_hashes(
    f: &Fx,
    token_map: Option<&Path>,
    ids: &[u32],
    li: usize,
    want_key: &str,
    rep: &mut Report,
) -> Result<Vec<i64>, String> {
    let Some(tmap) = token_map else {
        return f.i64s(want_key);
    };
    let g = Geo::from_config(
        &memra_gguf::config::ModelConfig::from_config_json(&f.ckpt.join("config.json"))
            .map_err(|e| e.to_string())?,
    )?;
    let ec = v::engram::EngramConsts::derive(&g, v::engram::token_map_from_file(tmap)?)?;
    let all = ec.hash_prefill(ids)?;
    let li_e = ec
        .layer_ids
        .iter()
        .position(|&x| x == li)
        .ok_or("engram layer not in constants")?;
    let (nl, cols) = (ec.layer_ids.len(), ec.cols());
    let mine: Vec<i64> = (0..ids.len())
        .flat_map(|t| all[(t * nl + li_e) * cols..(t * nl + li_e + 1) * cols].to_vec())
        .collect();
    if f.has(want_key) {
        let want = f.i64s(want_key)?;
        // the capture holds the ids of its own step: the prompt at prefill, one token per decode step
        let tail = &mine[mine.len() - want.len()..];
        let same = tail.iter().zip(&want).filter(|(a, b)| a == b).count();
        rep.raw(
            &format!(
                "layer{li}/engram_hash{}",
                if want.len() < mine.len() {
                    "_decode"
                } else {
                    ""
                }
            ),
            same == want.len(),
            "identical hash ids to the vendor NgramHashState",
            format!("{same}/{}", want.len()),
        );
        return Ok(tail.to_vec());
    }
    Ok(mine)
}

/// Decode steps of one layer after its prefill: each step's stream and pre-mix (and the CSA2 state
/// of the sources it reads) from the capture, this layer's own ring, compressor pair and caches
/// carried from the prefill, one token at position prompt + step - 1.
#[allow(clippy::too_many_arguments)]
fn decode_steps(
    cx: &mut Ctx,
    stm: &memra_gguf::safetensors::StModel,
    f: &Fx,
    lw: &LayerW,
    sh: &mut Shared,
    ls: &mut LayerState,
    token_map: Option<&Path>,
    ids: &[u32],
    decode: usize,
    tol: f64,
    fl: Option<&Fx>,
    saver: &mut Saver,
    rep: &mut Report,
) -> Result<(), String> {
    let li = lw.id;
    let g = cx.g.clone();
    let gen_ids: Vec<u32> = f.i64s("gen_ids")?.into_iter().map(|x| x as u32).collect();
    for step in 1..=decode.min(gen_ids.len()) {
        let ps = format!("s{step}/L{li}");
        if !f.has(&format!("{ps}/out/x")) {
            break;
        }
        let pos = ids.len() + step - 1;
        load_shared(f, &ps, li, &g, 1, sh, &cx.ops)?;
        let mut tr = Trace::default();
        let h = if lw.engram.is_some() {
            let mut ext = ids.to_vec();
            ext.extend_from_slice(&gen_ids[..step]);
            let hk = format!("{ps}/engram/hash");
            let all = engram_hashes(f, token_map, &ext, li, &hk, rep)?;
            let cols = g.engram_cols();
            let last = all[all.len() - cols..].to_vec();
            let h = f.dev(&cx.ops, &format!("{ps}/engram/x"))?;
            cx.engram(stm, lw, &h, &last, 1, &mut tr)?;
            h
        } else {
            f.dev(&cx.ops, &format!("{ps}/in/x"))?
        };
        let pre = f.dev(&cx.ops, &format!("{ps}/in/pre_mix"))?;
        let (h2, p2) = cx.block_forward(lw, &h, &pre, pos, 1, sh, ls, &mut tr)?;
        for (nm, got, key) in [
            ("attn_out", tr.attn_out.as_ref(), format!("{ps}/attn/out")),
            ("moe_out", tr.moe_out.as_ref(), format!("{ps}/moe/out")),
            ("out_stream", Some(&h2), format!("{ps}/out/x")),
        ] {
            if let (Some(b), true) = (got, f.has(&key)) {
                saver.put(&key, "bf16", &f.shape(&key), &b.host(&cx.ops.st)?)?;
                let (r, m, _, c) = stream_metrics(&bf(&cx.ops, b)?, &f.f32s(&key)?, 1);
                let k0 = key.replacen(&format!("s{step}/"), "s0/", 1);
                let (pass, rule) = match floor_tok_p99(fl, f, &k0, ids.len())? {
                    Some(p99) => (
                        r <= p99.max(1e-4),
                        format!("rel_l2 <= the floor's prefill per-token p99 ({p99:.3e})"),
                    ),
                    None => (r <= tol, format!("rel_l2 <= {tol:e} (no floor)")),
                };
                rep.raw(
                    &format!("layer{li}/decode{step}/{nm}"),
                    pass,
                    &rule,
                    format!("rel_l2={r:.3e} maxabs/max={m:.3e} cos={c:.8} pos {pos}"),
                );
            }
        }
        let pm = stat(
            &f32h(&cx.ops, &p2)?,
            &f.f32s(&format!("{ps}/out/pre_mix"))?,
            false,
        );
        let (ppass, prule) =
            match floor_tok_p99(fl, f, &format!("s0/L{li}/out/pre_mix"), ids.len())? {
                Some(p99) => (
                    pm.rel_l2 <= p99.max(1e-4),
                    format!("rel_l2 <= the floor's prefill per-token p99 ({p99:.3e})"),
                ),
                None => (pm.rel_l2 <= tol, format!("rel_l2 <= {tol:e} (no floor)")),
            };
        rep.raw(
            &format!("layer{li}/decode{step}/out_pre_mix"),
            ppass,
            &prule,
            format!("rel_l2={:.3e}", pm.rel_l2),
        );
        if !tr.route_sel.is_empty() && f.has(&format!("{ps}/gate/out1")) {
            let want = f.i64s(&format!("{ps}/gate/out1"))?;
            let mine: Vec<i64> = tr.route_sel.iter().map(|&x| x as i64).collect();
            let agree = route_agree(&mine, &want, 1, g.topk);
            rep.raw(
                &format!("layer{li}/decode{step}/route"),
                true,
                "reported (one token)",
                format!("{agree}/1 token sets equal"),
            );
        }
        if let (Some(ib), true) = (tr.idx_all.as_ref(), f.has(&format!("{ps}/sattn/idx"))) {
            let mut a: Vec<i32> = bytes_i32(&ib.host(&cx.ops.st)?)
                .into_iter()
                .filter(|&x| x >= 0)
                .collect();
            let mut b: Vec<i32> = f
                .i64s(&format!("{ps}/sattn/idx"))?
                .into_iter()
                .map(|x| x as i32)
                .filter(|&x| x >= 0)
                .collect();
            a.sort();
            b.sort();
            let inter = a.iter().filter(|x| b.binary_search(x).is_ok()).count();
            let jac = inter as f64 / (a.len() + b.len() - inter).max(1) as f64;
            rep.raw(
                &format!("layer{li}/decode{step}/attn_idx"),
                jac >= 0.99,
                "Jaccard >= 0.99",
                format!("Jaccard {jac:.5}, {} vs {} slots", a.len(), b.len()),
            );
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn layer_gate(
    ckpt: &Path,
    cap: &Path,
    layers: &[usize],
    mint: bool,
    chain: usize,
    token_map: Option<&Path>,
    tol: f64,
    floor: Option<&Path>,
    decode: usize,
    save: Option<&Path>,
    head: bool,
    rep: &mut Report,
) -> Result<String, String> {
    if let Some(d) = save {
        std::fs::create_dir_all(d).map_err(|e| e.to_string())?;
    }
    let saver = std::cell::RefCell::new(Saver {
        dir: save.map(Path::to_path_buf),
        ..Default::default()
    });
    let fl = match floor {
        Some(d) => Some(Fx::open(d)?),
        None => None,
    };
    let floor_rel = |key: &str, want: &[f32], s: usize| -> Result<Option<f64>, String> {
        match &fl {
            Some(x) if x.has(key) => Ok(Some(stream_metrics(&x.f32s(key)?, want, s).0)),
            _ => Ok(None),
        }
    };
    let e = memra_engine::Engine::new(0).map_err(|e| e.to_string())?;
    let st: Arc<cudarc::driver::CudaStream> = e.stream();
    let mc = memra_gguf::config::ModelConfig::from_config_json(&ckpt.join("config.json"))
        .map_err(|e| e.to_string())?;
    let g = Geo::from_config(&mc)?;
    let stm = memra_gguf::safetensors::StModel::open(ckpt).map_err(|e| e.to_string())?;
    let mut f = Fx::open(cap)?;
    f.ckpt = ckpt.to_path_buf();
    let ids: Vec<u32> = f
        .i64s("prompt_ids")?
        .into_iter()
        .map(|x| x as u32)
        .collect();
    let s = ids.len();
    let mut cx = Ctx::new(st.clone(), g.clone(), (s + 64).max(1024))?;
    cx.trace = true;
    if let Some(c) = std::env::args()
        .skip_while(|x| x != "--chunk")
        .nth(1)
        .and_then(|x| x.parse().ok())
    {
        // slicing never changes a result (every sliced step is per row): `--chunk` proves it
        cx.chunk = c;
    }
    let mut summary = Vec::new();
    for &l0 in layers {
        let mut sh = Shared::default();
        let pfx = format!("s0/L{l0}");
        load_shared(&f, &pfx, l0, &g, s, &mut sh, &cx.ops)?;
        // an Engram layer starts from the stream before its lookup (the previous layer's output)
        let h_key = if g.engram_layers.contains(&l0) && f.has(&format!("{pfx}/engram/x")) {
            format!("{pfx}/engram/x")
        } else {
            format!("{pfx}/in/x")
        };
        let mut h = f.dev(&cx.ops, &h_key)?;
        let mut pre = f.dev(&cx.ops, &format!("{pfx}/in/pre_mix"))?;
        let mut last = l0;
        for li in l0..(l0 + chain.max(1)).min(g.compress_ratios.len() - 3) {
            let p = format!("s0/L{li}");
            if !f.has(&format!("{p}/out/x")) {
                break;
            }
            let t0 = std::time::Instant::now();
            let lw = LayerW::load(&cx.ops, &stm, &g, li, mint)?;
            let load_s = t0.elapsed().as_secs_f64();
            let mut tr = Trace::default();
            if lw.engram.is_some() {
                let hashes =
                    engram_hashes(&f, token_map, &ids, li, &format!("{p}/engram/hash"), rep)?;
                cx.engram(&stm, &lw, &h, &hashes, s, &mut tr)?;
                if f.has(&format!("{p}/engram/out")) {
                    let key = format!("{p}/engram/out");
                    let want = f.f32s(&key)?;
                    let (r, m, w, c) = stream_metrics(&bf(&cx.ops, &h)?, &want, s);
                    let (pass, rule) = if li > l0 {
                        (true, "reported (chained)".to_string())
                    } else {
                        band(r, floor_rel(&key, &want, s)?, tol)
                    };
                    rep.raw(
                        &format!(
                            "layer{li}/engram_out{}",
                            if li > l0 { "_chained" } else { "" }
                        ),
                        pass,
                        &rule,
                        format!("rel_l2={r:.3e} maxabs/max={m:.3e} worst_tok={w:.3e} cos={c:.8}"),
                    );
                }
            }
            let t1 = std::time::Instant::now();
            let mut ls = LayerState::default();
            let (h2, pre2) = cx.block_forward(&lw, &h, &pre, 0, s, &mut sh, &mut ls, &mut tr)?;
            let fwd_s = t1.elapsed().as_secs_f64();
            // per-op checks against the capture where it has them
            let cmp =
                |rep: &mut Report, nm: &str, got: Option<&Buf>, key: &str| -> Result<(), String> {
                    if let (Some(b), true) = (got, f.has(key)) {
                        let want = f.f32s(key)?;
                        saver
                            .borrow_mut()
                            .put(key, "bf16", &f.shape(key), &b.host(&cx.ops.st)?)?;
                        let (r, m, w, c) = stream_metrics(&bf(&cx.ops, b)?, &want, s);
                        let fr = floor_rel(key, &want, s)?;
                        let (pass, rule) = band(r, fr, tol);
                        rep.raw(
                            &format!("layer{li}/{nm}"),
                            pass,
                            &rule,
                            format!(
                                "rel_l2={r:.3e} maxabs/max={m:.3e} worst_tok={w:.3e} cos={c:.8}"
                            ),
                        );
                    }
                    Ok(())
                };
            if li == l0 {
                cmp(rep, "attn_in", tr.attn_in.as_ref(), &format!("{p}/attn/x"))?;
                cmp(rep, "qr", tr.qr.as_ref(), &format!("{p}/qr/out"))?;
                cmp(rep, "latent", tr.latent.as_ref(), &format!("{p}/comp/out"))?;
                cmp(rep, "attn_q", tr.q.as_ref(), &format!("{p}/sattn/q"))?;
                cmp(rep, "attn_kv", tr.kv_all.as_ref(), &format!("{p}/sattn/kv"))?;
                cmp(rep, "attn_o", tr.attn_o.as_ref(), &format!("{p}/sattn/o"))?;
                cmp(
                    rep,
                    "attn_out",
                    tr.attn_out.as_ref(),
                    &format!("{p}/attn/out"),
                )?;
                cmp(rep, "mid_stream", tr.mid.as_ref(), &format!("{p}/hcpost0"))?;
                cmp(rep, "moe_in", tr.moe_in.as_ref(), &format!("{p}/moe/x"))?;
                cmp(rep, "moe_out", tr.moe_out.as_ref(), &format!("{p}/moe/out"))?;
                if let (Some(ib), true) = (tr.idx_all.as_ref(), f.has(&format!("{p}/sattn/idx"))) {
                    let got: Vec<i64> = bytes_i32(&ib.host(&st)?)
                        .into_iter()
                        .map(|x| x as i64)
                        .collect();
                    let want = f.i64s(&format!("{p}/sattn/idx"))?;
                    let w = want.len() / s;
                    let mut same_rows = 0;
                    let mut jacc = 0f64;
                    for r in 0..s {
                        let mut a: Vec<i64> = got[r * w..(r + 1) * w]
                            .iter()
                            .copied()
                            .filter(|&x| x >= 0)
                            .collect();
                        let mut b: Vec<i64> = want[r * w..(r + 1) * w]
                            .iter()
                            .copied()
                            .filter(|&x| x >= 0)
                            .collect();
                        a.sort();
                        b.sort();
                        if a == b {
                            same_rows += 1;
                        }
                        let inter = a.iter().filter(|x| b.binary_search(x).is_ok()).count();
                        let uni = a.len() + b.len() - inter;
                        jacc += if uni == 0 {
                            1.0
                        } else {
                            inter as f64 / uni as f64
                        };
                    }
                    let jm = jacc / s as f64;
                    rep.raw(
                        &format!("layer{li}/attn_idx"),
                        jm >= 0.99,
                        "mean per-row Jaccard >= 0.99",
                        format!("{same_rows}/{s} rows identical, mean Jaccard {jm:.5}, width {w}"),
                    );
                }
                if !tr.route_sel.is_empty() && f.has(&format!("{p}/gate/out1")) {
                    let want = f.i64s(&format!("{p}/gate/out1"))?;
                    let mine: Vec<i64> = tr.route_sel.iter().map(|&x| x as i64).collect();
                    let k = g.topk;
                    let agree = route_agree(&mine, &want, s, k);
                    let fagree = match &fl {
                        Some(x) if x.has(&format!("{p}/gate/out1")) => Some(route_agree(
                            &x.i64s(&format!("{p}/gate/out1"))?,
                            &want,
                            s,
                            k,
                        )),
                        _ => None,
                    };
                    let (pass, rule) = match fagree {
                        Some(fa) => (
                            agree + s / 100 >= fa,
                            format!("token expert sets agree within 1% of the floor's {fa}/{s}"),
                        ),
                        None => (
                            agree * 100 >= s * 99,
                            ">= 99% of token expert sets agree (no floor)".into(),
                        ),
                    };
                    rep.raw(
                        &format!("layer{li}/route"),
                        pass,
                        &rule,
                        format!("{agree}/{s} token sets equal"),
                    );
                }
            }
            let okey = format!("{p}/out/x");
            let owant = f.f32s(&okey)?;
            saver
                .borrow_mut()
                .put(&okey, "bf16", &f.shape(&okey), &h2.host(&cx.ops.st)?)?;
            saver.borrow_mut().put(
                &format!("{p}/out/pre_mix"),
                "f32",
                &f.shape(&format!("{p}/out/pre_mix")),
                &pre2.host(&cx.ops.st)?,
            )?;
            if !tr.route_sel.is_empty() {
                saver.borrow_mut().put(
                    &format!("{p}/gate/out1"),
                    "i32",
                    &[s, g.topk],
                    &v::i32_bytes(&tr.route_sel),
                )?;
            }
            let (r, m, w, c) = stream_metrics(&bf(&cx.ops, &h2)?, &owant, s);
            // a chained layer starts from this rewrite's own previous output, not the capture's
            let (pass, rule) = band(
                r,
                if li > l0 {
                    None
                } else {
                    floor_rel(&okey, &owant, s)?
                },
                tol,
            );
            // a chained layer runs on this rewrite's own previous output: its drift is reported and the
            // chain is judged where the vendor judges a forward, at the head (`--head`)
            let (pass, rule) = if li > l0 {
                (true, "reported (chained)".to_string())
            } else {
                (pass, rule)
            };
            rep.raw(&format!("layer{li}/out_stream{}", if li > l0 { "_chained" } else { "" }), pass, &rule, format!("rel_l2={r:.3e} maxabs/max={m:.3e} worst_tok={w:.3e} cos={c:.8} load {load_s:.1}s fwd {fwd_s:.2}s"));
            let pkey = format!("{p}/out/pre_mix");
            let pwant = f.f32s(&pkey)?;
            let pm = stat(&f32h(&cx.ops, &pre2)?, &pwant, false);
            let (ppass, prule) = band(
                pm.rel_l2,
                if li > l0 {
                    None
                } else {
                    floor_rel(&pkey, &pwant, s)?
                },
                tol,
            );
            let (ppass, prule) = if li > l0 {
                (true, "reported (chained)".to_string())
            } else {
                (ppass, prule)
            };
            rep.raw(
                &format!(
                    "layer{li}/out_pre_mix{}",
                    if li > l0 { "_chained" } else { "" }
                ),
                ppass,
                &prule,
                format!("rel_l2={:.3e} max_abs={:.3e}", pm.rel_l2, pm.max_abs),
            );
            summary.push(format!(
                "{{\"layer\": {li}, \"rel_l2\": {r:e}, \"cos\": {c}, \"worst_tok\": {w:e}}}"
            ));
            if li == l0 && decode > 0 {
                decode_steps(
                    &mut cx,
                    &stm,
                    &f,
                    &lw,
                    &mut sh,
                    &mut ls,
                    token_map,
                    &ids,
                    decode,
                    tol,
                    fl.as_ref(),
                    &mut saver.borrow_mut(),
                    rep,
                )?;
            }
            h = h2;
            pre = pre2;
            last = li;
        }
        // after the last trunk layer: collapse with the carried pre-mix, final norm, the fp32 head
        // on the newest token, against the capture's logits
        if head && last + 1 == g.compress_ratios.len() - 3 && f.has("logits") {
            let (hc, d) = (g.hc, g.dim);
            let off = (s - 1) * hc * d * 2;
            let hl = cx.ops.buf(hc * d * 2)?;
            v::copy_d2d(&cx.ops, &h, off, &hl, 0, hc * d * 2)?;
            let pl = cx.ops.buf(hc * 4)?;
            v::copy_d2d(&cx.ops, &pre, (s - 1) * hc * 4, &pl, 0, hc * 4)?;
            let xl = cx.ops.buf(d * 2)?;
            ck("hc_pre", unsafe {
                ffi::memra_v41_hc_pre(
                    hl.p(&st).m(),
                    pl.p(&st).f(),
                    1,
                    hc as i32,
                    d as i32,
                    xl.p(&st).m(),
                    d as i64,
                    cx.ops.sv(),
                )
            })?;
            let nw = v::load_vec_bf16(&cx.ops, &stm, "norm.weight", d)?;
            cx.ops
                .rmsnorm(xl.p(&st), d, &nw, xl.p(&st), d, 1, d, g.norm_eps)?;
            let vocab = stm.info("head.weight").ok_or("head.weight missing")?.shape[0] as usize;
            let hw = v::load_bf16(&cx.ops, &stm, "head.weight", vocab, d)?;
            let lg = cx.ops.buf(vocab * 4)?;
            cx.ops
                .linear_bf16(xl.p(&st), d, 1, &hw, lg.p(&st), vocab, true)?;
            let got = f32h(&cx.ops, &lg)?;
            let want = f.f32s("logits")?; // [steps][vocab], step 0 = the prefill's last token
            let want = &want[..vocab];
            let am = |x: &[f32]| {
                x.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map(|(i, _)| i)
                    .unwrap()
            };
            let lse = |x: &[f32]| {
                let m = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
                m + x.iter().map(|&v| (v as f64 - m).exp()).sum::<f64>().ln()
            };
            let (lg_, lw_) = (lse(&got), lse(want));
            let kl: f64 = want
                .iter()
                .zip(&got)
                .map(|(&w, &q)| {
                    let pw = (w as f64 - lw_).exp();
                    pw * ((w as f64 - lw_) - (q as f64 - lg_))
                })
                .sum();
            let mut top_w: Vec<usize> = (0..vocab).collect();
            top_w.sort_by(|a, b| want[*b].total_cmp(&want[*a]));
            let mut top_g: Vec<usize> = (0..vocab).collect();
            top_g.sort_by(|a, b| got[*b].total_cmp(&got[*a]));
            let t5 = top_w[..5].iter().filter(|i| top_g[..5].contains(i)).count();
            rep.raw(
                "head/next_token",
                am(&got) == am(want),
                "argmax equal to the vendor's; KL(vendor || rewrite) reported",
                format!(
                    "argmax {} vs {}, top-5 overlap {t5}/5, KL {kl:.3e}, chained from layer {l0}",
                    am(&got),
                    am(want)
                ),
            );
            summary.push(format!(
                "{{\"head_kl\": {kl:e}, \"argmax_equal\": {}}}",
                am(&got) == am(want)
            ));
        }
    }
    saver.borrow().close()?;
    Ok(format!(
        "\"tokens\": {s}, \"mint\": {mint}, \"layers\": [{}]",
        summary.join(", ")
    ))
}

// ------------------------------------------------------------------------------------------------
// N-rank gates

/// Byte identity of an N-rank tensor against the single-card one; on a mismatch the bf16 or f32
/// distance is reported too (diagnosis, not a band).
/// One decode step's integer outcome against the vendor's: equal bytes (width included), or where
/// they first differ.
fn int_rows(got: &[i32], want: &[i64]) -> (bool, String) {
    if got.len() != want.len() {
        return (
            false,
            format!("{} entries against the vendor's {}", got.len(), want.len()),
        );
    }
    match got.iter().zip(want).position(|(&a, &b)| a as i64 != b) {
        None => (true, format!("{} entries identical", want.len())),
        Some(i) => (
            false,
            format!(
                "first difference at entry {i} of {} ({} against {})",
                want.len(),
                got[i],
                want[i]
            ),
        ),
    }
}

/// The decode program below and across the window, per layer, at the integer level the two
/// programs differ in: each decode step from the capture's inputs (stream, pre-mix, the CSA2 state
/// of the sources it reads) with this layer's own ring and compressor pair carried from its
/// prefill, then the step's attention index row (window ring, oldest slot first, -1 for empty
/// slots, then the compressed selection; width and order), the compressor's emission on a ratio-2
/// kv source, the indexer's selection on an index source and the Engram hash ids, each against the
/// vendor decode program's own bytes. The red arm runs the same position under prefill semantics
/// (one prefill of every position up to it, the program a loop that re-prefills would run) on the
/// layers that hold all their CSA2 state themselves, and asserts the byte check rejects it at every
/// position where the vendor's own prefill row is not its decode row: everywhere but `window - 1`
/// (below it the prefill row is narrower, past it the ring row holds slots, not positions). Streams are reported (rel L2 against the vendor), not gated: below the
/// window the two programs sit equally far from the vendor there (`decode/README.md`).
fn decode_gate(
    ckpt: &Path,
    cap: &Path,
    layers: &[usize],
    token_map: &Path,
    rep: &mut Report,
) -> Result<String, String> {
    let e = memra_engine::Engine::new(0).map_err(|e| e.to_string())?;
    let st: Arc<cudarc::driver::CudaStream> = e.stream();
    let mc = memra_gguf::config::ModelConfig::from_config_json(&ckpt.join("config.json"))
        .map_err(|e| e.to_string())?;
    let g = Geo::from_config(&mc)?;
    let stm = memra_gguf::safetensors::StModel::open(ckpt).map_err(|e| e.to_string())?;
    let mut f = Fx::open(cap)?;
    f.ckpt = ckpt.to_path_buf();
    let ec = v::engram::EngramConsts::derive(&g, v::engram::token_map_from_file(token_map)?)?;
    let ids: Vec<u32> = f
        .i64s("prompt_ids")?
        .into_iter()
        .map(|x| x as u32)
        .collect();
    let fed: Vec<u32> = f.i64s("gen_ids")?.into_iter().map(|x| x as u32).collect();
    let p = ids.len();
    let l0 = *layers.first().ok_or("no layers")?;
    let steps = (1..=fed.len())
        .take_while(|k| f.has(&format!("s{k}/L{l0}/out/x")))
        .count();
    if steps == 0 {
        return Err("the capture holds no decode steps".into());
    }
    let win = g.window;
    let mut cx = Ctx::new(st.clone(), g.clone(), (p + steps + 64).max(1024))?;
    cx.trace = true;
    let row = g.hc * g.dim * 2;
    // the Engram hash columns of layer `li` for positions `from..` of `seq`
    let cols_of = |seq: &[u32], from: usize, li: usize| -> Result<Vec<i64>, String> {
        let all = ec.hash_span(seq, from)?;
        let sl = ec
            .layer_ids
            .iter()
            .position(|&x| x == li)
            .ok_or("engram layer not in the constants")?;
        let (nl, cols) = (ec.layer_ids.len(), ec.cols());
        Ok((0..seq.len() - from)
            .flat_map(|t| all[(t * nl + sl) * cols..(t * nl + sl + 1) * cols].to_vec())
            .collect())
    };
    let mut seq: Vec<u32> = ids.clone();
    seq.extend_from_slice(&fed[..steps]);
    let mut red_layers = Vec::new();
    let mut grow_layers = Vec::new();
    for &l in layers {
        let lw = LayerW::load(&cx.ops, &stm, &g, l, false)?;
        let engram = lw.engram.is_some();
        let in_key = |k: usize| {
            if engram {
                format!("s{k}/L{l}/engram/x")
            } else {
                format!("s{k}/L{l}/in/x")
            }
        };
        // the prefill from the capture's inputs seeds this layer's ring and compressor pair
        let mut sh = Shared::default();
        load_shared(&f, &format!("s0/L{l}"), l, &g, p, &mut sh, &cx.ops)?;
        let mut h_rows = f.bytes(&in_key(0))?;
        let mut pre_rows = f.bytes(&format!("s0/L{l}/in/pre_mix"))?;
        let h = Buf::from_bytes(&st, &h_rows)?;
        let pre = Buf::from_bytes(&st, &pre_rows)?;
        let mut tr = Trace::default();
        if engram {
            cx.engram(&stm, &lw, &h, &cols_of(&ids, 0, l)?, p, &mut tr)?;
        }
        let mut ls = LayerState::default();
        cx.block_forward(&lw, &h, &pre, 0, p, &mut sh, &mut ls, &mut tr)?;
        // the red arm runs on layers whose CSA2 state is all their own at prefill
        let ratio = g.compress_ratios[l];
        let red = ratio == 0 || (lw.kv_source && lw.index_source);
        if red {
            red_layers.push(l);
        }
        for k in 1..=steps {
            let pos = p + k - 1;
            let ps = format!("s{k}/L{l}");
            load_shared(&f, &ps, l, &g, 1, &mut sh, &cx.ops)?;
            let hb = f.bytes(&in_key(k))?;
            let pb = f.bytes(&format!("{ps}/in/pre_mix"))?;
            let h = Buf::from_bytes(&st, &hb)?;
            let pre = Buf::from_bytes(&st, &pb)?;
            let mut tr = Trace::default();
            if engram {
                let mine = cols_of(&seq[..pos + 1], pos, l)?;
                let want = f.i64s(&format!("{ps}/engram/hash"))?;
                let diff = mine.iter().zip(&want).filter(|(a, b)| a != b).count()
                    + mine.len().abs_diff(want.len());
                rep.raw(
                    &format!("layer{l}/decode{k}/engram_hash"),
                    diff == 0,
                    "the vendor NgramHashState's ids for the new token, exactly",
                    format!("pos {pos}: {} ids, {diff} differ", want.len()),
                );
                cx.engram(&stm, &lw, &h, &mine, 1, &mut tr)?;
            }
            let (h2, _p2) = cx.block_forward(&lw, &h, &pre, pos, 1, &mut sh, &mut ls, &mut tr)?;
            let want_idx = f.i64s(&format!("{ps}/sattn/idx"))?;
            let got_idx = bytes_i32(
                &tr.idx_all
                    .as_ref()
                    .ok_or("no index row traced")?
                    .host(&st)?,
            );
            let (same, extra) = int_rows(&got_idx, &want_idx);
            rep.raw(
                &format!("layer{l}/decode{k}/attn_idx"),
                same,
                "the vendor decode program's attention index row, byte for byte (window ring oldest slot first, -1 for empty slots, then the compressed selection)",
                format!("pos {pos}: {extra}"),
            );
            if lw.kv_source && ratio == 2 {
                let mine = tr.latent.is_some();
                let want = f.has(&format!("{ps}/comp/out"));
                rep.raw(
                    &format!("layer{l}/decode{k}/compressor_emits"),
                    mine == want,
                    "the ratio-2 compressor pools exactly when the vendor's does (a pair completes)",
                    format!("pos {pos}: memra {mine}, vendor {want}"),
                );
            }
            if lw.index_source && f.has(&format!("{ps}/out/topk_idxs")) {
                let got = bytes_i32(&sh.topk.as_ref().ok_or("no selection")?.host(&st)?);
                let want = f.i64s(&format!("{ps}/out/topk_idxs"))?;
                let (same, extra) = int_rows(&got, &want);
                rep.raw(
                    &format!("layer{l}/decode{k}/selection"),
                    same,
                    "the indexer's decode selection, byte for byte",
                    format!("pos {pos}: {extra}"),
                );
            }
            let want_x = f.f32s(&format!("{ps}/out/x"))?;
            let (r, _, _, c) = stream_metrics(&bf(&cx.ops, &h2)?, &want_x, 1);
            rep.raw(
                &format!("layer{l}/decode{k}/stream"),
                true,
                "reported: rel L2 against the vendor's decode output",
                format!("pos {pos}: rel_l2={r:.3e} cos={c:.8}"),
            );
            h_rows.extend_from_slice(&hb[..row]);
            pre_rows.extend_from_slice(&pb);
            if !red {
                continue;
            }
            // red arm: position `pos` under prefill semantics, one prefill of positions 0..=pos
            let s = pos + 1;
            let hr = Buf::from_bytes(&st, &h_rows)?;
            let pr = Buf::from_bytes(&st, &pre_rows)?;
            let mut tr2 = Trace::default();
            if engram {
                cx.engram(&stm, &lw, &hr, &cols_of(&seq[..s], 0, l)?, s, &mut tr2)?;
            }
            let (mut sh2, mut ls2) = (Shared::default(), LayerState::default());
            let (hx, _) = cx.block_forward(&lw, &hr, &pr, 0, s, &mut sh2, &mut ls2, &mut tr2)?;
            let all = bytes_i32(
                &tr2.idx_all
                    .as_ref()
                    .ok_or("no index rows traced")?
                    .host(&st)?,
            );
            let w = all.len() / s;
            let last = &all[(s - 1) * w..];
            let (same, extra) = int_rows(last, &want_idx);
            // the vendor's two rows are the same bytes at one position only: window - 1, where the
            // prefill row lists positions 0..window and the full ring lists slots 0..window
            // (below it the prefill row is narrower; past it the ring holds slots, not positions)
            let expect_same = pos + 1 == win;
            rep.raw(
                &format!("layer{l}/decode{k}/red_prefill_attn_idx"),
                same == expect_same,
                "red arm (re-prefill): the byte check rejects the prefill program's row at every position where the vendor's prefill row is not its decode row (all but position window - 1, where both list 0..window)",
                format!(
                    "pos {pos}: {} ({extra})",
                    if same { "accepted" } else { "rejected" }
                ),
            );
            let hx_last = bf(&cx.ops, &hx)?;
            let n = hx_last.len() / s;
            let (r2, _, _, _) = stream_metrics(&hx_last[(s - 1) * n..], &want_x, 1);
            rep.raw(
                &format!("layer{l}/decode{k}/red_prefill_stream"),
                true,
                "reported: the re-prefill's rel L2 against the vendor's decode output",
                format!("pos {pos}: rel_l2={r2:.3e} (decode program {r:.3e})"),
            );
        }
        eprintln!("layer {l}: {steps} decode steps from position {p}");
        if lw.kv_source && lw.index_source && ratio > 0 {
            let h0 = f.bytes(&in_key(0))?;
            let p0 = f.bytes(&format!("s0/L{l}/in/pre_mix"))?;
            let cols = |seq: &[u32], from: usize| -> Result<Option<Vec<i64>>, String> {
                if engram {
                    cols_of(seq, from, l).map(Some)
                } else {
                    Ok(None)
                }
            };
            grow_case(&mut cx, &stm, &lw, &h0, &p0, &ids, &cols, rep)?;
            grow_layers.push(l);
        }
    }
    Ok(format!(
        "\"prompt\": {p}, \"decode_steps\": {steps}, \"positions\": [{}, {}], \"window\": {win}, \"layers\": {layers:?}, \"red_arm_layers\": {red_layers:?}, \"grow_layers\": {grow_layers:?}",
        p,
        p + steps - 1
    ))
}

/// The Engram hash columns of a layer for positions `from..` of a sequence (None: not an Engram
/// layer).
type ColsFn<'a> = dyn Fn(&[u32], usize) -> Result<Option<Vec<i64>>, String> + 'a;
/// One arm of the growth case: the call's output, the compressed KV rows, the index-key rows, and
/// the compressed KV's capacity after the prefill and after the call.
type GrowArm = (Vec<u8>, Vec<u8>, Vec<u8>, usize, usize);

/// The growth case of the decode gate on a layer that owns its compressed KV and index keys: a
/// prefill of more than [`v::SMALL_M`] rows sizes them for exactly its rows, then a one-token
/// call (a second-stream branch: at most `SMALL_M` rows, no trace) completes one more compressed
/// row, so both grow inside the call. The same two calls with room for the row (nothing grows)
/// are the reference: the call's output and both buffers' rows must be the same bytes.
///
/// Both arms run under the second stream's fault injection: every branch the second stream
/// starts first waits 20 ms ([`v::SIDE_DELAY_NS`]), and a main-stream buffer is filled with 0xFF
/// as it drops ([`v::POISON_FREE`]). A grown buffer copied on the second stream from a
/// main-stream buffer the call already let go reads the fill (red: the bytes differ); copied on
/// the main stream before the branch starts, it reads the rows.
#[allow(clippy::too_many_arguments)]
fn grow_case(
    cx: &mut Ctx,
    stm: &memra_gguf::safetensors::StModel,
    lw: &LayerW,
    h_rows: &[u8],
    pre_rows: &[u8],
    ids: &[u32],
    cols: &ColsFn<'_>,
    rep: &mut Report,
) -> Result<(), String> {
    use std::sync::atomic::Ordering::Relaxed;
    let g = cx.g.clone();
    let st = cx.ops.st.clone();
    let l = lw.id;
    let ratio = lw.ratio;
    let row = g.hc * g.dim * 2;
    let prow = g.hc * 4;
    let p = h_rows.len() / row;
    // the prefill's length: more rows than a second-stream call takes, and one the next token
    // completes a compressed row after (odd at ratio 2: the next token closes a pair)
    let q = (v::SMALL_M + 1..p)
        .rev()
        .find(|&q| (q + 1) % ratio == 0)
        .ok_or_else(|| {
            format!("layer {l}: a prompt of {p} tokens is too short for the growth case")
        })?;
    let (was_trace, was_hint) = (cx.trace, cx.rows_hint);
    let arm = |cx: &mut Ctx, hint: usize| -> Result<GrowArm, String> {
        cx.trace = false;
        cx.rows_hint = hint;
        let (mut sh, mut ls) = (Shared::default(), LayerState::default());
        let h = Buf::from_bytes(&st, &h_rows[..q * row])?;
        let pre = Buf::from_bytes(&st, &pre_rows[..q * prow])?;
        if let Some(c) = cols(&ids[..q], 0)? {
            cx.engram(stm, lw, &h, &c, q, &mut Trace::default())?;
        }
        cx.block_forward(lw, &h, &pre, 0, q, &mut sh, &mut ls, &mut Trace::default())?;
        let cap0 = sh.compress_cap;
        let hd = Buf::from_bytes(&st, &h_rows[q * row..(q + 1) * row])?;
        let pd = Buf::from_bytes(&st, &pre_rows[q * prow..(q + 1) * prow])?;
        if let Some(c) = cols(&ids[..q + 1], q)? {
            cx.engram(stm, lw, &hd, &c, 1, &mut Trace::default())?;
        }
        let (h2, _) =
            cx.block_forward(lw, &hd, &pd, q, 1, &mut sh, &mut ls, &mut Trace::default())?;
        let n_c = sh.n_c;
        let mut kv = sh
            .compress_kv
            .as_ref()
            .ok_or("no compressed KV")?
            .host(&st)?;
        kv.truncate(n_c * g.head_dim * 2);
        let mut ik = sh.index_k.as_ref().ok_or("no index keys")?.host(&st)?;
        ik.truncate(n_c * g.index_dim * 2);
        Ok((h2.host(&st)?, kv, ik, cap0, sh.compress_cap))
    };
    v::SIDE_DELAY_NS.store(20_000_000, Relaxed);
    v::POISON_FREE.store(true, Relaxed);
    let room = cx.max_pos;
    let tight = arm(cx, q);
    let roomy = arm(cx, room);
    v::SIDE_DELAY_NS.store(0, Relaxed);
    v::POISON_FREE.store(false, Relaxed);
    cx.trace = was_trace;
    cx.rows_hint = was_hint;
    let (th, tkv, tik, cap0, cap1) = tight?;
    let (rh, rkv, rik, _, _) = roomy?;
    rep.raw(
        &format!("layer{l}/grow/rows"),
        cap1 > cap0 && cap0 == q / ratio,
        "the case grows: the prefill's compressed KV holds exactly its rows and the one-token call that completes the next row grows it",
        format!("prefill {q} tokens, ratio {ratio}: {cap0} rows, then {cap1}"),
    );
    same_bytes(rep, &format!("layer{l}/grow/stream"), &th, &rh, "bf16");
    same_bytes(
        rep,
        &format!("layer{l}/grow/compress_kv"),
        &tkv,
        &rkv,
        "bf16",
    );
    same_bytes(rep, &format!("layer{l}/grow/index_k"), &tik, &rik, "bf16");
    Ok(())
}

fn same_bytes(rep: &mut Report, name: &str, got: &[u8], want: &[u8], dtype: &str) {
    let diff =
        got.iter().zip(want).filter(|(a, b)| a != b).count() + got.len().abs_diff(want.len());
    let extra = if diff == 0 {
        format!("{} bytes identical", want.len())
    } else {
        let st = match dtype {
            "bf16" => stat(&bf16_to_f32(got), &bf16_to_f32(want), true),
            "f32" => stat(&bytes_f32(got), &bytes_f32(want), false),
            _ => Stat::default(),
        };
        format!(
            "{diff} of {} bytes differ (rel_l2 {:.3e}, max_abs {:.3e})",
            want.len(),
            st.rel_l2,
            st.max_abs
        )
    };
    rep.raw(
        name,
        diff == 0,
        "byte-identical to the single-card block_forward",
        extra,
    );
}

fn opt_host(ops: &Ops, b: Option<&Buf>) -> Result<Option<Vec<u8>>, String> {
    b.map(|x| x.host(&ops.st)).transpose()
}

/// One step's outputs and state: (stream, pre-mix, trace, CSA2 state, layer state), per rank.
type RankOut<'a> = (
    &'a [Buf],
    &'a [Buf],
    &'a [Trace],
    &'a [Shared],
    &'a [LayerState],
);

/// Compare every rank's replicated state and outputs with the single-card run of the same step.
#[allow(clippy::too_many_arguments)]
fn tp_compare(
    rep: &mut Report,
    tag: &str,
    cx: &Ctx,
    grp: &v::tp::Group,
    want: (&Buf, &Buf, &Trace, &Shared, &LayerState),
    got: RankOut<'_>,
) -> Result<(), String> {
    let (h2, p2, tr, sh, ls) = want;
    let (h2s, p2s, trs, shs, lss) = got;
    let o = &cx.ops;
    let wh = h2.host(&o.st)?;
    let wp = p2.host(&o.st)?;
    let wa = opt_host(o, tr.attn_out.as_ref())?;
    let wm = opt_host(o, tr.moe_out.as_ref())?;
    let rows_c = |b: Option<&Buf>, n: usize, rb: usize| -> Result<Option<Vec<u8>>, String> {
        Ok(opt_host(o, b)?.map(|mut v| {
            v.truncate(n * rb);
            v
        }))
    };
    let g = &cx.g;
    let wkv = rows_c(sh.compress_kv.as_ref(), sh.n_c, g.head_dim * 2)?;
    let wik = rows_c(sh.index_k.as_ref(), sh.n_c, g.index_dim * 2)?;
    let wtk = opt_host(o, sh.topk.as_ref())?;
    let wcand = opt_host(o, sh.cand.as_ref())?;
    let wring = opt_host(o, ls.ring.as_ref())?;
    let wckv = opt_host(o, ls.c_kv.as_ref())?;
    let wcsc = opt_host(o, ls.c_sc.as_ref())?;
    // the hashes of both sides (logged, not checks): a mismatch then shows at once whether the
    // single-card reference or the ranks moved between runs
    let hash = |b: &[u8]| -> String {
        use sha2::Digest;
        sha2::Sha256::digest(b)[..8]
            .iter()
            .map(|x| format!("{x:02x}"))
            .collect()
    };
    println!("HASH {tag}/single/out_stream {}", hash(&wh));
    if let Some(w) = wm.as_ref() {
        println!("HASH {tag}/single/moe_out {}", hash(w));
    }
    for r in 0..grp.n() {
        let ro = &grp.cx[r].ops;
        let pre = format!("{tag}/rank{r}");
        println!("HASH {pre}/out_stream {}", hash(&h2s[r].host(&ro.st)?));
        if let Some(b) = trs[r].moe_out.as_ref() {
            println!("HASH {pre}/moe_out {}", hash(&b.host(&ro.st)?));
        }
        same_bytes(
            rep,
            &format!("{pre}/out_stream"),
            &h2s[r].host(&ro.st)?,
            &wh,
            "bf16",
        );
        same_bytes(
            rep,
            &format!("{pre}/out_pre_mix"),
            &p2s[r].host(&ro.st)?,
            &wp,
            "f32",
        );
        if let (Some(w), Some(b)) = (wa.as_ref(), trs[r].attn_out.as_ref()) {
            same_bytes(rep, &format!("{pre}/attn_out"), &b.host(&ro.st)?, w, "bf16");
        }
        if let (Some(w), Some(b)) = (wm.as_ref(), trs[r].moe_out.as_ref()) {
            same_bytes(rep, &format!("{pre}/moe_out"), &b.host(&ro.st)?, w, "bf16");
        }
        let gh = |b: Option<&Buf>, n: usize, rb: usize| -> Result<Option<Vec<u8>>, String> {
            Ok(opt_host(ro, b)?.map(|mut v| {
                v.truncate(n * rb);
                v
            }))
        };
        let pairs = [
            (
                "compress_kv",
                wkv.clone(),
                gh(shs[r].compress_kv.as_ref(), shs[r].n_c, g.head_dim * 2)?,
                "bf16",
            ),
            (
                "index_k",
                wik.clone(),
                gh(shs[r].index_k.as_ref(), shs[r].n_c, g.index_dim * 2)?,
                "bf16",
            ),
            (
                "topk",
                wtk.clone(),
                opt_host(ro, shs[r].topk.as_ref())?,
                "i32",
            ),
            (
                // each rank keeps only the candidate rows it selects (`Shared::cand_row0`):
                // compare those rows of the single-card pool
                "candidates",
                wcand.as_ref().map(|w| {
                    let (a, b) = cand_rows(&shs[r], w.len());
                    w[a..b].to_vec()
                }),
                opt_host(ro, shs[r].cand.as_ref())?.map(|mut v| {
                    let (a, b) = cand_rows(&shs[r], wcand.as_ref().map_or(v.len(), |w| w.len()));
                    v.truncate(b - a);
                    v
                }),
                "u8",
            ),
            (
                "window_ring",
                wring.clone(),
                opt_host(ro, lss[r].ring.as_ref())?,
                "bf16",
            ),
            (
                "pair_kv",
                wckv.clone(),
                opt_host(ro, lss[r].c_kv.as_ref())?,
                "f32",
            ),
            (
                "pair_score",
                wcsc.clone(),
                opt_host(ro, lss[r].c_sc.as_ref())?,
                "f32",
            ),
        ];
        for (nm, w, gt, dt) in pairs {
            match (w, gt) {
                (Some(w), Some(gt)) => same_bytes(rep, &format!("{pre}/state/{nm}"), &gt, &w, dt),
                (None, None) => {}
                (w, gt) => rep.raw(
                    &format!("{pre}/state/{nm}"),
                    false,
                    "the same state exists on both sides",
                    format!("single-card {}, rank {}", w.is_some(), gt.is_some()),
                ),
            }
        }
    }
    if !tr.route_sel.is_empty() {
        // every rank routes its own replicated copy (a rank run with nothing traced, as the
        // graph door's repeats are, recorded no routing: its outputs and state are compared)
        for (r, t) in trs.iter().enumerate() {
            if t.route_sel.is_empty() && (!grp.trace || grp.n() == 1) {
                // (one rank runs the single-card MoE, which records under the rank's own flag)
                continue;
            }
            let same = t.route_sel == tr.route_sel && t.route_w == tr.route_w;
            rep.raw(
                &format!("{tag}/rank{r}/route"),
                same,
                "identical expert ids and weights",
                format!("{} selections", tr.route_sel.len()),
            );
        }
    }
    Ok(())
}

/// The byte range of the single-card candidate pool (`whole` bytes) that a rank's pool holds:
/// its rows from `cand_row0`, as many as its buffer has and the whole pool reaches.
fn cand_rows(sh: &Shared, whole: usize) -> (usize, usize) {
    let ld = sh.cand_ld.max(1);
    let rows = sh.cand.as_ref().map_or(0, |b| b.len / ld);
    let a = (sh.cand_row0 * ld).min(whole);
    let b = ((sh.cand_row0 + rows) * ld).min(whole);
    (a, b)
}

/// `dsv41-gate tp`: the N-rank program (`dsv41_gpu::tp`) against the single-card `block_forward`
/// on the same captured inputs, per layer, prefill and decode, byte for byte; plus the N-rank
/// output against the vendor capture (reported, the single-card gate owns that band).
// allow: the rank loops walk several per-rank vectors in step
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn tp_gate(
    ckpt: &Path,
    cap: &Path,
    layers: &[usize],
    devices: &[usize],
    token_map: Option<&Path>,
    decode: usize,
    chunk: Option<usize>,
    rep: &mut Report,
) -> Result<String, String> {
    let mc = memra_gguf::config::ModelConfig::from_config_json(&ckpt.join("config.json"))
        .map_err(|e| e.to_string())?;
    let g = Geo::from_config(&mc)?;
    let stm = memra_gguf::safetensors::StModel::open(ckpt).map_err(|e| e.to_string())?;
    let mut f = Fx::open(cap)?;
    f.ckpt = ckpt.to_path_buf();
    let ids: Vec<u32> = f
        .i64s("prompt_ids")?
        .into_iter()
        .map(|x| x as u32)
        .collect();
    let s = ids.len();
    let max_pos = (s + 64).max(1024);
    let mut grp = v::tp::Group::new(devices, &g, max_pos)?;
    grp.trace = true;
    if grp.transport == memra_engine::tp_transport::TpTransport::DevicePush && grp.n() > 1 {
        // the small gathers inside kernels: 20,000 back to back, every consumer checking every
        // producer's region; then the control, the word stored before the data, which must show
        let (diff, flagged) = grp.gather_stress(20_000, false)?;
        rep.raw(
            "stress/push_gather",
            diff == 0 && flagged == 0,
            "every consumer reads what every producer pushed",
            format!(
                "20000 gathers on devices {devices:?}: {diff} words differ, {flagged} ranks flagged"
            ),
        );
        let (diff, flagged) = grp.gather_stress(2_000, true)?;
        rep.raw(
            "stress/push_gather_control_word_first",
            diff > 0 && flagged == 0,
            "the word before the data is caught (words differ)",
            format!(
                "2000 gathers on devices {devices:?}: {diff} words differ, {flagged} ranks flagged"
            ),
        );
    }
    let e = memra_engine::Engine::new(devices[0]).map_err(|e| e.to_string())?;
    let mut cx = Ctx::new(e.stream(), g.clone(), max_pos)?;
    cx.trace = true;
    if let Some(c) = chunk {
        cx.chunk = c;
        for c2 in grp.cx.iter_mut() {
            c2.chunk = c;
        }
    }
    let n = grp.n();
    let mut summary = Vec::new();
    // decode steps repeated through the graph door's captured spans (the door records only
    // the in-kernel gathers: under another transport it changes nothing and nothing is repeated)
    let mut graph_steps = 0u64;
    let graph_can = n == 1 || grp.transport == memra_engine::tp_transport::TpTransport::DevicePush;
    for &l in layers {
        let pfx = format!("s0/L{l}");
        if !f.has(&format!("{pfx}/out/x")) {
            return Err(format!("capture has no layer {l}"));
        }
        let h_key = if g.engram_layers.contains(&l) && f.has(&format!("{pfx}/engram/x")) {
            format!("{pfx}/engram/x")
        } else {
            format!("{pfx}/in/x")
        };
        let t0 = std::time::Instant::now();
        let lw = LayerW::load(&cx.ops, &stm, &g, l, false)?;
        let lws = grp.load_layer(&stm, &g, l, false)?;
        let load_s = t0.elapsed().as_secs_f64();
        let mut sh = Shared::default();
        load_shared(&f, &pfx, l, &g, s, &mut sh, &cx.ops)?;
        let h = f.dev(&cx.ops, &h_key)?;
        let pre = f.dev(&cx.ops, &format!("{pfx}/in/pre_mix"))?;
        let mut shs: Vec<Shared> = (0..n).map(|_| Shared::default()).collect();
        let mut hs = Vec::with_capacity(n);
        let mut pres = Vec::with_capacity(n);
        for r in 0..n {
            let ops = &grp.cx[r].ops;
            load_shared(&f, &pfx, l, &g, s, &mut shs[r], ops)?;
            hs.push(f.dev(ops, &h_key)?);
            pres.push(f.dev(ops, &format!("{pfx}/in/pre_mix"))?);
        }
        let mut tr = Trace::default();
        let mut trs: Vec<Trace> = (0..n).map(|_| Trace::default()).collect();
        if lw.engram.is_some() {
            let hashes = engram_hashes(&f, token_map, &ids, l, &format!("{pfx}/engram/hash"), rep)?;
            cx.engram(&stm, &lw, &h, &hashes, s, &mut tr)?;
            for r in 0..n {
                grp.cx[r].engram(&stm, &lws[r], &hs[r], &hashes, s, &mut trs[r])?;
            }
            let want = h.host(&cx.ops.st)?;
            for r in 0..n {
                same_bytes(
                    rep,
                    &format!("layer{l}/prefill/rank{r}/engram_out"),
                    &hs[r].host(&grp.cx[r].ops.st)?,
                    &want,
                    "bf16",
                );
            }
        }
        let mut ls = LayerState::default();
        let mut lss: Vec<LayerState> = (0..n).map(|_| LayerState::default()).collect();
        let t1 = std::time::Instant::now();
        let (h2, p2) = cx.block_forward(&lw, &h, &pre, 0, s, &mut sh, &mut ls, &mut tr)?;
        let ref_s = t1.elapsed().as_secs_f64();
        grp.stats = Default::default();
        let t2 = std::time::Instant::now();
        let (h2s, p2s) = grp.block_forward(&lws, &hs, &pres, 0, s, &mut shs, &mut lss, &mut trs)?;
        let tp_s = t2.elapsed().as_secs_f64();
        let st = grp.stats;
        tp_compare(
            rep,
            &format!("layer{l}/prefill"),
            &cx,
            &grp,
            (&h2, &p2, &tr, &sh, &ls),
            (&h2s, &p2s, &trs, &shs, &lss),
        )?;
        // the vendor capture, reported (the single-card layer gate holds the band)
        let want = f.f32s(&format!("{pfx}/out/x"))?;
        let (r_tp, _, _, c_tp) = stream_metrics(&bf(&grp.cx[0].ops, &h2s[0])?, &want, s);
        let (r_1, _, _, _) = stream_metrics(&bf(&cx.ops, &h2)?, &want, s);
        rep.raw(
            &format!("layer{l}/prefill/vs_vendor"),
            true,
            "reported: the N-rank stream against the vendor capture, next to the single-card one",
            format!("rel_l2 N-rank {r_tp:.3e} / single-card {r_1:.3e}, cos {c_tp:.8}"),
        );
        // Calls of 9 to 32 rows keep the small-row bodies, issued over groups of rows, and the
        // shared expert and the mHC mixes on the second stream. The prompt's first 12, 16 and 32
        // rows as their own call, twice: with every family's limit at 0 and nothing on the
        // second stream (the pipelined bodies in stream order, the reference), then with the
        // limits at 32 and the second stream's own limits, on the single card and on every
        // rank; each output of the second against the first. Not on a layer that
        // reads another layer's selection: the capture's selection rows index the full prompt's
        // key rows, not a shorter call's.
        if lw.ratio == 0 || lw.index_source {
            let (hb, pb) = (h.host(&cx.ops.st)?, pre.host(&cx.ops.st)?);
            let (rh, rp) = (g.hc * g.dim * 2, g.hc * 4);
            for rows in [12usize, 16, 32] {
                if rows >= s {
                    continue;
                }
                let tag = format!("layer{l}/rows{rows}");
                let mut want: Option<(Vec<u8>, Vec<u8>)> = None;
                let side: Vec<usize> = v::SIDE_LIMITS
                    .iter()
                    .map(|x| x.load(std::sync::atomic::Ordering::Relaxed))
                    .collect();
                for wide in [false, true] {
                    // every family through its groups up to 32 rows, whatever the defaults are
                    if wide {
                        v::set_wide_rows(32, 32, 32, 32, 32);
                    } else {
                        v::set_wide_rows(0, 0, 0, 0, 0);
                    }
                    for (x, &was) in v::SIDE_LIMITS.iter().zip(&side) {
                        x.store(
                            if wide { was } else { 0 },
                            std::sync::atomic::Ordering::Relaxed,
                        );
                    }
                    let mut sh1 = Shared::default();
                    load_shared(&f, &pfx, l, &g, s, &mut sh1, &cx.ops)?;
                    let (mut ls1, mut tr1) = (LayerState::default(), Trace::default());
                    let h1 = Buf::from_bytes(&cx.ops.st, &hb[..rows * rh])?;
                    let p1 = Buf::from_bytes(&cx.ops.st, &pb[..rows * rp])?;
                    let (o, op) =
                        cx.block_forward(&lw, &h1, &p1, 0, rows, &mut sh1, &mut ls1, &mut tr1)?;
                    let (o, op) = (o.host(&cx.ops.st)?, op.host(&cx.ops.st)?);
                    let mut shn: Vec<Shared> = (0..n).map(|_| Shared::default()).collect();
                    let (mut hn, mut pn) = (Vec::with_capacity(n), Vec::with_capacity(n));
                    for r in 0..n {
                        let ops = &grp.cx[r].ops;
                        load_shared(&f, &pfx, l, &g, s, &mut shn[r], ops)?;
                        hn.push(Buf::from_bytes(&ops.st, &hb[..rows * rh])?);
                        pn.push(Buf::from_bytes(&ops.st, &pb[..rows * rp])?);
                    }
                    let mut lsn: Vec<LayerState> = (0..n).map(|_| LayerState::default()).collect();
                    let mut trn: Vec<Trace> = (0..n).map(|_| Trace::default()).collect();
                    let (on, opn) =
                        grp.block_forward(&lws, &hn, &pn, 0, rows, &mut shn, &mut lsn, &mut trn)?;
                    let Some((wh, wp)) = want.as_ref() else {
                        // the reference pass: the ranks against the single card, as everywhere
                        for r in 0..n {
                            let st = &grp.cx[r].ops.st;
                            same_bytes(
                                rep,
                                &format!("{tag}/pipelined/rank{r}/out_stream"),
                                &on[r].host(st)?,
                                &o,
                                "bf16",
                            );
                        }
                        want = Some((o, op));
                        continue;
                    };
                    same_bytes(rep, &format!("{tag}/single/out_stream"), &o, wh, "bf16");
                    same_bytes(rep, &format!("{tag}/single/out_pre_mix"), &op, wp, "f32");
                    for r in 0..n {
                        let st = &grp.cx[r].ops.st;
                        same_bytes(
                            rep,
                            &format!("{tag}/rank{r}/out_stream"),
                            &on[r].host(st)?,
                            wh,
                            "bf16",
                        );
                        same_bytes(
                            rep,
                            &format!("{tag}/rank{r}/out_pre_mix"),
                            &opn[r].host(st)?,
                            wp,
                            "f32",
                        );
                    }
                }
            }
            // back to the group's own limits
            v::set_wide_rows_default(n);
        }
        println!(
            "layer {l}: load {load_s:.1}s single-card {ref_s:.2}s {n}-rank {tp_s:.2}s, {} gathers {} copies {:.1} MB",
            st.collectives,
            st.copies,
            st.bytes as f64 / 1e6
        );
        let mut dec = Vec::new();
        if decode > 0 {
            let gen_ids: Vec<u32> = f.i64s("gen_ids")?.into_iter().map(|x| x as u32).collect();
            for step in 1..=decode.min(gen_ids.len()) {
                let ps = format!("s{step}/L{l}");
                if !f.has(&format!("{ps}/out/x")) {
                    break;
                }
                let pos = s + step - 1;
                load_shared(&f, &ps, l, &g, 1, &mut sh, &cx.ops)?;
                for r in 0..n {
                    load_shared(&f, &ps, l, &g, 1, &mut shs[r], &grp.cx[r].ops)?;
                }
                let key = if lw.engram.is_some() {
                    format!("{ps}/engram/x")
                } else {
                    format!("{ps}/in/x")
                };
                let mut tr = Trace::default();
                let mut trs: Vec<Trace> = (0..n).map(|_| Trace::default()).collect();
                let h = f.dev(&cx.ops, &key)?;
                let hs: Vec<Buf> = (0..n)
                    .map(|r| f.dev(&grp.cx[r].ops, &key))
                    .collect::<Result<_, _>>()?;
                if lw.engram.is_some() {
                    let mut ext = ids.clone();
                    ext.extend_from_slice(&gen_ids[..step]);
                    let all =
                        engram_hashes(&f, token_map, &ext, l, &format!("{ps}/engram/hash"), rep)?;
                    let last = all[all.len() - g.engram_cols()..].to_vec();
                    cx.engram(&stm, &lw, &h, &last, 1, &mut tr)?;
                    for r in 0..n {
                        grp.cx[r].engram(&stm, &lws[r], &hs[r], &last, 1, &mut trs[r])?;
                    }
                }
                let pre = f.dev(&cx.ops, &format!("{ps}/in/pre_mix"))?;
                let pres: Vec<Buf> = (0..n)
                    .map(|r| f.dev(&grp.cx[r].ops, &format!("{ps}/in/pre_mix")))
                    .collect::<Result<_, _>>()?;
                let (h2, p2) =
                    cx.block_forward(&lw, &h, &pre, pos, 1, &mut sh, &mut ls, &mut tr)?;
                let (h2s, p2s) =
                    grp.block_forward(&lws, &hs, &pres, pos, 1, &mut shs, &mut lss, &mut trs)?;
                tp_compare(
                    rep,
                    &format!("layer{l}/decode{step}"),
                    &cx,
                    &grp,
                    (&h2, &p2, &tr, &sh, &ls),
                    (&h2s, &p2s, &trs, &shs, &lss),
                )?;
                if grp.graphs.on && graph_can {
                    // the graph door: the same step again with nothing traced (a traced step
                    // reads on the host and is never recorded). The step writes its state at
                    // its position, so a repeat rewrites the same bytes: the first repeat runs
                    // launch by launch, the second records the layer's second half and replays
                    // it self-checked, the third and fourth replay it; each is compared whole.
                    drop((h2s, p2s));
                    grp.trace = false;
                    for k in 0..GRAPH_REPEATS {
                        let mut tg: Vec<Trace> = (0..n).map(|_| Trace::default()).collect();
                        let (hg, pg) = grp
                            .block_forward(&lws, &hs, &pres, pos, 1, &mut shs, &mut lss, &mut tg)?;
                        tp_compare(
                            rep,
                            &format!("layer{l}/decode{step}/graph{k}"),
                            &cx,
                            &grp,
                            (&h2, &p2, &tr, &sh, &ls),
                            (&hg, &pg, &tg, &shs, &lss),
                        )?;
                    }
                    grp.trace = true;
                    graph_steps += 1;
                }
                dec.push(pos.to_string());
            }
        }
        summary.push(format!(
            "{{\"layer\": {l}, \"single_card_s\": {ref_s:.3}, \"tp_s\": {tp_s:.3}, \"gathers\": {}, \"gather_bytes\": {}, \"decode_positions\": [{}]}}",
            st.collectives,
            st.bytes,
            dec.join(", ")
        ));
    }
    if grp.graphs.on && graph_can {
        // not vacuous: every gated decode step replayed its layer's captured half at least
        // twice (a step whose inputs moved runs launch by launch and is counted in `moved`)
        let gs = grp.graphs.stats;
        rep.raw(
            "graph/replayed",
            graph_steps > 0 && gs.replays >= 2 * graph_steps && gs.checks > 0,
            "MEMRA_DSV41_DECODE_GRAPH=1: the decode steps ran through the captured spans",
            format!(
                "{graph_steps} steps x {GRAPH_REPEATS} repeats: {} captures, {} replays ({} self-checked), {} moved, {} refused",
                gs.captures, gs.replays, gs.checks, gs.moved, gs.refused
            ),
        );
    }
    Ok(format!(
        "\"tokens\": {s}, \"ranks\": {n}, \"devices\": {devices:?}, \"transport\": {:?}, \"layers\": [{}]",
        grp.transport.name(),
        summary.join(", ")
    ))
}

/// Repeats of a gated decode step under the graph door (see the loop in [`tp_gate`]).
const GRAPH_REPEATS: usize = 4;

/// `dsv41-gate idx`: the indexer choice at N ranks on a captured prefill layer. Arms: the model's
/// replicated indexer on one rank (the reference), the same indexer with the query rows split
/// across ranks plus a gather of the selection, and the head split (each rank scores its index
/// heads into fp32 partials, the partials gathered, added, rounded, masked) plus the top-k on
/// every rank. Reports time per arm (`--reps` runs, median), bytes moved and selection agreement.
// allow: the rank loops walk several per-rank vectors in step
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn idx_gate(
    ckpt: &Path,
    cap: &Path,
    layer: usize,
    devices: &[usize],
    slice: usize,
    reps: usize,
    rows_cap: Option<usize>,
    last: Option<usize>,
    rep: &mut Report,
) -> Result<String, String> {
    let mc = memra_gguf::config::ModelConfig::from_config_json(&ckpt.join("config.json"))
        .map_err(|e| e.to_string())?;
    let g = Geo::from_config(&mc)?;
    if !g.index_sources.contains(&layer) {
        return Err(format!("layer {layer} is not an index source"));
    }
    let stm = memra_gguf::safetensors::StModel::open(ckpt).map_err(|e| e.to_string())?;
    let mut f = Fx::open(cap)?;
    f.ckpt = ckpt.to_path_buf();
    let s_full = f.i64s("prompt_ids")?.len();
    let s = rows_cap.unwrap_or(s_full).min(s_full);
    let ratio = g.compress_ratios[layer];
    let mut grp = v::tp::Group::new(devices, &g, (s_full + 64).max(1024))?;
    let n = grp.n();
    let lws = grp.load_layer(&stm, &g, layer, false)?;
    if lws[0].uses_cand {
        return Err(format!(
            "layer {layer} rescoring inside a candidate pool: not benched"
        ));
    }
    let pfx = format!("s0/L{layer}");
    // the attention input from the capture: attn_norm(hc_pre(stream, pre_mix)), then qr; index
    // keys: the capture's (a Reindex layer) or this layer's own from its latent (a kv source)
    let mut xs = Vec::new();
    let mut qrs = Vec::new();
    let mut iks = Vec::new();
    let mut n_c = 0;
    for r in 0..n {
        let c = &mut grp.cx[r];
        let st = c.ops.st.clone();
        let h = f.dev(&c.ops, &format!("{pfx}/in/x"))?;
        let pre = f.dev(&c.ops, &format!("{pfx}/in/pre_mix"))?;
        let mut tr = Trace::default();
        if lws[r].engram.is_some() {
            return Err("an Engram layer is not an index bench layer".into());
        }
        let x = c.hc_pre_norm(&h, &pre, &lws[r].attn_norm, s_full)?;
        let qr = c.ops.buf(s_full * g.q_lora * 2)?;
        c.ops.linear_q(
            x.p(&st),
            g.dim,
            s_full,
            &lws[r].wq_a,
            qr.p(&st),
            g.q_lora,
            false,
            &mut c.scr,
        )?;
        c.ops.rmsnorm(
            qr.p(&st),
            g.q_lora,
            &lws[r].q_norm,
            qr.p(&st),
            g.q_lora,
            s_full,
            g.q_lora,
            g.norm_eps,
        )?;
        let mut sh = Shared::default();
        load_shared(&f, &pfx, layer, &g, s_full, &mut sh, &c.ops)?;
        if sh.index_k.is_none() {
            // a kv source: run the layer once to produce its keys (one rank's replicated state)
            let mut ls = LayerState::default();
            c.keys_only(&lws[r], &x, s_full, &mut sh, &mut ls, &mut tr)?;
        }
        n_c = s / ratio;
        iks.push(sh.index_k.take().ok_or("no index keys")?);
        xs.push(x);
        qrs.push(qr);
    }
    let tw = g.index_topk.min(n_c);
    // `--last N`: only the last N rows (N = 1 is one decode-shaped row over every key)
    let first = last.map(|k| s.saturating_sub(k)).unwrap_or(0);
    let med = |v: &mut Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    let (mut t_rep, mut t_rows, mut t_split, mut t_gather) = (vec![], vec![], vec![], vec![]);
    let (mut s_rep, mut s_rows, mut s_split) = (vec![], vec![], vec![]);
    for it in 0..reps.max(1) + 1 {
        let (a, ta) = grp.cx[0]
            .index_replicated_bench(&lws[0], &xs[0], &qrs[0], &iks[0], n_c, s, slice, first)?;
        let (b, tb) = grp.index_rows_bench(&lws, &xs, &qrs, &iks, n_c, s, slice, first)?;
        grp.stats = Default::default();
        grp.timed = true;
        let (c, tc) =
            grp.index_split_bench(&lws, &xs, &qrs, &iks, n_c, s, ratio, tw, slice, first)?;
        grp.timed = false;
        if it > 0 {
            t_gather.push(grp.stats.secs);
        }
        if it > 0 {
            t_rep.push(ta);
            t_rows.push(tb);
            t_split.push(tc);
        }
        s_rep = a;
        s_rows = b;
        s_split = c;
    }
    let split_bytes = grp.stats.bytes;
    let rows_bytes = ((s - first) * tw * 4 * (n - 1)) as u64;
    rep.raw(
        &format!("layer{layer}/index/rows_split_selection"),
        s_rows == s_rep,
        "row split = replicated indexer, byte for byte",
        format!("{} selection words", s_rep.len()),
    );
    // head split vs replicated: rows whose selected set is identical, and the mean Jaccard
    let (mut same_rows, mut jac) = (0usize, 0f64);
    for t in first..s {
        let mut a: Vec<i32> = s_rep[t * tw..(t + 1) * tw]
            .iter()
            .copied()
            .filter(|&x| x >= 0)
            .collect();
        let mut b: Vec<i32> = s_split[t * tw..(t + 1) * tw]
            .iter()
            .copied()
            .filter(|&x| x >= 0)
            .collect();
        a.sort_unstable();
        b.sort_unstable();
        if a == b {
            same_rows += 1;
        }
        let inter = a.iter().filter(|x| b.binary_search(x).is_ok()).count();
        let uni = a.len() + b.len() - inter;
        jac += if uni == 0 {
            1.0
        } else {
            inter as f64 / uni as f64
        };
    }
    let (tr_, tro, ts, tg) = (
        med(&mut t_rep),
        med(&mut t_rows),
        med(&mut t_split),
        med(&mut t_gather),
    );
    rep.raw(
        &format!("layer{layer}/index/head_split_selection"),
        true,
        "reported: the head split adds the head sum in another order, so its bf16 score and top-k can move",
        format!(
            "{same_rows}/{} rows with the identical set, mean Jaccard {:.6}",
            s - first,
            jac / (s - first).max(1) as f64
        ),
    );
    rep.raw(
        &format!("layer{layer}/index/time"),
        true,
        "reported: median of the timed runs (first run is warm-up), host wall time with device syncs",
        format!(
            "replicated {:.2} ms, rows split {:.2} ms ({} B gathered), head split {:.2} ms ({} B gathered in {:.2} ms, {:.1} GB/s), {} rows, {n_c} keys, {n} ranks, {}",
            tr_ * 1e3,
            tro * 1e3,
            rows_bytes,
            ts * 1e3,
            split_bytes,
            tg * 1e3,
            split_bytes as f64 / tg.max(1e-12) / 1e9,
            s - first,
            grp.transport.name()
        ),
    );
    Ok(format!(
        "\"layer\": {layer}, \"rows\": {}, \"first_row\": {first}, \"keys\": {n_c}, \"ranks\": {n}, \"devices\": {devices:?}, \"transport\": {:?}, \"slice\": {slice}, \"ms\": {{\"replicated\": {:.3}, \"rows_split\": {:.3}, \"head_split\": {:.3}}}, \"head_split_gather_ms\": {:.3}, \"bytes\": {{\"rows_split\": {rows_bytes}, \"head_split\": {split_bytes}}}, \"head_split_same_rows\": {same_rows}, \"head_split_mean_jaccard\": {:.6}",
        s - first,
        grp.transport.name(),
        tr_ * 1e3,
        tro * 1e3,
        ts * 1e3,
        tg * 1e3,
        jac / (s - first).max(1) as f64
    ))
}

fn main() {
    let mut a: Vec<String> = std::env::args().collect();
    // --dense-form F, anywhere on the line: the form of every dense FP8/FP4 GEMM of the run
    // (`dsv41_gpu::GEMM_FORM`; 4 runs the small-row body at every call of at most 16 rows, so a card
    // whose default keeps CUTLASS at those widths gates the body in place). The bytes do not depend on it.
    if let Some(i) = a.iter().position(|x| x == "--dense-form") {
        let f: Vec<String> = a.drain(i..(i + 2).min(a.len())).collect();
        match f.get(1).and_then(|x| x.parse::<i32>().ok()) {
            Some(form) => v::GEMM_FORM.store(form, std::sync::atomic::Ordering::Relaxed),
            None => {
                eprintln!("--dense-form needs a form number");
                std::process::exit(2);
            }
        }
    }
    let flag = |k: &str| {
        a.iter()
            .position(|x| x == k)
            .and_then(|i| a.get(i + 1))
            .cloned()
    };
    let has = |k: &str| a.iter().any(|x| x == k);
    let mut rep = Report::default();
    let r = match a.get(1).map(|s| s.as_str()) {
        Some("kernels") if a.len() > 2 => kernels(Path::new(&a[2]), &mut rep).map(|_| format!("\"gate\": \"kernels\", \"fixtures\": {:?}", a[2])),
        Some("layer") if a.len() > 3 => {
            let layers: Vec<usize> = flag("--layers").unwrap_or_default().split(',').filter_map(|x| x.parse().ok()).collect();
            let chain = flag("--chain").and_then(|x| x.parse().ok()).unwrap_or(1);
            let tol = flag("--tol").and_then(|x| x.parse().ok()).unwrap_or(0.02);
            let token_map = flag("--token-map").map(PathBuf::from);
            let floor = flag("--floor").map(PathBuf::from);
            let decode = flag("--decode").and_then(|x| x.parse().ok()).unwrap_or(0);
            let save = flag("--save").map(PathBuf::from);
            layer_gate(Path::new(&a[2]), Path::new(&a[3]), &layers, has("--mint"), chain, token_map.as_deref(), tol, floor.as_deref(), decode, save.as_deref(), has("--head"), &mut rep)
                .map(|s| format!("\"gate\": \"layer\", \"capture\": {:?}, {s}", a[3]))
        }
        Some("decode") if a.len() > 3 => {
            let layers: Vec<usize> = flag("--layers").unwrap_or_default().split(',').filter_map(|x| x.parse().ok()).collect();
            match flag("--token-map") {
                Some(tm) => decode_gate(Path::new(&a[2]), Path::new(&a[3]), &layers, Path::new(&tm), &mut rep)
                    .map(|s| format!("\"gate\": \"decode\", \"capture\": {:?}, {s}", a[3])),
                None => Err("decode: --token-map FILE (the Engram hash of each new token)".into()),
            }
        }
        Some("tp") if a.len() > 3 => {
            let layers: Vec<usize> = flag("--layers").unwrap_or_default().split(',').filter_map(|x| x.parse().ok()).collect();
            let devices: Vec<usize> = flag("--devices").unwrap_or_else(|| "0,1".into()).split(',').filter_map(|x| x.parse().ok()).collect();
            let token_map = flag("--token-map").map(PathBuf::from);
            let decode = flag("--decode").and_then(|x| x.parse().ok()).unwrap_or(0);
            let chunk = flag("--chunk").and_then(|x| x.parse().ok());
            tp_gate(Path::new(&a[2]), Path::new(&a[3]), &layers, &devices, token_map.as_deref(), decode, chunk, &mut rep)
                .map(|s| format!("\"gate\": \"tp\", \"capture\": {:?}, {s}", a[3]))
        }
        Some("idx") if a.len() > 3 => {
            let layer = flag("--layer").and_then(|x| x.parse().ok()).unwrap_or(20);
            let devices: Vec<usize> = flag("--devices").unwrap_or_else(|| "0,1".into()).split(',').filter_map(|x| x.parse().ok()).collect();
            let slice = flag("--slice").and_then(|x| x.parse().ok()).unwrap_or(2048);
            let reps = flag("--reps").and_then(|x| x.parse().ok()).unwrap_or(3);
            let rows = flag("--rows").and_then(|x| x.parse().ok());
            let last = flag("--last").and_then(|x| x.parse().ok());
            idx_gate(Path::new(&a[2]), Path::new(&a[3]), layer, &devices, slice, reps, rows, last, &mut rep)
                .map(|s| format!("\"gate\": \"idx\", \"capture\": {:?}, {s}", a[3]))
        }
        _ => Err("usage: dsv41-gate [--dense-form F] kernels FIXTURE_ROOT [--out R.json] | layer CKPT CAPTURE --layers L,.. [--mint] [--chain N] [--tol X] [--token-map FILE] [--floor CPU_FLOOR_DIR] [--decode N] [--save DIR] [--head] [--chunk N] [--out R.json] | decode CKPT CAPTURE --layers L,.. --token-map FILE [--out R.json] | tp CKPT CAPTURE --layers L,.. [--devices 0,1] [--token-map FILE] [--decode N] [--chunk N] [--out R.json] | idx CKPT CAPTURE --layer L [--devices 0,1] [--slice N] [--reps N] [--rows N] [--last N] [--out R.json]".into()),
    };
    let header = match r {
        Ok(h) => h,
        Err(e) => {
            eprintln!("dsv41-gate: {e}");
            rep.raw("run", false, "the gate ran to completion", e.clone());
            format!("\"error\": {e:?}")
        }
    };
    let js = rep.json(&header);
    if let Some(o) = flag("--out")
        && let Err(e) = std::fs::write(&o, &js)
    {
        eprintln!("write {o}: {e}");
    }
    let fails = rep.checks.iter().filter(|c| !c.pass).count();
    println!("dsv41-gate: {} checks, {fails} failed", rep.checks.len());
    std::process::exit(if fails == 0 { 0 } else { 1 });
}
