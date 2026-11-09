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
//!                    [--engram-consts DIR] [--floor CPU_FLOOR_DIR] [--out RECEIPT.json]

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
        let s = ops.buf(m * k / 32)?;
        ops.act_quant(x.p(&st), k, Dp::NULL, m, k, q.p(&st), s.p(&st))?;
        let y = ops.buf(m * k * 2)?;
        v::copy_d2d(&ops, &x, 0, &y, 0, m * k * 2)?;
        ops.fp8_qdq(y.p(&st), k, m, k)?;
        let qh = q.host(&st)?;
        let sh_ = s.host(&st)?;
        let wq = f.bytes("q")?;
        let ws = f.bytes("s")?;
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
            "bit-exact ue8m0 scales",
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
        let w = QW {
            w: f.dev(&ops, "w")?,
            s: f.dev(&ops, "ws")?,
            n,
            k,
            fmt: QFmt::Fp8Block32,
        };
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
                let w = QW {
                    w: f.dev(&ops, &format!("L{lid}/wq_b"))?,
                    s: f.dev(&ops, &format!("L{lid}/wq_b_s"))?,
                    n: 4096,
                    k: 1280,
                    fmt: QFmt::Fp8Block32,
                };
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
        rep.f32rel(
            "moe/route_w",
            &f32h(&ops, &selw)?,
            &f.f32s("route_w")?,
            // the fp32 router GEMM's summation order over 5120 inputs: ~1e-6 relative
            1e-5,
        );
        let ex = |tag: &str, i: usize, n: usize, k: usize| -> Result<QW, String> {
            let w = f.dev(&ops, &format!("{tag}/w{i}"))?;
            let s = f.dev(&ops, &format!("{tag}/w{i}_s"))?;
            let fmt = if f.dtype(&format!("{tag}/w{i}")) == "f8e4m3" {
                QFmt::Fp8Block32
            } else {
                QFmt::Mxfp4
            };
            Ok(QW { w, s, n, k, fmt })
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
        // the minted NVFP4 form of the same expert must give the same bits
        let mint = |w: &QW| -> Result<QW, String> {
            let (sc, s2) = v::mint_mxfp4_scales(&w.s.host(&st)?)?;
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
        let wkv = QW {
            w: f.dev(&ops, "wkv")?,
            s: f.dev(&ops, "wkv_s")?,
            n: 25600,
            k: 6144,
            fmt: QFmt::Fp8Block32,
        };
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
    if l > g.cand_source && f.has(&k("candidates")) && f.shape(&k("candidates"))[1] == rows {
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
    }
    Ok(())
}

/// Engram hash ids `[tokens][cols]` of layer `li` for `ids`: the native host hash when the vendor
/// constants are given (checked against the capture's ids at `want_key`), else the capture's.
fn engram_hashes(
    f: &Fx,
    consts: Option<&Path>,
    ids: &[u32],
    li: usize,
    want_key: &str,
    rep: &mut Report,
) -> Result<Vec<i64>, String> {
    let Some(cpath) = consts else {
        return f.i64s(want_key);
    };
    let ec = v::engram::EngramConsts::load(
        &cpath.join("engram_constants.json"),
        &cpath.join("engram_constants.token_map.i32"),
    )?;
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
    consts: Option<&Path>,
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
            let all = engram_hashes(f, consts, &ext, li, &hk, rep)?;
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
    consts: Option<&Path>,
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
    let f = Fx::open(cap)?;
    let ids: Vec<u32> = f
        .i64s("prompt_ids")?
        .into_iter()
        .map(|x| x as u32)
        .collect();
    let s = ids.len();
    let mut cx = Ctx::new(st.clone(), g.clone(), (s + 64).max(1024))?;
    cx.trace = true;
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
                let hashes = engram_hashes(&f, consts, &ids, li, &format!("{p}/engram/hash"), rep)?;
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
                    consts,
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

fn main() {
    let a: Vec<String> = std::env::args().collect();
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
            let consts = flag("--engram-consts").map(PathBuf::from);
            let floor = flag("--floor").map(PathBuf::from);
            let decode = flag("--decode").and_then(|x| x.parse().ok()).unwrap_or(0);
            let save = flag("--save").map(PathBuf::from);
            layer_gate(Path::new(&a[2]), Path::new(&a[3]), &layers, has("--mint"), chain, consts.as_deref(), tol, floor.as_deref(), decode, save.as_deref(), has("--head"), &mut rep)
                .map(|s| format!("\"gate\": \"layer\", \"capture\": {:?}, {s}", a[3]))
        }
        _ => Err("usage: dsv41-gate kernels FIXTURE_ROOT [--out R.json] | layer CKPT CAPTURE --layers L,.. [--mint] [--chain N] [--tol X] [--engram-consts DIR] [--floor CPU_FLOOR_DIR] [--decode N] [--save DIR] [--head] [--out R.json]".into()),
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
