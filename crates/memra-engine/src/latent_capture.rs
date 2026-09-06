//! Single-invocation operand capture and replay. No model loading or quantizer tuning.
//! Captures fixed indices so codec/reader differences cannot be confused with routing changes.
use crate::{Engine, hybrid::HybridModel, latent_nvfp4_ffi::Nvfp4LatentOps};
use cudarc::driver::CudaSlice;
use memra_kv::latent_nvfp4::{DevicePlane, PackedRow};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::ffi::CString;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::os::fd::{AsRawFd, FromRawFd};
use std::path::{Component, Path, PathBuf};
use std::sync::{
    OnceLock,
    atomic::{AtomicUsize, Ordering},
};

type Res<T> = Result<T, Box<dyn std::error::Error>>;
const MIN_VISIBLE: usize = 217_744;
const SELECT: usize = 11;
const MAX_ELEMENTS: usize = 384 * 1024 * 1024; // all persisted f32/i32 planes, <=1.5 GiB
static CONFIG: OnceLock<Result<Option<PathBuf>, String>> = OnceLock::new();
static ELIGIBLE: AtomicUsize = AtomicUsize::new(0);

#[derive(Clone, Copy, Debug)]
pub(crate) struct Shape {
    pub t: usize,
    pub visible: usize,
    pub heads: usize,
    pub dn: usize,
    pub dv: usize,
    pub rank: usize,
    pub slots: usize,
}
impl Shape {
    fn counts(self) -> Res<[usize; 6]> {
        if self.t == 0
            || self.t > 4096
            || self.visible < MIN_VISIBLE
            || self.visible > 262_144
            || self.t > self.visible
            || self.heads == 0
            || self.heads > 128
            || self.dn == 0
            || self.dn > 256
            || self.dv == 0
            || self.dv > 256
            || self.rank != 512
            || self.slots == 0
            || self.slots > 16_384
            || !self.dn.is_multiple_of(4)
            || !self.dv.is_multiple_of(4)
        {
            return Err("capture geometry outside bounded NoPE rank512 diagnostic".into());
        }
        // Every factor is bounded above before multiplying.
        let n = [
            self.t * self.heads * self.dn,
            self.heads * self.rank * self.dn,
            self.heads * self.dv * self.rank,
            self.visible * self.rank,
            self.t * self.slots,
            self.t * self.heads * self.dv,
        ];
        if n.iter().sum::<usize>() > MAX_ELEMENTS {
            return Err("capture total exceeds 1.5 GiB".into());
        }
        Ok(n)
    }
    fn indices(self, idx: &[i32]) -> Res<()> {
        if idx.len() != self.t * self.slots {
            return Err("index extent mismatch".into());
        }
        let start = self.visible - self.t;
        for (q, row) in idx.chunks_exact(self.slots).enumerate() {
            if !row.iter().any(|x| *x >= 0) {
                return Err("query has no selected history".into());
            }
            if row
                .iter()
                .any(|x| *x < -1 || (*x >= 0 && *x as usize > start + q))
            {
                return Err(format!("noncausal or invalid selection at query {q}").into());
            }
        }
        Ok(())
    }
}

/// Descriptor-relative no-follow traversal: reject symlinks and '..' in every component.
struct Directory(File);
impl Directory {
    fn open(path: &Path, create: bool) -> Res<Self> {
        if !path.is_absolute() {
            return Err("diagnostic directory must be absolute".into());
        }
        let parts: Vec<_> = path.components().collect();
        if parts.len() < 2
            || parts
                .iter()
                .skip(1)
                .any(|c| !matches!(c, Component::Normal(_)))
        {
            return Err("diagnostic path needs normal components and a non-root leaf".into());
        }
        let mut dir = File::open("/")?;
        for (i, part) in parts.iter().enumerate().skip(1) {
            use std::os::unix::ffi::OsStrExt;
            let Component::Normal(name) = part else {
                unreachable!()
            };
            let name = CString::new(name.as_bytes())?;
            if create && i == parts.len() - 1 {
                // mkdirat is exclusive; existing directories are not adopted.
                if unsafe { libc::mkdirat(dir.as_raw_fd(), name.as_ptr(), 0o700) } != 0 {
                    return Err(std::io::Error::last_os_error().into());
                }
            }
            let fd = unsafe {
                libc::openat(
                    dir.as_raw_fd(),
                    name.as_ptr(),
                    libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
                )
            };
            if fd < 0 {
                return Err(std::io::Error::last_os_error().into());
            }
            dir = unsafe { File::from_raw_fd(fd) };
        }
        Ok(Self(dir))
    }
    fn file(&self, name: &str, write: bool) -> Res<File> {
        if name.is_empty() || name.contains('/') || name == "." || name == ".." {
            return Err("invalid capture member".into());
        }
        let name = CString::new(name)?;
        let flags = libc::O_NOFOLLOW
            | libc::O_CLOEXEC
            | if write {
                libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL
            } else {
                libc::O_RDONLY | libc::O_NONBLOCK
            };
        let fd = unsafe { libc::openat(self.0.as_raw_fd(), name.as_ptr(), flags, 0o600) };
        if fd < 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        let f = unsafe { File::from_raw_fd(fd) };
        if !f.metadata()?.is_file() {
            return Err("capture member is not a regular file".into());
        }
        Ok(f)
    }
    fn text(&self, name: &str, value: &str) -> Res<()> {
        let mut f = self.file(name, true)?;
        f.write_all(value.as_bytes())?;
        f.sync_all()?;
        self.0.sync_all()?;
        Ok(())
    }
    fn words(&self, name: &str, words: impl Iterator<Item = u32>, count: usize) -> Res<String> {
        let mut f = BufWriter::new(self.file(name, true)?);
        let mut h = Sha256::new();
        let mut seen = 0;
        for word in words {
            let b = word.to_le_bytes();
            f.write_all(&b)?;
            h.update(b);
            seen += 1;
        }
        if seen != count {
            return Err("capture writer count mismatch".into());
        }
        f.flush()?;
        f.get_ref().sync_all()?;
        Ok(format!("{name} {count} {:x}\n", h.finalize()))
    }
    fn read_words(&self, name: &str, count: usize, hash: &str) -> Res<Vec<u32>> {
        if count > MAX_ELEMENTS {
            return Err("read exceeds capture bound".into());
        }
        let f = self.file(name, false)?;
        if f.metadata()?.len() != (count as u64) * 4 {
            return Err("capture member size mismatch".into());
        }
        let mut f = BufReader::new(f);
        let mut h = Sha256::new();
        let mut v = Vec::with_capacity(count);
        for _ in 0..count {
            let mut b = [0; 4];
            f.read_exact(&mut b)?;
            h.update(b);
            v.push(u32::from_le_bytes(b));
        }
        if format!("{:x}", h.finalize()) != hash {
            return Err("capture member SHA256 mismatch".into());
        }
        Ok(v)
    }
}
const NAMES: [&str; 6] = [
    "q_nope.f32",
    "wk_b.f32",
    "wv_b.f32",
    "latent.f32",
    "idx.i32",
    "original.f32",
];
fn finite(v: &[f32]) -> Res<()> {
    if v.iter().any(|v| !v.is_finite()) {
        return Err("nonfinite captured/replayed operand".into());
    }
    Ok(())
}
fn executable_hash() -> Res<String> {
    let mut f = File::open(std::env::current_exe()?)?;
    let mut h = Sha256::new();
    let mut b = [0; 65536];
    loop {
        let n = f.read(&mut b)?;
        if n == 0 {
            break;
        }
        h.update(&b[..n]);
    }
    Ok(format!("{:x}", h.finalize()))
}

#[allow(clippy::too_many_arguments)] // direct private TC call-site operands
pub(crate) fn capture(
    e: &Engine,
    is_f32: bool,
    wk: &CudaSlice<f32>,
    wv: &CudaSlice<f32>,
    q: &CudaSlice<f32>,
    latent: &CudaSlice<f32>,
    idx: &CudaSlice<i32>,
    original: &CudaSlice<f32>,
    s: Shape,
    scale: f32,
) -> Res<()> {
    let config = CONFIG.get_or_init(|| match std::env::var_os("MEMRA_LATENT_CAPTURE_DIR") {
        None => Ok(None),
        Some(v) if v.is_empty() => Err("MEMRA_LATENT_CAPTURE_DIR cannot be empty".into()),
        Some(v) => Ok(Some(PathBuf::from(v))),
    });
    let path = match config {
        Ok(Some(p)) => p,
        Ok(None) => return Ok(()),
        Err(e) => return Err(e.clone().into()),
    };
    if !is_f32 || s.visible < MIN_VISIBLE {
        return Ok(());
    }
    let ordinal = ELIGIBLE.fetch_add(1, Ordering::Relaxed) + 1;
    if ordinal != SELECT {
        return Ok(());
    }
    let counts = s.counts()?;
    if !scale.is_finite()
        || scale <= 0.
        || [
            q.len(),
            wk.len(),
            wv.len(),
            counts[3],
            idx.len(),
            original.len(),
        ] != counts
        || latent.len() < counts[3]
    {
        return Err("capture operand extent/scale mismatch".into());
    }
    let dir = Directory::open(path, true)?;
    let mut manifest = format!(
        "latent-capture-v1\nshape {} {} {} {} {} {} {}\nscale_bits {}\nordinal {ordinal}\ndevice {}\nbinary_sha256 {}\n",
        s.t,
        s.visible,
        s.heads,
        s.dn,
        s.dv,
        s.rank,
        s.slots,
        scale.to_bits(),
        e.ctx().ordinal(),
        executable_hash()?
    );
    // One host readback at a time; never copy unused latent capacity.
    for (i, d) in [(0, q), (1, wk), (2, wv), (3, latent), (5, original)] {
        let v = e.dtoh_view(&d.slice(..counts[i]))?;
        finite(&v)?;
        manifest.push_str(&dir.words(NAMES[i], v.iter().map(|x| x.to_bits()), v.len())?);
    }
    let indices = e.dtoh_i32(idx)?;
    s.indices(&indices)?;
    manifest.push_str(&dir.words(NAMES[4], indices.iter().map(|x| *x as u32), indices.len())?);
    dir.text("manifest.txt", &manifest)?; // completion marker, only after durable operands
    eprintln!(
        "[latent-capture] complete ordinal={ordinal} visible={} t={} path={}",
        s.visible,
        s.t,
        path.display()
    );
    Ok(())
}

struct Inputs {
    shape: Shape,
    device: usize,
    scale: f32,
    q: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    latent: Vec<f32>,
    idx: Vec<i32>,
    original: Vec<f32>,
    manifest: String,
}
fn read(path: &Path) -> Res<Inputs> {
    let dir = Directory::open(path, false)?;
    let f = dir.file("manifest.txt", false)?;
    if f.metadata()?.len() > 8192 {
        return Err("oversized capture manifest".into());
    }
    let mut text = String::new();
    f.take(8193).read_to_string(&mut text)?;
    let mut lines = text.lines();
    if lines.next() != Some("latent-capture-v1") {
        return Err("unsupported capture version".into());
    }
    let mut fields = BTreeMap::new();
    for line in lines {
        let (k, v) = line.split_once(' ').ok_or("invalid capture line")?;
        if fields.insert(k, v).is_some() {
            return Err("duplicate capture field".into());
        }
    }
    if fields.len() != 11 {
        return Err("unexpected capture field census".into());
    }
    let dims: Vec<usize> = fields
        .get("shape")
        .ok_or("missing shape")?
        .split_whitespace()
        .map(str::parse)
        .collect::<Result<_, _>>()?;
    if dims.len() != 7 {
        return Err("invalid shape".into());
    }
    let s = Shape {
        t: dims[0],
        visible: dims[1],
        heads: dims[2],
        dn: dims[3],
        dv: dims[4],
        rank: dims[5],
        slots: dims[6],
    };
    let counts = s.counts()?;
    let scale = f32::from_bits(fields.get("scale_bits").ok_or("missing scale")?.parse()?);
    if !scale.is_finite() || scale <= 0. || fields.get("ordinal") != Some(&"11") {
        return Err("invalid capture selection/scale".into());
    }
    let device: usize = fields
        .get("device")
        .ok_or("missing captured device")?
        .parse()?;
    let binary = fields
        .get("binary_sha256")
        .ok_or("missing captured binary")?;
    if binary.len() != 64 || !binary.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("invalid captured binary SHA256".into());
    }
    let mut planes = Vec::new();
    for (i, name) in NAMES.iter().enumerate() {
        let desc: Vec<_> = fields
            .get(name)
            .ok_or("missing plane")?
            .split_whitespace()
            .collect();
        if desc.len() != 2 || desc[0].parse::<usize>()? != counts[i] {
            return Err("plane descriptor mismatch".into());
        }
        planes.push(dir.read_words(name, counts[i], desc[1])?);
    }
    let mut p = planes.into_iter();
    let floats = |words: Vec<u32>| -> Res<Vec<f32>> {
        let v = words.into_iter().map(f32::from_bits).collect::<Vec<_>>();
        finite(&v)?;
        Ok(v)
    };
    let q = floats(p.next().unwrap())?;
    let wk = floats(p.next().unwrap())?;
    let wv = floats(p.next().unwrap())?;
    let latent = floats(p.next().unwrap())?;
    let idx = p
        .next()
        .unwrap()
        .into_iter()
        .map(|x| x as i32)
        .collect::<Vec<_>>();
    s.indices(&idx)?;
    Ok(Inputs {
        shape: s,
        device,
        scale,
        q,
        wk,
        wv,
        latent,
        idx,
        original: floats(p.next().unwrap())?,
        manifest: text,
    })
}
fn compare(name: &str, a: &[f32], b: &[f32], report: &mut String) -> Res<bool> {
    if a.len() != b.len() {
        return Err("comparison extent mismatch".into());
    }
    finite(a)?;
    finite(b)?;
    let mut bad = 0usize;
    let mut max = 0f64;
    let mut sq = 0f64;
    for (a, b) in a.iter().zip(b) {
        bad += usize::from(a.to_bits() != b.to_bits());
        let d = *a as f64 - *b as f64;
        max = max.max(d.abs());
        sq += d * d;
    }
    report.push_str(&format!(
        "{name} elements={} bit_mismatches={bad} max_abs={max:.9e} rms={:.9e}\n",
        a.len(),
        (sq / a.len().max(1) as f64).sqrt()
    ));
    Ok(bad == 0)
}

/// Standalone diagnostic entry point. Validation happens before creating a CUDA engine.
pub fn check(input: &Path, result: &Path, device: usize) -> Res<()> {
    if std::env::var_os("MEMRA_LATENT_CAPTURE_DIR").is_some() {
        return Err("unset capture knob for checker".into());
    }
    let data = read(input)?;
    if device != data.device {
        return Err(format!(
            "checker device {device} differs from captured ordinal {}",
            data.device
        )
        .into());
    }
    let dir = Directory::open(result, true)?;
    dir.text("input-manifest.txt", &data.manifest)?;
    let s = data.shape;
    let e = Engine::new(device)?;
    let wk = e.htod(&data.wk)?;
    let wv = e.htod(&data.wv)?;
    let q = e.htod(&data.q)?;
    let idx = e.htod_i32(&data.idx)?;
    let original = e.htod(&data.latent)?;
    let mut quant = DevicePlane::new(&e, s.rank, s.visible)?;
    // Append in bounded chunks including the true prefill start boundary and final row.
    let start = s.visible - s.t;
    let mut pos = 0;
    while pos < s.visible {
        let limit = if pos < start {
            start
        } else if pos < s.visible - 1 {
            s.visible - 1
        } else {
            s.visible
        };
        let end = (pos + 4096).min(limit);
        let rows = e.htod(&data.latent[pos * s.rank..end * s.rank])?;
        quant.append(&e, &rows, pos)?;
        quant.check(&e)?;
        pos = end;
    }
    let packed = e.dtoh_u8(&quant.payload)?;
    let scales = e.dtoh_u8(&quant.scales)?;
    let macros = e.dtoh(&quant.macros)?;
    let mut decoded = Vec::with_capacity(data.latent.len());
    let mut codec_bad = 0usize;
    for (row, values) in data.latent.chunks_exact(s.rank).enumerate() {
        let cpu = PackedRow::encode(values)?;
        if packed[row * 256..(row + 1) * 256] != cpu.payload
            || scales[row * 32..(row + 1) * 32] != cpu.block_scales
            || macros[row].to_bits() != cpu.macro_scale.to_bits()
        {
            codec_bad += 1;
        }
        decoded.extend(cpu.decode()?);
    }
    let mut report = format!(
        "latent-capture-check-v1\nchecker_binary_sha256 {}\ndevice {}\ncodec_mismatched_rows {codec_bad}\n",
        executable_hash()?,
        device
    );
    let mut passed = codec_bad == 0;
    let deq = e.htod(&decoded)?;
    // Gather selected indices in small batches: never allocate t*slots*rank history.
    let mut gather_bad = 0usize;
    let mut selections = data.idx.clone();
    selections.extend([
        -1,
        0,
        (start.saturating_sub(1)) as i32,
        start as i32,
        (s.visible - 1) as i32,
    ]);
    selections.sort_unstable();
    selections.dedup();
    for positions in selections.chunks(256) {
        let ids = e.htod_i32(positions)?;
        let actual = quant.gather(&e, &ids, s.visible)?;
        let actual = e.dtoh(&actual)?;
        quant.check(&e)?;
        for (i, pos) in positions.iter().enumerate() {
            for c in 0..s.rank {
                let want = if *pos < 0 {
                    0.
                } else {
                    decoded[*pos as usize * s.rank + c]
                };
                gather_bad += usize::from(actual[i * s.rank + c].to_bits() != want.to_bits());
            }
        }
    }
    report.push_str(&format!("gather_bit_mismatches {gather_bad}\n"));
    passed &= gather_bad == 0;
    let bf_native = quant.bf16_history(&e, s.visible)?;
    let bf_cpu = e.f32_to_bf16(&deq, decoded.len())?;
    let bf_bad = e
        .dtoh_u8(&bf_native)?
        .iter()
        .zip(e.dtoh_u8(&bf_cpu)?)
        .filter(|(a, b)| **a != *b)
        .count();
    quant.check(&e)?;
    passed &= bf_bad == 0;
    report.push_str(&format!("bf16_byte_mismatches {bf_bad}\n"));
    drop(bf_native);
    drop(bf_cpu);
    let tc = |latent: &CudaSlice<f32>, compressed: Option<&mut DevicePlane>| -> Res<Vec<f32>> {
        let out = HybridModel::mla_tc_prefill_chain(
            &e, &wk, &wv, &q, latent, compressed, &idx, s.slots, s.t, s.visible, s.heads, s.dn,
            s.dv, s.rank, data.scale,
        )?
        .ok_or("TC chain declined captured geometry")?;
        e.dtoh(&out)
    };
    let golden = tc(&original, None)?;
    passed &= compare("original_replay", &data.original, &golden, &mut report)?;
    let cpu_tc = tc(&deq, None)?;
    compare("quantization_effect_same_TC", &golden, &cpu_tc, &mut report)?; // difference expected, not a gate
    drop(golden);
    let empty = e.uninit(0)?;
    let native_tc = tc(&empty, Some(&mut quant))?;
    quant.check(&e)?;
    passed &= compare("native_vs_CPU_dequant_TC", &cpu_tc, &native_tc, &mut report)?;
    drop(cpu_tc);
    drop(native_tc);
    // TC replay above retains the full shape. Scalar readers compare at most
    // three original query/index rows, sharing one absorbed-query operand.
    let mut query_ids = vec![0, s.t / 2, s.t - 1];
    query_ids.dedup();
    let direct_t = query_ids.len();
    report.push_str(&format!(
        "direct_original_t {}\ndirect_query_ids {:?}\n",
        s.t, query_ids
    ));
    let q_width = s.heads * s.dn;
    let mut queries = Vec::with_capacity(direct_t * q_width);
    let mut indices = Vec::with_capacity(direct_t * s.slots);
    for &i in &query_ids {
        queries.extend_from_slice(&data.q[i * q_width..(i + 1) * q_width]);
        indices.extend_from_slice(&data.idx[i * s.slots..(i + 1) * s.slots]);
    }
    let direct_q = e.htod(&queries)?;
    let direct_idx = e.htod_i32(&indices)?;
    let mut q_lat = e.uninit(direct_t * s.heads * s.rank)?;
    e.mla_absorb_q(&direct_q, &wk, &mut q_lat, direct_t, s.heads, s.dn, s.rank)?;
    let native = quant.attend(
        &e,
        &q_lat,
        &direct_idx,
        s.heads,
        direct_t,
        s.slots,
        s.visible,
        data.scale,
    )?;
    quant.check(&e)?;
    let mut cpu = e.uninit(direct_t * s.heads * s.rank)?;
    e.mla_attn_gathered(
        &q_lat,
        &empty,
        &deq,
        &direct_idx,
        &mut cpu,
        s.heads,
        s.rank,
        0,
        direct_t,
        s.slots,
        data.scale,
    )?;
    passed &= compare(
        "native_direct_vs_CPU_dequant_reader",
        &e.dtoh(&cpu)?,
        &e.dtoh(&native)?,
        &mut report,
    )?;
    report.push_str(&format!(
        "passed {passed}\nquantization_effect_is_not_NLL_or_quality_verdict true\n"
    ));
    dir.text("result.txt", &report)?;
    print!("{report}");
    if !passed {
        return Err("captured operand replay mismatch; inspect result.txt".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Scratch(PathBuf);
    impl Scratch {
        fn new() -> Self {
            use std::os::unix::fs::DirBuilderExt;
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            let base = std::env::temp_dir().canonicalize().unwrap();
            loop {
                let path = base.join(format!(
                    "latent-capture-cpu-{}-{}",
                    std::process::id(),
                    NEXT.fetch_add(1, Ordering::Relaxed)
                ));
                match std::fs::DirBuilder::new().mode(0o700).create(&path) {
                    Ok(()) => return Self(path),
                    Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                    Err(e) => panic!("scratch directory: {e}"),
                }
            }
        }
    }
    impl Drop for Scratch {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.0).unwrap();
        }
    }

    #[test]
    fn private_exclusive_directory_and_members() {
        use std::os::unix::fs::PermissionsExt;
        let root = Scratch::new();
        let path = root.0.join("capture");
        let dir = Directory::open(&path, true).unwrap();
        assert_eq!(
            std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o700
        );
        dir.text("receipt", "original").unwrap();
        assert_eq!(
            std::fs::metadata(path.join("receipt"))
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o600
        );
        assert!(Directory::open(&path, true).is_err());
        assert!(dir.text("receipt", "replacement").is_err());
        assert_eq!(
            std::fs::read_to_string(path.join("receipt")).unwrap(),
            "original"
        );
        for name in ["", ".", "..", "../escape", "nested/file"] {
            assert!(dir.file(name, true).is_err());
        }
        assert!(Directory::open(Path::new("relative"), true).is_err());
        assert!(Directory::open(Path::new("/"), true).is_err());
        assert!(Directory::open(&root.0.join("missing/leaf"), true).is_err());
        assert!(Directory::open(&path.join("../escape"), true).is_err());
    }

    #[test]
    fn rejects_symlink_parents_leaves_and_members() {
        use std::os::unix::fs::symlink;
        let root = Scratch::new();
        let real = root.0.join("real");
        let dir = Directory::open(&real, true).unwrap();
        dir.text("data", "original").unwrap();
        let alias = root.0.join("alias");
        symlink(&real, &alias).unwrap();
        assert!(Directory::open(&alias, false).is_err());
        assert!(Directory::open(&alias.join("new"), true).is_err());
        symlink(real.join("data"), real.join("link")).unwrap();
        assert!(dir.file("link", false).is_err());
        assert!(dir.file("link", true).is_err());
        symlink(real.join("missing"), real.join("dangling")).unwrap();
        assert!(dir.file("dangling", false).is_err());
        assert!(dir.file("dangling", true).is_err());
        std::fs::create_dir(real.join("subdir")).unwrap();
        assert!(dir.file("subdir", false).is_err());
    }

    #[test]
    fn words_require_exact_count_size_and_hash() {
        let root = Scratch::new();
        let dir = Directory::open(&root.0.join("capture"), true).unwrap();
        let words = [0, 1, u32::MAX, 1f32.to_bits()];
        let desc = dir.words("plane", words.into_iter(), words.len()).unwrap();
        let hash = desc.split_whitespace().last().unwrap();
        assert_eq!(dir.read_words("plane", words.len(), hash).unwrap(), words);
        assert!(dir.read_words("plane", words.len() - 1, hash).is_err());
        assert!(
            dir.read_words("plane", words.len(), &"0".repeat(64))
                .is_err()
        );
        assert!(dir.read_words("plane", usize::MAX, hash).is_err());
        assert!(dir.words("partial", [0].into_iter(), 2).is_err());
        assert!(dir.file("manifest.txt", false).is_err());
    }

    #[test]
    fn rejects_manifest_identity_before_any_plane_read() {
        let root = Scratch::new();
        for (i, (device, binary)) in [
            ("bad", "a".repeat(64)),
            ("0", "bad".into()),
            ("0", "g".repeat(64)),
        ]
        .into_iter()
        .enumerate()
        {
            let path = root.0.join(format!("capture-{i}"));
            let dir = Directory::open(&path, true).unwrap();
            let mut text = format!(
                "latent-capture-v1\nshape 2 {MIN_VISIBLE} 64 192 128 512 2\nscale_bits {}\nordinal 11\ndevice {device}\nbinary_sha256 {binary}\n",
                1f32.to_bits()
            );
            for name in NAMES {
                text.push_str(&format!("{name} 0 {}\n", "0".repeat(64)));
            }
            dir.text("manifest.txt", &text).unwrap();
            let error = read(&path).err().expect("bad identity must fail");
            assert!(!error.to_string().contains("plane descriptor"), "{error}");
        }
    }

    #[test]
    fn comparison_requires_finite_operands_and_preserves_signed_zero() {
        let mut report = String::new();
        assert!(compare("same", &[1.], &[1.], &mut report).unwrap());
        assert!(!compare("zero", &[0.], &[-0.], &mut report).unwrap());
        assert!(compare("nan", &[f32::NAN], &[0.], &mut report).is_err());
        assert!(compare("size", &[1.], &[], &mut report).is_err());
    }

    #[test]
    fn bounds_and_causality() {
        let s = Shape {
            t: 2,
            visible: MIN_VISIBLE,
            heads: 64,
            dn: 192,
            dv: 128,
            rank: 512,
            slots: 2,
        };
        assert!(s.counts().is_ok());
        assert!(
            s.indices(&[0, (MIN_VISIBLE - 2) as i32, -1, (MIN_VISIBLE - 1) as i32])
                .is_ok()
        );
        assert!(s.indices(&[0, (MIN_VISIBLE - 1) as i32, 0, 1]).is_err());
        assert!(s.indices(&[-1, -1, 0, 1]).is_err());
        assert!(
            Shape {
                visible: usize::MAX,
                ..s
            }
            .counts()
            .is_err()
        );
        assert!(Shape { rank: 511, ..s }.counts().is_err());
    }
}
