//! Opt-in native CPU backend for Hy3 routed experts.
//!
//! `MEMRA_CPU_EXPERT_LIB=/path/libmemra-cpu-experts.so` dynamically loads memra's stable v2 C ABI,
//! implemented by `tools/memra_cpu_experts.cpp`. The companion owns its packed-format decoders,
//! activation quantizer, AVX2/AVX-VNNI dots, cache, and positioned-I/O path; no external inference
//! runtime is linked or loaded. It consumes original host-resident GGUF bytes and returns one f32
//! hidden state contribution to CUDA.

use std::ffi::{CStr, CString, c_char, c_void};
use std::os::unix::ffi::OsStrExt as _;
use std::os::unix::io::AsRawFd as _;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, SyncSender};

use crate::hybrid::MoeWeights;
use crate::model::{ExpertKeepalive, ExpertSource};
use crate::spill_pread::DiskReadSource;
use crate::{
    QT_BF16, QT_F32, QT_IQ3_S, QT_IQ4_XS, QT_NVFP4, QT_Q2_K, QT_Q3_K, QT_Q4_0, QT_Q4_K, QT_Q5_K,
    QT_Q6_K, QT_Q8_0,
};
use memra_gguf::bound_disk::{
    BoundDiskView, BoundReadMode, FileGeneration,
    ffi::{ScopedDiskInfoV1, ScopedDiskReaderHandle, ScopedDiskReaderV1},
};

#[repr(C)]
#[derive(Clone, Copy)]
struct CpuProjectionV2 {
    weights: *const u8,
    qtype: i32,
    in_features: i32,
    out_features: i32,
    row_bytes: usize,
    byte_len: usize,
    file_fd: i32,
    file_offset: u64,
    scale: f32,
}

#[repr(C)]
struct CpuExpertV2 {
    gate: CpuProjectionV2,
    up: CpuProjectionV2,
    down: CpuProjectionV2,
    route_weight: f32,
}

#[derive(Clone)]
pub(crate) struct OwnedProjection {
    weights: usize,
    qtype: i32,
    in_features: i32,
    out_features: i32,
    row_bytes: usize,
    byte_len: usize,
    file: Option<std::sync::Arc<std::fs::File>>,
    file_offset: u64,
    scoped_reader: Option<std::sync::Arc<ScopedProjectionReaders>>,
    scale: f32,
}

#[derive(Clone)]
struct OwnedExpert {
    gate: OwnedProjection,
    up: OwnedProjection,
    down: OwnedProjection,
    route_weight: f32,
}

/// Owned request sent to the persistent CPU worker thread. Weight addresses point into immutable
/// `HostExps` stores. The caller must join the worker before the borrowed `MoeWeights` can be
/// dropped. `CpuExpertTicket::drop` enforces that lifetime even when the GPU path returns early.
pub(crate) struct CpuExpertJob {
    experts: Vec<OwnedExpert>,
    input: Vec<f32>,
    output_features: usize,
    threads: i32,
}

/// Lane-3 M3: one EXPERT evaluated for several stream rows in a single companion call
/// (memra_cpu_expert_rows_v2 — weight decode amortized across rows). Output layout:
/// [m_r, n_embd] per-row contributions with route weights applied, or (`raw`) the bare
/// down-projection rows for [`accumulate_expert_exact`] (memra_cpu_expert_rows_raw_v2).
pub(crate) struct CpuRowsJob {
    expert: OwnedExpert,
    inputs: Vec<f32>,
    route_weights: Vec<f32>,
    output_features: usize,
    threads: i32,
    raw: bool,
}

pub(crate) enum CpuJob {
    Token(CpuExpertJob),
    Rows(CpuRowsJob),
}

type AbiVersionFn = unsafe extern "C" fn() -> u32;
type MoeTokenFn = unsafe extern "C" fn(
    *const CpuExpertV2,
    i32,
    *const f32,
    *mut f32,
    i32,
    *mut c_char,
    usize,
) -> i32;
type CacheStatsFn = unsafe extern "C" fn(*mut u64, *mut u64, *mut u64, *mut u64);
type ProfileStatsFn = unsafe extern "C" fn(*mut u64, *mut u64, *mut u64, *mut u64);
type PrefetchFn = unsafe extern "C" fn(*const CpuProjectionV2, i32, *mut i8, usize) -> i32;
type RowsFn = unsafe extern "C" fn(
    *const CpuExpertV2,
    *const f32,
    i32,
    *const f32,
    *mut f32,
    i32,
    *mut c_char,
    usize,
) -> i32;
type RowsRawFn = unsafe extern "C" fn(
    *const CpuExpertV2,
    *const f32,
    i32,
    *mut f32,
    i32,
    *mut c_char,
    usize,
) -> i32;

type ScopedMoeTokenFn = unsafe extern "C" fn(
    *const CpuExpertV2,
    i32,
    *const f32,
    *mut f32,
    i32,
    *mut c_char,
    usize,
    *const *const ScopedDiskReaderV1,
) -> i32;
type ScopedRowsFn = unsafe extern "C" fn(
    *const CpuExpertV2,
    *const f32,
    i32,
    *const f32,
    *mut f32,
    i32,
    *mut c_char,
    usize,
    *const *const ScopedDiskReaderV1,
) -> i32;
type ScopedRowsRawFn = unsafe extern "C" fn(
    *const CpuExpertV2,
    *const f32,
    i32,
    *mut f32,
    i32,
    *mut c_char,
    usize,
    *const *const ScopedDiskReaderV1,
) -> i32;
type MirrorRowsRawFn = unsafe extern "C" fn(
    *const CpuExpertV2,
    *const f32,
    i32,
    *mut f32,
    i32,
    *mut c_char,
    usize,
    *const *const ScopedDiskReaderV1,
    *const *const ScopedDiskReaderV1,
) -> i32;

type ScopedPrefetchFn = unsafe extern "C" fn(
    *const CpuProjectionV2,
    i32,
    *mut c_char,
    usize,
    *const *const ScopedDiskReaderV1,
) -> i32;

type MirrorSpecFn = unsafe extern "C" fn(
    *const ScopedDiskInfoV1,
    *mut ScopedDiskInfoV1,
    *mut c_char,
    usize,
    *mut usize,
    *mut c_char,
    usize,
) -> i32;
type MirrorTokenFn = unsafe extern "C" fn(
    *const CpuExpertV2,
    i32,
    *const f32,
    *mut f32,
    i32,
    *mut c_char,
    usize,
    *const *const ScopedDiskReaderV1,
    *const *const ScopedDiskReaderV1,
) -> i32;
type MirrorRowsFn = unsafe extern "C" fn(
    *const CpuExpertV2,
    *const f32,
    i32,
    *const f32,
    *mut f32,
    i32,
    *mut c_char,
    usize,
    *const *const ScopedDiskReaderV1,
    *const *const ScopedDiskReaderV1,
) -> i32;
type MirrorPrefetchFn = unsafe extern "C" fn(
    *const CpuProjectionV2,
    i32,
    *mut c_char,
    usize,
    *const *const ScopedDiskReaderV1,
    *const *const ScopedDiskReaderV1,
) -> i32;

struct CpuBackend {
    // Kept open for process lifetime so the native memra function pointers remain valid.
    _handle: usize,
    moe_token: MoeTokenFn,
    cache_stats: CacheStatsFn,
    profile_stats: ProfileStatsFn,
    // Optional (added after ABI v2 shipped): absent in older companions, prefetch disabled.
    prefetch: Option<PrefetchFn>,
    rows: Option<RowsFn>,
    rows_raw: Option<RowsRawFn>,
    scoped_token: Option<ScopedMoeTokenFn>,
    scoped_rows: Option<ScopedRowsFn>,
    scoped_rows_raw: Option<ScopedRowsRawFn>,
    scoped_prefetch: Option<ScopedPrefetchFn>,
    mirror_spec: Option<MirrorSpecFn>,
    mirror_token: Option<MirrorTokenFn>,
    mirror_rows: Option<MirrorRowsFn>,
    mirror_rows_raw: Option<MirrorRowsRawFn>,
    mirror_prefetch: Option<MirrorPrefetchFn>,
}

// The dlopen handle names process-global immutable code after initialization.
unsafe impl Send for CpuBackend {}
unsafe impl Sync for CpuBackend {}

static BACKEND: OnceLock<Result<CpuBackend, String>> = OnceLock::new();
static CALLS: AtomicU64 = AtomicU64::new(0);
static EXPERTS: AtomicU64 = AtomicU64::new(0);
static WALL_NS: AtomicU64 = AtomicU64::new(0);
static EXPOSED_WAIT_NS: AtomicU64 = AtomicU64::new(0);
static GPU_RESIDENT_0: AtomicU64 = AtomicU64::new(0);
static GPU_RESIDENT_1: AtomicU64 = AtomicU64::new(0);
static GPU_RESIDENT_2: AtomicU64 = AtomicU64::new(0);
struct CpuRequest {
    job: CpuJob,
    reply: SyncSender<Result<Vec<f32>, String>>,
}

struct CpuExecutor {
    sender: SyncSender<CpuRequest>,
}

pub(crate) struct CpuExpertTicket {
    receiver: Option<Receiver<Result<Vec<f32>, String>>>,
}

static EXECUTOR: OnceLock<Result<CpuExecutor, String>> = OnceLock::new();

pub(crate) fn configured() -> bool {
    std::env::var_os("MEMRA_CPU_EXPERT_LIB").is_some()
}

fn threads_from_env() -> Result<i32, String> {
    let value = match std::env::var("MEMRA_CPU_EXPERT_THREADS") {
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => None,
        Err(error) => return Err(format!("cannot read MEMRA_CPU_EXPERT_THREADS: {error}")),
    };
    parse_thread_count(value.as_deref())
}

fn parse_thread_count(value: Option<&str>) -> Result<i32, String> {
    let raw = value.unwrap_or("8");
    let threads = raw
        .parse::<i32>()
        .map_err(|_| format!("MEMRA_CPU_EXPERT_THREADS={raw:?} is not an integer"))?;
    if !(1..=256).contains(&threads) {
        return Err(format!(
            "MEMRA_CPU_EXPERT_THREADS={threads} is outside 1..=256"
        ));
    }
    Ok(threads)
}

fn dl_error(context: &str) -> String {
    // SAFETY: dlerror returns either null or a process-owned NUL-terminated diagnostic.
    let detail = unsafe {
        let pointer = libc::dlerror();
        if pointer.is_null() {
            "unknown dynamic-loader error".to_string()
        } else {
            CStr::from_ptr(pointer).to_string_lossy().into_owned()
        }
    };
    format!("{context}: {detail}")
}

fn load_symbol(handle: *mut c_void, name: &'static [u8]) -> Result<*mut c_void, String> {
    debug_assert_eq!(name.last(), Some(&0));
    // SAFETY: the name is statically NUL-terminated and the handle stays open for process life.
    unsafe {
        libc::dlerror();
    }
    let symbol = unsafe { libc::dlsym(handle, name.as_ptr().cast()) };
    if symbol.is_null() {
        Err(dl_error(&format!(
            "missing symbol {}",
            String::from_utf8_lossy(&name[..name.len() - 1])
        )))
    } else {
        Ok(symbol)
    }
}

fn load_backend_from_path(path: &std::ffi::OsStr) -> Result<CpuBackend, String> {
    let c_path = CString::new(path.as_bytes())
        .map_err(|_| "MEMRA_CPU_EXPERT_LIB contains a NUL byte".to_string())?;
    // SAFETY: c_path is NUL-terminated. The successful handle is intentionally never closed.
    let handle = unsafe { libc::dlopen(c_path.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
    if handle.is_null() {
        return Err(dl_error(&format!("cannot load {}", path.to_string_lossy())));
    }
    let result = (|| {
        let version_symbol = load_symbol(handle, b"memra_cpu_experts_abi_version\0")?;
        let token_symbol = load_symbol(handle, b"memra_cpu_moe_token_v2\0")?;
        let stats_symbol = load_symbol(handle, b"memra_cpu_expert_cache_stats_v2\0")?;
        let profile_symbol = load_symbol(handle, b"memra_cpu_expert_profile_stats_v2\0")?;
        // SAFETY: the companion library exports these exact v2 C signatures.
        let version: AbiVersionFn = unsafe { std::mem::transmute(version_symbol) };
        let abi = unsafe { version() };
        require_abi_v2(abi)?;
        let moe_token: MoeTokenFn = unsafe { std::mem::transmute(token_symbol) };
        let cache_stats: CacheStatsFn = unsafe { std::mem::transmute(stats_symbol) };
        let profile_stats: ProfileStatsFn = unsafe { std::mem::transmute(profile_symbol) };
        // Optional symbol: older companions predate speculative prefetch.
        let prefetch: Option<PrefetchFn> = load_symbol(handle, b"memra_cpu_expert_prefetch_v2\0")
            .ok()
            // SAFETY: when present the companion exports this exact v2 C signature.
            .map(|symbol| unsafe { std::mem::transmute::<*mut c_void, PrefetchFn>(symbol) });
        eprintln!(
            "[memra] experimental CPU expert backend: {} (threads={})",
            path.to_string_lossy(),
            threads_from_env()?,
        );
        let rows: Option<RowsFn> = load_symbol(handle, b"memra_cpu_expert_rows_v2\0")
            .ok()
            // SAFETY: when present the companion exports this exact rows ABI signature.
            .map(|symbol| unsafe { std::mem::transmute::<*mut c_void, RowsFn>(symbol) });
        let rows_raw: Option<RowsRawFn> = load_symbol(handle, b"memra_cpu_expert_rows_raw_v2\0")
            .ok()
            .map(|symbol| unsafe { std::mem::transmute::<*mut c_void, RowsRawFn>(symbol) });
        // Optional extension leaves existing ABI2 libraries usable for unbound sources.
        // Scoped sources require all three symbols; no raw-descriptor fallback is permitted.
        let scoped_token = load_symbol(handle, b"memra_cpu_moe_token_scoped_v1\0")
            .ok()
            .map(|symbol| unsafe { std::mem::transmute::<*mut c_void, ScopedMoeTokenFn>(symbol) });
        let scoped_rows = load_symbol(handle, b"memra_cpu_expert_rows_scoped_v1\0")
            .ok()
            .map(|symbol| unsafe { std::mem::transmute::<*mut c_void, ScopedRowsFn>(symbol) });
        let scoped_prefetch = load_symbol(handle, b"memra_cpu_expert_prefetch_scoped_v1\0")
            .ok()
            .map(|symbol| unsafe { std::mem::transmute::<*mut c_void, ScopedPrefetchFn>(symbol) });
        let mirror_spec = load_symbol(handle, b"memra_cpu_expert_mirror_spec_v1\0")
            .ok()
            .map(|s| unsafe { std::mem::transmute::<*mut c_void, MirrorSpecFn>(s) });
        let mirror_token = load_symbol(handle, b"memra_cpu_moe_token_scoped_mirrors_v1\0")
            .ok()
            .map(|s| unsafe { std::mem::transmute::<*mut c_void, MirrorTokenFn>(s) });
        let mirror_rows = load_symbol(handle, b"memra_cpu_expert_rows_scoped_mirrors_v1\0")
            .ok()
            .map(|s| unsafe { std::mem::transmute::<*mut c_void, MirrorRowsFn>(s) });
        let mirror_prefetch = load_symbol(handle, b"memra_cpu_expert_prefetch_scoped_mirrors_v1\0")
            .ok()
            .map(|s| unsafe { std::mem::transmute::<*mut c_void, MirrorPrefetchFn>(s) });
        let scoped_rows_raw = load_symbol(handle, b"memra_cpu_expert_rows_raw_scoped_v1\0")
            .ok()
            .map(|s| unsafe { std::mem::transmute::<*mut c_void, ScopedRowsRawFn>(s) });
        let mirror_rows_raw = load_symbol(handle, b"memra_cpu_expert_rows_raw_scoped_mirrors_v1\0")
            .ok()
            .map(|s| unsafe { std::mem::transmute::<*mut c_void, MirrorRowsRawFn>(s) });
        Ok(CpuBackend {
            _handle: handle as usize,
            moe_token,
            cache_stats,
            profile_stats,
            prefetch,
            rows,
            rows_raw,
            scoped_token,
            scoped_rows,
            scoped_rows_raw,
            scoped_prefetch,
            mirror_spec,
            mirror_token,
            mirror_rows,
            mirror_rows_raw,
            mirror_prefetch,
        })
    })();
    if result.is_err() {
        // SAFETY: no function pointer escapes the failed initialization path.
        unsafe {
            libc::dlclose(handle);
        }
    }
    result
}

fn require_abi_v2(abi: u32) -> Result<(), String> {
    if abi == 2 {
        Ok(())
    } else {
        Err(format!(
            "CPU expert ABI {abi} is incompatible; memra requires native v2"
        ))
    }
}

fn load_backend() -> Result<CpuBackend, String> {
    let path = std::env::var_os("MEMRA_CPU_EXPERT_LIB")
        .ok_or_else(|| "MEMRA_CPU_EXPERT_LIB is not set".to_string())?;
    load_backend_from_path(&path)
}

fn backend() -> Result<&'static CpuBackend, String> {
    match BACKEND.get_or_init(load_backend) {
        Ok(backend) => Ok(backend),
        Err(error) => Err(error.clone()),
    }
}

fn native_qtype(qtype: i32) -> Result<i32, String> {
    match qtype {
        QT_F32 | QT_Q4_0 | QT_Q8_0 | QT_Q2_K | QT_Q3_K | QT_Q4_K | QT_Q5_K | QT_Q6_K | QT_IQ3_S
        | QT_IQ4_XS | QT_BF16 | QT_NVFP4 => Ok(qtype),
        other => Err(format!(
            "native CPU expert backend does not support memra qtype {other}"
        )),
    }
}

struct ScopedProjectionReaders {
    primary: ScopedDiskReaderHandle,
    alternate: Option<ScopedDiskReaderHandle>,
}
impl ScopedProjectionReaders {
    fn as_abi(&self) -> ScopedDiskReaderV1 {
        self.primary.as_abi()
    }
    #[cfg(test)]
    fn lifetime(&self) -> memra_gguf::bound_disk::ffi::ScopedDiskReaderLifetime {
        self.primary.lifetime()
    }
}
fn scoped_reader(view: BoundDiskView) -> Result<std::sync::Arc<ScopedProjectionReaders>, String> {
    let backend = backend()?;
    if backend.scoped_token.is_none()
        || backend.scoped_rows.is_none()
        || backend.scoped_prefetch.is_none()
    {
        return Err("CPU companion lacks the complete scoped-reader extension; refusing raw-descriptor fallback".into());
    }
    let direct = match std::env::var_os("MEMRA_CPU_EXPERT_IO") {
        None => false,
        Some(value) if value == "buffered" => false,
        Some(value) if value == "direct" => view.direct_aligned(),
        Some(value) => {
            return Err(format!(
                "invalid MEMRA_CPU_EXPERT_IO={value:?}: expected buffered or direct"
            ));
        }
    };
    let mode = if direct {
        BoundReadMode::Direct
    } else {
        BoundReadMode::Buffered
    };
    let primary = view
        .reader(mode)
        .map_err(|e| e.to_string())?
        .into_ffi_handle();
    let alternate = if direct
        && std::env::var_os("MEMRA_CPU_EXPERT_MIRROR_MAP").is_some_and(|p| !p.is_empty())
    {
        if backend.mirror_token.is_none()
            || backend.mirror_rows.is_none()
            || backend.mirror_prefetch.is_none()
        {
            return Err("CPU companion lacks the verified scoped mirror extension".into());
        }
        let query = backend
            .mirror_spec
            .ok_or("CPU companion lacks mirror declaration query")?;
        let abi = primary.as_abi();
        let mut source = ScopedDiskInfoV1::default();
        let code = unsafe { (abi.info)(abi.context, &mut source) };
        if code != 0 {
            return Err(std::io::Error::from_raw_os_error(code).to_string());
        }
        let mut target = ScopedDiskInfoV1::default();
        let mut path = vec![0u8; 1024];
        let mut needed = 0;
        let mut error = vec![0i8; 1024];
        let mut status = unsafe {
            query(
                &source,
                &mut target,
                path.as_mut_ptr().cast(),
                path.len(),
                &mut needed,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status == 2 {
            if needed == 0 || needed > 4096 {
                return Err("scoped mirror path exceeds the supported OS path length".into());
            }
            path.resize(needed, 0);
            status = unsafe {
                query(
                    &source,
                    &mut target,
                    path.as_mut_ptr().cast(),
                    path.len(),
                    &mut needed,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
        }
        if status != 1 {
            *error.last_mut().unwrap() = 0;
            let detail = unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy();
            return Err(format!(
                "active scoped mirror declaration failed: status={status} {detail}"
            ));
        }
        let path = CStr::from_bytes_with_nul(
            path.get(..needed)
                .ok_or("invalid scoped mirror path length")?,
        )
        .map_err(|_| "scoped mirror path is not a single terminated OS path")?;
        let generation = |v: &ScopedDiskInfoV1| FileGeneration {
            device: v.device,
            inode: v.inode,
            file_bytes: v.file_bytes,
            ctime_seconds: v.ctime_seconds,
            ctime_nanoseconds: v.ctime_nanoseconds,
        };
        let path = std::path::Path::new(std::ffi::OsStr::from_bytes(path.to_bytes()));
        Some(
            view.verified_mirror_reader(path, generation(&source), generation(&target))
                .map_err(|e| format!("scoped mirror verification failed: {e}"))?
                .into_ffi_handle(),
        )
    } else {
        None
    };
    Ok(std::sync::Arc::new(ScopedProjectionReaders {
        primary,
        alternate,
    }))
}

struct ScopedArguments {
    abi: Vec<Option<ScopedDiskReaderV1>>,
    alternate: Vec<Option<ScopedDiskReaderV1>>,
}
impl ScopedArguments {
    fn new<'a, I>(projections: I) -> Self
    where
        I: IntoIterator<Item = &'a OwnedProjection>,
        I::IntoIter: Clone,
    {
        let projections = projections.into_iter();
        let alternate = if projections.clone().any(|p| {
            p.scoped_reader
                .as_ref()
                .is_some_and(|r| r.alternate.is_some())
        }) {
            projections
                .clone()
                .map(|p| {
                    p.scoped_reader
                        .as_ref()
                        .and_then(|r| r.alternate.as_ref())
                        .map(|r| r.as_abi())
                })
                .collect()
        } else {
            Vec::new()
        };
        let abi = if projections.clone().any(|p| p.scoped_reader.is_some()) {
            projections
                .map(|p| p.scoped_reader.as_ref().map(|r| r.as_abi()))
                .collect()
        } else {
            Vec::new() // no additional ABI arrays for the unchanged raw-source call
        };
        Self { abi, alternate }
    }
    fn needed(&self) -> bool {
        !self.abi.is_empty()
    }
    fn mirrored(&self) -> bool {
        !self.alternate.is_empty()
    }
    fn alternate_pointers(&self) -> Vec<*const ScopedDiskReaderV1> {
        self.alternate
            .iter()
            .map(|a| a.as_ref().map_or(std::ptr::null(), |a| a as *const _))
            .collect()
    }
    fn pointers(&self) -> Vec<*const ScopedDiskReaderV1> {
        self.abi
            .iter()
            .map(|a| a.as_ref().map_or(std::ptr::null(), |a| a as *const _))
            .collect()
    }
}

fn projection(exps: &crate::model::HostExps, expert: usize) -> Result<OwnedProjection, String> {
    let layout = exps.expert_layout(expert);
    let bytes = exps.expert_bytes(expert);
    if bytes.len() != layout.len || layout.len != layout.row_bytes * exps.out_f {
        return Err(format!(
            "expert {expert} extent mismatch: bytes={} layout={} rows={}x{}",
            bytes.len(),
            layout.len,
            exps.out_f,
            layout.row_bytes
        ));
    }
    let (file, file_offset, scoped_reader) = match exps.expert_source(expert) {
        ExpertSource::Disk {
            read: DiskReadSource::Unbound { file, offset },
            len,
            ..
        } => {
            if len != layout.len {
                return Err(format!(
                    "expert {expert} disk extent {len} differs from layout {}",
                    layout.len
                ));
            }
            (Some(file.clone()), offset, None)
        }
        ExpertSource::Disk {
            read: DiskReadSource::Bound(view),
            len,
            ..
        } => {
            if len != layout.len {
                return Err("scoped CPU expert extent differs from its layout".into());
            }
            (
                None,
                0,
                Some(scoped_reader(view).map_err(|e| format!("CPU expert {expert}: {e}"))?),
            )
        }

        ExpertSource::Memory {
            keepalive: Some(ExpertKeepalive::Pinned(_)),
            ..
        }
        | ExpertSource::Memory {
            keepalive: Some(ExpertKeepalive::Buffer(_)),
            ..
        } => {
            return Err(format!(
                "expert {expert} is in CUDA write-combined host memory; CPU reads are disabled"
            ));
        }
        ExpertSource::Memory { .. } => (None, 0, None),
    };
    Ok(OwnedProjection {
        weights: if scoped_reader.is_some() {
            0
        } else {
            bytes.as_ptr() as usize
        },
        qtype: native_qtype(layout.qtype)?,
        in_features: i32::try_from(exps.in_f)
            .map_err(|_| format!("expert input width {} exceeds i32", exps.in_f))?,
        out_features: i32::try_from(exps.out_f)
            .map_err(|_| format!("expert output width {} exceeds i32", exps.out_f))?,
        row_bytes: layout.row_bytes,
        byte_len: layout.len,
        file,
        file_offset,
        scoped_reader,
        scale: exps.macro_scale(expert),
    })
}

pub(crate) fn prepare_job(
    weights: &MoeWeights,
    _layer: u16,
    selected: &[(usize, f32)],
    input: &[f32],
) -> Result<CpuExpertJob, String> {
    if selected.is_empty() {
        return Err("cannot prepare an empty CPU expert job".to_string());
    }
    if input.len() != weights.gate_exps.in_f {
        return Err(format!(
            "CPU expert input has {} values, expected {}",
            input.len(),
            weights.gate_exps.in_f
        ));
    }
    let mut experts = Vec::with_capacity(selected.len());
    for &(expert, route_weight) in selected {
        if expert >= weights.gate_exps.n_expert {
            return Err(format!("CPU expert id {expert} is out of range"));
        }
        if weights
            .active_experts
            .as_ref()
            .is_some_and(|active| !active[expert])
        {
            return Err(format!("router selected pruned CPU expert id {expert}"));
        }
        experts.push(OwnedExpert {
            gate: projection(&weights.gate_exps, expert)?,
            up: projection(&weights.up_exps, expert)?,
            down: projection(&weights.down_exps, expert)?,
            route_weight,
        });
    }
    Ok(CpuExpertJob {
        experts,
        input: input.to_vec(),
        output_features: weights.down_exps.out_f,
        threads: threads_from_env()?,
    })
}

fn ffi_projection(value: &OwnedProjection) -> CpuProjectionV2 {
    CpuProjectionV2 {
        weights: value.weights as *const u8,
        qtype: value.qtype,
        in_features: value.in_features,
        out_features: value.out_features,
        row_bytes: value.row_bytes,
        byte_len: value.byte_len,
        file_fd: value.file.as_ref().map_or(-1, |file| file.as_raw_fd()),
        file_offset: value.file_offset,
        scale: value.scale,
    }
}

fn ffi_expert(expert: &OwnedExpert) -> CpuExpertV2 {
    CpuExpertV2 {
        gate: ffi_projection(&expert.gate),
        up: ffi_projection(&expert.up),
        down: ffi_projection(&expert.down),
        route_weight: expert.route_weight,
    }
}

fn execute(job: CpuJob) -> Result<Vec<f32>, String> {
    // A pinned executor sizes its OMP team to its own core group, overriding the job's
    // process-wide thread count (set at prepare time, before the executor is known).
    let width = EXECUTOR_THREADS.with(|slot| slot.get());
    let job = if width > 0 {
        match job {
            CpuJob::Token(mut j) => {
                j.threads = width;
                CpuJob::Token(j)
            }
            CpuJob::Rows(mut j) => {
                j.threads = width;
                CpuJob::Rows(j)
            }
        }
    } else {
        job
    };
    match job {
        CpuJob::Token(job) => execute_token(job),
        CpuJob::Rows(job) => execute_rows(job),
    }
}

fn execute_rows(job: CpuRowsJob) -> Result<Vec<f32>, String> {
    let backend = backend()?;
    let readers = ScopedArguments::new([&job.expert.gate, &job.expert.up, &job.expert.down]);
    let pointers = readers.pointers();
    let alternate = readers.alternate_pointers();
    let expert = ffi_expert(&job.expert);
    let m_r = job.route_weights.len();
    let mut output = vec![0.0f32; m_r * job.output_features];
    let mut error = vec![0i8; 1024];
    let start = std::time::Instant::now();
    // SAFETY: memory descriptors borrow immutable model storage until the caller joins;
    // scoped descriptors retain their reader in this job. ABI arrays outlive the call, and
    // the companion retains scoped contexts before any queued read. All spans match dimensions.
    let status = unsafe {
        if job.raw {
            if readers.mirrored() {
                let call = backend
                    .mirror_rows_raw
                    .ok_or("CPU companion lacks scoped mirror raw rows entrypoint")?;
                call(
                    &expert,
                    job.inputs.as_ptr(),
                    m_r as i32,
                    output.as_mut_ptr(),
                    job.threads,
                    error.as_mut_ptr(),
                    error.len(),
                    pointers.as_ptr(),
                    alternate.as_ptr(),
                )
            } else if readers.needed() {
                let call = backend
                    .scoped_rows_raw
                    .ok_or("CPU companion lacks scoped raw rows entrypoint")?;
                call(
                    &expert,
                    job.inputs.as_ptr(),
                    m_r as i32,
                    output.as_mut_ptr(),
                    job.threads,
                    error.as_mut_ptr(),
                    error.len(),
                    pointers.as_ptr(),
                )
            } else {
                let call = backend
                    .rows_raw
                    .ok_or("companion library lacks memra_cpu_expert_rows_raw_v2")?;
                call(
                    &expert,
                    job.inputs.as_ptr(),
                    m_r as i32,
                    output.as_mut_ptr(),
                    job.threads,
                    error.as_mut_ptr(),
                    error.len(),
                )
            }
        } else if readers.mirrored() {
            let call = backend
                .mirror_rows
                .ok_or("CPU companion lacks scoped mirror rows entrypoint")?;
            call(
                &expert,
                job.inputs.as_ptr(),
                m_r as i32,
                job.route_weights.as_ptr(),
                output.as_mut_ptr(),
                job.threads,
                error.as_mut_ptr(),
                error.len(),
                pointers.as_ptr(),
                alternate.as_ptr(),
            )
        } else if readers.needed() {
            let call = backend
                .scoped_rows
                .ok_or("CPU companion lacks scoped rows entrypoint")?;
            call(
                &expert,
                job.inputs.as_ptr(),
                m_r as i32,
                job.route_weights.as_ptr(),
                output.as_mut_ptr(),
                job.threads,
                error.as_mut_ptr(),
                error.len(),
                pointers.as_ptr(),
            )
        } else {
            let call = backend
                .rows
                .ok_or("companion library lacks memra_cpu_expert_rows_v2")?;
            call(
                &expert,
                job.inputs.as_ptr(),
                m_r as i32,
                job.route_weights.as_ptr(),
                output.as_mut_ptr(),
                job.threads,
                error.as_mut_ptr(),
                error.len(),
            )
        }
    };
    WALL_NS.fetch_add(
        start.elapsed().as_nanos().min(u64::MAX as u128) as u64,
        Ordering::Relaxed,
    );
    CALLS.fetch_add(1, Ordering::Relaxed);
    EXPERTS.fetch_add(m_r as u64, Ordering::Relaxed);
    if status != 0 {
        if let Some(last) = error.last_mut() {
            *last = 0;
        }
        // SAFETY: the fixed-size buffer contains at least the NUL written above.
        let message = unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy();
        return Err(format!("CPU expert rows backend failed: {message}"));
    }
    Ok(output)
}

fn execute_token(job: CpuExpertJob) -> Result<Vec<f32>, String> {
    let backend = backend()?;
    let experts: Vec<_> = job.experts.iter().map(ffi_expert).collect();
    let readers = ScopedArguments::new(job.experts.iter().flat_map(|e| [&e.gate, &e.up, &e.down]));
    let pointers = readers.pointers();
    let alternate = readers.alternate_pointers();
    let count =
        i32::try_from(experts.len()).map_err(|_| "CPU expert count exceeds i32".to_string())?;
    let mut output = vec![0.0f32; job.output_features];
    let mut error = vec![0i8; 1024];
    let start = std::time::Instant::now();
    // SAFETY: memory descriptors borrow immutable model storage until the caller joins;
    // scoped descriptors retain their reader in this job. Both ABI arrays outlive the call,
    // and the companion retains contexts before queued reads. Input/output/error spans match.
    let status = unsafe {
        if readers.mirrored() {
            let call = backend
                .mirror_token
                .ok_or("CPU companion lacks scoped mirror token entrypoint")?;
            call(
                experts.as_ptr(),
                count,
                job.input.as_ptr(),
                output.as_mut_ptr(),
                job.threads,
                error.as_mut_ptr(),
                error.len(),
                pointers.as_ptr(),
                alternate.as_ptr(),
            )
        } else if readers.needed() {
            let call = backend
                .scoped_token
                .ok_or("CPU companion lacks scoped token entrypoint")?;
            call(
                experts.as_ptr(),
                count,
                job.input.as_ptr(),
                output.as_mut_ptr(),
                job.threads,
                error.as_mut_ptr(),
                error.len(),
                pointers.as_ptr(),
            )
        } else {
            (backend.moe_token)(
                experts.as_ptr(),
                count,
                job.input.as_ptr(),
                output.as_mut_ptr(),
                job.threads,
                error.as_mut_ptr(),
                error.len(),
            )
        }
    };
    WALL_NS.fetch_add(
        start.elapsed().as_nanos().min(u64::MAX as u128) as u64,
        Ordering::Relaxed,
    );
    CALLS.fetch_add(1, Ordering::Relaxed);
    EXPERTS.fetch_add(experts.len() as u64, Ordering::Relaxed);
    if status != 0 {
        // Defend the Rust side even if a broken companion violates the ABI contract.
        if let Some(last) = error.last_mut() {
            *last = 0;
        }
        // SAFETY: the fixed-size buffer contains at least the NUL written above.
        let message = unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy();
        return Err(format!("CPU expert backend failed: {message}"));
    }
    Ok(output)
}

// Per-executor OMP width, set on the executor thread so `execute` can size its team to the
// core group that thread is pinned to (heterogeneous groups want different widths).
thread_local! {
    static EXECUTOR_THREADS: std::cell::Cell<i32> = const { std::cell::Cell::new(0) };
}

/// Parse `"0-7;8-15"` (semicolon-separated core lists, each `a-b` or `a,b,c`) into cpu sets.
fn parse_cpusets(spec: &str) -> Result<Vec<Vec<usize>>, String> {
    let mut groups = Vec::new();
    for group in spec.split(';').filter(|g| !g.trim().is_empty()) {
        let mut cpus = Vec::new();
        for part in group.split(',') {
            let part = part.trim();
            match part.split_once('-') {
                Some((lo, hi)) => {
                    let (lo, hi) = (
                        lo.trim()
                            .parse::<usize>()
                            .map_err(|_| format!("bad cpu range {part:?}"))?,
                        hi.trim()
                            .parse::<usize>()
                            .map_err(|_| format!("bad cpu range {part:?}"))?,
                    );
                    if lo > hi || hi >= 4096 {
                        return Err(format!("bad cpu range {part:?}"));
                    }
                    cpus.extend(lo..=hi);
                }
                None => cpus.push(
                    part.parse::<usize>()
                        .map_err(|_| format!("bad cpu id {part:?}"))?,
                ),
            }
        }
        if cpus.is_empty() {
            return Err(format!("empty cpu group in {spec:?}"));
        }
        groups.push(cpus);
    }
    if groups.is_empty() {
        return Err(format!("no cpu groups in {spec:?}"));
    }
    Ok(groups)
}

fn pin_current_thread(cpus: &[usize]) {
    // SAFETY: zeroed cpu_set_t then CPU_SET of validated ids; affinity applies to this thread.
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        for &cpu in cpus {
            libc::CPU_SET(cpu, &mut set);
        }
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
    }
}

fn start_executor() -> Result<CpuExecutor, String> {
    // MEMRA_CPU_EXPERT_EXECUTORS > 1 (lane-3 cross-stream overlap): several worker threads
    // drain one queue, so stream A's expert compute runs under stream B's reads. Callers
    // should split MEMRA_CPU_EXPERT_THREADS across executors (each companion call spawns its
    // own OMP team). Default 1 = the established serial behavior.
    //
    // MEMRA_CPU_EXPERT_EXECUTOR_CPUSETS="0-7;8-15" pins each executor to its own core group —
    // ASYMMETRIC partitioning, the shape the 2026-07-23 receipt identified as the only viable
    // way to use heterogeneous cores: one OMP team per group, so a slow group never straggles
    // a fast group's barrier (naive widening of ONE team across P+E cores measured
    // catastrophic: compute 2.8 -> 5.2 s at 16 threads, 14.6 s at 20). Executor count follows
    // the group count when set, and each team is sized to its group unless
    // MEMRA_CPU_EXPERT_EXECUTOR_THREADS="8;16" overrides per group.
    // NOTE: a global GOMP_CPU_AFFINITY overrides per-thread affinity for OMP teams — leave it
    // unset (and widen the process taskset) for the pinning to take effect.
    let cpusets = match std::env::var("MEMRA_CPU_EXPERT_EXECUTOR_CPUSETS") {
        Ok(spec) if !spec.trim().is_empty() => Some(parse_cpusets(&spec)?),
        _ => None,
    };
    let group_threads = match std::env::var("MEMRA_CPU_EXPERT_EXECUTOR_THREADS") {
        Ok(spec) if !spec.trim().is_empty() => Some(
            spec.split(';')
                .filter(|s| !s.trim().is_empty())
                .map(|s| {
                    s.trim()
                        .parse::<i32>()
                        .map_err(|_| format!("bad thread count {s:?}"))
                })
                .collect::<Result<Vec<_>, _>>()?,
        ),
        _ => None,
    };
    let executors = match &cpusets {
        Some(groups) => groups.len(),
        None => std::env::var("MEMRA_CPU_EXPERT_EXECUTORS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|&n| (1..=8).contains(&n))
            .unwrap_or(1),
    };
    if let Some(groups) = &cpusets {
        let widths: Vec<String> = groups
            .iter()
            .enumerate()
            .map(|(i, g)| {
                let t = group_threads
                    .as_ref()
                    .and_then(|v| v.get(i).copied())
                    .unwrap_or(g.len() as i32);
                format!("{}cpus/{}thr", g.len(), t)
            })
            .collect();
        eprintln!(
            "[memra] cpu expert executors pinned: {}",
            widths.join(" + ")
        );
    }
    let (sender, receiver) = std::sync::mpsc::sync_channel::<CpuRequest>(executors.max(1));
    let receiver = std::sync::Arc::new(std::sync::Mutex::new(receiver));
    for index in 0..executors {
        let receiver = std::sync::Arc::clone(&receiver);
        let cpus = cpusets.as_ref().map(|g| g[index].clone());
        let threads = group_threads
            .as_ref()
            .and_then(|v| v.get(index).copied())
            .or_else(|| cpus.as_ref().map(|c| c.len() as i32));
        std::thread::Builder::new()
            .name(format!("memra-cpu-executor-{index}"))
            .spawn(move || {
                if let Some(cpus) = &cpus {
                    pin_current_thread(cpus);
                }
                if let Some(threads) = threads {
                    EXECUTOR_THREADS.with(|slot| slot.set(threads));
                }
                loop {
                    let request = {
                        let guard = receiver.lock().expect("cpu executor queue poisoned");
                        guard.recv()
                    };
                    let Ok(request) = request else { return };
                    let result = execute(request.job);
                    let _ = request.reply.send(result);
                }
            })
            .map_err(|error| format!("cannot start persistent CPU expert executor: {error}"))?;
    }
    Ok(CpuExecutor { sender })
}

fn executor() -> Result<&'static CpuExecutor, String> {
    match EXECUTOR.get_or_init(start_executor) {
        Ok(executor) => Ok(executor),
        Err(error) => Err(error.clone()),
    }
}

pub(crate) fn submit(job: CpuExpertJob) -> Result<CpuExpertTicket, String> {
    submit_any(CpuJob::Token(job))
}

fn submit_any(job: CpuJob) -> Result<CpuExpertTicket, String> {
    let (reply, receiver) = std::sync::mpsc::sync_channel(1);
    executor()?
        .sender
        .send(CpuRequest { job, reply })
        .map_err(|_| "persistent CPU expert executor stopped".to_string())?;
    Ok(CpuExpertTicket {
        receiver: Some(receiver),
    })
}

/// Build a rows job: ONE expert, several (input-row, route-weight) pairs.
pub(crate) fn prepare_rows_job(
    weights: &MoeWeights,
    expert: usize,
    rows: &[(&[f32], f32)],
) -> Result<CpuRowsJob, String> {
    if rows.is_empty() || rows.len() > 64 {
        return Err("CPU rows job needs 1..=64 rows".to_string());
    }
    if expert >= weights.gate_exps.n_expert {
        return Err(format!("CPU rows expert id {expert} is out of range"));
    }
    if weights
        .active_experts
        .as_ref()
        .is_some_and(|active| !active[expert])
    {
        return Err(format!(
            "router selected pruned CPU rows expert id {expert}"
        ));
    }
    let n_embd = weights.gate_exps.in_f;
    let mut inputs = Vec::with_capacity(rows.len() * n_embd);
    let mut route_weights = Vec::with_capacity(rows.len());
    for (input, weight) in rows {
        if input.len() != n_embd {
            return Err("CPU rows input width mismatch".to_string());
        }
        inputs.extend_from_slice(input);
        route_weights.push(*weight);
    }
    Ok(CpuRowsJob {
        expert: OwnedExpert {
            gate: projection(&weights.gate_exps, expert)?,
            up: projection(&weights.up_exps, expert)?,
            down: projection(&weights.down_exps, expert)?,
            route_weight: 0.0,
        },
        inputs,
        route_weights,
        output_features: weights.down_exps.out_f,
        threads: threads_from_env()?,
        raw: false,
    })
}

/// The raw twin of [`prepare_rows_job`]: bare down-projection rows, no route weight and no down
/// scale applied, for [`accumulate_expert_exact`]. The job carries weight 1.0 per row only as a
/// placeholder; the caller folds the real weight in the accumulate step.
pub(crate) fn prepare_rows_raw_job(
    weights: &MoeWeights,
    expert: usize,
    rows: &[&[f32]],
) -> Result<CpuRowsJob, String> {
    let weighted: Vec<(&[f32], f32)> = rows.iter().map(|row| (*row, 1.0f32)).collect();
    let mut job = prepare_rows_job(weights, expert, &weighted)?;
    job.raw = true;
    let b = backend()?;
    let readers = ScopedArguments::new([&job.expert.gate, &job.expert.up, &job.expert.down]);
    if readers.mirrored() && b.mirror_rows_raw.is_none() {
        return Err("CPU companion lacks scoped mirror raw rows entrypoint".into());
    }
    if !readers.mirrored() && readers.needed() && b.scoped_rows_raw.is_none() {
        return Err("CPU companion lacks scoped raw rows entrypoint".into());
    }
    if !readers.needed() && b.rows_raw.is_none() {
        return Err("companion library lacks memra_cpu_expert_rows_raw_v2".into());
    }
    Ok(job)
}

/// True when the companion exports the raw rows entry point the exact lockstep program needs.
pub(crate) fn rows_raw_supported() -> bool {
    backend().is_ok_and(|b| b.rows_raw.is_some())
}

/// True when `expert` can take the rows kernel at all: the companion's rows path serves
/// quantized projections only (`memra_cpu_experts.cpp`, "CPU expert rows path serves quantized
/// experts only"), while the one-job program also takes F32 and BF16. A caller that would send
/// an unquantized expert down the rows path must fall back to the one-job program for that row
/// set instead of surfacing the companion's refusal mid-step.
pub(crate) fn rows_raw_admits(weights: &MoeWeights, expert: usize) -> bool {
    [&weights.gate_exps, &weights.up_exps, &weights.down_exps]
        .iter()
        .all(|exps| !matches!(exps.expert_layout(expert).qtype, QT_F32 | QT_BF16))
}

/// Submit an already prepared job; the persistent executor queue is bounded (one slot per
/// executor), so a dispatcher that has GPU work to launch should submit from a helper thread.
pub(crate) fn submit_job(job: CpuJob) -> Result<CpuExpertTicket, String> {
    submit_any(job)
}

/// The down-projection scale the companion applies for `expert`: the same field
/// [`prepare_job`] hands to `memra_cpu_moe_token_v2`, so an exact re-accumulation reads the
/// value the single-token program used, not a lookalike.
pub(crate) fn down_scale(weights: &MoeWeights, expert: usize) -> Result<f32, String> {
    Ok(projection(&weights.down_exps, expert)?.scale)
}

/// Fold one expert's raw down-projection row into a running sum EXACTLY as
/// `memra_cpu_moe_token_v2` does: `sum = fma(y, route_weight * down_scale, sum)`, the scale a
/// single f32 product, the fold a fused multiply-add, from a zero start, in job (selection)
/// order. `f32::mul_add` is the correctly rounded FMA, `std::fma`'s twin. Given bit-identical
/// per-expert rows (the multi-row kernel's per-row contract), a row accumulated this way over
/// its CPU experts in selection order is bit-identical to the one-job program, which is what
/// lets the lockstep multi-row arm share expert decodes across streams without changing a
/// stream's bytes (memra#577).
pub(crate) fn accumulate_expert_exact(
    sum: &mut [f32],
    y: &[f32],
    route_weight: f32,
    down_scale: f32,
) {
    debug_assert_eq!(sum.len(), y.len());
    let scale = route_weight * down_scale;
    for (acc, &v) in sum.iter_mut().zip(y) {
        *acc = v.mul_add(scale, *acc);
    }
}

pub(crate) fn record_incomplete_gpu_residency(resident_projections: usize) {
    match resident_projections {
        0 => GPU_RESIDENT_0.fetch_add(1, Ordering::Relaxed),
        1 => GPU_RESIDENT_1.fetch_add(1, Ordering::Relaxed),
        2 => GPU_RESIDENT_2.fetch_add(1, Ordering::Relaxed),
        _ => return,
    };
}

pub(crate) fn incomplete_gpu_residency_stats() -> (u64, u64, u64) {
    (
        GPU_RESIDENT_0.load(Ordering::Relaxed),
        GPU_RESIDENT_1.load(Ordering::Relaxed),
        GPU_RESIDENT_2.load(Ordering::Relaxed),
    )
}

impl CpuExpertTicket {
    pub(crate) fn wait(mut self) -> Result<Vec<f32>, String> {
        let start = std::time::Instant::now();
        let result = self
            .receiver
            .take()
            .expect("CPU expert ticket receiver is present until wait")
            .recv()
            .map_err(|_| "persistent CPU expert executor dropped a result".to_string())?;
        EXPOSED_WAIT_NS.fetch_add(
            start.elapsed().as_nanos().min(u64::MAX as u128) as u64,
            Ordering::Relaxed,
        );
        result
    }
}

impl Drop for CpuExpertTicket {
    fn drop(&mut self) {
        // Jobs borrow immutable model bytes through raw pointers. Every early-return path must
        // therefore wait until the persistent worker has stopped reading those bytes.
        if let Some(receiver) = self.receiver.take() {
            let _ = receiver.recv();
        }
    }
}

// ---- prediction-guided speculative prefetch (increment 2) --------------------------------
// Grounded in research/moe/expert-prefetch-prediction-pilot.md: applying layer j's router to
// layer k's MoE input predicts the deep half's routed experts at 84-100% argmax precision for
// j-k <= 4. A dedicated worker thread (never the decode thread) scores lookahead layers with
// host copies of the router weights, filters HBM-resident and pruned experts, and hands the
// predicted-and-missing projections to the companion's detached cold-insert prefetch.

struct PredictLayer {
    /// Router weights transposed to [n_expert][n_embd] for sequential dot products.
    router_t: Vec<f32>,
    bias: Option<Vec<f32>>,
    active: Option<Vec<bool>>,
    n_embd: usize,
    n_expert: usize,
    n_used: usize,
    sig: (f32, bool),
    /// Prebuilt per-expert projection descriptors (gate/up/down); None for pruned ids. The
    /// OwnedProjection keeps the backing file handles alive for the process lifetime.
    experts: Vec<Option<[OwnedProjection; 3]>>,
}

struct Predictor {
    sender: SyncSender<(u16, Vec<f32>)>,
    submitted: AtomicU64,
    dropped: AtomicU64,
}

static PREDICTOR: OnceLock<Option<Predictor>> = OnceLock::new();

fn prefetch_depth_from_env() -> usize {
    std::env::var("MEMRA_MOE_PREFETCH")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&depth| (1..=8).contains(&depth))
        .unwrap_or(0)
}

/// Build the predictor from per-layer inputs and start its worker. Called once from the model
/// after residency freeze (the resident snapshot must be final). `layers` carries, per MoE
/// layer index, everything the worker needs — fully owned, no model references escape.
pub(crate) fn start_prefetch_predictor(
    layers: Vec<(u16, PredictLayerInit)>,
    resident: std::collections::HashSet<(u16, u8, u16)>,
) -> Result<(), String> {
    let depth = prefetch_depth_from_env();
    if depth == 0 {
        return Err("MEMRA_MOE_PREFETCH is not enabled".to_string());
    }
    let backend = backend()?;
    if backend.prefetch.is_none() {
        return Err("companion library lacks memra_cpu_expert_prefetch_v2".to_string());
    }
    let top = std::env::var("MEMRA_MOE_PREFETCH_TOP")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&t| (1..=8).contains(&t))
        .unwrap_or(1);
    let min_layer = std::env::var("MEMRA_MOE_PREFETCH_MIN_LAYER")
        .ok()
        .and_then(|value| value.parse::<u16>().ok())
        .unwrap_or(40);
    let mut table: std::collections::HashMap<u16, PredictLayer> = Default::default();
    for (layer_index, init) in layers {
        let mut experts = Vec::with_capacity(init.weights_n_expert);
        for expert in 0..init.weights_n_expert {
            experts.push(init.build_expert(expert));
        }
        let n_embd = init.n_embd;
        let n_expert = init.weights_n_expert;
        // Transpose [n_embd, n_expert] -> [n_expert][n_embd].
        let mut router_t = vec![0.0f32; n_embd * n_expert];
        for row in 0..n_embd {
            for expert in 0..n_expert {
                router_t[expert * n_embd + row] = init.router[row * n_expert + expert];
            }
        }
        table.insert(
            layer_index,
            PredictLayer {
                router_t,
                bias: init.bias,
                active: init.active,
                n_embd,
                n_expert,
                n_used: init.n_used,
                sig: init.sig,
                experts,
            },
        );
    }
    let (sender, receiver) = std::sync::mpsc::sync_channel::<(u16, Vec<f32>)>(8);
    std::thread::Builder::new()
        .name("memra-moe-prefetch".to_string())
        .spawn(move || prefetch_worker(receiver, table, resident, depth, top, min_layer))
        .map_err(|error| format!("cannot spawn prefetch worker: {error}"))?;
    let created = PREDICTOR
        .set(Some(Predictor {
            sender,
            submitted: AtomicU64::new(0),
            dropped: AtomicU64::new(0),
        }))
        .is_ok();
    if !created {
        return Err("prefetch predictor already started".to_string());
    }
    eprintln!("[memra] moe prefetch predictor: depth={depth} top={top}");
    Ok(())
}

/// Per-layer construction inputs, fully owned.
pub(crate) struct PredictLayerInit {
    pub router: Vec<f32>,
    pub bias: Option<Vec<f32>>,
    pub active: Option<Vec<bool>>,
    pub n_embd: usize,
    pub n_used: usize,
    pub sig: (f32, bool),
    pub weights_n_expert: usize,
    pub gate: Vec<Option<crate::cpu_experts::OwnedProjection>>,
    pub up: Vec<Option<crate::cpu_experts::OwnedProjection>>,
    pub down: Vec<Option<crate::cpu_experts::OwnedProjection>>,
}

impl PredictLayerInit {
    fn build_expert(&self, expert: usize) -> Option<[OwnedProjection; 3]> {
        Some([
            self.gate[expert].clone()?,
            self.up[expert].clone()?,
            self.down[expert].clone()?,
        ])
    }
}

/// Predictor eligibility excludes pruned, CUDA write-combined and unsupported CPU encodings.
/// Binding, reader and descriptor errors still propagate instead of becoming optional absence.
pub(crate) fn predictor_projection(
    exps: &crate::model::HostExps,
    expert: usize,
) -> Result<Option<OwnedProjection>, String> {
    let layout = exps.expert_layout(expert);
    if layout.len == 0
        || native_qtype(layout.qtype).is_err()
        || matches!(
            exps.expert_source(expert),
            ExpertSource::Memory {
                keepalive: Some(ExpertKeepalive::Pinned(_) | ExpertKeepalive::Buffer(_)),
                ..
            }
        )
    {
        return Ok(None);
    }
    projection(exps, expert).map(Some)
}

/// Fire-and-forget: hand layer il's MoE input to the predictor. Never blocks the decode
/// thread — a full channel drops the sample (speculation must not backpressure decode).
pub(crate) fn predictor_submit(layer: u16, input: &[f32]) {
    let Some(Some(predictor)) = PREDICTOR.get().map(Option::as_ref) else {
        return;
    };
    match predictor.sender.try_send((layer, input.to_vec())) {
        Ok(()) => {
            predictor.submitted.fetch_add(1, Ordering::Relaxed);
        }
        Err(_) => {
            predictor.dropped.fetch_add(1, Ordering::Relaxed);
        }
    }
}

pub(crate) fn predictor_enabled() -> bool {
    matches!(PREDICTOR.get(), Some(Some(_)))
}

pub(crate) fn predictor_stats() -> (u64, u64) {
    match PREDICTOR.get().map(Option::as_ref) {
        Some(Some(p)) => (
            p.submitted.load(Ordering::Relaxed),
            p.dropped.load(Ordering::Relaxed),
        ),
        _ => (0, 0),
    }
}

fn submit_projection_prefetch(
    backend: &CpuBackend,
    projections: &[&OwnedProjection],
    error: &mut [i8],
) -> Result<i32, String> {
    if projections.is_empty() {
        return Ok(0);
    }
    let descs: Vec<_> = projections.iter().map(|p| ffi_projection(p)).collect();
    let readers = ScopedArguments::new(projections.iter().copied());
    let pointers = readers.pointers();
    let alternate = readers.alternate_pointers();
    let count =
        i32::try_from(descs.len()).map_err(|_| "CPU prefetch projection count exceeds i32")?;
    // SAFETY: both arrays live through the call. The companion copies descriptors and retains
    // scoped contexts before detached work is queued; projection owners are live during handoff.
    Ok(unsafe {
        if readers.mirrored() {
            let call = backend
                .mirror_prefetch
                .ok_or("CPU companion lacks scoped mirror prefetch entrypoint")?;
            call(
                descs.as_ptr(),
                count,
                error.as_mut_ptr(),
                error.len(),
                pointers.as_ptr(),
                alternate.as_ptr(),
            )
        } else if readers.needed() {
            let call = backend
                .scoped_prefetch
                .ok_or("CPU companion lacks scoped prefetch entrypoint")?;
            call(
                descs.as_ptr(),
                count,
                error.as_mut_ptr(),
                error.len(),
                pointers.as_ptr(),
            )
        } else {
            let call = backend
                .prefetch
                .ok_or("CPU companion lacks ABI2 prefetch entrypoint")?;
            call(descs.as_ptr(), count, error.as_mut_ptr(), error.len())
        }
    })
}

fn prefetch_worker(
    receiver: Receiver<(u16, Vec<f32>)>,
    table: std::collections::HashMap<u16, PredictLayer>,
    resident: std::collections::HashSet<(u16, u8, u16)>,
    depth: usize,
    top: usize,
    min_layer: u16,
) {
    let Ok(backend) = backend() else { return };
    if backend.prefetch.is_none() {
        return;
    }
    let mut error = vec![0i8; 512];
    while let Ok((layer, input)) = receiver.recv() {
        for d in 1..=depth {
            let target = layer + d as u16;
            if target < min_layer {
                continue; // pilot precision only justifies the deep half (k >= ~40)
            }
            let Some(predict) = table.get(&target) else {
                continue;
            };
            if input.len() != predict.n_embd {
                continue;
            }
            // logits = router @ x with sigmoid selection parity to the runtime oracle.
            let mut logits = vec![0.0f32; predict.n_expert];
            for (expert, logit) in logits.iter_mut().enumerate() {
                let row = &predict.router_t[expert * predict.n_embd..(expert + 1) * predict.n_embd];
                *logit = row
                    .iter()
                    .zip(&input)
                    .map(|(weight, value)| weight * value)
                    .sum();
            }
            let Ok((sel, _weights)) = crate::hybrid::HybridModel::moe_route_sigmoid_host_public(
                &logits,
                1,
                predict.n_expert,
                predict.n_used,
                predict.bias.as_deref(),
                predict.sig.0,
                predict.sig.1,
                predict.active.as_deref(),
            ) else {
                continue;
            };
            let mut owned_projections = Vec::new();
            for &expert in sel.iter().take(top) {
                let expert_index = expert as usize;
                let Some(Some(projections)) = predict.experts.get(expert_index) else {
                    continue;
                };
                for (proj_index, owned) in projections.iter().enumerate() {
                    if owned.file.is_none() && owned.scoped_reader.is_none() {
                        continue; // memory-backed, nothing to read
                    }
                    if resident.contains(&(target, proj_index as u8, expert as u16)) {
                        continue; // HBM-resident, never CPU-routed
                    }
                    owned_projections.push(owned);
                }
            }
            if owned_projections.is_empty() {
                continue;
            }
            // Speculative I/O remains best effort; construction already propagated binding
            // and extension-admission errors before this worker was started.
            let _ = submit_projection_prefetch(backend, &owned_projections, &mut error);
        }
    }
}

pub(crate) fn exposed_wait_ns() -> u64 {
    EXPOSED_WAIT_NS.load(Ordering::Relaxed)
}

pub(crate) fn stats() -> (u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64) {
    let mut cache_hits = 0;
    let mut cache_misses = 0;
    let mut read_bytes = 0;
    let mut resident_bytes = 0;
    let mut prepare_ns = 0;
    let mut io_ns = 0;
    let mut insert_ns = 0;
    let mut compute_ns = 0;
    if let Ok(backend) = backend() {
        // SAFETY: all four outputs point to initialized u64 storage owned by this call.
        unsafe {
            (backend.cache_stats)(
                &mut cache_hits,
                &mut cache_misses,
                &mut read_bytes,
                &mut resident_bytes,
            );
            (backend.profile_stats)(&mut prepare_ns, &mut io_ns, &mut insert_ns, &mut compute_ns);
        }
    }
    (
        CALLS.load(Ordering::Relaxed),
        EXPERTS.load(Ordering::Relaxed),
        WALL_NS.load(Ordering::Relaxed),
        cache_hits,
        cache_misses,
        read_bytes,
        resident_bytes,
        prepare_ns,
        io_ns,
        insert_ns,
        compute_ns,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_supported_native_qtypes_without_translation() {
        for qtype in [QT_Q2_K, QT_Q3_K, QT_Q4_K, QT_IQ3_S, QT_IQ4_XS, QT_Q8_0] {
            assert_eq!(native_qtype(qtype).unwrap(), qtype);
        }
    }

    #[test]
    fn rejects_missing_symbols_and_wrong_abi() {
        let error = match load_backend_from_path(std::ffi::OsStr::new("libc.so.6")) {
            Ok(_) => panic!("libc unexpectedly provided the CPU expert ABI"),
            Err(error) => error,
        };
        assert!(error.contains("missing symbol memra_cpu_experts_abi_version"));
        assert!(require_abi_v2(0).is_err());
        assert!(require_abi_v2(1).is_err());
        assert!(require_abi_v2(2).is_ok());
    }

    #[test]
    fn validates_thread_count() {
        assert_eq!(parse_thread_count(None).unwrap(), 8);
        assert_eq!(parse_thread_count(Some("1")).unwrap(), 1);
        assert_eq!(parse_thread_count(Some("256")).unwrap(), 256);
        for invalid in ["", "0", "257", "eight"] {
            assert!(parse_thread_count(Some(invalid)).is_err());
        }
    }

    #[test]
    fn dropped_ticket_joins_outstanding_worker() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};

        let (reply, receiver) = std::sync::mpsc::sync_channel(1);
        let (release, start) = std::sync::mpsc::sync_channel(0);
        let completed = Arc::new(AtomicBool::new(false));
        let worker_completed = Arc::clone(&completed);
        let worker = std::thread::spawn(move || {
            start.recv().unwrap();
            worker_completed.store(true, Ordering::Release);
            reply.send(Ok(Vec::new())).unwrap();
        });
        let ticket = CpuExpertTicket {
            receiver: Some(receiver),
        };
        release.send(()).unwrap();
        drop(ticket);
        assert!(completed.load(Ordering::Acquire));
        worker.join().unwrap();
    }

    #[test]
    fn scoped_arguments_leave_unbound_calls_empty() {
        let p = OwnedProjection {
            weights: 1,
            qtype: QT_Q8_0,
            in_features: 32,
            out_features: 32,
            row_bytes: 34,
            byte_len: 1088,
            file: None,
            file_offset: 0,
            scoped_reader: None,
            scale: 1.0,
        };
        let args = ScopedArguments::new([&p, &p, &p]);
        assert!(!args.needed());
        assert_eq!(args.abi.capacity(), 0);
        assert!(args.pointers().is_empty());
    }

    #[test]
    #[ignore = "requires the scoped native CPU companion; no CUDA initialization"]
    fn scoped_bridge_native_token_rows_and_opened_inode_parity() {
        use crate::model::{HostBuf, HostExps};
        use memra_gguf::{
            GgufFile,
            bound_source::BoundTensorSource,
            source::GgufSource,
            tensor_contract::{LayerTensor, TensorId},
        };
        let path =
            std::env::temp_dir().join(format!("memra-cpu-bridge-{}.gguf", std::process::id()));
        memra_gguf::micro_gguf::write_glm_dsa_micro(&path, 541).unwrap();
        let (scoped_job, scoped_rows, scoped_raw, expected, expected_rows, expected_raw) = {
            let file = GgufFile::open(&path).unwrap();
            let source = GgufSource(&file);
            let bound = BoundTensorSource::compile(&source).unwrap();
            let mut raw = Vec::new();
            let mut scoped = Vec::new();
            for tensor in [
                LayerTensor::MoeExpertGateBank,
                LayerTensor::MoeExpertUpBank,
                LayerTensor::MoeExpertDownBank,
            ] {
                let id = TensorId::Layer { index: 1, tensor };
                let value = bound.tensor(&id).unwrap();
                assert_eq!(value.ggml_type, memra_gguf::GgmlType::Q8_0);
                let (in_f, out_f, n_expert) = (
                    value.ne[0] as usize,
                    value.ne[1] as usize,
                    value.ne[2] as usize,
                );
                let stride = value.bytes.len() / n_expert;
                let make = |bytes| HostExps {
                    bytes,
                    tiers: None,
                    qtype: QT_Q8_0,
                    in_f,
                    out_f,
                    n_expert,
                    row_bytes: stride / out_f,
                    expert_stride: stride,
                    layouts: None,
                    macros: None,
                    fp8_blk: None,
                };
                raw.push(make(HostBuf::Paged(value.bytes.to_vec())));
                scoped.push(make(HostBuf::Bounded(bound.disk(&id).unwrap().unwrap())));
            }
            let experts = |banks: &[HostExps]| {
                [0usize, 3]
                    .into_iter()
                    .map(|e| OwnedExpert {
                        gate: projection(&banks[0], e).unwrap(),
                        up: projection(&banks[1], e).unwrap(),
                        down: projection(&banks[2], e).unwrap(),
                        route_weight: if e == 0 { 0.7 } else { 0.3 },
                    })
                    .collect::<Vec<_>>()
            };
            let input: Vec<_> = (0..raw[0].in_f)
                .map(|i| (i as i32 % 13 - 6) as f32 * 0.01)
                .collect();
            let job = |experts| CpuExpertJob {
                experts,
                input: input.clone(),
                output_features: raw[2].out_f,
                threads: 4,
            };
            let expected = execute_token(job(experts(&raw))).unwrap();
            let scoped_job = job(experts(&scoped));
            for expert in &scoped_job.experts {
                for p in [&expert.gate, &expert.up, &expert.down] {
                    assert_eq!(ffi_projection(p).weights, std::ptr::null());
                    assert_eq!(ffi_projection(p).file_fd, -1);
                }
            }
            let rows = |expert| CpuRowsJob {
                expert,
                inputs: [input.clone(), input.clone()].concat(),
                route_weights: vec![0.7, 0.3],
                output_features: raw[2].out_f,
                threads: 4,
                raw: false,
            };
            let expected_rows = execute_rows(rows(experts(&raw).remove(0))).unwrap();
            let scoped_rows = rows(experts(&scoped).remove(0));
            let mut raw_job = rows(experts(&raw).remove(0));
            raw_job.raw = true;
            raw_job.expert.down.scale = 0.37;
            let expected_raw = execute_rows(raw_job).unwrap();
            let mut scoped_raw = rows(experts(&scoped).remove(0));
            scoped_raw.raw = true;
            scoped_raw.expert.down.scale = 0.37;
            (
                scoped_job,
                scoped_rows,
                scoped_raw,
                expected,
                expected_rows,
                expected_raw,
            )
        };
        // Scoped jobs retain only the exact opened readers after all model/source stores drop.
        std::fs::remove_file(&path).unwrap();
        std::fs::write(&path, b"replacement does not contain model weights").unwrap();
        let actual = execute_token(scoped_job).unwrap();
        let actual_rows = execute_rows(scoped_rows).unwrap();
        let actual_raw = execute_rows(scoped_raw).unwrap();
        assert_eq!(
            actual_raw.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            expected_raw.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(
            actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(
            actual_rows.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            expected_rows
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>()
        );
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    #[ignore = "requires a legacy ABI2-only companion symbol fixture; no CUDA initialization"]
    fn scoped_bridge_refuses_legacy_companion_without_fallback() {
        use memra_gguf::{
            GgufFile, bound_source::BoundTensorSource, source::GgufSource,
            tensor_contract::TensorId,
        };
        let path = std::env::temp_dir().join(format!(
            "memra-cpu-missing-extension-{}.gguf",
            std::process::id()
        ));
        memra_gguf::micro_gguf::write_glm_dsa_micro(&path, 541).unwrap();
        let file = GgufFile::open(&path).unwrap();
        let source = GgufSource(&file);
        let bound = BoundTensorSource::compile(&source).unwrap();
        let view = bound.disk(&TensorId::TokenEmbedding).unwrap().unwrap();
        let error = match scoped_reader(view) {
            Ok(_) => panic!("legacy companion accepted a scoped source"),
            Err(error) => error,
        };
        assert!(
            error.contains("lacks the complete scoped-reader extension"),
            "{error}"
        );
        let id = TensorId::Layer {
            index: 1,
            tensor: memra_gguf::tensor_contract::LayerTensor::MoeExpertGateBank,
        };
        let tensor = bound.tensor(&id).unwrap();
        let (in_f, out_f, n_expert) = (
            tensor.ne[0] as usize,
            tensor.ne[1] as usize,
            tensor.ne[2] as usize,
        );
        let stride = tensor.bytes.len() / n_expert;
        let experts = crate::model::HostExps {
            bytes: crate::model::HostBuf::Bounded(bound.disk(&id).unwrap().unwrap()),
            tiers: None,
            qtype: QT_Q8_0,
            in_f,
            out_f,
            n_expert,
            row_bytes: stride / out_f,
            expert_stride: stride,
            layouts: None,
            macros: None,
            fp8_blk: None,
        };
        assert!(
            predictor_projection(&experts, 0).is_err(),
            "predictor hid scoped admission error as optional absence"
        );
        std::fs::remove_file(path).unwrap();
    }

    struct AlignedFixture(std::path::PathBuf);
    impl Drop for AlignedFixture {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.0).unwrap();
        }
    }
    fn aligned_fixture(label: &str) -> AlignedFixture {
        use memra_gguf::{
            config::{HfConfig, ModelConfig},
            model_packs,
            tensor_contract::CheckpointDialect,
        };
        let dir =
            std::env::temp_dir().join(format!("memra-aligned-cpu-{label}-{}", std::process::id()));
        std::fs::create_dir(&dir).unwrap();
        let config = r#"{"model_type":"qwen3_moe","num_hidden_layers":1,"hidden_size":128,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":64,"intermediate_size":512,"vocab_size":64,"max_position_embeddings":128,"num_experts":4,"num_experts_per_tok":2,"moe_intermediate_size":512}"#;
        let cfg = ModelConfig::from_hf(&HfConfig::parse(config));
        let pack = model_packs::for_config(&cfg).unwrap();
        let plan = pack.compile_plan(&cfg).unwrap();
        let contract = pack
            .compile_tensor_contract(
                &cfg,
                &plan,
                CheckpointDialect::Gguf,
                pack.contract_options(&cfg),
            )
            .unwrap();
        let mut bytes = Vec::new();
        let mut rows = Vec::new();
        for r in contract.requirements.iter().filter(|r| r.required) {
            let name = &r.names[0];
            let n = r.shape.iter().product::<u64>() as usize;
            let values: Vec<_> = (0..n).map(|i| (i as i32 % 19 - 9) as f32 * 0.01).collect();
            let (qtype, data) = if name.contains("_exps.weight") {
                ("Q8_0", memra_gguf::nvfp4_repack::f32_to_q8_0(&values))
            } else {
                ("F32", values.iter().flat_map(|x| x.to_le_bytes()).collect())
            };
            bytes.resize(bytes.len().div_ceil(4096) * 4096, 0);
            let offset = bytes.len();
            bytes.extend(&data);
            rows.push(format!("{name:?}:{{\"file\":\"weights.bin\",\"offset\":{offset},\"qtype\":{qtype:?},\"ne\":{:?},\"bytes\":{}}}",r.shape,data.len()));
        }
        std::fs::write(dir.join("weights.bin"), bytes).unwrap();
        std::fs::write(dir.join("config.json"), config).unwrap();
        std::fs::write(
            dir.join("manifest.json"),
            format!(
                "{{\"format\":\"memra-repack-v1\",\"tensors\":{{{}}}}}",
                rows.join(",")
            ),
        )
        .unwrap();
        AlignedFixture(dir)
    }
    fn aligned_banks(source: &dyn memra_gguf::source::TensorSource) -> Vec<crate::model::HostExps> {
        use memra_gguf::{
            bound_source::BoundTensorSource,
            tensor_contract::{LayerTensor, TensorId},
        };
        let bound = BoundTensorSource::compile(source).unwrap();
        [
            LayerTensor::MoeExpertGateBank,
            LayerTensor::MoeExpertUpBank,
            LayerTensor::MoeExpertDownBank,
        ]
        .into_iter()
        .map(|tensor| {
            let id = TensorId::Layer { index: 0, tensor };
            let value = bound.tensor(&id).unwrap();
            let view = bound.disk(&id).unwrap().unwrap();
            let (in_f, out_f, n_expert) = (
                value.ne[0] as usize,
                value.ne[1] as usize,
                value.ne[2] as usize,
            );
            let stride = value.bytes.len() / n_expert;
            assert!(view.direct_aligned());
            assert_eq!(stride % 4096, 0);
            crate::model::HostExps {
                bytes: crate::model::HostBuf::Bounded(view),
                tiers: None,
                qtype: QT_Q8_0,
                in_f,
                out_f,
                n_expert,
                row_bytes: stride / out_f,
                expert_stride: stride,
                layouts: None,
                macros: None,
                fp8_blk: None,
            }
        })
        .collect()
    }
    fn reader_info(p: &OwnedProjection) -> memra_gguf::bound_disk::ffi::ScopedDiskInfoV1 {
        let abi = p.scoped_reader.as_ref().unwrap().as_abi();
        let mut info = Default::default();
        assert_eq!(unsafe { (abi.info)(abi.context, &mut info) }, 0);
        info
    }

    #[test]
    #[ignore = "requires scoped CPU companion; verifies actual aligned Rust reader mode"]
    fn scoped_bridge_aligned_direct_token_rows_parity() {
        let fixture = aligned_fixture("direct");
        let (scoped, rows, expected, expected_rows) = {
            let source = memra_gguf::source::Hy3RepackSource::open(&fixture.0).unwrap();
            let banks = aligned_banks(&source);
            let input: Vec<_> = (0..128).map(|i| (i % 11 - 5) as f32 * 0.01).collect();
            let mut experts = Vec::new();
            let mut memory = Vec::new();
            let mut owners = Vec::new();
            for e in [0, 3] {
                let mut scoped: Vec<_> = banks.iter().map(|b| projection(b, e).unwrap()).collect();
                for (bank, p) in banks.iter().zip(&scoped) {
                    let info = reader_info(p);
                    assert_eq!(info.offset % 4096, 0);
                    assert_eq!(info.len % 4096, 0);
                    let direct = std::env::var("MEMRA_CPU_EXPERT_IO").as_deref() == Ok("direct");
                    assert_eq!(info.direct, u32::from(direct));
                    println!(
                        "RUST_SCOPED_IO offset={} len={} direct={}",
                        info.offset, info.len, info.direct
                    );
                    owners.push(bank.expert_bytes(e).to_vec());
                }
                let mut raw: Vec<_> = scoped
                    .iter()
                    .enumerate()
                    .map(|(i, p)| {
                        let mut raw = p.clone();
                        raw.scoped_reader = None;
                        raw.weights = owners[owners.len() - 3 + i].as_ptr() as usize;
                        raw
                    })
                    .collect();
                let weight = if e == 0 { 0.7 } else { 0.3 };
                memory.push(OwnedExpert {
                    gate: raw.remove(0),
                    up: raw.remove(0),
                    down: raw.remove(0),
                    route_weight: weight,
                });
                experts.push(OwnedExpert {
                    gate: scoped.remove(0),
                    up: scoped.remove(0),
                    down: scoped.remove(0),
                    route_weight: weight,
                });
            }
            let expected = execute_token(CpuExpertJob {
                experts: memory.clone(),
                input: input.clone(),
                output_features: 128,
                threads: 4,
            })
            .unwrap();
            let mkrows = |expert| CpuRowsJob {
                expert,
                inputs: [input.clone(), input.clone()].concat(),
                route_weights: vec![0.7, 0.3],
                output_features: 128,
                threads: 4,
                raw: false,
            };
            let expected_rows = execute_rows(mkrows(memory.remove(0))).unwrap();
            let rows = mkrows(experts[0].clone());
            (
                CpuExpertJob {
                    experts,
                    input,
                    output_features: 128,
                    threads: 4,
                },
                rows,
                expected,
                expected_rows,
            )
        };
        let path = fixture.0.join("weights.bin");
        std::fs::remove_file(&path).unwrap();
        std::fs::write(path, b"replacement").unwrap();
        let before = stats();
        let expected_bytes: usize = scoped
            .experts
            .iter()
            .flat_map(|e| [&e.gate, &e.up, &e.down])
            .map(|p| p.byte_len)
            .sum();
        let got = execute_token(scoped).unwrap();
        let after = stats();
        assert_eq!(after.4 - before.4, 6); // exactly six cold expert projections
        assert_eq!(after.5 - before.5, expected_bytes as u64);
        let got_rows = execute_rows(rows).unwrap();
        println!(
            "RUST_SCOPED_CACHE cold_misses={} read_bytes={} aligned_windows=6",
            after.4 - before.4,
            after.5 - before.5
        );

        assert_eq!(
            got.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(
            got_rows.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            expected_rows
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    #[ignore = "requires scoped companion with direct mode and a nonempty mirror-map request"]
    fn scoped_bridge_aligned_mirror_request_refuses() {
        assert_eq!(std::env::var("MEMRA_CPU_EXPERT_IO").unwrap(), "direct");
        assert!(std::env::var_os("MEMRA_CPU_EXPERT_MIRROR_MAP").is_some_and(|p| !p.is_empty()));
        let fixture = aligned_fixture("mirror");
        let source = memra_gguf::source::Hy3RepackSource::open(&fixture.0).unwrap();
        let banks = aligned_banks(&source);
        let error = match projection(&banks[0], 0) {
            Ok(_) => panic!("active mirror request silently accepted"),
            Err(e) => e,
        };
        assert!(error.contains("mirror"), "{error}");
    }

    #[test]
    #[ignore = "requires test-observer companion; uses the production Rust prefetch submission hook"]
    fn scoped_bridge_prefetch_submission_retains_reads_after_argument_drop() {
        use memra_gguf::bound_disk::ffi::ScopedDiskInfoV1;
        type Pause = unsafe extern "C" fn(i32);
        type Entries = unsafe extern "C" fn() -> u64;
        type Take = unsafe extern "C" fn(*const ScopedDiskInfoV1, *mut u8, usize) -> i32;
        let backend = backend().unwrap();
        let handle = backend._handle as *mut c_void;
        let pause: Pause = unsafe {
            std::mem::transmute(
                load_symbol(handle, b"memra_cpu_test_pause_scoped_reads\0").unwrap(),
            )
        };
        let entries: Entries = unsafe {
            std::mem::transmute(
                load_symbol(handle, b"memra_cpu_test_scoped_read_entries\0").unwrap(),
            )
        };
        let take: Take = unsafe {
            std::mem::transmute(load_symbol(handle, b"memra_cpu_test_take_annex\0").unwrap())
        };
        let stats: CacheStatsFn = unsafe {
            std::mem::transmute(
                load_symbol(handle, b"memra_cpu_expert_prefetch_stats_v2\0").unwrap(),
            )
        };
        struct Resume(Pause);
        impl Drop for Resume {
            fn drop(&mut self) {
                unsafe { (self.0)(0) }
            }
        }
        let fixture = aligned_fixture("prefetch");
        unsafe { pause(1) };
        let _resume = Resume(pause);
        let (weak, metadata, expected) = {
            let source = memra_gguf::source::Hy3RepackSource::open(&fixture.0).unwrap();
            let banks = aligned_banks(&source);
            let mut projections = Vec::new();
            let mut expected = Vec::new();
            for e in [0, 3] {
                for bank in &banks {
                    expected.push(bank.expert_bytes(e).to_vec());
                    projections.push(projection(bank, e).unwrap());
                }
            }
            let weak: Vec<_> = projections
                .iter()
                .map(|p| p.scoped_reader.as_ref().unwrap().lifetime())
                .collect();
            let metadata: Vec<_> = projections.iter().map(reader_info).collect();
            for info in &metadata {
                assert_eq!(info.offset % 4096, 0);
                assert_eq!(info.len % 4096, 0);
                let direct = std::env::var("MEMRA_CPU_EXPERT_IO").as_deref() == Ok("direct");
                assert_eq!(info.direct, u32::from(direct));
                println!(
                    "RUST_PREFETCH_IO offset={} len={} direct={}",
                    info.offset, info.len, info.direct
                );
            }
            let refs: Vec<_> = projections.iter().collect();
            let mut error = vec![0i8; 512];
            assert_eq!(
                submit_projection_prefetch(backend, &refs, &mut error).unwrap(),
                6
            );
            (weak, metadata, expected)
        };
        let path = fixture.0.join("weights.bin");
        std::fs::remove_file(&path).unwrap();
        std::fs::write(path, b"replacement").unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while unsafe { entries() } == 0 && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let (mut submitted, mut promoted, mut expired, mut inflight) = (0, 0, 0, 0);
        unsafe { stats(&mut submitted, &mut promoted, &mut expired, &mut inflight) };
        assert!(unsafe { entries() } > 0);
        assert_eq!(submitted, 6);
        assert_eq!(inflight, 6);
        assert!(weak.iter().all(|w| w.is_alive()));
        unsafe { pause(0) };
        while weak.iter().any(|w| w.is_alive()) && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        unsafe { stats(&mut submitted, &mut promoted, &mut expired, &mut inflight) };
        assert_eq!(inflight, 0);
        assert!(weak.iter().all(|w| !w.is_alive()));
        for (info, expected) in metadata.iter().zip(expected) {
            let mut bytes = vec![0; expected.len()];
            assert_eq!(unsafe { take(info, bytes.as_mut_ptr(), bytes.len()) }, 0);
            assert_eq!(bytes, expected);
        }
        println!(
            "RUST_PREFETCH_PASS submitted=6 blocked_after_argument_drop=1 retained=6 final_live=0 inflight=0 annex_bytes=1 router_executed=0"
        );
    }

    struct MirrorFixture {
        source: AlignedFixture,
        alternate_dir: Option<std::path::PathBuf>,
        mirror: std::path::PathBuf,
        legacy: std::path::PathBuf,
    }
    impl Drop for MirrorFixture {
        fn drop(&mut self) {
            if let Some(dir) = &self.alternate_dir {
                std::fs::remove_dir_all(dir).unwrap();
            }
        }
    }
    fn mirror_fixture(case: &str) -> MirrorFixture {
        use std::os::unix::fs::MetadataExt;
        let source = aligned_fixture(case);
        let source_path = source.0.join("weights.bin");
        let legacy = source.0.join("legacy.bin");
        std::fs::copy(&source_path, &legacy).unwrap();
        let alternate_dir = if case == "same-fs" {
            None
        } else {
            let dir = std::path::PathBuf::from(format!(
                "/dev/shm/memra-541-mirror-{case}-{}",
                std::process::id()
            ));
            std::fs::create_dir(&dir).unwrap();
            Some(dir)
        };
        let mirror = alternate_dir
            .as_ref()
            .unwrap_or(&source.0)
            .join("alternate.bin");
        std::fs::copy(&source_path, &mirror).unwrap();
        if case == "wrong-bytes" {
            use std::io::{Seek, SeekFrom, Write};
            let base = memra_gguf::source::Hy3RepackSource::open(&source.0).unwrap();
            let banks = aligned_banks(&base);
            let crate::model::HostBuf::Bounded(view) = &banks[0].bytes else {
                unreachable!()
            };
            let handle = view
                .reader(BoundReadMode::Buffered)
                .unwrap()
                .into_ffi_handle();
            let abi = handle.as_abi();
            let mut info = Default::default();
            assert_eq!(unsafe { (abi.info)(abi.context, &mut info) }, 0);
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .open(&mirror)
                .unwrap();
            file.seek(SeekFrom::Start(info.offset)).unwrap();
            file.write_all(&[view.bytes()[0] ^ 1]).unwrap();
            file.sync_all().unwrap();
        }
        let m = std::fs::metadata(&mirror).unwrap();
        let mut rows = Vec::new();
        for path in [&source_path, &legacy] {
            let a = std::fs::metadata(path).unwrap();
            if case != "same-fs" {
                assert_ne!(
                    a.dev(),
                    m.dev(),
                    "mirror control needs distinct filesystems"
                );
            }
            let source_ctime = a.ctime_nsec() + i64::from(case == "source-generation");
            let target_ctime = m.ctime_nsec() + i64::from(case == "target-generation");
            rows.push(format!(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                a.dev(),
                a.ino(),
                a.len(),
                a.ctime(),
                source_ctime,
                m.dev(),
                m.ino(),
                m.len(),
                m.ctime(),
                target_ctime,
                mirror.display()
            ));
        }
        let map = std::path::PathBuf::from(
            std::env::var_os("MEMRA_CPU_EXPERT_MIRROR_MAP").expect("mirror map path required"),
        );
        assert!(
            map.starts_with(std::env::temp_dir())
                && !map
                    .components()
                    .any(|c| matches!(c, std::path::Component::ParentDir))
        );
        std::fs::write(map, rows.join("\n") + "\n").unwrap();
        MirrorFixture {
            source,
            alternate_dir,
            mirror,
            legacy,
        }
    }
    fn mirror_negative(case: &str, expected: &str) {
        let fixture = mirror_fixture(case);
        let source = memra_gguf::source::Hy3RepackSource::open(&fixture.source.0).unwrap();
        let banks = aligned_banks(&source);
        let error = match projection(&banks[0], 0) {
            Ok(_) => panic!("invalid mirror accepted"),
            Err(e) => e,
        };
        assert!(error.contains(expected), "{error}");
    }
    #[test]
    #[ignore = "requires Linux scoped mirror companion and an isolated map path"]
    fn scoped_mirror_wrong_bytes_refuse() {
        mirror_negative("wrong-bytes", "mirror bytes differ");
    }
    #[test]
    #[ignore = "requires Linux scoped mirror companion and an isolated map path"]
    fn scoped_mirror_wrong_source_generation_refuses() {
        mirror_negative("source-generation", "source generation differs");
    }
    #[test]
    #[ignore = "requires Linux scoped mirror companion and an isolated map path"]
    fn scoped_mirror_wrong_target_generation_refuses() {
        mirror_negative("target-generation", "mirror generation differs");
    }
    #[test]
    #[ignore = "requires Linux scoped mirror companion and an isolated map path"]
    fn scoped_mirror_same_filesystem_refuses() {
        mirror_negative("same-fs", "distinct filesystems");
    }

    #[test]
    #[ignore = "requires Linux scoped mirror companion and isolated scratch"]
    fn scoped_mirror_cached_generation_change_refuses() {
        use std::io::Write;
        let fixture = mirror_fixture("cached-generation");
        let source = memra_gguf::source::Hy3RepackSource::open(&fixture.source.0).unwrap();
        let banks = aligned_banks(&source);
        let owner = projection(&banks[0], 0).unwrap();
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .open(&fixture.mirror)
            .unwrap();
        file.write_all(b"changed after verification").unwrap();
        file.sync_all().unwrap();
        let error = match projection(&banks[0], 0) {
            Ok(_) => panic!("mutated cached mirror accepted"),
            Err(e) => e,
        };
        assert!(error.contains("mirror generation changed"), "{error}");
        drop(owner);
    }

    #[test]
    #[ignore = "requires Linux scoped mirror observer companion and distinct filesystem scratch"]
    fn scoped_mirror_token_rows_and_detached_prefetch_match_legacy() {
        use memra_gguf::bound_disk::ffi::ScopedDiskInfoV1;
        assert_eq!(std::env::var("MEMRA_CPU_EXPERT_IO").unwrap(), "direct");
        let fixture = mirror_fixture("positive");
        let backend = backend().unwrap();
        let handle = backend._handle as *mut c_void;
        type Pause = unsafe extern "C" fn(i32);
        type Entries = unsafe extern "C" fn() -> u64;
        type Take = unsafe extern "C" fn(*const ScopedDiskInfoV1, *mut u8, usize) -> i32;
        let pause: Pause = unsafe {
            std::mem::transmute(
                load_symbol(handle, b"memra_cpu_test_pause_scoped_reads\0").unwrap(),
            )
        };
        let entries: Entries = unsafe {
            std::mem::transmute(
                load_symbol(handle, b"memra_cpu_test_scoped_read_entries\0").unwrap(),
            )
        };
        let take: Take = unsafe {
            std::mem::transmute(load_symbol(handle, b"memra_cpu_test_take_annex\0").unwrap())
        };
        let counts: CacheStatsFn = unsafe {
            std::mem::transmute(load_symbol(handle, b"memra_cpu_test_scoped_io_counts\0").unwrap())
        };
        let stats: CacheStatsFn = unsafe {
            std::mem::transmute(
                load_symbol(handle, b"memra_cpu_expert_prefetch_stats_v2\0").unwrap(),
            )
        };
        struct Resume(Pause);
        impl Drop for Resume {
            fn drop(&mut self) {
                unsafe { (self.0)(0) }
            }
        }
        let (
            job,
            rows,
            raw_rows,
            legacy_job,
            legacy_rows,
            legacy_raw_rows,
            prefetch,
            metadata,
            expected,
            weak,
        ) = {
            let source = memra_gguf::source::Hy3RepackSource::open(&fixture.source.0).unwrap();
            let banks = aligned_banks(&source);
            let legacy_file = std::sync::Arc::new(std::fs::File::open(&fixture.legacy).unwrap());
            let mut selected = Vec::new();
            let mut old = Vec::new();
            let mut prefetch = Vec::new();
            let mut metadata = Vec::new();
            let mut expected = Vec::new();
            let mut weak = Vec::new();
            let mut fd_count = None;
            for e in [0, 3, 1, 2] {
                let mut group = Vec::new();
                let mut legacy_group = Vec::new();
                for bank in &banks {
                    let p = projection(bank, e).unwrap();
                    let pair = p.scoped_reader.as_ref().unwrap();
                    let primary = reader_info(&p);
                    let a = pair.alternate.as_ref().expect("mirror reader required");
                    let abi = a.as_abi();
                    let mut alternate = Default::default();
                    assert_eq!(unsafe { (abi.info)(abi.context, &mut alternate) }, 0);
                    assert_ne!(primary.device, alternate.device);
                    assert_eq!(
                        (primary.offset, primary.len, primary.direct),
                        (alternate.offset, alternate.len, alternate.direct)
                    );
                    assert_eq!(primary.direct, 1);
                    let now = std::fs::read_dir("/proc/self/fd").unwrap().count();
                    if let Some(previous) = fd_count {
                        assert_eq!(
                            now, previous,
                            "expert windows created an extra persistent mirror fd"
                        );
                    } else {
                        fd_count = Some(now);
                    }
                    if e == 1 || e == 2 {
                        weak.push(pair.primary.lifetime());
                        weak.push(a.lifetime());
                        metadata.push(primary);
                        expected.push(bank.expert_bytes(e).to_vec());
                        prefetch.push(p);
                    } else {
                        let mut raw = p.clone();
                        raw.scoped_reader = None;
                        raw.file = Some(legacy_file.clone());
                        raw.file_offset = primary.offset;
                        legacy_group.push(raw);
                        group.push(p);
                    }
                }
                if e == 0 || e == 3 {
                    let weight = if e == 0 { 0.7 } else { 0.3 };
                    selected.push(OwnedExpert {
                        gate: group.remove(0),
                        up: group.remove(0),
                        down: group.remove(0),
                        route_weight: weight,
                    });
                    old.push(OwnedExpert {
                        gate: legacy_group.remove(0),
                        up: legacy_group.remove(0),
                        down: legacy_group.remove(0),
                        route_weight: weight,
                    });
                }
            }
            let input: Vec<_> = (0..128).map(|i| (i % 11 - 5) as f32 * 0.01).collect();
            let row = |expert| CpuRowsJob {
                expert,
                inputs: [input.clone(), input.clone()].concat(),
                route_weights: vec![0.7, 0.3],
                output_features: 128,
                threads: 4,
                raw: false,
            };
            let rows = row(selected[0].clone());
            let legacy_rows = row(old[0].clone());
            let mut raw_rows = row(selected[0].clone());
            raw_rows.raw = true;
            raw_rows.expert.down.scale = 0.37;
            let mut legacy_raw_rows = row(old[0].clone());
            legacy_raw_rows.raw = true;
            legacy_raw_rows.expert.down.scale = 0.37;
            (
                CpuExpertJob {
                    experts: selected,
                    input: input.clone(),
                    output_features: 128,
                    threads: 4,
                },
                rows,
                raw_rows,
                CpuExpertJob {
                    experts: old,
                    input,
                    output_features: 128,
                    threads: 4,
                },
                legacy_rows,
                legacy_raw_rows,
                prefetch,
                metadata,
                expected,
                weak,
            )
        };
        unsafe {
            pause(1);
            pause(0)
        };
        let expected_token = execute_token(legacy_job).unwrap();
        let got = execute_token(job).unwrap();
        assert_eq!(
            got.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            expected_token
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>()
        );
        let (mut primary, mut alternate, mut pbytes, mut abytes) = (0, 0, 0, 0);
        unsafe { counts(&mut primary, &mut alternate, &mut pbytes, &mut abytes) };
        assert_eq!((primary, alternate, pbytes, abytes), (6, 6, 196608, 221184));
        let expected_raw = execute_rows(legacy_raw_rows).unwrap();
        let actual_raw = execute_rows(raw_rows).unwrap();
        assert_eq!(
            actual_raw.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            expected_raw.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
        let expected_rows = execute_rows(legacy_rows).unwrap();
        let got = execute_rows(rows).unwrap();
        assert_eq!(
            got.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            expected_rows
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>()
        );
        unsafe { pause(1) };
        let _resume = Resume(pause);
        let refs: Vec<_> = prefetch.iter().collect();
        let mut error = vec![0i8; 512];
        assert_eq!(
            submit_projection_prefetch(backend, &refs, &mut error).unwrap(),
            6
        );
        drop(refs);
        drop(prefetch);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while unsafe { entries() } == 0 && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let (mut submitted, mut promoted, mut expired, mut inflight) = (0, 0, 0, 0);
        unsafe { stats(&mut submitted, &mut promoted, &mut expired, &mut inflight) };
        assert_eq!((submitted, inflight), (6, 6));
        assert!(weak.iter().all(|w| w.is_alive()));
        unsafe { pause(0) };
        while weak.iter().any(|w| w.is_alive()) && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        unsafe { stats(&mut submitted, &mut promoted, &mut expired, &mut inflight) };
        assert_eq!(inflight, 0);
        assert!(weak.iter().all(|w| !w.is_alive()));
        unsafe { counts(&mut primary, &mut alternate, &mut pbytes, &mut abytes) };
        assert_eq!((primary, alternate, pbytes, abytes), (6, 6, 196608, 221184));
        for (info, expected) in metadata.iter().zip(expected) {
            let mut actual = vec![0; expected.len()];
            assert_eq!(unsafe { take(info, actual.as_mut_ptr(), actual.len()) }, 0);
            assert_eq!(actual, expected);
        }
        println!(
            "RUST_MIRROR_PASS legacy_token_rows_bits=1 primary_reads=6 alternate_reads=6 primary_bytes=196608 alternate_bytes=221184 retained_contexts=12 final_live=0 inflight=0 annex_bytes=1 shared_fd=1"
        );
    }

    #[test]
    #[ignore = "requires mirror observer companion with inflight cap exactly three"]
    fn scoped_mirror_inflight_waits_for_final_half_and_enforces_cap() {
        mirror_counter_control(false);
    }

    #[test]
    #[ignore = "requires mirror observer companion; injects an alternate-half EOF"]
    fn scoped_mirror_inflight_error_half_balances() {
        mirror_counter_control(true);
    }

    fn mirror_counter_control(fail_alternate: bool) {
        assert_eq!(
            std::env::var("MEMRA_CPU_EXPERT_PREFETCH_MAX_INFLIGHT").unwrap(),
            "3"
        );
        let fixture = mirror_fixture("signed-counter");
        let backend = backend().unwrap();
        let handle = backend._handle as *mut c_void;
        type Pause = unsafe extern "C" fn(i32);
        type Signed = unsafe extern "C" fn() -> i32;
        type Completed = unsafe extern "C" fn(*mut u64, *mut u64);
        type Take = unsafe extern "C" fn(*const ScopedDiskInfoV1, *mut u8, usize) -> i32;
        let pause: Pause = unsafe {
            std::mem::transmute(
                load_symbol(handle, b"memra_cpu_test_pause_scoped_reads\0").unwrap(),
            )
        };
        let signed: Signed = unsafe {
            std::mem::transmute(
                load_symbol(handle, b"memra_cpu_test_signed_prefetch_inflight\0").unwrap(),
            )
        };
        let completed: Completed = unsafe {
            std::mem::transmute(load_symbol(handle, b"memra_cpu_test_completed_halves\0").unwrap())
        };
        let take: Take = unsafe {
            std::mem::transmute(load_symbol(handle, b"memra_cpu_test_take_annex\0").unwrap())
        };
        struct Resume(Pause);
        impl Drop for Resume {
            fn drop(&mut self) {
                unsafe { (self.0)(0) }
            }
        }
        let source = memra_gguf::source::Hy3RepackSource::open(&fixture.source.0).unwrap();
        let banks = aligned_banks(&source);
        let first: Vec<_> = banks.iter().map(|b| projection(b, 0).unwrap()).collect();
        let next: Vec<_> = banks.iter().map(|b| projection(b, 3).unwrap()).collect();
        let metadata: Vec<_> = first.iter().map(reader_info).collect();
        let mut error = vec![0i8; 512];
        unsafe { pause(2) };
        let _resume = Resume(pause);
        assert_eq!(
            submit_projection_prefetch(backend, &first.iter().collect::<Vec<_>>(), &mut error)
                .unwrap(),
            3
        );
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        let (mut primary, mut alternate) = (0, 0);
        while primary < 3 && std::time::Instant::now() < deadline {
            unsafe { completed(&mut primary, &mut alternate) };
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let primaries_before = primary;
        let alternates_before = alternate;
        let blocked_counter = unsafe { signed() };
        let excess = if fail_alternate {
            std::fs::OpenOptions::new()
                .write(true)
                .open(&fixture.mirror)
                .unwrap()
                .set_len(0)
                .unwrap();
            0
        } else {
            submit_projection_prefetch(backend, &next.iter().collect::<Vec<_>>(), &mut error)
                .unwrap()
        };
        unsafe { pause(0) };
        let expected_jobs = 3u64 + excess.max(0) as u64;
        while (primary < expected_jobs || alternate < expected_jobs)
            && std::time::Instant::now() < deadline
        {
            unsafe { completed(&mut primary, &mut alternate) };
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let final_counter = unsafe { signed() };
        println!(
            "MIRROR_SIGNED_CONTROL error_half={fail_alternate} primaries_before_release={primaries_before} alternates_before_release={alternates_before} blocked_counter={blocked_counter} excess_admitted={excess} final_counter={final_counter} completed_primary={primary} completed_alternate={alternate}"
        );
        // Drain before assertions: the old counter must fail without stranding blocked I/O.
        assert_eq!(primaries_before, 3);
        assert_eq!(alternates_before, 0);
        assert_eq!(primary, expected_jobs);
        assert_eq!(alternate, expected_jobs);
        assert_eq!(
            blocked_counter, 3,
            "projection charge disappeared before its alternate half finished"
        );
        assert_eq!(
            excess, 0,
            "prefetch cap admitted projections while all three slots remained charged"
        );
        assert_eq!(
            final_counter, 0,
            "signed inflight counter did not balance exactly"
        );
        for info in &metadata {
            let mut data = vec![0; info.len as usize];
            let status = unsafe { take(info, data.as_mut_ptr(), data.len()) };
            if fail_alternate {
                assert_ne!(
                    status, 0,
                    "failed mirrored projection published annex bytes"
                );
            } else {
                assert_eq!(status, 0);
            }
        }
        if fail_alternate {
            println!(
                "MIRROR_SIGNED_ERROR_PASS charged_until_final_half=1 exact_zero=1 failed_annex_absent=1"
            );
            return;
        }
        assert_eq!(
            submit_projection_prefetch(backend, &first.iter().collect::<Vec<_>>(), &mut error)
                .unwrap(),
            3,
            "balanced counter could not retry"
        );
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while unsafe { signed() } != 0 && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert_eq!(unsafe { signed() }, 0);
        println!("MIRROR_SIGNED_PASS cap=3 charged_until_final_half=1 exact_zero=1 retry=1");
    }
}
