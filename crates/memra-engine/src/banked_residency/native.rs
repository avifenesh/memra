//! Qualification-only immutable-GGUF expert bank installer. No runtime env door,
//! artifact conversion, external source, or alternative numeric executor.
use crate::{
    Engine,
    banked_residency::{ExpertBankBudget, ExpertBankRefusal, gpu_bank_budget, host_bank_budget},
    hybrid::{Ffn, HybridModel, MoeWeights},
};
use memra_gguf::{
    GgufFile,
    expert_banks::{ExpertBankBlock, ExpertBankCatalog, ExpertBankProjection, expert_bank_catalog},
    model_packs,
    source::census_from_gguf,
    tensor_contract::{
        CheckpointDialect, ContractOptions, ExpertTensor, TensorCensusEntry, TensorContract,
    },
};
use memra_tier::{bank::*, contracts::*, tier::governor::Governor};
use sha2::{Digest as _, Sha256};
use std::{
    cell::{Cell, RefCell},
    collections::BTreeMap,
    fs::File,
    os::unix::fs::FileExt,
    rc::Rc,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        mpsc,
    },
    time::Instant,
};

const APPROVED_SHA: &str = "df27a780435b7b45c2597536112ea3cb091f8544c3d0c3318d9f4258b31f7adf";
struct FileReader {
    file: Arc<File>,
    ranges: BTreeMap<TensorId, (u64, u64)>,
    reads: Rc<Cell<u64>>,
    /// `--expert-bank-stages` only: host wall of every `read_exact_at` (DAY40 `pread`).
    pread_ns: Option<Rc<Cell<u64>>>,
}
impl ExactReader for FileReader {
    fn storage_bytes(&self, tensor: &TensorId) -> Result<u64> {
        Ok(self.ranges.get(tensor).ok_or(Error::NotFound)?.1)
    }
    fn read_exact(&mut self, tensor: &TensorId, offset: u64, dst: &mut [u8]) -> Result<()> {
        let &(start, len) = self.ranges.get(tensor).ok_or(Error::NotFound)?;
        if offset
            .checked_add(dst.len() as u64)
            .ok_or(Error::Overflow)?
            > len
        {
            return Err(Error::InvalidLayout);
        }
        let started = self.pread_ns.as_ref().map(|_| Instant::now());
        self.file
            .read_exact_at(dst, start.checked_add(offset).ok_or(Error::Overflow)?)?;
        if let (Some(started), Some(total)) = (started, &self.pread_ns) {
            total.set(total.get().saturating_add(elapsed_ns(started)));
        }
        self.reads.set(self.reads.get() + 1);
        Ok(())
    }
}
fn elapsed_ns(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

/// Day 45 (`research/spill-c-20260919/DAY45.md`): one record the host fill reads, its exact range
/// in the authenticated artifact.
struct FillJob {
    local: ExpertDispatchId,
    offset: u64,
    len: usize,
}
/// A record a fill worker read and checksummed, handed to the owner by value.
struct FillDone {
    local: ExpertDispatchId,
    bytes: HostBytes,
    digest: Digest,
}

/// Where a fill worker takes a record's buffer: `None` means none is free right now (the worker
/// retries against the stop flag). The installer's factory takes from the pinned pool above its
/// demand reserve (DAY47); the CUDA-free tests use heap buffers.
type FillBuffers = Arc<dyn Fn(usize) -> Option<HostBytes> + Send + Sync>;

/// Day 47 (`research/spill-c-20260919/DAY47.md`): the door's pinned host tier. One cached
/// allocation (`CU_MEMHOSTALLOC_PORTABLE`, never write-combined: the fill and the verify read these
/// bytes on the CPU) carved into fixed buffers per host-plan class, the class's planned slots
/// plus `POOL_HEADROOM`. Each buffer is owned by at most one `PooledBuffer` at a time (the free
/// lists sit under one mutex) and zeroed the first time it is handed out; dropping the
/// `PooledBuffer` returns it. The allocation is freed when the pool and its last buffer are gone.
struct PinnedPool {
    base: *mut u8,
    context: Arc<cudarc::driver::CudaContext>,
    classes: Vec<PoolClass>,
    state: std::sync::Mutex<PoolState>,
    bytes: usize,
}
struct PoolClass {
    capacity: usize,
    start: usize,
}
struct PoolState {
    free: Vec<Vec<usize>>,
    touched: Vec<Vec<bool>>,
}
// SAFETY: `base` is one pinned host allocation that never moves and is freed only in `Drop`, after
// every `PooledBuffer` (each holds an `Arc` of the pool) is gone. Each buffer range is taken and
// returned under `state`'s mutex, so at most one `PooledBuffer` ever reaches a given range, and
// the pool itself exposes no byte access.
unsafe impl Send for PinnedPool {}
unsafe impl Sync for PinnedPool {}
/// Buffers per class beyond the planned slots: the leases I1 keeps open, the fill's queued and
/// in-worker records, and one.
const POOL_HEADROOM: usize = crate::moe_cache::BANKED_INFLIGHT + 1 + 64 + 8 + 1;
/// Buffers per class the fill never takes, so a demand's read always finds one.
const FILL_RESERVE: usize = crate::moe_cache::BANKED_INFLIGHT + 2;
impl PinnedPool {
    fn new(
        context: Arc<cudarc::driver::CudaContext>,
        classes: &[(u64, usize)],
    ) -> std::result::Result<Arc<Self>, Box<dyn std::error::Error>> {
        let mut layout = Vec::with_capacity(classes.len());
        let mut counts = Vec::with_capacity(classes.len());
        let mut total = 0usize;
        for &(bytes, slots) in classes {
            let capacity = usize::try_from(bytes)?;
            let count = slots.checked_add(POOL_HEADROOM).ok_or(Error::Overflow)?;
            layout.push(PoolClass {
                capacity,
                start: total,
            });
            counts.push(count);
            total = capacity
                .checked_mul(count)
                .and_then(|n| total.checked_add(n))
                .ok_or(Error::Overflow)?;
        }
        context.bind_to_thread()?;
        // SAFETY: documented FFI (`cuMemHostAlloc`); the returned range is owned by this pool.
        let base = unsafe {
            cudarc::driver::result::malloc_host(
                total.max(1),
                cudarc::driver::sys::CU_MEMHOSTALLOC_PORTABLE,
            )?
        }
        .cast::<u8>();
        let state = PoolState {
            free: counts.iter().map(|&n| (0..n).rev().collect()).collect(),
            touched: counts.iter().map(|&n| vec![false; n]).collect(),
        };
        Ok(Arc::new(Self {
            base,
            context,
            classes: layout,
            state: std::sync::Mutex::new(state),
            bytes: total,
        }))
    }
    /// A buffer of exactly `len` bytes from the smallest class that holds it and has more than
    /// `reserve` free, or `None`.
    fn take(self: &Arc<Self>, len: usize, reserve: usize) -> Option<PooledBuffer> {
        let (class, index, first) = {
            let mut state = self.state.lock().ok()?;
            let class = self.classes.iter().enumerate().position(|(class, spec)| {
                spec.capacity >= len && state.free[class].len() > reserve
            })?;
            let index = state.free[class].pop()?;
            let first = !std::mem::replace(&mut state.touched[class][index], true);
            (class, index, first)
        };
        let spec = &self.classes[class];
        // SAFETY: `index < count` of this class, so the range lies inside the allocation, and it
        // was just popped from the free list, so no other `PooledBuffer` holds it.
        let ptr = unsafe { self.base.add(spec.start + index * spec.capacity) };
        if first {
            // SAFETY: the exclusive range above; zeroed once so every later slice is initialized.
            unsafe { ptr.write_bytes(0, spec.capacity) };
        }
        Some(PooledBuffer {
            pool: self.clone(),
            class,
            index,
            ptr,
            len,
        })
    }
    fn give_back(&self, class: usize, index: usize) {
        if let Ok(mut state) = self.state.lock() {
            state.free[class].push(index);
        }
    }
}
impl Drop for PinnedPool {
    fn drop(&mut self) {
        let _ = self.context.bind_to_thread();
        // SAFETY: every `PooledBuffer` holds an `Arc` of this pool, so none is alive here.
        let _ = unsafe { cudarc::driver::result::free_host(self.base.cast()) };
    }
}
/// One pooled buffer: exclusive owner of `len` bytes at `ptr` inside the pool.
struct PooledBuffer {
    pool: Arc<PinnedPool>,
    class: usize,
    index: usize,
    ptr: *mut u8,
    len: usize,
}
// SAFETY: the buffer exclusively owns its range (see `PinnedPool::take`); moving it to another
// thread moves that ownership, and the pool it returns to is `Sync`.
unsafe impl Send for PooledBuffer {}
impl HostBuffer for PooledBuffer {
    fn as_slice(&self) -> &[u8] {
        // SAFETY: an exclusive, initialized (zeroed at first take), pinned range of `len` bytes.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
    fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: as above, and `&mut self` makes the access unique.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}
impl Drop for PooledBuffer {
    fn drop(&mut self) {
        self.pool.give_back(self.class, self.index);
    }
}
/// The bank's buffer source: demand reads take from the pool with no reserve.
struct PoolSource(Arc<PinnedPool>);
impl HostBufferSource for PoolSource {
    fn take(&mut self, len: usize) -> Option<Box<dyn HostBuffer>> {
        self.0
            .take(len, 0)
            .map(|buffer| Box::new(buffer) as Box<dyn HostBuffer>)
    }
}
/// The fill's counts, shared by the workers (reads, read errors), the owner's intake
/// (admitted, dropped, refused, errors, the admission time) and the gate's close line.
#[derive(Default)]
struct FillCounts {
    reads: AtomicU64,
    read_errors: AtomicU64,
    admitted: AtomicU64,
    dropped: AtomicU64,
    refused: AtomicU64,
    errors: AtomicU64,
    admit_ns: AtomicU64,
}
impl FillCounts {
    fn line(&self) -> String {
        let get = |a: &AtomicU64| a.load(Ordering::Relaxed);
        format!(
            "fill_reads={} fill_read_errors={} fill_admitted={} fill_dropped={} fill_refused={} fill_errors={} fill_ns={}",
            get(&self.reads),
            get(&self.read_errors),
            get(&self.admitted),
            get(&self.dropped),
            get(&self.refused),
            get(&self.errors),
            get(&self.admit_ns)
        )
    }
}
/// The fill threads, held by the gate: a stop flag the owner raises when the tier is full and
/// the gate raises at close, and the handles the gate joins.
struct FillWorkers {
    stop: Arc<AtomicBool>,
    handles: Vec<std::thread::JoinHandle<()>>,
}
impl Drop for FillWorkers {
    /// Every exit path stops and joins the workers, the installer's error returns included:
    /// each worker retries a full channel against the stop flag, so a join returns after at
    /// most one read and one checksum.
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
    }
}
/// Owner-side intake of finished fills (lives in `TracedDispatch`, on the owner thread).
struct FillIntake {
    rx: mpsc::Receiver<FillDone>,
    counts: Arc<FillCounts>,
    stop: Arc<AtomicBool>,
}
/// Start the fill over `jobs`: each worker takes the next job from a shared cursor, `pread`s
/// its range from the authenticated inode, checksums it with the contract `checksum`, and
/// offers it to the owner over a channel of 64. A full channel is retried until the owner
/// drains it or the stop flag is raised, so no worker blocks past a stop.
fn start_fill(
    file: Arc<File>,
    jobs: Vec<FillJob>,
    counts: Arc<FillCounts>,
    buffers: FillBuffers,
) -> (FillWorkers, FillIntake) {
    let (tx, rx) = mpsc::sync_channel::<FillDone>(64);
    let stop = Arc::new(AtomicBool::new(false));
    let jobs = Arc::new(jobs);
    let cursor = Arc::new(AtomicUsize::new(0));
    let threads = std::thread::available_parallelism()
        .map(|n| n.get() / 2)
        .unwrap_or(1)
        .clamp(1, 8);
    let handles = (0..threads)
        .map(|_| {
            let (file, jobs, cursor, counts, stop, tx, buffers) = (
                file.clone(),
                jobs.clone(),
                cursor.clone(),
                counts.clone(),
                stop.clone(),
                tx.clone(),
                buffers.clone(),
            );
            std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    let index = cursor.fetch_add(1, Ordering::Relaxed);
                    let Some(job) = jobs.get(index) else { break };
                    let mut bytes = loop {
                        if let Some(bytes) = buffers(job.len) {
                            break bytes;
                        }
                        if stop.load(Ordering::Relaxed) {
                            return;
                        }
                        std::thread::sleep(std::time::Duration::from_micros(200));
                    };
                    if file.read_exact_at(bytes.bytes_mut(), job.offset).is_err() {
                        counts.read_errors.fetch_add(1, Ordering::Relaxed);
                        break;
                    }
                    counts.reads.fetch_add(1, Ordering::Relaxed);
                    let digest = checksum(bytes.bytes());
                    let mut done = FillDone {
                        local: job.local,
                        bytes,
                        digest,
                    };
                    loop {
                        match tx.try_send(done) {
                            Ok(()) => break,
                            Err(mpsc::TrySendError::Full(back)) => {
                                if stop.load(Ordering::Relaxed) {
                                    return;
                                }
                                done = back;
                                std::thread::sleep(std::time::Duration::from_micros(200));
                            }
                            Err(mpsc::TrySendError::Disconnected(_)) => return,
                        }
                    }
                }
            })
        })
        .collect();
    (
        FillWorkers {
            stop: stop.clone(),
            handles,
        },
        FillIntake { rx, counts, stop },
    )
}
struct Heat;
impl Hotness<ExpertDomain> for Heat {
    fn demand(&mut self, _: &BankId) {}
    fn score(&self, _: &BankId) -> u64 {
        0
    }
}
/// !Send owner guard, retained by the gate on the CUDA thread for the entire run.
/// The Engine itself contains only its Send+Sync proxy.
pub struct BankedExpertGate<'a> {
    engine: &'a Engine,
    max_bytes: usize,
    owner: ExpertBankOwner,
    reads: Rc<Cell<u64>>,
    budget: SharedBudget,
    metadata: ChargedLease,
    stage_clock: bool,
    fill: Option<FillWorkers>,
    fill_counts: Arc<FillCounts>,
}
impl BankedExpertGate<'_> {
    /// `--expert-bank-stages`: print the door's cumulative stage line for `phase` (DAY40).
    /// A no-op without the flag.
    pub fn print_stage_line(&self, phase: &str) {
        if !self.stage_clock {
            return;
        }
        match self.engine.expert_bank_stage_line() {
            Ok(Some(line)) => eprintln!(
                "[experts-via-tier] stages phase={phase} {line} | proc {}",
                proc_memory_fields()
            ),
            Ok(None) => eprintln!("[experts-via-tier] stages phase={phase} absent"),
            Err(err) => eprintln!("[experts-via-tier] stages phase={phase} refused: {err}"),
        }
    }
}
impl Drop for BankedExpertGate<'_> {
    fn drop(&mut self) {
        if let Some(fill) = &self.fill {
            fill.stop.store(true, Ordering::Relaxed);
        }
        // DAY46: finish every in-flight lease (one stream drain) before the registry closes;
        // `close` refuses while any lease is open.
        let retired = self
            .engine
            .with_moe_cache(self.max_bytes, |cache, _| cache.retire_all_banked());
        if let Err(err) = retired {
            eprintln!("[experts-via-tier] in-flight retire refused: {err}");
        }
        self.print_stage_line("close");
        let report = self.engine.with_moe_cache(self.max_bytes, |cache, _| {
            let (slots, allocated_bytes, evictions) = cache.bank_pressure();
            eprintln!("[expert-gpu-slru] slots={slots} allocated_bytes={allocated_bytes} evictions={evictions}");
            Ok(())
        });
        if let Err(err) = report {
            eprintln!("[expert-gpu-slru] report refused: {err}");
        }
        eprintln!(
            "[experts-via-tier] physical_reads={} owner_close={:?}",
            self.reads.get(),
            self.owner.close()
        );
        if let Some(fill) = self.fill.take() {
            drop(fill); // stop and join (FillWorkers::drop)
            eprintln!("[experts-via-tier] fill {}", self.fill_counts.line());
        }
        // A still-pinned owner refuses release; never force-credit unknown DMA.
        let _ = self.budget.borrow_mut().release(&self.metadata);
    }
}

/// `/proc/self/status` memory fields as `key_kb=value` tokens (DAY44 section 1a: recorded
/// context of the stage lines, no clause reads them). Missing fields print nothing.
fn proc_memory_fields() -> String {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    ["VmRSS", "RssAnon", "RssFile", "RssShmem", "VmPin", "VmLck"]
        .iter()
        .filter_map(|key| {
            let line = status.lines().find(|l| l.starts_with(&format!("{key}:")))?;
            let kib = line.split_whitespace().nth(1)?;
            Some(format!("{}_kb={kib}", key.to_ascii_lowercase()))
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn catalog_refusal(detail: impl std::fmt::Display) -> Box<dyn std::error::Error> {
    ExpertBankRefusal(format!("experts-via-tier expert catalog refused: {detail}")).into()
}

fn hex(digest: &[u8]) -> String {
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

/// The routed-expert catalog of the loaded model, derived from its compiled plan and the
/// tensor contract its model pack compiles for a GGUF artifact, bound against the
/// artifact's tensor census. No checkpoint name is spelled here; a plan without MoE
/// projections, a contract entry that is missing, ambiguous or shape-incompatible, and a
/// scale plane the artifact carries for a bank are typed refusals.
fn plan_catalog(
    model: &HybridModel,
    gguf: &GgufFile,
) -> std::result::Result<ExpertBankCatalog, Box<dyn std::error::Error>> {
    let full_census = census_from_gguf(gguf);
    let pack = model_packs::for_config(&model.cfg);
    // Head ownership is the pack's declaration (memra#541), the same verdict the loader used.
    let output_head =
        memra_gguf::checkpoint_binding::output_head_for(pack, &model.cfg, &full_census)
            .map_err(|err| catalog_refusal(err.to_string()))?;
    let options = ContractOptions { output_head };
    let contract = match pack {
        Some(pack) => {
            pack.compile_tensor_contract(&model.cfg, &model.plan, CheckpointDialect::Gguf, options)
        }
        None => TensorContract::for_plan(&model.plan, CheckpointDialect::Gguf, options),
    }
    .map_err(|err| catalog_refusal(format!("tensor contract did not compile: {err}")))?;
    let census: Vec<TensorCensusEntry> = full_census
        .tensors
        .into_iter()
        .map(|row| row.entry)
        .collect();
    expert_bank_catalog(&model.plan, &contract, &census).map_err(catalog_refusal)
}

impl Engine {
    /// Explicit default-OFF qualification door for the approved exact artifact.
    /// Call after loading and before the first forward; keeps the existing native
    /// SLRU slot addresses, expert kernels and routing unchanged. Resident slabs,
    /// scale-bearing formats and parallel/frozen paths are intentionally refused.
    ///
    /// `budget` is the gate binary's parsed CLI budget (`ExpertBankBudget`); the installer
    /// reads no argv and no environment for it. A budget the bank cannot hold, or an expert
    /// catalog the plan and tensor contract cannot bind for the artifact, returns the typed
    /// `ExpertBankRefusal` before any bank, CUDA slot or source read exists.
    pub fn install_expert_bank_gate(
        &self,
        model: &HybridModel,
        gguf: &GgufFile,
        budget: ExpertBankBudget,
    ) -> std::result::Result<BankedExpertGate<'_>, Box<dyn std::error::Error>> {
        if gguf.n_shards() != 1 || !Engine::moe_cache_enabled() || !model.mtp_extra.is_empty() {
            return Err(
                "experts-via-tier requires one immutable GGUF, cache, and at most one MTP head"
                    .into(),
            );
        }
        let stage_clock = budget.stage_clock;
        let clock = |started: Instant| stage_clock.then(|| elapsed_ns(started));
        let install_started = Instant::now();
        // Authenticate the already-open inode, not a pathname reopened after load.
        let file = gguf.opened_file().clone();
        let mut h = Sha256::new();
        let mut buf = vec![0u8; 4 * 1024 * 1024];
        let len = file.metadata()?.len();
        let mut offset = 0;
        while offset < len {
            let n = buf.len().min((len - offset) as usize);
            file.read_exact_at(&mut buf[..n], offset)?;
            h.update(&buf[..n]);
            offset += n as u64;
        }
        let artifact: Digest = h.finalize().into();
        let sha_ns = clock(install_started);
        let actual = hex(&artifact);
        if actual != APPROVED_SHA {
            return Err("experts-via-tier artifact SHA256 mismatch".into());
        }
        // The catalog comes from the compiled plan and the artifact's tensor contract; the
        // loaded HostExps only supply the bytes and the router mask for each named bank.
        let catalog_started = Instant::now();
        let catalog = plan_catalog(model, gguf)?;
        let catalog_ns = clock(catalog_started);
        if model.layers.len() != model.plan.layers.len() {
            return Err(catalog_refusal(format!(
                "loaded model has {} layers, compiled plan has {}",
                model.layers.len(),
                model.plan.layers.len()
            )));
        }
        let reads = Rc::new(Cell::new(0));
        let pread_ns = stage_clock.then(|| Rc::new(Cell::new(0)));
        let mut reader = FileReader {
            file,
            ranges: BTreeMap::new(),
            reads: reads.clone(),
            pread_ns: pread_ns.clone(),
        };
        let records_started = Instant::now();
        let mut entries = vec![];
        let mut ids = BTreeMap::new();
        let mut max_bytes = 0;
        let mut records = Sha256::new();
        let mut record_count = 0u64;
        let mut banked_blocks = 0usize;
        let mut mtp_banked = false;
        // DAY44: the loaded banks' host storage, by class, over every retained projection.
        let mut storage: BTreeMap<&'static str, (usize, usize)> = BTreeMap::new();
        for block in catalog.blocks() {
            let Some(first) = block.first() else {
                return Err(Error::InvalidLayout.into());
            };
            if block.len() != memra_gguf::expert_banks::PROJECTIONS_PER_BLOCK
                || block.iter().any(|p| p.block != first.block)
            {
                return Err(Error::InvalidLayout.into());
            }
            let (layer, moe): (u16, &MoeWeights) = match first.block {
                ExpertBankBlock::Trunk { position } => match &model.layers[position].ffn {
                    Ffn::Moe(moe) => (u16::try_from(position)?, moe),
                    Ffn::Dense { .. } => {
                        return Err(catalog_refusal(format!(
                            "plan layer {position} routes experts but the loaded layer is dense"
                        )));
                    }
                },
                ExpertBankBlock::Mtp { depth } => match model.mtp.as_ref().map(|head| &head.ffn) {
                    // run-gen loads without the MTP head; the plan still names its bank.
                    None => continue,
                    Some(Ffn::Moe(moe)) if depth == 0 => {
                        mtp_banked = true;
                        (u16::MAX, moe)
                    }
                    Some(Ffn::Moe(_)) => {
                        return Err(catalog_refusal(format!(
                            "plan MTP block depth {depth} has no loaded head"
                        )));
                    }
                    Some(Ffn::Dense { .. }) => {
                        return Err(catalog_refusal(format!(
                            "plan MTP block depth {depth} routes experts but the loaded head is dense"
                        )));
                    }
                },
            };
            if moe.dev_exps.is_some()
                || moe.step_ep.is_some()
                || moe.step_tp.is_some()
                || moe.glm5_ep.is_some()
                || moe.glm5_tp_split.is_some()
            {
                return Err("experts-via-tier refuses resident or parallel expert bypasses; use the cache baseline".into());
            }
            banked_blocks += 1;
            for projection in block {
                let (proj, tier_projection, host) = match projection.projection {
                    ExpertTensor::Gate => (0u8, Projection::Gate, &moe.gate_exps),
                    ExpertTensor::Up => (1, Projection::Up, &moe.up_exps),
                    ExpertTensor::Down => (2, Projection::Down, &moe.down_exps),
                };
                let entry = storage.entry(host.bytes.storage_kind()).or_insert((0, 0));
                entry.0 += 1;
                entry.1 += host.bytes.len();
                bank_projection(
                    gguf,
                    artifact,
                    projection,
                    layer,
                    proj,
                    tier_projection,
                    host,
                    moe.active_experts.as_deref(),
                    &mut reader,
                    &mut entries,
                    &mut ids,
                    &mut max_bytes,
                    &mut records,
                    &mut record_count,
                )?;
            }
        }
        if let Some(head) = &model.mtp
            && matches!(head.ffn, Ffn::Moe(_))
            && !mtp_banked
        {
            return Err(catalog_refusal(
                "loaded MTP head routes experts but the compiled plan has no MTP expert bank",
            ));
        }
        if ids.is_empty() || max_bytes == 0 || max_bytes > 16 * 1024 * 1024 {
            return Err("experts-via-tier empty or oversized bank".into());
        }
        let records_ns = clock(records_started);
        let kind = |k: &str| storage.get(k).copied().unwrap_or((0, 0));
        let ((mm, mb), (pn, pb), (pg, gb)) = (kind("mmap"), kind("pinned"), kind("paged"));
        eprintln!(
            "[experts-via-tier] host expert storage: mmap={mm} ({mb} bytes) pinned={pn} ({pb} bytes) paged={pg} ({gb} bytes)"
        );
        let setup_started = Instant::now();
        eprintln!(
            "[experts-via-tier] catalog blocks={} banked={banked_blocks} projections={} catalog_sha256={} records={record_count} records_sha256={}",
            catalog.blocks().count(),
            catalog.projections.len(),
            hex(&Sha256::digest(catalog.identity().as_bytes())),
            hex(&records.finalize()),
        );
        // Gate-only CLI budgets. Without a GPU budget, native slot sizing (MEMRA_MOE_SLOTS or
        // auto) is untouched; with one, the exact count is fixed here, before any allocation,
        // and a MEMRA_MOE_SLOTS request alongside it is a conflict, never a silent loser.
        let host_bytes = budget.host_bytes;
        // Day 43: the host tier is planned per record size under the budget, refused above
        // three quarters of the host's MemAvailable read now (never an environment variable).
        let record_sizes = entries
            .iter()
            .filter_map(|(_, record)| record.as_ref())
            .map(|record| record.layout.storage_bytes())
            .collect::<std::result::Result<Vec<u64>, _>>()?;
        let meminfo = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
        let ceiling = crate::banked_residency::host_bank_ceiling(&meminfo).ok_or_else(|| {
            ExpertBankRefusal(
                "experts-via-tier host memory unknown: /proc/meminfo carries no MemAvailable"
                    .to_owned(),
            )
        })?;
        let plan = host_bank_budget(host_bytes, &record_sizes, ceiling)?;
        let slots = plan.records_held;
        eprintln!(
            "[experts-via-tier] host_bank_plan requested={host_bytes} planned={} classes={:?} records_held={slots} ceiling={ceiling}",
            plan.planned_bytes, plan.classes
        );
        let gpu_slots = match budget.gpu_bytes {
            Some(bytes) => {
                if std::env::var_os("MEMRA_MOE_SLOTS").is_some() {
                    return Err(ExpertBankRefusal(format!(
                        "experts-via-tier GPU bank budget conflicts with MEMRA_MOE_SLOTS (requested {bytes})"
                    ))
                    .into());
                }
                let (free, _total) = self.ctx().mem_get_info()?;
                let hard_bytes = crate::moe_cache::hard_slot_bytes(free, max_bytes as usize);
                let slots = gpu_bank_budget(bytes, max_bytes, hard_bytes as u64)?;
                eprintln!(
                    "[experts-via-tier] gpu_bank_budget bytes={bytes} slots={slots} hard_ceiling={hard_bytes}"
                );
                Some(slots)
            }
            None => None,
        };
        // DAY46: leases stay open until their copy event completes, so the registry, the bank's
        // tickets and the governor's in-flight and staging dimensions hold the in-flight queue
        // plus the demand being admitted.
        let open_leases = crate::moe_cache::BANKED_INFLIGHT + 1;
        let mut capacity = TierBudget::zero(1);
        // The planned payload, a per-record metadata allowance and the previous fixed 512 MiB
        // as headroom for staging and open tickets.
        capacity.pageable = plan
            .planned_bytes
            .checked_add(slots as u64 * 4096)
            .and_then(|n| n.checked_add(512 * 1024 * 1024))
            .ok_or(Error::Overflow)?;
        capacity.staging = max_bytes
            .checked_mul(open_leases as u64)
            .ok_or(Error::Overflow)?;
        capacity.inflight = open_leases as u64;
        let budget: SharedBudget = Rc::new(RefCell::new(Governor::new(
            capacity,
            TierBudget::zero(1),
            1,
            0,
            Arc::new(|| 0),
        )?));
        // DAY45: the host fill's jobs, one per retained record in catalog order, each the record's
        // exact range in the authenticated inode (the reader's tensor start plus the segment's
        // offset, the position `ReadWork` would read). Door records are single-segment payload.
        let fill_file = reader.file.clone();
        let mut fill_jobs = Vec::with_capacity(entries.len());
        for (id, record) in &entries {
            let Some(record) = record else { continue };
            let [segment] = record.layout.segments.as_slice() else {
                return Err(Error::InvalidLayout.into());
            };
            let tensor = segment.tensor.as_ref().ok_or(Error::InvalidLayout)?;
            let &(start, _) = reader.ranges.get(tensor).ok_or(Error::NotFound)?;
            fill_jobs.push(FillJob {
                local: dispatch_id(&id.record)?,
                offset: start.checked_add(segment.offset).ok_or(Error::Overflow)?,
                len: usize::try_from(segment.storage_bytes)?,
            });
        }
        let bank: BankService<ExpertDomain, Heat, FileReader> = BankService::new(
            Catalog::new(LayoutClass::PerRecord, entries)?,
            budget.clone(),
            Heat,
            reader,
            CoalescingPolicy {
                granularity: 1,
                slot_bytes: max_bytes,
            },
            BankLimits {
                cache_bytes: plan.planned_bytes,
                batch_bytes: max_bytes,
                items: 1,
                tickets: open_leases,
            },
        )?;
        let bank = if stage_clock {
            bank.with_stage_clock()
        } else {
            bank
        };
        // DAY47: the host tier lives in one cached pinned pool; every read lands in a buffer
        // from it, and the fill takes from it above the demand reserve.
        let pool = PinnedPool::new(self.ctx().clone(), &plan.classes)?;
        eprintln!(
            "[experts-via-tier] host pinned pool bytes={} classes={} headroom={POOL_HEADROOM} fill_reserve={FILL_RESERVE}",
            pool.bytes,
            pool.classes.len()
        );
        let bank = bank.with_host_buffers(Box::new(PoolSource(pool.clone())))?;
        let fill_pool = pool.clone();
        let fill_buffers: FillBuffers = Arc::new(move |len| {
            fill_pool
                .take(len, FILL_RESERVE)
                .map(|buffer| HostBytes::Pooled(Box::new(buffer)))
        });
        let mut request = BudgetRequest {
            bytes: TierBudget::zero(1),
            priority: Priority::Demand,
            deadline: Deadline(u64::MAX),
            tenant: artifact,
        };
        request.bytes.pageable = bank.slru_metadata_bytes(slots)?;
        let metadata = budget.borrow_mut().reserve(&request)?;
        let bank = bank.with_slru(SlruPolicy::new(&plan.classes)?, &metadata)?;
        request.bytes = TierBudget::zero(1);
        let dispatch = SlruExpertDispatch::new(
            bank,
            ids.clone(),
            request,
            Epochs {
                state: 0,
                src_gen: 0,
                dst_gen: 0,
            },
        )?;
        let fill_counts = Arc::new(FillCounts::default());
        let (fill_workers, fill_intake) =
            start_fill(fill_file, fill_jobs, fill_counts.clone(), fill_buffers);
        let owner = ExpertBankOwner::register(
            Box::new(TracedDispatch {
                inner: dispatch,
                ids,
                occupants: BTreeMap::new(),
                clock: stage_clock.then(|| OwnerClock {
                    pread_ns: pread_ns.clone().unwrap_or_default(),
                    reads: reads.clone(),
                    ..OwnerClock::default()
                }),
                fill: Some(fill_intake),
            }),
            open_leases,
        );
        let owner = owner?;
        if let Some(slots) = gpu_slots {
            self.build_moe_cache_exact(max_bytes as usize, slots)?;
        }
        self.with_moe_cache(max_bytes as usize, |cache, _| {
            cache.install_banked(owner.proxy(), stage_clock)
        })?;
        if let (Some(sha), Some(catalog), Some(records), Some(setup)) =
            (sha_ns, catalog_ns, records_ns, clock(setup_started))
        {
            eprintln!(
                "[experts-via-tier] install sha_ns={sha} catalog_ns={catalog} records_ns={records} setup_ns={setup}"
            );
        }
        eprintln!(
            "[experts-via-tier] installed artifact_sha256={actual} host_slots={slots} max_expert_bytes={max_bytes}"
        );
        Ok(BankedExpertGate {
            engine: self,
            max_bytes: max_bytes as usize,
            owner,
            reads,
            budget,
            metadata,
            stage_clock,
            fill: Some(fill_workers),
            fill_counts,
        })
    }
}

/// Catalog one projection bank: every retained expert record of the tensor the catalog
/// named, checked byte-for-byte against the loaded HostExps, with its checksum folded into
/// the running records digest. Scale planes on the loaded bank are a typed refusal: the
/// native installer consumes payload bytes only.
#[allow(clippy::too_many_arguments)] // allow: the installer's accumulators are threaded through one call per projection rather than a struct that outlives the loop
fn bank_projection(
    gguf: &GgufFile,
    artifact: Digest,
    projection: &ExpertBankProjection,
    layer: u16,
    proj: u8,
    tier_projection: Projection,
    host: &crate::model::HostExps,
    active_experts: Option<&[bool]>,
    reader: &mut FileReader,
    entries: &mut Vec<(BankId, Option<CatalogRecord>)>,
    ids: &mut BTreeMap<ExpertDispatchId, BankId>,
    max_bytes: &mut u64,
    records: &mut Sha256,
    record_count: &mut u64,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    if host.macros.is_some() || host.fp8_blk.is_some() {
        return Err(catalog_refusal(format!(
            "loaded bank {} carries scale planes (macro or block scales) the native installer does not consume",
            projection.name
        )));
    }
    if host.n_expert != projection.expert_count as usize {
        return Err(catalog_refusal(format!(
            "plan routes {} over {} experts but the loaded bank holds {}",
            projection.name, projection.expert_count, host.n_expert
        )));
    }
    let info = gguf
        .find(&projection.name)
        .ok_or("missing immutable expert tensor")?;
    if info.n_bytes != projection.physical_bytes {
        return Err("catalog tensor bytes differ from the GGUF tensor table".into());
    }
    let raw = gguf.tensor_data(info);
    let (start, _) = gguf.tensor_file_range(info);
    let tensor = TensorId {
        version: WIRE_VERSION,
        artifact,
        name: projection.name.clone(),
    };
    // The loader's original-width router mask when an overlay pruned experts, else every
    // expert of the bank. Never inferred from a dense repacked vector.
    let active: Vec<bool> = active_experts
        .map(<[bool]>::to_vec)
        .unwrap_or_else(|| vec![true; host.n_expert]);
    let mut sources = BTreeMap::new();
    for (expert, &retained) in active.iter().enumerate() {
        if !retained {
            continue;
        }
        let layout = host.expert_layout(expert);
        let bytes = raw
            .get(
                layout.offset
                    ..layout
                        .offset
                        .checked_add(layout.len)
                        .ok_or(Error::Overflow)?,
            )
            .ok_or(Error::InvalidLayout)?;
        if bytes != host.expert_bytes(expert) {
            return Err("native expert bytes differ from pinned GGUF".into());
        }
        let mut source = tensor.clone();
        if host.tiers.is_some() {
            source.name = format!("{}.expert.{expert}", tensor.name);
            reader.ranges.insert(
                source.clone(),
                ((start + layout.offset) as u64, layout.len as u64),
            );
        } else {
            reader
                .ranges
                .insert(source.clone(), (start as u64, raw.len() as u64));
        }
        let digest = checksum(bytes);
        records.update(digest);
        *record_count += 1;
        sources.insert(
            expert as u32,
            ExpertSource {
                tensor: source,
                split: host.tiers.is_some(),
                scales: vec![],
                checksums: vec![digest],
            },
        );
    }
    let mapping = crate::banked_residency::map_host_exps(
        host,
        tensor,
        u32::from(layer),
        tier_projection,
        &active,
        &sources,
    )?;
    *max_bytes = (*max_bytes).max(mapping.max_expert_bytes);
    for id in mapping.ids {
        let RecordId::Expert { original_id, .. } = id.record else {
            return Err(Error::InvalidLayout.into());
        };
        let record = if active[original_id as usize] {
            Some(mapping.catalog.record(&id)?.clone())
        } else {
            None
        };
        ids.insert((layer, proj, u16::try_from(original_id)?), id.clone());
        entries.push((id, record));
    }
    Ok(())
}

// Complete host-bank demand trace, not a model-route or GPU-hit trace. IDs and
// extents are validated by the exact catalog; the bounded occupant map has at
// most host_slots entries. No numeric data or machine identity is logged.
struct TracedDispatch {
    inner: SlruExpertDispatch<Heat, FileReader>,
    ids: BTreeMap<ExpertDispatchId, BankId>,
    occupants: BTreeMap<usize, ExpertDispatchId>,
    /// `--expert-bank-stages` only (DAY40): the owner side of the door's stage clock.
    clock: Option<OwnerClock>,
    /// DAY45: finished host fills, admitted at the start of each demand.
    fill: Option<FillIntake>,
}
impl TracedDispatch {
    /// Offer up to `limit` finished fills to the bank (DAY45 section 1 (b)); a full tier raises
    /// the fill's stop flag.
    fn drain_fill(&mut self, limit: usize) {
        let Some(fill) = &self.fill else { return };
        let started = Instant::now();
        for _ in 0..limit {
            let Ok(done) = fill.rx.try_recv() else { break };
            let local = done.local;
            match self.inner.admit_filled(local, done.bytes, done.digest) {
                Ok(FillOutcome::Admitted) => {
                    fill.counts.admitted.fetch_add(1, Ordering::Relaxed);
                }
                Ok(FillOutcome::Dropped) => {
                    fill.counts.dropped.fetch_add(1, Ordering::Relaxed);
                }
                Ok(FillOutcome::Full) => {
                    fill.counts.dropped.fetch_add(1, Ordering::Relaxed);
                    fill.stop.store(true, Ordering::Relaxed);
                }
                Ok(FillOutcome::Refused) => {
                    fill.counts.refused.fetch_add(1, Ordering::Relaxed);
                    eprintln!(
                        "[experts-via-tier] fill refused {}:{}:{}: checksum mismatch",
                        local.0, local.1, local.2
                    );
                }
                Err(err) => {
                    if fill.counts.errors.fetch_add(1, Ordering::Relaxed) == 0 {
                        eprintln!(
                            "[experts-via-tier] fill admission error {}:{}:{}: {err:?}",
                            local.0, local.1, local.2
                        );
                    }
                }
            }
        }
        fill.counts
            .admit_ns
            .fetch_add(elapsed_ns(started), Ordering::Relaxed);
    }
}
/// Owner-thread half of the stage clock: host-tier hits and misses, the inner demand, the
/// trace print, and the reader's positioned reads (shared with `FileReader`).
#[derive(Default)]
struct OwnerClock {
    host_hits: u64,
    host_misses: u64,
    inner_demand_ns: u64,
    trace_ns: u64,
    pread_ns: Rc<Cell<u64>>,
    reads: Rc<Cell<u64>>,
}
impl ExpertDispatchBank for TracedDispatch {
    fn validate(&self, local: ExpertDispatchId, bytes: usize) -> Result<()> {
        self.inner.validate(local, bytes)
    }
    fn demand(&mut self, local: ExpertDispatchId, bytes: usize) -> Result<ExpertDemand> {
        self.drain_fill(32);
        let id = self.ids.get(&local).ok_or(Error::NotFound)?;
        let hit = self
            .inner
            .bank()
            .slru_policy()
            .ok_or(Error::Incomplete)?
            .resident(id)
            .is_some();
        let demand_started = self.clock.as_ref().map(|_| Instant::now());
        let demand = self.inner.demand(local, bytes);
        if let (Some(started), Some(clock)) = (demand_started, self.clock.as_mut()) {
            clock.inner_demand_ns = clock.inner_demand_ns.saturating_add(elapsed_ns(started));
            if hit {
                clock.host_hits += 1;
            } else {
                clock.host_misses += 1;
            }
        }
        let demand = demand?;
        let slot = self
            .inner
            .bank()
            .slru_policy()
            .ok_or(Error::Incomplete)?
            .resident(id)
            .ok_or(Error::Incomplete)?;
        let victim = self
            .occupants
            .insert(slot, local)
            .filter(|old| *old != local);
        let trace_started = self.clock.as_ref().map(|_| Instant::now());
        eprintln!(
            "[expert-host-slru] key={}:{}:{} bytes={} slot={} hit={} victim={}",
            local.0,
            local.1,
            local.2,
            bytes,
            slot,
            hit,
            victim
                .map(|v| format!("{}:{}:{}", v.0, v.1, v.2))
                .unwrap_or_else(|| "-".into())
        );
        if let (Some(started), Some(clock)) = (trace_started, self.clock.as_mut()) {
            clock.trace_ns = clock.trace_ns.saturating_add(elapsed_ns(started));
        }
        Ok(demand)
    }
    fn finish(&mut self, demand: ExpertDemand) -> Result<()> {
        self.inner.finish(demand)
    }
    fn stage_report(&self) -> Option<String> {
        let clock = self.clock.as_ref()?;
        let fill = self
            .fill
            .as_ref()
            .map(|f| f.counts.line())
            .unwrap_or_else(|| "fill absent".to_owned());
        Some(format!(
            "| owner host_hits={} host_misses={} inner_demand_ns={} trace_ns={} reads={} pread_ns={} | {fill} | bank {}",
            clock.host_hits,
            clock.host_misses,
            clock.inner_demand_ns,
            clock.trace_ns,
            clock.reads.get(),
            clock.pread_ns.get(),
            self.inner
                .stage_report()
                .unwrap_or_else(|| "absent".to_owned())
        ))
    }
}

#[cfg(test)]
mod day44_census {
    //! DAY44 (`research/spill-c-20260919/DAY44.md`): the door's mapped-expert load option is a
    //! gate-binary statement and the loader's mapped branch sits behind it, nowhere else.
    const MODEL: &str = include_str!("../model.rs");
    const RUN_GEN: &str = include_str!("../bin/run_gen.rs");
    const RUN_SPEC: &str = include_str!("../bin/run_spec.rs");
    const LIB: &str = include_str!("../lib.rs");

    #[test]
    fn the_mapped_branch_is_taken_only_under_the_door_option() {
        let branch = "None if e.expert_host_mapped() => Some(door_mapped_extent(src, name)?),";
        assert_eq!(
            MODEL.matches(branch).count(),
            1,
            "one guarded mapped branch"
        );
        assert_eq!(
            MODEL.matches("door_mapped_extent(").count(),
            2,
            "the helper's definition and its one guarded call"
        );
        // The split-tensor copy path refuses under the option instead of pinning.
        assert!(
            MODEL.contains("the door's mapped expert banks do not cover a split stacked tensor")
        );
        // Only the gate binaries set the option, from the parsed door, before the model loads.
        let set = "e.set_expert_host_mapped(expert_bank.is_some());";
        assert_eq!(RUN_GEN.matches(set).count(), 1);
        assert_eq!(RUN_SPEC.matches(set).count(), 1);
        for (name, src) in [("run_gen", RUN_GEN), ("run_spec", RUN_SPEC)] {
            let at = src.find(set).unwrap();
            let load = src.find("HybridModel::load").unwrap();
            assert!(at < load, "{name} sets the option after a load");
        }
        assert_eq!(LIB.matches("fn set_expert_host_mapped").count(), 1);
        assert!(LIB.contains("expert_host_mapped: std::sync::atomic::AtomicBool::new(false),"));
    }
}

#[cfg(test)]
mod day45_fill {
    //! DAY45: the fill workers read each job's exact range, digest it with the contract
    //! `checksum`, hand the bytes over by value, and stop and join promptly even when the owner
    //! never drains the channel.
    use super::*;
    use std::io::Write as _;

    fn heap() -> FillBuffers {
        Arc::new(|len| Some(HostBytes::Heap(vec![0u8; len])))
    }

    fn artifact(len: usize) -> (std::path::PathBuf, Vec<u8>) {
        let bytes: Vec<u8> = (0..len).map(|i| (i * 31 % 251) as u8).collect();
        let path = std::env::temp_dir().join(format!("c45-fill-{}-{len}.bin", std::process::id()));
        std::fs::File::create(&path)
            .unwrap()
            .write_all(&bytes)
            .unwrap();
        (path, bytes)
    }

    #[test]
    fn the_fill_reads_each_range_and_digests_it() {
        let (path, bytes) = artifact(1 << 16);
        let file = Arc::new(File::open(&path).unwrap());
        let ranges = [(3u64, 4096usize), (10_000, 777), (40_000, 12_345)];
        let jobs = ranges
            .iter()
            .enumerate()
            .map(|(i, &(offset, len))| FillJob {
                local: (0, 0, i as u16),
                offset,
                len,
            })
            .collect();
        let counts = Arc::new(FillCounts::default());
        let (workers, intake) = start_fill(file, jobs, counts.clone(), heap());
        let mut got = BTreeMap::new();
        for _ in 0..ranges.len() {
            let done = intake
                .rx
                .recv_timeout(std::time::Duration::from_secs(10))
                .unwrap();
            got.insert(done.local.2, (done.bytes, done.digest));
        }
        for (i, &(offset, len)) in ranges.iter().enumerate() {
            let want = &bytes[offset as usize..offset as usize + len];
            let (b, d) = &got[&(i as u16)];
            assert_eq!(b.bytes(), want);
            assert_eq!(*d, checksum(want));
        }
        drop(workers);
        assert_eq!(counts.reads.load(Ordering::Relaxed), ranges.len() as u64);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn workers_stop_and_join_with_an_undrained_channel() {
        let (path, _) = artifact(1 << 16);
        let file = Arc::new(File::open(&path).unwrap());
        let jobs = (0..10_000u32)
            .map(|i| FillJob {
                local: (1, 2, (i % 60_000) as u16),
                offset: u64::from(i % 60),
                len: 1024,
            })
            .collect();
        let counts = Arc::new(FillCounts::default());
        let (workers, _intake) = start_fill(file, jobs, counts.clone(), heap());
        std::thread::sleep(std::time::Duration::from_millis(50));
        let started = Instant::now();
        drop(workers);
        assert!(
            started.elapsed() < std::time::Duration::from_secs(2),
            "the join hung"
        );
        // The channel holds at most 64, so the workers stopped long before the 10,000 jobs.
        assert!(counts.reads.load(Ordering::Relaxed) < 10_000);
        std::fs::remove_file(path).ok();
    }
}

#[cfg(test)]
mod day46_census {
    //! DAY46 (`research/spill-c-20260919/DAY46.md`): the door's miss path drains nothing on
    //! success. Its only stream drains are the two error branches of `stage_banked` and the
    //! teardown's `retire_all_banked`; retirement waits on the oldest copy's event, never the
    //! stream; every `finish` of a copied lease sits behind an event check or the teardown drain.
    const CACHE: &str = include_str!("../moe_cache.rs");

    fn body(name: &str) -> &'static str {
        let start = CACHE
            .find(&format!("fn {name}("))
            .unwrap_or_else(|| panic!("{name} missing"));
        let rest = &CACHE[start..];
        let end = rest[1..]
            .find("\n    fn ")
            .or_else(|| rest[1..].find("\n    pub(crate) fn "))
            .map(|i| i + 1)
            .unwrap_or(rest.len());
        &rest[..end]
    }

    #[test]
    fn the_miss_path_drains_nothing_on_success() {
        let admit = body("admit_banked");
        assert!(!admit.contains("synchronize"), "admit_banked drains");
        assert!(admit.contains("self.retire_banked(&bank)?;"));
        assert!(admit.contains("self.banked_inflight.push_back((token, done));"));
        let stage = body("stage_banked");
        assert_eq!(
            stage.matches("e.stream().synchronize()").count(),
            2,
            "stage_banked drains beyond its two error branches"
        );
        assert!(stage.contains("e.stream().record_event(None)"));
        let retire = body("retire_banked");
        assert!(
            !retire.contains("stream().synchronize()"),
            "retire waits on the stream"
        );
        assert!(retire.contains("done.synchronize()?;"));
        assert!(retire.contains("if !done.is_complete()"));
        let teardown = body("retire_all_banked");
        assert_eq!(
            teardown
                .matches("self.compute_stream.synchronize()?;")
                .count(),
            1
        );
        assert!(CACHE.contains("pub(crate) const BANKED_INFLIGHT: usize = 32;"));
    }
}
