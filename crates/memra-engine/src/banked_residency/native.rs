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
        self.engine.set_expert_bank_prefetch(false);
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
        let mut traced = TracedDispatch {
            inner: dispatch,
            ids,
            occupants: BTreeMap::new(),
            clock: stage_clock.then(|| OwnerClock {
                pread_ns: pread_ns.clone().unwrap_or_default(),
                reads: reads.clone(),
                ..OwnerClock::default()
            }),
            fill: Some(fill_intake),
            trace: String::with_capacity(TRACE_CHUNK + 256),
        };
        // DAY57 (I10): the fill completes inside the install, as the legacy's pinned host copy
        // completes inside its load, so no decode demand races it.
        let fill_started = Instant::now();
        if traced.complete_fill(FILL_WAIT_STALL) {
            eprintln!(
                "[experts-via-tier] fill complete before decode in {:.1} ms: {}",
                fill_started.elapsed().as_secs_f64() * 1e3,
                fill_counts.line()
            );
        } else {
            eprintln!(
                "[experts-via-tier] fill wait stopped: no completion in {} s ({})",
                FILL_WAIT_STALL.as_secs(),
                fill_counts.line()
            );
        }
        let owner = ExpertBankOwner::register(Box::new(traced), open_leases);
        let owner = owner?;
        if let Some(slots) = gpu_slots {
            self.build_moe_cache_exact(max_bytes as usize, slots)?;
        }
        self.with_moe_cache(max_bytes as usize, |cache, _| {
            cache.install_banked(owner.proxy(), stage_clock)
        })?;
        self.set_expert_bank_prefetch(true);
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

/// DAY49: threads for the installer's record pass, the fill's bound (`min(8, cores / 2)`).
fn record_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get() / 2)
        .unwrap_or(1)
        .clamp(1, 8)
}

/// DAY49: the order-independent per-expert work of the record pass (the byte compare of the
/// artifact's bytes with the loaded bank's, and the contract `checksum`) on `threads` scoped
/// threads over contiguous expert ranges. `Ok` holds each expert's digest in expert order
/// (`None` for a masked expert); `Err` names the first expert, in expert order, whose bytes
/// differ, whichever thread saw it.
fn record_digests(
    pairs: &[Option<(&[u8], &[u8])>],
    threads: usize,
) -> std::result::Result<Vec<Option<Digest>>, usize> {
    let chunk = pairs.len().div_ceil(threads.max(1)).max(1);
    let parts: Vec<Vec<std::result::Result<Option<Digest>, ()>>> = std::thread::scope(|scope| {
        let handles: Vec<_> = pairs
            .chunks(chunk)
            .map(|part| {
                scope.spawn(move || {
                    part.iter()
                        .map(|pair| match pair {
                            None => Ok(None),
                            Some((artifact, loaded)) if artifact == loaded => {
                                Ok(Some(checksum(artifact)))
                            }
                            Some(_) => Err(()),
                        })
                        .collect()
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("record pass thread panicked"))
            .collect()
    });
    let mut out = Vec::with_capacity(pairs.len());
    for (expert, result) in parts.into_iter().flatten().enumerate() {
        match result {
            Ok(digest) => out.push(digest),
            Err(()) => return Err(expert),
        }
    }
    Ok(out)
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
    // DAY49: the per-expert compare and checksum are independent; they run on scoped threads,
    // and everything order-dependent below stays serial in expert order.
    let mut pairs: Vec<Option<(&[u8], &[u8])>> = Vec::with_capacity(active.len());
    let mut slicing_error = None;
    for (expert, &retained) in active.iter().enumerate() {
        if !retained {
            pairs.push(None);
            continue;
        }
        let layout = host.expert_layout(expert);
        let bytes = layout
            .offset
            .checked_add(layout.len)
            .ok_or(Error::Overflow)
            .and_then(|end| raw.get(layout.offset..end).ok_or(Error::InvalidLayout));
        match bytes {
            Ok(bytes) => pairs.push(Some((bytes, host.expert_bytes(expert)))),
            Err(err) => {
                slicing_error = Some(err);
                break;
            }
        }
    }
    // A mismatch before a slicing error refuses first, as the serial pass would.
    let digests = record_digests(&pairs, record_threads())
        .map_err(|_| "native expert bytes differ from pinned GGUF")?;
    if let Some(err) = slicing_error {
        return Err(err.into());
    }
    let mut sources = BTreeMap::new();
    for (expert, &retained) in active.iter().enumerate() {
        if !retained {
            continue;
        }
        let layout = host.expert_layout(expert);
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
        let digest = digests[expert].ok_or(Error::InvalidLayout)?;
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
    /// DAY48 (I5): the host-demand trace, byte for byte as the unbuffered lines were, written to
    /// stderr in one call whenever it passes `TRACE_CHUNK` (always at a line end) and at close.
    trace: String,
}
/// DAY48: bytes of trace buffered before one stderr write.
const TRACE_CHUNK: usize = 64 * 1024;
impl Drop for TracedDispatch {
    fn drop(&mut self) {
        self.flush_trace();
    }
}
impl TracedDispatch {
    fn flush_trace(&mut self) {
        if !self.trace.is_empty() {
            use std::io::Write as _;
            let _ = std::io::stderr().write_all(self.trace.as_bytes());
            self.trace.clear();
        }
    }
    /// Offer up to `limit` finished fills to the bank (DAY45 section 1 (b)); a full tier raises
    /// the fill's stop flag.
    fn drain_fill(&mut self, limit: usize) {
        let Some(fill) = &self.fill else { return };
        let started = Instant::now();
        for _ in 0..limit {
            let Ok(done) = fill.rx.try_recv() else { break };
            admit_one_fill(&mut self.inner, fill, done);
        }
        fill.counts
            .admit_ns
            .fetch_add(elapsed_ns(started), Ordering::Relaxed);
    }
    /// DAY57 (I10): admit finished fills until every worker has exited (the channel
    /// disconnects: jobs exhausted, the tier full, or a read error), so decode never races the
    /// fill. Bounded: no completion for `stall` stops the wait and leaves the rest of the fill to
    /// the demand path's `drain_fill`, as before. Returns whether the fill completed.
    fn complete_fill(&mut self, stall: std::time::Duration) -> bool {
        let Some(fill) = &self.fill else { return true };
        let inner = &mut self.inner;
        complete_fill_with(fill, stall, |done| admit_one_fill(inner, fill, done))
    }
}

/// DAY57 (I10): the installer's wait, over any admission (the door's is `admit_one_fill`): every
/// finished fill is admitted in arrival order until the channel disconnects (`true`), or no
/// completion arrives for `stall` (`false`). Owner-side admission time lands in `admit_ns`.
fn complete_fill_with(
    fill: &FillIntake,
    stall: std::time::Duration,
    mut admit: impl FnMut(FillDone),
) -> bool {
    loop {
        match fill.rx.recv_timeout(stall) {
            Ok(done) => {
                let started = Instant::now();
                admit(done);
                fill.counts
                    .admit_ns
                    .fetch_add(elapsed_ns(started), Ordering::Relaxed);
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => return true,
            Err(mpsc::RecvTimeoutError::Timeout) => return false,
        }
    }
}

/// One finished fill offered to the bank (DAY45 section 1 (b)): the admission both the demand
/// path's `drain_fill` and the installer's `complete_fill` (DAY57) run. A full tier raises the
/// fill's stop flag; a refused checksum and the first admission error print their line.
fn admit_one_fill(
    inner: &mut SlruExpertDispatch<Heat, FileReader>,
    fill: &FillIntake,
    done: FillDone,
) {
    let local = done.local;
    match inner.admit_filled(local, done.bytes, done.digest) {
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

/// DAY58 (I5f): one host-demand trace line, byte for byte the line day 48 printed
/// (`[expert-host-slru] key={}:{}:{} bytes={} slot={} hit={} victim={}` with the victim as
/// `layer:proj:expert` or `-`), written with direct pushes: no formatting machinery, no allocation.
fn push_trace_line(
    out: &mut String,
    key: ExpertDispatchId,
    bytes: usize,
    slot: usize,
    hit: bool,
    victim: Option<ExpertDispatchId>,
) {
    let key3 = |out: &mut String, k: ExpertDispatchId| {
        push_decimal(out, u64::from(k.0));
        out.push(':');
        push_decimal(out, u64::from(k.1));
        out.push(':');
        push_decimal(out, u64::from(k.2));
    };
    out.push_str("[expert-host-slru] key=");
    key3(out, key);
    out.push_str(" bytes=");
    push_decimal(out, bytes as u64);
    out.push_str(" slot=");
    push_decimal(out, slot as u64);
    out.push_str(if hit {
        " hit=true victim="
    } else {
        " hit=false victim="
    });
    match victim {
        Some(v) => key3(out, v),
        None => out.push('-'),
    }
    out.push('\n');
}

/// DAY58: `n` in decimal, as `{}` prints it.
fn push_decimal(out: &mut String, mut n: u64) {
    let mut digits = [0u8; 20];
    let mut at = digits.len();
    loop {
        at -= 1;
        digits[at] = b'0' + (n % 10) as u8;
        n /= 10;
        if n == 0 {
            break;
        }
    }
    for &d in &digits[at..] {
        out.push(char::from(d));
    }
}

/// DAY57 (I10): how long the installer waits for the next finished fill before it stops waiting.
const FILL_WAIT_STALL: std::time::Duration = std::time::Duration::from_secs(10);
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
        let before = self
            .inner
            .bank()
            .slru_policy()
            .ok_or(Error::Incomplete)?
            .resident(id);
        let hit = before.is_some();
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
        // DAY61 (I11 change 5): a host hit keeps its record's slot (publish's SLRU `hit` relinks
        // the queues, never the slot), so the pre-demand lookup is the slot; a miss reads the
        // slot its publication reserved.
        let slot = match before {
            Some(slot) => slot,
            None => self
                .inner
                .bank()
                .slru_policy()
                .ok_or(Error::Incomplete)?
                .resident(id)
                .ok_or(Error::Incomplete)?,
        };
        let victim = self
            .occupants
            .insert(slot, local)
            .filter(|old| *old != local);
        let trace_started = self.clock.as_ref().map(|_| Instant::now());
        push_trace_line(&mut self.trace, local, bytes, slot, hit, victim);
        if self.trace.len() >= TRACE_CHUNK {
            self.flush_trace();
        }
        if let (Some(started), Some(clock)) = (trace_started, self.clock.as_mut()) {
            clock.trace_ns = clock.trace_ns.saturating_add(elapsed_ns(started));
        }
        Ok(demand)
    }
    fn finish(&mut self, demand: ExpertDemand) -> Result<()> {
        self.inner.finish(demand)
    }
    fn host_resident(&self, local: ExpertDispatchId) -> Result<bool> {
        self.inner.host_resident(local)
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

    /// DAY57 (I10): the installer's wait admits every job once and returns when the workers
    /// have exited; a fill that cannot progress (no buffer ever) stops the wait at the stall
    /// bound, and the workers still stop and join.
    #[test]
    fn the_installer_wait_admits_every_fill_then_returns_and_a_stall_is_bounded() {
        // Its own file (the path is keyed by length): the module's tests run in parallel.
        let (path, _) = artifact((1 << 16) + 57);
        let jobs = |n: u32| {
            (0..n)
                .map(|i| FillJob {
                    local: (0, 1, i as u16),
                    offset: u64::from(i % 60),
                    len: 1024,
                })
                .collect::<Vec<_>>()
        };
        let counts = Arc::new(FillCounts::default());
        let file = Arc::new(File::open(&path).unwrap());
        let (workers, intake) = start_fill(file, jobs(500), counts.clone(), heap());
        let mut seen = std::collections::BTreeSet::new();
        let done = complete_fill_with(&intake, std::time::Duration::from_secs(10), |d| {
            assert!(seen.insert(d.local.2), "a job admitted twice");
        });
        assert!(done, "the wait returns when every worker has exited");
        assert_eq!(seen.len(), 500);
        assert_eq!(counts.reads.load(Ordering::Relaxed), 500);
        drop(workers);
        // No buffer is ever available: the workers spin on the pool until stopped.
        let counts = Arc::new(FillCounts::default());
        let file = Arc::new(File::open(&path).unwrap());
        let never: FillBuffers = Arc::new(|_| None);
        let (workers, intake) = start_fill(file, jobs(10), counts.clone(), never);
        let started = Instant::now();
        let done = complete_fill_with(&intake, std::time::Duration::from_millis(200), |_| {
            panic!("nothing can complete")
        });
        assert!(!done, "a stalled fill stops the wait");
        assert!(started.elapsed() < std::time::Duration::from_secs(5));
        let joined = Instant::now();
        drop(workers);
        assert!(
            joined.elapsed() < std::time::Duration::from_secs(2),
            "the join hung"
        );
        std::fs::remove_file(path).ok();
    }

    /// DAY57 census: the demand path and the installer admit through the one function, and no
    /// other code calls the bank's `admit_filled`.
    #[test]
    fn both_fill_admissions_share_one_function() {
        let src = include_str!("native.rs");
        let code = &src[..src.find("#[cfg(test)]").expect("the tests")];
        assert_eq!(
            code.matches("admit_one_fill(").count(),
            3,
            "definition and two callers"
        );
        assert_eq!(
            code.matches(".admit_filled(").count(),
            1,
            "only admit_one_fill admits"
        );
        let drain = &code[code.find("    fn drain_fill(").unwrap()..];
        let drain = &drain[..drain.find("\n    }\n").unwrap()];
        assert!(drain.contains("admit_one_fill(&mut self.inner, fill, done)"));
        let complete = &code[code.find("    fn complete_fill(").unwrap()..];
        let complete = &complete[..complete.find("\n    }\n").unwrap()];
        assert!(complete.contains("admit_one_fill(inner, fill, done)"));
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
        // DAY50 wrapped the event in an `Arc` (a prefetch's copy event is shared with `pending`).
        assert!(admit.contains("self.banked_inflight.push_back((token, Arc::new(done)));"));
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

#[cfg(test)]
mod day48_census {
    //! DAY48: the validate memo admits a pair only after the proxy accepted it, and a pair it
    //! does not hold still reaches the proxy.
    const CACHE: &str = include_str!("../moe_cache.rs");

    #[test]
    fn the_memo_holds_only_pairs_the_proxy_accepted() {
        // DAY58 (I8f): the dense memo's guard.
        let guarded = "        if !self.banked_validated.holds(id, bytes) {\n            bank.validate(local, bytes)?;\n            self.banked_validated.insert(id, bytes);\n        }";
        // Two sites since DAY50 (the demand and the door's prefetch), each behind the guard.
        assert_eq!(
            CACHE.matches(guarded).count(),
            2,
            "the memo's guarded insertions"
        );
        assert_eq!(CACHE.matches("banked_validated.insert(").count(), 2);
        assert_eq!(
            CACHE.matches("bank.validate(").count(),
            2,
            "validate call sites"
        );
    }
}

#[cfg(test)]
mod day48_trace_census {
    //! DAY48 (I5): the host-demand trace keeps its line format and has no unbuffered print left;
    //! the buffer is flushed at the chunk bound and when the dispatch closes.
    const SRC: &str = include_str!("native.rs");

    #[test]
    fn the_trace_is_buffered_with_its_format_unchanged() {
        // DAY58 (I5f): the line is written by `push_trace_line` alone; its bytes are pinned against
        // day 48's format by `the_direct_trace_line_is_day_48s_format_byte_for_byte`.
        let code = &SRC[..SRC.find("#[cfg(test)]").expect("the tests")];
        assert_eq!(code.matches("push_trace_line(&mut self.trace, ").count(), 1);
        assert_eq!(
            code.matches("writeln!(\n                self.trace")
                .count(),
            0
        );
        let unbuffered = concat!("eprintln!(\n", "            \"[expert-host-slru]");
        assert_eq!(
            SRC.matches(unbuffered).count(),
            0,
            "an unbuffered trace print remains"
        );
        assert!(SRC.contains("if self.trace.len() >= TRACE_CHUNK {"));
        assert!(SRC.contains("impl Drop for TracedDispatch {"));
    }

    /// DAY58 (I5f): the direct writer prints day 48's line byte for byte, edge values and a
    /// randomized set of keys, sizes, slots, hits and victims.
    #[test]
    fn the_direct_trace_line_is_day_48s_format_byte_for_byte() {
        let day48 = |k: (u16, u8, u16), b: usize, s: usize, h: bool, v: Option<(u16, u8, u16)>| {
            format!(
                "[expert-host-slru] key={}:{}:{} bytes={} slot={} hit={} victim={}\n",
                k.0,
                k.1,
                k.2,
                b,
                s,
                h,
                v.map(|v| format!("{}:{}:{}", v.0, v.1, v.2))
                    .unwrap_or_else(|| "-".into())
            )
        };
        let mut cases = vec![
            ((0, 0, 0), 0, 0, false, None),
            (
                (u16::MAX, u8::MAX, u16::MAX),
                usize::MAX,
                usize::MAX,
                true,
                Some((u16::MAX, u8::MAX, u16::MAX)),
            ),
            ((9, 10, 99), 100, 1000, true, Some((0, 0, 0))),
        ];
        let mut x: u64 = 0x9e37_79b9_7f4a_7c15;
        for _ in 0..20_000 {
            let mut next = || {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                x
            };
            let key = (next() as u16, next() as u8, next() as u16);
            let victim = (next() % 3 != 0).then(|| (next() as u16, next() as u8, next() as u16));
            cases.push((
                key,
                (next() % 2_000_000) as usize,
                (next() % 40_000) as usize,
                next() % 2 == 0,
                victim,
            ));
        }
        let mut out = String::new();
        for (k, b, s, h, v) in cases {
            out.clear();
            super::push_trace_line(&mut out, k, b, s, h, v);
            assert_eq!(out, day48(k, b, s, h, v));
        }
    }
}

#[cfg(test)]
mod day58_memo {
    //! DAY58 (I8f): the dense memo decides exactly as the map it replaced, step by step.
    use crate::moe_cache::{BlockId, ValidatedMemo};
    use std::collections::HashMap;

    #[test]
    fn the_dense_memo_matches_a_map_oracle() {
        let (mut memo, mut oracle) = (ValidatedMemo::default(), HashMap::<BlockId, usize>::new());
        let mut x: u64 = 0x2545_f491_4f6c_dd1d;
        for step in 0..200_000 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let id = BlockId::new(
                (x % 48) as u16,
                ((x >> 8) % 3) as u8,
                ((x >> 16) % 256) as u16,
            );
            let bytes = [450_560usize, 557_056, 860_160, 0][((x >> 32) % 4) as usize];
            let held = memo.holds(id, bytes);
            assert_eq!(
                held,
                bytes != 0 && oracle.get(&id) == Some(&bytes),
                "step {step}"
            );
            // The caller's guard: a pair not held reaches the proxy; the proxy accepts nonzero sizes.
            if !held && bytes != 0 {
                memo.insert(id, bytes);
                oracle.insert(id, bytes);
            }
        }
    }
}

#[cfg(test)]
mod day49_record_pass {
    //! DAY49: the parallel record pass gives the serial pass's digests in expert order and names
    //! the same first mismatch, for any thread count.
    use super::*;

    #[test]
    fn parallel_digests_equal_the_serial_pass() {
        let slabs: Vec<Vec<u8>> = (0..37u32)
            .map(|e| {
                (0..(1000 + e as usize * 13))
                    .map(|i| (i as u32 * 7 + e) as u8)
                    .collect()
            })
            .collect();
        let mut copies = slabs.clone();
        let mask: Vec<bool> = (0..37).map(|e| e % 5 != 3).collect();
        let pairs = |copies: &Vec<Vec<u8>>| -> Vec<Option<(Vec<u8>, Vec<u8>)>> {
            slabs
                .iter()
                .zip(copies)
                .zip(&mask)
                .map(|((a, b), &keep)| keep.then(|| (a.clone(), b.clone())))
                .collect()
        };
        let owned = pairs(&copies);
        let view: Vec<Option<(&[u8], &[u8])>> = owned
            .iter()
            .map(|p| p.as_ref().map(|(a, b)| (a.as_slice(), b.as_slice())))
            .collect();
        let serial: Vec<Option<Digest>> =
            view.iter().map(|p| p.map(|(a, _)| checksum(a))).collect();
        for threads in [1, 2, 3, 8, 64] {
            assert_eq!(
                record_digests(&view, threads),
                Ok(serial.clone()),
                "threads={threads}"
            );
        }
        // Two planted mismatches: the earlier expert is named, whatever thread sees which.
        copies[29][5] ^= 1;
        copies[11][0] ^= 1;
        let owned = pairs(&copies);
        let view: Vec<Option<(&[u8], &[u8])>> = owned
            .iter()
            .map(|p| p.as_ref().map(|(a, b)| (a.as_slice(), b.as_slice())))
            .collect();
        for threads in [1, 2, 3, 8, 64] {
            assert_eq!(record_digests(&view, threads), Err(11), "threads={threads}");
        }
    }
}

#[cfg(test)]
mod day50_census {
    //! DAY50: a pending prefetch is consumed only after the compute stream waits on its copy,
    //! the door's prefetch takes its lease through the proxy, and only the door's installer
    //! turns the forward's prefetch condition on.
    const CACHE: &str = include_str!("../moe_cache.rs");
    const FORWARD: &str = include_str!("../hybrid_forward.rs");
    const SRC: &str = include_str!("native.rs");

    #[test]
    fn prefetch_goes_through_the_owner_and_consumption_waits() {
        let consume = CACHE
            .find("if let Some(pending) = self.pending.remove(&id) {\n            if let Err(err) = e.compute_wait(pending.ready.as_ref()) {")
            .expect("consumption waits first");
        let publish = CACHE[consume..]
            .find("self.publish(id, pending.slot);")
            .unwrap();
        let wait = CACHE[consume..].find("e.compute_wait(").unwrap();
        assert!(wait < publish, "published before the wait");
        let prefetch = &CACHE[CACHE.find("fn prefetch_banked(").unwrap()..];
        let prefetch = &prefetch[..prefetch.find("\n    fn ").unwrap_or(prefetch.len())];
        assert!(prefetch.contains("bank.host_resident(local)?"));
        assert!(prefetch.contains("bank.demand(local, bytes)"));
        assert!(prefetch.contains("stage_on_copy_stream(e, payload, &mut self.slots[slot])"));
        assert_eq!(FORWARD.matches("e.expert_bank_prefetch()").count(), 1);
        // Count in this file's code, not in these tests' own literals.
        let code = &SRC[..SRC.find("#[cfg(test)]").unwrap()];
        assert_eq!(
            code.matches("self.set_expert_bank_prefetch(true);").count(),
            1
        );
    }
}

#[cfg(test)]
mod day60_census {
    //! DAY60: the dispatch clock is log only. Every use of the cache's clock field is a presence
    //! check, a counter update through `as_mut`, the line through `as_ref`, or the timing helper;
    //! no decision reads a counter, and the unclocked paths are the programs as they were.
    const CACHE: &str = include_str!("../moe_cache.rs");

    #[test]
    fn the_dispatch_clock_only_times_and_counts() {
        let code = &CACHE[..CACHE.find("#[cfg(test)]").unwrap_or(CACHE.len())];
        let mut uses = std::collections::BTreeMap::new();
        for (i, _) in code.match_indices("self.dispatch_clock") {
            let rest = &code[i + "self.dispatch_clock".len()..];
            let tail: String = rest
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_' || *c == '.')
                .collect();
            *uses.entry(tail).or_insert(0usize) += 1;
        }
        let allowed = [".is_none", ".is_some", ".as_mut", ".as_ref", "_mark"];
        for (tail, n) in &uses {
            assert!(
                allowed.contains(&tail.as_str()),
                "a dispatch clock use outside the log-only forms: {tail} ({n})"
            );
        }
        // The two wrappers return the unclocked result unchanged.
        assert_eq!(
            code.matches("let result = self.dispatch_source_unclocked(id, source, e);")
                .count(),
            1
        );
        assert_eq!(
            code.matches("let result = self.prefetch_source_unclocked(id, source, keep, e);")
                .count(),
            1
        );
        assert!(code.matches("        result\n    }").count() >= 2);
    }
}

#[cfg(test)]
mod day61_profile {
    //! DAY61 section 1 (`research/spill-c-20260919/DAY61.md`): the door's host-hit lease on the
    //! CPU, profiled. `#[ignore]`: a log-only instrument that decides nothing. The owner stack is
    //! built as the installer builds it, from synthetic bytes in a temporary file.
    use super::*;
    use std::collections::BTreeSet;
    use std::io::Write as _;

    const LAYERS: u16 = 40;
    const EXPERTS: u16 = 256;
    const LEN: u64 = 64;
    const CYCLES: usize = 200_000;
    const REPEATS: usize = 5;

    struct View;
    impl HostExpsView for View {
        fn is_uniform_layout(&self) -> bool {
            true
        }
        fn n_expert(&self) -> usize {
            usize::from(EXPERTS)
        }
        fn max_expert_bytes(&self) -> u64 {
            LEN
        }
        fn expert_layout(&self, e: usize) -> Result<ExpertMetadata> {
            Ok(ExpertMetadata {
                offset: e as u64 * LEN,
                len: LEN,
                qtype: 12,
                row_bytes: 16,
            })
        }
    }
    struct HeapBuf(Vec<u8>);
    impl HostBuffer for HeapBuf {
        fn as_slice(&self) -> &[u8] {
            &self.0
        }
        fn as_mut_slice(&mut self) -> &mut [u8] {
            &mut self.0
        }
    }
    /// A heap stand-in for the pinned pool, so a lease lends through the pooled-buffer branch as
    /// on the card.
    struct HeapSource;
    impl HostBufferSource for HeapSource {
        fn take(&mut self, len: usize) -> Option<Box<dyn HostBuffer>> {
            Some(Box::new(HeapBuf(vec![0; len])))
        }
    }
    struct Stack {
        traced: TracedDispatch,
        entries: Vec<(BankId, Option<CatalogRecord>)>,
        ids: BTreeMap<ExpertDispatchId, BankId>,
        request: BudgetRequest,
        capacity: TierBudget,
        _metadata: ChargedLease,
    }
    fn artifact(tag: &str) -> (std::path::PathBuf, Vec<u8>) {
        let len = usize::from(LAYERS) * 3 * usize::from(EXPERTS) * LEN as usize;
        let bytes: Vec<u8> = (0..len).map(|i| (i * 31 % 251) as u8).collect();
        let path = std::env::temp_dir().join(format!("c61-{tag}-{}.bin", std::process::id()));
        File::create(&path).unwrap().write_all(&bytes).unwrap();
        (path, bytes)
    }
    fn stack(path: &std::path::Path, bytes: &[u8], stage_clock: bool) -> Stack {
        let artifact = [7u8; 32];
        let mut reader = FileReader {
            file: Arc::new(File::open(path).unwrap()),
            ranges: BTreeMap::new(),
            reads: Rc::new(Cell::new(0)),
            pread_ns: None,
        };
        let mut entries = Vec::new();
        let mut ids = BTreeMap::new();
        let tensor_bytes = u64::from(EXPERTS) * LEN;
        for layer in 0..LAYERS {
            for (proj, (name, projection)) in [
                ("gate", Projection::Gate),
                ("up", Projection::Up),
                ("down", Projection::Down),
            ]
            .into_iter()
            .enumerate()
            {
                let start = (usize::from(layer) * 3 + proj) as u64 * tensor_bytes;
                let tensor = TensorId {
                    version: WIRE_VERSION,
                    artifact,
                    name: format!("blk.{layer}.ffn_{name}_exps.weight"),
                };
                reader.ranges.insert(tensor.clone(), (start, tensor_bytes));
                let mut sources = BTreeMap::new();
                for e in 0..u32::from(EXPERTS) {
                    let at = (start + u64::from(e) * LEN) as usize;
                    sources.insert(
                        e,
                        ExpertSource {
                            tensor: tensor.clone(),
                            split: false,
                            scales: vec![],
                            checksums: vec![checksum(&bytes[at..at + LEN as usize])],
                        },
                    );
                }
                let active = vec![true; usize::from(EXPERTS)];
                let mapping = host_exps_catalog(
                    &View,
                    tensor,
                    u32::from(layer),
                    projection,
                    &active,
                    &sources,
                )
                .unwrap();
                for id in mapping.ids {
                    let record = mapping.catalog.record(&id).unwrap().clone();
                    ids.insert(dispatch_id(&id.record).unwrap(), id.clone());
                    entries.push((id, Some(record)));
                }
            }
        }
        let slots = entries.len();
        let open = crate::moe_cache::BANKED_INFLIGHT + 1;
        let planned = slots as u64 * LEN;
        let mut capacity = TierBudget::zero(1);
        capacity.pageable = planned + slots as u64 * 4096 + 512 * 1024 * 1024;
        capacity.staging = LEN * open as u64;
        capacity.inflight = open as u64;
        let budget: SharedBudget = Rc::new(RefCell::new(
            Governor::new(capacity.clone(), TierBudget::zero(1), 1, 0, Arc::new(|| 0)).unwrap(),
        ));
        let bank: BankService<ExpertDomain, Heat, FileReader> = BankService::new(
            Catalog::new(LayoutClass::PerRecord, entries.clone()).unwrap(),
            budget.clone(),
            Heat,
            reader,
            CoalescingPolicy {
                granularity: 1,
                slot_bytes: LEN,
            },
            BankLimits {
                cache_bytes: planned,
                batch_bytes: LEN,
                items: 1,
                tickets: open,
            },
        )
        .unwrap();
        let bank = if stage_clock {
            bank.with_stage_clock()
        } else {
            bank
        };
        let bank = bank.with_host_buffers(Box::new(HeapSource)).unwrap();
        let mut request = BudgetRequest {
            bytes: TierBudget::zero(1),
            priority: Priority::Demand,
            deadline: Deadline(u64::MAX),
            tenant: artifact,
        };
        request.bytes.pageable = bank.slru_metadata_bytes(slots).unwrap();
        let metadata = budget.borrow_mut().reserve(&request).unwrap();
        let bank = bank
            .with_slru(SlruPolicy::new(&[(LEN, slots)]).unwrap(), &metadata)
            .unwrap();
        request.bytes = TierBudget::zero(1);
        let dispatch = SlruExpertDispatch::new(
            bank,
            ids.clone(),
            request.clone(),
            Epochs {
                state: 0,
                src_gen: 0,
                dst_gen: 0,
            },
        )
        .unwrap();
        Stack {
            traced: TracedDispatch {
                inner: dispatch,
                ids: ids.clone(),
                occupants: BTreeMap::new(),
                clock: None,
                fill: None,
                trace: String::with_capacity(TRACE_CHUNK + 256),
            },
            entries,
            ids,
            request,
            capacity,
            _metadata: metadata,
        }
    }
    /// One fixed seeded routing: per layer 8 distinct experts, each expert's three projections.
    fn order() -> Vec<ExpertDispatchId> {
        let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut out = Vec::with_capacity(CYCLES + 1024);
        while out.len() < CYCLES {
            for layer in 0..LAYERS {
                let mut chosen: Vec<u16> = Vec::with_capacity(8);
                while chosen.len() < 8 {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    let e = (s % u64::from(EXPERTS)) as u16;
                    if !chosen.contains(&e) {
                        chosen.push(e);
                    }
                }
                for e in chosen {
                    for proj in 0..3u8 {
                        out.push((layer, proj, e));
                    }
                }
            }
        }
        out.truncate(CYCLES);
        out
    }
    fn ns(d: std::time::Duration) -> u64 {
        u64::try_from(d.as_nanos()).unwrap_or(u64::MAX)
    }
    /// The median repeat (by its total) of `REPEATS` runs of `run`, each returning its per-part ns.
    fn median_repeat<const N: usize>(mut run: impl FnMut() -> [u64; N]) -> [u64; N] {
        let mut repeats: Vec<[u64; N]> = (0..REPEATS).map(|_| run()).collect();
        repeats.sort_by_key(|r| r[N - 1]);
        repeats[REPEATS / 2]
    }
    fn per_cycle(total: u64) -> f64 {
        total as f64 / CYCLES as f64
    }

    /// I11 change 1 on the door-shaped catalog (30720 records): the allowance `Catalog::new`
    /// memoizes is the per-ticket formula it replaced.
    #[test]
    fn the_door_catalog_allowance_is_the_formula() {
        let (path, bytes) = artifact("allowance");
        let s = stack(&path, &bytes, false);
        let catalog = Catalog::new(LayoutClass::PerRecord, s.entries.clone()).unwrap();
        for (id, record) in &s.entries {
            let layout = &record.as_ref().unwrap().layout;
            let formula =
                (id.encode().unwrap().len() + layout.encode().unwrap().len() + 1024) as u64;
            assert_eq!(catalog.metadata_allowance(id).unwrap(), formula);
            // DAY64 (I14 change 1): the hashed index finds every record of the door-shaped catalog.
            assert_eq!(catalog.record(id).unwrap().layout, *layout);
        }
        drop(s);
        std::fs::remove_file(path).ok();
    }

    /// I11 changes 4 and 5: over the fill (every record a host miss) and 20000 routed host-hit
    /// cycles, every demand's trace line names the slot a fresh SLRU lookup finds after the
    /// demand, with the hit flag the pre-demand lookup gave; the demanded record is the one
    /// leased.
    #[test]
    fn the_trace_slot_is_the_fresh_slot() {
        let (path, bytes) = artifact("slot");
        let mut s = stack(&path, &bytes, false);
        let ids = s.ids.clone();
        let check = |traced: &mut TracedDispatch, local: ExpertDispatchId, hit: bool| {
            let demand = traced.demand(local, LEN as usize).unwrap();
            assert_eq!(dispatch_id(&demand.lease.id().record).unwrap(), local);
            let id = &ids[&local];
            let fresh = traced
                .inner
                .bank()
                .slru_policy()
                .unwrap()
                .resident(id)
                .unwrap();
            let line = traced.trace.lines().last().unwrap().to_owned();
            assert!(
                line.contains(&format!(" slot={fresh} hit={hit} ")),
                "{line} (fresh slot {fresh})"
            );
            traced.trace.clear();
            traced.finish(demand).unwrap();
        };
        for local in ids.keys() {
            check(&mut s.traced, *local, false);
        }
        for &local in order().iter().take(20_000) {
            check(&mut s.traced, local, true);
        }
        drop(s);
        std::fs::remove_file(path).ok();
    }

    #[test]
    #[ignore = "DAY61 profile: log only, run by hand under the CPU cap"]
    fn host_hit_lease_profile() {
        let (path, bytes) = artifact("profile");
        let seq = order();
        let open = crate::moe_cache::BANKED_INFLIGHT + 1;
        println!(
            "DAY61 PROFILE records={} cycles={CYCLES} repeats={REPEATS} record_bytes={LEN}",
            usize::from(LAYERS) * 3 * usize::from(EXPERTS)
        );

        // P1 and P2: the door's prefetch sequence through the owner proxy.
        let s = stack(&path, &bytes, false);
        let ids = s.ids.clone();
        let mut owner = ExpertBankOwner::register(Box::new(s.traced), open).unwrap();
        let proxy = owner.proxy();
        for local in ids.keys() {
            let token = proxy.demand(*local, LEN as usize).unwrap();
            proxy.finish(&token).unwrap();
        }
        assert!(ids.keys().all(|l| proxy.host_resident(*l).unwrap()));
        let p1 = median_repeat(|| {
            let mut part = [0u64; 5];
            let started = Instant::now();
            for &local in &seq {
                let a = Instant::now();
                assert!(proxy.host_resident(local).unwrap());
                let b = Instant::now();
                let token = proxy.demand(local, LEN as usize).unwrap();
                let c = Instant::now();
                std::hint::black_box(proxy.with_bytes(&token, |b| b[0]).unwrap());
                let d = Instant::now();
                proxy.finish(&token).unwrap();
                let f = Instant::now();
                part[0] += ns(b - a);
                part[1] += ns(c - b);
                part[2] += ns(d - c);
                part[3] += ns(f - d);
            }
            part[4] = ns(started.elapsed());
            part
        });
        println!(
            "DAY61 P1 proxy per_cycle_ns host_resident={:.1} demand={:.1} with_bytes={:.1} finish={:.1} cycle={:.1}",
            per_cycle(p1[0]),
            per_cycle(p1[1]),
            per_cycle(p1[2]),
            per_cycle(p1[3]),
            per_cycle(p1[4])
        );
        let p2 = median_repeat(|| {
            let started = Instant::now();
            for &local in &seq {
                assert!(proxy.host_resident(local).unwrap());
                let token = proxy.demand(local, LEN as usize).unwrap();
                std::hint::black_box(proxy.with_bytes(&token, |b| b[0]).unwrap());
                proxy.finish(&token).unwrap();
            }
            [ns(started.elapsed())]
        });
        println!(
            "DAY61 P2 proxy unbracketed per_cycle_ns cycle={:.1} brackets={:.1}",
            per_cycle(p2[0]),
            per_cycle(p1[4]) - per_cycle(p2[0])
        );
        owner.close().unwrap();
        drop(owner);

        // P3: the bank alone, called directly, with its stage clock.
        let mut s = stack(&path, &bytes, true);
        for local in ids.keys() {
            let demand = s.traced.inner.demand(*local, LEN as usize).unwrap();
            s.traced.inner.finish(demand).unwrap();
        }
        let before = *s.traced.inner.bank().stage_times().unwrap();
        let p3 = median_repeat(|| {
            let mut part = [0u64; 3];
            let started = Instant::now();
            for &local in &seq {
                let a = Instant::now();
                let demand = s.traced.inner.demand(local, LEN as usize).unwrap();
                let b = Instant::now();
                s.traced.inner.finish(demand).unwrap();
                let c = Instant::now();
                part[0] += ns(b - a);
                part[1] += ns(c - b);
            }
            part[2] = ns(started.elapsed());
            part
        });
        let after = *s.traced.inner.bank().stage_times().unwrap();
        let cycles = (CYCLES * REPEATS) as f64;
        let d = |a: u64, b: u64| (a - b) as f64 / cycles;
        println!(
            "DAY61 P3 bank per_cycle_ns demand={:.1} finish={:.1} cycle={:.1} | stage clock per cycle (all repeats): stages={:.2} stage={:.1} alloc={:.1} publish={:.1} retire={:.1} collect={:.1}",
            per_cycle(p3[0]),
            per_cycle(p3[1]),
            per_cycle(p3[2]),
            d(after.stages, before.stages),
            d(after.stage_ns, before.stage_ns),
            d(after.alloc_ns, before.alloc_ns),
            d(after.publish_ns, before.publish_ns),
            d(after.retire_ns, before.retire_ns),
            d(after.collect_ns, before.collect_ns)
        );
        // DAY63 section 1: the split inside stage, publish and the retire side.
        println!(
            "DAY63 P3 split per cycle (all repeats): stage_lookup={:.1} stage_cache={:.1} stage_charge={:.1} publish_output={:.1} publish_policy={:.1} host_use={:.1} retire_only={:.1} ack={:.1} ack_release={:.1}",
            d(after.stage_lookup_ns, before.stage_lookup_ns),
            d(after.stage_cache_ns, before.stage_cache_ns),
            d(after.stage_charge_ns, before.stage_charge_ns),
            d(after.publish_output_ns, before.publish_output_ns),
            d(after.publish_policy_ns, before.publish_policy_ns),
            d(after.host_use_ns, before.host_use_ns),
            d(after.retire_only_ns, before.retire_only_ns),
            d(after.ack_ns, before.ack_ns),
            d(after.ack_release_ns, before.ack_release_ns)
        );
        // DAY63 P7: one clone of the leased record's `BankLease` and its drop, inside the same routed cycle.
        let p7 = median_repeat(|| {
            let mut part = [0u64; 2];
            let started = Instant::now();
            for &local in &seq {
                let demand = s.traced.inner.demand(local, LEN as usize).unwrap();
                let a = Instant::now();
                std::hint::black_box(demand.lease.clone());
                part[0] += ns(a.elapsed());
                s.traced.inner.finish(demand).unwrap();
            }
            part[1] = ns(started.elapsed());
            part
        });
        println!(
            "DAY63 P7 lease clone and drop per cycle ns={:.1} (cycle {:.1})",
            per_cycle(p7[0]),
            per_cycle(p7[1])
        );

        // P4: the parts inside `stage` its clock does not split, each alone over the same ids.
        let catalog = Catalog::new(LayoutClass::PerRecord, s.entries.clone()).unwrap();
        let bank_ids: Vec<BankId> = seq.iter().map(|l| s.ids[l].clone()).collect();
        let time = |f: &mut dyn FnMut(usize)| {
            median_repeat(|| {
                let started = Instant::now();
                for i in 0..CYCLES {
                    f(i);
                }
                [ns(started.elapsed())]
            })[0]
        };
        let encode = time(&mut |i| {
            let id = &bank_ids[i];
            let n = id.encode().unwrap().len()
                + catalog.record(id).unwrap().layout.encode().unwrap().len();
            std::hint::black_box(n);
        });
        let clone = time(&mut |i| {
            std::hint::black_box(vec![s.ids[&seq[i]].clone()]);
        });
        let map = time(&mut |i| {
            std::hint::black_box(s.ids.get(&seq[i]));
        });
        let lookup = time(&mut |i| {
            std::hint::black_box(catalog.record(&bank_ids[i]).unwrap());
        });
        let set = time(&mut |i| {
            let unique: BTreeSet<BankId> = std::iter::once(bank_ids[i].clone()).collect();
            std::hint::black_box(unique);
        });
        let request = time(&mut |_| {
            let r = s.request.clone();
            r.validate().unwrap();
            std::hint::black_box(r);
        });
        let mut governor = Governor::new(
            s.capacity.clone(),
            TierBudget::zero(1),
            1,
            0,
            Arc::new(|| 0),
        )
        .unwrap();
        let mut queue = s.request.clone();
        queue.bytes.pageable = 2048;
        queue.bytes.inflight = 1;
        let reserve = time(&mut |_| {
            let lease = governor.reserve(&queue).unwrap();
            governor.release(&lease).unwrap();
        });
        println!(
            "DAY61 P4 per_op_ns encode_two={:.1} bankid_clone_vec={:.1} ids_map_get={:.1} catalog_record={:.1} btreeset_one={:.1} request_clone_validate={:.1} governor_reserve_release={:.1}",
            per_cycle(encode),
            per_cycle(clone),
            per_cycle(map),
            per_cycle(lookup),
            per_cycle(set),
            per_cycle(request),
            per_cycle(reserve)
        );
        // P4b (added after the first run, DAY61 section 1a): the SLRU's lookup, the std hash of
        // one BankId alone, and the owner stack's host_resident without the proxy.
        let slru = time(&mut |i| {
            let policy = s.traced.inner.bank().slru_policy().unwrap();
            std::hint::black_box(policy.resident(&bank_ids[i]));
        });
        let hash = time(&mut |i| {
            use std::hash::{BuildHasher as _, RandomState};
            thread_local!(static STATE: RandomState = RandomState::new());
            std::hint::black_box(STATE.with(|st| st.hash_one(&bank_ids[i])));
        });
        let traced_resident = time(&mut |i| {
            std::hint::black_box(s.traced.host_resident(seq[i]).unwrap());
        });
        println!(
            "DAY61 P4b per_op_ns slru_resident={:.1} siphash_bankid={:.1} traced_host_resident={:.1}",
            per_cycle(slru),
            per_cycle(hash),
            per_cycle(traced_resident)
        );
        drop(s);
        std::fs::remove_file(path).ok();
    }
}

#[cfg(test)]
mod day61_census {
    //! DAY61 (I12): the cache retires finished leases where a lease is taken. `admit_banked`
    //! calls `retire_banked` once, after its GPU-hit and prefetched-consumption arms and before
    //! its demand; the prefetch retires before its bound check and its demand; nothing else in
    //! the cache demands a lease.
    const CACHE: &str = include_str!("../moe_cache.rs");

    fn body(name: &str) -> &'static str {
        let code = &CACHE[..CACHE.find("#[cfg(test)]").unwrap_or(CACHE.len())];
        let start = code.find(&format!("fn {name}(")).unwrap();
        let rest = &code[start..];
        &rest[..rest[1..].find("\n    fn ").map_or(rest.len(), |i| i + 1)]
    }

    #[test]
    fn leases_retire_where_a_lease_is_taken() {
        let code = &CACHE[..CACHE.find("#[cfg(test)]").unwrap_or(CACHE.len())];
        assert_eq!(code.matches("bank.demand(local, bytes)").count(), 2);
        let admit = body("admit_banked");
        assert_eq!(admit.matches("self.retire_banked(&bank)?;").count(), 1);
        let retire = admit.find("self.retire_banked(&bank)?;").unwrap();
        let hit = admit
            .find("if let Some(slot) = self.table.get(&id)")
            .unwrap();
        let consume = admit
            .find("if let Some(pending) = self.pending.remove(&id)")
            .unwrap();
        let demand = admit.find("bank.demand(local, bytes)").unwrap();
        assert!(hit < retire && consume < retire && retire < demand);
        let prefetch = body("prefetch_banked");
        let retire = prefetch.find("self.retire_banked(&bank)?;").unwrap();
        let bound = prefetch.find(">= BANKED_INFLIGHT").unwrap();
        let demand = prefetch.find("bank.demand(local, bytes)").unwrap();
        assert!(retire < bound && bound < demand);
    }
}
