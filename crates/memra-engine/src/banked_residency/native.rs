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
    sync::Arc,
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
}
impl BankedExpertGate<'_> {
    /// `--expert-bank-stages`: print the door's cumulative stage line for `phase` (DAY40).
    /// A no-op without the flag.
    pub fn print_stage_line(&self, phase: &str) {
        if !self.stage_clock {
            return;
        }
        match self.engine.expert_bank_stage_line() {
            Ok(Some(line)) => eprintln!("[experts-via-tier] stages phase={phase} {line}"),
            Ok(None) => eprintln!("[experts-via-tier] stages phase={phase} absent"),
            Err(err) => eprintln!("[experts-via-tier] stages phase={phase} refused: {err}"),
        }
    }
}
impl Drop for BankedExpertGate<'_> {
    fn drop(&mut self) {
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
        // A still-pinned owner refuses release; never force-credit unknown DMA.
        let _ = self.budget.borrow_mut().release(&self.metadata);
    }
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
        let slots = host_bank_budget(host_bytes, max_bytes)?;
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
        let mut capacity = TierBudget::zero(1);
        capacity.pageable = 512 * 1024 * 1024;
        capacity.staging = max_bytes;
        capacity.inflight = 1;
        let budget: SharedBudget = Rc::new(RefCell::new(Governor::new(
            capacity,
            TierBudget::zero(1),
            1,
            0,
            Arc::new(|| 0),
        )?));
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
                cache_bytes: host_bytes,
                batch_bytes: max_bytes,
                items: 1,
                tickets: 1,
            },
        )?;
        let bank = if stage_clock {
            bank.with_stage_clock()
        } else {
            bank
        };
        let mut request = BudgetRequest {
            bytes: TierBudget::zero(1),
            priority: Priority::Demand,
            deadline: Deadline(u64::MAX),
            tenant: artifact,
        };
        request.bytes.pageable = bank.slru_metadata_bytes(slots)?;
        let metadata = budget.borrow_mut().reserve(&request)?;
        let bank = bank.with_slru(SlruPolicy::new(&[(max_bytes, slots)])?, &metadata)?;
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
            }),
            1,
        )?;
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
        Some(format!(
            "| owner host_hits={} host_misses={} inner_demand_ns={} trace_ns={} reads={} pread_ns={} | bank {}",
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
