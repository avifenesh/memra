//! Qualification-only immutable-GGUF expert bank installer. No runtime env door,
//! artifact conversion, external source, or alternative numeric executor.
use crate::{
    Engine,
    hybrid::{Ffn, HybridModel, MoeWeights},
};
use memra_gguf::GgufFile;
use memra_tier::{bank::*, contracts::*, tier::governor::Governor};
use sha2::{Digest as _, Sha256};
use std::{
    cell::{Cell, RefCell},
    collections::BTreeMap,
    fs::File,
    os::unix::fs::FileExt,
    rc::Rc,
    sync::Arc,
};

const APPROVED_SHA: &str = "df27a780435b7b45c2597536112ea3cb091f8544c3d0c3318d9f4258b31f7adf";
struct FileReader {
    file: Arc<File>,
    ranges: BTreeMap<TensorId, (u64, u64)>,
    reads: Rc<Cell<u64>>,
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
        self.file
            .read_exact_at(dst, start.checked_add(offset).ok_or(Error::Overflow)?)?;
        self.reads.set(self.reads.get() + 1);
        Ok(())
    }
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
pub struct BankedExpertGate {
    owner: ExpertBankOwner,
    reads: Rc<Cell<u64>>,
    budget: SharedBudget,
    metadata: ChargedLease,
}
impl Drop for BankedExpertGate {
    fn drop(&mut self) {
        eprintln!(
            "[experts-via-tier] physical_reads={} owner_close={:?}",
            self.reads.get(),
            self.owner.close()
        );
        // A still-pinned owner refuses release; never force-credit unknown DMA.
        let _ = self.budget.borrow_mut().release(&self.metadata);
    }
}
impl Engine {
    /// Explicit default-OFF qualification door for the approved exact artifact.
    /// Call after loading and before the first forward; keeps the existing native
    /// SLRU slot addresses, expert kernels and routing unchanged. Resident slabs,
    /// scale-bearing formats and parallel/frozen paths are intentionally refused.
    pub fn install_expert_bank_gate(
        &self,
        model: &HybridModel,
        gguf: &GgufFile,
    ) -> std::result::Result<BankedExpertGate, Box<dyn std::error::Error>> {
        if gguf.n_shards() != 1 || !Engine::moe_cache_enabled() || !model.mtp_extra.is_empty() {
            return Err(
                "experts-via-tier requires one immutable GGUF, cache, and at most one MTP head"
                    .into(),
            );
        }
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
        let actual: String = artifact.iter().map(|b| format!("{b:02x}")).collect();
        if actual != APPROVED_SHA {
            return Err("experts-via-tier artifact SHA256 mismatch".into());
        }
        let reads = Rc::new(Cell::new(0));
        let mut reader = FileReader {
            file,
            ranges: BTreeMap::new(),
            reads: reads.clone(),
        };
        let mut entries = vec![];
        let mut ids = BTreeMap::new();
        let mut max_bytes = 0;
        let mut add = |layer: u16,
                       checkpoint_layer: usize,
                       moe: &MoeWeights|
         -> std::result::Result<(), Box<dyn std::error::Error>> {
            if moe.dev_exps.is_some()
                || moe.step_ep.is_some()
                || moe.step_tp.is_some()
                || moe.glm5_ep.is_some()
                || moe.glm5_tp_split.is_some()
            {
                return Err("experts-via-tier refuses resident or parallel expert bypasses; use the cache baseline".into());
            }
            for (proj, projection, name, host) in [
                (0, Projection::Gate, "gate", &moe.gate_exps),
                (1, Projection::Up, "up", &moe.up_exps),
                (2, Projection::Down, "down", &moe.down_exps),
            ] {
                if host.macros.is_some() || host.fp8_blk.is_some() {
                    return Err("experts-via-tier payload-only gate refuses scale planes".into());
                }
                let name = format!("blk.{checkpoint_layer}.ffn_{name}_exps.weight");
                let info = gguf.find(&name).ok_or("missing immutable expert tensor")?;
                let raw = gguf.tensor_data(info);
                let (start, _) = gguf.tensor_file_range(info);
                let tensor = TensorId {
                    version: WIRE_VERSION,
                    artifact,
                    name,
                };
                let active = moe
                    .active_experts
                    .clone()
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
                    sources.insert(
                        expert as u32,
                        ExpertSource {
                            tensor: source,
                            split: host.tiers.is_some(),
                            scales: vec![],
                            checksums: vec![checksum(bytes)],
                        },
                    );
                }
                let mapping = crate::banked_residency::map_host_exps(
                    host,
                    tensor,
                    u32::from(layer),
                    projection,
                    &active,
                    &sources,
                )?;
                max_bytes = max_bytes.max(mapping.max_expert_bytes);
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
            }
            Ok(())
        };
        for (layer, block) in model.layers.iter().enumerate() {
            if let Ffn::Moe(moe) = &block.ffn {
                add(u16::try_from(layer)?, layer, moe)?;
            }
        }
        if let Some(head) = &model.mtp
            && let Ffn::Moe(moe) = &head.ffn
        {
            add(u16::MAX, model.cfg.n_layer as usize, moe)?;
        }
        if ids.is_empty() || max_bytes == 0 || max_bytes > 16 * 1024 * 1024 {
            return Err("experts-via-tier empty or oversized bank".into());
        }
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
        let bank = BankService::new(
            Catalog::new(LayoutClass::PerRecord, entries)?,
            budget.clone(),
            Heat,
            reader,
            CoalescingPolicy {
                granularity: 1,
                slot_bytes: max_bytes,
            },
            BankLimits {
                cache_bytes: 256 * 1024 * 1024,
                batch_bytes: max_bytes,
                items: 1,
                tickets: 1,
            },
        )?;
        let slots = 16;
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
            ids,
            request,
            Epochs {
                state: 0,
                src_gen: 0,
                dst_gen: 0,
            },
        )?;
        let owner = ExpertBankOwner::register(Box::new(dispatch), 1)?;
        self.with_moe_cache(max_bytes as usize, |cache, _| {
            cache.install_banked(owner.proxy())
        })?;
        eprintln!(
            "[experts-via-tier] installed artifact_sha256={actual} host_slots={slots} max_expert_bytes={max_bytes}"
        );
        Ok(BankedExpertGate {
            owner,
            reads,
            budget,
            metadata,
        })
    }
}
