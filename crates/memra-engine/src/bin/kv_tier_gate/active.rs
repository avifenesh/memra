//! Gate-only native active K/V handoff. No scheduler/default/serving registration.
//! Recurrent state and counters stay resident; every full-history K/V allocation is
//! removed from the cache before restore. No token can execute during that interval.
use super::tier_transfer::{CudaPinnedLease, CudaTransfers};
use memra_engine::Engine;
use memra_kv::{Cache, KvLayer, KvPlane};
use memra_tier::{bank::SharedBudget, contracts::*, tier::governor::Governor};
use std::{cell::RefCell, fs, io::Write, path::Path, rc::Rc, sync::Arc};

type Transfers = Rc<RefCell<CudaTransfers>>;
const EPOCHS: Epochs = Epochs {
    state: 1,
    src_gen: 1,
    dst_gen: 1,
};
fn request() -> BudgetRequest {
    BudgetRequest {
        bytes: TierBudget::zero(1),
        priority: Priority::Demand,
        deadline: Deadline(u64::MAX),
        tenant: [0; 32],
    }
}

/// Operands remain backend-owned until their real consumer event retires. This
/// descriptor cannot expose a pointer; `with_destination` revalidates publication.
struct NativeOperands {
    ticket: TransferTicket,
    allocation: u64,
}
struct NativeMaterializer {
    transfers: Transfers,
    expected: StateBundle,
    ticket: TransferTicket,
    allocation: u64,
    live: bool,
}
impl KvMaterializer for NativeMaterializer {
    type Operands = NativeOperands;
    fn materialize(
        &mut self,
        b: &StateBundle,
        p: &ProgramIdentity,
        ready: &ReadyView<'_>,
        current: Epochs,
    ) -> Result<NativeOperands> {
        current.require(EPOCHS)?;
        b.validate()?;
        if p != &self.expected.program || &b.program != p {
            return Err(Error::ProgramMismatch);
        }
        if b != &self.expected {
            return Err(Error::InvalidLayout);
        }
        if ready.ticket() != self.ticket || ready.destination().allocation_id() != self.allocation {
            return Err(Error::ForeignLease);
        }
        ready.ticket().epochs.require(current)?;
        if ready.destination().device() != b.id.owner {
            return Err(Error::WrongOwner);
        }
        if ready.destination().bytes() < b.layout.storage_bytes()? {
            return Err(Error::Capacity);
        }
        if self.live {
            return Err(Error::Busy);
        }
        self.live = true;
        Ok(NativeOperands {
            ticket: self.ticket,
            allocation: self.allocation,
        })
    }
    fn retire(&mut self, op: &NativeOperands, consumer_done: FenceId) -> Result<()> {
        if !self.live || op.ticket != self.ticket || op.allocation != self.allocation {
            return Err(Error::ForeignLease);
        }
        // Backend verifies the exact retained CUDA event, not just fence integers.
        self.transfers
            .borrow_mut()
            .retire(&op.ticket, Some(consumer_done))?;
        self.live = false;
        Ok(())
    }
}

struct HostPlane {
    host: CudaPinnedLease,
    ticket: TransferTicket,
    bundle: StateBundle,
    capacity: usize,
    source_owners_before_release: usize,
    source_owners_after_release: usize,
    vmm: Option<KvPlane>,
    released_bytes: usize,
    granularity: usize,
}
fn bundle(
    program: &ProgramIdentity,
    group: u32,
    role: Role,
    row: usize,
    tokens: usize,
    bytes: &[u8],
) -> Result<StateBundle> {
    let encoding = match role {
        Role::Key => b"q8_0".as_slice(),
        Role::Value => b"q5_1".as_slice(),
        _ => return Err(Error::Unsupported),
    };
    if row.checked_mul(tokens) != Some(bytes.len()) || tokens == 0 {
        return Err(Error::InvalidLayout);
    }
    let b = StateBundle {
        version: 1,
        id: KvBlockId {
            version: 1,
            namespace: program.namespace()?,
            parent: [0; 32],
            tokens: program.position,
            start: 0,
            end: tokens as u64,
            group,
            owner: 0,
            epoch: EPOCHS.state,
        },
        program: program.clone(),
        kind: StateKind::Active,
        committed_high_water: tokens as u64,
        layout: RecordLayout {
            version: 1,
            segments: vec![ByteSegment {
                version: 1,
                group,
                page: 0,
                owner: 0,
                role,
                tensor: None,
                offset: 0,
                valid_bytes: bytes.len() as u64,
                storage_bytes: bytes.len() as u64,
                alignment: 1,
                encoding: EncodingId {
                    version: 1,
                    program: digest("native-kv-encoding", encoding),
                    row_bytes: row as u64,
                },
            }],
            requirements: vec![GroupRequirement {
                version: 1,
                group,
                owner: 0,
                role,
                page_count: 1,
                pages: PageRequirement::AllPages,
            }],
        },
        owner_aliases: vec![],
        checksums: vec![checksum(bytes)],
    };
    b.validate()?;
    Ok(b)
}

fn demote(e: &Engine, t: &Transfers, backing: KvPlane, b: StateBundle) -> super::Result<HostPlane> {
    let capacity = backing.len();
    let is_vmm = backing.is_vmm();
    let bytes = b.layout.storage_bytes()? as usize;
    let mut t = t.borrow_mut();
    let device = t.register_device(backing, EPOCHS.src_gen, request())?;
    let keep = t.retain_device(&device)?;
    let host = t.alloc_host(bytes, request())?;
    let producer = t.record_producer(EPOCHS.src_gen)?;
    let ticket = t
        .d2h(CopyOp {
            host,
            device,
            bytes: bytes as u64,
            epochs: EPOCHS,
            producer_fence: Some(producer),
        })
        .map_err(|rejected| rejected.error)?;
    t.synchronize(&ticket)?;
    let Destination::Host(host) = t.take_destination(&ticket, 0, EPOCHS)? else {
        return Err("D2H returned non-host destination".into());
    };
    if checksum(host.bytes()?) != b.checksums[0] {
        return Err("native D2H checksum mismatch".into());
    }
    t.retire_source(&ticket)?;
    let (
        source_owners_before_release,
        source_owners_after_release,
        vmm,
        released_bytes,
        granularity,
    ) = if is_vmm {
        let mut plane = t.take_plane(&keep)?;
        let granularity = plane.granularity().ok_or("missing VMM granularity")?;
        let released = plane.demote_prefix(bytes)?;
        (1, 1, Some(plane), released, granularity)
    } else {
        let (before, after) = t.release_device_observed(&keep)?;
        (before, after, None, 0, 0)
    };
    t.release_producer(producer)?;
    e.stream().synchronize()?;
    Ok(HostPlane {
        host,
        ticket,
        bundle: b,
        capacity,
        source_owners_before_release,
        source_owners_after_release,
        vmm,
        released_bytes,
        granularity,
    })
}
fn restore(e: &Engine, transfers: &Transfers, plane: HostPlane) -> super::Result<KvPlane> {
    let HostPlane {
        host,
        ticket: d2h,
        bundle,
        capacity,
        vmm,
        released_bytes,
        ..
    } = plane;
    if checksum(host.bytes()?) != bundle.checksums[0] {
        return Err("host residency checksum mismatch".into());
    }
    let bytes = host.valid_bytes();
    let (ticket, keep) = {
        let mut t = transfers.borrow_mut();
        let device = if let Some(mut plane) = vmm {
            if plane.remap_prefix()? != released_bytes {
                return Err("VMM remap byte mismatch".into());
            }
            t.register_device(plane, EPOCHS.dst_gen, request())?
        } else {
            t.alloc_device(capacity, EPOCHS.dst_gen, request())?
        };
        let keep = t.retain_device(&device)?;
        let ticket = t
            .h2d(CopyOp {
                host,
                device,
                bytes,
                epochs: EPOCHS,
                producer_fence: None,
            })
            .map_err(|rejected| rejected.error)?;
        t.synchronize(&ticket)?;
        (ticket, keep)
    };
    let mut materializer = NativeMaterializer {
        transfers: transfers.clone(),
        expected: bundle.clone(),
        ticket,
        allocation: keep.allocation_id(),
        live: false,
    };
    let op = {
        let mut t = transfers.borrow_mut();
        let ready = t.ready_view(&ticket, 0, EPOCHS)?;
        materializer.materialize(&bundle, &bundle.program, &ready, EPOCHS)?
    };
    let consumer = {
        let mut t = transfers.borrow_mut();
        t.with_destination(&ticket, 0, EPOCHS, |buffer, stream| {
            let restored = stream
                .clone_dtoh(&buffer.slice(..bytes as usize))
                .map_err(|_| Error::Quarantined)?;
            if checksum(&restored) != bundle.checksums[0] {
                return Err(Error::Corrupt);
            }
            Ok(())
        })?;
        let consumer = t.record_consumer(&ticket)?;
        e.stream().synchronize()?;
        t.retire_source(&ticket)?; // Releases host only after observed H2D completion.
        consumer
    };
    materializer.retire(&op, consumer)?;
    let mut t = transfers.borrow_mut();
    t.acknowledge(&ticket)?;
    // The D2H destination host no longer lives after the H2D source retirement.
    let consumer = t.record_consumer(&d2h)?;
    e.stream().synchronize()?;
    t.retire(&d2h, Some(consumer))?;
    t.acknowledge(&d2h)?;
    Ok(t.take_plane(&keep)?) // No D2D; original native operand type/accounting returns to Cache.
}

/// Returns only after all cache slots have been restored, or aborts the gate.
/// This gate is the exclusive scheduler: it cannot decode with a suspended cache.
pub fn roundtrip(
    e: &Engine,
    cache: &mut Cache,
    program: ProgramIdentity,
    out: &Path,
    diagnostic: bool,
) -> super::Result<bool> {
    let mut capacity = TierBudget::zero(1);
    capacity.device[0] = 2 << 30;
    capacity.pinned = 2 << 30;
    capacity.inflight = 256;
    let governor: SharedBudget = Rc::new(RefCell::new(Governor::new(
        capacity,
        TierBudget::zero(1),
        256,
        0,
        Arc::new(|| 0),
    )?));
    let transfers = Rc::new(RefCell::new(CudaTransfers::new(e.stream(), governor)?));
    e.stream().synchronize()?;
    e.pool_trim_to_zero(); // Equal trim before/after isolates newly freed source allocations.
    let free_before_spare_reserve = e.ctx().mem_get_info()?.0;
    let spare = if diagnostic {
        let planes = cache.kv.iter().flatten().flat_map(|l| [&l.k, &l.v]);
        let bytes: usize = planes.clone().map(|p| p.physical_bytes()).sum();
        let granularity = planes
            .filter_map(|p| p.granularity())
            .next()
            .ok_or("REFUSED: diagnostic needs VMM planes")?;
        Some(SpareVa::reserve(bytes, granularity)?)
    } else {
        None
    };
    let before = e.ctx().mem_get_info()?.0;
    let pool_before = e.pool_reserved_used();
    let mut suspended: Vec<(usize, KvLayer, HostPlane, HostPlane)> = vec![];
    let mut allocated = 0usize;
    let mut logical = 0usize;
    let mut physical = 0usize;
    let mut vmm_manifest =
        String::from("layer\trole\tvirtual_address\tvalid_bytes\tphysical_bytes\tgranularity\n");
    let mut manifest = String::from("layer\trole\tvalid_bytes\tbundle_sha256\n");
    for i in 0..cache.kv.len() {
        if cache.kv[i].as_ref().is_none_or(|kv| kv.len == 0) {
            continue;
        }
        let mut layer = cache.kv[i].take().ok_or("missing KV layer")?;
        let geometry =
            memra_kv::tiered::materializer::NativeGeometry::from_layer(&layer, i as u32, 0)?;
        if geometry.tokens != cache.pos as u64 || layer.base_d.is_some() {
            return Err("unsupported active geometry".into());
        }
        let mut planes = vec![];
        for (role, row, buffer) in [
            (Role::Key, layer.k_tok_bytes, &mut layer.k),
            (Role::Value, layer.v_tok_bytes, &mut layer.v),
        ] {
            let valid = cache.pos.checked_mul(row).ok_or("active extent overflow")?;
            let raw = e.dtoh_u8_view(&buffer.slice(..valid))?;
            let b = bundle(&program, i as u32, role, row, cache.pos, &raw)?;
            manifest.push_str(&format!(
                "{i}\t{role:?}\t{valid}\t{}\n",
                super::hash(&b.encode()?)
            ));
            allocated = allocated
                .checked_add(buffer.len())
                .ok_or("allocation sum overflow")?;
            logical = logical.checked_add(valid).ok_or("logical sum overflow")?;
            physical += buffer.physical_bytes();
            if let Some(address) = buffer.virtual_address() {
                vmm_manifest.push_str(&format!(
                    "{i}\t{role:?}\t{address}\t{valid}\t{}\t{}\n",
                    buffer.physical_bytes(),
                    buffer.granularity().unwrap()
                ));
            }
            // Placeholder belongs to a detached KvLayer only; no reader can see it.
            let backing = std::mem::replace(buffer, e.alloc_u8(1)?.into());
            planes.push(demote(e, &transfers, backing, b)?);
        }
        let v = planes.pop().ok_or("missing V")?;
        let k = planes.pop().ok_or("missing K")?;
        suspended.push((i, layer, k, v));
    }
    if suspended.is_empty() {
        return Err("REFUSED: active gate requires nonempty native K/V state".into());
    }
    e.stream().synchronize()?;
    let after_before_trim = e.ctx().mem_get_info()?.0;
    let pool_demoted = e.pool_reserved_used();
    let trimmed = checked_trim(e)?;
    let after = e.ctx().mem_get_info()?.0;
    let pool_trimmed = e.pool_reserved_used();
    let source_owners_before_release: usize = suspended
        .iter()
        .map(|(_, _, k, v)| k.source_owners_before_release + v.source_owners_before_release)
        .sum();
    let source_owners_after_release: usize = suspended
        .iter()
        .map(|(_, _, k, v)| k.source_owners_after_release + v.source_owners_after_release)
        .sum();
    let source_drops = suspended
        .iter()
        .flat_map(|(_, _, k, v)| [k, v])
        .filter(|p| p.source_owners_before_release == 1 && p.source_owners_after_release == 0)
        .count();
    let vmm_released_bytes: usize = suspended
        .iter()
        .map(|(_, _, k, v)| k.released_bytes + v.released_bytes)
        .sum();
    let vmm_granularity = suspended
        .iter()
        .map(|(_, _, k, _)| k.granularity)
        .max()
        .unwrap_or(0);
    let diagnosis = if vmm_granularity != 0 {
        "RECLAIM-DIAG: VMM fixed-VA chunk release"
    } else if source_owners_after_release != 0 {
        "RECLAIM-DIAG: source still owned by transfer backing Rc"
    } else if after > after_before_trim {
        "RECLAIM-DIAG: async-pool retention"
    } else {
        "RECLAIM-DIAG: freed but not observable"
    };
    fs::write(
        out.join("reclaim-diagnosis.txt"),
        format!(
            "{diagnosis}\nsource_owners_before_release={source_owners_before_release}\nsource_owners_after_release={source_owners_after_release}\nsource_slice_drops={source_drops}\ndevice_registry_after_demote={}\nfree_before_trim_bytes={after_before_trim}\nfree_after_trim_bytes={after}\npool_before_reserved={}\npool_before_used={}\npool_demoted_reserved={}\npool_demoted_used={}\npool_trimmed_reserved={}\npool_trimmed_used={}\ntrim_api=cuDeviceGetDefaultMemPool+cuMemPoolTrimTo\ntrim_api_success=true\n",
            transfers.borrow().device_registry_len(),
            pool_before.0,
            pool_before.1,
            pool_demoted.0,
            pool_demoted.1,
            pool_trimmed.0,
            pool_trimmed.1
        ),
    )?;
    eprintln!("{diagnosis}");
    let mut residual_probe = None;
    if let Some(spare) = spare {
        let spare_bytes = spare.bytes;
        spare.release()?;
        let after_va_free = e.ctx().mem_get_info()?.0;
        // SAFETY: Engine's CUDA context is bound to this owner thread; wait for all its work.
        unsafe { cudarc::driver::sys::cuCtxSynchronize().result()? };
        let after_ctx_sync = e.ctx().mem_get_info()?.0;
        residual_probe = Some((after_va_free, after_ctx_sync));
        fs::write(
            out.join("residual-diagnostic.txt"),
            format!(
                "free_before_spare_reserve_bytes={free_before_spare_reserve}\nfree_after_spare_reserve_bytes={before}\nspare_reserved_unmapped_bytes={spare_bytes}\nfree_after_demote_bytes={after}\nfree_after_spare_va_free_bytes={after_va_free}\nfree_after_context_sync_bytes={after_ctx_sync}\nspare_va_free_success=true\ncontext_sync_success=true\n"
            ),
        )?;
    }

    let used = transfers.borrow().used();
    fs::write(out.join("active-bundles.tsv"), manifest)?;
    fs::write(out.join("vmm-planes.tsv"), vmm_manifest)?;
    fs::write(
        out.join("active-reclaim.txt"),
        format!(
            "committed={}\nlayers={}\nsource_allocation_bytes={allocated}\nsource_physical_bytes={physical}\nlogical_d2h_bytes={logical}\ndevice_charged_after_demote={}\npinned_charged_after_demote={}\nfree_before_bytes={before}\nfree_after_demote_bytes={after}\npool_trim_released_bytes={trimmed}\n",
            cache.pos,
            suspended.len(),
            used.device[0],
            used.pinned
        ),
    )?;
    if used.device[0] != 0 || used.pinned != logical as u64 {
        return Err("active demotion accounting mismatch; see active-reclaim.txt".into());
    }
    let demote_count = suspended.len() * 2;
    let mut reload_count = 0;
    // Counts planes whose fixed VA was actually OBSERVED (Some before and Some-equal after); a pooled
    // plane has no VA to compare, so it never counts and the receipt cannot claim the property.
    let mut fixed_va_observed = 0usize;
    for (i, mut layer, k, v) in suspended {
        let k_address = k.vmm.as_ref().and_then(KvPlane::virtual_address);
        let v_address = v.vmm.as_ref().and_then(KvPlane::virtual_address);
        layer.k = restore(e, &transfers, k)?;
        layer.v = restore(e, &transfers, v)?;
        if layer.k.virtual_address() != k_address || layer.v.virtual_address() != v_address {
            return Err("VMM fixed virtual address changed during restore".into());
        }
        for (before, after) in [
            (k_address, layer.k.virtual_address()),
            (v_address, layer.v.virtual_address()),
        ] {
            if before.is_some() && before == after {
                fixed_va_observed += 1;
            }
        }
        reload_count += 2;
        cache.kv[i] = Some(layer);
    }
    if transfers.borrow().used() != TierBudget::zero(1) {
        return Err("active transfer budget not fully retired".into());
    }
    e.stream().synchronize()?;
    e.pool_trim_to_zero();
    let restored = e.ctx().mem_get_info()?.0;
    let observation = super::reclaim_contract::observe(
        before,
        after,
        restored,
        vmm_released_bytes,
        vmm_granularity,
    );
    let residual_class = if vmm_granularity == 0 {
        "not-applicable-pooled"
    } else if observation.residual == 0 {
        "none"
    } else if let Some((va_free, ctx_sync)) = residual_probe {
        if va_free as i128 - after as i128 == observation.residual && ctx_sync == va_free {
            "spare-VA-release-sensitive-driver-accounting"
        } else if ctx_sync as i128 - va_free as i128 == observation.residual && va_free == after {
            "deferred-driver-release-completed-by-context-sync"
        } else {
            "unclassified"
        }
    } else {
        "unclassified"
    };
    // Pooled runs have no released-chunk quantity to bound against, so they can report an observed,
    // leak-free rise but can never publish the G1 label: `g1_reclaim_qualified=not-applicable-pooled`.
    let reclaim_observed = if vmm_granularity != 0 {
        observation.bounded_no_leak
    } else {
        after > before && restored == before
    };
    let reclaimed = vmm_granularity != 0 && reclaim_observed && residual_class != "unclassified";
    let g1_reclaim_qualified = if vmm_granularity == 0 {
        "not-applicable-pooled".to_string()
    } else {
        reclaimed.to_string()
    };
    let residual_bytes = if vmm_granularity == 0 {
        "not-applicable-pooled".to_string()
    } else {
        observation.residual.to_string()
    };
    if diagnostic {
        let mut probe = fs::OpenOptions::new()
            .append(true)
            .open(out.join("residual-diagnostic.txt"))?;
        writeln!(
            probe,
            "residual_bytes={residual_bytes}\nresidual_class={residual_class}\nfree_after_restore_bytes={restored}",
        )?;
    }
    let vmm_fixed_va_restored = if vmm_granularity == 0 {
        "not-applicable-pooled"
    } else if fixed_va_observed == reload_count {
        "true"
    } else {
        "false"
    };
    let mut metrics = fs::OpenOptions::new()
        .append(true)
        .open(out.join("active-reclaim.txt"))?;
    writeln!(
        metrics,
        "vmm_fixed_va_restored={vmm_fixed_va_restored}\nvmm_retained_edge_and_capacity_bytes={}\nvmm_granularity_bytes={vmm_granularity}\nvmm_released_chunk_bytes={vmm_released_bytes}\ndemote_count={demote_count}\nreload_count={reload_count}\nfree_after_restore_bytes={restored}\nreclaimed_bytes={}\nreacquired_bytes={}\nreclaim_observed={reclaim_observed}\nreclaim_exact_equal={}\nresidual_bytes={residual_bytes}\nresidual_class={residual_class}\ng1_reclaim_qualified={g1_reclaim_qualified}",
        if vmm_granularity == 0 {
            0
        } else {
            physical - vmm_released_bytes
        },
        after as i128 - before as i128,
        after as i128 - restored as i128,
        observation.exact,
    )?;
    metrics.sync_all()?;
    Ok(reclaimed)
}

/// Unlike the engine's best-effort trim, a failed diagnostic API is an error.
fn checked_trim(e: &Engine) -> super::Result<usize> {
    use cudarc::driver::{result::device, sys};
    e.ctx().bind_to_thread()?;
    let dev = device::get(e.ctx().ordinal() as i32)?;
    let before = e.pool_reserved_used().0;
    let mut pool = std::ptr::null_mut();
    // SAFETY: current context owns dev; pool is a live output pointer.
    unsafe { sys::cuDeviceGetDefaultMemPool(&mut pool, dev).result()? };
    // SAFETY: pool was successfully obtained above; trim only frees unused blocks.
    unsafe { sys::cuMemPoolTrimTo(pool, 0).result()? };
    Ok(before.saturating_sub(e.pool_reserved_used().0))
}

/// A diagnostic-only VA reservation: never mapped and never exposed to kernels.
struct SpareVa {
    address: Option<u64>,
    bytes: usize,
}
impl SpareVa {
    fn reserve(bytes: usize, granularity: usize) -> super::Result<Self> {
        let mut address = 0;
        // SAFETY: bytes is the sum of queried-granularity-aligned plane capacities; output is live.
        unsafe {
            cudarc::driver::sys::cuMemAddressReserve(&mut address, bytes, granularity, 0, 0)
                .result()?
        };
        Ok(Self {
            address: Some(address),
            bytes,
        })
    }
    fn release(mut self) -> super::Result<()> {
        // SAFETY: exclusively owned reservation was never mapped or handed to any consumer.
        unsafe {
            cudarc::driver::sys::cuMemAddressFree(self.address.unwrap(), self.bytes).result()?
        };
        self.address = None;
        Ok(())
    }
}
impl Drop for SpareVa {
    fn drop(&mut self) {
        if let Some(address) = self.address {
            // SAFETY: cleanup of this owner's still-unmapped, unexposed reservation.
            if let Err(e) =
                unsafe { cudarc::driver::sys::cuMemAddressFree(address, self.bytes).result() }
            {
                eprintln!("spare VA cleanup failed: {e}");
            }
        }
    }
}
