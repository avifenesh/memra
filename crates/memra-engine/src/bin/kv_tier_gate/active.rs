//! Gate-only native active K/V handoff. No scheduler/default/serving registration.
//! Recurrent state and counters stay resident; every full-history K/V allocation is
//! removed from the cache before restore. No token can execute during that interval.
use super::tier_transfer::{CudaPinnedLease, CudaTransfers};
use cudarc::driver::CudaSlice;
use memra_engine::Engine;
use memra_kv::{Cache, KvLayer};
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

fn demote(
    e: &Engine,
    t: &Transfers,
    backing: CudaSlice<u8>,
    b: StateBundle,
) -> super::Result<HostPlane> {
    let capacity = backing.len();
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
    let (source_owners_before_release, source_owners_after_release) =
        t.release_device_observed(&keep)?;
    t.release_producer(producer)?;
    e.stream().synchronize()?;
    Ok(HostPlane {
        host,
        ticket,
        bundle: b,
        capacity,
        source_owners_before_release,
        source_owners_after_release,
    })
}
fn restore(e: &Engine, transfers: &Transfers, plane: HostPlane) -> super::Result<CudaSlice<u8>> {
    let HostPlane {
        host,
        ticket: d2h,
        bundle,
        capacity,
        ..
    } = plane;
    if checksum(host.bytes()?) != bundle.checksums[0] {
        return Err("host residency checksum mismatch".into());
    }
    let bytes = host.valid_bytes();
    let (ticket, keep) = {
        let mut t = transfers.borrow_mut();
        let device = t.alloc_device(capacity, EPOCHS.dst_gen, request())?;
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
    Ok(t.take_device(&keep)?) // No D2D; original native operand type/accounting returns to Cache.
}

/// Returns only after all cache slots have been restored, or aborts the gate.
/// This gate is the exclusive scheduler: it cannot decode with a suspended cache.
pub fn roundtrip(
    e: &Engine,
    cache: &mut Cache,
    program: ProgramIdentity,
    out: &Path,
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
    let before = e.ctx().mem_get_info()?.0;
    let pool_before = e.pool_reserved_used();
    let mut suspended: Vec<(usize, KvLayer, HostPlane, HostPlane)> = vec![];
    let mut allocated = 0usize;
    let mut logical = 0usize;
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
            // Placeholder belongs to a detached KvLayer only; no reader can see it.
            let backing = std::mem::replace(buffer, e.alloc_u8(1)?);
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
    let diagnosis = if source_owners_after_release != 0 {
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
    let used = transfers.borrow().used();
    fs::write(out.join("active-bundles.tsv"), manifest)?;
    fs::write(
        out.join("active-reclaim.txt"),
        format!(
            "committed={}\nlayers={}\nsource_allocation_bytes={allocated}\nlogical_d2h_bytes={logical}\ndevice_charged_after_demote={}\npinned_charged_after_demote={}\nfree_before_bytes={before}\nfree_after_demote_bytes={after}\npool_trim_released_bytes={trimmed}\n",
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
    for (i, mut layer, k, v) in suspended {
        layer.k = restore(e, &transfers, k)?;
        layer.v = restore(e, &transfers, v)?;
        reload_count += 2;
        cache.kv[i] = Some(layer);
    }
    if transfers.borrow().used() != TierBudget::zero(1) {
        return Err("active transfer budget not fully retired".into());
    }
    e.stream().synchronize()?;
    e.pool_trim_to_zero();
    let restored = e.ctx().mem_get_info()?.0;
    let reclaimed = after > before && restored < after;
    let mut metrics = fs::OpenOptions::new()
        .append(true)
        .open(out.join("active-reclaim.txt"))?;
    writeln!(
        metrics,
        "demote_count={demote_count}\nreload_count={reload_count}\nfree_after_restore_bytes={restored}\nreclaimed_bytes={}\nreacquired_bytes={}\nreclaim_observed={reclaimed}",
        after as i128 - before as i128,
        after as i128 - restored as i128
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
