//! Gate-only fault arms over the frozen tier contracts (lane D, day 11). Every arm drives the
//! active roundtrip's own `demote`/`restore` calls and injects exactly one fault at a
//! documented contract call on the first full-history K plane; the contract's own answers
//! are the evidence (`fault_contract`). No second cancellation, integrity or admission
//! mechanism lives here, and no token executes on an incomplete cache: a layer whose K plane
//! did not come back whole never re-enters the cache. Pooled allocator only.
use super::active::{self, EPOCHS, HostPlane, Transfers, request};
use super::fault_contract::{Arm, Check, Verdict, checks_tsv, verdict};
use super::tier_transfer::CudaTransfers;
use memra_engine::Engine;
use memra_kv::{Cache, KvLayer, KvPlane};
use memra_tier::{bank::SharedBudget, contracts::*, tier::governor::Governor};
use std::{cell::RefCell, fs, path::Path, rc::Rc, sync::Arc};

/// The active roundtrip's governor envelope, repeated here so the arms admit under the same
/// ceilings; `host-budget-short` lowers the pinned dimension below the demoted bytes.
const DEVICE_CAPACITY: u64 = 2 << 30;
const PINNED_CAPACITY: u64 = 2 << 30;
/// A second tenant of the same governor (the gate's own requests use the zero tenant).
const COMPETITOR: Digest = [1; 32];

/// Evidence gathered while the arm runs; the caller adds `restored-identical` from its own
/// capture and then `finish` renders the verdict and the receipt.
pub struct Pending {
    arm: Arm,
    checks: Vec<Check>,
    observations: Vec<(String, String)>,
    faulted: Option<(usize, Role, usize)>,
}
impl Pending {
    pub fn check(
        &mut self,
        name: &'static str,
        expected: impl Into<String>,
        observed: impl Into<String>,
    ) {
        self.checks.push(Check::new(name, expected, observed));
    }
    fn observe(&mut self, key: &str, value: impl std::fmt::Display) {
        self.observations.push((key.to_owned(), value.to_string()));
    }
}

/// What the faulted K plane became after the demote phase.
enum Slot {
    /// `cancel-demote`: the source plane came back through `take_plane` and sits in the layer.
    Resident,
    Demoted(Box<HostPlane>),
    /// `missing-host`: the untaken D2H ticket was retired and acknowledged; only its id remains.
    Removed(TransferTicket),
}

fn hex(digest: &Digest) -> String {
    digest.iter().map(|b| format!("{b:02x}")).collect()
}
fn resident_layers(cache: &Cache) -> usize {
    cache
        .kv
        .iter()
        .flatten()
        .filter(|kv| kv.len == cache.pos && kv.k.len() > 1)
        .count()
}
fn zero_budget(transfers: &Transfers) -> String {
    (transfers.borrow().used() == TierBudget::zero(1)).to_string()
}

/// Runs the arm against the suspended-and-restored program of `active::roundtrip`, faulting the
/// first K plane. Returns the recorded checks; the cache is whole again only for arms whose
/// `Arm::restores_cache` is true.
pub fn run(
    e: &Engine,
    cache: &mut Cache,
    program: ProgramIdentity,
    arm: Arm,
) -> super::Result<Pending> {
    let mut p = Pending {
        arm,
        checks: vec![],
        observations: vec![],
        faulted: None,
    };
    let whole_state = active::whole_state_bytes(cache)?;
    let mut capacity = TierBudget::zero(1);
    capacity.device[0] = DEVICE_CAPACITY;
    capacity.pinned = if arm == Arm::HostBudgetShort {
        (whole_state as u64)
            .checked_sub(1)
            .ok_or("host-budget-short needs a nonempty state")?
    } else {
        PINNED_CAPACITY
    };
    capacity.inflight = 256;
    p.observe("whole_state_pinned_bytes", whole_state);
    p.observe("governor_pinned_capacity_bytes", capacity.pinned);
    p.observe("governor_device_capacity_bytes", capacity.device[0]);
    let governor: SharedBudget = Rc::new(RefCell::new(Governor::new(
        capacity,
        TierBudget::zero(1),
        256,
        0,
        Arc::new(|| 0),
    )?));
    let transfers: Transfers = Rc::new(RefCell::new(CudaTransfers::new(
        e.stream(),
        governor.clone(),
    )?));
    e.stream().synchronize()?;
    let resident_before = resident_layers(cache);
    // Whole-state admission is the roundtrip's first contract call: the governor's own
    // `reserve` for every byte the demotion would pin, before any layer is taken.
    let admission = active::admit_whole_state(&governor, whole_state);
    if arm == Arm::HostBudgetShort {
        p.check(
            "whole-state-admission",
            "Err(Capacity)",
            format!("{admission:?}"),
        );
        p.check(
            "layers-resident",
            resident_before.to_string(),
            resident_layers(cache).to_string(),
        );
        p.check(
            "pinned-charged",
            "0",
            transfers.borrow().used().pinned.to_string(),
        );
        p.check("budget-zero", "true", zero_budget(&transfers));
        return Ok(p);
    }
    admission?;

    // Demote: the first K plane takes the arm's fault; every other plane goes through `demote`.
    let mut suspended: Vec<(usize, KvLayer, Slot, HostPlane)> = vec![];
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
        let mut k_slot = None;
        let mut v_plane = None;
        for (role, row, buffer) in [
            (Role::Key, layer.k_tok_bytes, &mut layer.k),
            (Role::Value, layer.v_tok_bytes, &mut layer.v),
        ] {
            let valid = cache.pos.checked_mul(row).ok_or("active extent overflow")?;
            let raw = e.dtoh_u8_view(&buffer.slice(..valid))?;
            let b = active::bundle(&program, i as u32, role, row, cache.pos, &raw)?;
            // Placeholder belongs to a detached KvLayer only; no reader can see it.
            let backing = std::mem::replace(buffer, e.alloc_u8(1)?.into());
            if role == Role::Value {
                v_plane = Some(active::demote(e, &transfers, backing, b)?);
            } else if p.faulted.is_some() {
                k_slot = Some(Slot::Demoted(Box::new(active::demote(
                    e, &transfers, backing, b,
                )?)));
            } else {
                p.faulted = Some((i, role, valid));
                k_slot = Some(match arm {
                    Arm::CancelDemote => {
                        *buffer = cancel_demote(&transfers, backing, b, &mut p)?;
                        Slot::Resident
                    }
                    Arm::MissingHost => {
                        Slot::Removed(demote_untaken_then_remove(&transfers, backing, b, &mut p)?)
                    }
                    Arm::CorruptHost => {
                        let mut plane = active::demote(e, &transfers, backing, b)?;
                        corrupt(e, &transfers, &mut plane, &mut p)?;
                        Slot::Demoted(Box::new(plane))
                    }
                    Arm::CancelRestore | Arm::DeviceShort | Arm::RequireResident => {
                        Slot::Demoted(Box::new(active::demote(e, &transfers, backing, b)?))
                    }
                    Arm::HostBudgetShort => return Err("host-budget-short refused above".into()),
                });
            }
        }
        suspended.push((
            i,
            layer,
            k_slot.ok_or("missing K")?,
            v_plane.ok_or("missing V")?,
        ));
    }
    if suspended.is_empty() {
        return Err("REFUSED: active gate requires nonempty native K/V state".into());
    }
    let faulted_layer = p.faulted.map(|(i, _, _)| i);
    e.stream().synchronize()?;
    p.observe("suspended_layers", suspended.len());
    p.observe(
        "pinned_charged_after_demote_bytes",
        transfers.borrow().used().pinned,
    );
    p.observe(
        "device_charged_after_demote_bytes",
        transfers.borrow().used().device[0],
    );
    if arm == Arm::RequireResident {
        // The engine's only continuation gate, asked while every full-history plane is demoted.
        // Recorded, not acted on: no token executes here.
        let gate = cache
            .ensure_usable("kv-tier-gate continuation")
            .map_err(|error| error.to_string());
        p.observe("continuation_gate_on_suspended_cache", format!("{gate:?}"));
    }
    if arm == Arm::DeviceShort {
        let Some((_, _, Slot::Demoted(plane), _)) = suspended.first() else {
            return Err("device-short expects a demoted K plane".into());
        };
        device_short(&governor, &transfers, plane, &mut p)?;
    }

    // Restore: the faulted K plane answers per arm; every other plane goes through `restore`.
    for (i, mut layer, k, v) in suspended {
        let faulted = faulted_layer == Some(i);
        let mut whole = true;
        match k {
            Slot::Resident => {}
            Slot::Demoted(plane) if faulted && arm == Arm::CancelRestore => {
                cancel_restore(e, &transfers, *plane, &mut p)?;
                whole = false;
            }
            Slot::Demoted(plane) if faulted && arm == Arm::CorruptHost => {
                let registry = transfers.borrow().device_registry_len();
                let restored = active::restore(e, &transfers, *plane);
                let observed = match &restored {
                    Ok(_) => "Ok(plane)".to_owned(),
                    Err(error) => error.to_string(),
                };
                drop(restored);
                p.check("restore-integrity", "Corrupt", observed);
                p.check(
                    "device-registry-unchanged",
                    registry.to_string(),
                    transfers.borrow().device_registry_len().to_string(),
                );
                whole = false;
            }
            Slot::Demoted(plane) => layer.k = active::restore(e, &transfers, *plane)?,
            Slot::Removed(ticket) => {
                let retake = transfers.borrow_mut().take_destination(&ticket, 0, EPOCHS);
                p.check(
                    "retake-removed-copy",
                    "Err(UnknownTicket)",
                    format!("{retake:?}"),
                );
                whole = false;
            }
        }
        layer.v = active::restore(e, &transfers, v)?;
        if whole {
            cache.kv[i] = Some(layer);
        } else {
            // Fail closed: an incomplete layer never re-enters the cache, so no token can
            // address it; its restored V plane and the placeholder are released here.
            drop(layer);
        }
    }
    e.stream().synchronize()?;
    p.check("budget-zero", "true", zero_budget(&transfers));
    p.observe(
        "device_registry_after_drain",
        transfers.borrow().device_registry_len(),
    );
    Ok(p)
}

/// D2H submitted and observed complete, then `TransferEngine::cancel` before `take_destination`
/// (the contract's cancellation seam: publication revocation). The source plane was never
/// modified and returns through `take_plane`; the unpublished host copy is released by `retire`.
fn cancel_demote(
    transfers: &Transfers,
    backing: KvPlane,
    b: StateBundle,
    p: &mut Pending,
) -> super::Result<KvPlane> {
    let capacity = backing.len();
    let bytes = b.layout.storage_bytes()? as usize;
    let mut t = transfers.borrow_mut();
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
    // Documented point: producer completion observed, destination not yet taken.
    p.check(
        "cancel",
        "Ok(PublicationRevoked)",
        format!("{:?}", t.cancel(&ticket)),
    );
    p.check(
        "take-after-cancel",
        "Err(Cancelled)",
        format!("{:?}", t.take_destination(&ticket, 0, EPOCHS)),
    );
    p.check("retire", "Ok(())", format!("{:?}", t.retire(&ticket, None)));
    p.check(
        "acknowledge",
        "Ok(())",
        format!("{:?}", t.acknowledge(&ticket)),
    );
    t.release_producer(producer)?;
    // The first plane is the only one in flight here, so a released copy reads as zero.
    p.check("pinned-after-cancel", "0", t.used().pinned.to_string());
    let plane = t.take_plane(&keep)?;
    p.check(
        "source-returned",
        format!("len={capacity} vmm=false"),
        format!("len={} vmm={}", plane.len(), plane.is_vmm()),
    );
    Ok(plane)
}

/// D2H completed but never taken: the engine's entry is the copy's only owner (a tier that has
/// not published its host copy yet). Retiring and acknowledging that untaken ticket removes the
/// copy; the restore's `take_destination` then meets the contract's `UnknownTicket`.
fn demote_untaken_then_remove(
    transfers: &Transfers,
    backing: KvPlane,
    b: StateBundle,
    p: &mut Pending,
) -> super::Result<TransferTicket> {
    let bytes = b.layout.storage_bytes()? as usize;
    let mut t = transfers.borrow_mut();
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
    let completion = t.poll(&ticket)?;
    let segment = &completion.items[0].segments[0];
    p.check(
        "completion-checksum",
        format!("producer_done=true checksum={}", hex(&b.checksums[0])),
        format!(
            "producer_done={} checksum={}",
            segment.producer_done,
            segment.checksum.as_ref().map_or("none".to_owned(), hex)
        ),
    );
    t.retire_source(&ticket)?;
    t.release_device_observed(&keep)?;
    t.release_producer(producer)?;
    let pinned_before = t.used().pinned;
    let retire = t.retire(&ticket, None);
    let acknowledge = t.acknowledge(&ticket);
    p.check(
        "remove",
        "retire=Ok(()) acknowledge=Ok(())",
        format!("retire={retire:?} acknowledge={acknowledge:?}"),
    );
    p.check(
        "pinned-released",
        bytes.to_string(),
        pinned_before.saturating_sub(t.used().pinned).to_string(),
    );
    Ok(ticket)
}

/// Flip one byte of the demoted copy. The contract refuses the write while the D2H ticket
/// still owns a twin of the copy (`Busy`); after the legal drain of that ticket (a taken
/// destination outlives acknowledgement) the gate's handle is the sole owner and the write
/// lands. `restore` then meets `StateBundle::verify` (`Corrupt`) before touching the device.
fn corrupt(
    e: &Engine,
    transfers: &Transfers,
    plane: &mut HostPlane,
    p: &mut Pending,
) -> super::Result<()> {
    let mut flipped = plane.host.bytes()?.to_vec();
    let index = flipped.len() / 2;
    flipped[index] ^= 0x01;
    p.observe("corrupt_byte_index", index);
    p.check(
        "write-under-live-ticket",
        "Err(Busy)",
        format!("{:?}", plane.host.write(&flipped)),
    );
    {
        let mut t = transfers.borrow_mut();
        let consumer = t.record_consumer(&plane.ticket)?;
        e.stream().synchronize()?;
        t.retire(&plane.ticket, Some(consumer))?;
        t.acknowledge(&plane.ticket)?;
    }
    p.check(
        "write-sole-owner",
        "Ok(())",
        format!("{:?}", plane.host.write(&flipped)),
    );
    Ok(())
}

/// A competing tenant reserves the governor's device dimension down to one byte less than the
/// restore needs; `alloc_device`, restore's first contract call, must answer `Capacity` and the
/// host copy must still verify. The competitor is released before the real restore.
fn device_short(
    governor: &SharedBudget,
    transfers: &Transfers,
    plane: &HostPlane,
    p: &mut Pending,
) -> super::Result<()> {
    let needed = plane.capacity as u64;
    let headroom = needed.checked_sub(1).ok_or("empty restore allocation")?;
    let mut competitor = request();
    competitor.tenant = COMPETITOR;
    competitor.bytes.device[0] = DEVICE_CAPACITY
        .checked_sub(headroom)
        .ok_or("restore allocation exceeds the device budget")?;
    let lease = governor.borrow_mut().reserve(&competitor)?;
    p.observe("competitor_device_bytes", competitor.bytes.device[0]);
    p.observe("device_headroom_bytes", headroom);
    p.observe("restore_device_bytes", needed);
    let registry = transfers.borrow().device_registry_len();
    let probe = transfers
        .borrow_mut()
        .alloc_device(plane.capacity, EPOCHS.dst_gen, request());
    let observed = match &probe {
        Ok(_) => "Ok(lease)".to_owned(),
        Err(error) => format!("Err({error:?})"),
    };
    if let Ok(admitted) = probe {
        transfers.borrow_mut().release_device(&admitted)?;
    }
    p.check("restore-admission", "Err(Capacity)", observed);
    p.check(
        "device-registry-unchanged",
        registry.to_string(),
        transfers.borrow().device_registry_len().to_string(),
    );
    p.check(
        "host-copy-intact",
        "Ok(())",
        format!("{:?}", plane.bundle.verify(&[plane.host.bytes()?.to_vec()])),
    );
    governor.borrow_mut().release(&lease)?;
    Ok(())
}

/// H2D submitted and observed complete, then `TransferEngine::cancel` before `ready_view`
/// publishes. Every publication path answers `Cancelled`; `retire` releases the H2D source
/// handle, which was the gate's only handle on the demoted copy. The D2H entry's twin cannot be
/// taken again (`AlreadyReleased`), so the copy is drained, never restored: no token executes.
fn cancel_restore(
    e: &Engine,
    transfers: &Transfers,
    plane: HostPlane,
    p: &mut Pending,
) -> super::Result<()> {
    let HostPlane {
        host,
        ticket: d2h,
        bundle,
        capacity,
        ..
    } = plane;
    bundle.verify(&[host.bytes()?.to_vec()])?;
    let bytes = host.valid_bytes();
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
    // Documented point: producer completion observed, before ready_view publishes.
    p.check(
        "cancel",
        "Ok(PublicationRevoked)",
        format!("{:?}", t.cancel(&ticket)),
    );
    p.check(
        "ready-view-after-cancel",
        "Err(Cancelled)",
        format!("{:?}", t.ready_view(&ticket, 0, EPOCHS).map(|_| ())),
    );
    p.check(
        "with-destination-after-cancel",
        "Err(Cancelled)",
        format!(
            "{:?}",
            t.with_destination(&ticket, 0, EPOCHS, |_, _| Ok(()))
        ),
    );
    p.check(
        "take-after-cancel",
        "Err(Cancelled)",
        format!("{:?}", t.take_destination(&ticket, 0, EPOCHS)),
    );
    let pinned_before = t.used().pinned;
    p.check("retire", "Ok(())", format!("{:?}", t.retire(&ticket, None)));
    p.check(
        "acknowledge",
        "Ok(())",
        format!("{:?}", t.acknowledge(&ticket)),
    );
    // The D2H entry still owns the physical copy and its charge: intact in accounting only.
    p.observe("pinned_before_h2d_retire_bytes", pinned_before);
    p.observe("pinned_after_h2d_retire_bytes", t.used().pinned);
    p.check(
        "retake-demoted-copy",
        "Err(AlreadyReleased)",
        format!("{:?}", t.take_destination(&d2h, 0, EPOCHS)),
    );
    let consumer = t.record_consumer(&d2h)?;
    e.stream().synchronize()?;
    t.retire(&d2h, Some(consumer))?;
    t.acknowledge(&d2h)?;
    p.observe("pinned_after_d2h_acknowledge_bytes", t.used().pinned);
    p.observe("h2d_source_bytes", bytes);
    // The unpublished destination never enters the cache; it is released here.
    drop(t.take_plane(&keep)?);
    Ok(())
}

/// Render the verdict and the receipt (`FAULT-ARM.txt`, `fault-checks.tsv`).
pub fn finish(p: Pending, out: &Path) -> super::Result<Verdict> {
    let v = verdict(p.arm, &p.checks);
    fs::write(out.join("fault-checks.tsv"), checks_tsv(&p.checks))?;
    let (layer, role, bytes) = p.faulted.map_or_else(
        || ("none".to_owned(), "none".to_owned(), "0".to_owned()),
        |(i, r, n)| (i.to_string(), format!("{r:?}"), n.to_string()),
    );
    let outcome = if v.pass {
        "PASS"
    } else if v.line.starts_with("REFUSED:") {
        "REFUSED"
    } else {
        "FAILED"
    };
    let mut text = format!(
        "arm={}\nverdict={outcome}\nline={}\ncontinues={}\nrestores_cache={}\nfaulted_layer={layer}\nfaulted_role={role}\nfaulted_valid_bytes={bytes}\nchecks={}\nchecks_failed={}\nchecks_missing={}\nrequired_checks={}\n",
        p.arm.name(),
        v.line,
        p.arm.continues(),
        p.arm.restores_cache(),
        p.checks.len(),
        v.failed.join(","),
        v.missing.join(","),
        p.arm.required_checks().join(","),
    );
    for (key, value) in &p.observations {
        text.push_str(&format!("observation.{key}={value}\n"));
    }
    if let Some(finding) = p.arm.refusal() {
        text.push_str(&format!("finding={finding}\n"));
    }
    fs::write(out.join("FAULT-ARM.txt"), text)?;
    Ok(v)
}
