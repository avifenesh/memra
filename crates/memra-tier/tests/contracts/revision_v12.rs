//! Additive CPU schedules. Hooks observe/control fixtures, never CUDA event truth.
use memra_tier::contracts::*;

/// Exercise the completion contract at each consumer's boundary. Mutations are
/// local snapshots: this does NOT prove a backend handles physical short I/O.
pub fn framed_logical_bytes(c: &Completion, expected: &[Vec<SegmentExpectation>], device: bool) {
    c.require(&c.ticket, expected, device).unwrap();
    for (i, item) in c.items.iter().enumerate() {
        for (j, segment) in item.segments.iter().enumerate() {
            assert!(segment.valid_bytes > 0);
            for io in [0, segment.valid_bytes, segment.valid_bytes + 8192, u64::MAX] {
                let mut framed = c.clone();
                framed.items[i].segments[j].io_bytes = io;
                framed.require(&c.ticket, expected, device).unwrap();
                framed.items[i].segments[j].valid_bytes -= 1;
                assert_eq!(
                    framed.require(&c.ticket, expected, device),
                    Err(Error::ShortIo {
                        expected: segment.valid_bytes,
                        actual: segment.valid_bytes - 1,
                    })
                );
            }
            let mut corrupt = c.clone();
            corrupt.items[i].segments[j].checksum.as_mut().unwrap()[0] ^= 1;
            assert_eq!(
                corrupt.require(&c.ticket, expected, device),
                Err(Error::Corrupt)
            );
            let mut missing = c.clone();
            missing.items[i].segments.remove(j);
            assert_eq!(
                missing.require(&c.ticket, expected, device),
                Err(Error::Incomplete)
            );
        }
    }
}

/// Snapshot comes from TransferEngine::poll, not a reconstructed aggregate.
pub fn transfer_completion_bytes<T: TransferEngine>(
    t: &mut T,
    ticket: &TransferTicket,
    expected: &[Vec<SegmentExpectation>],
    device: bool,
) {
    framed_logical_bytes(&t.poll(ticket).unwrap(), expected, device);
}
pub fn peer_completion_bytes<P: PeerBackend>(
    p: &mut P,
    ticket: &TransferTicket,
    expected: &[Vec<SegmentExpectation>],
) {
    framed_logical_bytes(&p.poll(ticket).unwrap(), expected, true);
}
/// Bank/row traits expose no poll: the hook must return their actual completion.
pub fn bank_completion_bytes<B: BankedResidency>(
    b: &B,
    ticket: &TransferTicket,
    expected: &[Vec<SegmentExpectation>],
    observe: impl FnOnce(&B, &TransferTicket) -> Completion,
) {
    framed_logical_bytes(&observe(b, ticket), expected, false);
}
pub fn row_completion_bytes<R: RowService>(
    r: &R,
    ticket: &TransferTicket,
    expected: &[Vec<SegmentExpectation>],
    observe: impl FnOnce(&R, &TransferTicket) -> Completion,
) {
    framed_logical_bytes(&observe(r, ticket), expected, false);
}
/// ObjectStore exposes logical chunk reads, not completion telemetry.
pub fn object_logical_bytes<S: ObjectStore>(
    s: &mut S,
    manifest: &ObjectManifest,
    request: &BudgetRequest,
    payloads: &[Vec<u8>],
) {
    assert_eq!(manifest.chunks.len(), payloads.len());
    let lease = s.lease(manifest, request).unwrap();
    for (i, (chunk, payload)) in manifest.chunks.iter().zip(payloads).enumerate() {
        let mut dst = vec![0xa5; chunk.storage_bytes as usize];
        let n = s.read(&lease, i as u32, &mut dst).unwrap();
        assert_eq!(n, chunk.valid_bytes);
        assert_eq!(&dst[..n as usize], payload);
    }
    s.release(&lease).unwrap();
}

/// No owner method can observe driver completion: the caller is responsible for
/// establishing producer/wait/last-use truth before creating these observations.
pub fn ready_owner(
    owner: &mut DeviceOwner,
    destination: &DeviceLease,
    foreign: &DeviceOwner,
    c: &Completion,
    expected: &[Vec<SegmentExpectation>],
) {
    let epochs = c.ticket.epochs;
    assert!(matches!(
        owner.ready_view(destination, c, expected, epochs),
        Err(Error::UnknownTicket)
    ));
    assert!(matches!(
        foreign.ready_view(destination, c, expected, epochs),
        Err(Error::WrongOwner)
    ));
    let stale = TransferTicket {
        epochs: Epochs {
            dst_gen: epochs.dst_gen + 1,
            ..epochs
        },
        ..c.ticket
    };
    assert_eq!(
        owner.bind_destination(stale, destination),
        Err(Error::StaleEpoch)
    );
    owner.bind_destination(c.ticket, destination).unwrap();
    for current in [
        Epochs {
            state: epochs.state + 1,
            ..epochs
        },
        Epochs {
            src_gen: epochs.src_gen + 1,
            ..epochs
        },
        Epochs {
            dst_gen: epochs.dst_gen + 1,
            ..epochs
        },
    ] {
        assert!(matches!(
            owner.ready_view(destination, c, expected, current),
            Err(Error::StaleEpoch)
        ));
    }
    for i in 0..c.items.len() {
        for j in 0..c.items[i].segments.len() {
            for field in 0..5 {
                let mut bad = c.clone();
                let s = &mut bad.items[i].segments[j];
                match field {
                    0 => s.consumer_fence.as_mut().unwrap().issuer = foreign.issuer(),
                    1 => s.consumer_fence.as_mut().unwrap().owner += 1,
                    2 => s.consumer_fence.as_mut().unwrap().generation += 1,
                    3 => s.consumer_fence = None,
                    _ => s.consumer_fenced = false,
                }
                assert!(
                    owner
                        .ready_view(destination, &bad, expected, epochs)
                        .is_err()
                );
            }
        }
    }
    let view = owner.ready_view(destination, c, expected, epochs).unwrap();
    assert_eq!(
        view.destination().allocation_id(),
        destination.allocation_id()
    );
    assert_eq!(view.ticket(), c.ticket);
    // Installed wait is NOT consumer completion: binding still prevents release.
    assert_eq!(owner.release(destination), Err(Error::Busy));
    // Fake caller has now explicitly driven all uses to retirement.
    owner.retire_binding(&c.ticket).unwrap();
    assert!(matches!(
        owner.ready_view(destination, c, expected, epochs),
        Err(Error::UnknownTicket)
    ));
}

#[derive(Clone, Copy, Debug)]
pub enum InstallCase {
    Valid,
    MissingSource,
    AmbiguousSource,
    WrongLayout,
    WrongGeneration,
    MissingObject,
    WrongManifest,
    MissingSpec,
    AmbiguousSpec,
    WrongLength,
}
/// A fresh source/pool per case prevents a previous success masking a refusal.
pub fn bank_source_install(mut install: impl FnMut(InstallCase) -> Result<()>) {
    use InstallCase::*;
    for (case, expected) in [
        (Valid, Ok(())),
        (MissingSource, Err(Error::NotFound)),
        (AmbiguousSource, Err(Error::Conflict)),
        (WrongLayout, Err(Error::InvalidLayout)),
        (WrongGeneration, Err(Error::InvalidLayout)),
        (MissingObject, Err(Error::NotFound)),
        (WrongManifest, Err(Error::InvalidLayout)),
        (MissingSpec, Err(Error::NotFound)),
        (AmbiguousSpec, Err(Error::Conflict)),
        (WrongLength, Err(Error::InvalidLayout)),
    ] {
        assert_eq!(install(case), expected, "{case:?}");
    }
}

/// Run unchanged against distinct materializer implementations/record programs.
/// Capture hooks compare complete bytes/order and observe retired refusal.
pub fn kv_materializer<M: KvMaterializer>(
    m: &mut M,
    bundle: &StateBundle,
    ready: &ReadyView<'_>,
    done: FenceId,
    mut capture: impl FnMut(&M, &M::Operands, bool),
) {
    let epochs = ready.ticket().epochs;
    for field in 0..10 {
        let mut wrong = bundle.program.clone();
        let fields = [
            &mut wrong.artifact,
            &mut wrong.serialized_plan,
            &mut wrong.numeric,
            &mut wrong.stream,
            &mut wrong.tokenizer,
            &mut wrong.template,
            &mut wrong.adapter,
            &mut wrong.modality,
            &mut wrong.position,
            &mut wrong.tenant_salt,
        ];
        fields.into_iter().nth(field).unwrap()[0] ^= 1;
        assert!(matches!(
            m.materialize(bundle, &wrong, ready, epochs),
            Err(Error::ProgramMismatch)
        ));
    }
    for stale in [
        Epochs {
            state: epochs.state + 1,
            ..epochs
        },
        Epochs {
            src_gen: epochs.src_gen + 1,
            ..epochs
        },
        Epochs {
            dst_gen: epochs.dst_gen + 1,
            ..epochs
        },
    ] {
        assert!(matches!(
            m.materialize(bundle, &bundle.program, ready, stale),
            Err(Error::StaleEpoch)
        ));
    }
    for i in 0..bundle.layout.segments.len() {
        let mut missing = bundle.clone();
        missing.layout.segments.remove(i);
        assert!(
            m.materialize(&missing, &missing.program, ready, epochs)
                .is_err()
        );
        let mut corrupt = bundle.clone();
        corrupt.checksums[i][0] ^= 1;
        assert!(
            m.materialize(&corrupt, &corrupt.program, ready, epochs)
                .is_err()
        );
    }
    let operands = m
        .materialize(bundle, &bundle.program, ready, epochs)
        .unwrap();
    capture(m, &operands, true);
    for wrong in [
        FenceId {
            issuer: done.issuer + 1,
            ..done
        },
        FenceId {
            owner: done.owner + 1,
            ..done
        },
        FenceId {
            generation: done.generation + 1,
            ..done
        },
    ] {
        assert_eq!(m.retire(&operands, wrong), Err(Error::WrongOwner));
        capture(m, &operands, true);
    }
    m.retire(&operands, done).unwrap();
    capture(m, &operands, false);
    assert_eq!(m.retire(&operands, done), Err(Error::AlreadyReleased));
}

/// TierStore does not expose I/O counters. It must preserve the entire bundle
/// (including distinct logical/storage lengths), not replace lengths with framing.
/// Transfer completion checks belong at its TransferEngine binding above.
pub fn tier_logical_bytes<T: TierStore>(
    store: &mut T,
    reservation: &TierReservation,
    expected: &StateBundle,
    epochs: Epochs,
) {
    let ready = store.ready(reservation, &expected.program, epochs).unwrap();
    assert_eq!(&ready.bundle, expected);
    assert!(
        expected
            .layout
            .segments
            .iter()
            .any(|s| s.storage_bytes != s.valid_bytes)
    );
}
