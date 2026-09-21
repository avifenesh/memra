//! Real CUDA conformance + byte roundtrip gate. Run ONLY through tier-battery.
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, sys};
use memra_engine::tier_transfer::{CudaPinnedLease, CudaTransfers, GraphPin};
use memra_tier::conformance as v1;
use memra_tier::{bank::SharedBudget, contracts::*, tier::governor::Governor};
use std::{
    cell::RefCell,
    rc::Rc,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

fn epochs() -> Epochs {
    Epochs {
        state: 1,
        src_gen: 7,
        dst_gen: 7,
    }
}
fn request() -> BudgetRequest {
    BudgetRequest {
        bytes: TierBudget::zero(1),
        priority: Priority::Demand,
        deadline: Deadline(u64::MAX),
        tenant: [1; 32],
    }
}
fn setup() -> (CudaTransfers, Arc<CudaStream>, SharedBudget) {
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    let cap = TierBudget {
        version: 1,
        device: vec![2 << 30],
        peer: vec![0],
        replicas: vec![0],
        pinned: 1 << 30,
        pageable: 1 << 30,
        staging: 1 << 30,
        loaders: 64,
        inflight: 64,
        nvme: 0,
    };
    let gov: SharedBudget = Rc::new(RefCell::new(
        Governor::new(cap, TierBudget::zero(1), 64, 0, Arc::new(|| 0)).unwrap(),
    ));
    (
        CudaTransfers::new(stream.clone(), gov.clone()).unwrap(),
        stream,
        gov,
    )
}
fn expected(bytes: &[u8]) -> Vec<Vec<SegmentExpectation>> {
    vec![vec![SegmentExpectation {
        valid_bytes: bytes.len() as u64,
        io_bytes: bytes.len() as u64,
        checksum: checksum(bytes),
    }]]
}
fn h2d(t: &mut CudaTransfers, bytes: &[u8]) -> (CopyOp<CudaPinnedLease>, DeviceLease) {
    let mut host = t.alloc_host(bytes.len(), request()).unwrap();
    host.write(bytes).unwrap();
    let device = t
        .alloc_device(bytes.len(), epochs().dst_gen, request())
        .unwrap();
    let keep = t.retain_device(&device).unwrap();
    (
        CopyOp {
            host,
            device,
            bytes: bytes.len() as u64,
            epochs: epochs(),
            producer_fence: None,
        },
        keep,
    )
}
/// Drain a cancelled H2D ticket. Day-11 rule 1: the cancelled restore holds its untouched
/// source for the caller, so retirement waits until every accepted item's source is recovered.
fn clean(t: &mut CudaTransfers, ticket: &TransferTicket, items: &[u32], device: &DeviceLease) {
    t.synchronize(ticket).unwrap();
    assert_eq!(t.retire(ticket, None), Err(Error::Busy));
    for item in items {
        drop(t.recover_source(ticket, *item).unwrap());
    }
    t.retire(ticket, None).unwrap();
    t.acknowledge(ticket).unwrap();
    t.release_device(device).unwrap();
}
fn conformance() {
    let (mut t, stream, gov) = setup();
    let bytes = [3, 20, 37, 0];
    let (op, keep) = h2d(&mut t, &bytes);
    let ticket = t.h2d(op).unwrap();
    v1::transfer_cancel(&mut t, ticket, |t, ticket| {
        t.synchronize(ticket).unwrap();
        // Day-11 rule 1: the frozen schedule is unchanged; its completion hook recovers the
        // source before the ticket may retire.
        assert_eq!(t.retire(ticket, None), Err(Error::Busy));
        let host = t.recover_source(ticket, 0).unwrap();
        assert_eq!(host.bytes().unwrap(), &bytes[..]);
        drop(host);
        t.retire(ticket, None).unwrap();
    });
    t.release_device(&keep).unwrap();
    println!("PASS v1 transfer_cancel native CUDA");

    let (op, keep) = h2d(&mut t, &bytes);
    let ticket = t.h2d(op).unwrap();
    v1::transfer_complete_cancel(&mut t, ticket, &expected(&bytes), |t, ticket| {
        t.synchronize(ticket).unwrap()
    });
    clean(&mut t, &ticket, &[0], &keep);
    println!("PASS v1.1 transfer_complete_cancel native CUDA");

    let (op, keep) = h2d(&mut t, &bytes);
    let ticket = t.h2d(op).unwrap();
    let mut graph = Some(t.pin_graph(&ticket).unwrap());
    v1::transfer_lifetime(
        &mut t,
        ticket,
        |t, ticket, step| match step {
            v1::LifetimeStep::Unknown => {
                t.quarantine_observation(ticket).unwrap();
                assert_eq!(t.retire_source(ticket), Err(Error::Quarantined));
            }
            v1::LifetimeStep::Producer => stream.synchronize().unwrap(),
            v1::LifetimeStep::Recover => t.synchronize(ticket).unwrap(),
            v1::LifetimeStep::Consumer => {
                assert_eq!(t.retire(ticket, None), Err(Error::Busy));
            }
            v1::LifetimeStep::Graph => {
                drop(graph.take());
                // Day-11 rule 1: idle and revoked, the source is still held for the caller.
                assert_eq!(t.retire(ticket, None), Err(Error::Busy));
                drop(t.recover_source(ticket, 0).unwrap());
                t.retire(ticket, None).unwrap();
            }
        },
        |t| t.used(),
    );
    t.release_device(&keep).unwrap();
    println!(
        "PASS v1.1 transfer_lifetime native events + injected observation loss + graph retention"
    );

    let host = t.alloc_host(4, request()).unwrap();
    let read = TransferOp::NvmeRead(ReadPlan {
        object: ObjectKey {
            version: 1,
            artifact: [0; 32],
            semantic_id: [0; 32],
            layout: [0; 32],
            generation: 7,
        },
        chunk: 0,
        destination: host,
        epochs: epochs(),
    });
    let rejected = v1::transfer_zero_accept(&mut t, vec![read], |ops| {
        assert!(matches!(&ops[0], TransferOp::NvmeRead(_)));
    });
    drop(rejected);
    println!("PASS v1.1 transfer_zero_accept Unsupported NVMe preserves owned input");
    let (a, keep_a) = h2d(&mut t, &bytes);
    let (b, keep_b) = h2d(&mut t, &bytes);
    let (mut c, keep_c) = h2d(&mut t, &bytes);
    c.bytes = 5; // preflight reject only this item, without submitting it
    let batch = t
        .submit_batch(vec![
            TransferOp::H2d(a),
            TransferOp::H2d(c),
            TransferOp::H2d(b),
        ])
        .unwrap();
    t.synchronize(&batch.ticket).unwrap();
    let e = vec![expected(&bytes)[0].clone(); 3];
    v1::acceptance(&batch, &t.poll(&batch.ticket).unwrap(), &e, 1, None);
    assert!(matches!(
        t.take_destination(&batch.ticket, 0, epochs()),
        Err(Error::Rejected)
    ));
    t.cancel(&batch.ticket).unwrap();
    // The rejected sibling was returned at submission; it has no source to recover.
    assert!(matches!(
        t.recover_source(&batch.ticket, 1),
        Err(Error::Rejected)
    ));
    clean(&mut t, &batch.ticket, &[0, 2], &keep_a);
    t.release_device(&keep_b).unwrap();
    drop(batch);
    t.release_device(&keep_c).unwrap();
    println!(
        "PASS v1.1 acceptance exhaustive native mixed batch; rejected sibling blocks publication"
    );

    let (op, keep) = h2d(&mut t, &bytes);
    let ticket = t.h2d(op).unwrap();
    t.synchronize(&ticket).unwrap();
    v1::transfer_completion_bytes(&mut t, &ticket, &expected(&bytes), true);
    for stale in v1::stale_epochs(epochs()) {
        assert!(matches!(
            t.ready_view(&ticket, 0, stale),
            Err(Error::StaleEpoch)
        ));
    }
    assert_eq!(
        t.ready_view(&ticket, 0, epochs())
            .unwrap()
            .destination()
            .allocation_id(),
        keep.allocation_id()
    );
    assert_eq!(t.cancel(&ticket), Ok(CancelState::AlreadyPublished));
    let destination = t.take_destination(&ticket, 0, epochs()).unwrap();
    assert!(matches!(
        t.take_destination(&ticket, 0, epochs()),
        Err(Error::AlreadyReleased)
    ));
    assert_eq!(t.retire(&ticket, None), Err(Error::Busy));
    let f = t.record_consumer(&ticket).unwrap();
    let forged = FenceId {
        sequence: f.sequence + 1,
        ..f
    };
    assert_eq!(t.retire(&ticket, Some(forged)), Err(Error::WrongOwner));
    stream.synchronize().unwrap();
    t.retire(&ticket, Some(f)).unwrap();
    t.acknowledge(&ticket).unwrap();
    drop(destination);
    t.release_device(&keep).unwrap();
    println!(
        "PASS v1.2 transfer_completion_bytes native CUDA; stale epochs, ready publication, take once, authentic consumer fence"
    );
    // A live destination consumer on an allocation also blocks its D2H-source
    // retirement. The source must not become reusable just because DMA finished.
    let (op, keep) = h2d(&mut t, &bytes);
    let up = t.h2d(op).unwrap();
    t.synchronize(&up).unwrap();
    t.with_destination(&up, 0, epochs(), |device, stream| {
        assert_eq!(stream.clone_dtoh(device).unwrap(), bytes);
        Ok(())
    })
    .unwrap();
    let producer = t.record_producer(epochs().src_gen).unwrap();
    let down_host = t.alloc_host(bytes.len(), request()).unwrap();
    let down_device = t.retain_device(&keep).unwrap();
    let down = t
        .d2h(CopyOp {
            host: down_host,
            device: down_device,
            bytes: bytes.len() as u64,
            epochs: epochs(),
            producer_fence: Some(producer),
        })
        .unwrap();
    t.synchronize(&down).unwrap();
    assert_eq!(t.retire_source(&down), Err(Error::Busy));
    assert!(matches!(t.take_device(&keep), Err(Error::Busy)));
    let consumer = t.record_consumer(&up).unwrap();
    stream.synchronize().unwrap();
    t.retire(&up, Some(consumer)).unwrap();
    t.acknowledge(&up).unwrap();
    t.retire_source(&down).unwrap();
    t.release_device(&keep).unwrap();
    let Destination::Host(host) = t.take_destination(&down, 0, epochs()).unwrap() else {
        panic!("D2H destination is not host")
    };
    assert_eq!(host.bytes().unwrap(), bytes);
    let consumer = t.record_consumer(&down).unwrap();
    stream.synchronize().unwrap();
    t.retire(&down, Some(consumer)).unwrap();
    t.acknowledge(&down).unwrap();
    assert_eq!(host.bytes().unwrap(), bytes);
    assert_eq!(gov.borrow().used().pinned, bytes.len() as u64);
    drop(host);
    t.release_producer(producer).unwrap();
    println!(
        "PASS additive source retirement Busy while source consumer bound; host destination survives source release"
    );
    canonical_source_retirement(&mut t, &stream);
    canonical_hand_back(&mut t, &stream);
    destination_drop_retention(&mut t, &stream);
    cancelled_restore_recovers_source(&mut t, &stream);
    cancel_refused_after_source_consumed(&mut t);
    assert_eq!(gov.borrow().used(), TierBudget::zero(1));
    println!("PASS native governor zero after controlled drain");
}

// Day-11 rule 1 (lead ruling 9), recoverable outcome: the producer is held pending by a native
// host function, a source graph pin and a real source consumer event are live; every advance is
// an observed CUDA event or a released pin, never a flag. The frozen schedules are unchanged.
fn cancelled_restore_recovers_source(t: &mut CudaTransfers, stream: &Arc<CudaStream>) {
    let bytes = vec![47u8; 4096];
    let (op, keep) = h2d(t, &bytes);
    let hold = ProducerHold::new(stream);
    let ticket = t.h2d(op).unwrap();
    assert!(!t.poll(&ticket).unwrap().producer_done);
    let mut source_graph = Some(t.pin_source_graph(&ticket).unwrap());
    let charged = t.used();
    let host = v1::transfer_cancel_recovers_source(
        t,
        ticket,
        |t, ticket, step| match step {
            v1::RecoverStep::Unknown => t.quarantine_observation(ticket).unwrap(),
            v1::RecoverStep::Producer => {
                hold.release();
                stream.synchronize().unwrap();
            }
            v1::RecoverStep::Recover => t.synchronize(ticket).unwrap(),
            v1::RecoverStep::SourceConsumer => {
                t.record_source_consumer(ticket).unwrap();
                stream.synchronize().unwrap();
            }
            v1::RecoverStep::SourceGraph => drop(source_graph.take()),
        },
        |t| {
            assert_eq!(t.used(), charged);
            assert_eq!(t.device_registry_len(), 1);
        },
    );
    // The untouched source is the caller's again and carries its own pinned charge.
    assert_eq!(host.bytes().unwrap(), bytes);
    assert_eq!(t.used().pinned, bytes.len() as u64);
    assert_eq!(t.used().inflight, 0);
    drop(host);
    t.release_device(&keep).unwrap();
    assert_eq!(t.used(), TierBudget::zero(1));
    println!("PASS rule cancelled-restore-recovers-source native CUDA");
}
// Day-11 rule 1, forbidden order: once the source left through per-side retirement, cancel is
// refused and no recovery is offered.
fn cancel_refused_after_source_consumed(t: &mut CudaTransfers) {
    let bytes = vec![53u8; 4096];
    let (op, keep) = h2d(t, &bytes);
    let ticket = t.h2d(op).unwrap();
    v1::transfer_cancel_refused_after_source_consumed(t, ticket, |t, ticket| {
        t.synchronize(ticket).unwrap()
    });
    t.release_device(&keep).unwrap();
    assert_eq!(t.used(), TierBudget::zero(1));
    println!("PASS rule cancel-refused-after-source-consumed native CUDA");
}
// A native stream callback holds production pending without launching a
// spin kernel or synthesizing completion. It never calls a CUDA API. RAII opens
// the latch even if a fixture assertion unwinds.
struct ProducerHold(Arc<AtomicBool>);
impl ProducerHold {
    fn new(stream: &Arc<CudaStream>) -> Self {
        stream.synchronize().unwrap();
        let latch = Arc::new(AtomicBool::new(false));
        let raw = Box::into_raw(Box::new(latch.clone()));
        let result =
            unsafe { sys::cuLaunchHostFunc(stream.cu_stream(), Some(held_producer), raw.cast()) };
        if result != sys::CUresult::CUDA_SUCCESS {
            drop(unsafe { Box::from_raw(raw) });
            panic!("cannot hold native producer: {result:?}");
        }
        Self(latch)
    }
    fn release(&self) {
        self.0.store(true, Ordering::Release);
    }
}
impl Drop for ProducerHold {
    fn drop(&mut self) {
        self.release();
    }
}
unsafe extern "C" fn held_producer(raw: *mut std::ffi::c_void) {
    let latch = unsafe { Box::from_raw(raw.cast::<Arc<AtomicBool>>()) };
    while !latch.load(Ordering::Acquire) {
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}

// All lifetime advances use native CUDA events or real lease/graph retention.
// Observation loss is explicitly injected; recovery queries the recorded event.
fn canonical_source_retirement(t: &mut CudaTransfers, stream: &Arc<CudaStream>) {
    let bytes = vec![11u8; 4096];
    // The source is also a published destination of an earlier transfer. Its
    // live native consumer binding must block source retirement independently.
    let (op, keep) = h2d(t, &bytes);
    let up = t.h2d(op).unwrap();
    t.synchronize(&up).unwrap();
    t.with_destination(&up, 0, epochs(), |device, stream| {
        assert_eq!(stream.clone_dtoh(device).unwrap(), bytes);
        Ok(())
    })
    .unwrap();
    let host = t.alloc_host(bytes.len(), request()).unwrap();
    let device = t.retain_device(&keep).unwrap();
    let hold = ProducerHold::new(stream);
    let producer = t.record_producer(epochs().src_gen).unwrap();
    let down = t
        .d2h(CopyOp {
            host,
            device,
            bytes: bytes.len() as u64,
            epochs: epochs(),
            producer_fence: Some(producer),
        })
        .unwrap();
    assert!(!t.poll(&down).unwrap().producer_done);
    let mut source_graph = Some(t.pin_source_graph(&down).unwrap());
    let mut destination_graph = Some(t.pin_destination_graph(&down).unwrap());
    let destination = RefCell::new(None::<CudaPinnedLease>);
    let consumer = RefCell::new(None);
    v1::transfer_source_retirement(
        t,
        down,
        |t, ticket, step| {
            match step {
                v1::SourceStep::Unknown => t.quarantine_observation(ticket).unwrap(),
                v1::SourceStep::Producer => {
                    hold.release();
                    stream.synchronize().unwrap();
                }
                v1::SourceStep::Recover => {
                    t.synchronize(ticket).unwrap();
                    let Destination::Host(mut host) =
                        t.take_destination(ticket, 0, epochs()).unwrap()
                    else {
                        panic!("D2H destination is not host")
                    };
                    assert_eq!(host.write(&bytes), Err(Error::Busy));
                    *destination.borrow_mut() = Some(host);
                }
                v1::SourceStep::SourceConsumer => {
                    let f = t.record_consumer(&up).unwrap();
                    t.record_source_consumer(ticket).unwrap();
                    stream.synchronize().unwrap();
                    t.retire(&up, Some(f)).unwrap();
                    t.acknowledge(&up).unwrap();
                }
                v1::SourceStep::SourceGraph => {
                    drop(source_graph.take());
                }
                v1::SourceStep::DestinationConsumer => {
                    // A CPU read of the pinned destination is complete before the
                    // owner records the consumer completion event.
                    assert_eq!(
                        destination.borrow().as_ref().unwrap().bytes().unwrap(),
                        bytes
                    );
                    let f = t.record_consumer(ticket).unwrap();
                    stream.synchronize().unwrap();
                    *consumer.borrow_mut() = Some(f);
                    assert_eq!(t.retire(ticket, Some(f)), Err(Error::Busy));
                }
                v1::SourceStep::DestinationGraph => {
                    drop(destination_graph.take());
                    t.retire(ticket, *consumer.borrow()).unwrap();
                }
            }
        },
        |t| t.release_device(&keep),
        |t| {
            assert!(t.used().pinned >= bytes.len() as u64);
            if let Some(host) = destination.borrow().as_ref() {
                assert_eq!(host.bytes().unwrap(), bytes);
            }
        },
    );
    assert_eq!(t.used().device[0], 0);
    assert_eq!(t.used().inflight, 0);
    assert_eq!(t.used().pinned, bytes.len() as u64);
    let mut host = destination.into_inner().unwrap();
    host.write(&bytes).unwrap(); // exclusive again after acknowledgement
    drop(host);
    t.release_producer(producer).unwrap();
    assert_eq!(t.used(), TierBudget::zero(1));
    println!("PASS v1.3 transfer_source_retirement native CUDA");
}

fn destination_drop_retention(t: &mut CudaTransfers, stream: &Arc<CudaStream>) {
    let device = t.alloc_device(4096, epochs().src_gen, request()).unwrap();
    let keep = t.retain_device(&device).unwrap();
    let host = t.alloc_host(4096, request()).unwrap();
    let producer = t.record_producer(epochs().src_gen).unwrap();
    let ticket = t
        .d2h(CopyOp {
            host,
            device,
            bytes: 4096,
            epochs: epochs(),
            producer_fence: Some(producer),
        })
        .unwrap();
    let graph = t.pin_destination_graph(&ticket).unwrap();
    t.synchronize(&ticket).unwrap();
    let Destination::Host(host) = t.take_destination(&ticket, 0, epochs()).unwrap() else {
        panic!("D2H destination is not host")
    };
    let device = t.retain_device(&keep).unwrap();
    let before = t.used();
    let rejected = t
        .d2h(CopyOp {
            host,
            device,
            bytes: 4096,
            epochs: epochs(),
            producer_fence: Some(producer),
        })
        .unwrap_err();
    assert_eq!(rejected.error, Error::Busy);
    assert_eq!(rejected.op.host.bytes().unwrap(), vec![0; 4096]);
    assert_eq!(t.used(), before);
    let consumer = t.record_consumer(&ticket).unwrap();
    stream.synchronize().unwrap();
    drop(rejected);
    t.retire_source(&ticket).unwrap();
    t.release_device(&keep).unwrap();
    assert_eq!(t.used().pinned, 4096);
    assert_eq!(t.retire(&ticket, Some(consumer)), Err(Error::Busy));
    drop(graph);
    t.retire(&ticket, Some(consumer)).unwrap();
    assert_eq!(t.used().pinned, 4096);
    t.acknowledge(&ticket).unwrap();
    t.release_producer(producer).unwrap();
    assert_eq!(t.used(), TierBudget::zero(1));
    println!(
        "PASS additive dropped destination retains backing and charge until graph retirement and acknowledgement"
    );
}

struct HandBack<'a> {
    foreign: CudaTransfers,
    t: &'a mut CudaTransfers,
    stream: &'a Arc<CudaStream>,
    ticket: TransferTicket,
    lease: DeviceLease,
    graph: Option<GraphPin>,
    consumer: Option<FenceId>,
    hold: ProducerHold,
    pointer: u64,
    bytes: Vec<u8>,
}
impl v1::DeviceHandBackFixture for HandBack<'_> {
    type Backing = CudaSlice<u8>;
    fn take(&mut self) -> Result<Self::Backing> {
        self.t.take_device(&self.lease)
    }
    fn advance(&mut self, step: v1::HandBackStep) {
        match step {
            v1::HandBackStep::Unknown => self.t.quarantine_observation(&self.ticket).unwrap(),
            v1::HandBackStep::Producer => {
                self.hold.release();
                self.stream.synchronize().unwrap();
            }
            v1::HandBackStep::Recover => {
                self.t.synchronize(&self.ticket).unwrap();
                self.pointer = self
                    .t
                    .with_destination(&self.ticket, 0, epochs(), |device, stream| {
                        assert_eq!(stream.clone_dtoh(device).unwrap(), self.bytes);
                        let (ptr, _guard) = device.device_ptr(stream);
                        Ok(ptr)
                    })
                    .unwrap();
            }
            v1::HandBackStep::Consumer => {
                self.consumer = Some(self.t.record_consumer(&self.ticket).unwrap());
                self.stream.synchronize().unwrap();
                assert_eq!(self.t.retire(&self.ticket, self.consumer), Err(Error::Busy));
            }
            v1::HandBackStep::Graph => {
                drop(self.graph.take());
                self.t.retire(&self.ticket, self.consumer).unwrap();
                self.t.acknowledge(&self.ticket).unwrap();
            }
        }
    }
    fn assert_registered_and_charged(&mut self) {
        assert_eq!(self.t.device_registry_len(), 1);
        assert_eq!(self.t.used().device[0], self.bytes.len() as u64);
    }
    fn assert_returned_once(&mut self, backing: Self::Backing) {
        let (ptr, guard) = backing.device_ptr(self.stream);
        assert_eq!(ptr, self.pointer);
        drop(guard);
        assert_eq!(self.stream.clone_dtoh(&backing).unwrap(), self.bytes);
        assert_eq!(self.t.device_registry_len(), 0);
        assert_eq!(self.t.used(), TierBudget::zero(1));
        drop(backing);
    }
    fn reject_foreign(&mut self) {
        assert!(matches!(
            self.foreign.take_device(&self.lease),
            Err(Error::ForeignLease | Error::WrongOwner)
        ));
        assert_eq!(self.foreign.used(), TierBudget::zero(1));
    }
}
fn canonical_hand_back(t: &mut CudaTransfers, stream: &Arc<CudaStream>) {
    // Context/new_stream setup may synchronize the primary context. Construct
    // the foreign owner BEFORE holding work pending on the tested stream.
    let (foreign, _, _) = setup();
    let bytes = vec![29u8; 4096];
    let (op, lease) = h2d(t, &bytes);
    let hold = ProducerHold::new(stream);
    let ticket = t.h2d(op).unwrap();
    assert!(!t.poll(&ticket).unwrap().producer_done);
    let graph = Some(t.pin_destination_graph(&ticket).unwrap());
    let mut fixture = HandBack {
        foreign,
        t,
        stream,
        ticket,
        lease,
        graph,
        consumer: None,
        hold,
        pointer: 0,
        bytes,
    };
    v1::device_hand_back(&mut fixture);
    println!("PASS v1.3 device_hand_back native CUDA");
}

fn roundtrip() {
    let (mut t, stream, gov) = setup();
    for n in [4096usize, 65536, 1048576, 16777216, 67108864, 268435456] {
        let bytes: Vec<u8> = (0..n)
            .map(|i| ((i.wrapping_mul(37) ^ (i >> 8) ^ (i >> 17)) & 255) as u8)
            .collect();
        let source = stream.clone_htod(&bytes).unwrap();
        let source = t
            .register_device(source, epochs().src_gen, request())
            .unwrap();
        let producer = t.record_producer(epochs().src_gen).unwrap();
        let host = t.alloc_host(n, request()).unwrap();
        let source_copy = t.retain_device(&source).unwrap();
        let down = t
            .d2h(CopyOp {
                host,
                device: source_copy,
                bytes: n as u64,
                epochs: epochs(),
                producer_fence: Some(producer),
            })
            .unwrap();
        t.synchronize(&down).unwrap();
        v1::transfer_completion_bytes(&mut t, &down, &expected(&bytes), false);
        let Destination::Host(host) = t.take_destination(&down, 0, epochs()).unwrap() else {
            panic!("D2H destination is not host")
        };
        assert!(matches!(t.take_device(&source), Err(Error::Busy)));
        let graph = t.pin_graph(&down).unwrap();
        assert_eq!(t.retire_source(&down), Err(Error::Busy));
        drop(graph);
        t.retire_source(&down).unwrap();
        t.retire_source(&down).unwrap(); // additive API is idempotent
        assert!(!t.retired(&down).unwrap());
        t.release_device(&source).unwrap();
        assert_eq!(gov.borrow().used().device[0], 0);
        assert_eq!(gov.borrow().used().pinned, n as u64);
        assert_eq!(host.bytes().unwrap(), bytes);
        assert_eq!(checksum(host.bytes().unwrap()), checksum(&bytes));
        assert_eq!(t.retire(&down, None), Err(Error::Busy));
        let target = t.alloc_device(n, epochs().dst_gen, request()).unwrap();
        let keep = t.retain_device(&target).unwrap();
        let up = t
            .h2d(CopyOp {
                host,
                device: target,
                bytes: n as u64,
                epochs: epochs(),
                producer_fence: None,
            })
            .unwrap();
        t.synchronize(&up).unwrap();
        v1::transfer_completion_bytes(&mut t, &up, &expected(&bytes), true);
        let allocation_ptr = t
            .with_destination(&up, 0, epochs(), |device, stream| {
                let (ptr, _guard) = device.device_ptr(stream);
                Ok(ptr)
            })
            .unwrap();
        let Destination::Device(destination) = t.take_destination(&up, 0, epochs()).unwrap() else {
            panic!("H2D destination is not device")
        };
        assert!(matches!(t.take_device(&destination), Err(Error::Busy)));
        t.retire_source(&up).unwrap();
        // The earlier D2H ticket still owns the host until acknowledgement.
        assert_eq!(gov.borrow().used().pinned, n as u64);
        let down_done = t.record_consumer(&down).unwrap();
        stream.synchronize().unwrap();
        t.retire(&down, Some(down_done)).unwrap();
        t.acknowledge(&down).unwrap();
        assert_eq!(gov.borrow().used().pinned, 0);
        assert!(!t.retired(&up).unwrap());
        assert!(matches!(t.take_device(&destination), Err(Error::Busy)));
        let up_done = t.record_consumer(&up).unwrap();
        stream.synchronize().unwrap();
        t.retire(&up, Some(up_done)).unwrap();
        t.acknowledge(&up).unwrap();
        assert!(matches!(t.take_device(&destination), Err(Error::Busy))); // retained keep
        drop(keep);
        let operand = t.take_device(&destination).unwrap();
        assert_eq!(operand.len(), n);
        {
            let (ptr, _guard) = operand.device_ptr(&stream);
            assert_eq!(
                ptr, allocation_ptr,
                "hand-back changed the native allocation"
            );
        }
        assert_eq!(gov.borrow().used().device[0], 0);
        assert!(matches!(
            t.take_device(&destination),
            Err(Error::ForeignLease)
        ));
        let actual = stream.clone_dtoh(&operand).unwrap();
        let want = checksum(&bytes);
        let got = checksum(&actual);
        assert_eq!(got, want, "real destination bytes differ at size {n}");
        assert_eq!(actual, bytes);
        drop(operand);

        t.release_producer(producer).unwrap();
        assert_eq!(gov.borrow().used(), TierBudget::zero(1));
        let hex = |d: Digest| d.iter().map(|b| format!("{b:02x}")).collect::<String>();
        println!(
            "PASS native D2H-H2D roundtrip bytes={n} N=1 expected_sha256={} actual_sha256={} byte_exact=true source_freed_host_live=true handback_no_copy=true governor_zero=true",
            hex(want),
            hex(got)
        );
    }
}
/// One timed D2H then H2D roundtrip of the contract under one pinned arm (WP-A day 13,
/// `research/spill-a-20260919/PINNED-FLAGS.md` section 6). The phases are the engine's own calls in
/// the order the door runs them: allocation (page-lock plus the zero fill), the D2H (submit to the
/// owner stream's synchronize: the DMA, no host read), the engine's completion hash
/// (`synchronize(&ticket)` on an already complete copy is `progress`'s SHA-256 over the
/// destination), the bind hash (`checksum(host.bytes())` over the taken destination, the read
/// `bind_tier_image` does), the byte compare, the H2D (submit to stream synchronize) and the
/// H2D-source hash (Option C's completion read). Byte exactness covers both legs.
struct PinnedSample {
    arm: memra_engine::tier_transfer::PinnedKind,
    driver_flags: u32,
    alloc_ms: f64,
    d2h_ms: f64,
    engine_hash_ms: f64,
    bind_hash_ms: f64,
    compare_ms: f64,
    h2d_ms: f64,
    source_hash_ms: f64,
    byte_exact: bool,
}
fn pinned_roundtrip(
    t: &mut CudaTransfers,
    stream: &Arc<CudaStream>,
    gov: &SharedBudget,
    pattern: &[u8],
    want: Digest,
    arm: memra_engine::tier_transfer::PinnedKind,
) -> PinnedSample {
    use std::time::Instant;
    let ms = |t0: Instant| t0.elapsed().as_secs_f64() * 1000.0;
    let n = pattern.len();
    // The device source is staged outside every timed phase.
    let source = stream.clone_htod(pattern).unwrap();
    stream.synchronize().unwrap();
    let source = t
        .register_device(source, epochs().src_gen, request())
        .unwrap();
    let producer = t.record_producer(epochs().src_gen).unwrap();
    let t0 = Instant::now();
    let host = t.alloc_host_kind(n, request(), arm).unwrap();
    let alloc_ms = ms(t0);
    assert_eq!(host.pinned_kind(), Some(arm));
    let mut driver_flags: std::ffi::c_uint = u32::MAX;
    // SAFETY: the pointer is a live cuMemHostAlloc allocation owned by `host` for the duration of
    // the call; the query writes only `driver_flags`.
    unsafe {
        sys::cuMemHostGetFlags(&mut driver_flags, host.bytes().unwrap().as_ptr() as *mut _)
            .result()
            .unwrap()
    };
    let source_copy = t.retain_device(&source).unwrap();
    let t0 = Instant::now();
    let down = t
        .d2h(CopyOp {
            host,
            device: source_copy,
            bytes: n as u64,
            epochs: epochs(),
            producer_fence: Some(producer),
        })
        .unwrap();
    stream.synchronize().unwrap();
    let d2h_ms = ms(t0);
    let t0 = Instant::now();
    t.synchronize(&down).unwrap();
    let engine_hash_ms = ms(t0);
    let Destination::Host(host) = t.take_destination(&down, 0, epochs()).unwrap() else {
        panic!("D2H destination is not host")
    };
    let t0 = Instant::now();
    let got = checksum(host.bytes().unwrap());
    let bind_hash_ms = ms(t0);
    let t0 = Instant::now();
    let d2h_exact = got == want && host.bytes().unwrap() == pattern;
    let compare_ms = ms(t0);
    t.retire_source(&down).unwrap();
    t.release_device(&source).unwrap();
    let target = t.alloc_device(n, epochs().dst_gen, request()).unwrap();
    let keep = t.retain_device(&target).unwrap();
    let t0 = Instant::now();
    let up = t
        .h2d(CopyOp {
            host,
            device: target,
            bytes: n as u64,
            epochs: epochs(),
            producer_fence: None,
        })
        .unwrap();
    stream.synchronize().unwrap();
    let h2d_ms = ms(t0);
    let t0 = Instant::now();
    t.synchronize(&up).unwrap();
    let source_hash_ms = ms(t0);
    let restored = t
        .with_destination(&up, 0, epochs(), |device, stream| {
            stream
                .clone_dtoh(&device.slice(..n))
                .map_err(|_| Error::Quarantined)
        })
        .unwrap();
    let h2d_exact = restored == pattern && checksum(&restored) == want;
    let Destination::Device(destination) = t.take_destination(&up, 0, epochs()).unwrap() else {
        panic!("H2D destination is not device")
    };
    t.retire_source(&up).unwrap();
    let down_done = t.record_consumer(&down).unwrap();
    stream.synchronize().unwrap();
    t.retire(&down, Some(down_done)).unwrap();
    t.acknowledge(&down).unwrap();
    let up_done = t.record_consumer(&up).unwrap();
    stream.synchronize().unwrap();
    t.retire(&up, Some(up_done)).unwrap();
    t.acknowledge(&up).unwrap();
    drop(keep);
    drop(t.take_device(&destination).unwrap());
    t.release_producer(producer).unwrap();
    assert_eq!(gov.borrow().used(), TierBudget::zero(1));
    PinnedSample {
        arm,
        driver_flags,
        alloc_ms,
        d2h_ms,
        engine_hash_ms,
        bind_hash_ms,
        compare_ms,
        h2d_ms,
        source_hash_ms,
        byte_exact: d2h_exact && h2d_exact,
    }
}
/// One timed phase of a `PinnedSample`, in milliseconds.
type Phase = fn(&PinnedSample) -> f64;
fn median(values: &[f64]) -> f64 {
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n == 0 {
        f64::NAN
    } else if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}
/// `tier-transfer-gate pinned-ab [--bytes N] [--pairs N]`: the write-combined arm (A, today's
/// default) against the cached arm (B) in one process, one CUDA context and one collector lock
/// hold: one untimed warm-up roundtrip per arm, then A B A B (order 1, `pairs` pairs) and B A B A
/// (order 2), one line per roundtrip, medians per arm per order and pooled, and the pre-registered
/// rule of `research/spill-a-20260919/DAY13.md` evaluated for this card. The collector reads the
/// one `RESULT` line. Run ONLY through tier-battery (the rig lock and the 250 ms telemetry).
fn pinned_ab(bytes: usize, pairs: usize) {
    use memra_engine::tier_transfer::PinnedKind;
    assert!(bytes >= 4096 && pairs >= 1, "bytes >= 4096, pairs >= 1");
    let (mut t, stream, gov) = setup();
    let pattern: Vec<u8> = (0..bytes)
        .map(|i| ((i.wrapping_mul(37) ^ (i >> 8) ^ (i >> 17)) & 255) as u8)
        .collect();
    let want = checksum(&pattern);
    let (a, b) = (PinnedKind::WriteCombined, PinnedKind::Cached);
    let line = |tag: &str, order: usize, pair: usize, s: &PinnedSample| {
        println!(
            "PINNED-AB {tag} order={order} pair={pair} arm={} bytes={bytes} driver_flags={} alloc_ms={:.3} d2h_ms={:.3} engine_hash_ms={:.3} bind_hash_ms={:.3} compare_ms={:.3} h2d_ms={:.3} source_hash_ms={:.3} byte_exact={}",
            s.arm.name(),
            s.driver_flags,
            s.alloc_ms,
            s.d2h_ms,
            s.engine_hash_ms,
            s.bind_hash_ms,
            s.compare_ms,
            s.h2d_ms,
            s.source_hash_ms,
            s.byte_exact
        );
    };
    let mut warm_exact = true;
    for arm in [a, b] {
        let s = pinned_roundtrip(&mut t, &stream, &gov, &pattern, want, arm);
        warm_exact &= s.byte_exact;
        line("warmup", 0, 0, &s);
    }
    // samples[order][pair] = (first arm's sample, second arm's sample); order 1 is A then B.
    let mut samples: Vec<Vec<(PinnedSample, PinnedSample)>> = vec![];
    for (order, (first, second)) in [(a, b), (b, a)].into_iter().enumerate() {
        let mut rows = vec![];
        for pair in 1..=pairs {
            let x = pinned_roundtrip(&mut t, &stream, &gov, &pattern, want, first);
            line("timed", order + 1, pair, &x);
            let y = pinned_roundtrip(&mut t, &stream, &gov, &pattern, want, second);
            line("timed", order + 1, pair, &y);
            rows.push((x, y));
        }
        samples.push(rows);
    }
    let pick = |order: usize, arm: PinnedKind, f: Phase| -> Vec<f64> {
        samples[order]
            .iter()
            .map(|(x, y)| if x.arm == arm { f(x) } else { f(y) })
            .collect()
    };
    let phases: [(&str, Phase); 7] = [
        ("alloc_ms", |s| s.alloc_ms),
        ("d2h_ms", |s| s.d2h_ms),
        ("engine_hash_ms", |s| s.engine_hash_ms),
        ("bind_hash_ms", |s| s.bind_hash_ms),
        ("compare_ms", |s| s.compare_ms),
        ("h2d_ms", |s| s.h2d_ms),
        ("source_hash_ms", |s| s.source_hash_ms),
    ];
    let mut medians: Vec<String> = vec![];
    for (name, f) in phases {
        for arm in [a, b] {
            let o1 = pick(0, arm, f);
            let o2 = pick(1, arm, f);
            let pooled: Vec<f64> = o1.iter().chain(o2.iter()).copied().collect();
            println!(
                "PINNED-AB median {name} arm={} order1={:.3} (N={}) order2={:.3} (N={}) pooled={:.3} (N={}) min={:.3} max={:.3}",
                arm.name(),
                median(&o1),
                o1.len(),
                median(&o2),
                o2.len(),
                median(&pooled),
                pooled.len(),
                pooled.iter().cloned().fold(f64::INFINITY, f64::min),
                pooled.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            );
            medians.push(format!(
                "\"{name}.{}\":{{\"order1\":{:.3},\"order2\":{:.3},\"pooled\":{:.3},\"n_per_order\":{},\"n_pooled\":{}}}",
                arm.name(),
                median(&o1),
                median(&o2),
                median(&pooled),
                o1.len(),
                pooled.len()
            ));
        }
    }
    // The pre-registered rule (DAY13.md): the cached arm wins on this card iff byte exactness
    // holds in every cell, its bind-hash wall is below the write-combined arm's at every pair of
    // both orders and in both orders' medians, and its D2H wall is not above the write-combined
    // arm's at any pair (the DMA is predicted flag-independent; a regression there is a loss).
    // The strict D2H reading (cached strictly below at every pair) is reported beside it.
    let all_exact = warm_exact
        && samples
            .iter()
            .flatten()
            .all(|(x, y)| x.byte_exact && y.byte_exact);
    let flags_ok = samples.iter().flatten().all(|(x, y)| {
        x.driver_flags == x.arm.host_alloc_flags() && y.driver_flags == y.arm.host_alloc_flags()
    });
    let per_pair = |f: Phase, cmp: fn(f64, f64) -> bool| -> (usize, usize) {
        let mut hits = 0;
        let mut total = 0;
        for (x, y) in samples.iter().flatten() {
            let (wc, cached) = if x.arm == a { (x, y) } else { (y, x) };
            total += 1;
            if cmp(f(cached), f(wc)) {
                hits += 1;
            }
        }
        (hits, total)
    };
    let bind = per_pair(|s| s.bind_hash_ms, |c, w| c < w);
    let engine = per_pair(|s| s.engine_hash_ms, |c, w| c < w);
    let d2h_no_regression = per_pair(|s| s.d2h_ms, |c, w| c <= w);
    let d2h_strict = per_pair(|s| s.d2h_ms, |c, w| c < w);
    let h2d_no_regression = per_pair(|s| s.h2d_ms, |c, w| c <= w);
    let medians_win = (0..2).all(|o| {
        median(&pick(o, b, |s| s.bind_hash_ms)) < median(&pick(o, a, |s| s.bind_hash_ms))
            && median(&pick(o, b, |s| s.d2h_ms)) <= median(&pick(o, a, |s| s.d2h_ms))
    });
    let cached_wins = all_exact
        && flags_ok
        && bind.0 == bind.1
        && d2h_no_regression.0 == d2h_no_regression.1
        && medians_win;
    let cached_wins_strict_d2h = cached_wins && d2h_strict.0 == d2h_strict.1;
    let verdict = |win: bool| {
        if win {
            "wins-on-this-card"
        } else {
            "inconclusive"
        }
    };
    println!(
        "PINNED-AB rule byte_exact_all={all_exact} driver_flags_honoured={flags_ok} bind_hash_cached_below_wc={}/{} engine_hash_cached_below_wc={}/{} d2h_cached_not_above_wc={}/{} d2h_cached_strictly_below_wc={}/{} h2d_cached_not_above_wc={}/{} medians_both_orders={medians_win} cached_arm={} cached_arm_strict_d2h_reading={}",
        bind.0,
        bind.1,
        engine.0,
        engine.1,
        d2h_no_regression.0,
        d2h_no_regression.1,
        d2h_strict.0,
        d2h_strict.1,
        h2d_no_regression.0,
        h2d_no_regression.1,
        verdict(cached_wins),
        verdict(cached_wins_strict_d2h),
    );
    let pair = |(hits, total): (usize, usize)| format!("[{hits},{total}]");
    let rows: Vec<String> = samples
        .iter()
        .enumerate()
        .flat_map(|(o, rows)| {
            rows.iter().enumerate().flat_map(move |(p, (x, y))| {
                [x, y].into_iter().map(move |s| {
                    format!(
                        "{{\"order\":{},\"pair\":{},\"arm\":\"{}\",\"driver_flags\":{},\"alloc_ms\":{:.3},\"d2h_ms\":{:.3},\"engine_hash_ms\":{:.3},\"bind_hash_ms\":{:.3},\"compare_ms\":{:.3},\"h2d_ms\":{:.3},\"source_hash_ms\":{:.3},\"byte_exact\":{}}}",
                        o + 1,
                        p + 1,
                        s.arm.name(),
                        s.driver_flags,
                        s.alloc_ms,
                        s.d2h_ms,
                        s.engine_hash_ms,
                        s.bind_hash_ms,
                        s.compare_ms,
                        s.h2d_ms,
                        s.source_hash_ms,
                        s.byte_exact
                    )
                })
            })
        })
        .collect();
    println!(
        "RESULT {{\"cell\":\"pinned-ab\",\"bytes\":{bytes},\"pairs_per_order\":{pairs},\"arms\":{{\"A\":\"{}\",\"B\":\"{}\"}},\"orders\":[\"A B\",\"B A\"],\"warmup_per_arm\":1,\"byte_exact_all\":{all_exact},\"driver_flags_honoured\":{flags_ok},\"per_pair\":{{\"bind_hash_cached_below_wc\":{},\"engine_hash_cached_below_wc\":{},\"d2h_cached_not_above_wc\":{},\"d2h_cached_strictly_below_wc\":{},\"h2d_cached_not_above_wc\":{}}},\"medians_both_orders\":{medians_win},\"cached_arm\":\"{}\",\"cached_arm_strict_d2h_reading\":\"{}\",\"medians_ms\":{{{}}},\"samples\":[{}],\"qualification\":false,\"scope\":\"one card, one window, executed-not-qualified; the target-card cell of the MEMRA_KV_HOST_CONTRACTS decide-by review, not a default change\"}}",
        a.name(),
        b.name(),
        pair(bind),
        pair(engine),
        pair(d2h_no_regression),
        pair(d2h_strict),
        pair(h2d_no_regression),
        verdict(cached_wins),
        verdict(cached_wins_strict_d2h),
        medians.join(","),
        rows.join(",")
    );
    assert!(all_exact, "byte exactness failed in at least one cell");
    assert!(flags_ok, "the driver did not record an arm's flag bits");
}
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let case = args
        .get(1)
        .expect("case: conformance|roundtrip|pinned-ab [--bytes N] [--pairs N]");
    match case.as_str() {
        "conformance" => conformance(),
        "roundtrip" => roundtrip(),
        "pinned-ab" => {
            let mut bytes = 160usize << 20;
            let mut pairs = 5usize;
            let mut rest = args[2..].iter();
            while let Some(key) = rest.next() {
                let value = rest.next().expect("missing value");
                match key.as_str() {
                    "--bytes" => bytes = value.parse().expect("--bytes N"),
                    "--pairs" => pairs = value.parse().expect("--pairs N"),
                    _ => panic!("unknown argument {key}"),
                }
            }
            pinned_ab(bytes, pairs)
        }
        _ => panic!("unknown case"),
    }
}
