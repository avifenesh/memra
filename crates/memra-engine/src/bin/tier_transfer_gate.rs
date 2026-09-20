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
fn clean(t: &mut CudaTransfers, ticket: &TransferTicket, device: &DeviceLease) {
    t.synchronize(ticket).unwrap();
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
        t.retire(ticket, None).unwrap();
    });
    t.release_device(&keep).unwrap();
    println!("PASS v1 transfer_cancel native CUDA");

    let (op, keep) = h2d(&mut t, &bytes);
    let ticket = t.h2d(op).unwrap();
    v1::transfer_complete_cancel(&mut t, ticket, &expected(&bytes), |t, ticket| {
        t.synchronize(ticket).unwrap()
    });
    clean(&mut t, &ticket, &keep);
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
    clean(&mut t, &batch.ticket, &keep_a);
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
    assert_eq!(gov.borrow().used(), TierBudget::zero(1));
    println!("PASS native governor zero after controlled drain");
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
    let destination = t.take_destination(&ticket, 0, epochs()).unwrap();
    let consumer = t.record_consumer(&ticket).unwrap();
    stream.synchronize().unwrap();
    drop(destination);
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
        let (mut foreign, _, _) = setup();
        assert!(matches!(
            foreign.take_device(&self.lease),
            Err(Error::ForeignLease | Error::WrongOwner)
        ));
        assert_eq!(foreign.used(), TierBudget::zero(1));
    }
}
fn canonical_hand_back(t: &mut CudaTransfers, stream: &Arc<CudaStream>) {
    let bytes = vec![29u8; 4096];
    let (op, lease) = h2d(t, &bytes);
    let hold = ProducerHold::new(stream);
    let ticket = t.h2d(op).unwrap();
    assert!(!t.poll(&ticket).unwrap().producer_done);
    let graph = Some(t.pin_destination_graph(&ticket).unwrap());
    let mut fixture = HandBack {
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
fn main() {
    let case = std::env::args()
        .nth(1)
        .expect("case: conformance|roundtrip");
    match case.as_str() {
        "conformance" => conformance(),
        "roundtrip" => roundtrip(),
        _ => panic!("unknown case"),
    }
}
