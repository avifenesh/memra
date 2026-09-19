//! Real CUDA conformance + byte roundtrip gate. Run ONLY through tier-battery.
use cudarc::driver::{CudaContext, CudaStream};
use memra_engine::tier_transfer::{CudaPinnedLease, CudaTransfers};
use memra_tier::{bank::SharedBudget, contracts::*, tier::governor::Governor};
use std::{cell::RefCell, rc::Rc, sync::Arc};
#[allow(dead_code)]
#[path = "../../../memra-tier/tests/contracts/conformance.rs"]
mod v1;
#[allow(dead_code)]
#[path = "../../../memra-tier/tests/contracts/revision_v11.rs"]
mod v11;
#[allow(dead_code)]
#[path = "../../../memra-tier/tests/contracts/revision_v12.rs"]
mod v12;

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
    v11::transfer_complete_cancel(&mut t, ticket, &expected(&bytes), |t, ticket| {
        t.synchronize(ticket).unwrap()
    });
    clean(&mut t, &ticket, &keep);
    println!("PASS v1.1 transfer_complete_cancel native CUDA");

    let (op, keep) = h2d(&mut t, &bytes);
    let ticket = t.h2d(op).unwrap();
    let mut graph = Some(t.pin_graph(&ticket).unwrap());
    v11::transfer_lifetime(
        &mut t,
        ticket,
        |t, ticket, step| match step {
            v11::LifetimeStep::Unknown => t.quarantine_observation(ticket).unwrap(),
            v11::LifetimeStep::Producer => stream.synchronize().unwrap(),
            v11::LifetimeStep::Recover => t.synchronize(ticket).unwrap(),
            v11::LifetimeStep::Consumer => {
                assert_eq!(t.retire(ticket, None), Err(Error::Busy));
            }
            v11::LifetimeStep::Graph => {
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
    let rejected = v11::transfer_zero_accept(&mut t, vec![read], |ops| {
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
    v11::acceptance(&batch, &t.poll(&batch.ticket).unwrap(), &e, 1, None);
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
    v12::transfer_completion_bytes(&mut t, &ticket, &expected(&bytes), true);
    for stale in v11::stale_epochs(epochs()) {
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
    assert_eq!(gov.borrow().used(), TierBudget::zero(1));
    println!("PASS native governor zero after controlled drain");
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
        v12::transfer_completion_bytes(&mut t, &down, &expected(&bytes), false);
        let Destination::Host(host) = t.take_destination(&down, 0, epochs()).unwrap() else {
            panic!("D2H destination is not host")
        };
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
        v12::transfer_completion_bytes(&mut t, &up, &expected(&bytes), true);
        let actual = t
            .with_destination(&up, 0, epochs(), |device, stream| {
                stream.clone_dtoh(device).map_err(|_| Error::Quarantined)
            })
            .unwrap();
        let want = checksum(&bytes);
        let got = checksum(&actual);
        assert_eq!(got, want, "real destination bytes differ at size {n}");
        assert_eq!(actual, bytes);
        let up_done = t.record_consumer(&up).unwrap();
        stream.synchronize().unwrap();
        t.retire(&up, Some(up_done)).unwrap();
        t.acknowledge(&up).unwrap();
        let down_done = t.record_consumer(&down).unwrap();
        stream.synchronize().unwrap();
        t.retire(&down, Some(down_done)).unwrap();
        t.acknowledge(&down).unwrap();
        t.release_device(&keep).unwrap();
        t.release_device(&source).unwrap();
        t.release_producer(producer).unwrap();
        assert_eq!(gov.borrow().used(), TierBudget::zero(1));
        let hex = |d: Digest| d.iter().map(|b| format!("{b:02x}")).collect::<String>();
        println!(
            "PASS native D2H-H2D roundtrip bytes={n} N=1 expected_sha256={} actual_sha256={} byte_exact=true governor_zero=true",
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
