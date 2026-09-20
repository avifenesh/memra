//! Shared v1.2 schedule bound to B's actual two CPU materializers; no GPU proof.
use memra_kv::tiered::materializer::*;
use memra_tier::conformance as schedule;
use memra_tier::contracts::*;
#[allow(dead_code)]
#[path = "../../memra-tier/tests/contracts/support.rs"]
mod support;
use support::{Governor, bytes, epochs, program, request};

fn fixture(native: bool) -> (StateBundle, Vec<Vec<u8>>) {
    let p = program();
    let mut segments = vec![];
    let mut payloads = vec![];
    for (i, (role, n, encoding)) in if native {
        vec![
            (Role::Key, 34, digest("native-kv-encoding", b"q8_0")),
            (Role::Value, 24, digest("native-kv-encoding", b"q5_1")),
        ]
    } else {
        vec![(Role::Payload, 264, digest("opaque-test", b"packed-record"))]
    }
    .into_iter()
    .enumerate()
    {
        let storage = if native { 64 } else { 512 };
        let mut payload = bytes(n);
        payload.resize(storage, 0);
        payloads.push(payload);
        segments.push(ByteSegment {
            version: 1,
            group: 0,
            page: 0,
            owner: 0,
            role,
            tensor: None,
            offset: (i * storage) as u64,
            valid_bytes: n as u64,
            storage_bytes: storage as u64,
            alignment: 64,
            encoding: EncodingId {
                version: 1,
                program: encoding,
                row_bytes: n as u64,
            },
        });
    }
    let b = StateBundle {
        version: 1,
        id: KvBlockId::new(&p, [0; 32], &[42], 0, 0, 0, epochs().state).unwrap(),
        program: p,
        checksums: segments
            .iter()
            .zip(&payloads)
            .map(|(s, p)| checksum(&p[..s.valid_bytes as usize]))
            .collect(),
        layout: RecordLayout {
            version: 1,
            requirements: segments
                .iter()
                .map(|s| GroupRequirement {
                    version: 1,
                    group: 0,
                    owner: 0,
                    role: s.role,
                    page_count: 1,
                    pages: PageRequirement::AllPages,
                })
                .collect(),
            segments,
        },
        kind: StateKind::Active,
        committed_high_water: 1,
        owner_aliases: vec![],
    };
    b.verify(&payloads).unwrap();
    (b, payloads)
}
fn run(native: bool) {
    let (b, payloads) = fixture(native);
    let mut g = Governor::new(10000);
    let mut req = request(0, Priority::MandatoryActive);
    req.bytes.device[0] = b.layout.storage_bytes().unwrap();
    let charge = g.reserve(&req).unwrap();
    let mut owner = DeviceOwner::new(0);
    let image: Box<dyn std::any::Any> = if native {
        Box::new(NativeKvImage {
            bundle: b.clone(),
            payloads: payloads.clone(),
        })
    } else {
        Box::new(PackedImage {
            bundle: b.clone(),
            payloads: payloads.clone(),
        })
    };
    let device = owner
        .register(epochs().dst_gen, req.bytes.device[0], image, &charge)
        .unwrap();
    let ticket = TransferTicket {
        issuer: owner.issuer(),
        sequence: 1,
        epochs: epochs(),
    };
    owner.bind_destination(ticket, &device).unwrap();
    let fence = FenceId {
        issuer: owner.issuer(),
        owner: 0,
        generation: epochs().dst_gen,
        sequence: 2,
    };
    let expected: Vec<_> = b
        .layout
        .segments
        .iter()
        .zip(&b.checksums)
        .map(|(s, hash)| {
            vec![SegmentExpectation {
                valid_bytes: s.valid_bytes,
                io_bytes: s.valid_bytes,
                checksum: *hash,
            }]
        })
        .collect();
    let completion = Completion {
        ticket,
        producer_done: true,
        consumer_fenced: true,
        items: b
            .layout
            .segments
            .iter()
            .zip(&b.checksums)
            .enumerate()
            .map(|(i, (s, hash))| ItemOutcome {
                item: i as u32,
                accepted: true,
                segments: vec![SegmentCompletion {
                    segment: 0,
                    status: ItemStatus::Complete,
                    valid_bytes: s.valid_bytes,
                    io_bytes: s.storage_bytes + 4096,
                    checksum: Some(*hash),
                    epochs: epochs(),
                    producer_done: true,
                    consumer_fenced: true,
                    consumer_fence: Some(fence),
                    error: None,
                }],
            })
            .collect(),
    };
    schedule::framed_logical_bytes(&completion, &expected, true);
    let ready = owner
        .ready_view(&device, &completion, &expected, epochs())
        .unwrap();
    if native {
        let mut m = QwenMaterializer::new(
            &owner,
            b.program.clone(),
            NativeGeometry {
                group: 0,
                owner: 0,
                tokens: 1,
                k_token_bytes: 34,
                v_token_bytes: 24,
            },
        );
        schedule::kv_materializer(&mut m, &b, &ready, fence, |m, op, live| {
            if live {
                assert_eq!(op.allocation_id(), device.allocation_id());
                assert_eq!(
                    m.capture(op).unwrap(),
                    (payloads[0][..34].to_vec(), payloads[1][..24].to_vec())
                );
            } else {
                assert!(m.capture(op).is_err());
            }
            assert_eq!(g.release(&charge), Err(Error::Busy));
        });
    } else {
        let mut m = PackedMaterializer::new(&owner, b.program.clone(), b.layout.clone());
        schedule::kv_materializer(&mut m, &b, &ready, fence, |m, op, live| {
            if live {
                assert_eq!(op.allocation_id(), device.allocation_id());
                assert_eq!(m.capture(op).unwrap(), payloads);
            } else {
                assert!(m.capture(op).is_err());
            }
            assert_eq!(g.release(&charge), Err(Error::Busy));
        });
    }
    owner.retire_binding(&ticket).unwrap();
    owner.release(&device).unwrap();
    g.release(&charge).unwrap();
    assert_eq!(g.used, TierBudget::zero(2));
}
#[test]
fn revision_v12_native_q8_q5_materializer() {
    run(true);
}
#[test]
fn revision_v12_opaque_264_byte_materializer() {
    run(false);
}
