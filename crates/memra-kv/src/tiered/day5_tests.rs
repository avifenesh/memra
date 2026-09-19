//! Actual B admission -> D exported CPU capacity -> B materializer schedule.
//! This seam test does not claim CUDA/PCIe transport or an attention kernel.
use super::*;
use memra_tier::peer::{test_support::FakePeerCapacity, topology::LinkHealth};

#[test]
fn b_admission_d_capacity_local_materializer_and_retirement_share_one_governor() {
    let g = Rc::new(RefCell::new(governor(1024, TierBudget::zero(2))));
    let mut peer = FakePeerCapacity::new(g.clone(), 2, epochs().state);
    peer.set_route(0, 1, true, true, LinkHealth::AtMaximum)
        .unwrap();
    let b = bundle();
    let mut admission = plan(&b);
    admission.source = Tier::PeerGpu(1);
    let hit = Lookup {
        tier: admission.source,
        storage_bytes: 128,
    };
    for (context, pool, health) in [
        (false, true, LinkHealth::AtMaximum),
        (true, false, LinkHealth::AtMaximum),
        (true, true, LinkHealth::Downgraded),
        (true, true, LinkHealth::Unknown),
    ] {
        peer.set_route(1, 0, context, pool, health).unwrap();
        assert!(matches!(
            integration::reserve_peer_source(&mut peer, &hit, &admission),
            Err(Error::Unsupported)
        ));
        // An unavailable direction does not revoke the reverse direction.
        let mut reverse = admission.clone();
        reverse.source = Tier::PeerGpu(0);
        reverse.target_device = 1;
        let reverse_hit = Lookup {
            tier: reverse.source,
            storage_bytes: 128,
        };
        let lease = integration::reserve_peer_source(&mut peer, &reverse_hit, &reverse).unwrap();
        peer.release(&lease).unwrap();
        assert_eq!(g.borrow().used(), TierBudget::zero(2));
    }
    peer.set_route(1, 0, true, true, LinkHealth::AtMaximum)
        .unwrap();
    peer.state += 1;
    assert!(matches!(
        integration::reserve_peer_source(&mut peer, &hit, &admission),
        Err(Error::StaleEpoch)
    ));
    peer.state = epochs().state;
    let source = integration::reserve_peer_source(&mut peer, &hit, &admission).unwrap();
    assert_eq!(source.device.device(), 1);
    assert_eq!(source.device.generation(), epochs().src_gen);
    let source_pin = peer.owners[1].retain(&source.device).unwrap();
    let copied_payloads = {
        let storage = peer.owners[1]
            .resolve::<RefCell<Vec<u8>>>(&source.device)
            .unwrap();
        *storage.borrow_mut() = payloads().concat();
        let bytes = storage.borrow();
        bytes
            .chunks_exact(64)
            .map(<[u8]>::to_vec)
            .collect::<Vec<_>>()
    };
    assert_eq!(copied_payloads, payloads());
    assert_eq!(peer.release(&source), Err(Error::Busy));
    assert_eq!(g.borrow().used().device, vec![0, 128]);
    assert_eq!(g.borrow().used().peer, vec![0, 128]);

    let destination_charge = g.borrow_mut().reserve(&admission.request).unwrap();
    let mut local = DeviceOwner::new(0);
    let destination = local
        .register(
            epochs().dst_gen,
            128,
            Box::new(NativeKvImage {
                bundle: b.clone(),
                payloads: copied_payloads,
            }),
            &destination_charge,
        )
        .unwrap();
    let ticket = TransferTicket {
        issuer: local.issuer(),
        sequence: 1,
        epochs: epochs(),
    };
    local.bind_destination(ticket, &destination).unwrap();
    let mut completion = Completion {
        ticket,
        producer_done: true,
        consumer_fenced: false,
        items: b
            .layout
            .segments
            .iter()
            .enumerate()
            .map(|(i, s)| ItemOutcome {
                item: i as u32,
                accepted: true,
                segments: vec![SegmentCompletion {
                    segment: 0,
                    status: ItemStatus::Complete,
                    valid_bytes: s.valid_bytes,
                    io_bytes: s.storage_bytes,
                    checksum: Some(b.checksums[i]),
                    epochs: epochs(),
                    producer_done: true,
                    consumer_fenced: true,
                    consumer_fence: Some(FenceId {
                        issuer: local.issuer(),
                        owner: 0,
                        generation: epochs().dst_gen,
                        sequence: 1,
                    }),
                    error: None,
                }],
            })
            .collect(),
    };
    assert!(matches!(
        local.ready_view(&destination, &completion, &expected(&b), epochs()),
        Err(Error::NotReady)
    ));
    completion.consumer_fenced = true;
    let view = local
        .ready_view(&destination, &completion, &expected(&b), epochs())
        .unwrap();
    let mut materializer = QwenMaterializer::new(
        &local,
        program(),
        NativeGeometry {
            group: 0,
            owner: 0,
            tokens: 1,
            k_token_bytes: 34,
            v_token_bytes: 24,
        },
    );
    let operands = materializer
        .materialize(&b, &program(), &view, epochs())
        .unwrap();
    assert_eq!(
        materializer.capture(&operands).unwrap(),
        (payloads()[0][..34].to_vec(), payloads()[1][..24].to_vec())
    );
    assert_eq!(g.borrow().used().device, vec![128, 128]);
    assert_eq!(peer.release(&source), Err(Error::Busy));
    materializer
        .retire(
            &operands,
            FenceId {
                issuer: local.issuer(),
                owner: 0,
                generation: epochs().dst_gen,
                sequence: 2,
            },
        )
        .unwrap();
    drop(operands);
    drop(materializer);
    local.retire_binding(&ticket).unwrap();
    local.release(&destination).unwrap();
    g.borrow_mut().release(&destination_charge).unwrap();
    drop(source_pin);
    peer.release(&source).unwrap();
    assert_eq!(peer.release(&source), Err(Error::ForeignLease));
    assert_eq!(g.borrow().used(), TierBudget::zero(2));
}
