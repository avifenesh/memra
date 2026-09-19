//! CPU fixture evidence only. D's private test fake is deliberately not imported.
use super::*;
use super::{policy::*, scheduler::*};

fn complete(owner: &DeviceOwner, b: &StateBundle, ticket: TransferTicket) -> Completion {
    Completion {
        ticket,
        producer_done: true,
        consumer_fenced: true,
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
                    epochs: ticket.epochs,
                    producer_done: true,
                    consumer_fenced: true,
                    consumer_fence: Some(FenceId {
                        issuer: owner.issuer(),
                        owner: 0,
                        generation: ticket.epochs.dst_gen,
                        sequence: 1,
                    }),
                    error: None,
                }],
            })
            .collect(),
    }
}

fn packed(pages: PageRequirement) -> (StateBundle, Vec<Vec<u8>>) {
    let mut b = bundle();
    b.id = KvBlockId::new(&b.program, [0; 32], &[42, 43, 44], 0, 0, 0, 7).unwrap();
    b.committed_high_water = 3;
    let first = match pages {
        PageRequirement::AllPages => 0,
        PageRequirement::TrailingPages(n) => 3 - n,
    };
    b.layout.requirements = vec![GroupRequirement {
        version: 1,
        group: 0,
        owner: 0,
        role: Role::Key,
        page_count: 3,
        pages,
    }];
    b.layout.segments = (first..3)
        .map(|page| ByteSegment {
            version: 1,
            group: 0,
            page,
            owner: 0,
            role: Role::Key,
            tensor: None,
            offset: page * 64,
            valid_bytes: 40,
            storage_bytes: 64,
            alignment: 64,
            encoding: EncodingId {
                version: 1,
                program: digest("opaque-fixture", b"fp8-record-with-scale"),
                row_bytes: 40,
            },
        })
        .collect();
    let bytes: Vec<Vec<u8>> = (first..3)
        .map(|page| {
            (0..64)
                .map(|i| if i < 40 { (page * 67 + i) as u8 } else { 0 })
                .collect()
        })
        .collect();
    b.checksums = bytes.iter().map(|p| checksum(&p[..40])).collect();
    b.verify(&bytes).unwrap();
    (b, bytes)
}

#[test]
fn opaque_packed_all_and_trailing_exact_bytes_identity_and_mixed_layout_refusal() {
    for pages in [PageRequirement::AllPages, PageRequirement::TrailingPages(2)] {
        let (b, bytes) = packed(pages);
        let mut owner = DeviceOwner::new(0);
        let mut g = governor(4096, TierBudget::zero(2));
        let mut r = request(0, Priority::MandatoryActive);
        r.bytes.device[0] = b.layout.storage_bytes().unwrap();
        let charge = g.reserve(&r).unwrap();
        let device = owner
            .register(
                31,
                r.bytes.device[0],
                Box::new(PackedImage {
                    bundle: b.clone(),
                    payloads: bytes.clone(),
                }),
                &charge,
            )
            .unwrap();
        let ticket = TransferTicket {
            issuer: owner.issuer(),
            sequence: 1,
            epochs: epochs(),
        };
        owner.bind_destination(ticket, &device).unwrap();
        let completion = complete(&owner, &b, ticket);
        let view = owner
            .ready_view(&device, &completion, &expected(&b), epochs())
            .unwrap();
        let mut m = PackedMaterializer::new(&owner, b.program.clone(), b.layout.clone());
        let mut wrong = b.program.clone();
        wrong.numeric[0] ^= 1;
        assert!(matches!(
            m.materialize(&b, &wrong, &view, epochs()),
            Err(Error::ProgramMismatch)
        ));
        let mut mixed = b.clone();
        mixed.layout.segments[1].encoding.program[0] ^= 1;
        assert!(m.materialize(&mixed, &b.program, &view, epochs()).is_err());
        let mut mixed_m = PackedMaterializer::new(&owner, b.program.clone(), mixed.layout.clone());
        assert!(matches!(
            mixed_m.materialize(&mixed, &b.program, &view, epochs()),
            Err(Error::Unsupported)
        ));
        let opposite = packed(if pages == PageRequirement::AllPages {
            PageRequirement::TrailingPages(2)
        } else {
            PageRequirement::AllPages
        })
        .0;
        assert!(
            m.materialize(&opposite, &b.program, &view, epochs())
                .is_err()
        );
        let op = m.materialize(&b, &b.program, &view, epochs()).unwrap();
        assert_eq!(m.capture(&op).unwrap(), bytes); // padding preserved too
        assert_eq!(op.allocation_id(), device.allocation_id());
        assert_eq!(g.release(&charge), Err(Error::Busy));
        m.retire(
            &op,
            FenceId {
                issuer: owner.issuer(),
                owner: 0,
                generation: 31,
                sequence: 2,
            },
        )
        .unwrap();
        assert!(m.capture(&op).is_err());
        drop(op);
        drop(m);
        drop(mixed_m);
        owner.retire_binding(&ticket).unwrap();
        owner.release(&device).unwrap();
        g.release(&charge).unwrap();
    }
}

#[test]
fn request_plan_records_fixture_policy_and_never_loads_recompute_winners() {
    let csv = include_str!("../../../../research/spill-b-20260919/fixtures/recompute-load.csv");
    for row in csv.lines().skip(1) {
        let c: Vec<_> = row.split(',').collect();
        let tokens: u64 = c[0].parse().unwrap();
        let bpt: u64 = c[1].parse().unwrap();
        let gb: f64 = c[2].parse().unwrap();
        let rate: f64 = c[3].parse().unwrap();
        let startup: f64 = c[4].parse().unwrap();
        let costs = PrefixCosts {
            prompt_tokens: tokens,
            reused_tokens: tokens,
            cold_prefill_ns: (tokens as f64 / rate * 1e9) as u64,
            suffix_prefill_ns: 0,
            read_ns: (tokens as f64 * bpt as f64 / gb) as u64,
            copy_ns: (startup * 1e9) as u64,
            materialize_ns: 0,
        };
        let (h, g, _, _) = fixture();
        let mut s = Scheduler::new(h, 2);
        let mut p = plan(&bundle());
        p.id = KvBlockId::new(&p.program, [0; 32], &vec![42; tokens as usize], 0, 0, 0, 0).unwrap();
        p.epochs.state = 0;
        p.committed_high_water = tokens;
        p.request.priority = Priority::Demand;
        let id = s
            .enqueue_with_costs(
                p,
                PrefetchPolicy::Wait,
                StateKind::ImmutablePrefix,
                costs,
                true,
            )
            .unwrap();
        let expected_decision = if c[5] == "load" {
            RestoreDecision::Load
        } else {
            RestoreDecision::Recompute
        };
        assert_eq!(
            s.restore_decision(id).unwrap().0,
            expected_decision,
            "{row}"
        );
        if expected_decision == RestoreDecision::Recompute {
            // There is deliberately no matching backing. A mistaken lookup/load would
            // produce NotFound instead of the explicit uncharged cold decision.
            for now in [1, 2, 3] {
                assert_eq!(
                    s.tick(now, &[], |_| Epochs {
                        state: 0,
                        ..epochs()
                    })
                    .unwrap(),
                    vec![(id, Ok(Progress::Recompute))]
                );
                assert_eq!(g.lock().unwrap().used(), TierBudget::zero(2));
            }
            assert!(matches!(
                s.consume(
                    id,
                    Epochs {
                        state: 0,
                        ..epochs()
                    },
                    |_| ()
                ),
                Err(Error::NotReady)
            ));
            s.release(id).unwrap();
        }
    }
}

#[test]
fn active_state_cannot_recompute_even_with_optional_priority() {
    for state in [0, 7] {
        let (h, _, _, _) = fixture();
        let mut s = Scheduler::new(h, 2);
        let mut p = plan(&bundle());
        p.request.priority = Priority::Demand;
        p.id.epoch = state;
        p.epochs.state = state;
        let id = s
            .enqueue_with_costs(
                p,
                PrefetchPolicy::Wait,
                StateKind::Active,
                PrefixCosts {
                    prompt_tokens: 1,
                    reused_tokens: 1,
                    cold_prefill_ns: 1,
                    suffix_prefill_ns: 2,
                    read_ns: 3,
                    copy_ns: 4,
                    materialize_ns: 5,
                },
                true,
            )
            .unwrap();
        assert_eq!(
            s.restore_decision(id).unwrap().0,
            RestoreDecision::RequireState
        );
        s.cancel(id).unwrap();
    }
}

/// B-owned fake for the frozen seam; no dependency on D's private implementation.
struct Capacity {
    gov: Shared,
    owners: [DeviceOwner; 2],
    routes: HashMap<(u32, u32), [bool; 3]>,
    released: HashMap<(u64, u64), bool>,
}
impl Capacity {
    fn new(gov: Shared) -> Self {
        Self {
            gov,
            owners: [DeviceOwner::new(0), DeviceOwner::new(1)],
            routes: [((0, 1), [true; 3]), ((1, 0), [true; 3])]
                .into_iter()
                .collect(),
            released: HashMap::new(),
        }
    }
}
impl PeerCapacity for Capacity {
    fn reserve(&mut self, plan: PeerPlan) -> Result<PeerLease> {
        if !self
            .routes
            .get(&(plan.owner_device, plan.consumer_device))
            .is_some_and(|flags| flags.iter().all(|x| *x))
        {
            return Err(Error::Unsupported);
        }
        memra_tier::peer::validate_peer_charge(&plan)?;
        if plan.epochs.state != epochs().state {
            return Err(Error::StaleEpoch);
        }
        let charge = self.gov.lock().unwrap().reserve(&plan.request)?;
        let device = match self.owners[plan.owner_device as usize].register(
            plan.epochs.dst_gen,
            plan.bytes,
            Box::new(NativeKvImage {
                bundle: bundle(),
                payloads: payloads(),
            }),
            &charge,
        ) {
            Ok(device) => device,
            Err(e) => {
                self.gov.lock().unwrap().release(&charge)?;
                return Err(e);
            }
        };
        self.released.insert(charge.id(), false);
        Ok(PeerLease {
            plan,
            charge,
            device,
        })
    }
    fn release(&mut self, lease: &PeerLease) -> Result<()> {
        let released = self
            .released
            .get_mut(&lease.charge.id())
            .ok_or(Error::ForeignLease)?;
        if !*released {
            self.owners[lease.device.device() as usize].release(&lease.device)?;
            *released = true;
        }
        self.gov.lock().unwrap().release(&lease.charge)?;
        self.released.remove(&lease.charge.id());
        Ok(())
    }
}

#[test]
fn peer_lookup_capacity_denial_downgrade_local_materialize_and_busy_retry() {
    let (mut h, g, _, _) = fixture();
    h.backing.as_mut().unwrap().route = Some(Tier::PeerGpu(1));
    let hit = h.lookup(&bundle().id, &[1], 0).unwrap().unwrap();
    assert_eq!(hit.tier, Tier::PeerGpu(1));
    let mut admission = plan(&bundle());
    admission.source = hit.tier;
    let mut capacity = Capacity::new(g.clone());
    // A stale advisory hit cannot authorize capacity after a grant or link changes.
    for control in 0..3 {
        capacity.routes.get_mut(&(1, 0)).unwrap()[control] = false;
        assert!(matches!(
            integration::reserve_peer_source(&mut capacity, &hit, &admission),
            Err(Error::Unsupported)
        ));
        assert_eq!(g.lock().unwrap().used(), TierBudget::zero(2));
        // Reverse direction still reserves and retires; grants are not one global bit.
        let mut reverse = admission.clone();
        reverse.source = Tier::PeerGpu(0);
        reverse.target_device = 1;
        let reverse_hit = Lookup {
            tier: reverse.source,
            storage_bytes: hit.storage_bytes,
        };
        let reverse_lease =
            integration::reserve_peer_source(&mut capacity, &reverse_hit, &reverse).unwrap();
        capacity.release(&reverse_lease).unwrap();
        capacity.routes.get_mut(&(1, 0)).unwrap()[control] = true;
    }
    let source = integration::reserve_peer_source(&mut capacity, &hit, &admission).unwrap();
    assert_eq!(source.device.device(), 1);
    assert_eq!(source.device.generation(), epochs().src_gen);
    let transfer_pin = capacity.owners[1].retain(&source.device).unwrap();
    assert_eq!(capacity.release(&source), Err(Error::Busy));
    assert_eq!(g.lock().unwrap().used().device, vec![0, 128]);
    assert_eq!(g.lock().unwrap().used().peer, vec![0, 128]);

    // Explicit CPU byte copy into a separately charged local destination. Native CUDA
    // copy submission/fences and D interoperability are intentionally not claimed here.
    let (copied_bundle, copied_payloads) = {
        let image = capacity.owners[1]
            .resolve::<NativeKvImage>(&source.device)
            .unwrap();
        (image.bundle.clone(), image.payloads.clone())
    };
    let mut local = DeviceOwner::new(0);
    let destination_charge = g.lock().unwrap().reserve(&admission.request).unwrap();
    let destination = local
        .register(
            31,
            128,
            Box::new(NativeKvImage {
                bundle: copied_bundle.clone(),
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
    let mut completion = complete(&local, &copied_bundle, ticket);
    completion.consumer_fenced = false;
    assert!(matches!(
        local.ready_view(
            &destination,
            &completion,
            &expected(&copied_bundle),
            epochs()
        ),
        Err(Error::NotReady)
    ));
    completion.consumer_fenced = true;
    let view = local
        .ready_view(
            &destination,
            &completion,
            &expected(&copied_bundle),
            epochs(),
        )
        .unwrap();
    let mut m = QwenMaterializer::new(
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
    let op = m
        .materialize(&copied_bundle, &program(), &view, epochs())
        .unwrap();
    assert_eq!(
        m.capture(&op).unwrap(),
        (payloads()[0][..34].to_vec(), payloads()[1][..24].to_vec())
    );
    assert_eq!(g.lock().unwrap().used().device, vec![128, 128]);
    assert_eq!(capacity.release(&source), Err(Error::Busy));
    m.retire(
        &op,
        FenceId {
            issuer: local.issuer(),
            owner: 0,
            generation: 31,
            sequence: 2,
        },
    )
    .unwrap();
    drop(op);
    drop(m);
    local.retire_binding(&ticket).unwrap();
    local.release(&destination).unwrap();
    g.lock().unwrap().release(&destination_charge).unwrap();
    drop(transfer_pin);
    capacity.release(&source).unwrap();
    assert_eq!(capacity.release(&source), Err(Error::ForeignLease));
    assert_eq!(g.lock().unwrap().used(), TierBudget::zero(2));
}

#[test]
fn program_residency_refuses_cross_tenant_before_charging_and_retains_full_identity() {
    let g = Arc::new(Mutex::new(governor(4096, TierBudget::zero(2))));
    let mut r = request(128, Priority::Backup);
    r.tenant[0] ^= 1;
    assert!(matches!(
        hostprefix::ResidentCharge::reserve_for_program(g.clone(), &program(), &r),
        Err(Error::ProgramMismatch)
    ));
    assert_eq!(g.lock().unwrap().used(), TierBudget::zero(2));
    r.tenant = program().tenant_salt;
    let guard = hostprefix::ResidentCharge::reserve_for_program(g.clone(), &program(), &r).unwrap();
    assert_eq!(guard.program(), Some(&program()));
    assert_eq!(g.lock().unwrap().used().pageable, 128);
    drop(guard);
    assert_eq!(g.lock().unwrap().used(), TierBudget::zero(2));
}

#[test]
fn fixture_load_winner_advances_frozen_plan_without_losing_decision() {
    let (mut h, g, c, _) = fixture();
    let b = bundle().seal().unwrap();
    h.backing.as_mut().unwrap().stored = b.clone();
    h.backing.as_mut().unwrap().route = Some(Tier::Nvme);
    let mut p = plan(&b);
    p.epochs.state = 0;
    p.request.priority = Priority::Demand;
    let current = p.epochs;
    let mut s = Scheduler::new(h, 2);
    let cost = PrefixCosts {
        prompt_tokens: 2,
        reused_tokens: 1,
        cold_prefill_ns: 1000,
        suffix_prefill_ns: 200,
        read_ns: 300,
        copy_ns: 100,
        materialize_ns: 100,
    };
    let id = s
        .enqueue_with_costs(
            p,
            PrefetchPolicy::Wait,
            StateKind::ImmutablePrefix,
            cost,
            true,
        )
        .unwrap();
    for now in 1..=3 {
        let events = s.tick(now, &[], |_| current).unwrap();
        assert!(events.iter().all(|(_, result)| result.is_ok()));
        let (decision, saved) = s.restore_decision(id).unwrap();
        assert_eq!(decision, RestoreDecision::Load);
        assert_eq!(saved.unwrap().materialize_ns, 100);
    }
    assert_eq!(
        s.consume(id, current, |b| b.bundle.id.clone()).unwrap(),
        b.id
    );
    let charged = g.lock().unwrap().used();
    assert_eq!(s.release(id), Err(Error::Busy));
    assert_eq!(g.lock().unwrap().used(), charged);
    c.borrow_mut().retired = true;
    s.release(id).unwrap();
    assert_eq!(g.lock().unwrap().used(), TierBudget::zero(2));
}
