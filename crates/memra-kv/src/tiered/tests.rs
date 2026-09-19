use super::*;
use crate::record::*;
use policy::*;
use std::{cell::RefCell, rc::Rc};

fn program() -> ProgramIdentity {
    ProgramIdentity {
        artifact: [1; 32],
        serialized_plan: [2; 32],
        numeric: [3; 32],
        stream: [4; 32],
        tokenizer: [5; 32],
        template: [6; 32],
        adapter: [7; 32],
        modality: [8; 32],
        position: [9; 32],
        tenant_salt: [10; 32],
    }
}
fn bundle() -> StateBundle {
    let p = program();
    let id = KvBlockId::new(&p, [0; 32], &[1, 2, 3], 0, 0, 0, 0).unwrap();
    let segments = vec![
        ByteSegment {
            group: 0,
            page: 0,
            owner: 0,
            role: 0,
            valid_bytes: 34,
            storage_bytes: 64,
            alignment: 64,
            encoding: digest(b"q8_0"),
        },
        ByteSegment {
            group: 0,
            page: 0,
            owner: 0,
            role: 1,
            valid_bytes: 24,
            storage_bytes: 64,
            alignment: 64,
            encoding: digest(b"q5_1"),
        },
    ];
    StateBundle {
        id,
        program: p,
        layout: RecordLayout {
            version: 1,
            page_count: 1,
            segments,
            requirements: vec![
                GroupRequirement {
                    group: 0,
                    owner: 0,
                    role: 0,
                    pages: PageRequirement::AllPages,
                },
                GroupRequirement {
                    group: 0,
                    owner: 0,
                    role: 1,
                    pages: PageRequirement::AllPages,
                },
            ],
        },
        committed_high_water: 3,
        owner_aliases: vec![(1, 0), (2, 0)],
        checksums: vec![digest(&[7; 34]), digest(&[8; 24])],
    }
}
fn budget(n: u64) -> TierBudget {
    TierBudget {
        devices: vec![n],
        pinned: n,
        staging: n,
        nvme: n,
        inflight: 1,
    }
}
fn admission(b: &StateBundle) -> TierAdmission {
    TierAdmission {
        id: b.id.clone(),
        program: b.program.clone(),
        expected_layout: b.layout.clone(),
        source: Tier::Nvme,
        target_device: 0,
        charge: budget(128),
        mandatory: true,
        active: None,
    }
}
#[derive(Clone)]
struct FakeStore {
    bundle: StateBundle,
    available: Rc<RefCell<bool>>,
    releases: Rc<RefCell<usize>>,
}
impl ObjectStore for FakeStore {
    fn lookup(&self, _: &KvBlockId) -> Vec<Lookup> {
        if !*self.available.borrow() {
            return vec![];
        }
        vec![
            Tier::Nvme,
            Tier::PinnedHost,
            Tier::PeerGpu(1),
            Tier::LocalGpu(0),
        ]
        .into_iter()
        .map(|tier| Lookup {
            tier,
            storage_bytes: 128,
        })
        .collect()
    }
    fn lease(&mut self, _: &KvBlockId, tier: Tier) -> Result<BlockLease, TierError> {
        if !*self.available.borrow() {
            return Err(TierError::Missing);
        }
        Ok(BlockLease {
            handle: 1,
            bundle: self.bundle.clone(),
            tier,
        })
    }
    fn evict(&mut self, _: &KvBlockId) -> Result<(), TierError> {
        *self.available.borrow_mut() = false;
        Ok(())
    }
    fn release(&mut self, _: BlockLease) {
        *self.releases.borrow_mut() += 1;
    }
}
#[derive(Clone)]
struct FakeTransfer {
    result: Rc<RefCell<Completion>>,
    retired: Rc<RefCell<bool>>,
    cancels: Rc<RefCell<usize>>,
    next: u64,
}
impl TransferEngine for FakeTransfer {
    fn prefetch(&mut self, _: &BlockLease) -> Result<TransferTicket, TierError> {
        self.next += 1;
        Ok(TransferTicket(self.next))
    }
    fn load(&mut self, _: &BlockLease, _: u32) -> Result<TransferTicket, TierError> {
        self.next += 1;
        Ok(TransferTicket(self.next))
    }
    fn poll(&mut self, _: TransferTicket) -> Completion {
        self.result.borrow().clone()
    }
    fn cancel(&mut self, _: TransferTicket) {
        *self.cancels.borrow_mut() += 1;
    }
    fn retired(&mut self, _: TransferTicket) -> bool {
        *self.retired.borrow()
    }
}
fn fixture() -> (
    Hierarchy<FakeStore, FakeTransfer>,
    FakeStore,
    FakeTransfer,
    StateBundle,
) {
    let b = bundle();
    let s = FakeStore {
        bundle: b.clone(),
        available: Rc::new(RefCell::new(true)),
        releases: Rc::new(RefCell::new(0)),
    };
    let t = FakeTransfer {
        result: Rc::new(RefCell::new(Completion {
            items: b
                .layout
                .segments
                .iter()
                .enumerate()
                .map(|(i, s)| ItemCompletion {
                    segment: i,
                    epoch: b.id.epoch,
                    status: ItemStatus::Complete,
                    bytes: s.storage_bytes,
                    checksum: b.checksums[i],
                })
                .collect(),
            producer_done: true,
            consumer_fenced: false,
        })),
        retired: Rc::new(RefCell::new(false)),
        cancels: Rc::new(RefCell::new(0)),
        next: 0,
    };
    let h = Hierarchy::new(
        s.clone(),
        t.clone(),
        budget(256),
        TierBudget {
            devices: vec![0],
            ..TierBudget::default()
        },
    )
    .unwrap();
    (h, s, t, b)
}
#[test]
fn identity_is_deterministic_and_every_program_field_matters() {
    let p = program();
    let id = KvBlockId::new(&p, [0; 32], &[1, 2, 3], 0, 0, 0, 0).unwrap();
    assert_eq!(id, bundle().id);
    let mut variants = vec![p.clone(); 10];
    variants[0].artifact[0] ^= 1;
    variants[1].serialized_plan[0] ^= 1;
    variants[2].numeric[0] ^= 1;
    variants[3].stream[0] ^= 1;
    variants[4].tokenizer[0] ^= 1;
    variants[5].template[0] ^= 1;
    variants[6].adapter[0] ^= 1;
    variants[7].modality[0] ^= 1;
    variants[8].position[0] ^= 1;
    variants[9].tenant_salt[0] ^= 1;
    for v in variants {
        assert_ne!(v.namespace(), p.namespace());
    }
    let mut changed = id.clone();
    changed.parent[0] = 1;
    assert_ne!(changed.hash(), id.hash());
    changed = id.clone();
    changed.epoch += 1;
    assert_ne!(changed.hash(), id.hash());
    let child = KvBlockId::new(&p, id.hash(), &[4], 3, 0, 0, 0).unwrap();
    assert_eq!(child.parent, id.hash());
    assert_ne!(
        id.tokens,
        KvBlockId::new(&p, [0; 32], &[1, 2, 4], 0, 0, 0, 0)
            .unwrap()
            .tokens
    );
    assert!(KvBlockId::new(&p, [0; 32], &[], 0, 0, 0, 0).is_err());
    assert_eq!(
        KvBlockId::new(&p, [0; 32], &[1], u64::MAX, 0, 0, 0),
        Err(TierError::Overflow)
    );
    assert_eq!(
        digest(b"abc"),
        [
            0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea, 0x41, 0x41, 0x40, 0xde, 0x5d, 0xae,
            0x22, 0x23, 0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c, 0xb4, 0x10, 0xff, 0x61,
            0xf2, 0x00, 0x15, 0xad
        ]
    );
}
#[test]
fn native_records_validate_valid_bytes_not_padding() {
    let b = bundle();
    let mut k = vec![7; 64];
    let mut v = vec![8; 64];
    k[34..].fill(0xee);
    v[24..].fill(0xdd);
    b.verify(&[k.clone(), v.clone()]).unwrap();
    k[1] ^= 1;
    assert_eq!(b.verify(&[k, v]), Err(TierError::Corrupt));
    assert_eq!(
        b.verify(&[vec![7; 34], vec![8; 64]]),
        Err(TierError::ShortRead)
    );
    assert_eq!(b.verify(&[]), Err(TierError::Incomplete));
}
#[test]
fn opaque_f32_fp8_fp4_and_trailing_requirements() {
    for (encoding, valid) in [
        (b"opaque-f32".as_slice(), 584),
        (b"opaque-fp8".as_slice(), 132),
        (b"opaque-fp4".as_slice(), 288),
    ] {
        let mut b = bundle();
        b.layout.page_count = 3;
        b.layout.segments.clear();
        b.layout.requirements = vec![
            GroupRequirement {
                group: 0,
                owner: 0,
                role: 0,
                pages: PageRequirement::AllPages,
            },
            GroupRequirement {
                group: 0,
                owner: 0,
                role: 1,
                pages: PageRequirement::TrailingPages(1),
            },
        ];
        for page in 0..3 {
            b.layout.segments.push(ByteSegment {
                group: 0,
                page,
                owner: 0,
                role: 0,
                valid_bytes: valid,
                storage_bytes: 4096,
                alignment: 4096,
                encoding: digest(encoding),
            });
        }
        let mut tail = b.layout.segments[2].clone();
        tail.role = 1;
        b.layout.segments.push(tail);
        b.checksums = vec![digest(&vec![9; valid as usize]); 4];
        b.verify(&vec![vec![9; 4096]; 4]).unwrap();
        assert_eq!(b.layout.storage_bytes().unwrap(), 16384); // owner aliases add ZERO; no TP division
        b.layout.segments.pop();
        b.checksums.pop();
        assert_eq!(b.validate(), Err(TierError::Incomplete));
    }
}
#[test]
fn malformed_layouts_and_missing_state_fail_closed() {
    let mut b = bundle();
    b.layout.segments[0].alignment = 3;
    assert!(b.validate().is_err());
    b = bundle();
    b.layout.segments[0].valid_bytes = 65;
    assert!(b.validate().is_err());
    b = bundle();
    b.layout.segments[1] = b.layout.segments[0].clone();
    assert!(b.validate().is_err());
    b = bundle();
    b.layout.requirements.clear();
    assert!(b.validate().is_err());
    b = bundle();
    b.layout.page_count = u64::MAX;
    assert_eq!(b.validate(), Err(TierError::Incomplete));
    b = bundle();
    b.owner_aliases.push((1, 0));
    assert!(b.validate().is_err());
    b = bundle();
    b.owner_aliases.push((3, 9));
    assert!(b.validate().is_err());
    b = bundle();
    b.committed_high_water = 2;
    assert!(b.validate().is_err());
}
#[test]
fn search_order_filters_ineligible_peers_and_lookup_is_advisory() {
    let (mut h, s, _, b) = fixture();
    assert_eq!(h.lookup(&b.id, &[1], 0).unwrap().tier, Tier::LocalGpu(0));
    assert_eq!(h.lookup(&b.id, &[1], 3).unwrap().tier, Tier::PeerGpu(1));
    assert_eq!(h.lookup(&b.id, &[], 3).unwrap().tier, Tier::PinnedHost);
    *s.available.borrow_mut() = false;
    assert_eq!(h.admit(admission(&b)), Err(TierError::Missing));
    assert_eq!(h.used().devices, vec![0]);
}
#[test]
fn disk_ready_is_not_gpu_ready_and_fence_required() {
    let (mut h, s, t, b) = fixture();
    let r = h.admit(admission(&b)).unwrap();
    assert_eq!(h.load(&r), Err(TierError::WrongState));
    h.prefetch(&r).unwrap();
    assert_eq!(h.advance(&r).unwrap(), Phase::HostReady);
    assert_eq!(
        h.with_ready(&r, &b.program, |_, _, _| ()),
        Err(TierError::WrongState)
    );
    h.load(&r).unwrap();
    assert_eq!(h.advance(&r).unwrap(), Phase::Loading);
    t.result.borrow_mut().consumer_fenced = true;
    assert_eq!(h.advance(&r).unwrap(), Phase::Ready);
    h.with_ready(&r, &b.program, |state, ready, _| {
        assert_eq!(ready.id(), &state.id);
        assert_eq!(ready.device(), 0)
    })
    .unwrap();
    assert_eq!(h.retire(&r), Err(TierError::Busy));
    h.cancel(&r).unwrap();
    assert!(!h.retire(&r).unwrap());
    assert_eq!(*s.releases.borrow(), 0);
    *t.retired.borrow_mut() = true;
    assert!(h.retire(&r).unwrap());
    assert_eq!(*s.releases.borrow(), 1);
    assert_eq!(h.used().devices, vec![0]);
}
#[test]
fn every_partial_short_reject_duplicate_corrupt_and_stale_item_refuses_batch() {
    for fault in 0..7 {
        let (mut h, _, t, b) = fixture();
        let r = h.admit(admission(&b)).unwrap();
        h.prefetch(&r).unwrap();
        {
            let mut c = t.result.borrow_mut();
            match fault {
                0 => {
                    c.items.pop();
                }
                1 => c.items[1].status = ItemStatus::Rejected,
                2 => c.items[1].bytes -= 1,
                3 => c.items[1].epoch += 1,
                4 => c.items[1].segment = 0,
                5 => c.items[1].checksum[0] ^= 1,
                _ => c.items[1].status = ItemStatus::Failed,
            }
        }
        assert!(h.advance(&r).is_err(), "fault {fault}");
        assert_eq!(h.phase(&r).unwrap(), Phase::Failed);
        assert!(!h.retire(&r).unwrap());
        assert_eq!(*t.cancels.borrow(), 1);
    }
}
#[test]
fn pending_and_unfinished_io_cannot_publish() {
    let (mut h, _, t, b) = fixture();
    let r = h.admit(admission(&b)).unwrap();
    h.prefetch(&r).unwrap();
    t.result.borrow_mut().items[1].status = ItemStatus::Pending;
    assert_eq!(h.advance(&r).unwrap(), Phase::Prefetching);
    t.result.borrow_mut().items[1].status = ItemStatus::Complete;
    t.result.borrow_mut().producer_done = false;
    assert_eq!(h.advance(&r).unwrap(), Phase::Prefetching);
}
#[test]
fn cancel_suppresses_late_publication_and_holds_budget_until_all_fences() {
    let (mut h, s, t, b) = fixture();
    let r = h.admit(admission(&b)).unwrap();
    h.prefetch(&r).unwrap();
    h.cancel(&r).unwrap();
    t.result.borrow_mut().consumer_fenced = true;
    assert_eq!(h.advance(&r).unwrap(), Phase::Cancelled);
    assert!(!h.retire(&r).unwrap());
    assert_eq!(h.used().devices, vec![128]);
    assert_eq!(*s.releases.borrow(), 0);
    assert_eq!(
        h.with_ready(&r, &b.program, |_, _, _| ()),
        Err(TierError::WrongState)
    );
    *t.retired.borrow_mut() = true;
    assert!(h.retire(&r).unwrap());
    assert_eq!(h.used().devices, vec![0]);
    assert_eq!(h.retire(&r), Err(TierError::Missing));
}
#[test]
fn mismatch_and_forged_identity_do_not_get_leased_or_restored() {
    let (mut h, s, t, b) = fixture();
    let mut a = admission(&b);
    a.program.numeric[0] ^= 1;
    assert_eq!(h.admit(a), Err(TierError::ProgramMismatch));
    assert_eq!(*s.releases.borrow(), 0);
    let mut a = admission(&b);
    a.id.epoch += 1;
    assert_eq!(h.admit(a), Err(TierError::StaleEpoch));
    assert_eq!(*s.releases.borrow(), 1);
    let r = h.admit(admission(&b)).unwrap();
    h.prefetch(&r).unwrap();
    h.advance(&r).unwrap();
    h.load(&r).unwrap();
    t.result.borrow_mut().consumer_fenced = true;
    h.advance(&r).unwrap();
    let mut p = program();
    p.stream[0] ^= 1;
    assert_eq!(
        h.with_ready(&r, &p, |_, _, _| ()),
        Err(TierError::ProgramMismatch)
    );
}
#[test]
fn mandatory_headroom_and_whole_operand_capacity_are_charged() {
    let (_, s, t, b) = fixture();
    let mut cap = budget(256);
    cap.inflight = 3;
    let headroom = TierBudget {
        devices: vec![128],
        ..TierBudget::default()
    };
    let mut h = Hierarchy::new(s, t, cap, headroom).unwrap();
    let mut a = admission(&b);
    a.mandatory = false;
    let r = h.admit(a.clone()).unwrap();
    assert_eq!(h.admit(a), Err(TierError::Capacity));
    h.admit(admission(&b)).unwrap();
    assert_eq!(h.used().devices, vec![256]);
    h.cancel(&r).unwrap();
    assert!(h.retire(&r).unwrap());
    let mut a = admission(&b);
    a.charge.devices[0] = 64;
    assert_eq!(h.admit(a), Err(TierError::Capacity));
}
#[test]
fn active_epochs_rollback_and_committed_high_water_reject_stale() {
    let mut a = ActiveEpoch::new(3);
    let old = bundle();
    assert!(a.accepts(&old));
    let epoch = a.fork(8).unwrap();
    assert!(!a.accepts(&old));
    assert_eq!(a.commit(0, 5), Err(TierError::StaleEpoch));
    assert_eq!(a.commit(epoch, 9), Err(TierError::Incomplete));
    a.commit(epoch, 5).unwrap();
    let mut new = old.clone();
    new.id.epoch = epoch;
    new.id.end = 5;
    new.committed_high_water = 5;
    assert!(a.accepts(&new));
    a.rollback(3).unwrap();
    assert!(!a.accepts(&new));
    assert_eq!(old.id.epoch, 0);
    assert_eq!(old.committed_high_water, 3); // old immutable prefix preserved
}
#[test]
fn write_prefetch_and_eviction_policies() {
    assert_eq!(
        write_action(WritePolicy::Through, true, false, 0, false),
        WriteAction::BackupBeforePublish
    );
    assert_eq!(
        write_action(WritePolicy::Back, true, false, 0, true),
        WriteAction::BackupBeforeEvict
    );
    assert_eq!(
        write_action(WritePolicy::Back, true, false, 0, false),
        WriteAction::RetainDirty
    );
    assert_eq!(
        write_action(
            WritePolicy::Selective { reuse_threshold: 2 },
            true,
            false,
            1,
            false
        ),
        WriteAction::RetainDirty
    );
    assert_eq!(
        write_action(
            WritePolicy::Selective { reuse_threshold: 2 },
            true,
            false,
            2,
            false
        ),
        WriteAction::BackupBeforePublish
    );
    assert_eq!(
        write_action(WritePolicy::Through, false, false, 9, true),
        WriteAction::Nothing
    );
    for mandatory in [false, true] {
        assert_eq!(
            prefetch_decision(PrefetchPolicy::Wait, false, true, 99, mandatory, true),
            PrefetchDecision::Wait
        );
        assert_eq!(
            prefetch_decision(
                PrefetchPolicy::Timeout { deadline_ms: 100 },
                false,
                true,
                99,
                mandatory,
                true
            ),
            PrefetchDecision::Wait
        );
        let stop = if mandatory {
            PrefetchDecision::Refuse
        } else {
            PrefetchDecision::SameProgramCold
        };
        assert_eq!(
            prefetch_decision(
                PrefetchPolicy::Timeout { deadline_ms: 100 },
                false,
                true,
                100,
                mandatory,
                true
            ),
            stop
        );
        assert_eq!(
            prefetch_decision(PrefetchPolicy::BestEffort, false, true, 0, mandatory, true),
            stop
        );
    }
    assert_eq!(
        prefetch_decision(PrefetchPolicy::BestEffort, false, true, 0, false, false),
        PrefetchDecision::Refuse
    );
    let base = EvictionCandidate {
        id: 1,
        last_use: 1,
        protected: true,
        leased: false,
        inflight: false,
        dirty: false,
        has_backing: true,
        mandatory: true,
    };
    let second = EvictionCandidate {
        id: 2,
        last_use: 2,
        protected: false,
        ..base
    };
    assert_eq!(
        eviction_victim(&[base, second], EvictionPolicy::Lru),
        Some(1)
    );
    assert_eq!(
        eviction_victim(&[base, second], EvictionPolicy::Slru),
        Some(2)
    );
    for c in [
        EvictionCandidate {
            leased: true,
            ..base
        },
        EvictionCandidate {
            inflight: true,
            ..base
        },
        EvictionCandidate {
            dirty: true,
            ..base
        },
        EvictionCandidate {
            has_backing: false,
            ..base
        },
    ] {
        assert_eq!(eviction_victim(&[c], EvictionPolicy::Lru), None);
    }
}
#[test]
fn measured_frontier_and_invalid_calibration() {
    assert_eq!(
        recompute_vs_load(1000, 1_000_000_000, 1.0, 1000.0, 0.0, false, true).unwrap(),
        RestoreDecision::Recompute
    );
    assert_eq!(
        recompute_vs_load(2000, 1_000_000_000, 1.0, 1000.0, 0.0, false, true).unwrap(),
        RestoreDecision::Load
    );
    assert_eq!(
        recompute_vs_load(2000, 1_000_000_000, 1.0, 1000.0, 2.0, false, true).unwrap(),
        RestoreDecision::Recompute
    );
    for speed in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(recompute_vs_load(1, 1, speed, 1.0, 0.0, false, true).is_err());
    }
    assert_eq!(
        recompute_vs_load(0, 0, 0.0, 0.0, 0.0, true, false).unwrap(),
        RestoreDecision::RequireState
    );
    assert_eq!(
        recompute_vs_load(0, 0, 1.0, 1.0, 0.0, false, false).unwrap(),
        RestoreDecision::Load
    );
}

#[test]
fn rollback_during_io_cancels_publication_and_evict_waits_for_fences() {
    let (mut h, s, t, b) = fixture();
    let active = Arc::new(Mutex::new(ActiveEpoch::new(3)));
    let mut a = admission(&b);
    a.active = Some(active.clone());
    let r = h.admit(a).unwrap();
    h.prefetch(&r).unwrap();
    assert_eq!(h.evict(&b.id), Err(TierError::Busy));
    active.lock().unwrap().rollback(2).unwrap();
    assert_eq!(h.advance(&r), Err(TierError::StaleEpoch));
    assert_eq!(h.phase(&r).unwrap(), Phase::Cancelled);
    assert!(!h.retire(&r).unwrap());
    assert_eq!(h.evict(&b.id), Err(TierError::Busy));
    *t.retired.borrow_mut() = true;
    assert!(h.retire(&r).unwrap());
    h.evict(&b.id).unwrap();
    assert!(!*s.available.borrow());
}

#[test]
fn stored_layout_substitution_is_refused_against_admitted_layout() {
    let (mut h, s, _, b) = fixture();
    let mut a = admission(&b);
    a.expected_layout.segments[0].encoding = digest(b"not-q8_0");
    assert_eq!(h.admit(a), Err(TierError::InvalidLayout));
    assert_eq!(*s.releases.borrow(), 1);
    assert_eq!(h.used().devices, vec![0]);
}

#[test]
fn shutdown_unknown_fence_quarantines_instead_of_releasing() {
    let (mut h, s, _, b) = fixture();
    let r = h.admit(admission(&b)).unwrap();
    h.prefetch(&r).unwrap();
    drop(h);
    assert_eq!(*s.releases.borrow(), 0);
    let (mut h, s, t, b) = fixture();
    let r = h.admit(admission(&b)).unwrap();
    h.prefetch(&r).unwrap();
    *t.retired.borrow_mut() = true;
    drop(h);
    assert_eq!(*s.releases.borrow(), 1);
}
