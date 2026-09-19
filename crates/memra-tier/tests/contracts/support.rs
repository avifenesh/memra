use memra_tier::contracts::*;
use std::{cell::RefCell, rc::Rc};
pub type Shared = Rc<RefCell<Governor>>;
pub fn bytes(n: usize) -> Vec<u8> {
    (0..n).map(|i| ((17 * i + 3) % 251) as u8).collect()
}
pub fn epochs() -> Epochs {
    Epochs {
        state: 7,
        src_gen: 19,
        dst_gen: 31,
    }
}
pub fn program() -> ProgramIdentity {
    ProgramIdentity {
        version: 1,
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
pub fn key() -> ObjectKey {
    ObjectKey {
        version: 1,
        artifact: [1; 32],
        semantic_id: [2; 32],
        layout: [3; 32],
        generation: 7,
    }
}
pub fn layout() -> RecordLayout {
    RecordLayout {
        version: 1,
        segments: vec![ByteSegment {
            version: 1,
            group: 0,
            page: 0,
            owner: 0,
            role: Role::Payload,
            tensor: None,
            offset: 0,
            valid_bytes: 3,
            storage_bytes: 4,
            alignment: 4,
            encoding: EncodingId {
                version: 1,
                program: [42; 32],
                row_bytes: 3,
            },
        }],
        requirements: vec![GroupRequirement {
            version: 1,
            group: 0,
            owner: 0,
            role: Role::Payload,
            page_count: 1,
            pages: PageRequirement::AllPages,
        }],
    }
}
pub fn bundle() -> StateBundle {
    let program = program();
    StateBundle {
        version: 1,
        id: KvBlockId::new(&program, [0; 32], &[1, 2, 3], 0, 0, 0, 7).unwrap(),
        program,
        layout: layout(),
        kind: StateKind::Active,
        committed_high_water: 3,
        owner_aliases: vec![],
        checksums: vec![checksum(&bytes(3))],
    }
}
pub fn request(pageable: u64, priority: Priority) -> BudgetRequest {
    let mut b = TierBudget::zero(2);
    b.pageable = pageable;
    BudgetRequest {
        bytes: b,
        priority,
        deadline: Deadline(100),
        tenant: [1; 32],
    }
}
pub struct Governor {
    pub issuer: LeaseIssuer,
    pub used: TierBudget,
    pub cap: TierBudget,
    pub headroom: TierBudget,
}
impl Governor {
    pub fn new(cap: u64) -> Self {
        let mut b = TierBudget::zero(2);
        b.pageable = cap;
        b.pinned = cap;
        b.staging = cap;
        b.loaders = cap;
        b.device = vec![cap; 2];
        b.peer = vec![cap; 2];
        b.replicas = vec![cap; 2];
        b.nvme = cap;
        b.inflight = cap;
        let mut headroom = TierBudget::zero(2);
        headroom.pageable = cap / 4;
        Self {
            issuer: LeaseIssuer::default(),
            used: TierBudget::zero(2),
            cap: b,
            headroom,
        }
    }
}
impl BudgetGovernor for Governor {
    fn reserve(&mut self, r: &BudgetRequest) -> Result<ChargedLease> {
        r.validate()?;
        if r.deadline.0 == 0 {
            return Err(Error::Deadline);
        }
        let cap = if r.priority == Priority::MandatoryActive {
            self.cap.clone()
        } else {
            self.cap.checked_sub(&self.headroom)?
        };
        if !r.bytes.fits(&self.used, &cap)? {
            return Err(Error::Capacity);
        }
        let next = self.used.checked_add(&r.bytes)?;
        let lease = self.issuer.issue(r.bytes.clone())?;
        self.used = next;
        Ok(lease)
    }
    fn used(&self) -> TierBudget {
        self.used.clone()
    }
    fn mark(&mut self, l: &ChargedLease, s: ChargeState) -> Result<()> {
        self.issuer.mark(l, s)
    }
    fn release(&mut self, l: &ChargedLease) -> Result<()> {
        self.issuer.release(l)?;
        self.used = self.used.checked_sub(l.bytes())?;
        Ok(())
    }
}
pub fn shared() -> Shared {
    Rc::new(RefCell::new(Governor::new(1024)))
}
pub fn expected(n: usize) -> Vec<Vec<SegmentExpectation>> {
    (0..n)
        .map(|_| {
            vec![SegmentExpectation {
                valid_bytes: 3,
                io_bytes: 4,
                checksum: checksum(&bytes(3)),
            }]
        })
        .collect()
}
pub fn completion(t: TransferTicket, n: usize) -> Completion {
    Completion {
        ticket: t,
        producer_done: true,
        consumer_fenced: true,
        items: (0..n)
            .map(|i| ItemOutcome {
                item: i as u32,
                accepted: true,
                segments: vec![SegmentCompletion {
                    segment: 0,
                    status: ItemStatus::Complete,
                    valid_bytes: 3,
                    io_bytes: 4,
                    checksum: Some(checksum(&bytes(3))),
                    epochs: t.epochs,
                    producer_done: true,
                    consumer_fenced: true,
                    consumer_fence: Some(FenceId {
                        issuer: 1,
                        owner: 0,
                        generation: t.epochs.dst_gen,
                        sequence: 1,
                    }),
                    error: None,
                }],
            })
            .collect(),
    }
}
