use super::*;

#[derive(Debug)]
struct Host(Vec<u8>);
impl PinnedLease for Host {
    fn storage_bytes(&self) -> u64 {
        self.0.len() as u64
    }
    fn valid_bytes(&self) -> u64 {
        self.storage_bytes()
    }
    fn alignment(&self) -> u32 {
        4
    }
    fn numa_node(&self) -> Option<u32> {
        None
    }
    fn bytes(&self) -> Result<&[u8]> {
        Ok(&self.0)
    }
}
// CPU capability/lifetime fake: no pinned host memory or CUDA is implied.
struct Fake {
    owner: DeviceOwner,
    op: Option<CopyOp<Host>>,
    completion: Option<Completion>,
    expected: Vec<Vec<SegmentExpectation>>,
    published: bool,
    cancelled: bool,
    done: bool,
    reject: bool,
}
impl Fake {
    fn new() -> Self {
        Self {
            owner: DeviceOwner::new(0),
            op: None,
            completion: None,
            expected: vec![],
            published: false,
            cancelled: false,
            done: false,
            reject: false,
        }
    }
    fn op(&mut self, bytes: Vec<u8>, charge: &ChargedLease) -> CopyOp<Host> {
        let device = self
            .owner
            .register(
                epochs().dst_gen,
                bytes.len() as u64,
                Box::new(vec![0u8; bytes.len()]),
                charge,
            )
            .unwrap();
        CopyOp {
            bytes: bytes.len() as u64,
            host: Host(bytes),
            device,
            epochs: epochs(),
            producer_fence: None,
        }
    }
    fn ready(&mut self) {
        let c = self.completion.as_mut().unwrap();
        c.consumer_fenced = true;
        c.items[0].segments[0].consumer_fenced = true;
        c.items[0].segments[0].consumer_fence = Some(FenceId {
            issuer: self.owner.issuer(),
            owner: 0,
            generation: epochs().dst_gen,
            sequence: 1,
        });
    }
}
impl TransferEngine for Fake {
    type Host = Host;
    fn h2d(
        &mut self,
        op: CopyOp<Host>,
    ) -> std::result::Result<TransferTicket, Rejected<CopyOp<Host>>> {
        if self.reject {
            return Err(Rejected {
                op,
                error: Error::Capacity,
            });
        }
        let t = TransferTicket {
            issuer: self.owner.issuer(),
            sequence: 1,
            epochs: op.epochs,
        };
        self.owner.bind_destination(t, &op.device).unwrap();
        self.expected = vec![vec![SegmentExpectation {
            valid_bytes: op.bytes,
            io_bytes: op.bytes,
            checksum: checksum(op.host.bytes().unwrap()),
        }]];
        self.completion = Some(Completion {
            ticket: t,
            producer_done: true,
            consumer_fenced: false,
            items: vec![ItemOutcome {
                item: 0,
                accepted: true,
                segments: vec![SegmentCompletion {
                    segment: 0,
                    status: ItemStatus::Complete,
                    valid_bytes: op.bytes,
                    io_bytes: op.bytes,
                    checksum: Some(self.expected[0][0].checksum),
                    epochs: op.epochs,
                    producer_done: true,
                    consumer_fenced: false,
                    consumer_fence: None,
                    error: None,
                }],
            }],
        });
        self.op = Some(op);
        Ok(t)
    }
    fn d2h(
        &mut self,
        op: CopyOp<Host>,
    ) -> std::result::Result<TransferTicket, Rejected<CopyOp<Host>>> {
        Err(Rejected {
            op,
            error: Error::Unsupported,
        })
    }
    fn p2p(
        &mut self,
        op: ContiguousCopy,
    ) -> std::result::Result<TransferTicket, Rejected<ContiguousCopy>> {
        Err(Rejected {
            op,
            error: Error::Unsupported,
        })
    }
    fn nvme_read(
        &mut self,
        op: ReadPlan<Host>,
    ) -> std::result::Result<TransferTicket, Rejected<ReadPlan<Host>>> {
        Err(Rejected {
            op,
            error: Error::Unsupported,
        })
    }
    fn submit_batch(&mut self, op: Vec<TransferOp<Host>>) -> Submission<TransferOp<Host>> {
        Err(Rejected {
            op,
            error: Error::Unsupported,
        })
    }
    fn poll(&mut self, _: &TransferTicket) -> Result<Completion> {
        self.completion.clone().ok_or(Error::UnknownTicket)
    }
    fn cancel(&mut self, _: &TransferTicket) -> Result<CancelState> {
        if self.published {
            Ok(CancelState::AlreadyPublished)
        } else {
            self.cancelled = true;
            Ok(CancelState::PublicationRevoked)
        }
    }
    fn ready_view(&mut self, _: &TransferTicket, _: u32, current: Epochs) -> Result<ReadyView<'_>> {
        if self.cancelled {
            return Err(Error::Cancelled);
        }
        let view = self.owner.ready_view(
            &self.op.as_ref().unwrap().device,
            self.completion.as_ref().unwrap(),
            &self.expected,
            current,
        )?;
        self.published = true;
        Ok(view)
    }
    fn take_destination(
        &mut self,
        t: &TransferTicket,
        i: u32,
        current: Epochs,
    ) -> Result<Destination<Host>> {
        self.ready_view(t, i, current)?;
        Ok(Destination::Device(
            self.owner.retain(&self.op.as_ref().unwrap().device)?,
        ))
    }
    fn retire(&mut self, _: &TransferTicket, _: Option<FenceId>) -> Result<()> {
        Ok(())
    }
    fn retired(&mut self, _: &TransferTicket) -> Result<bool> {
        Ok(self.done)
    }
    fn acknowledge(&mut self, t: &TransferTicket) -> Result<()> {
        if !self.done {
            return Err(Error::Busy);
        }
        self.owner.retire_binding(t)?;
        let op = self.op.take().unwrap();
        self.owner.release(&op.device)?;
        self.completion = None;
        Ok(())
    }
}
fn charge() -> (Rc<RefCell<Governor>>, ChargedLease) {
    let g = Rc::new(RefCell::new(Governor::new(1 << 20)));
    let mut req = request(0, Priority::Demand);
    req.bytes.device[0] = 1024;
    let c = g.borrow_mut().reserve(&req).unwrap();
    (g, c)
}
#[test]
fn row_device_publication_requires_wait_epochs_integrity_and_retirement() {
    let (gov, charge) = charge();
    let mut f = Fake::new();
    let op = f.op(vec![1, 2, 3, 4, 5, 6, 7, 8], &charge);
    let mut upload = RowUpload::submit(&mut f, op).unwrap();
    assert!(matches!(
        upload.publish(&mut f, epochs()),
        Err(Error::NotReady)
    ));
    assert!(!f.published);
    f.ready();
    let mut stale = epochs();
    stale.state += 1;
    assert!(matches!(
        upload.publish(&mut f, stale),
        Err(Error::StaleEpoch)
    ));
    let c = f.completion.clone().unwrap();
    f.completion.as_mut().unwrap().items[0].segments[0].valid_bytes = 4;
    assert!(matches!(
        upload.publish(&mut f, epochs()),
        Err(Error::ShortIo { .. })
    ));
    f.completion = Some(c.clone());
    f.completion.as_mut().unwrap().items[0].segments[0].checksum = Some([0; 32]);
    assert!(matches!(
        upload.publish(&mut f, epochs()),
        Err(Error::Corrupt)
    ));
    f.completion = Some(c.clone());
    f.completion.as_mut().unwrap().items[0].segments[0]
        .consumer_fence
        .as_mut()
        .unwrap()
        .issuer += 1;
    assert!(matches!(
        upload.publish(&mut f, epochs()),
        Err(Error::WrongOwner)
    ));
    f.completion = Some(c);
    let destination = upload.publish(&mut f, epochs()).unwrap();
    assert!(matches!(upload.publish(&mut f, epochs()), Err(Error::Busy)));
    assert_eq!(
        upload.cancel(&mut f).unwrap(),
        CancelState::AlreadyPublished
    );
    assert!(!upload.retire(&mut f, None).unwrap());
    assert_eq!(upload.acknowledge(&mut f), Err(Error::Busy));
    assert_eq!(gov.borrow_mut().release(&charge), Err(Error::Busy));
    drop(destination);
    f.done = true;
    assert!(upload.retire(&mut f, None).unwrap());
    upload.acknowledge(&mut f).unwrap();
    gov.borrow_mut().release(&charge).unwrap();
}
#[test]
fn row_device_rejection_returns_inputs_and_cancel_never_publishes() {
    let (gov, charge) = charge();
    let mut f = Fake::new();
    f.reject = true;
    let op = f.op(vec![0; 8], &charge);
    let rejected = RowUpload::submit(&mut f, op).unwrap_err();
    assert_eq!(rejected.error, Error::Capacity);
    f.owner.release(&rejected.op.device).unwrap();
    drop(rejected);
    f.reject = false;
    let op = f.op(vec![0; 8], &charge);
    let mut upload = RowUpload::submit(&mut f, op).unwrap();
    f.ready();
    assert_eq!(
        upload.cancel(&mut f).unwrap(),
        CancelState::PublicationRevoked
    );
    assert!(matches!(
        upload.publish(&mut f, epochs()),
        Err(Error::Cancelled)
    ));
    f.done = true;
    upload.acknowledge(&mut f).unwrap();
    gov.borrow_mut().release(&charge).unwrap();
}

#[test]
fn row_device_native_gather_submits_exact_coalesced_bits() {
    use super::ple_rows_native::native::{HostTable, PleRowsTier};
    let (gov, charge) = charge();
    let tier = PleRowsTier::new(gov.clone());
    let source = [
        f32::from_bits(0x80000000),
        f32::from_bits(0x7fc01234),
        1.,
        -2.,
    ];
    let rows = tier
        .gather(HostTable::F32(&source), 0, 2, &[1, 0, 1])
        .unwrap();
    let raw: Vec<_> = rows
        .iter()
        .flat_map(|f| f.to_bits().to_le_bytes())
        .collect();
    let mut f = Fake::new();
    let mut bad = raw.clone();
    bad[0] ^= 1;
    let op = f.op(bad, &charge);
    let rejected = rows.submit_device(&mut f, op).unwrap_err();
    assert_eq!(rejected.error, Error::Corrupt);
    assert!(f.op.is_none());
    f.owner.release(&rejected.op.device).unwrap();
    drop(rejected);
    let op = f.op(raw.clone(), &charge);
    let mut upload = rows.submit_device(&mut f, op).unwrap();
    assert_eq!(f.op.as_ref().unwrap().host.bytes().unwrap(), raw);
    drop(rows); // backend, not host expansion, now owns the DMA input
    assert_eq!(gov.borrow().used.pageable, 0);
    assert!(matches!(
        upload.publish(&mut f, epochs()),
        Err(Error::NotReady)
    ));
    f.ready();
    drop(upload.publish(&mut f, epochs()).unwrap());
    f.done = true;
    upload.acknowledge(&mut f).unwrap();
    gov.borrow_mut().release(&charge).unwrap();
    assert_eq!(gov.borrow().used, TierBudget::zero(2));
}
