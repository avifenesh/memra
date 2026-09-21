use super::*;
use std::thread;

struct CheckedBank {
    bank: SlruExpertDispatch<Heat, Reader>,
    thread: thread::ThreadId,
    calls: Rc<RefCell<Vec<&'static str>>>,
    fail_finish: bool,
}
impl ExpertDispatchBank for CheckedBank {
    fn validate(&self, id: ExpertDispatchId, bytes: usize) -> Result<()> {
        assert_eq!(thread::current().id(), self.thread);
        self.calls.borrow_mut().push("validate");
        self.bank.validate(id, bytes)
    }
    fn demand(&mut self, id: ExpertDispatchId, bytes: usize) -> Result<ExpertDemand> {
        assert_eq!(thread::current().id(), self.thread);
        self.calls.borrow_mut().push("stage/publish");
        self.bank.demand(id, bytes)
    }
    fn finish(&mut self, demand: ExpertDemand) -> Result<()> {
        assert_eq!(thread::current().id(), self.thread);
        self.calls.borrow_mut().push("finish");
        if std::mem::take(&mut self.fail_finish) {
            return Err(Error::NotReady);
        }
        self.bank.finish(demand)
    }
}
fn owner(fail_finish: bool) -> (ExpertBankOwner, Rc<RefCell<Vec<&'static str>>>) {
    let g = gov();
    let mut l = layout(9, 4, 16);
    l.segments.truncate(1);
    l.requirements.truncate(1);
    let id = bank_id(9, &l);
    let bank = BankService::new(
        Catalog::new(LayoutClass::Uniform, vec![(id.clone(), Some(record(l)))]).unwrap(),
        g.clone(),
        Heat::default(),
        Reader::default(),
        CoalescingPolicy {
            granularity: 1,
            slot_bytes: 32,
        },
        limits(32),
    )
    .unwrap();
    let mut req = request(bank.slru_metadata_bytes(1).unwrap(), Priority::Demand);
    let metadata = g.borrow_mut().reserve(&req).unwrap();
    let bank = bank
        .with_slru(SlruPolicy::new(&[(16, 1)]).unwrap(), &metadata)
        .unwrap();
    req.bytes = TierBudget::zero(2);
    let bank =
        SlruExpertDispatch::new(bank, BTreeMap::from([((2, 0, 9), id)]), req, epochs()).unwrap();
    let calls = Rc::new(RefCell::new(Vec::new()));
    let checked = CheckedBank {
        bank,
        thread: thread::current().id(),
        calls: calls.clone(),
        fail_finish,
    };
    (
        ExpertBankOwner::register(Box::new(checked), 1).unwrap(),
        calls,
    )
}
#[test]
fn proxy_and_token_are_send_sync_but_staging_is_owner_only() {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<ExpertBankProxy>();
    send_sync::<ExpertLeaseToken>();
    let (mut owner, calls) = owner(false);
    let proxy = owner.proxy();
    let token = proxy.demand((2, 0, 9), 16).unwrap();
    let other = proxy.clone();
    let token = thread::spawn(move || {
        assert_eq!(other.validate((2, 0, 9), 16), Err(Error::WrongOwner));
        assert_eq!(other.demand((2, 0, 9), 16), Err(Error::WrongOwner));
        assert_eq!(other.with_bytes(&token, |_| ()), Err(Error::WrongOwner));
        assert_eq!(other.finish(&token), Err(Error::WrongOwner));
        token
    })
    .join()
    .unwrap();
    assert_eq!(&*calls.borrow(), &["stage/publish"]);
    assert_eq!(proxy.with_bytes(&token, |bytes| bytes.len()), Ok(16));
    proxy.finish(&token).unwrap();
    owner.close().unwrap();
    assert_eq!(proxy.validate((2, 0, 9), 16), Err(Error::NotFound));
}
#[test]
fn proxy_bounds_pending_and_preserves_failed_finish_and_foreign_tokens() {
    let (mut owner1, _) = owner(true);
    let (mut owner2, _) = owner(false);
    let proxy = owner1.proxy();
    assert_eq!(proxy.validate((2, 0, 8), 16), Err(Error::NotFound));
    let token = proxy.demand((2, 0, 9), 16).unwrap();
    assert_eq!(proxy.demand((2, 0, 9), 16), Err(Error::Capacity));
    assert_eq!(owner1.close(), Err(Error::Busy));
    assert_eq!(owner2.proxy().finish(&token), Err(Error::ForeignLease));
    assert_eq!(proxy.finish(&token), Err(Error::NotReady));
    assert_eq!(proxy.with_bytes(&token, |b| b.len()), Ok(16));
    assert_eq!(
        proxy.with_bytes(&token, |_| proxy.validate((2, 0, 9), 16)),
        Ok(Err(Error::Busy))
    );
    proxy.finish(&token).unwrap();
    assert_eq!(proxy.finish(&token), Err(Error::UnknownTicket));
    assert_eq!(proxy.with_bytes(&token, |_| ()), Err(Error::UnknownTicket));
    owner1.close().unwrap();
    owner2.close().unwrap();
}
#[test]
fn dropping_owner_invalidates_proxy_without_auto_finishing_open_dma() {
    let (owner, calls) = owner(false);
    let proxy = owner.proxy();
    let token = proxy.demand((2, 0, 9), 16).unwrap();
    drop(owner);
    assert_eq!(proxy.finish(&token), Err(Error::NotFound));
    assert_eq!(&*calls.borrow(), &["stage/publish"]);
}
#[test]
fn token_identity_names_the_fixture_lease() {
    let (mut owner, _) = owner(false);
    let proxy = owner.proxy();
    let token = proxy.demand((2, 0, 9), 16).unwrap();
    assert_eq!(token.record(), (2, 0, 9));
    assert_eq!(token.artifact(), tensor().artifact);
    assert_eq!(token.epochs(), epochs());
    proxy.finish(&token).unwrap();
    owner.close().unwrap();
}
