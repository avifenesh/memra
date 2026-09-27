//! Day 47 (`research/spill-c-20260919/DAY47.md`): record buffers from a `HostBufferSource`. A read
//! lands straight in a pooled buffer, verifies and leases the same bytes the heap path would; a
//! buffer returns to its pool exactly once and only after its lease is released; an exhausted
//! source refuses the read and nothing is allocated or charged in its place.
use super::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

struct TestBuf {
    bytes: Vec<u8>,
    returned: Arc<AtomicUsize>,
}
impl HostBuffer for TestBuf {
    fn as_slice(&self) -> &[u8] {
        &self.bytes
    }
    fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.bytes
    }
}
impl Drop for TestBuf {
    fn drop(&mut self) {
        self.returned.fetch_add(1, Ordering::SeqCst);
    }
}
struct TestPool {
    left: usize,
    taken: Arc<AtomicUsize>,
    returned: Arc<AtomicUsize>,
}
impl HostBufferSource for TestPool {
    fn take(&mut self, len: usize) -> Option<Box<dyn HostBuffer>> {
        if self.left == 0 {
            return None;
        }
        self.left -= 1;
        self.taken.fetch_add(1, Ordering::SeqCst);
        // Poisoned contents: the read must overwrite every byte it leases.
        Some(Box::new(TestBuf {
            bytes: vec![0xEE; len],
            returned: self.returned.clone(),
        }))
    }
}

fn dispatch(
    pool: Option<TestPool>,
    slots: usize,
) -> (
    SlruExpertDispatch<Heat, Reader>,
    Vec<BankId>,
    Rc<RefCell<Governor>>,
) {
    let g = gov();
    let mut entries = Vec::new();
    let mut ids = Vec::new();
    for n in 0..6u32 {
        let mut l = layout(u64::from(n), 13, 16);
        l.segments.truncate(1);
        l.requirements.truncate(1);
        let id = bank_id(n, &l);
        ids.push(id.clone());
        entries.push((id, Some(record(l))));
    }
    let mut bank = BankService::<ExpertDomain, _, _>::new(
        Catalog::new(LayoutClass::Uniform, entries).unwrap(),
        g.clone(),
        Heat::default(),
        Reader::default(),
        CoalescingPolicy {
            granularity: 1,
            slot_bytes: 32,
        },
        limits(16 * slots as u64),
    )
    .unwrap();
    if let Some(pool) = pool {
        bank = bank.with_host_buffers(Box::new(pool)).unwrap();
    }
    let mut req = request(bank.slru_metadata_bytes(slots).unwrap(), Priority::Demand);
    let metadata = g.borrow_mut().reserve(&req).unwrap();
    let bank = bank
        .with_slru(SlruPolicy::new(&[(16, slots)]).unwrap(), &metadata)
        .unwrap();
    req.bytes = TierBudget::zero(2);
    let map = ids
        .iter()
        .map(|b| (dispatch_id(&b.record).unwrap(), b.clone()))
        .collect();
    (
        SlruExpertDispatch::new(bank, map, req, epochs()).unwrap(),
        ids,
        g,
    )
}

fn lent(d: &mut SlruExpertDispatch<Heat, Reader>, id: &BankId) -> Vec<u8> {
    let local = dispatch_id(&id.record).unwrap();
    let demand = d.demand(local, 16).unwrap();
    let bytes = match demand.lease.resource::<Vec<u8>>() {
        Ok(v) => v.clone(),
        Err(_) => demand
            .lease
            .resource::<Box<dyn HostBuffer>>()
            .unwrap()
            .as_slice()
            .to_vec(),
    };
    d.finish(demand).unwrap();
    bytes
}

#[test]
fn a_pooled_read_leases_the_same_bytes_as_the_heap_path() {
    let (taken, returned) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
    let pool = TestPool {
        left: 64,
        taken: taken.clone(),
        returned: returned.clone(),
    };
    let (mut heap, ids, _) = dispatch(None, 2);
    let (mut pooled, _, _) = dispatch(Some(pool), 2);
    for id in &ids {
        assert_eq!(lent(&mut heap, id), lent(&mut pooled, id), "{id:?}");
    }
    // Two host slots over six records: four evictions released four pooled buffers, each
    // exactly once; the two cached records still hold theirs.
    assert_eq!(taken.load(Ordering::SeqCst), 6);
    assert_eq!(returned.load(Ordering::SeqCst), 4);
    assert_eq!(pooled.bank().cached_records(), 2);
    drop(pooled);
    assert_eq!(returned.load(Ordering::SeqCst), 6);
}

#[test]
fn a_buffer_returns_only_after_its_lease_is_released() {
    let (taken, returned) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
    let pool = TestPool {
        left: 64,
        taken: taken.clone(),
        returned: returned.clone(),
    };
    let (mut d, ids, _) = dispatch(Some(pool), 1);
    let first = d.demand(dispatch_id(&ids[0].record).unwrap(), 16).unwrap();
    // A second record evicts the first from the one host slot while the first's ticket is still
    // open (a copy may still read it): the first's buffer must survive.
    let second = d.demand(dispatch_id(&ids[1].record).unwrap(), 16).unwrap();
    assert!(
        d.bank().slru_policy().unwrap().resident(&ids[0]).is_none(),
        "the first record was not evicted"
    );
    d.finish(second).unwrap();
    assert_eq!(
        returned.load(Ordering::SeqCst),
        0,
        "an evicted lease with an open ticket lost its buffer"
    );
    d.finish(first).unwrap();
    assert_eq!(
        returned.load(Ordering::SeqCst),
        1,
        "released after its last ticket"
    );
    assert_eq!(taken.load(Ordering::SeqCst), 2);
}

#[test]
fn an_exhausted_source_refuses_and_charges_nothing() {
    let (taken, returned) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
    let pool = TestPool {
        left: 1,
        taken: taken.clone(),
        returned: returned.clone(),
    };
    let (mut d, ids, g) = dispatch(Some(pool), 4);
    lent(&mut d, &ids[0]);
    let used = g.borrow().used();
    let refused = d.demand(dispatch_id(&ids[1].record).unwrap(), 16);
    assert_eq!(refused.err(), Some(Error::Capacity));
    assert_eq!(g.borrow().used(), used, "a refused read left a charge");
    assert!(d.bank().tickets().is_empty());
    assert_eq!(taken.load(Ordering::SeqCst), 1);
}
