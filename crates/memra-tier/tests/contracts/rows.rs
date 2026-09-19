//! Bounded byte-row fixture: one 4KiB slot, independent scale/payload layouts elsewhere.
use super::support::*;
use memra_tier::contracts::*;
use std::collections::{BTreeSet, HashMap};
struct Pending {
    ordered: Vec<BankLease>,
    failed: bool,
    cancelled: bool,
    published: bool,
    retired: bool,
}
struct PackedRows {
    gov: Shared,
    entries: HashMap<TransferTicket, Pending>,
    next: u64,
    read_pages: u64,
    fail_page: Option<u64>,
}
impl PackedRows {
    fn new() -> Self {
        Self {
            gov: std::rc::Rc::new(std::cell::RefCell::new(Governor::new(1_000_000))),
            entries: HashMap::new(),
            next: 0,
            read_pages: 0,
            fail_page: None,
        }
    }
    fn id(row: u64) -> BankId {
        let l = Self::layout(row);
        BankId {
            version: 1,
            tensor: l.segments[0].tensor.clone().unwrap(),
            record: RecordId::Row(row),
            layout: l.identity().unwrap(),
        }
    }
    fn layout(row: u64) -> RecordLayout {
        let mut l = layout();
        let s = &mut l.segments[0];
        s.tensor = Some(TensorId {
            version: 1,
            artifact: [8; 32],
            name: "table".into(),
        });
        s.offset = row * 264;
        s.valid_bytes = 264;
        s.storage_bytes = 264;
        s.alignment = 1;
        s.encoding.row_bytes = 264;
        l
    }
}
impl RowService for PackedRows {
    fn gather(&mut self, b: RowBatch) -> Result<TransferTicket> {
        if b.ids.is_empty() {
            return Err(Error::EmptyBatch);
        }
        let mut unique = HashMap::new();
        let mut pages = BTreeSet::new();
        for id in &b.ids {
            let RecordId::Row(n) = id.record else {
                return Err(Error::InvalidLayout);
            };
            if *id != Self::id(n) {
                return Err(Error::ProgramMismatch);
            }
            if !unique.contains_key(id) {
                let charge = self
                    .gov
                    .borrow_mut()
                    .reserve(&request(264, Priority::Demand))?;
                unique.insert(id.clone(), (vec![0; 264], charge));
            }
            for page in n * 264 / 4096..=(n * 264 + 263) / 4096 {
                pages.insert(page);
            }
        }
        let mut slot_request = request(0, Priority::Demand);
        slot_request.bytes.pinned = 4096;
        slot_request.bytes.staging = 4096;
        slot_request.bytes.inflight = 1;
        let slot_charge = self.gov.borrow_mut().reserve(&slot_request)?;
        let mut slot = vec![0u8; 4096];
        let mut failed = false;
        for page in pages {
            self.read_pages += 1;
            if self.fail_page == Some(page) {
                failed = true;
                continue;
            }
            for (i, v) in slot.iter_mut().enumerate() {
                *v = ((17 * (page * 4096 + i as u64) + 3) % 251) as u8;
            }
            for (id, (output, _)) in &mut unique {
                let RecordId::Row(row) = id.record else {
                    unreachable!()
                };
                let begin = row * 264;
                let end = begin + 264;
                let lo = begin.max(page * 4096);
                let hi = end.min((page + 1) * 4096);
                if lo < hi {
                    output[(lo - begin) as usize..(hi - begin) as usize].copy_from_slice(
                        &slot[(lo - page * 4096) as usize..(hi - page * 4096) as usize],
                    );
                }
            }
        }
        drop(slot);
        self.gov.borrow_mut().release(&slot_charge)?;
        let mut records = HashMap::new();
        for (id, (output, charge)) in unique {
            let RecordId::Row(n) = id.record else {
                unreachable!()
            };
            records.insert(
                id.clone(),
                BankLease::from_backend(
                    id,
                    Self::layout(n),
                    LayoutClass::PerRecord,
                    charge,
                    Box::new(output),
                )?,
            );
        }
        let ordered = b.ids.iter().map(|id| records[id].clone()).collect();
        self.next += 1;
        let t = TransferTicket {
            issuer: 88,
            sequence: self.next,
            epochs: b.epochs,
        };
        self.entries.insert(
            t,
            Pending {
                ordered,
                failed,
                cancelled: false,
                published: false,
                retired: false,
            },
        );
        Ok(t)
    }
    fn publish(&mut self, t: &TransferTicket, current: Epochs) -> Result<RowLease> {
        t.epochs.require(current)?;
        let e = self.entries.get_mut(t).ok_or(Error::UnknownTicket)?;
        if e.cancelled {
            return Err(Error::Cancelled);
        }
        if e.failed {
            return Err(Error::Incomplete);
        }
        if e.published {
            return Err(Error::Busy);
        }
        e.published = true;
        Ok(RowLease {
            records: e.ordered.clone(),
        })
    }
    fn cancel(&mut self, t: &TransferTicket) -> Result<CancelState> {
        let e = self.entries.get_mut(t).ok_or(Error::UnknownTicket)?;
        if e.published {
            Ok(CancelState::AlreadyPublished)
        } else {
            e.cancelled = true;
            Ok(CancelState::PublicationRevoked)
        }
    }
    fn retire(&mut self, t: &TransferTicket) -> Result<bool> {
        Ok(self.entries.get(t).ok_or(Error::UnknownTicket)?.retired)
    }
    fn release(&mut self, l: &RowLease) -> Result<()> {
        if self.entries.values().any(|e| {
            !e.retired
                && e.ordered
                    .iter()
                    .any(|r| l.records.iter().any(|x| x.charge().id() == r.charge().id()))
        }) {
            return Err(Error::Busy);
        }
        let mut seen = BTreeSet::new();
        for r in &l.records {
            if seen.insert(r.charge().id()) {
                r.retire_backing()?;
                self.gov.borrow_mut().release(r.charge())?;
            }
        }
        Ok(())
    }
}
#[test]
fn row_straddles_deduplicate_physical_reads_and_preserve_exact_duplicate_bytes() {
    let mut rows = PackedRows::new();
    let ids = vec![
        PackedRows::id(15),
        PackedRows::id(2),
        PackedRows::id(15),
        PackedRows::id(31),
    ];
    let t = rows
        .gather(RowBatch {
            ids,
            epochs: epochs(),
            request: request(0, Priority::Demand),
        })
        .unwrap();
    let l = rows.publish(&t, epochs()).unwrap();
    assert_eq!(rows.read_pages, 3);
    assert_eq!(rows.gov.borrow().used.pinned, 0);
    assert_eq!(rows.gov.borrow().used.pageable, 3 * 264);
    for r in &l.records {
        let RecordId::Row(n) = r.id().record else {
            panic!()
        };
        let b = r.resource::<Vec<u8>>().unwrap();
        assert_eq!(
            &*b,
            &(n * 264..n * 264 + 264)
                .map(|i| ((17 * i + 3) % 251) as u8)
                .collect::<Vec<_>>()
        );
    }
    assert_eq!(rows.release(&l), Err(Error::Busy));
    rows.entries.get_mut(&t).unwrap().retired = true;
    rows.release(&l).unwrap();
    assert_eq!(rows.gov.borrow().used.pageable, 0);
}
#[test]
fn row_batch_larger_than_pool_progresses_and_one_bad_physical_read_refuses_all_output() {
    for fail in [None, Some(1)] {
        let mut rows = PackedRows::new();
        rows.fail_page = fail;
        let ids = (0..32).map(PackedRows::id).collect();
        let t = rows
            .gather(RowBatch {
                ids,
                epochs: epochs(),
                request: request(0, Priority::Demand),
            })
            .unwrap();
        assert_eq!(rows.read_pages, 3);
        if fail.is_some() {
            assert!(matches!(rows.publish(&t, epochs()), Err(Error::Incomplete)));
        } else {
            assert_eq!(rows.publish(&t, epochs()).unwrap().records.len(), 32);
        }
    }
}
