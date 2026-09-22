//! Day-22 rule (WP-A, memra#536 Move 2 slice 3): the receipt term of the two D2D classes. The CPU
//! binding models the copy stream's completion event as a flag and the witness as data: the source
//! digest is the CPU oracle over the item's payload; the destination digest is the same oracle over
//! the payload (a matching witness), absent (a receipt-less item, the slice-1 shape) or the oracle
//! over a fresh zeroed plane the delayed copy has not reached (the `d2d-delay` fault's early reader,
//! the red arm). The caller's publish asks `Completion::require` first and latches on `Corrupt`.
use super::conformance::*;
use super::support::*;
use memra_tier::contracts::*;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Witness {
    Matching,
    Missing,
    EarlyReader,
}

struct ReceiptBatch {
    witness: Witness,
    event_done: bool,
    completion: Option<Completion>,
    payloads: Vec<Vec<u8>>,
    published: bool,
    latched: bool,
    retired: bool,
    acknowledged: bool,
}
impl ReceiptBatch {
    fn new(items: usize, witness: Witness) -> Self {
        Self {
            witness,
            event_done: false,
            completion: None,
            payloads: (0..items)
                .map(|i| (0..64 + 8 * i).map(|b| (b * 31 + i * 7) as u8).collect())
                .collect(),
            published: false,
            latched: false,
            retired: false,
            acknowledged: false,
        }
    }
    fn entry(&mut self, ticket: &TransferTicket) -> Result<&mut Completion> {
        match &mut self.completion {
            Some(c) if &c.ticket == ticket => Ok(c),
            _ => Err(Error::UnknownTicket),
        }
    }
    fn destination_digest(&self, i: usize) -> Option<Digest> {
        let p = &self.payloads[i];
        match self.witness {
            Witness::Matching => Some(receipt_digest(p)),
            Witness::Missing => None,
            // The early reader looked at the fresh destination before the delayed copy reached it.
            Witness::EarlyReader => Some(receipt_digest(&vec![0u8; p.len()])),
        }
    }
    fn expected(&self) -> Vec<Vec<SegmentExpectation>> {
        self.payloads
            .iter()
            .map(|p| {
                vec![SegmentExpectation {
                    valid_bytes: p.len() as u64,
                    io_bytes: p.len() as u64,
                    checksum: receipt_digest(p),
                }]
            })
            .collect()
    }
}
impl D2dReceiptFixture for ReceiptBatch {
    fn submit(&mut self) -> TransferTicket {
        let ticket = TransferTicket {
            issuer: 94,
            sequence: 1,
            epochs: epochs(),
        };
        let mut c = completion(ticket, self.payloads.len());
        c.producer_done = false;
        c.consumer_fenced = true;
        for item in &mut c.items {
            for s in &mut item.segments {
                s.status = ItemStatus::Pending;
                s.producer_done = false;
                s.checksum = None;
                s.valid_bytes = 0;
                s.consumer_fenced = true;
                s.consumer_fence = Some(FenceId {
                    issuer: 94,
                    owner: 0,
                    generation: epochs().dst_gen,
                    sequence: 1,
                });
            }
        }
        self.completion = Some(c);
        ticket
    }
    fn poll(&mut self, ticket: &TransferTicket) -> Result<Completion> {
        let done = self.event_done;
        let witnessed: Vec<(u64, Option<Digest>)> = (0..self.payloads.len())
            .map(|i| (self.payloads[i].len() as u64, self.destination_digest(i)))
            .collect();
        let c = self.entry(ticket)?;
        if done {
            for (item, (bytes, digest)) in c.items.iter_mut().zip(witnessed) {
                for s in &mut item.segments {
                    s.status = ItemStatus::Complete;
                    s.producer_done = true;
                    s.valid_bytes = bytes;
                    s.checksum = digest;
                }
            }
            c.producer_done = true;
        }
        Ok(c.clone())
    }
    fn landed(&mut self, ticket: &TransferTicket) -> Result<bool> {
        Ok(self.poll(ticket)?.producer_done)
    }
    fn receipt(&mut self, ticket: &TransferTicket) -> Result<Vec<ReceiptTerm>> {
        if !self.landed(ticket)? {
            return Err(Error::NotReady);
        }
        Ok((0..self.payloads.len())
            .map(|i| ReceiptTerm {
                source: receipt_digest(&self.payloads[i]),
                destination: self.destination_digest(i),
            })
            .collect())
    }
    fn require_receipt(&mut self, ticket: &TransferTicket) -> Result<()> {
        let c = self.poll(ticket)?;
        c.require(ticket, &self.expected(), true)
    }
    fn publish(&mut self, ticket: &TransferTicket) -> Result<()> {
        if self.latched {
            return Err(Error::Corrupt);
        }
        if self.published {
            return Err(Error::AlreadyReleased);
        }
        if !self.landed(ticket)? {
            return Err(Error::NotReady);
        }
        match self.require_receipt(ticket) {
            Ok(()) => {
                self.published = true;
                Ok(())
            }
            Err(Error::Corrupt) => {
                self.latched = true;
                Err(Error::Corrupt)
            }
            Err(e) => Err(e),
        }
    }
    fn latched(&self) -> bool {
        self.latched
    }
    fn retire(&mut self, ticket: &TransferTicket, consumer_done: Option<FenceId>) -> Result<()> {
        let c = self.poll(ticket)?;
        if self.retired {
            return Ok(());
        }
        if !c.producer_done {
            return Err(Error::Busy);
        }
        if consumer_done.is_some() {
            return Err(Error::WrongOwner);
        }
        self.retired = true;
        Ok(())
    }
    fn retired(&mut self, ticket: &TransferTicket) -> Result<bool> {
        self.entry(ticket)?;
        if self.acknowledged {
            return Err(Error::UnknownTicket);
        }
        Ok(self.retired)
    }
    fn acknowledge(&mut self, ticket: &TransferTicket) -> Result<()> {
        self.entry(ticket)?;
        if !self.retired {
            return Err(Error::Busy);
        }
        self.acknowledged = true;
        Ok(())
    }
    fn copy_completes(&mut self) {
        self.event_done = true;
    }
    fn submitted_bytes(&self) -> Vec<u64> {
        self.payloads.iter().map(|p| p.len() as u64).collect()
    }
}

/// Rules 1 and 2 on a two-item and a one-item batch: no receipt before the event, both digests
/// named and equal after it, the gate opens, one publication, retire, acknowledge.
#[test]
fn day22_d2d_receipt_matching_publishes_once() {
    let mut f = ReceiptBatch::new(2, Witness::Matching);
    d2d_receipt_witnessed(&mut f);
    assert!(f.published && f.retired && f.acknowledged && !f.latched);
    let mut one = ReceiptBatch::new(1, Witness::Matching);
    d2d_receipt_witnessed(&mut one);
    assert!(one.published && !one.latched);
}

/// Rule 3: a receipt-less item (the slice-1 and slice-2 shape) is refused `Corrupt`, the caller
/// latches, nothing publishes, the landed ticket still leaves cleanly.
#[test]
fn day22_receipt_less_item_is_refused_and_latches() {
    let mut f = ReceiptBatch::new(3, Witness::Missing);
    d2d_receipt_refused(&mut f);
    assert!(!f.published && f.latched && f.retired && f.acknowledged);
    let terms = {
        let mut g = ReceiptBatch::new(1, Witness::Missing);
        let t = g.submit();
        g.copy_completes();
        g.receipt(&t).unwrap()
    };
    assert!(
        terms[0].destination.is_none(),
        "no witness, no destination digest"
    );
}

/// Rule 4, the red arm: the `d2d-delay` early reader's digest of a fresh destination differs from
/// the source digest; refused `Corrupt`, latched, nothing published, the ticket leaves cleanly. The
/// matching schedule must FAIL on this fixture: a mismatch is never a tolerance.
#[test]
fn day22_red_arm_early_reader_receipt_is_refused_and_latches() {
    let mut f = ReceiptBatch::new(2, Witness::EarlyReader);
    d2d_receipt_refused(&mut f);
    assert!(!f.published && f.latched && f.retired && f.acknowledged);
    let mut g = ReceiptBatch::new(2, Witness::EarlyReader);
    let t = g.submit();
    g.copy_completes();
    for term in g.receipt(&t).unwrap() {
        assert!(
            term.destination.is_some(),
            "the early reader DID witness a digest"
        );
        assert_ne!(
            term.destination,
            Some(term.source),
            "and it is the stale plane's"
        );
    }
    let schedule = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut h = ReceiptBatch::new(2, Witness::EarlyReader);
        d2d_receipt_witnessed(&mut h);
    }));
    assert!(
        schedule.is_err(),
        "a mismatching receipt must not pass the matching schedule"
    );
}

/// The CPU oracle on fixed vectors: deterministic; a one-byte flip, a one-byte extension (a
/// trailing zero, which the zero-padded last word alone would not see) and a transposition each
/// change the digest; the lanes-plus-count split equals the whole.
#[test]
fn day22_receipt_digest_oracle_is_stable() {
    let base: Vec<u8> = (0..1000u32).map(|i| (i * 131 % 251) as u8).collect();
    let d = receipt_digest(&base);
    assert_eq!(d, receipt_digest(&base));
    assert_eq!(
        d,
        receipt_digest_from_lanes(receipt_lanes(&base), base.len() as u64)
    );
    let mut flip = base.clone();
    flip[517] ^= 0x01;
    assert_ne!(receipt_digest(&flip), d, "a one-byte flip");
    let mut ext = base.clone();
    ext.push(0);
    assert_ne!(receipt_digest(&ext), d, "a trailing zero byte");
    let mut swap = base.clone();
    swap.swap(8, 16);
    assert_ne!(receipt_digest(&swap), d, "a word transposition");
    assert_ne!(
        receipt_digest(&[]),
        receipt_digest(&[0]),
        "empty against one zero"
    );
    assert_ne!(
        receipt_digest(&[1, 2, 3, 4, 5, 6, 7, 8]),
        receipt_digest(&[1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0]),
        "one word against the same word plus a zero word"
    );
    assert_ne!(
        d,
        checksum(&base),
        "the receipt program is not the SHA-256 checksum"
    );
}
