//! Day-20 rule (WP-A, memra#536 Move 2 slice 1): the event-ordered publication of a D2D capture.
//! The CPU binding models the copy stream's completion event as a flag and the caller's
//! publication into the device prefix index as a state; a D2D item has no host bytes, so its
//! completion carries no checksum and `Completion::require` refuses it `Corrupt` (the slice-1
//! receipt clause). `publish_early` is the red arm: a caller that publishes on the copy's
//! ISSUE rather than on its completion event, the shape the worker change must never take.
use super::conformance::*;
use super::support::*;
use memra_tier::contracts::*;
use std::panic::{AssertUnwindSafe, catch_unwind};

struct CaptureBatch {
    publish_early: bool,
    event_done: bool,
    completion: Option<Completion>,
    bytes: Vec<u64>,
    published: bool,
    retired: bool,
    acknowledged: bool,
    destinations: usize,
}
impl CaptureBatch {
    fn new(items: usize) -> Self {
        Self {
            publish_early: false,
            event_done: false,
            completion: None,
            bytes: (0..items).map(|i| 64 + 8 * i as u64).collect(),
            published: false,
            retired: false,
            acknowledged: false,
            destinations: items,
        }
    }
    fn entry(&mut self, ticket: &TransferTicket) -> Result<&mut Completion> {
        match &mut self.completion {
            Some(c) if &c.ticket == ticket => Ok(c),
            _ => Err(Error::UnknownTicket),
        }
    }
}
impl D2dCaptureFixture for CaptureBatch {
    fn submit(&mut self) -> TransferTicket {
        let ticket = TransferTicket {
            issuer: 93,
            sequence: 1,
            epochs: epochs(),
        };
        // Every copy is issued on the copy stream behind the producer event: pending, no
        // checksum (no host bytes), fenced at submit as a D2H is (the destination's consumer is
        // the index publication, host-ordered after the event).
        let mut c = completion(ticket, self.bytes.len());
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
                    issuer: 93,
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
        let bytes = self.bytes.clone();
        let c = self.entry(ticket)?;
        if done {
            for (item, bytes) in c.items.iter_mut().zip(bytes) {
                for s in &mut item.segments {
                    s.status = ItemStatus::Complete;
                    s.producer_done = true;
                    s.valid_bytes = bytes;
                    s.checksum = None; // slice 1: no witnessed checksum term for a D2D
                }
            }
            c.producer_done = true;
        }
        Ok(c.clone())
    }
    fn landed(&mut self, ticket: &TransferTicket) -> Result<bool> {
        let c = self.poll(ticket)?;
        Ok(c.producer_done
            && c.items
                .iter()
                .filter(|i| i.accepted)
                .all(|i| i.segments.iter().all(|s| s.producer_done)))
    }
    fn publish(&mut self, ticket: &TransferTicket) -> Result<()> {
        // The caller's publication: the entry enters the device prefix index. It asks `landed`
        // first; the red arm asks nothing and publishes on issue.
        if self.published {
            return Err(Error::AlreadyReleased);
        }
        if !self.publish_early && !self.landed(ticket)? {
            return Err(Error::NotReady);
        }
        self.entry(ticket)?;
        self.published = true;
        Ok(())
    }
    fn require_receipt(&mut self, ticket: &TransferTicket) -> Result<()> {
        let c = self.poll(ticket)?;
        let expected: Vec<Vec<SegmentExpectation>> = self
            .bytes
            .iter()
            .map(|&b| {
                vec![SegmentExpectation {
                    valid_bytes: b,
                    io_bytes: b,
                    checksum: [0; 32],
                }]
            })
            .collect();
        c.require(ticket, &expected, true)
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
            return Err(Error::WrongOwner); // a capture records no consumer fence
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
    fn destination_back(&mut self, ticket: &TransferTicket) -> Result<usize> {
        if !self.acknowledged {
            self.entry(ticket)?;
            return Err(Error::Busy);
        }
        Ok(std::mem::take(&mut self.destinations))
    }
    fn submitted_bytes(&self) -> Vec<u64> {
        self.bytes.clone()
    }
}

/// The schedule on a two-item and a one-item batch: nothing publishes before the event, the
/// landed batch publishes once with exact bytes, retires without a consumer fence, acknowledges,
/// and every destination comes back.
#[test]
fn day20_d2d_capture_publishes_only_after_every_items_event() {
    let mut f = CaptureBatch::new(2);
    d2d_capture_publish(&mut f);
    assert!(f.published && f.retired && f.acknowledged);
    assert_eq!(f.destinations, 0, "both destinations left the engine");
    let mut one = CaptureBatch::new(1);
    d2d_capture_publish(&mut one);
    assert_eq!(one.destinations, 0);
}

/// Red arm: a caller that publishes on the copy's issue (before the event). The schedule fails,
/// and this is the shape of its failure: the index holds an entry whose planes are still being
/// written by the copy stream, the capture law's forbidden partial entry.
#[test]
fn day20_red_arm_publish_before_the_event_fails_the_schedule() {
    let mut f = CaptureBatch::new(2);
    f.publish_early = true;
    let schedule = catch_unwind(AssertUnwindSafe(|| {
        d2d_capture_publish(&mut f);
    }));
    assert!(
        schedule.is_err(),
        "a publish before the completion event must not pass the capture rule"
    );
    let mut f = CaptureBatch::new(2);
    f.publish_early = true;
    let ticket = f.submit();
    f.publish(&ticket).unwrap(); // the early publish "succeeded"
    assert!(
        !f.landed(&ticket).unwrap(),
        "the index names an entry whose copies have not landed"
    );
    assert_eq!(f.retire(&ticket, None), Err(Error::Busy));
}

/// The slice-1 receipt clause on its own: a D2D completion has no checksum term, so the
/// host-contract gate (`Completion::require`, device = true) refuses it before and after the
/// event; publication of a capture never goes through that gate until slice 3 witnesses the term.
#[test]
fn day20_d2d_item_without_a_witnessed_checksum_is_refused_by_the_host_contract_gate() {
    let mut f = CaptureBatch::new(3);
    let ticket = f.submit();
    assert!(matches!(
        f.require_receipt(&ticket),
        Err(Error::Corrupt | Error::NotReady)
    ));
    f.copy_completes();
    assert!(f.landed(&ticket).unwrap());
    assert_eq!(f.require_receipt(&ticket), Err(Error::Corrupt));
}

// ---------------------------------------------------------------------------
// Day 24 (WP-A, memra#536 Move 2 owed item 2, the capture half): the draft-bearing capture. The
// CPU binding models the class of every item, a per-class landing (the trunk's events fire first,
// the draft's later) and the slice-3 receipt term per item (a landed item's checksum equals the
// modelled source digest). `publish_trunk_only` is the red arm: a caller that publishes on the
// trunk's landing while a draft item is still running.

struct DraftCaptureBatch {
    publish_trunk_only: bool,
    classes: Vec<Role>,
    trunk_done: bool,
    draft_done: bool,
    completion: Option<Completion>,
    bytes: Vec<u64>,
    published: bool,
    retired: bool,
    acknowledged: bool,
    destinations: usize,
}
impl DraftCaptureBatch {
    /// `trunk` trunk items (K, V alternating) followed by the draft plane's K and V items.
    fn draft_bearing(trunk: usize) -> Self {
        let classes: Vec<Role> = (0..trunk)
            .map(|i| if i % 2 == 0 { Role::Key } else { Role::Value })
            .chain([Role::Draft, Role::Draft])
            .collect();
        let bytes = classes
            .iter()
            .enumerate()
            .map(|(i, r)| match r {
                Role::Draft => 24 + 8 * i as u64,
                _ => 64 + 8 * i as u64,
            })
            .collect();
        Self {
            publish_trunk_only: false,
            destinations: classes.len(),
            classes,
            trunk_done: false,
            draft_done: false,
            completion: None,
            bytes,
            published: false,
            retired: false,
            acknowledged: false,
        }
    }
    /// The modelled receipt term of item `i`: its class and byte count folded into 32 bytes (the
    /// source digest, taken behind the producer fence; the destination digest equals it once the
    /// item's copy landed).
    fn digest_of(&self, i: usize) -> Digest {
        let mut d = [0u8; 32];
        d[0] = match self.classes[i] {
            Role::Draft => 0xD7,
            Role::Key => 0x4B,
            Role::Value => 0x56,
            _ => 0x00,
        };
        d[1..9].copy_from_slice(&self.bytes[i].to_le_bytes());
        d[9] = i as u8;
        d
    }
    fn item_done(&self, i: usize) -> bool {
        match self.classes[i] {
            Role::Draft => self.draft_done,
            _ => self.trunk_done,
        }
    }
    fn entry(&mut self, ticket: &TransferTicket) -> Result<&mut Completion> {
        match &mut self.completion {
            Some(c) if &c.ticket == ticket => Ok(c),
            _ => Err(Error::UnknownTicket),
        }
    }
}
impl D2dCaptureFixture for DraftCaptureBatch {
    fn submit(&mut self) -> TransferTicket {
        let ticket = TransferTicket {
            issuer: 94,
            sequence: 1,
            epochs: epochs(),
        };
        let mut c = completion(ticket, self.bytes.len());
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
        let done: Vec<bool> = (0..self.bytes.len()).map(|i| self.item_done(i)).collect();
        let digests: Vec<Digest> = (0..self.bytes.len()).map(|i| self.digest_of(i)).collect();
        let bytes = self.bytes.clone();
        let c = self.entry(ticket)?;
        for (i, item) in c.items.iter_mut().enumerate() {
            if !done[i] {
                continue;
            }
            for s in &mut item.segments {
                s.status = ItemStatus::Complete;
                s.producer_done = true;
                s.valid_bytes = bytes[i];
                s.checksum = Some(digests[i]);
            }
        }
        // The batch's landing is every item's event, never a class's alone.
        c.producer_done = done.iter().all(|d| *d);
        Ok(c.clone())
    }
    fn landed(&mut self, ticket: &TransferTicket) -> Result<bool> {
        let c = self.poll(ticket)?;
        Ok(c.producer_done
            && c.items
                .iter()
                .filter(|i| i.accepted)
                .all(|i| i.segments.iter().all(|s| s.producer_done)))
    }
    fn publish(&mut self, ticket: &TransferTicket) -> Result<()> {
        if self.published {
            return Err(Error::AlreadyReleased);
        }
        // The red arm asks only the trunk; the rule asks every item of both classes.
        let ready = if self.publish_trunk_only {
            self.trunk_done
        } else {
            self.landed(ticket)?
        };
        if !ready {
            return Err(Error::NotReady);
        }
        self.entry(ticket)?;
        self.published = true;
        Ok(())
    }
    fn require_receipt(&mut self, ticket: &TransferTicket) -> Result<()> {
        let c = self.poll(ticket)?;
        let expected: Vec<Vec<SegmentExpectation>> = (0..self.bytes.len())
            .map(|i| {
                vec![SegmentExpectation {
                    valid_bytes: self.bytes[i],
                    io_bytes: self.bytes[i],
                    checksum: self.digest_of(i),
                }]
            })
            .collect();
        c.require(ticket, &expected, true)
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
        self.trunk_done = true;
        self.draft_done = true;
    }
    fn destination_back(&mut self, ticket: &TransferTicket) -> Result<usize> {
        if !self.acknowledged {
            self.entry(ticket)?;
            return Err(Error::Busy);
        }
        Ok(std::mem::take(&mut self.destinations))
    }
    fn submitted_bytes(&self) -> Vec<u64> {
        self.bytes.clone()
    }
}
impl D2dDraftCaptureFixture for DraftCaptureBatch {
    fn item_classes(&self) -> Vec<Role> {
        self.classes.clone()
    }
    fn trunk_completes(&mut self) {
        self.trunk_done = true;
    }
    fn witnessed_classes(&mut self, ticket: &TransferTicket) -> Result<Vec<Role>> {
        let c = self.poll(ticket)?;
        Ok(c.items
            .iter()
            .enumerate()
            .filter(|(i, item)| {
                item.segments.iter().all(|s| {
                    s.status == ItemStatus::Complete && s.checksum == Some(self.digest_of(*i))
                })
            })
            .map(|(i, _)| self.classes[i])
            .collect())
    }
}

/// The schedule on the 27B's shape (32 trunk items plus the draft plane's two, `items=34`) and the
/// 9B's (`items=18`): the trunk's landing alone publishes nothing, the whole landing publishes once
/// with both classes witnessed, and every destination of both classes comes back.
#[test]
fn day24_draft_bearing_capture_publishes_both_classes_or_neither() {
    let mut f = DraftCaptureBatch::draft_bearing(32);
    d2d_capture_draft_publish(&mut f);
    assert!(f.published && f.retired && f.acknowledged);
    assert_eq!(f.destinations, 0, "all 34 destinations left the engine");
    let mut small = DraftCaptureBatch::draft_bearing(16);
    d2d_capture_draft_publish(&mut small);
    assert_eq!(small.destinations, 0);
}

/// Red arm: a caller that publishes on the trunk's landing while the draft plane is still being
/// written. The schedule fails, and this is the shape of its failure: the index names an entry
/// whose draft plane is not landed, so the next hit's spec session would read rows the copy stream
/// is still writing.
#[test]
fn day24_red_arm_publish_with_the_draft_unlanded_fails_the_schedule() {
    let mut f = DraftCaptureBatch::draft_bearing(4);
    f.publish_trunk_only = true;
    let schedule = catch_unwind(AssertUnwindSafe(|| {
        d2d_capture_draft_published_with_the_draft_unlanded_fails(&mut f);
    }));
    assert!(
        schedule.is_err(),
        "a publish on the trunk's landing alone must not pass the draft-bearing capture rule"
    );
    let mut f = DraftCaptureBatch::draft_bearing(4);
    f.publish_trunk_only = true;
    let ticket = f.submit();
    f.trunk_completes();
    f.publish(&ticket).unwrap(); // the early publish "succeeded"
    assert!(
        !f.landed(&ticket).unwrap(),
        "the index names an entry whose draft plane has not landed"
    );
    assert_eq!(f.retire(&ticket, None), Err(Error::Busy));
    let witnessed = f.witnessed_classes(&ticket).unwrap();
    assert!(!witnessed.contains(&Role::Draft));
}

/// The receipt clause per class on its own: with the trunk landed and the draft running the
/// host-contract gate is closed; landed whole, it admits.
#[test]
fn day24_unwitnessed_draft_items_keep_the_host_contract_gate_closed() {
    let mut f = DraftCaptureBatch::draft_bearing(2);
    let ticket = f.submit();
    f.trunk_completes();
    assert!(matches!(
        f.require_receipt(&ticket),
        Err(Error::NotReady | Error::Corrupt)
    ));
    assert_eq!(
        f.witnessed_classes(&ticket).unwrap(),
        vec![Role::Key, Role::Value]
    );
    f.copy_completes();
    assert_eq!(f.require_receipt(&ticket), Ok(()));
    assert_eq!(
        f.witnessed_classes(&ticket).unwrap(),
        vec![Role::Key, Role::Value, Role::Draft, Role::Draft]
    );
}
