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
