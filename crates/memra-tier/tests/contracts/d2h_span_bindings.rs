//! Day-30 rule (WP-A, memra#536 Move 2 owed item 1, the D2H half): the typed f32 spans of a
//! demote batch. The CPU binding models the KV items' events and the spans' events as two flags
//! and the engine's ownership of each span as a slot. `take_on_items` is the red arm: a binding
//! that hands the spans back on the KV items' landing alone, the shape the engine must never take.
use super::conformance::*;
use super::support::*;
use memra_tier::contracts::*;
use std::panic::{AssertUnwindSafe, catch_unwind};

struct SpanBatch {
    take_on_items: bool,
    fail_next: bool,
    items_done: bool,
    spans_done: bool,
    quarantined: bool,
    completion: Option<Completion>,
    lens: Vec<u64>,
    attached: Option<Vec<u64>>,
    taken: bool,
    retired: bool,
    acknowledged: bool,
}
impl SpanBatch {
    fn new(items: usize, spans: usize) -> Self {
        Self {
            take_on_items: false,
            fail_next: false,
            items_done: false,
            spans_done: false,
            quarantined: false,
            completion: Some(completion(
                TransferTicket {
                    issuer: 94,
                    sequence: 1,
                    epochs: epochs(),
                },
                items,
            )),
            lens: (0..spans).map(|i| 4096 + 64 * i as u64).collect(),
            attached: None,
            taken: false,
            retired: false,
            acknowledged: false,
        }
    }
    fn live(&self, ticket: &TransferTicket) -> Result<()> {
        match &self.completion {
            Some(c) if &c.ticket == ticket && !self.retired => Ok(()),
            _ => Err(Error::UnknownTicket),
        }
    }
    fn landed(&self) -> bool {
        self.items_done && (self.attached.is_none() || self.spans_done || self.take_on_items)
    }
}
impl D2hSpanFixture for SpanBatch {
    fn submit(&mut self) -> TransferTicket {
        let c = self.completion.as_mut().unwrap();
        c.producer_done = false;
        for item in &mut c.items {
            for s in &mut item.segments {
                s.status = ItemStatus::Pending;
                s.producer_done = false;
            }
        }
        c.ticket
    }
    fn attach(&mut self, ticket: &TransferTicket) -> std::result::Result<(), (Error, usize)> {
        let n = self.lens.len();
        self.live(ticket).map_err(|e| (e, n))?;
        if self.attached.is_some() {
            return Err((Error::Busy, n));
        }
        if self.lens.contains(&0) {
            return Err((Error::InvalidLayout, n));
        }
        // The first span enqueued; a failure on the second quarantines the ticket, the engine
        // keeps every span.
        if std::mem::take(&mut self.fail_next) {
            self.quarantined = true;
        }
        self.attached = Some(self.lens.clone());
        Ok(())
    }
    fn attach_invalid(&mut self, ticket: &TransferTicket) -> (Error, usize) {
        let saved = std::mem::replace(&mut self.lens, vec![4096, 0]);
        let out = self
            .attach(ticket)
            .expect_err("a zero-byte span is refused");
        self.lens = saved;
        out
    }
    fn poll(&mut self, ticket: &TransferTicket) -> Result<Completion> {
        self.live(ticket)?;
        if self.quarantined {
            return Err(Error::Quarantined);
        }
        let items_done = self.items_done;
        let landed = self.landed();
        let c = self.completion.as_mut().unwrap();
        if items_done {
            for item in &mut c.items {
                for s in &mut item.segments {
                    s.status = ItemStatus::Complete;
                    s.producer_done = true;
                }
            }
        }
        c.producer_done = landed;
        Ok(c.clone())
    }
    fn take_spans(&mut self, ticket: &TransferTicket) -> Result<Vec<u64>> {
        self.live(ticket)?;
        if self.quarantined {
            return Err(Error::Quarantined);
        }
        if self.taken {
            return Err(Error::AlreadyReleased);
        }
        let Some(lens) = self.attached.clone() else {
            return Err(Error::UnknownTicket);
        };
        if !self.landed() {
            return Err(Error::NotReady);
        }
        self.taken = true;
        Ok(lens)
    }
    fn retire(&mut self, ticket: &TransferTicket) -> Result<()> {
        self.live(ticket)?;
        if self.quarantined {
            return Err(Error::Quarantined);
        }
        if !self.landed() || (self.attached.is_some() && !self.taken) {
            return Err(Error::Busy);
        }
        self.retired = true;
        Ok(())
    }
    fn acknowledge(&mut self, ticket: &TransferTicket) -> Result<()> {
        match &self.completion {
            Some(c) if &c.ticket == ticket && self.retired && !self.acknowledged => {
                self.acknowledged = true;
                Ok(())
            }
            _ => Err(Error::UnknownTicket),
        }
    }
    fn items_complete(&mut self) {
        self.items_done = true;
    }
    fn spans_complete(&mut self) {
        self.spans_done = true;
    }
    fn fail_enqueue(&mut self) {
        self.fail_next = true;
    }
    fn span_bytes(&self) -> Vec<u64> {
        self.lens.clone()
    }
}

#[test]
fn d2h_span_batch_lands_with_its_ticket_and_comes_back_once() {
    let mut f = SpanBatch::new(4, 12);
    d2h_span_batch(&mut f);
    assert!(f.acknowledged);
    // The KV items' host-contract gate is unchanged by the spans.
    let c = f.completion.as_ref().unwrap();
    assert!(c.require(&c.ticket, &expected(4), false).is_ok());
}

#[test]
fn d2h_span_enqueue_failure_quarantines_the_ticket() {
    let mut f = SpanBatch::new(4, 12);
    d2h_span_enqueue_failure_quarantines(&mut f);
    assert!(!f.taken && !f.retired, "the engine keeps every span");
}

#[test]
fn d2h_span_red_arm_a_take_on_the_items_landing_fails() {
    let mut f = SpanBatch::new(4, 12);
    f.take_on_items = true;
    let r = catch_unwind(AssertUnwindSafe(|| {
        d2h_span_taken_on_the_items_landing_fails(&mut f)
    }));
    assert!(r.is_err(), "the red arm must fail the rule");
    let mut f = SpanBatch::new(4, 12);
    d2h_span_taken_on_the_items_landing_fails(&mut f);
}
