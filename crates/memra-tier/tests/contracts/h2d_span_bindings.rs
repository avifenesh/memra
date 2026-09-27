//! Day-32 rule (WP-A, memra#536 Move 2 owed item 1, the H2D half): the typed f32 spans of a
//! promote batch. The CPU binding models the KV items' events and the spans' events as two flags,
//! the reader stream's wait as a third, the engine's ownership of each span as a slot, and every
//! read the reader stream issues as a record of whether the wait existed when it was issued.
//! `take_on_landing` is the red arm: a binding that hands the spans back on the batch's landing
//! alone, before the reader wait, the shape the engine must never take.
use super::conformance::*;
use super::support::*;
use memra_tier::contracts::*;
use std::panic::{AssertUnwindSafe, catch_unwind};

struct SpanBatch {
    take_on_landing: bool,
    fail_next: bool,
    items_done: bool,
    spans_done: bool,
    waited: bool,
    quarantined: bool,
    completion: Option<Completion>,
    lens: Vec<u64>,
    attached: Option<Vec<u64>>,
    taken: bool,
    reads: Vec<bool>,
    retired: bool,
    acknowledged: bool,
}
impl SpanBatch {
    fn new(items: usize, spans: usize) -> Self {
        Self {
            take_on_landing: false,
            fail_next: false,
            items_done: false,
            spans_done: false,
            waited: false,
            quarantined: false,
            completion: Some(completion(
                TransferTicket {
                    issuer: 95,
                    sequence: 1,
                    epochs: epochs(),
                },
                items,
            )),
            lens: (0..spans).map(|i| 4096 + 64 * i as u64).collect(),
            attached: None,
            taken: false,
            reads: Vec::new(),
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
        self.items_done && (self.attached.is_none() || self.spans_done)
    }
}
impl H2dSpanFixture for SpanBatch {
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
        if self.quarantined {
            return Err((Error::Quarantined, n));
        }
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
    fn install_reader_wait(&mut self, ticket: &TransferTicket) -> Result<()> {
        self.live(ticket)?;
        if self.quarantined {
            return Err(Error::Quarantined);
        }
        self.waited = true;
        Ok(())
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
        if !self.landed() || !(self.waited || self.take_on_landing) {
            return Err(Error::NotReady);
        }
        self.taken = true;
        Ok(lens)
    }
    fn reader_reads(&mut self) {
        if self.taken {
            self.reads.push(self.waited);
        }
    }
    fn readers_ordered(&self) -> bool {
        self.reads.iter().all(|&waited| waited)
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
fn h2d_span_batch_lands_with_its_ticket_and_is_read_behind_the_wait() {
    let mut f = SpanBatch::new(4, 12);
    h2d_span_batch(&mut f);
    assert!(f.acknowledged);
    assert_eq!(f.reads, vec![true], "one read, issued after the wait");
    // The KV items' host-contract gate is unchanged by the spans.
    let c = f.completion.as_ref().unwrap();
    assert!(c.require(&c.ticket, &expected(4), false).is_ok());
}

#[test]
fn h2d_span_enqueue_failure_quarantines_the_ticket() {
    let mut f = SpanBatch::new(4, 12);
    h2d_span_enqueue_failure_quarantines(&mut f);
    assert!(!f.taken && !f.retired, "the engine keeps every span");
}

#[test]
fn h2d_span_red_arm_an_owner_read_before_the_wait_fails() {
    let mut f = SpanBatch::new(4, 12);
    f.take_on_landing = true;
    let r = catch_unwind(AssertUnwindSafe(|| {
        h2d_span_read_before_its_wait_is_unordered(&mut f)
    }));
    assert!(r.is_err(), "the red arm must fail the rule");
    assert_eq!(
        f.reads,
        vec![false],
        "the red arm's read ran before the wait"
    );
    let mut f = SpanBatch::new(4, 12);
    h2d_span_read_before_its_wait_is_unordered(&mut f);
    assert_eq!(f.reads, vec![true]);
}

/// Day 33, rule 6: the filled batch. `copy_before_fill` is the red arm: a binding whose copies run
/// when offered, before the fill.
struct FilledBatch {
    copy_before_fill: bool,
    items_done: bool,
    filled: bool,
    copied: bool,
    read_unfilled: bool,
    waited: bool,
    taken: bool,
    completion: Completion,
    lens: Vec<u64>,
}
impl FilledBatch {
    fn new(items: usize, spans: usize) -> Self {
        Self {
            copy_before_fill: false,
            items_done: false,
            filled: false,
            copied: false,
            read_unfilled: false,
            waited: false,
            taken: false,
            completion: completion(
                TransferTicket {
                    issuer: 96,
                    sequence: 1,
                    epochs: epochs(),
                },
                items,
            ),
            lens: (0..spans).map(|i| 8192 + 32 * i as u64).collect(),
        }
    }
    fn landed(&self) -> bool {
        self.items_done && self.copied
    }
}
impl H2dFillFixture for FilledBatch {
    fn submit(&mut self) -> TransferTicket {
        self.completion.producer_done = false;
        self.completion.ticket
    }
    fn attach_filled(
        &mut self,
        ticket: &TransferTicket,
    ) -> std::result::Result<(), (Error, usize)> {
        if ticket != &self.completion.ticket {
            return Err((Error::UnknownTicket, self.lens.len()));
        }
        Ok(())
    }
    fn poll(&mut self, _ticket: &TransferTicket) -> Result<Completion> {
        self.completion.producer_done = self.landed();
        Ok(self.completion.clone())
    }
    fn install_reader_wait(&mut self, _ticket: &TransferTicket) -> Result<()> {
        self.waited = true;
        Ok(())
    }
    fn take_spans(&mut self, _ticket: &TransferTicket) -> Result<Vec<u64>> {
        if self.taken {
            return Err(Error::AlreadyReleased);
        }
        if !self.landed() || !self.waited {
            return Err(Error::NotReady);
        }
        self.taken = true;
        Ok(self.lens.clone())
    }
    fn items_complete(&mut self) {
        self.items_done = true;
    }
    fn fill_runs(&mut self) {
        self.filled = true;
    }
    fn copies_run(&mut self) {
        // Stream order: the copies wait for the fill; the red arm runs them anyway.
        if self.filled || self.copy_before_fill {
            if !self.filled {
                self.read_unfilled = true;
            }
            self.copied = true;
        }
    }
    fn copies_read_filled(&self) -> bool {
        !self.read_unfilled
    }
    fn span_bytes(&self) -> Vec<u64> {
        self.lens.clone()
    }
}

#[test]
fn h2d_span_fill_runs_ahead_of_every_copy() {
    let mut f = FilledBatch::new(4, 12);
    h2d_span_fill_ordered_before_its_copy(&mut f);
    assert!(f.taken && !f.read_unfilled);
}

#[test]
fn h2d_span_red_arm_a_copy_before_its_fill_fails() {
    let mut f = FilledBatch::new(4, 12);
    f.copy_before_fill = true;
    let r = catch_unwind(AssertUnwindSafe(|| {
        h2d_span_fill_ordered_before_its_copy(&mut f)
    }));
    assert!(r.is_err(), "the red arm must fail the rule");
    assert!(f.read_unfilled, "the red arm's copy read unfilled bytes");
}
