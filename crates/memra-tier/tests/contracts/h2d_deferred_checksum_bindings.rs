//! Day-34 rule (WP-A, `DAY34.md` design K): the deferred H2D checksum. The CPU binding models the
//! copies' events as a flag, the views out as a count and the supplied digests per item.
//! `land_on_copy` is the red arm: a binding that lands a deferred item on its copy alone.
use super::conformance::*;
use super::support::*;
use memra_tier::contracts::*;
use std::panic::{AssertUnwindSafe, catch_unwind};

struct Deferred {
    land_on_copy: bool,
    completion: Completion,
    receipts: Vec<Vec<SegmentExpectation>>,
    copies_done: bool,
    views_out: usize,
    supplied: bool,
    returned: bool,
    retired_source: bool,
}
impl Deferred {
    fn new(items: usize) -> Self {
        let ticket = TransferTicket {
            issuer: 97,
            sequence: 1,
            epochs: epochs(),
        };
        Self {
            land_on_copy: false,
            completion: completion(ticket, items),
            receipts: expected(items),
            copies_done: false,
            views_out: 0,
            supplied: false,
            returned: false,
            retired_source: false,
        }
    }
}
impl DeferredChecksumFixture for Deferred {
    fn submit(&mut self) -> TransferTicket {
        self.completion.producer_done = false;
        for item in &mut self.completion.items {
            for s in &mut item.segments {
                s.status = ItemStatus::Pending;
                s.producer_done = false;
                s.checksum = None;
            }
        }
        self.completion.ticket
    }
    fn defer(&mut self, _ticket: &TransferTicket) -> Result<usize> {
        self.views_out = self.completion.items.len();
        Ok(self.views_out)
    }
    fn copies_complete(&mut self) {
        self.copies_done = true;
    }
    fn poll(&mut self, _ticket: &TransferTicket) -> Result<Completion> {
        let landed = self.copies_done && (self.supplied || self.land_on_copy);
        for item in &mut self.completion.items {
            for s in &mut item.segments {
                if self.copies_done {
                    s.status = ItemStatus::Complete;
                    s.producer_done = landed;
                }
            }
        }
        self.completion.producer_done = landed;
        Ok(self.completion.clone())
    }
    fn supply(&mut self, _ticket: &TransferTicket, wrong: bool) -> Result<()> {
        if self.returned {
            return Err(Error::AlreadyReleased);
        }
        for (k, (item, want)) in self
            .completion
            .items
            .iter_mut()
            .zip(&self.receipts)
            .enumerate()
        {
            let mut d = want[0].checksum;
            if wrong && k == 0 {
                d[0] ^= 0xff;
            }
            item.segments[0].checksum = Some(d);
        }
        self.views_out = 0;
        self.returned = true;
        self.supplied = true;
        Ok(())
    }
    fn supply_again(&mut self, ticket: &TransferTicket) -> Result<()> {
        self.supply(ticket, false)
    }
    fn receipts(&self) -> Vec<Vec<SegmentExpectation>> {
        self.receipts.clone()
    }
    fn retire_source(&mut self, _ticket: &TransferTicket) -> Result<()> {
        if self.views_out > 0 || !self.completion.producer_done {
            return Err(Error::Busy);
        }
        self.retired_source = true;
        Ok(())
    }
    fn retire(&mut self, _ticket: &TransferTicket) -> Result<()> {
        if self.views_out > 0 || !self.retired_source {
            return Err(Error::Busy);
        }
        Ok(())
    }
}

#[test]
fn h2d_deferred_checksum_lands_with_its_digests_and_passes_the_gate() {
    let mut f = Deferred::new(6);
    h2d_deferred_checksum_lands_with_its_digests(&mut f);
    assert!(f.retired_source);
}

#[test]
fn h2d_deferred_checksum_mismatch_is_refused_corrupt() {
    let mut f = Deferred::new(6);
    h2d_deferred_checksum_mismatch_is_corrupt(&mut f);
}

#[test]
fn h2d_deferred_checksum_red_arm_a_copy_alone_fails() {
    let mut f = Deferred::new(6);
    f.land_on_copy = true;
    let r = catch_unwind(AssertUnwindSafe(|| {
        h2d_deferred_checksum_copy_alone_is_not_landed(&mut f)
    }));
    assert!(r.is_err(), "the red arm must fail the rule");
    let mut f = Deferred::new(6);
    h2d_deferred_checksum_copy_alone_is_not_landed(&mut f);
}
