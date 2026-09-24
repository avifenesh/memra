//! Day-35 rule (WP-A, `DAY35.md` design M1): the deferred D2H checksum. The CPU binding models the
//! copies' events as a flag, the views out as a count and the supplied digests per item.
//! `view_early` is the red arm (a view before the copy); `land_on_copy` the second red arm (a landing on
//! the copy alone).
use super::conformance::*;
use super::support::*;
use memra_tier::contracts::*;
use std::panic::{AssertUnwindSafe, catch_unwind};

struct Deferred {
    view_early: bool,
    land_on_copy: bool,
    completion: Completion,
    copies_done: bool,
    views_out: usize,
    supplied: bool,
    returned: bool,
    taken_back: bool,
}
impl Deferred {
    fn new(items: usize) -> Self {
        let ticket = TransferTicket {
            issuer: 98,
            sequence: 1,
            epochs: epochs(),
        };
        Self {
            view_early: false,
            land_on_copy: false,
            completion: completion(ticket, items),
            copies_done: false,
            views_out: 0,
            supplied: false,
            returned: false,
            taken_back: false,
        }
    }
}
impl D2hDeferredChecksumFixture for Deferred {
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
    fn views(&mut self, _ticket: &TransferTicket) -> Result<usize> {
        if !self.copies_done && !self.view_early {
            return Err(Error::NotReady);
        }
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
        let n = self.completion.items.len();
        for k in 0..n {
            let mut d = self.digest_for(k);
            if wrong && k == 0 {
                d[0] ^= 0xff;
            }
            self.completion.items[k].segments[0].checksum = Some(d);
        }
        self.views_out = 0;
        self.returned = true;
        self.supplied = true;
        Ok(())
    }
    fn supply_again(&mut self, ticket: &TransferTicket) -> Result<()> {
        self.supply(ticket, false)
    }
    fn digest_for(&self, item: usize) -> [u8; 32] {
        let mut d = [0u8; 32];
        d[0] = 0x5a;
        d[1] = item as u8;
        d
    }
    fn take_back(&mut self, _ticket: &TransferTicket) -> Result<()> {
        if self.views_out > 0 || !self.completion.producer_done {
            return Err(Error::Busy);
        }
        self.taken_back = true;
        Ok(())
    }
    fn retire(&mut self, _ticket: &TransferTicket) -> Result<()> {
        if self.views_out > 0 || !self.taken_back {
            return Err(Error::Busy);
        }
        Ok(())
    }
}

#[test]
fn d2h_deferred_checksum_lands_with_its_digests_as_its_receipts() {
    let mut f = Deferred::new(6);
    d2h_deferred_checksum_lands_with_its_digests(&mut f);
    assert!(f.taken_back);
}

#[test]
fn d2h_deferred_checksum_wrong_digest_becomes_the_receipt() {
    let mut f = Deferred::new(6);
    d2h_deferred_checksum_wrong_digest_is_the_receipt(&mut f);
}

#[test]
fn d2h_deferred_checksum_red_arm_a_view_before_the_copy_fails() {
    let mut f = Deferred::new(6);
    f.view_early = true;
    let r = catch_unwind(AssertUnwindSafe(|| {
        d2h_deferred_checksum_no_view_before_the_copy(&mut f)
    }));
    assert!(r.is_err(), "the red arm must fail the rule");
    let mut f = Deferred::new(6);
    d2h_deferred_checksum_no_view_before_the_copy(&mut f);
}

#[test]
fn d2h_deferred_checksum_red_arm_a_copy_alone_fails() {
    let mut f = Deferred::new(6);
    f.land_on_copy = true;
    let r = catch_unwind(AssertUnwindSafe(|| {
        d2h_deferred_checksum_copy_alone_is_not_landed(&mut f)
    }));
    assert!(r.is_err(), "the red arm must fail the rule");
    let mut f = Deferred::new(6);
    d2h_deferred_checksum_copy_alone_is_not_landed(&mut f);
}
