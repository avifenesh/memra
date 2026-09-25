//! Day-40 rule (WP-A, `DAY40.md` design S), revised on day 42 (`DAY42.md` design S2): the device
//! receipt of the recurrent f32 spans. The CPU binding models each span's device source, its
//! landed bytes (one flipped under the D2H flip), the resident plane the promote fills from (one
//! flipped under the H2D flip), the events as flags, and every digest as the four-lane program
//! `receipt_digest`. The red arms: `publish_early` (a caller that publishes a demote before its
//! span receipt is observed), `publish_blind` (a caller that ignores the digests), and
//! `h2d_land_on_copies` (a binding that lands an H2D span batch on its copies alone).
use super::conformance::*;
use memra_tier::contracts::*;
use std::panic::{AssertUnwindSafe, catch_unwind};

struct Spans {
    publish_early: bool,
    publish_blind: bool,
    h2d_land_on_copies: bool,
    sources: Vec<Vec<u8>>,
    landed: Vec<Vec<u8>>,
    destinations: Vec<Vec<u8>>,
    h2d: bool,
    copies_done: bool,
    sealed: bool,
    receipt_done: bool,
}
impl Spans {
    fn new(spans: usize) -> Self {
        Self {
            publish_early: false,
            publish_blind: false,
            h2d_land_on_copies: false,
            sources: (0..spans)
                .map(|i| (0..96 + 8 * i).map(|k| (k * 13 + i) as u8).collect())
                .collect(),
            landed: vec![],
            destinations: vec![],
            h2d: false,
            copies_done: false,
            sealed: false,
            receipt_done: false,
        }
    }
    fn ticket(&self) -> TransferTicket {
        TransferTicket {
            issuer: 97,
            sequence: 1 + self.h2d as u64,
            epochs: Epochs {
                state: 0,
                src_gen: 1,
                dst_gen: 1,
            },
        }
    }
}
impl SpanReceiptFixture for Spans {
    fn submit_d2h(&mut self, flip: bool) -> TransferTicket {
        self.h2d = false;
        self.copies_done = false;
        self.sealed = false;
        self.receipt_done = false;
        self.landed = self.sources.clone();
        if flip {
            self.landed[0][5] ^= 0x40;
        }
        self.ticket()
    }
    fn submit_h2d(&mut self, flip: bool) -> TransferTicket {
        self.h2d = true;
        self.copies_done = false;
        self.receipt_done = false;
        // The fill reads the resident planes (the landed bytes kept by the demote).
        self.destinations = self.landed.clone();
        if flip {
            self.destinations[0][5] ^= 0x40;
        }
        self.ticket()
    }
    fn copies_complete(&mut self) {
        self.copies_done = true;
    }
    fn d2h_landed(&mut self) -> bool {
        !self.h2d && self.copies_done
    }
    fn seal(&mut self) {
        self.sealed = true;
    }
    fn receipt_complete(&mut self) {
        self.receipt_done = true;
    }
    fn h2d_landed(&mut self) -> bool {
        self.h2d && self.copies_done && (self.receipt_done || self.h2d_land_on_copies)
    }
    fn d2h_pairs(&mut self) -> Option<Vec<(Digest, Digest)>> {
        (self.sealed && self.receipt_done).then(|| {
            self.sources
                .iter()
                .zip(&self.landed)
                .map(|(s, l)| (receipt_digest(s), receipt_digest(l)))
                .collect()
        })
    }
    fn h2d_digests(&mut self) -> Vec<Digest> {
        self.destinations
            .iter()
            .map(|d| receipt_digest(d))
            .collect()
    }
    fn publish_demote(&mut self, pairs: Option<&[(Digest, Digest)]>) -> bool {
        match pairs {
            None => self.publish_early,
            Some(pairs) => self.publish_blind || pairs.iter().all(|(s, l)| s == l),
        }
    }
    fn publish_promote(&mut self, kept: &[Digest], landed: &[Digest]) -> bool {
        self.publish_blind
            || (kept.len() == landed.len() && kept.iter().zip(landed).all(|(k, l)| k == l))
    }
    fn spans(&self) -> usize {
        self.sources.len()
    }
}

#[test]
fn span_receipt_clean_demote_and_promote_publish() {
    let mut f = Spans::new(5);
    span_receipt_lands_with_its_digests(&mut f);
}

#[test]
fn span_receipt_landed_flip_is_refused() {
    let mut f = Spans::new(5);
    span_receipt_landed_flip_refuses_the_demote(&mut f);
}

#[test]
fn span_receipt_resident_flip_is_refused() {
    let mut f = Spans::new(5);
    span_receipt_resident_flip_refuses_the_promote(&mut f);
}

#[test]
fn span_receipt_red_arms_fail_the_rule() {
    let mut f = Spans::new(5);
    f.publish_early = true;
    let r = catch_unwind(AssertUnwindSafe(|| {
        span_receipt_lands_with_its_digests(&mut f)
    }));
    assert!(
        r.is_err(),
        "a caller that publishes before the span receipt is observed must fail the rule"
    );
    let mut f = Spans::new(5);
    f.h2d_land_on_copies = true;
    let r = catch_unwind(AssertUnwindSafe(|| {
        span_receipt_lands_with_its_digests(&mut f)
    }));
    assert!(
        r.is_err(),
        "an H2D batch landed on its copies alone must fail the rule"
    );
    let mut f = Spans::new(5);
    f.publish_blind = true;
    let r = catch_unwind(AssertUnwindSafe(|| {
        span_receipt_landed_flip_refuses_the_demote(&mut f)
    }));
    assert!(
        r.is_err(),
        "a caller that publishes a differing demote must fail the rule"
    );
    let mut f = Spans::new(5);
    f.publish_blind = true;
    let r = catch_unwind(AssertUnwindSafe(|| {
        span_receipt_resident_flip_refuses_the_promote(&mut f)
    }));
    assert!(
        r.is_err(),
        "a caller that publishes a differing promote must fail the rule"
    );
}
