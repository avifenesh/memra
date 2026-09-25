//! Day-38 rule (WP-A, `DAY38.md` design G): the device receipt of a D2H batch. The CPU binding models
//! each item's source bytes, the copy (landed bytes, one flipped under `flip`), the copies' events and
//! the batch's receipt event as flags, and the receipt as the program over the source as the digest
//! read it. `land_on_copy` is the red arm: a binding that lands an item on its copy alone.
use super::conformance::*;
use super::support::*;
use memra_tier::contracts::*;
use std::panic::{AssertUnwindSafe, catch_unwind};

struct DeviceReceipt {
    land_on_copy: bool,
    completion: Completion,
    sources: Vec<Vec<u8>>,
    landed: Vec<Vec<u8>>,
    copies_done: bool,
    receipt_done: bool,
}
impl DeviceReceipt {
    fn new(items: usize) -> Self {
        let ticket = TransferTicket {
            issuer: 98,
            sequence: 1,
            epochs: epochs(),
        };
        Self {
            land_on_copy: false,
            completion: completion(ticket, items),
            sources: (0..items)
                .map(|i| (0..64 + i).map(|k| (k * 7 + i) as u8).collect())
                .collect(),
            landed: vec![],
            copies_done: false,
            receipt_done: false,
        }
    }
}
impl D2hDeviceReceiptFixture for DeviceReceipt {
    fn submit(&mut self, flip: bool) -> TransferTicket {
        self.completion.producer_done = false;
        for item in &mut self.completion.items {
            for s in &mut item.segments {
                s.status = ItemStatus::Pending;
                s.producer_done = false;
                s.checksum = None;
            }
        }
        // The digest read the sources as they are now; the copy reads them after the flip.
        self.landed = self.sources.clone();
        if flip {
            self.landed[0][5] ^= 0x40;
        }
        self.completion.ticket
    }
    fn copies_complete(&mut self) {
        self.copies_done = true;
    }
    fn receipt_complete(&mut self) {
        self.receipt_done = true;
    }
    fn poll(&mut self, _ticket: &TransferTicket) -> Result<Completion> {
        let landed = self.copies_done && (self.receipt_done || self.land_on_copy);
        for (i, item) in self.completion.items.iter_mut().enumerate() {
            for s in &mut item.segments {
                if self.copies_done {
                    s.status = ItemStatus::Complete;
                    s.producer_done = landed;
                    if landed {
                        s.checksum = Some(checksum(&self.sources[i]));
                    }
                }
            }
        }
        self.completion.producer_done = landed;
        Ok(self.completion.clone())
    }
    fn source_digest(&self, item: usize) -> Digest {
        checksum(&self.sources[item])
    }
    fn landed_digest(&self, item: usize) -> Digest {
        checksum(&self.landed[item])
    }
    fn items(&self) -> usize {
        self.sources.len()
    }
}

#[test]
fn d2h_device_receipt_lands_with_the_source_digest_and_the_witness_agrees() {
    let mut f = DeviceReceipt::new(6);
    d2h_device_receipt_lands_with_the_source_digest(&mut f);
}

#[test]
fn d2h_device_receipt_flip_is_seen_by_the_witness() {
    let mut f = DeviceReceipt::new(6);
    d2h_device_receipt_flip_differs_at_the_witness(&mut f);
}

#[test]
fn d2h_device_receipt_red_arm_a_copy_alone_fails() {
    let mut f = DeviceReceipt::new(6);
    f.land_on_copy = true;
    let r = catch_unwind(AssertUnwindSafe(|| {
        d2h_device_receipt_copy_alone_is_not_landed(&mut f)
    }));
    assert!(r.is_err(), "the red arm must fail the rule");
    let mut f = DeviceReceipt::new(6);
    d2h_device_receipt_copy_alone_is_not_landed(&mut f);
}
