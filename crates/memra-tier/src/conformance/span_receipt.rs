//! Day-40 rule (WP-A, memra#536 Move 2 owed item 1's strong form, `DAY40.md` design S), revised on
//! day 42 for design S2 (`DAY42.md` sections 1 and 1a): the device receipt of the recurrent f32
//! spans, both directions. Additive and unversioned beside the frozen schedules: every v1 to v1.3
//! schedule is byte-identical and `WIRE_VERSION` stays 1. Nothing here qualifies a backend by
//! itself.
//!
//! Rule, the span receipt (the program is the four-lane `receipt_digest`):
//!
//! 1. The D2H landing is the copies'. A D2H span batch lands when every span's copy is observed
//!    complete; its SOURCE digests were taken before the copies, so they are complete then. Its
//!    LANDED digests are taken after the landing (the seal) and read once the receipt's event is
//!    observed.
//! 2. The demote's witness. The caller publishes a demote only after its span receipt is observed,
//!    only if every span's landed digest equals its source digest, and keeps each source digest
//!    with the published image.
//! 3. The H2D landing includes its receipt. An H2D span batch lands only when every span's copy
//!    AND the batch's receipt (each span's DESTINATION digest, taken after its copy) are observed.
//! 4. The promote's witness. The caller publishes a promote only if every span's destination
//!    digest equals the source digest kept from that plane's demote; a byte that changed anywhere
//!    between the demote's source and the promote's destination makes them differ.
//!
//! The red arms: a caller that publishes a demote before its span receipt is observed, a caller
//! that publishes a span whose two digests differ, and a binding that lands an H2D span batch on
//! its copies alone.
use crate::contracts::*;

/// The fixture. `submit_d2h` issues a D2H span batch (with `flip`, one byte of span 0's landed
/// bytes changes after its copy and before its landing); `copies_complete` fires the span copies'
/// events; `d2h_landed` is the D2H batch's landing; `seal` enqueues the landed digests;
/// `receipt_complete` fires the pending receipt's event (the seal's, or the H2D batch's);
/// `d2h_pairs` the (source, landed) digests per span once the receipt is observed (`None` before).
/// `submit_h2d` issues the promote of the same planes (with `flip`, one byte of span 0's resident
/// bytes changes before the fill); `h2d_landed` and `h2d_digests` its landing and destination
/// digests. `publish_demote` and `publish_promote` are the caller's gates under test.
pub trait SpanReceiptFixture {
    fn submit_d2h(&mut self, flip: bool) -> TransferTicket;
    fn submit_h2d(&mut self, flip: bool) -> TransferTicket;
    fn copies_complete(&mut self);
    fn d2h_landed(&mut self) -> bool;
    fn seal(&mut self);
    fn receipt_complete(&mut self);
    fn h2d_landed(&mut self) -> bool;
    fn d2h_pairs(&mut self) -> Option<Vec<(Digest, Digest)>>;
    fn h2d_digests(&mut self) -> Vec<Digest>;
    fn publish_demote(&mut self, pairs: Option<&[(Digest, Digest)]>) -> bool;
    fn publish_promote(&mut self, kept: &[Digest], landed: &[Digest]) -> bool;
    fn spans(&self) -> usize;
}

/// The clean demote and promote: the D2H batch lands on its copies; no publication before the
/// receipt is observed; every pair equal, the demote published; the H2D batch lands only with its
/// receipt, its destination digests equal the kept sources, the promote published.
pub fn span_receipt_lands_with_its_digests<F: SpanReceiptFixture>(f: &mut F) {
    let _ = f.submit_d2h(false);
    assert!(!f.d2h_landed(), "no landing before the copies");
    f.copies_complete();
    assert!(f.d2h_landed(), "a D2H span batch lands on its copies");
    assert!(f.d2h_pairs().is_none(), "no pairs before the seal");
    assert!(
        !f.publish_demote(None),
        "no demote publishes before its span receipt is observed"
    );
    f.seal();
    assert!(
        f.d2h_pairs().is_none(),
        "no pairs before the receipt's event"
    );
    assert!(!f.publish_demote(None));
    f.receipt_complete();
    let pairs = f.d2h_pairs().expect("the receipt observed");
    assert_eq!(pairs.len(), f.spans());
    for (i, (src, landed)) in pairs.iter().enumerate() {
        assert_eq!(src, landed, "span {i}: landed equals source");
    }
    assert!(
        f.publish_demote(Some(&pairs)),
        "a witnessed demote publishes"
    );
    let kept: Vec<Digest> = pairs.iter().map(|(s, _)| *s).collect();
    let _ = f.submit_h2d(false);
    f.copies_complete();
    assert!(
        !f.h2d_landed(),
        "an H2D span batch does not land on its copies alone"
    );
    f.receipt_complete();
    assert!(f.h2d_landed());
    let landed = f.h2d_digests();
    assert_eq!(landed, kept, "the destinations are the demote's sources");
    assert!(
        f.publish_promote(&kept, &landed),
        "a witnessed promote publishes"
    );
}

/// Rule 2's refusal: a byte flipped after the copy and before the landing makes span 0's pair
/// differ, and the caller does not publish the demote.
pub fn span_receipt_landed_flip_refuses_the_demote<F: SpanReceiptFixture>(f: &mut F) {
    let _ = f.submit_d2h(true);
    f.copies_complete();
    assert!(f.d2h_landed());
    f.seal();
    f.receipt_complete();
    let pairs = f.d2h_pairs().expect("the receipt observed");
    assert_ne!(
        pairs[0].0, pairs[0].1,
        "span 0's landed digest shows the flip"
    );
    for (src, landed) in &pairs[1..] {
        assert_eq!(src, landed);
    }
    assert!(
        !f.publish_demote(Some(&pairs)),
        "a demote whose span differs is refused"
    );
}

/// Rule 4's refusal: a resident byte flipped between the demote and the promote's fill makes span
/// 0's destination digest differ from its kept source, and the caller does not publish the promote.
pub fn span_receipt_resident_flip_refuses_the_promote<F: SpanReceiptFixture>(f: &mut F) {
    let _ = f.submit_d2h(false);
    f.copies_complete();
    f.seal();
    f.receipt_complete();
    let pairs = f.d2h_pairs().expect("the receipt observed");
    assert!(f.publish_demote(Some(&pairs)));
    let kept: Vec<Digest> = pairs.iter().map(|(s, _)| *s).collect();
    let _ = f.submit_h2d(true);
    f.copies_complete();
    f.receipt_complete();
    let landed = f.h2d_digests();
    assert_ne!(
        landed[0], kept[0],
        "span 0's destination shows the resident flip"
    );
    assert!(
        !f.publish_promote(&kept, &landed),
        "a promote whose span differs is refused"
    );
}
