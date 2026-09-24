//! Day-22 rule (WP-A, memra#536 Move 2, slice 3): the receipt term of the two D2D classes. Additive
//! and unversioned beside the frozen schedules, in the day-11, day-19, day-20 and day-21 shape:
//! every v1 to v1.3 schedule is byte-identical and `WIRE_VERSION` stays 1. Native bindings read
//! real device digests; CPU bindings model the witness as data. Nothing here qualifies a backend by
//! itself.
//!
//! What a D2D item lacked after slices 1 and 2: `checksum = None`, so `Completion::require` refused
//! it `Corrupt` by construction and `capture_landed` / `restore_landed` proved only that the
//! completion EVENT fired. The receipt witnesses the bytes: a digest of the SOURCE span taken behind
//! the producer fence (it reads what the producer finished writing) and a digest of the
//! DESTINATION taken on the copy's stream after the copy and before the completion event, both read
//! by the host only after that event is observed complete. The item's `checksum` is the destination
//! digest and its expectation is the source digest, so the existing gate clause
//! (`s.checksum != Some(e.checksum) => Corrupt`) is the comparison, unchanged.
//!
//! The program (`receipt_digest`): `memra_tier::contracts::checksum` is SHA-256 over host bytes and
//! has no device twin, so the receipt is a copy-stream REDUCTION with this function as its CPU
//! oracle. The span's bytes are little-endian u64 words `w_j` (the last partial word zero-padded);
//! four lanes `L_l = sum_j mix64(w_j + (j + 1) * C_l)` in wrapping arithmetic; the digest is
//! `le_bytes(L_l ^ mix64(n + C_l))` for l = 0..4 with `n` the byte count. Wrapping sums are
//! order-independent, so the device's block and atomic order cannot move the value. This is a
//! receipt over KV BYTES, never a numeric program over tokens: no forward, no encoding, no tolerance.
//!
//! Rule, the witnessed receipt:
//!
//! 1. Not landed. Before every item's completion event is observed complete there is no receipt
//!    (`NotReady`) and a publication is refused `NotReady`.
//! 2. Landed, matching. The receipt names BOTH digests per item and they are equal; `require` is
//!    `Ok`; the caller publishes exactly once; `retire(ticket, None)`, then `acknowledge`.
//! 3. Landed, receipt-less. An item with no witnessed checksum is refused `Corrupt` (the slice-1
//!    clause, kept by name); the publish is refused; the caller latches; the ticket still retires
//!    and acknowledges (nothing leaked: the copy has landed).
//! 4. Landed, mismatching. The receipt names both digests and they DIFFER (the early reader of the
//!    `d2d-delay` fault, or a corrupted copy); `require` is `Corrupt`; the publish is refused; the
//!    caller latches; the ticket retires and acknowledges. Nothing is published or primed on.
use crate::contracts::*;

/// The four lane constants of the receipt program (fixed; part of the program's identity).
pub const RECEIPT_LANES: [u64; 4] = [
    0x9E37_79B9_7F4A_7C15,
    0xC2B2_AE3D_27D4_EB4F,
    0x1656_67B1_9E37_79F9,
    0x27D4_EB2F_1656_67C5,
];

/// The splitmix64 finalizer, the program's word mixer.
pub const fn mix64(mut z: u64) -> u64 {
    z ^= z >> 30;
    z = z.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z ^= z >> 27;
    z = z.wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    z
}

/// The four lane sums over a byte span: the value the device kernel accumulates.
pub fn receipt_lanes(bytes: &[u8]) -> [u64; 4] {
    let mut lanes = [0u64; 4];
    for (j, chunk) in bytes.chunks(8).enumerate() {
        let mut word = [0u8; 8];
        word[..chunk.len()].copy_from_slice(chunk);
        let v = u64::from_le_bytes(word);
        let j1 = j as u64 + 1;
        for (lane, c) in lanes.iter_mut().zip(RECEIPT_LANES) {
            *lane = lane.wrapping_add(mix64(v.wrapping_add(j1.wrapping_mul(c))));
        }
    }
    lanes
}

/// The digest from the lanes and the byte count: what the host computes from the kernel's lanes.
pub fn receipt_digest_from_lanes(lanes: [u64; 4], n: u64) -> Digest {
    let mut d = [0u8; 32];
    for (i, (lane, c)) in lanes.iter().zip(RECEIPT_LANES).enumerate() {
        let v = lane ^ mix64(n.wrapping_add(c));
        d[i * 8..i * 8 + 8].copy_from_slice(&v.to_le_bytes());
    }
    d
}

/// The CPU oracle of the D2D receipt digest over a byte span.
pub fn receipt_digest(bytes: &[u8]) -> Digest {
    receipt_digest_from_lanes(receipt_lanes(bytes), bytes.len() as u64)
}

/// One item's witnessed receipt: the source digest (behind the producer fence) and the destination
/// digest (after the copy), `None` where no witness exists.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReceiptTerm {
    pub source: Digest,
    pub destination: Option<Digest>,
}

/// The receipt fixture: one D2D batch (either class). `poll`, `landed`, `retire`, `retired` and
/// `acknowledge` are the engine's own answers; `receipt` is the engine's per-item receipt
/// (`NotReady` before the landing); `require_receipt` runs `Completion::require` as the host-contract
/// gate would (`device = true`); `publish` is the CALLER's publication (into the device prefix index
/// for a capture, the hand-over of the ready cache for a restore), which asks `require_receipt`
/// first, refuses on its error and LATCHES the route on `Corrupt`; `latched` reads that latch;
/// `copy_completes` fires every item's completion event.
pub trait D2dReceiptFixture {
    fn submit(&mut self) -> TransferTicket;
    fn poll(&mut self, ticket: &TransferTicket) -> Result<Completion>;
    fn landed(&mut self, ticket: &TransferTicket) -> Result<bool>;
    fn receipt(&mut self, ticket: &TransferTicket) -> Result<Vec<ReceiptTerm>>;
    fn require_receipt(&mut self, ticket: &TransferTicket) -> Result<()>;
    fn publish(&mut self, ticket: &TransferTicket) -> Result<()>;
    fn latched(&self) -> bool;
    fn retire(&mut self, ticket: &TransferTicket, consumer_done: Option<FenceId>) -> Result<()>;
    fn retired(&mut self, ticket: &TransferTicket) -> Result<bool>;
    fn acknowledge(&mut self, ticket: &TransferTicket) -> Result<()>;
    fn copy_completes(&mut self);
    /// The bytes each item was submitted with, in item order.
    fn submitted_bytes(&self) -> Vec<u64>;
}

fn landing<F: D2dReceiptFixture>(f: &mut F) -> (TransferTicket, Vec<u64>) {
    let ticket = f.submit();
    let expected = f.submitted_bytes();
    assert!(!expected.is_empty(), "a D2D batch has at least one item");
    // 1. Running: no receipt, no publication.
    assert!(!f.landed(&ticket).unwrap(), "a running copy is not landed");
    assert!(
        matches!(f.receipt(&ticket), Err(Error::NotReady)),
        "no receipt before the completion event"
    );
    assert!(
        matches!(f.publish(&ticket), Err(Error::NotReady)),
        "a publish before the event is refused NotReady"
    );
    assert!(!f.latched(), "a NotReady refusal is not a latch");
    f.copy_completes();
    let c = f.poll(&ticket).unwrap();
    assert!(c.producer_done);
    assert!(f.landed(&ticket).unwrap(), "every event observed complete");
    let delivered: Vec<u64> = c
        .items
        .iter()
        .filter(|i| i.accepted)
        .flat_map(|i| i.segments.iter().map(|s| s.valid_bytes))
        .collect();
    assert_eq!(delivered, expected, "each item delivered exactly its bytes");
    (ticket, expected)
}

/// The schedule, the matching arm (rules 1 and 2). The fixture starts before `submit`.
pub fn d2d_receipt_witnessed<F: D2dReceiptFixture>(f: &mut F) {
    let (ticket, expected) = landing(f);
    // 2. The receipt names both digests per item and they agree; the gate opens; publish once.
    let terms = f.receipt(&ticket).unwrap();
    assert_eq!(terms.len(), expected.len(), "one receipt term per item");
    for (i, t) in terms.iter().enumerate() {
        assert_eq!(
            t.destination,
            Some(t.source),
            "item {i}: the destination digest witnesses the source digest"
        );
    }
    let c = f.poll(&ticket).unwrap();
    for (item, t) in c.items.iter().zip(&terms) {
        assert_eq!(
            item.segments[0].checksum, t.destination,
            "the item's checksum IS the destination digest"
        );
    }
    f.require_receipt(&ticket)
        .expect("a matching receipt opens the host-contract gate");
    f.publish(&ticket).unwrap();
    assert!(!f.latched());
    assert!(
        matches!(f.publish(&ticket), Err(Error::AlreadyReleased)),
        "publication happens exactly once"
    );
    f.retire(&ticket, None).unwrap();
    assert!(f.retired(&ticket).unwrap());
    f.acknowledge(&ticket).unwrap();
}

/// The schedule, the refusal arms (rules 3 and 4): a fixture whose witness is missing (receipt-less)
/// or whose early reader read stale bytes (mismatching). Nothing publishes; the caller latches; the
/// ticket leaves cleanly.
pub fn d2d_receipt_refused<F: D2dReceiptFixture>(f: &mut F) {
    let (ticket, expected) = landing(f);
    let terms = f.receipt(&ticket).unwrap();
    assert_eq!(terms.len(), expected.len(), "one receipt term per item");
    assert!(
        terms.iter().any(|t| t.destination != Some(t.source)),
        "a refused receipt has at least one item without a matching witness"
    );
    assert_eq!(
        f.require_receipt(&ticket),
        Err(Error::Corrupt),
        "a missing or mismatching witness is refused Corrupt"
    );
    assert!(!f.latched(), "nothing latched before the caller's publish");
    assert_eq!(f.publish(&ticket), Err(Error::Corrupt));
    assert!(f.latched(), "the caller latches on a Corrupt receipt");
    assert!(
        matches!(f.publish(&ticket), Err(Error::Corrupt)),
        "a latched route never publishes the item"
    );
    // The copy landed, so the ticket leaves cleanly: retired and acknowledged, nothing leaked.
    f.retire(&ticket, None).unwrap();
    assert!(f.retired(&ticket).unwrap());
    f.acknowledge(&ticket).unwrap();
}
