//! Day-21 rule (WP-A, memra#536 Move 2 slice 2): the readiness of a D2D restore into a borrowed
//! destination from a pinned borrowed source. The CPU binding models the copy stream's completion
//! event as a flag, the reader (owner) stream as a log of installed waits and issued primes, and
//! the device LRU's pin on the source as a counter; a D2D item has no host bytes, so its completion
//! carries no checksum and `Completion::require` refuses it `Corrupt`. `prime_early` is the red
//! arm: a caller that re-admits its request and primes on the landed copy BEFORE the reader wait is
//! installed, the shape the worker change must never take.
//!
//! Day 23 (the draft-bearing restore, `d2d_restore_draft_ready`): the same fixture with a class per
//! item (`Role::Key`/`Role::Value` trunk rows, `Role::Draft` draft rows), a `witnessed` mode in which
//! a landed item carries its receipt term (the modelled destination digest equal to the source's),
//! and the deferred prime's first draft-head read on the reader stream. `draft_read_early` is the
//! red arm: a session built over the landed scratch before the wait.
use super::conformance::*;
use super::support::*;
use memra_tier::contracts::*;
use std::panic::{AssertUnwindSafe, catch_unwind};

const ITEM_EVENT: u64 = 1;

struct RestoreBatch {
    prime_early: bool,
    /// Day 23: the class of every item, in item order (trunk K/V, then the draft plane's K/V).
    classes: Vec<Role>,
    /// Day 23: a landed item carries its witnessed receipt term (the slice-3 program modelled).
    witnessed: bool,
    /// Day 23: one entry per draft-head read the reader stream issued: ordered at issue or not.
    draft_reads: Vec<bool>,
    draft_read_early: bool,
    event_done: bool,
    completion: Option<Completion>,
    bytes: Vec<u64>,
    reader_waits: Vec<u64>,
    /// One entry per prime the reader stream issued: ordered at issue or not.
    primes: Vec<bool>,
    fence_sequence: u64,
    pins: u32,
    retired: bool,
    acknowledged: bool,
}
impl RestoreBatch {
    fn new(items: usize) -> Self {
        Self {
            prime_early: false,
            classes: (0..items)
                .map(|i| if i % 2 == 0 { Role::Key } else { Role::Value })
                .collect(),
            witnessed: false,
            draft_reads: vec![],
            draft_read_early: false,
            event_done: false,
            completion: None,
            bytes: (0..items).map(|i| 96 + 16 * i as u64).collect(),
            reader_waits: vec![],
            primes: vec![],
            fence_sequence: 0,
            pins: 0,
            retired: false,
            acknowledged: false,
        }
    }
    /// Day 23: `trunk` trunk items followed by the draft plane's K and V items, witnessed.
    fn draft_bearing(trunk: usize) -> Self {
        let mut f = Self::new(trunk + 2);
        f.classes = (0..trunk)
            .map(|i| if i % 2 == 0 { Role::Key } else { Role::Value })
            .chain([Role::Draft, Role::Draft])
            .collect();
        f.witnessed = true;
        f
    }
    /// The modelled receipt term of item `i`: its class and byte count folded into 32 bytes (the
    /// source digest; the destination digest equals it once the copy landed).
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
    fn entry(&mut self, ticket: &TransferTicket) -> Result<&mut Completion> {
        match &mut self.completion {
            Some(c) if &c.ticket == ticket => Ok(c),
            _ => Err(Error::UnknownTicket),
        }
    }
    fn fence_segments(&mut self) {
        self.fence_sequence += 1;
        let fence = FenceId {
            issuer: 95,
            owner: 0,
            generation: epochs().dst_gen,
            sequence: self.fence_sequence,
        };
        let c = self.completion.as_mut().unwrap();
        for item in &mut c.items {
            for s in &mut item.segments {
                s.consumer_fenced = true;
                s.consumer_fence = Some(fence);
            }
        }
        c.consumer_fenced = true;
    }
}
impl D2dRestoreFixture for RestoreBatch {
    fn submit(&mut self) -> TransferTicket {
        let ticket = TransferTicket {
            issuer: 95,
            sequence: 1,
            epochs: epochs(),
        };
        // The caller pins the source, then every copy is issued on the copy stream behind the
        // producer event: pending, no checksum (no host bytes), UNFENCED (rule 3: the consumer is
        // the reader stream and its wait is installed at the settle).
        self.pins += 1;
        let mut c = completion(ticket, self.bytes.len());
        c.producer_done = false;
        c.consumer_fenced = false;
        for item in &mut c.items {
            for s in &mut item.segments {
                s.status = ItemStatus::Pending;
                s.producer_done = false;
                s.checksum = None;
                s.valid_bytes = 0;
                s.consumer_fenced = false;
                s.consumer_fence = None;
            }
        }
        self.completion = Some(c);
        if self.prime_early {
            // The red arm: the request is re-admitted on issue and primes at once.
            self.prime_issues();
        }
        if self.draft_read_early {
            // Day 23's red arm: a session built over the scratch at issue reads draft rows at once.
            self.draft_read_issues();
        }
        ticket
    }
    fn poll(&mut self, ticket: &TransferTicket) -> Result<Completion> {
        let done = self.event_done;
        let bytes = self.bytes.clone();
        let digests: Vec<Option<Digest>> = (0..bytes.len())
            .map(|i| self.witnessed.then(|| self.digest_of(i)))
            .collect();
        let c = self.entry(ticket)?;
        if done {
            for ((item, bytes), digest) in c.items.iter_mut().zip(bytes).zip(digests) {
                for s in &mut item.segments {
                    s.status = ItemStatus::Complete;
                    s.producer_done = true;
                    s.valid_bytes = bytes;
                    // slice 2: no witnessed checksum term for a D2D; day 23 models slice 3's
                    // witness (destination digest = source digest) in the witnessed mode.
                    s.checksum = digest;
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
    fn ready(&mut self, ticket: &TransferTicket) -> Result<bool> {
        // The caller's predicate: landed AND every item fenced by an installed reader wait.
        let landed = self.landed(ticket)?;
        let c = self.poll(ticket)?;
        Ok(landed
            && c.consumer_fenced
            && c.items.iter().filter(|i| i.accepted).all(|i| {
                i.segments
                    .iter()
                    .all(|s| s.consumer_fenced && s.consumer_fence.is_some())
            }))
    }
    fn require_receipt(&mut self, ticket: &TransferTicket) -> Result<()> {
        let c = self.poll(ticket)?;
        let expected: Vec<Vec<SegmentExpectation>> = self
            .bytes
            .iter()
            .enumerate()
            .map(|(i, &b)| {
                vec![SegmentExpectation {
                    valid_bytes: b,
                    io_bytes: b,
                    // The source digest is the expectation (slice 3); receipt-less: never met.
                    checksum: if self.witnessed {
                        self.digest_of(i)
                    } else {
                        [0; 32]
                    },
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
            return Err(Error::WrongOwner); // a restore records no consumer fence of its own
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
    fn install_reader_wait(&mut self, ticket: &TransferTicket) {
        self.entry(ticket).unwrap();
        self.reader_waits.push(ITEM_EVENT);
        self.fence_segments();
    }
    fn prime_issues(&mut self) {
        self.primes.push(self.reader_waits.contains(&ITEM_EVENT));
    }
    fn primes_ordered(&self) -> bool {
        self.primes.iter().all(|ordered| *ordered)
    }
    fn source_pinned(&self) -> bool {
        self.pins > 0
    }
    fn release_source(&mut self) {
        self.pins = self.pins.saturating_sub(1);
    }
    fn submitted_bytes(&self) -> Vec<u64> {
        self.bytes.clone()
    }
}

impl D2dDraftRestoreFixture for RestoreBatch {
    fn item_classes(&self) -> Vec<Role> {
        self.classes.clone()
    }
    fn draft_read_issues(&mut self) {
        self.draft_reads
            .push(self.reader_waits.contains(&ITEM_EVENT));
    }
    fn draft_reads_ordered(&self) -> bool {
        self.draft_reads.iter().all(|ordered| *ordered)
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

/// The schedule on a two-item and a one-item batch: nothing is ready before the event, a landed
/// copy is not ready without the reader wait, the installed wait makes it ready and orders the
/// prime, the ticket retires without a consumer fence and acknowledges, the pin outlives both.
#[test]
fn day21_d2d_restore_is_ready_only_after_the_landing_and_the_installed_reader_wait() {
    let mut f = RestoreBatch::new(2);
    d2d_restore_ready(&mut f);
    assert!(f.retired && f.acknowledged);
    assert_eq!(f.reader_waits, vec![ITEM_EVENT]);
    assert_eq!(f.primes, vec![true]);
    assert_eq!(f.pins, 0, "the pin was handed over after acknowledge");
    let mut one = RestoreBatch::new(1);
    d2d_restore_ready(&mut one);
    assert_eq!(one.primes, vec![true]);
    let mut early = RestoreBatch::new(2);
    d2d_restore_primed_before_its_wait_is_unordered(&mut early);
    assert_eq!(early.primes, vec![false]);
}

/// Red arm: a caller that re-admits its request and primes on the copy's ISSUE (before the wait,
/// before even the landing). The schedule fails, and this is the shape of its failure: the
/// request's first token is produced on a cache the copy stream is still writing, the
/// one-program law's forbidden half-restored prime.
#[test]
fn day21_red_arm_prime_before_the_reader_wait_fails_the_schedule() {
    let mut f = RestoreBatch::new(2);
    f.prime_early = true;
    let schedule = catch_unwind(AssertUnwindSafe(|| {
        d2d_restore_ready(&mut f);
    }));
    assert!(
        schedule.is_err(),
        "a prime before the installed reader wait must not pass the restore rule"
    );
    let mut f = RestoreBatch::new(2);
    f.prime_early = true;
    let ticket = f.submit();
    assert!(!f.primes_ordered(), "the early prime is unordered");
    assert!(!f.landed(&ticket).unwrap());
    f.copy_completes();
    f.install_reader_wait(&ticket);
    assert!(f.ready(&ticket).unwrap());
    assert!(
        !f.primes_ordered(),
        "the landing and a later wait do not order the early prime"
    );
}

/// The slice-2 receipt clause on its own: a restore completion has no checksum term, so the
/// host-contract gate refuses it before and after the event and after the wait.
#[test]
fn day21_d2d_restore_item_without_a_witnessed_checksum_is_refused_by_the_host_contract_gate() {
    let mut f = RestoreBatch::new(3);
    let ticket = f.submit();
    assert!(matches!(
        f.require_receipt(&ticket),
        Err(Error::Corrupt | Error::NotReady)
    ));
    f.copy_completes();
    f.install_reader_wait(&ticket);
    assert!(f.ready(&ticket).unwrap());
    assert_eq!(f.require_receipt(&ticket), Err(Error::Corrupt));
}

/// Day 23: the draft-bearing batch (16 trunk items and the draft plane's two, the 9B's `items=18`;
/// then 32 and two, the 27B's `items=34`): both classes land, are witnessed and are fenced by the
/// one install; the prime and the draft-head read after it are ordered; the pin outlives the ticket.
#[test]
fn day23_draft_bearing_restore_is_ready_for_both_classes_only_after_the_installed_wait() {
    for trunk in [16usize, 32] {
        let mut f = RestoreBatch::draft_bearing(trunk);
        d2d_restore_draft_ready(&mut f);
        assert!(f.retired && f.acknowledged);
        assert_eq!(
            f.reader_waits,
            vec![ITEM_EVENT],
            "one install fences both classes"
        );
        assert_eq!(f.primes, vec![true]);
        assert_eq!(f.draft_reads, vec![true]);
        assert_eq!(f.pins, 0, "the one pin was handed over after acknowledge");
        assert_eq!(
            f.classes.iter().filter(|r| **r == Role::Draft).count(),
            2,
            "the draft plane is its K and V items"
        );
        assert_eq!(f.classes.len(), trunk + 2);
    }
}

/// Day 23's red arm: a session built over the landed scratch before the installed wait (its first
/// draft-head read at issue) fails the schedule, and the read stays unordered under a later wait.
#[test]
fn day23_red_arm_draft_read_before_the_reader_wait_fails_the_schedule() {
    let mut f = RestoreBatch::draft_bearing(4);
    f.draft_read_early = true;
    let schedule = catch_unwind(AssertUnwindSafe(|| {
        d2d_restore_draft_ready(&mut f);
    }));
    assert!(
        schedule.is_err(),
        "a draft-head read before the installed reader wait must not pass the draft rule"
    );
    let mut early = RestoreBatch::draft_bearing(4);
    d2d_restore_draft_read_before_its_wait_is_unordered(&mut early);
    assert_eq!(early.draft_reads, vec![false]);
    assert!(
        early
            .ready(&TransferTicket {
                issuer: 95,
                sequence: 1,
                epochs: epochs(),
            })
            .unwrap()
    );
}

/// Day 23, the receipt clause over the draft class: a draft-bearing batch whose items land with no
/// witnessed term (the slice-2 shape) is refused by the host-contract gate after the landing and
/// after the wait, and `witnessed_classes` names none of them; the draft rule itself fails on it.
#[test]
fn day23_unwitnessed_draft_items_are_refused_by_the_host_contract_gate() {
    let mut f = RestoreBatch::draft_bearing(4);
    f.witnessed = false;
    let schedule = catch_unwind(AssertUnwindSafe(|| {
        d2d_restore_draft_ready(&mut f);
    }));
    assert!(
        schedule.is_err(),
        "a draft-bearing batch without the witnessed term does not pass the draft rule"
    );
    let mut f = RestoreBatch::draft_bearing(4);
    f.witnessed = false;
    let ticket = f.submit();
    f.copy_completes();
    assert!(f.landed(&ticket).unwrap());
    assert!(f.witnessed_classes(&ticket).unwrap().is_empty());
    f.install_reader_wait(&ticket);
    assert!(f.ready(&ticket).unwrap());
    assert_eq!(f.require_receipt(&ticket), Err(Error::Corrupt));
}
