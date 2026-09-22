//! Day-19 rule 3 (WP-A, memra#536 Move 1): the reader fence of an H2D issued off the owner
//! stream. The CPU binding models two streams as logs: the copy stream owns the item's completion
//! event; the reader stream holds installed waits and issued reads. A read is ordered only if a
//! wait on the item's event existed on the reader stream when it was issued. `flag_only` is the
//! red arm: a settle that sets `consumer_fenced` without installing the wait (a flag, not an
//! event), the shape the engine change must never take.
use super::conformance::*;
use super::support::*;
use memra_tier::contracts::*;
use std::panic::{AssertUnwindSafe, catch_unwind};

const ITEM_EVENT: u64 = 1;

struct TwoStreams {
    install: ReaderWaitInstall,
    flag_only: bool,
    copy_event_done: bool,
    /// The consumer fence recorded on the reader stream and whether its event has fired.
    consumer: Option<(FenceId, bool)>,
    reader_waits: Vec<u64>,
    /// One entry per read the reader stream issued: ordered at issue or not.
    reads: Vec<bool>,
    fence_sequence: u64,
    completion: Option<Completion>,
    published: bool,
    retired: bool,
    acknowledged: bool,
}
impl TwoStreams {
    fn new(install: ReaderWaitInstall) -> Self {
        Self {
            install,
            flag_only: false,
            copy_event_done: false,
            consumer: None,
            reader_waits: vec![],
            reads: vec![],
            fence_sequence: 0,
            completion: None,
            published: false,
            retired: false,
            acknowledged: false,
        }
    }
    fn next_fence(&mut self, generation: u64) -> FenceId {
        self.fence_sequence += 1;
        FenceId {
            issuer: 91,
            owner: 0,
            generation,
            sequence: self.fence_sequence,
        }
    }
    fn fence_segments(&mut self) {
        let fence = self.next_fence(epochs().dst_gen);
        let c = self.completion.as_mut().unwrap();
        for item in &mut c.items {
            for s in &mut item.segments {
                s.consumer_fenced = true;
                s.consumer_fence = Some(fence);
            }
        }
        c.consumer_fenced = true;
    }
    fn entry(&mut self, ticket: &TransferTicket) -> Result<&mut Completion> {
        match &mut self.completion {
            Some(c) if &c.ticket == ticket => Ok(c),
            _ => Err(Error::UnknownTicket),
        }
    }
}
impl ReaderFenceFixture for TwoStreams {
    fn submit(&mut self) -> TransferTicket {
        let ticket = TransferTicket {
            issuer: 91,
            sequence: 1,
            epochs: epochs(),
        };
        // The copy is issued on the copy stream: its completion event is pending, its checksum
        // unknown, and the item is fenced only under the at-submit install.
        let mut c = completion(ticket, 1);
        c.producer_done = false;
        c.consumer_fenced = false;
        for item in &mut c.items {
            for s in &mut item.segments {
                s.status = ItemStatus::Pending;
                s.producer_done = false;
                s.checksum = None;
                s.consumer_fenced = false;
                s.consumer_fence = None;
            }
        }
        self.completion = Some(c);
        if self.install == ReaderWaitInstall::AtSubmit {
            self.reader_waits.push(ITEM_EVENT);
            self.fence_segments();
        }
        ticket
    }
    fn poll(&mut self, ticket: &TransferTicket) -> Result<Completion> {
        let done = self.copy_event_done;
        let c = self.entry(ticket)?;
        if done {
            for item in &mut c.items {
                for s in &mut item.segments {
                    s.status = ItemStatus::Complete;
                    s.producer_done = true;
                    s.checksum = Some(checksum(&bytes(3)));
                }
            }
            c.producer_done = true;
        }
        Ok(c.clone())
    }
    fn ready_view(&mut self, ticket: &TransferTicket) -> Result<()> {
        // The engine's own gate: `Completion::require` with `device = true`, as
        // `DeviceOwner::ready_view` runs it, fences included.
        let c = self.poll(ticket)?;
        c.require(ticket, &expected(1), true)?;
        self.published = true;
        Ok(())
    }
    fn take_destination(&mut self, ticket: &TransferTicket) -> Result<()> {
        self.ready_view(ticket)
    }
    fn record_consumer(&mut self, ticket: &TransferTicket) -> Result<FenceId> {
        self.entry(ticket)?;
        if !self.published || self.retired {
            return Err(Error::NotReady);
        }
        let f = self.next_fence(ticket.epochs.dst_gen);
        self.consumer = Some((f, false));
        Ok(f)
    }
    fn retire(&mut self, ticket: &TransferTicket, consumer_done: Option<FenceId>) -> Result<()> {
        let c = self.poll(ticket)?;
        if self.retired {
            return Ok(());
        }
        if !c.producer_done {
            return Err(Error::Busy);
        }
        match consumer_done {
            Some(f) => {
                let (actual, done) = self.consumer.ok_or(Error::WrongOwner)?;
                if actual != f {
                    return Err(Error::WrongOwner);
                }
                if !done {
                    return Err(Error::Busy);
                }
            }
            None if self.published => return Err(Error::Busy),
            None => {}
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
        self.copy_event_done = true;
    }
    fn install_reader_wait(&mut self, ticket: &TransferTicket) {
        self.entry(ticket).unwrap();
        if self.install == ReaderWaitInstall::AtSubmit {
            return; // installed inside `submit`
        }
        if !self.flag_only {
            self.reader_waits.push(ITEM_EVENT);
        }
        self.fence_segments();
    }
    fn reader_issues(&mut self) {
        self.reads.push(self.reader_waits.contains(&ITEM_EVENT));
    }
    fn reader_completes(&mut self) {
        if let Some((_, done)) = &mut self.consumer {
            *done = true;
        }
    }
    fn readers_ordered(&self) -> bool {
        self.reads.iter().all(|ordered| *ordered)
    }
}

/// The engine's day-18 program: the owner (reader) stream waits on the item's event at submit.
/// `consumer_fenced` is true from submission; nothing is readable until the copy lands.
#[test]
fn day19_h2d_reader_fence_at_submit_is_the_engines_day18_program() {
    let mut f = TwoStreams::new(ReaderWaitInstall::AtSubmit);
    h2d_reader_fence(&mut f, ReaderWaitInstall::AtSubmit);
    assert_eq!(f.reader_waits, vec![ITEM_EVENT]);
    assert_eq!(f.reads, vec![true]);
}

/// The owed program: the wait installed at the settle, after the completion is observed and
/// before `ready_view`, as a wait on the reader stream. `consumer_fenced` turns true there and
/// only there; a landed copy without it is not readable; a read before it is unordered.
#[test]
fn day19_h2d_reader_fence_at_settle_with_a_wait_on_the_reader_stream() {
    let mut f = TwoStreams::new(ReaderWaitInstall::AtSettle);
    h2d_reader_fence(&mut f, ReaderWaitInstall::AtSettle);
    assert_eq!(f.reader_waits, vec![ITEM_EVENT]);
    assert_eq!(f.reads, vec![true]);
    let mut early = TwoStreams::new(ReaderWaitInstall::AtSettle);
    h2d_reader_issued_before_its_wait_is_unordered(&mut early);
    assert_eq!(early.reads, vec![false]);
}

/// Red arm: the settle sets `consumer_fenced` and installs NO wait on the reader stream (a flag
/// where the contract wants an event). The schedule fails, and this is the shape of its failure:
/// the flag publishes the destination and the reader's very next read is unordered.
#[test]
fn day19_red_arm_settle_time_flag_without_a_reader_wait_fails_the_schedule() {
    let mut f = TwoStreams::new(ReaderWaitInstall::AtSettle);
    f.flag_only = true;
    let schedule = catch_unwind(AssertUnwindSafe(|| {
        h2d_reader_fence(&mut f, ReaderWaitInstall::AtSettle);
    }));
    assert!(
        schedule.is_err(),
        "a settle-time flag without a reader wait must not pass rule 3"
    );
    let mut f = TwoStreams::new(ReaderWaitInstall::AtSettle);
    f.flag_only = true;
    let ticket = f.submit();
    f.copy_completes();
    assert!(matches!(f.ready_view(&ticket), Err(Error::NotReady)));
    f.install_reader_wait(&ticket);
    f.ready_view(&ticket).unwrap(); // the flag published it
    assert!(
        f.reader_waits.is_empty(),
        "no wait exists on the reader stream"
    );
    f.reader_issues();
    assert!(
        !f.readers_ordered(),
        "the published destination is read unordered"
    );
}
