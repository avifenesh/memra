//! Bounded CPU positioned-read workers, generalized from spill_pread's ownership pattern.
//! No CUDA calls. The existing expert adapter is deliberately unchanged before freeze.
pub mod direct;
pub mod retirement;
pub mod transfer;
use crate::contracts::{Epochs, Error, Result, TransferTicket};
use crate::pool::PinnedLease;
use std::collections::HashMap;
use std::io;
use std::sync::{Arc, Mutex, mpsc};
use std::thread::{self, JoinHandle};

pub trait ReadAt: Send + Sync + 'static {
    fn read_at(&self, dst: &mut [u8], offset: u64) -> io::Result<usize>;
}
#[cfg(unix)]
impl ReadAt for std::fs::File {
    fn read_at(&self, dst: &mut [u8], offset: u64) -> io::Result<usize> {
        std::os::unix::fs::FileExt::read_at(self, dst, offset)
    }
}
/// Exact-loop behavior matches spill_pread::pread_exact_at; errors retain OS code.
pub fn read_exact_at(source: &dyn ReadAt, dst: &mut [u8], offset: u64) -> Result<u64> {
    offset
        .checked_add(dst.len() as u64)
        .ok_or(Error::Overflow)?;
    let mut done = 0;
    while done < dst.len() {
        match source.read_at(&mut dst[done..], offset + done as u64) {
            Ok(0) => {
                return Err(Error::ShortIo {
                    expected: dst.len() as u64,
                    actual: done as u64,
                });
            }
            Ok(n) if n <= dst.len() - done => done += n,
            Ok(_) => return Err(Error::InvalidLayout),
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e.into()),
        }
    }
    Ok(done as u64)
}
pub struct ReadRequest {
    pub source: Arc<dyn ReadAt>,
    pub offset: u64,
    pub lease: PinnedLease,
    pub epochs: Epochs,
}
pub struct SubmitError {
    pub error: Error,
    pub request: ReadRequest,
}
impl std::fmt::Debug for SubmitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.error.fmt(f)
    }
}
pub struct ReadCompletion {
    pub ticket: TransferTicket,
    pub lease: PinnedLease,
    pub result: Result<u64>,
    pub cancelled: bool,
}
struct Job {
    ticket: TransferTicket,
    request: ReadRequest,
}
pub struct BoundedReader {
    sender: Option<mpsc::SyncSender<Job>>,
    receiver: mpsc::Receiver<ReadCompletion>,
    threads: Vec<JoinHandle<()>>,
    pending: HashMap<TransferTicket, bool>,
    limit: usize,
    next_id: u64,
    issuer: u64,
}
impl BoundedReader {
    pub fn new(workers: usize, max_inflight: usize) -> Result<Self> {
        if workers == 0 || max_inflight == 0 || workers > max_inflight {
            return Err(Error::InvalidLayout);
        }
        let (sender, requests) = mpsc::sync_channel::<Job>(max_inflight);
        let (completions, receiver) = mpsc::channel();
        let requests = Arc::new(Mutex::new(requests));
        let mut threads: Vec<JoinHandle<()>> = Vec::new();
        for index in 0..workers {
            let requests = requests.clone();
            let completions = completions.clone();
            match thread::Builder::new()
                .name(format!("memra-tier-io-{index}"))
                .spawn(move || {
                    loop {
                        let job = match requests.lock().expect("reader queue poisoned").recv() {
                            Ok(job) => job,
                            Err(_) => break,
                        };
                        let Job {
                            ticket,
                            mut request,
                        } = job;
                        // A malicious/faulty fake backend panic must still return ownership.
                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            request.lease.read_into(|dst| {
                                read_exact_at(request.source.as_ref(), dst, request.offset)
                            })
                        }))
                        .unwrap_or(Err(Error::Quarantined));
                        if completions
                            .send(ReadCompletion {
                                ticket,
                                lease: request.lease,
                                result,
                                cancelled: false,
                            })
                            .is_err()
                        {
                            break;
                        }
                    }
                }) {
                Ok(t) => threads.push(t),
                Err(e) => {
                    drop(sender);
                    for t in threads {
                        let _ = t.join();
                    }
                    return Err(e.into());
                }
            }
        }
        drop(completions);
        Ok(Self {
            sender: Some(sender),
            receiver,
            threads,
            pending: HashMap::new(),
            limit: max_inflight,
            next_id: 1,
            issuer: crate::contracts::DeviceOwner::new(0).issuer(),
        })
    }
    /// Accepted set is bounded until COMPLETIONS are consumed, not merely dequeued by workers.
    #[allow(clippy::result_large_err)] // Rejection returns owned backing without an allocation.
    pub fn submit(
        &mut self,
        request: ReadRequest,
    ) -> std::result::Result<TransferTicket, SubmitError> {
        let reject = |error, request| Err(SubmitError { error, request });
        if self.pending.len() >= self.limit {
            return reject(Error::Capacity, request);
        }
        let Some(next) = self.next_id.checked_add(1) else {
            return reject(Error::Overflow, request);
        };
        let ticket = TransferTicket {
            issuer: self.issuer,
            sequence: self.next_id,
            epochs: request.epochs,
        };
        let Some(sender) = &self.sender else {
            return reject(Error::NotReady, request);
        };
        match sender.try_send(Job { ticket, request }) {
            Ok(()) => {
                self.pending.insert(ticket, false);
                self.next_id = next;
                Ok(ticket)
            }
            Err(mpsc::TrySendError::Full(job)) => reject(Error::Capacity, job.request),
            Err(mpsc::TrySendError::Disconnected(job)) => reject(Error::NotReady, job.request),
        }
    }
    pub fn cancel(&mut self, ticket: TransferTicket) -> Result<()> {
        *self.pending.get_mut(&ticket).ok_or(Error::UnknownTicket)? = true;
        Ok(()) // no buffer/slot is freed here
    }
    fn accept(&mut self, mut completion: ReadCompletion) -> Result<ReadCompletion> {
        completion.cancelled = self
            .pending
            .remove(&completion.ticket)
            .ok_or(Error::UnknownTicket)?;
        if completion.cancelled {
            completion.result = Err(Error::Cancelled);
        }
        Ok(completion)
    }
    pub fn poll(&mut self) -> Result<Option<ReadCompletion>> {
        match self.receiver.try_recv() {
            Ok(c) => self.accept(c).map(Some),
            Err(mpsc::TryRecvError::Empty) => Ok(None),
            Err(mpsc::TryRecvError::Disconnected) if self.pending.is_empty() => Ok(None),
            Err(_) => Err(Error::Quarantined),
        }
    }
    /// CPU fixture/CLI only. Engine scheduler uses nonblocking poll.
    pub fn wait(&mut self) -> Result<ReadCompletion> {
        if self.pending.is_empty() {
            return Err(Error::UnknownTicket);
        }
        let c = self.receiver.recv().map_err(|_| Error::Quarantined)?;
        self.accept(c)
    }
}
impl Drop for BoundedReader {
    fn drop(&mut self) {
        self.sender.take();
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
        // Receiver stays alive through join; queued owned completions drop afterward.
    }
}
