use super::*;

impl<S: BackgroundStore> CpuTransfers<S> {
    /// Enable before submitting any ticket. Root metadata/admission/file open stay
    /// on this owner; only retained file descriptors and CPU slots reach workers.
    /// No device handles, governor or publication rights cross the channel.
    pub fn enable_background(&mut self, workers: usize) -> Result<()> {
        if !self.entries.is_empty() || self.background.is_some() {
            return Err(Error::Busy);
        }
        self.background = Some(Background {
            reader: BoundedReader::new(workers, self.limit)?,
            pending: HashMap::new(),
            prepare: S::prepare_extent,
            release: S::release_prepared,
        });
        Ok(())
    }
}
fn finish(e: &mut Entry) {
    e.driven = e
        .completion
        .items
        .iter()
        .all(|i| !i.accepted || i.segments.iter().all(|s| s.producer_done));
    e.completion.producer_done = e.driven;
}
fn failed(s: &mut SegmentCompletion, error: Error) {
    s.status = if error == Error::Cancelled {
        ItemStatus::Cancelled
    } else {
        ItemStatus::Failed
    };
    s.error = Some(error);
    s.producer_done = true;
}
impl<S: ObjectStore> CpuTransfers<S> {
    pub(super) fn drive_background(&mut self, ticket: &TransferTicket) -> Result<()> {
        let e = self.entries.get_mut(ticket).ok_or(Error::UnknownTicket)?;
        let bg = self.background.as_mut().ok_or(Error::Unsupported)?;
        for (i, slot) in e.ops.iter_mut().enumerate() {
            let Some(op) = slot.take() else {
                continue;
            };
            let segment = &mut e.completion.items[i].segments[0];
            if segment.producer_done {
                *slot = Some(op);
                continue;
            }
            if e.cancelled {
                failed(segment, Error::Cancelled);
                *slot = Some(op);
                continue;
            }
            let prepared = match (bg.prepare)(
                &mut self.store,
                &op.object,
                op.chunk,
                op.destination.valid_bytes() as u64,
                &self.request,
            ) {
                Ok(prepared) => prepared,
                Err(error) => {
                    failed(segment, error);
                    *slot = Some(op);
                    continue;
                }
            };
            let ReadPlan {
                object,
                chunk,
                destination,
                epochs,
            } = op;
            e.extents.push(prepared.lease); // charge survives completion/cancel until explicit retirement
            match bg.reader.submit(ReadRequest {
                source: prepared.source,
                offset: 0,
                lease: destination,
                epochs,
            }) {
                Ok(worker_ticket) => {
                    bg.pending
                        .insert(worker_ticket, (*ticket, i, object, chunk));
                }
                Err(rejected) => {
                    failed(segment, rejected.error);
                    *slot = Some(ReadPlan {
                        object,
                        chunk,
                        destination: rejected.request.lease,
                        epochs,
                    });
                }
            }
        }
        finish(e);
        Ok(())
    }
    /// Drain at most the configured in-flight bound, without blocking. Completion
    /// harvesting/publication occurs only on the caller (future CUDA owner).
    pub fn progress(&mut self) -> Result<usize> {
        let Some(bg) = self.background.as_mut() else {
            return Ok(0);
        };
        let mut count = 0;
        while count < self.limit {
            let Some(done) = bg.reader.poll()? else {
                break;
            };
            let (ticket, item, object, chunk) = bg
                .pending
                .remove(&done.ticket)
                .ok_or(Error::UnknownTicket)?;
            let e = self.entries.get_mut(&ticket).ok_or(Error::UnknownTicket)?;
            let s = &mut e.completion.items[item].segments[0];
            if e.cancelled {
                failed(s, Error::Cancelled);
            } else {
                match done.result {
                    Ok(n) => {
                        s.status = ItemStatus::Complete;
                        s.valid_bytes = n;
                        s.io_bytes = (crate::object_store::padded_len(n as usize)?
                            + crate::object_store::ALIGNMENT)
                            as u64;
                        s.checksum = Some(checksum(done.lease.bytes()?));
                        s.producer_done = true;
                    }
                    Err(error) => failed(s, error),
                }
            }
            e.ops[item] = Some(ReadPlan {
                object,
                chunk,
                destination: done.lease,
                epochs: ticket.epochs,
            });
            finish(e);
            count += 1;
        }
        Ok(count)
    }
}
