//! One coalesced row-window upload through the frozen transfer contract.
//! This is an ownership/publication adapter, not a CUDA implementation. The caller
//! supplies admitted host/device leases and drives its backend on the device owner.
use crate::contracts::*;

/// An accepted upload stays addressable after every error. There is deliberately
/// no Drop reclamation: unknown DMA/consumer status must stay backend-owned.
#[derive(Debug)]
pub struct RowUpload {
    ticket: TransferTicket,
    expected: [Vec<SegmentExpectation>; 1],
    allocation: u64,
    device: u32,
    published: bool,
    cancelled: bool,
}
impl RowUpload {
    #[allow(clippy::result_large_err)]
    pub fn submit<T: TransferEngine>(
        transfers: &mut T,
        op: CopyOp<T::Host>,
    ) -> std::result::Result<Self, Rejected<CopyOp<T::Host>>> {
        let validate = (|| {
            op.validate(CopyDirection::HostToDevice, op.epochs)?;
            if op.bytes != op.host.valid_bytes() || op.bytes != op.device.bytes() {
                return Err(Error::InvalidLayout);
            }
            let bytes = op.host.bytes()?;
            if bytes.len() as u64 != op.bytes {
                return Err(Error::InvalidLayout);
            }
            Ok(SegmentExpectation {
                valid_bytes: op.bytes,
                io_bytes: op.bytes,
                checksum: checksum(bytes),
            })
        })();
        let expected = match validate {
            Ok(expected) => [vec![expected]],
            Err(error) => return Err(Rejected { op, error }),
        };
        let allocation = op.device.allocation_id();
        let device = op.device.device();
        let ticket = transfers.h2d(op)?;
        Ok(Self {
            ticket,
            expected,
            allocation,
            device,
            published: false,
            cancelled: false,
        })
    }
    pub fn ticket(&self) -> TransferTicket {
        self.ticket
    }

    /// Producer completion alone is insufficient. Validate exact bytes/checksum,
    /// current epochs AND the backend's owner-sealed consumer-ready capability.
    /// Only then may the caller take the actual device buffer read by projection.
    pub fn publish<T: TransferEngine>(
        &mut self,
        transfers: &mut T,
        current: Epochs,
    ) -> Result<DeviceLease> {
        if self.cancelled {
            return Err(Error::Cancelled);
        }
        if self.published {
            return Err(Error::Busy);
        }
        self.ticket.epochs.require(current)?;
        transfers
            .poll(&self.ticket)?
            .require(&self.ticket, &self.expected, true)?;
        let ready = transfers.ready_view(&self.ticket, 0, current)?;
        if ready.ticket() != self.ticket
            || ready.destination().allocation_id() != self.allocation
            || ready.destination().device() != self.device
            || ready.destination().generation() != current.dst_gen
            || ready.destination().bytes() != self.expected[0][0].valid_bytes
        {
            return Err(Error::ForeignLease);
        }
        match transfers.take_destination(&self.ticket, 0, current)? {
            Destination::Device(device) => {
                self.published = true;
                Ok(device)
            }
            Destination::Host(_) => Err(Error::InvalidLayout),
        }
    }
    pub fn cancel<T: TransferEngine>(&mut self, transfers: &mut T) -> Result<CancelState> {
        let state = transfers.cancel(&self.ticket)?;
        if state == CancelState::PublicationRevoked {
            self.cancelled = true;
        }
        Ok(state)
    }
    /// Register the REAL last-use fence, then separately observe retirement.
    /// No acknowledge or budget release occurs merely because a wait was installed.
    pub fn retire<T: TransferEngine>(
        &self,
        transfers: &mut T,
        done: Option<FenceId>,
    ) -> Result<bool> {
        transfers.retire(&self.ticket, done)?;
        transfers.retired(&self.ticket)
    }
    pub fn acknowledge<T: TransferEngine>(&self, transfers: &mut T) -> Result<()> {
        if !transfers.retired(&self.ticket)? {
            return Err(Error::Busy);
        }
        transfers.acknowledge(&self.ticket)
    }
}
