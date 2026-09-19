//! CPU-only retirement fixture; never evidence of actual CUDA fence completion.
use crate::contracts::{Epochs, Error, Result};
use crate::pool::PinnedLease;

/// Real adapters must source each signal from disk/DMA/consumer owners.
pub struct FakeTransferLease {
    lease: Option<PinnedLease>,
    epoch: Epochs,
    disk_done: bool,
    dma_done: bool,
    consumer_done: bool,
    graph_done: bool,
    cancelled: bool,
}
impl FakeTransferLease {
    pub fn new(lease: PinnedLease, epoch: Epochs) -> Self {
        Self {
            lease: Some(lease),
            epoch,
            disk_done: false,
            dma_done: false,
            consumer_done: false,
            graph_done: false,
            cancelled: false,
        }
    }
    pub fn cancel(&mut self) {
        self.cancelled = true;
    }
    pub fn publication_allowed(&self) -> bool {
        !self.cancelled && self.disk_done && self.dma_done
    }
    pub fn observe(
        &mut self,
        epoch: Epochs,
        disk_done: bool,
        dma_done: bool,
        consumer_done: bool,
        graph_done: bool,
    ) -> Result<()> {
        if epoch != self.epoch {
            return Err(Error::StaleEpoch);
        }
        self.disk_done |= disk_done;
        self.dma_done |= dma_done;
        self.consumer_done |= consumer_done;
        self.graph_done |= graph_done;
        Ok(())
    }
    pub fn retired(&self) -> bool {
        self.disk_done && self.dma_done && self.consumer_done && self.graph_done
    }
    pub fn release(&mut self) -> Result<()> {
        if !self.retired() {
            return Err(Error::NotReady);
        }
        self.lease.take().ok_or(Error::UnknownTicket)?.release();
        Ok(())
    }
}
impl Drop for FakeTransferLease {
    fn drop(&mut self) {
        if let Some(lease) = self.lease.take() {
            // Losing state is not proof that any asynchronous use ended.
            lease.quarantine();
        }
    }
}
