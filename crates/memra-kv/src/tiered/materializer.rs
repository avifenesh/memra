//! Native q8_0 K / q5_1 V operand mapping. No codec, alternate KV format or attention math.
//! The CPU binding resolves real retained opaque bytes through DeviceOwner. CUDA binding
//! must replace NativeKvImage with the existing KvLayer buffers on its owner thread;
//! the geometry/ordering validation here is shared, but no CUDA path is installed.
use memra_tier::contracts::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
static NEXT_MATERIALIZER: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Debug)]
pub struct NativeGeometry {
    pub group: u32,
    pub owner: u32,
    pub tokens: u64,
    pub k_token_bytes: u64,
    pub v_token_bytes: u64,
}
impl NativeGeometry {
    /// Capture the EXISTING attention operands, not a newly selectable format.
    pub fn from_layer(layer: &crate::KvLayer, group: u32, owner: u32) -> Result<Self> {
        if layer.ring.is_some() {
            return Err(Error::Unsupported);
        }
        if !layer.kv_dim_k.is_multiple_of(32)
            || !layer.kv_dim_v.is_multiple_of(32)
            || layer.k_tok_bytes != layer.kv_dim_k / 32 * 34
            || layer.v_tok_bytes != layer.kv_dim_v / 32 * 24
        {
            return Err(Error::InvalidLayout);
        }
        Ok(Self {
            group,
            owner,
            tokens: layer.len as u64,
            k_token_bytes: layer.k_tok_bytes as u64,
            v_token_bytes: layer.v_tok_bytes as u64,
        })
    }
    pub fn validate(&self, b: &StateBundle) -> Result<()> {
        b.validate()?;
        if self.tokens == 0
            || self.k_token_bytes == 0
            || self.v_token_bytes == 0
            || !self.k_token_bytes.is_multiple_of(34)
            || !self.v_token_bytes.is_multiple_of(24)
            || b.id.start != 0
            || b.id.end != self.tokens
            || b.committed_high_water != self.tokens
            || b.id.group != self.group
            || b.id.owner != self.owner
        {
            return Err(Error::InvalidLayout);
        }
        if b.layout.requirements.len() != 2
            || b.layout.requirements.iter().any(|r| {
                r.group != self.group
                    || r.owner != self.owner
                    || r.pages != PageRequirement::AllPages
                    || !matches!(r.role, Role::Key | Role::Value)
            })
        {
            return Err(Error::Unsupported);
        }
        for (role, name, row) in [
            (Role::Key, b"q8_0".as_slice(), self.k_token_bytes),
            (Role::Value, b"q5_1".as_slice(), self.v_token_bytes),
        ] {
            let mut bytes = 0u64;
            for (page, s) in b
                .layout
                .segments
                .iter()
                .filter(|s| s.role == role)
                .enumerate()
            {
                if s.group != self.group
                    || s.owner != self.owner
                    || s.page != page as u64
                    || s.encoding.program != digest("native-kv-encoding", name)
                    || s.encoding.row_bytes != row
                    || s.valid_bytes % row != 0
                {
                    return Err(Error::InvalidLayout);
                }
                bytes = bytes.checked_add(s.valid_bytes).ok_or(Error::Overflow)?;
            }
            if bytes != self.tokens.checked_mul(row).ok_or(Error::Overflow)? {
                return Err(Error::Incomplete);
            }
        }
        Ok(())
    }
}
/// Retained CPU fixture backing; never advertised as CUDA memory. Layout order and padding
/// are preserved verbatim; native K/V ordering is checked before any operand is returned.
pub struct NativeKvImage {
    pub bundle: StateBundle,
    pub payloads: Vec<Vec<u8>>,
}
pub struct QwenOperands {
    materializer: u64,
    device: DeviceLease,
    ticket: TransferTicket,
    binding: u64,
}
impl QwenOperands {
    pub fn allocation_id(&self) -> u64 {
        self.device.allocation_id()
    }
}
pub struct QwenMaterializer<'a> {
    id: u64,
    owner: &'a DeviceOwner,
    program: ProgramIdentity,
    geometry: NativeGeometry,
    active: HashMap<u64, DeviceLease>,
    next: u64,
}
impl<'a> QwenMaterializer<'a> {
    pub fn new(owner: &'a DeviceOwner, program: ProgramIdentity, geometry: NativeGeometry) -> Self {
        Self {
            id: NEXT_MATERIALIZER
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
                .expect("materializer id exhausted"),
            owner,
            program,
            geometry,
            active: HashMap::new(),
            next: 0,
        }
    }
    /// CPU byte oracle: same [token,kv_head,dim] block order as KvLayer.k/v.
    /// No dequantization or chunk-wise softmax. Only live retained operands may be read.
    pub fn capture(&self, op: &QwenOperands) -> Result<(Vec<u8>, Vec<u8>)> {
        if op.materializer != self.id || !self.active.contains_key(&op.binding) {
            return Err(Error::AlreadyReleased);
        }
        let image = self.owner.resolve::<NativeKvImage>(&op.device)?;
        let mut k = vec![];
        let mut v = vec![];
        for (s, bytes) in image.bundle.layout.segments.iter().zip(&image.payloads) {
            let dest = match s.role {
                Role::Key => &mut k,
                Role::Value => &mut v,
                _ => return Err(Error::Unsupported),
            };
            dest.extend_from_slice(&bytes[..s.valid_bytes as usize]);
        }
        Ok((k, v))
    }
}
impl KvMaterializer for QwenMaterializer<'_> {
    type Operands = QwenOperands;
    fn materialize(
        &mut self,
        b: &StateBundle,
        program: &ProgramIdentity,
        ready: &ReadyView<'_>,
        current: Epochs,
    ) -> Result<QwenOperands> {
        ready.ticket().epochs.require(current)?;
        if program != &self.program || &b.program != program {
            return Err(Error::ProgramMismatch);
        }
        if b.id.epoch != current.state {
            return Err(Error::StaleEpoch);
        }
        self.geometry.validate(b)?;
        if ready.destination().device() != self.geometry.owner {
            return Err(Error::WrongOwner);
        }
        if ready.destination().bytes() < b.layout.storage_bytes()? {
            return Err(Error::Capacity);
        }
        let image = self.owner.resolve::<NativeKvImage>(ready.destination())?;
        if &image.bundle != b {
            return Err(Error::Conflict);
        }
        b.verify(&image.payloads)?;
        let next = self.next.checked_add(1).ok_or(Error::Overflow)?;
        let device = self.owner.retain(ready.destination())?;
        let pin = self.owner.retain(ready.destination())?;
        self.next = next;
        self.active.insert(next, pin);
        Ok(QwenOperands {
            materializer: self.id,
            device,
            ticket: ready.ticket(),
            binding: next,
        })
    }
    fn retire(&mut self, op: &QwenOperands, done: FenceId) -> Result<()> {
        if op.materializer != self.id {
            return Err(Error::ForeignLease);
        }
        if done.issuer != self.owner.issuer()
            || done.owner != self.geometry.owner
            || done.generation != op.ticket.epochs.dst_gen
        {
            return Err(Error::WrongOwner);
        }
        // Caller supplies an OBSERVED completed last-use fence; TransferEngine still
        // retains the destination binding through its disk/DMA/consumer/graph drain.
        self.active
            .remove(&op.binding)
            .ok_or(Error::AlreadyReleased)?;
        Ok(())
    }
}
impl Drop for QwenMaterializer<'_> {
    fn drop(&mut self) {
        // Lost consumer notification is unknown, not permission to recycle an address.
        for (_, lease) in self.active.drain() {
            std::mem::forget(lease);
        }
    }
}
