//! Identity-only sidecar for the EXISTING HostPrefixEntry API. Legacy entries are
//! unbound and can still use legacy synchronous promotion, but cannot issue tier leases.
//! A metadata lease is NOT a payload/DMA lease: KvBacking must separately retain the
//! existing host image and governor pin before submitting any asynchronous transfer.
use memra_tier::contracts::*;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

#[derive(Default)]
pub struct IdentitySlot(Option<Arc<Proof>>);
struct Proof {
    bundle: StateBundle,
    model_generation: Arc<()>,
    live: AtomicBool,
}
pub struct IdentityLease(Arc<Proof>);
impl IdentitySlot {
    /// Call only after native complete-state capture and physical immutable sealing.
    /// No handoff import is automatically trusted as generic v1.
    pub fn bind(
        &mut self,
        bundle: StateBundle,
        tokens: &[u32],
        program: &ProgramIdentity,
        expected_layout: &RecordLayout,
        model_generation: Arc<()>,
    ) -> Result<()> {
        bundle.require(program, expected_layout, 0, tokens.len() as u64)?;
        if bundle.kind != StateKind::ImmutablePrefix
            || bundle.id.start != 0
            || bundle.committed_high_water != tokens.len() as u64
        {
            return Err(Error::InvalidLayout);
        }
        let expected = KvBlockId::new(
            program,
            bundle.id.parent,
            tokens,
            0,
            bundle.id.group,
            bundle.id.owner,
            0,
        )?;
        if bundle.id != expected {
            return Err(Error::ProgramMismatch);
        }
        self.invalidate();
        self.0 = Some(Arc::new(Proof {
            bundle,
            model_generation,
            live: AtomicBool::new(true),
        }));
        Ok(())
    }
    pub fn lease(&self, program: &ProgramIdentity, generation: &Arc<()>) -> Result<IdentityLease> {
        let proof = self.0.as_ref().ok_or(Error::Unsupported)?;
        let lease = IdentityLease(proof.clone());
        lease.require(program, generation)?;
        Ok(lease)
    }
    pub fn invalidate(&mut self) {
        if let Some(proof) = self.0.take() {
            proof.live.store(false, Ordering::Release);
        }
    }
}
impl Drop for IdentitySlot {
    fn drop(&mut self) {
        self.invalidate();
    }
}
impl IdentityLease {
    pub fn require(&self, program: &ProgramIdentity, generation: &Arc<()>) -> Result<&StateBundle> {
        if !self.0.live.load(Ordering::Acquire)
            || !Arc::ptr_eq(&self.0.model_generation, generation)
        {
            return Err(Error::StaleEpoch);
        }
        if &self.0.bundle.program != program {
            return Err(Error::ProgramMismatch);
        }
        Ok(&self.0.bundle)
    }
}
