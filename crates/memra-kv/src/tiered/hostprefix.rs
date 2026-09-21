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
    /// True while at least one `IdentityLease` handed out by [`Self::lease`] is alive. The
    /// lease is the only other holder of the proof, so the count is exact. An unbound slot and
    /// a bound slot whose leases all dropped are unleased. Owners use this to leave a leased
    /// image alone (the host tier's tenant-share reclaim, memra#384).
    pub fn leased(&self) -> bool {
        self.0.as_ref().is_some_and(|p| Arc::strong_count(p) > 1)
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

/// CPU owned immutable image, not CUDA storage. Copy committed payloads BEFORE sealing
/// metadata; no mutable borrow or Arc to an active generation is retained.
pub struct SealedImage {
    bundle: StateBundle,
    payloads: Vec<Vec<u8>>,
}
impl SealedImage {
    pub fn copy_committed(
        active: &super::ActiveEpoch,
        bundle: &StateBundle,
        payloads: &[Vec<u8>],
    ) -> Result<Self> {
        active.require(bundle)?;
        bundle.verify(payloads)?;
        let copied = payloads.to_vec();
        let bundle = bundle.seal()?;
        bundle.verify(&copied)?;
        Ok(Self {
            bundle,
            payloads: copied,
        })
    }
    pub fn bundle(&self) -> &StateBundle {
        &self.bundle
    }
    pub fn payloads(&self) -> &[Vec<u8>] {
        &self.payloads
    }
}

/// Injected shared governor; runtime owners must NOT create a prefix-only governor.
pub type SharedGovernor = Arc<std::sync::Mutex<dyn BudgetGovernor + Send>>;

/// The ONE derivation of `ProgramIdentity.tenant_salt` from a PC-ISO cache namespace string
/// (lead ruling 13, lane/spill-c-20260919 day 13). The caller passes the exact string it
/// feeds its metering row (`memra-server` `auth::meter_key(&pool_key.1)`), so the share-cap
/// row and the governor tenant name the same tenant; no second derivation exists anywhere.
/// The empty string is a namespace too: it is the default single-tenant namespace of a
/// server without a keyring (`auth::meter_key` documents `""` the same way) and derives its
/// own salt, distinct from every keyring namespace (`t:<tenant>\x1f<salt>`), which can
/// never be empty.
pub fn tenant_salt(scope_namespace: &str) -> Digest {
    digest("tenant-salt", scope_namespace.as_bytes())
}

/// Build the governor a runtime owner injects everywhere (lead ruling 15: the server's
/// governor, held once and cloned into every consumer; never a prefix-only ledger). The
/// capacities, headroom, queue and dirty limits are the owner's; the clock is monotonic
/// nanoseconds in the same domain as every request deadline.
pub fn shared_governor(
    capacity: TierBudget,
    mandatory_headroom: TierBudget,
    queue_limit: usize,
    dirty_limit: u64,
    clock: Arc<dyn Fn() -> u64 + Send + Sync>,
) -> Result<SharedGovernor> {
    let governor = memra_tier::tier::Governor::new(
        capacity,
        mandatory_headroom,
        queue_limit,
        dirty_limit,
        clock,
    )?;
    let shared: SharedGovernor = Arc::new(std::sync::Mutex::new(governor));
    Ok(shared)
}
/// Synchronous owner-only residency charge. Declare this AFTER payload fields so Rust
/// destroys the bytes before crediting them. Async DMA requires explicit retirement,
/// not this guard. A failed release retains the governor/lease rather than granting credit.
pub struct ResidentCharge {
    governor: SharedGovernor,
    lease: Option<ChargedLease>,
    program: Option<ProgramIdentity>,
}
impl ResidentCharge {
    pub fn reserve(governor: SharedGovernor, request: &BudgetRequest) -> Result<Self> {
        let lease = governor
            .lock()
            .map_err(|_| Error::Quarantined)?
            .reserve(request)?;
        Ok(Self {
            governor,
            lease: Some(lease),
            program: None,
        })
    }
    /// Program-bound runtime residency. Reject a tenant mismatch BEFORE quota changes;
    /// keep the complete identity on the payload guard, not just the lookup sidecar.
    pub fn reserve_for_program(
        governor: SharedGovernor,
        program: &ProgramIdentity,
        request: &BudgetRequest,
    ) -> Result<Self> {
        program.validate()?;
        if request.tenant != program.tenant_salt {
            return Err(Error::ProgramMismatch);
        }
        let mut guard = Self::reserve(governor, request)?;
        guard.program = Some(program.clone());
        Ok(guard)
    }
    pub fn program(&self) -> Option<&ProgramIdentity> {
        self.program.as_ref()
    }
    pub fn lease(&self) -> &ChargedLease {
        self.lease.as_ref().expect("live residency guard")
    }
}
impl Drop for ResidentCharge {
    fn drop(&mut self) {
        let Some(lease) = self.lease.take() else {
            return;
        };
        let result = self
            .governor
            .lock()
            .map_err(|_| Error::Quarantined)
            .and_then(|mut g| g.release(&lease));
        if let Err(error) = result {
            eprintln!("[kv-tier] residency release quarantined: {error:?}");
            std::mem::forget((self.governor.clone(), lease));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tenant_salt_is_one_derivation_of_the_namespace_string() {
        let a = tenant_salt("t:acme\u{1f}salt-a");
        assert_eq!(a, tenant_salt("t:acme\u{1f}salt-a"));
        assert_ne!(a, tenant_salt("t:acme\u{1f}salt-b"));
        assert_ne!(a, tenant_salt("t:blue\u{1f}salt-a"));
        assert_ne!(a, tenant_salt("salt-a"));
        // The derivation is the framed contract digest, not a bare hash of the string.
        assert_eq!(a, digest("tenant-salt", b"t:acme\x1fsalt-a"));
    }

    #[test]
    fn empty_namespace_is_the_default_single_tenant_namespace_not_a_refusal() {
        // A server without a keyring keys every request's pool on `""` (its documented
        // default namespace), so the salt must exist and must not collide with any keyring
        // namespace, which always carries the `t:` prefix and the separator.
        let default_ns = tenant_salt("");
        assert_eq!(default_ns, tenant_salt(""));
        assert_ne!(default_ns, tenant_salt("t:default\u{1f}"));
        assert_ne!(default_ns, tenant_salt("t:\u{1f}"));
        assert_ne!(default_ns, [0; 32]);
    }

    #[test]
    fn unbound_slot_is_never_leased() {
        // The positive half (bound, leased while an IdentityLease lives, unleased after it
        // drops) needs a program and a bundle and lives with the host tier's reclaim tests
        // in memra-server (`host_cache_tenant_share_reclaim_skips_leased_entries_*`).
        let slot = IdentitySlot::default();
        assert!(!slot.leased());
    }

    #[test]
    fn shared_governor_is_the_injected_trait_object_and_charges_through_it() {
        let mut capacity = TierBudget::zero(1);
        capacity.pinned = 10;
        let governor =
            shared_governor(capacity, TierBudget::zero(1), 0, 0, Arc::new(|| 0)).unwrap();
        let mut bytes = TierBudget::zero(1);
        bytes.pinned = 6;
        let request = BudgetRequest {
            bytes,
            priority: Priority::Backup,
            deadline: Deadline(u64::MAX),
            tenant: tenant_salt(""),
        };
        let lease = governor.lock().unwrap().reserve(&request).unwrap();
        assert_eq!(governor.lock().unwrap().used().pinned, 6);
        assert_eq!(
            governor.lock().unwrap().reserve(&request).err(),
            Some(Error::Capacity)
        );
        governor.lock().unwrap().release(&lease).unwrap();
        assert_eq!(governor.lock().unwrap().used().pinned, 0);
        let mut bad = TierBudget::zero(1);
        bad.pinned = 1;
        let mut headroom = TierBudget::zero(1);
        headroom.pinned = 2;
        assert!(shared_governor(bad, headroom, 0, 0, Arc::new(|| 0)).is_err());
    }
}
