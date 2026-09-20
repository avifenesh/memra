//! Protected execution scopes. Only boundaries inspect external process state; token
//! admission reads the current generation and a fixed surface mask.

use memra_gguf::execution_manifest::{RewriteAdmission, RewriteSurface};
use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// All mutable access passes through one invalidation point. This owns the data;
/// there is no shared mutable alias that can bypass the generation.
pub(crate) struct TrackedProgram<T> {
    data: T,
    generation: Arc<ProgramGeneration>,
}

impl<T> TrackedProgram<T> {
    pub(crate) fn new(data: T, generation: Arc<ProgramGeneration>) -> Self {
        Self { data, generation }
    }
}

impl<T> std::ops::Deref for TrackedProgram<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.data
    }
}

impl<T> std::ops::DerefMut for TrackedProgram<T> {
    fn deref_mut(&mut self) -> &mut T {
        self.generation.mutate();
        &mut self.data
    }
}

#[derive(Default)]
pub(crate) struct ProgramGeneration {
    epoch: AtomicU64,
    mutations: AtomicU64,
}

impl ProgramGeneration {
    pub(crate) fn mutations(&self) -> u64 {
        self.mutations.load(Ordering::Acquire)
    }

    pub(crate) fn revoke(&self) {
        self.epoch.fetch_add(1, Ordering::AcqRel);
    }

    pub(crate) fn mutate(&self) {
        self.mutations.fetch_add(1, Ordering::AcqRel);
        self.revoke();
    }
}

fn surface_bit(surface: RewriteSurface) -> u16 {
    1 << surface as u16
}

const SURFACES: [RewriteSurface; 8] = [
    RewriteSurface::CarriedPrime,
    RewriteSurface::ForwardFreshKv,
    RewriteSurface::DecodeEager,
    RewriteSurface::DecodeBatch,
    RewriteSurface::DecodeGraph,
    RewriteSurface::MtpSpec,
    RewriteSurface::Glm5Spec,
    RewriteSurface::Pipeline,
];

/// A request/session's validated program. Creation checks process libraries and
/// numerical environment. Mutation and every bundle reinstall revoke old snapshots.
/// Retention preserves origin, not permission to skip later external-state checks.
/// Every outermost entry validates the boundary before activating this snapshot;
/// nested entries share the existing protected scope.
#[derive(Clone)]
pub struct RewriteExecutionSnapshot {
    generation: Arc<ProgramGeneration>,
    epoch: u64,
    mask: u16,
    pipeline: bool,
    qualified: bool,
}

impl RewriteExecutionSnapshot {
    pub(crate) fn protect<'a, T>(
        generation: &Arc<ProgramGeneration>,
        admission: &RewriteAdmission,
        pipeline: bool,
        program: &'a T,
        validate_boundary: impl FnOnce() -> Result<(), String>,
    ) -> Result<RewriteExecutionGuard<'a>, String> {
        let snapshot = match active_execution(generation) {
            Some(active) => Self::from_active(generation, active),
            None => Self::new(generation, admission, pipeline),
        };
        snapshot.enter(generation, program, validate_boundary)
    }

    pub(crate) fn validated(
        generation: &Arc<ProgramGeneration>,
        admission: &RewriteAdmission,
        pipeline: bool,
        validate_boundary: impl FnOnce() -> Result<(), String>,
    ) -> Result<Self, String> {
        if let Err(error) = validate_boundary() {
            generation.revoke();
            return Err(error);
        }
        Ok(Self::new(generation, admission, pipeline))
    }
    pub(crate) fn from_active(
        generation: &Arc<ProgramGeneration>,
        active: ActiveExecution,
    ) -> Self {
        Self {
            generation: generation.clone(),
            epoch: active.epoch,
            mask: active.mask,
            pipeline: active.pipeline,
            qualified: active.qualified,
        }
    }

    pub(crate) fn allows(&self, surface: RewriteSurface) -> bool {
        self.current(&self.generation)
            && ActiveExecution {
                key: Arc::as_ptr(&self.generation) as usize,
                epoch: self.epoch,
                mask: self.mask,
                pipeline: self.pipeline,
                qualified: self.qualified,
            }
            .allows(surface)
    }
    fn new(
        generation: &Arc<ProgramGeneration>,
        admission: &RewriteAdmission,
        pipeline: bool,
    ) -> Self {
        let mask = SURFACES.into_iter().fold(0, |mask, surface| {
            mask | if admission.allows(surface) {
                surface_bit(surface)
            } else {
                0
            }
        });
        Self {
            generation: generation.clone(),
            epoch: generation.epoch.load(Ordering::Acquire),
            mask,
            pipeline,
            qualified: admission.is_qualified(),
        }
    }

    pub(crate) fn current(&self, generation: &Arc<ProgramGeneration>) -> bool {
        Arc::ptr_eq(&self.generation, generation)
            && self.epoch == generation.epoch.load(Ordering::Acquire)
    }

    pub(crate) fn enter<'a, T>(
        &self,
        generation: &Arc<ProgramGeneration>,
        _program: &'a T,
        validate_boundary: impl FnOnce() -> Result<(), String>,
    ) -> Result<RewriteExecutionGuard<'a>, String> {
        if !self.current(generation) {
            return Err("rewrite execution snapshot was revoked; validate the current program at a request boundary".into());
        }
        // A retained origin cannot establish its own authority. Only an existing,
        // current scope permits us to reuse validation. In particular, dropping the
        // previous guard and resuming a graph or worker tick always checks again.
        if active_execution(generation).is_none_or(|active| active.epoch != self.epoch) {
            if let Err(error) = validate_boundary() {
                generation.revoke();
                return Err(error);
            }
            // Validation may itself revoke the program. Never rebase the retained
            // origin onto the new epoch, even if the callback otherwise succeeded.
            if !self.current(generation) {
                return Err(
                    "rewrite execution snapshot was revoked during boundary validation".into(),
                );
            }
        }
        let entry = ActiveExecution {
            key: Arc::as_ptr(generation) as usize,
            epoch: self.epoch,
            mask: self.mask,
            pipeline: self.pipeline,
            qualified: self.qualified,
        };
        let id = ACTIVE.with(|active| {
            let mut active = active.borrow_mut();
            active.next += 1;
            let id = active.next;
            active.entries.push((id, entry));
            id
        });
        Ok(RewriteExecutionGuard {
            id,
            _generation: generation.clone(),
            _borrow: PhantomData,
            _thread: PhantomData,
        })
    }
}

#[derive(Clone, Copy)]
pub(crate) struct ActiveExecution {
    key: usize,
    epoch: u64,
    mask: u16,
    pipeline: bool,
    qualified: bool,
}

impl ActiveExecution {
    pub(crate) fn allows(self, surface: RewriteSurface) -> bool {
        self.mask & surface_bit(surface) != 0
            && (!self.qualified
                || !self.pipeline
                || self.mask & surface_bit(RewriteSurface::Pipeline) != 0)
    }

    pub(crate) fn qualified(self) -> bool {
        self.qualified
    }
}

#[derive(Default)]
struct ActiveScopes {
    next: u64,
    entries: Vec<(u64, ActiveExecution)>,
}

thread_local! {
    static ACTIVE: RefCell<ActiveScopes> = RefCell::default();
}

pub(crate) fn active_execution(generation: &Arc<ProgramGeneration>) -> Option<ActiveExecution> {
    ACTIVE.with(|active| {
        active
            .borrow()
            .entries
            .last()
            .map(|(_, entry)| *entry)
            .filter(|entry| entry.key == Arc::as_ptr(generation) as usize)
            .map(|mut entry| {
                if entry.epoch != generation.epoch.load(Ordering::Acquire) {
                    entry.mask = 0;
                    entry.qualified = false;
                }
                entry
            })
    })
}

/// Holds an immutable borrow of the model for the scope, including nested eager,
/// graph, speculative and pipeline calls. Deliberately cannot move to another thread.
#[must_use]
pub struct RewriteExecutionGuard<'a> {
    id: u64,
    // Retain the identity even if a caller forgets a guard. A later model must not
    // reuse the pointer of a leaked thread-local entry and inherit its permissions.
    _generation: Arc<ProgramGeneration>,
    _borrow: PhantomData<&'a ()>,
    _thread: PhantomData<Rc<()>>,
}

impl Drop for RewriteExecutionGuard<'_> {
    fn drop(&mut self) {
        ACTIVE.with(|active| {
            let mut active = active.borrow_mut();
            if let Some(index) = active.entries.iter().position(|(id, _)| *id == self.id) {
                active.entries.remove(index);
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn repeated_protected_tokens_do_not_read_inventory_or_serialize_program() {
        // The token API has no Debug/Serialize bound and never reads this payload.
        // Metadata work, if accidentally added here, must not run on token admission.
        struct Program;
        impl std::fmt::Debug for Program {
            fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                panic!("model/plan serialization on a protected token path")
            }
        }
        let generation = Arc::default();
        let program = TrackedProgram::new(Program, Arc::clone(&generation));
        let inventories = Cell::new(0);
        let snapshot =
            RewriteExecutionSnapshot::new(&generation, &RewriteAdmission::LegacyUnbundled, false);
        let validate = || {
            inventories.set(inventories.get() + 1);
            Ok(())
        };
        {
            let _outer = RewriteExecutionSnapshot::protect(
                &generation,
                &RewriteAdmission::LegacyUnbundled,
                false,
                &program,
                validate,
            )
            .unwrap();
            for _ in 0..10_000 {
                let _scope = snapshot.enter(&generation, &program, validate).unwrap();
                let active = active_execution(&generation).unwrap();
                for surface in SURFACES {
                    assert!(active.allows(surface));
                }
                let nested = RewriteExecutionSnapshot::from_active(&generation, active);
                let _nested = nested.enter(&generation, &program, validate).unwrap();
                assert!(
                    active_execution(&generation)
                        .unwrap()
                        .allows(RewriteSurface::DecodeEager)
                );
            }
            assert_eq!(inventories.get(), 1);
        }
        assert!(active_execution(&generation).is_none());
        // A later standalone resume is a new boundary, even with the same origin.
        let _resume = snapshot.enter(&generation, &program, validate).unwrap();
        assert_eq!(inventories.get(), 2);
    }

    #[test]
    fn same_shape_mutation_and_every_reinstall_revoke_retained_snapshots() {
        let generation = Arc::default();
        let mut program = TrackedProgram::new(vec![1u8, 2], Arc::clone(&generation));
        let old =
            RewriteExecutionSnapshot::new(&generation, &RewriteAdmission::LegacyUnbundled, false);
        assert!(old.enter(&generation, &program, || Ok(())).is_ok());
        program[0] = 9;
        assert_eq!(generation.mutations(), 1);
        assert!(old.enter(&generation, &program, || Ok(())).is_err());
        let newer =
            RewriteExecutionSnapshot::new(&generation, &RewriteAdmission::LegacyUnbundled, false);
        generation.revoke(); // Both successful and failed reinstall enter here first.
        assert!(newer.enter(&generation, &program, || Ok(())).is_err());
        assert_eq!(generation.mutations(), 1);
        let latest =
            RewriteExecutionSnapshot::new(&generation, &RewriteAdmission::LegacyUnbundled, false);
        assert!(latest.enter(&generation, &program, || Ok(())).is_ok());
        assert!(old.enter(&generation, &program, || Ok(())).is_err());
    }

    #[test]
    fn boundary_drift_revokes_inflight_scopes_and_cannot_reactivate_them() {
        let generation = Arc::default();
        let program = ();
        let old =
            RewriteExecutionSnapshot::new(&generation, &RewriteAdmission::LegacyUnbundled, false);
        let _scope = old.enter(&generation, &program, || Ok(())).unwrap();
        let error = RewriteExecutionSnapshot::validated(
            &generation,
            &RewriteAdmission::LegacyUnbundled,
            false,
            || Err("library/environment changed".into()),
        );
        assert!(error.is_err());
        assert!(
            !active_execution(&generation)
                .unwrap()
                .allows(RewriteSurface::DecodeEager)
        );
        let restored = RewriteExecutionSnapshot::validated(
            &generation,
            &RewriteAdmission::LegacyUnbundled,
            false,
            || Ok(()),
        )
        .unwrap();
        assert!(restored.enter(&generation, &program, || Ok(())).is_ok());
        assert!(
            !active_execution(&generation)
                .unwrap()
                .allows(RewriteSurface::DecodeEager)
        );
        assert!(old.enter(&generation, &program, || Ok(())).is_err());
    }

    #[test]
    fn snapshots_are_model_bound_and_pipeline_permission_is_checked_once() {
        let generation = Arc::default();
        let other = Arc::default();
        let mut snapshot =
            RewriteExecutionSnapshot::new(&generation, &RewriteAdmission::LegacyUnbundled, true);
        assert!(snapshot.enter(&other, &(), || Ok(())).is_err());
        snapshot.qualified = true;
        snapshot.mask = surface_bit(RewriteSurface::DecodeEager);
        assert!(!snapshot.allows(RewriteSurface::DecodeEager));
        snapshot.mask |= surface_bit(RewriteSurface::Pipeline);
        assert!(snapshot.allows(RewriteSurface::DecodeEager));
        assert!(!snapshot.allows(RewriteSurface::DecodeGraph));
    }

    #[test]
    fn out_of_order_scope_drops_do_not_restore_an_ended_scope() {
        let generation = Arc::default();
        let snapshot =
            RewriteExecutionSnapshot::new(&generation, &RewriteAdmission::LegacyUnbundled, false);
        let first = snapshot.enter(&generation, &(), || Ok(())).unwrap();
        let second = snapshot.enter(&generation, &(), || Ok(())).unwrap();
        drop(first);
        assert!(active_execution(&generation).is_some());
        drop(second);
        assert!(active_execution(&generation).is_none());
    }
}

#[cfg(test)]
#[path = "execution_snapshot/reentry_tests.rs"]
mod reentry_tests;
