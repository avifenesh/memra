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
        // The snapshot keeps the epoch that was validated. A revocation during
        // validation leaves it stale instead of blessing the newer epoch.
        let epoch = generation.epoch.load(Ordering::Acquire);
        if !boundary_validated(generation, epoch)
            && let Err(error) = validate_boundary()
        {
            generation.revoke();
            return Err(error);
        }
        Ok(Self::at_epoch(generation, admission, pipeline, epoch))
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
        Self::at_epoch(
            generation,
            admission,
            pipeline,
            generation.epoch.load(Ordering::Acquire),
        )
    }

    fn at_epoch(
        generation: &Arc<ProgramGeneration>,
        admission: &RewriteAdmission,
        pipeline: bool,
        epoch: u64,
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
            epoch,
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
        // current scope or a held boundary at this epoch permits us to reuse
        // validation. Dropping the previous guard and resuming a graph or worker
        // tick outside a held boundary always checks again.
        let validated = active_execution(generation)
            .is_some_and(|active| active.epoch == self.epoch)
            || boundary_validated(generation, self.epoch);
        if !validated {
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

#[derive(Default)]
struct BoundaryScopes {
    next: u64,
    /// (guard id, generation key, validated epoch).
    entries: Vec<(u64, usize, u64)>,
}

thread_local! {
    static ACTIVE: RefCell<ActiveScopes> = RefCell::default();
    static BOUNDARIES: RefCell<BoundaryScopes> = RefCell::default();
}

/// True only while this thread holds a boundary validated at `epoch` and the
/// generation has not moved since. A boundary grants no surface.
fn boundary_validated(generation: &Arc<ProgramGeneration>, epoch: u64) -> bool {
    let key = Arc::as_ptr(generation) as usize;
    epoch == generation.epoch.load(Ordering::Acquire)
        && BOUNDARIES.with(|scopes| {
            scopes
                .borrow()
                .entries
                .iter()
                .any(|&(_, held, validated)| held == key && validated == epoch)
        })
}

/// One external-state validation for one model, held across a worker tick. While it
/// is held, snapshots and scopes of that model that are still current at the
/// validated epoch enter without rescanning libraries or the environment. It grants
/// no rewrite surface. Every entry still checks its own snapshot generation, so a
/// revoked session refuses and cannot borrow a peer's validation. External state
/// must stay fixed for the tick; a change is detected at the next boundary.
#[must_use]
pub struct RewriteBoundaryGuard<'a> {
    id: u64,
    // Same reason as RewriteExecutionGuard: a leaked entry keeps its generation
    // alive, so a later model cannot reuse the pointer key.
    _generation: Arc<ProgramGeneration>,
    _borrow: PhantomData<&'a ()>,
    _thread: PhantomData<Rc<()>>,
}

impl RewriteBoundaryGuard<'_> {
    pub(crate) fn hold<'a, T>(
        generation: &Arc<ProgramGeneration>,
        _program: &'a T,
        validate_boundary: impl FnOnce() -> Result<(), String>,
    ) -> Result<RewriteBoundaryGuard<'a>, String> {
        let epoch = generation.epoch.load(Ordering::Acquire);
        if !boundary_validated(generation, epoch) {
            // Same policy as every other boundary: observed drift revokes.
            if let Err(error) = validate_boundary() {
                generation.revoke();
                return Err(error);
            }
            if generation.epoch.load(Ordering::Acquire) != epoch {
                return Err("rewrite boundary was revoked during validation".into());
            }
        }
        let key = Arc::as_ptr(generation) as usize;
        let id = BOUNDARIES.with(|scopes| {
            let mut scopes = scopes.borrow_mut();
            scopes.next += 1;
            let id = scopes.next;
            scopes.entries.push((id, key, epoch));
            id
        });
        Ok(RewriteBoundaryGuard {
            id,
            _generation: generation.clone(),
            _borrow: PhantomData,
            _thread: PhantomData,
        })
    }
}

impl Drop for RewriteBoundaryGuard<'_> {
    fn drop(&mut self) {
        BOUNDARIES.with(|scopes| {
            let mut scopes = scopes.borrow_mut();
            if let Some(index) = scopes.entries.iter().position(|(id, _, _)| *id == self.id) {
                scopes.entries.remove(index);
            }
        });
    }
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

    /// One worker tick for N sessions of one model, in the order the scheduler runs
    /// them: per-session sample/emit and token emit, a fallback capability probe, a
    /// batched step (check every participant, enter the first), a protected graph
    /// or prime call, and the retire sweep.
    fn run_tick<T>(
        generation: &Arc<ProgramGeneration>,
        program: &T,
        sessions: &[RewriteExecutionSnapshot],
        validate: impl Fn() -> Result<(), String> + Copy,
    ) {
        let admission = RewriteAdmission::LegacyUnbundled;
        for session in sessions {
            let _sample_emit = session.enter(generation, program, validate).unwrap();
            let active = active_execution(generation).unwrap();
            let nested = RewriteExecutionSnapshot::from_active(generation, active);
            let _nested = nested.enter(generation, program, validate).unwrap();
        }
        for session in sessions {
            let _token_emit = session.enter(generation, program, validate).unwrap();
        }
        // An unscoped `rewrite_allowed` probe, as in the batched-prime candidate loop.
        let probe =
            RewriteExecutionSnapshot::validated(generation, &admission, false, validate).unwrap();
        assert!(probe.allows(RewriteSurface::CarriedPrime));
        for chunk in sessions.chunks(8) {
            assert!(chunk.iter().all(|session| session.current(generation)));
            let _batched = chunk[0].enter(generation, program, validate).unwrap();
        }
        let _graph =
            RewriteExecutionSnapshot::protect(generation, &admission, false, program, validate)
                .unwrap();
        for session in sessions {
            let _retire = session.enter(generation, program, validate).unwrap();
        }
    }

    #[test]
    fn one_boundary_per_model_per_tick_validates_once_for_n_sessions() {
        let generation = Arc::default();
        let program = TrackedProgram::new((), Arc::clone(&generation));
        let inventories = Cell::new(0);
        let validate = || {
            inventories.set(inventories.get() + 1);
            Ok(())
        };
        let admission = RewriteAdmission::LegacyUnbundled;
        let sessions: Vec<_> = (0..64)
            .map(|_| {
                RewriteExecutionSnapshot::validated(&generation, &admission, false, validate)
                    .unwrap()
            })
            .collect();
        inventories.set(0);
        // Without a boundary every outermost entry pays external validation.
        run_tick(&generation, &program, &sessions, validate);
        assert!(inventories.get() > sessions.len());
        for _ in 0..100 {
            inventories.set(0);
            {
                let _tick = RewriteBoundaryGuard::hold(&generation, &program, validate).unwrap();
                run_tick(&generation, &program, &sessions, validate);
            }
            assert_eq!(inventories.get(), 1);
        }
        // The boundary ends with the tick. A later entry is a new boundary.
        let _resume = sessions[0].enter(&generation, &program, validate).unwrap();
        assert_eq!(inventories.get(), 2);
    }

    #[test]
    fn tick_boundaries_are_per_model_and_do_not_depend_on_stack_order() {
        let first: Arc<ProgramGeneration> = Arc::default();
        let second: Arc<ProgramGeneration> = Arc::default();
        let inventories = Cell::new(0);
        let validate = || {
            inventories.set(inventories.get() + 1);
            Ok(())
        };
        let admission = RewriteAdmission::LegacyUnbundled;
        let a = RewriteExecutionSnapshot::validated(&first, &admission, false, validate).unwrap();
        let b = RewriteExecutionSnapshot::validated(&second, &admission, false, validate).unwrap();
        inventories.set(0);
        let _first_tick = RewriteBoundaryGuard::hold(&first, &(), validate).unwrap();
        let _second_tick = RewriteBoundaryGuard::hold(&second, &(), validate).unwrap();
        for _ in 0..1_000 {
            let _a = a.enter(&first, &(), validate).unwrap();
            let _b = b.enter(&second, &(), validate).unwrap();
            let _a_again = a.enter(&first, &(), validate).unwrap();
        }
        assert_eq!(inventories.get(), 2);
        // A boundary never lends one model's validation to another model.
        let other: Arc<ProgramGeneration> = Arc::default();
        let c = RewriteExecutionSnapshot::new(&other, &admission, false);
        let _c = c.enter(&other, &(), validate).unwrap();
        assert_eq!(inventories.get(), 3);
    }

    #[test]
    fn revoked_session_refuses_inside_a_held_tick_boundary() {
        let generation = Arc::default();
        let inventories = Cell::new(0);
        let validate = || {
            inventories.set(inventories.get() + 1);
            Ok(())
        };
        let admission = RewriteAdmission::LegacyUnbundled;
        let old =
            RewriteExecutionSnapshot::validated(&generation, &admission, false, validate).unwrap();
        generation.revoke(); // A reinstall or a drift seen at an earlier boundary.
        let newer =
            RewriteExecutionSnapshot::validated(&generation, &admission, false, validate).unwrap();
        inventories.set(0);
        let tick = RewriteBoundaryGuard::hold(&generation, &(), validate).unwrap();
        let error = old
            .enter(&generation, &(), validate)
            .err()
            .expect("revoked session borrowed the tick boundary");
        assert!(error.contains("was revoked"), "{error}");
        assert!(active_execution(&generation).is_none());
        // A current peer does not lend its scope to the revoked session.
        let peer = newer.enter(&generation, &(), validate).unwrap();
        assert!(old.enter(&generation, &(), validate).is_err());
        assert!(!old.allows(RewriteSurface::DecodeEager));
        assert!(![&old, &newer].iter().all(|s| s.current(&generation)));
        drop(peer);
        assert_eq!(inventories.get(), 1);
        // A revocation during the tick ends the boundary's reuse at once.
        generation.revoke();
        assert!(newer.enter(&generation, &(), validate).is_err());
        let fresh =
            RewriteExecutionSnapshot::validated(&generation, &admission, false, validate).unwrap();
        assert_eq!(inventories.get(), 2);
        let _fresh = fresh.enter(&generation, &(), validate).unwrap();
        assert_eq!(inventories.get(), 3);
        drop(tick);
    }

    #[test]
    fn failed_tick_boundary_holds_nothing_and_revokes() {
        let generation = Arc::default();
        let admission = RewriteAdmission::LegacyUnbundled;
        let session =
            RewriteExecutionSnapshot::validated(&generation, &admission, false, || Ok(())).unwrap();
        let error = RewriteBoundaryGuard::hold(&generation, &(), || {
            Err("library/environment changed".into())
        })
        .err()
        .expect("drift held a tick boundary");
        assert_eq!(error, "library/environment changed");
        let epoch = generation.epoch.load(Ordering::Acquire);
        assert!(!boundary_validated(&generation, epoch));
        assert!(session.enter(&generation, &(), || Ok(())).is_err());
        // A validator that revokes cannot leave a boundary behind either.
        let error = RewriteBoundaryGuard::hold(&generation, &(), || {
            generation.revoke();
            Ok(())
        })
        .err()
        .expect("revoked boundary was held");
        assert!(error.contains("revoked during validation"));
        let epoch = generation.epoch.load(Ordering::Acquire);
        assert!(!boundary_validated(&generation, epoch));
    }
}

#[cfg(test)]
#[path = "execution_snapshot/reentry_tests.rs"]
mod reentry_tests;
