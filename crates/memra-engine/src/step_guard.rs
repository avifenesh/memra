//! THE BATCHED DECODE STEP GUARD (WP-B day 49, `research/spill-b-20260919/DAY49.md` 1.5, OWED O14).
//!
//! A batched decode step that fails with a CUDA OOM can be run again only if it wrote no
//! session state: the linear-attention conv ring is updated IN PLACE, so no host-side marker
//! (lengths, positions, ping-pong parity) can tell a torn step from a clean one. This guard is
//! set by the code that writes the state: the serve worker arms it before the call (`Armed`),
//! the batched entry marks `Entered`, the generic unsplit body marks `Generic` before its
//! pre-layer setup, and `decode_batch_layers` marks `Touched` immediately before every
//! state-writing statement. `recoverable` holds only for `Armed` (the failure came before the
//! engine call) and `Generic` (the generic body failed before any state write).

use std::cell::Cell;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepState {
    Idle,
    Armed,
    Entered,
    Generic,
    Touched,
}

thread_local! {
    static STATE: Cell<StepState> = const { Cell::new(StepState::Idle) };
}

/// The worker arms the guard for one batched call.
pub fn arm() {
    STATE.with(|s| s.set(StepState::Armed));
}

/// The state the last call left, and back to `Idle`.
pub fn take() -> StepState {
    STATE.with(|s| s.replace(StepState::Idle))
}

/// A failure the worker may retry after a reclaim: no session state was written.
pub fn recoverable(state: StepState) -> bool {
    matches!(state, StepState::Armed | StepState::Generic)
}

fn advance(to: StepState) {
    STATE.with(|s| {
        let now = s.get();
        // Only forward, and only while armed: an unarmed call (a test, a CLI) never records.
        let next = match (now, to) {
            (StepState::Idle, _) => StepState::Idle,
            (StepState::Touched, _) => StepState::Touched,
            (_, StepState::Touched) => StepState::Touched,
            (StepState::Armed, StepState::Entered) => StepState::Entered,
            (StepState::Entered, StepState::Generic) => StepState::Generic,
            (other, _) => other,
        };
        s.set(next);
    });
}

/// The batched entry (`decode_step_batch_sampled_lean_masked_schedule`) is reached.
pub(crate) fn entered() {
    advance(StepState::Entered);
}

/// The generic unsplit body begins its pre-layer setup (no state is written before the layers).
pub(crate) fn generic() {
    advance(StepState::Generic);
}

/// Immediately before a statement that writes session state.
pub(crate) fn touched() {
    advance(StepState::Touched);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_a_call_that_wrote_no_state_is_recoverable() {
        arm();
        assert!(recoverable(take()), "failed before the engine call");
        arm();
        entered();
        assert_eq!(take(), StepState::Entered);
        assert!(
            !recoverable(StepState::Entered),
            "a non-generic program is not tracked"
        );
        arm();
        entered();
        generic();
        assert!(
            recoverable(take()),
            "the generic body before any state write"
        );
        arm();
        entered();
        generic();
        touched();
        generic();
        assert_eq!(take(), StepState::Touched, "touched is sticky");
        // Unarmed calls never record.
        entered();
        touched();
        assert_eq!(take(), StepState::Idle);
        // Generic without Entered does not skip the entry mark.
        arm();
        generic();
        assert_eq!(take(), StepState::Armed);
    }
}
