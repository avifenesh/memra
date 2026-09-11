//! DSV4 door reach: what a door's declared default actually resolves to on the
//! program a request runs under.
//!
//! Why this module exists (memra #454, #458). Every DSV4 door merged since #374
//! was measured honestly, on the tuned bench program every `dsv4_*_gate` binary
//! pins: attention TP/EP, the matrix expert program, the drafter off, the
//! small-kernel diet on, and the gate-only fused-GU arm turned on AFTER load
//! through `Dsv4Gpu::set_grouped_gu_fuse_for_gate`. What was never checked is
//! whether that program is the one a customer request takes.
//!
//! **Half of that gap closed on 2026-09-11** (memra #461, owner ruling): the
//! matrix expert program is the SERVED program now, not a default-OFF door, so
//! `SERVED_PROGRAM.matrix_moe` is true and the expert-program axis no longer
//! separates the bench from the served path. The other half did not move and
//! cannot: the served path is PP-2 with a resident DSpark drafter and chunked
//! prefill, and the engine refuses the union with TP/EP in two independent
//! places: the TP/EP topology guard refuses MTP/DSpark state, and
//! `prefill_with_cache_chunked` refuses a batched prime at all under
//! `topology.is_tp_ep()`, "DSV4 TP/EP vertical slice currently admits only a
//! single-token prime; batched replicated cache hydration is not wired". So
//! TP/EP can neither chunk nor serve, and either refusal alone makes those two
//! programs disjoint. Both are cited by FUNCTION and refusal text on purpose:
//! line numbers in this file move under every lane that touches it.
//!
//! A door admitted only under TP/EP is not "inert today", it is PERMANENTLY
//! unreachable on the served path: the program it needs cannot take a customer
//! request at all, for two independent structural reasons (memra #457), so no
//! program decision rescues it. `AdmittingProgram` is that axis and
//! `served_disposition` is where the futures are decided.
//!
//! What the two flips did to the counts, written down because the counts are
//! the claim: six of nine merged default-ON doors were inert, the matrix flip
//! took that to FOUR of seven, and the PP-2 norm port takes it to ONE. The two
//! that left with the matrix flip were the split-K pair, and they left by
//! DELETION rather than by rescue: their admitting program can serve, but the
//! arm they need is a gate-only function called after load, so no serving
//! process could reach them on any ordering.
//!
//! THREE more left by RESCUE, which is a different thing and the first time it
//! has happened here: the norm doors (`norm-fuse` #404, `norm-fuse2` #426,
//! `norm2-wide` #430), which the owner ruled FIXED rather than deleted. Their
//! admission term never needed a topology (no norm kernel, and no consumer of a
//! norm kernel's output, reads a rank, a shard or a topology plan), and what
//! made TP/EP look like a precondition was a SEPARATE kernel limit: the pack
//! launchers pinned grid 1, which forced a `t == 1` call-site guard that only
//! the drafter-off bench program satisfied on every shape. The launchers now
//! take `rows` and run the identical per-row 128-thread tree, so both terms are
//! gone together and `chains_f32`, the one real precondition, is what is left.
//! The one door still inert is inert for a reason of its own rather than for a
//! program reason. Doors still go inert SILENTLY, because an unset value
//! resolves to `Ok(admitted)` with `admitted == false`, which is why this
//! registry exists rather than a comment.
//!
//! So this module holds three things, and the load path uses the same code the
//! gate does:
//!
//! 1. `Dsv4Program`, a description of a running program, with the two constants
//!    that matter: `SERVED_PROGRAM` (what a customer request takes today) and
//!    `TUNED_BENCH_PROGRAM` (what every DSV4 gate binary pins). Since the flip
//!    they differ on topology and gate-only arms, no longer on expert program.
//! 2. The door policy functions themselves. `Dsv4Gpu::load` calls exactly these,
//!    so the gate cannot drift into being a second, kinder model of admission.
//! 3. `DSV4_DOORS`, one row per door: the default and the served-path state the
//!    registries CLAIM, checked by `declaration_violations` against what the
//!    policy functions actually return. A door whose claim stops being true
//!    fails `cargo test`, which is the only place this family has ever been
//!    caught.

/// Same error type the rest of the DSV4 admission path uses.
pub type Res<T> = Result<T, String>;

/// The shapes a door can be on for. Two of the engaged doors are `m == 1` only,
/// which is decode; prefill is 95-97% of a request's GPU seconds at this
/// family's real input:output ratio, so "on" without a shape is not an answer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DoorShape {
    /// Every routed shape the program issues, prefill chunks included.
    AllRoutedShapes,
    /// Single-row transactions only (`m == 1`): decode, never a prefill chunk.
    DecodeOnlyM1,
}

/// What a door resolves to for a given program with the environment unset,
/// which is the state a default claim is a claim ABOUT.
///
/// The three not-engaged states are kept apart on purpose, because they have
/// three different fixes and the six inert DSV4F doors are not all the same
/// case. `Off` is an admission predicate saying no on a program that HAS the
/// call site; `OffProgram` is a door whose call site exists only on a program
/// we do not serve, which the program decision resolves; `NoServingCaller` is a
/// door no serving request can reach by construction, whatever we decide about
/// programs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DoorState {
    /// The door is on and its call sites are reachable on this program.
    On(DoorShape),
    /// The door resolved off. Its call sites exist on this program, so porting
    /// or widening the admission predicate is a real option.
    Off,
    /// The door refuses this program at load rather than resolving.
    RefusedAtLoad,
    /// Reachable, but only on a program this one is not. Names the program that
    /// does reach it, because that program's fate is this door's fate.
    OffProgram(&'static str),
    /// No serving request can reach this door on ANY program: its only caller is
    /// a gate binary arming an instrument. Dead code on every serving path.
    NoServingCaller(&'static str),
}

impl DoorState {
    pub fn engaged(self) -> bool {
        matches!(self, DoorState::On(_))
    }
}

/// Which program admits a door at all. This is the axis that decides what to DO
/// about an inert door, because the three answers have three different futures.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdmittingProgram {
    /// The served program itself admits it: it engages, or its own predicate
    /// says no for a reason that has nothing to do with which program we run.
    ServedProgram,
    /// Only the matrix expert executor reaches it. That program CAN serve, and a
    /// sibling lane measured it on the served path at +42% to +140% prefill, so
    /// whether these doors ever engage is the memra #461 matrix verdict.
    MatrixExecutor,
    /// Only the all-layer TP/EP topology admits it, and TP/EP cannot serve a
    /// customer request at all (memra #457, two independent refusals). No
    /// foreseeable program decision makes these reachable.
    TpEpOnly,
    /// Only a gate binary arming an instrument reaches it. No serving caller
    /// exists on any program.
    GateInstrumentOnly,
}

/// What to DO about a door, given its resolved state and the program that
/// admits it. Printed in the load receipt and pinned per door by the gate, so
/// the inert doors cannot collapse into one recommendation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DoorDisposition {
    /// Engaged on the served program. Nothing to decide.
    Engaged,
    /// The program that admits this door cannot serve a request at all, so no
    /// foreseeable change makes it reachable: removable on its own evidence,
    /// with no dependency on any other verdict.
    PermanentlyUnreachable,
    /// The door belongs to another program that CAN serve. It resolves itself if
    /// that program becomes the default, and leaves with it if it does not.
    FollowsTheProgramDecision,
    /// Nothing about a serving program can make this door engage: it is a gate
    /// instrument's input filed as a product door.
    ReclassifyAsGateInput,
    /// The served program admits it and its own predicate resolved off, so the
    /// question is about the predicate rather than about programs.
    PortAdmissionOrRedeclare,
    /// Refused at load. The configuration is illegal, not silently degraded.
    RefusedConfiguration,
}

/// Facts about the programs a door can live on, kept as values rather than baked
/// into per-door constants, so the red arm can ask what changes if TP/EP ever
/// learns to serve.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProgramFacts {
    /// Can the all-layer TP/EP topology take a customer request?
    pub tp_ep_can_serve: bool,
    /// Can the matrix expert program take one? Yes, and measured.
    pub matrix_can_serve: bool,
}

/// Today's facts: TP/EP is refused twice over, the matrix program serves.
pub const PROGRAM_FACTS: ProgramFacts = ProgramFacts {
    tp_ep_can_serve: false,
    matrix_can_serve: true,
};

/// A program: the set of admission inputs every DSV4 door reads. Constructed at
/// load from the resolved configuration, and as a constant for the two programs
/// the registry makes claims about.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dsv4Program {
    pub name: &'static str,
    /// `Dsv4TopologyPlan::is_tp_ep()`: the all-layer TP/EP topology.
    pub tp_ep: bool,
    /// f32x chains.
    pub chains_f32: bool,
    /// f32x dots.
    pub dots_f32: bool,
    /// The matrix expert executor (`dsv4_grouped.rs`), which is the loaded
    /// default since 2026-09-11. The reference program never enters that file,
    /// and a process only runs it by calling
    /// `arm_reference_expert_program_for_gate` before load.
    pub matrix_moe: bool,
    /// A drafter (DSpark/MTP) is resident, so decode steps carry `tmax > 1` or a
    /// tap destination and are not the plain fused decode shape.
    pub drafter_resident: bool,
    /// The process arms the gate-only fused-GU seam
    /// (`Dsv4Gpu::set_grouped_gu_fuse_for_gate`). No serving process does: the
    /// arm is `MEMRA_F16G_GU_FUSE`, kept OFF and research-only in FLAGS.md, and
    /// the server never calls the setter.
    pub gate_armed_gu_fuse: bool,
    /// This checkpoint's HC geometry is `rows == 24 && w == 16384`.
    pub hc_geometry_24x16384: bool,
    /// Whether this program can serve a customer request AT ALL. TP/EP cannot:
    /// `prefill_with_cache_chunked` refuses a batched prime under
    /// `topology.is_tp_ep()` ("admits only a single-token prime; batched
    /// replicated cache hydration is not wired"), and the topology guard
    /// separately refuses MTP/DSpark state, which the served spec route
    /// requires. The two refusals are independent: replicating the drafter per
    /// rank would still leave TP/EP unable to chunk (memra #457).
    pub can_serve: bool,
}

/// What a customer request runs today: PP-2 (`Dsv4TopologyPlan::pp_ep`, so
/// `is_tp_ep()` is false), the MATRIX expert program, DSpark resident, chunked
/// prefill, and no gate-only arms.
///
/// `matrix_moe` was false until 2026-09-11. The owner flipped it on the memra
/// #461 class verdict (NEW numeric class, 4.464% top-1 drift on the whole tape
/// against 3.571% for the already-accepted HC S16 door on the same tape, 179/300
/// against 179/300 on the paired accuracy control, McNemar p = 1.000000) plus the
/// `EP=off` gain the serve-economics lane measured on this exact arm: 135.7/135.5
/// tok/s reference against 173.8/173.8 matrix at a 3,686-token prompt, disjoint
/// by 28.0%. There is no `MEMRA_DSV4_MOE_PROGRAM` door any more: the matrix
/// program is what loads, and the scalar reference executor is reachable only
/// through `arm_reference_expert_program_for_gate`, which no serving process
/// calls and which exists so every CLASS row still has its reference arm.
pub const SERVED_PROGRAM: Dsv4Program = Dsv4Program {
    name: "served",
    tp_ep: false,
    chains_f32: true,
    dots_f32: true,
    matrix_moe: true,
    drafter_resident: true,
    gate_armed_gu_fuse: false,
    hc_geometry_24x16384: true,
    can_serve: true,
};

/// What every `dsv4_*_gate` binary pins, and what every merged door was measured
/// on: TP/EP, the matrix expert program, the drafter off, no chunked prefill,
/// and the gate-only fused-GU arm on.
pub const TUNED_BENCH_PROGRAM: Dsv4Program = Dsv4Program {
    name: "tuned-bench",
    tp_ep: true,
    chains_f32: true,
    dots_f32: true,
    matrix_moe: true,
    drafter_resident: false,
    gate_armed_gu_fuse: true,
    hc_geometry_24x16384: true,
    // The whole point: this program cannot take a customer request.
    can_serve: false,
};

// ---------------------------------------------------------------------------
// Policy functions. `Dsv4Gpu::load` calls these; so does the registry below.
// ---------------------------------------------------------------------------

/// memra #458, RESOLVED BY DELETION 2026-09-11. The matrix program's plain
/// gate/up split-K arm computed gate and up in ONE fused launch, so it required
/// the fused-GU arm; `gate_up` refused with "split-K requires plain fused GU"
/// when it was not on. That arm was gate-only: `MEMRA_F16G_GU_FUSE` is OFF and
/// research-only, and the only way to turn it on was
/// `Dsv4Gpu::set_grouped_gu_fuse_for_gate`, which every gate binary called after
/// load and no serving process called at all. Split-K had been default ON since
/// #392, so `MEMRA_DSV4_MOE_PROGRAM=matrix` on a serving stack booted CLEAN and
/// then failed 100% of requests as `engine_error` 500s. memra #462 replaced that
/// outage with a load-time refusal.
///
/// With the matrix program as the LOADED DEFAULT (memra #461, owner flip
/// 2026-09-11), that refusal would have fired on every serving boot. Arming
/// split-K for serving was never the fix: its precondition is a function only a
/// gate calls after load, so a serving process cannot meet it on any ordering.
/// Under the owner's door rule ("a door is the default or it is deleted") both
/// `MEMRA_DSV4_MOE_M1_SPLITK` (#392) and `MEMRA_DSV4_SPLITK_FAST` (#425) were
/// therefore REMOVED in the flip lane, with their kernels, their gate binaries
/// and their registry rows. Their +10.226982%/+10.023951% and
/// +1.727312%/+1.581907% were measured on a bench-armed configuration no serving
/// process can reach, and that is recorded in the FLAGS.md removed-doors ledger
/// rather than carried as a default. This paragraph is the tombstone: the next
/// lane that wants split-K on the served path owns removing the fused-GU
/// precondition first, and then owes a served receipt before any default.
///
/// The scalar reference expert executor survives, and only here. It is the arm
/// every CLASS row compares against (`dsv4_moe_program_class_gate`,
/// `dsv4_program_accuracy`), so it must stay reachable, and it must be
/// unreachable from a serving process, because a request that silently took the
/// slower numeric class is exactly the incident #461 was opened about. So it has
/// no environment variable: a gate binary calls this before `Dsv4Gpu::load`, and
/// nothing else does.
static REFERENCE_PROGRAM_ARMED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Load the scalar reference expert program in THIS process instead of the
/// matrix default. Gate binaries that own a CLASS comparison call it before
/// load; no serving path calls it, and there is no environment variable that
/// sets it.
pub fn arm_reference_expert_program_for_gate() {
    REFERENCE_PROGRAM_ARMED.store(true, std::sync::atomic::Ordering::SeqCst);
}

/// Undo the arm, so a gate binary that loads both programs in one process can
/// put the default back rather than leaking the reference arm into a later load.
pub fn disarm_reference_expert_program_for_gate() {
    REFERENCE_PROGRAM_ARMED.store(false, std::sync::atomic::Ordering::SeqCst);
}

/// The program `Dsv4Gpu::load` resolves: matrix unless a gate armed the
/// reference executor. This is the whole of the program decision; there is no
/// environment read behind it.
pub fn matrix_expert_program_resolved() -> bool {
    !REFERENCE_PROGRAM_ARMED.load(std::sync::atomic::Ordering::SeqCst)
}

// ---------------------------------------------------------------------------
// Per-door resolution against a program, with the environment unset.
// ---------------------------------------------------------------------------

/// A door's declared default, as the registries state it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeclaredDefault {
    On,
    Off,
}

pub struct DoorRow {
    /// Door name as `docs/FLAGS.md` and `docs/KERNELS.md` know it.
    pub name: &'static str,
    /// The environment variable that carries the door, if it has one.
    pub env: &'static str,
    /// The PR the door merged on.
    pub merged: &'static str,
    /// The default the registries declare.
    pub declared_default: DeclaredDefault,
    /// What the door resolves to on the SERVED program with the variable unset.
    /// This is the claim under test: change an admission predicate and this row
    /// stops being true, loudly.
    pub declared_served: DoorState,
    /// What it resolves to on the tuned bench program every gate pins. The row
    /// exists so a resolver that ignores its program argument cannot pass.
    pub declared_bench: DoorState,
    /// Resolution, environment unset, for one program.
    pub resolve: fn(&Dsv4Program) -> DoorState,
    /// Which program admits this door at all. With `PROGRAM_FACTS` this is what
    /// turns "inert" into a decision.
    pub admitted_by: AdmittingProgram,
    /// The pooled forward and reverse percentages the door merged on, as
    /// `docs/FLAGS.md` records them. Kept as numbers rather than prose because
    /// the instrument floor below is a comparison, not an opinion.
    pub merged_gain_pct: (f64, f64),
    /// The hardware those percentages were measured on.
    pub measured_on: &'static str,
}

/// The dev pair drifts about 5% MONOTONICALLY within one arm across a single
/// run (OFF rows climbing 50.417848 to 52.936086 on 2026-09-10), so a lever
/// whose ceiling is under about 2% of step time is below that instrument's
/// resolution whatever its steady rows looked like. Darklanes
/// `KNEE:dev-pair-run-drift-swamps-sub-2pct-levers`, the finding that deleted
/// the split vocab head door (darklanes #595).
pub const DEV_PAIR_INSTRUMENT_FLOOR_PCT: f64 = 2.0;

/// Whether a door's merged magnitude is below what the pair it was measured on
/// can resolve. Not a re-litigation of any merge: it is the honest scope of what
/// the receipt establishes.
pub fn below_instrument_floor(row: &DoorRow, floor_pct: f64) -> bool {
    let (forward, reverse) = row.merged_gain_pct;
    forward.abs() < floor_pct && reverse.abs() < floor_pct
}

/// What to do about a door on the served path: its resolved state, then the
/// program that admits it and whether that program can take a request.
pub fn served_disposition(row: &DoorRow, facts: &ProgramFacts) -> DoorDisposition {
    let state = (row.resolve)(&SERVED_PROGRAM);
    if state.engaged() {
        return DoorDisposition::Engaged;
    }
    match state {
        DoorState::RefusedAtLoad => DoorDisposition::RefusedConfiguration,
        _ => match row.admitted_by {
            AdmittingProgram::GateInstrumentOnly => DoorDisposition::ReclassifyAsGateInput,
            AdmittingProgram::TpEpOnly if !facts.tp_ep_can_serve => {
                DoorDisposition::PermanentlyUnreachable
            }
            AdmittingProgram::MatrixExecutor if facts.matrix_can_serve => {
                DoorDisposition::FollowsTheProgramDecision
            }
            // A door on a program that cannot serve is permanently unreachable
            // whichever program that is; anything else is about the predicate.
            AdmittingProgram::MatrixExecutor => DoorDisposition::PermanentlyUnreachable,
            AdmittingProgram::TpEpOnly | AdmittingProgram::ServedProgram => {
                DoorDisposition::PortAdmissionOrRedeclare
            }
        },
    }
}

/// Doors with no evidence for their default in EITHER direction: inert on the
/// served path, so the default buys nothing, AND measured below the instrument
/// floor of the pair they were qualified on, so the number that justified the
/// default is not one that pair can resolve. Strictly stronger than either fact
/// alone, and a cleaner argument for removal than "we do not use it".
pub fn doors_without_evidence_in_either_direction(
    rows: &[DoorRow],
    facts: &ProgramFacts,
    floor_pct: f64,
) -> Vec<&'static str> {
    rows.iter()
        // memra #470. The complaint is about a door DEFAULTING ON with nothing to justify
        // it. A default-OFF door is off by decision, so its magnitude is its
        // evidence rather than a missing justification, and including it would
        // also break this function's own invariant: at a floor above every
        // magnitude the set must be exactly `inert_default_on_doors`.
        .filter(|row| row.declared_default == DeclaredDefault::On)
        .filter(|row| served_disposition(row, facts) != DoorDisposition::Engaged)
        .filter(|row| below_instrument_floor(row, floor_pct))
        .map(|row| row.name)
        .collect()
}

/// Doors that DO engage on the served path but whose merged magnitude the pair
/// they were measured on cannot resolve. Their defaults reach a customer; the
/// evidence for those defaults is what is thin.
pub fn engaged_but_below_the_floor(rows: &[DoorRow], floor_pct: f64) -> Vec<&'static str> {
    rows.iter()
        .filter(|row| (row.resolve)(&SERVED_PROGRAM).engaged())
        .filter(|row| below_instrument_floor(row, floor_pct))
        .map(|row| row.name)
        .collect()
}

static NOT_PRESENT: std::env::VarError = std::env::VarError::NotPresent;

fn unset() -> Result<&'static str, &'static std::env::VarError> {
    Err(&NOT_PRESENT)
}




fn resolve_replay_cadence(p: &Dsv4Program) -> DoorState {
    // `arm_full_token_replay_for_gate` is the only caller, its admission needs
    // the gate-only fused-GU arm and refuses host split-K/DSpark state, and its
    // own FLAGS.md row says it: "No automatic arming of eager or serving
    // requests". A served request never reaches the cadence choice.
    if p.gate_armed_gu_fuse && p.tp_ep && !p.drafter_resident {
        DoorState::On(DoorShape::DecodeOnlyM1)
    } else {
        DoorState::NoServingCaller(
            "full-token replay is armed per request by a gate binary only; no serving or eager \
             request arms it, so the cadence choice has no caller",
        )
    }
}

fn resolve_dense_exact_tail(_p: &Dsv4Program) -> DoorState {
    // Generic dense entry points, so the served program does reach it, but
    // `Dsv4DenseExactTailControlScope control(m != 1)` suppresses it for every
    // m > 1: decode gets it, prefill chunks never do.
    DoorState::On(DoorShape::DecodeOnlyM1)
}

fn resolve_dense_fast(_p: &Dsv4Program) -> DoorState {
    // Same m == 1 scope as the dense tail (cu/dsv4_dense_m1_exact_tail.cuh).
    DoorState::On(DoorShape::DecodeOnlyM1)
}

fn resolve_hc_dot_split(p: &Dsv4Program) -> DoorState {
    // `dots_f32 && rows == 24 && w == 16384`, no topology term at all.
    if p.dots_f32 && p.hc_geometry_24x16384 {
        DoorState::On(DoorShape::AllRoutedShapes)
    } else {
        DoorState::Off
    }
}

/// The nine merged DSV4F doors, with the default and served-path state the
/// registries claim for each. A tenth, `MEMRA_DSV4_DENSE_TILE`, existed for one
/// afternoon and was REMOVED rather than carried: its receipt was positive
/// beyond the box's noise on a same-class change, and a same-class win with a
/// clean receipt becomes the code (owner, 2026-09-10). See the removed-doors
/// ledger in `docs/FLAGS.md`.
pub const DSV4_DOORS: &[DoorRow] = &[
    DoorRow {
        name: "replay cadence",
        env: "MEMRA_DSV4_REPLAY_CADENCE",
        merged: "#374",
        declared_default: DeclaredDefault::On,
        declared_served: DoorState::NoServingCaller(
            "full-token replay is armed per request by a gate binary only; no serving or eager \
             request arms it, so the cadence choice has no caller",
        ),
        declared_bench: DoorState::On(DoorShape::DecodeOnlyM1),
        resolve: resolve_replay_cadence,
        admitted_by: AdmittingProgram::GateInstrumentOnly,
        merged_gain_pct: (1.007107, 1.074545),
        measured_on: "2x RTX PRO 6000 Blackwell Max-Q dev pair",
    },
    DoorRow {
        name: "dense exact-tail transport",
        env: "MEMRA_DSV4_DENSE_EXACT_TAIL",
        merged: "#374",
        declared_default: DeclaredDefault::On,
        declared_served: DoorState::On(DoorShape::DecodeOnlyM1),
        declared_bench: DoorState::On(DoorShape::DecodeOnlyM1),
        resolve: resolve_dense_exact_tail,
        admitted_by: AdmittingProgram::ServedProgram,
        merged_gain_pct: (0.459578, 0.397451),
        measured_on: "2x RTX PRO 6000 Blackwell Max-Q dev pair",
    },
    DoorRow {
        name: "dense-fast",
        env: "MEMRA_DSV4_DENSE_FAST",
        merged: "#404",
        declared_default: DeclaredDefault::On,
        declared_served: DoorState::On(DoorShape::DecodeOnlyM1),
        declared_bench: DoorState::On(DoorShape::DecodeOnlyM1),
        resolve: resolve_dense_fast,
        admitted_by: AdmittingProgram::ServedProgram,
        merged_gain_pct: (1.618979, 1.609496),
        measured_on: "2x RTX PRO 6000 Blackwell Max-Q dev pair, composed with norm-fuse",
    },
    DoorRow {
        name: "HC dot split S16",
        env: "MEMRA_DSV4_HC_DOT_SPLIT",
        merged: "#418",
        declared_default: DeclaredDefault::On,
        declared_served: DoorState::On(DoorShape::AllRoutedShapes),
        declared_bench: DoorState::On(DoorShape::AllRoutedShapes),
        resolve: resolve_hc_dot_split,
        admitted_by: AdmittingProgram::ServedProgram,
        merged_gain_pct: (2.701174, 2.591532),
        measured_on: "2x RTX PRO 6000 Blackwell Max-Q dev pair",
    },
];

// ---------------------------------------------------------------------------
// Door-set derivation. The registry above is an explicit list, and an explicit
// list is exactly what stops covering doors added later. So the list is not
// trusted: these functions DERIVE the door-name set from the engine sources
// that read the environment, and the gate fails until every derived name is
// either a declared door or an exemption with a written reason. A door added to
// the engine is covered by construction: the person adding it gets a red test
// naming their variable.
// ---------------------------------------------------------------------------

/// Every engine source that could carry a DSV4 door, DERIVED from the tree
/// rather than listed: `src/dsv4*` and `cu/dsv4*` plus the grouped expert
/// kernels. Listing files by hand is the same disease as listing doors by hand,
/// one level up, so the list is a directory read and a new `dsv4_*.rs` file is
/// scanned the day it lands. Gate binaries are excluded on purpose: a gate
/// process is allowed to read instruments.
pub fn door_name_source_files(crate_root: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    for (dir, prefixes) in [("src", &["dsv4"][..]), ("cu", &["dsv4", "moe_f16"][..])] {
        let mut entries: Vec<_> = std::fs::read_dir(crate_root.join(dir))
            .unwrap_or_else(|e| panic!("read {dir}: {e}"))
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| path.is_file())
            .filter(|path| {
                path.file_name().and_then(|n| n.to_str()).is_some_and(|n| {
                    // This file names every door and every exemption, so scanning
                    // it would make both the coverage check and the stale-exemption
                    // check pass on their own text.
                    n != "dsv4_doors.rs" && prefixes.iter().any(|p| n.starts_with(p))
                })
            })
            .collect();
        entries.sort();
        out.append(&mut entries);
    }
    out
}

/// Every `MEMRA_DSV4_*` / `MEMRA_F16G_*` / `MEMRA_MOE_F16G*` name that appears
/// in the given source text. Pure, so the red arm can hand it a source that
/// contains a door nobody declared.
pub fn door_names_in_source(text: &str) -> std::collections::BTreeSet<String> {
    let bytes = text.as_bytes();
    let mut names = std::collections::BTreeSet::new();
    let mut i = 0;
    while let Some(hit) = text[i..].find("MEMRA_") {
        let start = i + hit;
        let mut end = start;
        while end < bytes.len()
            && (bytes[end].is_ascii_uppercase()
                || bytes[end].is_ascii_digit()
                || bytes[end] == b'_')
        {
            end += 1;
        }
        let name = &text[start..end];
        let carries_a_suffix = !name.ends_with('_');
        if carries_a_suffix
            && (name.starts_with("MEMRA_DSV4_")
                || name.starts_with("MEMRA_F16G_")
                || name.starts_with("MEMRA_MOE_F16G"))
        {
            names.insert(name.to_string());
        }
        i = end.max(start + 1);
    }
    names
}

/// Names the sources read that no `DSV4_DOORS` row declares and no exemption
/// excuses. Non-empty means the registry has stopped covering the engine.
pub fn undeclared_door_names(
    names: &std::collections::BTreeSet<String>,
    declared: &[&str],
    exempt: &[(&str, &str)],
) -> Vec<String> {
    names
        .iter()
        .filter(|name| {
            !declared.contains(&name.as_str()) && !exempt.iter().any(|(e, _)| e == &name.as_str())
        })
        .cloned()
        .collect()
}

/// Exemptions that no longer name anything the engine reads. An exemption list
/// that outlives its reasons is a blanket excuse for the next door.
pub fn stale_exemptions(
    names: &std::collections::BTreeSet<String>,
    exempt: &[(&str, &str)],
) -> Vec<String> {
    exempt
        .iter()
        .filter(|(name, _)| !names.contains(*name))
        .map(|(name, _)| name.to_string())
        .collect()
}

/// One violation of a declared door claim.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DoorViolation {
    pub door: &'static str,
    pub program: &'static str,
    pub declared: DoorState,
    pub resolved: DoorState,
}

impl std::fmt::Display for DoorViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "door {:?} on the {} program: registries declare {:?}, admission code resolves {:?}",
            self.door, self.program, self.declared, self.resolved
        )
    }
}

/// Check every declared door claim against what the admission code returns for
/// both programs. Empty means the registries and the engine agree.
///
/// Taking the rows as an argument rather than reading `DSV4_DOORS` directly is
/// what lets the red arm feed this a knowingly false row and see it fail.
pub fn declaration_violations(rows: &[DoorRow]) -> Vec<DoorViolation> {
    let mut out = Vec::new();
    for row in rows {
        for (program, declared) in [
            (&SERVED_PROGRAM, row.declared_served),
            (&TUNED_BENCH_PROGRAM, row.declared_bench),
        ] {
            let resolved = (row.resolve)(program);
            if resolved != declared {
                out.push(DoorViolation {
                    door: row.name,
                    program: program.name,
                    declared,
                    resolved,
                });
            }
        }
    }
    out
}

/// The doors whose declared default is ON and which do not engage on the served
/// program. This is the number memra #454 is about; it is THREE since the PP-2
/// norm port, and the gate pins it so it cannot grow silently.
pub fn inert_default_on_doors(rows: &[DoorRow]) -> Vec<&'static str> {
    rows.iter()
        .filter(|row| row.declared_default == DeclaredDefault::On)
        .filter(|row| !(row.resolve)(&SERVED_PROGRAM).engaged())
        .map(|row| row.name)
        .collect()
}

/// One line per door for the load receipt, so an operator reading a boot log can
/// see which doors this process actually got.
pub fn door_receipt_lines(program: &Dsv4Program) -> Vec<String> {
    DSV4_DOORS
        .iter()
        .map(|row| {
            format!(
                "[dsv4-doors] program={} door={:?} env={} merged={} declared_default={:?} resolved={:?} disposition={:?}",
                program.name,
                row.name,
                row.env,
                row.merged,
                row.declared_default,
                (row.resolve)(program),
                served_disposition(row, &PROGRAM_FACTS)
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    /// Why the arms below use a SYNTHETIC row instead of a real door.
    ///
    /// Several of these arms need a door that CANNOT engage on the served
    /// program, and they had been keyed to whichever real door was stuck at the
    /// time. That is a red arm with a maintenance schedule, and it came due
    /// twice in two days: the matrix flip deleted `graph split-K`, and the PP-2
    /// norm port makes `norm2-wide` reachable, turning this arm's lie into a
    /// truth. A red arm that goes green because someone FIXED its subject has
    /// stopped checking, silently, which is the failure mode this registry
    /// exists to catch.
    ///
    /// So the stand-in is defined here, depends on nothing else in this file,
    /// and stays correct whichever real doors exist: a door admitted only under
    /// all-layer TP/EP, which the served program is not and (memra #457, two
    /// independent refusals) cannot become. Unlike a real door, nobody can fix
    /// it, because there is nothing behind it to fix.
    ///
    /// Taken verbatim from the norm-fuse PP-2 port lane's trial branch
    /// (`trial-normpp2-on-482`, `c3a4099b2`) rather than re-invented, so the two
    /// lanes cannot drift into two framings of the same subject.
    pub(super) const SYNTHETIC_TP_EP_ONLY: &str =
        "synthetic red-arm door: admitted only under all-layer TP/EP, which cannot serve";

    pub(super) fn resolve_synthetic_tp_ep_only(p: &Dsv4Program) -> DoorState {
        if p.tp_ep {
            DoorState::On(DoorShape::AllRoutedShapes)
        } else {
            DoorState::OffProgram(SYNTHETIC_TP_EP_ONLY)
        }
    }

    /// The stand-in as a ROW, for arms that need a `DoorRow` rather than a
    /// resolver. Required where a derived subject list is allowed to reach zero:
    /// since the PP-2 norm port the registry holds no real TP/EP-only door, and
    /// a `for` over an empty list passes, so this row is what keeps the
    /// permanence disposition actually asserted.
    pub(super) const SYNTHETIC_TP_EP_ONLY_ROW: DoorRow = DoorRow {
        name: "a future TP/EP-only door",
        // A real declared name on purpose: the coverage scan in this module
        // reads this file, so an invented `MEMRA_DSV4_` literal here would be
        // reported as an undeclared door. The row's teeth are in its resolver,
        // not in this label, and #482's synthetic rows label themselves the
        // same way.
        env: "MEMRA_DSV4_REPLAY_CADENCE",
        merged: "#374",
        declared_default: DeclaredDefault::On,
        declared_served: DoorState::OffProgram(SYNTHETIC_TP_EP_ONLY),
        declared_bench: DoorState::On(DoorShape::AllRoutedShapes),
        resolve: resolve_synthetic_tp_ep_only,
        admitted_by: AdmittingProgram::TpEpOnly,
        merged_gain_pct: (0.0, 0.0),
        measured_on: "synthetic: a stand-in, not a claim about any box",
    };

    /// The stand-in must be non-vacuous in BOTH directions, or every arm built
    /// on it proves nothing: inert on the served program, live on the bench.
    #[test]
    fn the_synthetic_stand_in_is_inert_on_served_and_live_on_the_bench() {
        assert!(!resolve_synthetic_tp_ep_only(&SERVED_PROGRAM).engaged());
        assert!(resolve_synthetic_tp_ep_only(&TUNED_BENCH_PROGRAM).engaged());
    }

    use super::*;

    /// The gate. Every declared default/served-path claim in the registry must be
    /// what the admission code returns for the program a request actually takes.
    #[test]
    fn declared_door_state_matches_resolved_door_state() {
        let violations = declaration_violations(DSV4_DOORS);
        assert!(
            violations.is_empty(),
            "door declarations disagree with admission code:\n{}",
            violations
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join("\n")
        );
    }

    /// Red arm 1: the checker goes red on a door that claims a default it does
    /// not get. A check that has never failed is not a check, so fail it here.
    #[test]
    fn a_door_that_claims_an_engagement_it_does_not_get_is_caught() {
        const LIAR: &[DoorRow] = &[DoorRow {
            name: "a TP/EP-only door, claiming the served path",
            env: "MEMRA_DSV4_REPLAY_CADENCE",
            merged: "#374",
            declared_default: DeclaredDefault::On,
            // The lie, and it is a lie about a SYNTHETIC door on purpose: see
            // the note on `resolve_synthetic_tp_ep_only`. This arm used to lie
            // with norm2-wide, and the PP-2 port makes that claim TRUE.
            declared_served: DoorState::On(DoorShape::AllRoutedShapes),
            declared_bench: DoorState::On(DoorShape::AllRoutedShapes),
            resolve: resolve_synthetic_tp_ep_only,
            admitted_by: AdmittingProgram::TpEpOnly,
            merged_gain_pct: (0.0, 0.0),
            measured_on: "synthetic: the liar arm, not a claim about any box",
        }];
        let violations = declaration_violations(LIAR);
        assert_eq!(violations.len(), 1, "{violations:?}");
        assert_eq!(violations[0].program, "served");
        assert_eq!(
            violations[0].resolved,
            DoorState::OffProgram(SYNTHETIC_TP_EP_ONLY)
        );
    }

    /// Red arm 2: a door claimed at the wrong SHAPE is caught too. Decode-only
    /// is not "on" when prefill is 95-97% of the bill.
    #[test]
    fn a_decode_only_door_claimed_for_every_shape_is_caught() {
        const LIAR: &[DoorRow] = &[DoorRow {
            name: "dense-fast, claiming prefill",
            env: "MEMRA_DSV4_DENSE_FAST",
            merged: "#404",
            declared_default: DeclaredDefault::On,
            declared_served: DoorState::On(DoorShape::AllRoutedShapes),
            declared_bench: DoorState::On(DoorShape::AllRoutedShapes),
            resolve: resolve_dense_fast,
            admitted_by: AdmittingProgram::ServedProgram,
            merged_gain_pct: (1.618979, 1.609496),
            measured_on: "2x RTX PRO 6000 Blackwell Max-Q dev pair",
        }];
        assert_eq!(declaration_violations(LIAR).len(), 2);
    }

    /// Non-vacuity, crosswise: the two programs must disagree about at least one
    /// door in each direction, so a resolver that ignores its program argument,
    /// or two programs that have drifted into being the same program, cannot
    /// pass. This is the exact failure mode that made two gates on main panic:
    /// an expectation keyed to a constant instead of to resolved state.
    #[test]
    fn the_served_and_bench_programs_resolve_doors_differently() {
        assert_ne!(SERVED_PROGRAM, TUNED_BENCH_PROGRAM);
        assert!(!DSV4_DOORS.is_empty(), "an empty registry loops zero times");
        let differing: Vec<_> = DSV4_DOORS
            .iter()
            .filter(|row| (row.resolve)(&SERVED_PROGRAM) != (row.resolve)(&TUNED_BENCH_PROGRAM))
            .map(|row| row.name)
            .collect();
        // One since the PP-2 norm port. Before it four: the matrix flip had
        // already deleted the two split-K doors that used to differ purely
        // because the served program was the reference one, and the expert
        // program itself no longer separates the two programs. The port then
        // took the three norm doors from TP/EP-only to served, so one
        // disagreement is left. Non-vacuity for this test does not rest on the
        // count: the assertion above it requires the list to be non-empty and
        // the synthetic stand-in keeps a permanent disagreement in the file.
        assert_eq!(differing.len(), 1, "{differing:?}");
        for row in DSV4_DOORS {
            // Nothing may be engaged on the served path and dead on the bench:
            // that direction would mean the bench is measuring less than what
            // serves, which is a different bug with the same shape.
            if (row.resolve)(&SERVED_PROGRAM).engaged() {
                assert!(
                    (row.resolve)(&TUNED_BENCH_PROGRAM).engaged(),
                    "{} engages on the served path and not on the bench",
                    row.name
                );
            }
        }
    }

    /// The count memra #454 reports, pinned. Making a door reachable, or
    /// changing its declared default to match reality, or deleting it, all move
    /// this number, and all three are decisions that must be written down. It
    /// moved from six to three when the PP-2 norm port took the three norm doors
    /// off the TP/EP-only list: this assertion is the record of that decision.
    #[test]
    fn one_default_on_door_is_inert_on_the_served_path() {
        // Six before 2026-09-11. The matrix flip removed the two split-K rows
        // (deleted outright, not rescued: their precondition is gate-only), and
        // it moved nothing else, because the remaining four were inert for the
        // TP/EP reason an expert-program decision was never going to fix. The
        // PP-2 norm port fixed that reason for three of them by RESCUE, so what
        // is left is the one door inert for a reason of its own.
        assert_eq!(
            inert_default_on_doors(DSV4_DOORS),
            vec!["replay cadence"]
        );
    }

    /// The scalar reference executor has exactly one way in, and a serving
    /// process does not have it. The red arm is the DEFAULT: if
    /// `matrix_expert_program_resolved` ever answers false without the arm, a
    /// request has silently taken the slower numeric class, which is the
    /// incident memra #461 was opened about.
    #[test]
    fn the_reference_expert_program_is_reachable_only_through_the_gate_arm() {
        disarm_reference_expert_program_for_gate();
        assert!(
            matrix_expert_program_resolved(),
            "unarmed, the loaded program must be matrix"
        );
        // No environment variable reaches it: the old door name is gone from the
        // engine, and setting it changes nothing.
        // SAFETY: single-threaded test, restored immediately below.
        unsafe { std::env::set_var("MEMRA_DSV4_MOE_PROGRAM", "reference") };
        assert!(
            matrix_expert_program_resolved(),
            "the removed door must not resurrect the reference program"
        );
        unsafe { std::env::remove_var("MEMRA_DSV4_MOE_PROGRAM") };

        arm_reference_expert_program_for_gate();
        assert!(
            !matrix_expert_program_resolved(),
            "a gate binary that armed the reference arm must get it"
        );
        disarm_reference_expert_program_for_gate();
        assert!(matrix_expert_program_resolved());
    }

    #[test]
    fn every_declared_door_has_a_flags_row() {
        let flags = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/FLAGS.md"),
        )
        .expect("docs/FLAGS.md");
        for row in DSV4_DOORS {
            assert!(
                flags.contains(row.env),
                "door {:?} has no docs/FLAGS.md row for {}",
                row.name,
                row.env
            );
        }
    }

    /// The exemption list: names the engine reads that are NOT doors. Each is a
    /// program selector, a geometry or tuning parameter, a gate-only instrument
    /// or a diagnostic, and each carries the reason it is not a merged default a
    /// performance claim rests on.
    const EXEMPT: &[(&str, &str)] = &[
        (
            "MEMRA_DSV4_AR_PHASE",
            "gate-only all-reduce phase instrument, refuses to load unarmed",
        ),
        ("MEMRA_DSV4_BENCH_PROFILE", "bench profile selector"),
        ("MEMRA_DSV4_DECODE_PATH", "program selector"),
        ("MEMRA_DSV4_DENSE_ARM", "program selector"),
        ("MEMRA_DSV4_DOTS_ARM", "program selector"),
        ("MEMRA_DSV4_DRAFTER", "program selector"),
        ("MEMRA_DSV4_DSPARK_CHAIN", "drafter program selector"),
        ("MEMRA_DSV4_DSPARK_FUSED_MOE", "drafter program selector"),
        ("MEMRA_DSV4_DSPARK_HEAD_ARM", "drafter program selector"),
        ("MEMRA_DSV4_DSPARK_MARKOV", "drafter program selector"),
        ("MEMRA_DSV4_DSPARK_PROPOSAL", "drafter program selector"),
        ("MEMRA_DSV4_EP", "topology selector"),
        ("MEMRA_DSV4_EP_ROUTE_STATS", "diagnostic counter"),
        ("MEMRA_DSV4_EXPERT_ARM", "program selector"),
        ("MEMRA_DSV4_GROUPED_ROUTE", "program selector"),
        ("MEMRA_DSV4_HAVE_NVTX", "build-time profiling switch"),
        ("MEMRA_DSV4_INDEXER_SCORE", "program selector"),
        ("MEMRA_DSV4_NVTX", "profiling ranges"),
        ("MEMRA_DSV4_PEER_PROBE_POISON", "gate-only fault injection"),
        ("MEMRA_DSV4_PREFILL_DRAFT", "program selector"),
        ("MEMRA_DSV4_PREFILL_HEAD", "program selector"),
        ("MEMRA_DSV4_PREFILL_MOE", "program selector"),
        ("MEMRA_DSV4_PREFILL_PROGRESS", "diagnostic"),
        ("MEMRA_DSV4_ROUND_PROFILE", "profiling"),
        ("MEMRA_DSV4_SAMPLER", "program selector"),
        ("MEMRA_DSV4_SAMPLE_SORT", "program selector"),
        ("MEMRA_DSV4_SINK_SCORE", "program selector"),
        (
            "MEMRA_DSV4_SMALL_KERNEL_DIET",
            "TP/EP-only program selector, refuses elsewhere",
        ),
        ("MEMRA_DSV4_SPEC_DEPTH", "spec parameter"),
        ("MEMRA_DSV4_VERIFY_TOPK", "program selector"),
        ("MEMRA_DSV4_VT", "spec threshold parameter"),
        ("MEMRA_DSV4_VT_FLOOR", "spec threshold parameter"),
        ("MEMRA_DSV4_VT_TAU", "spec threshold parameter"),
        ("MEMRA_F16G_BDB", "grouped visitor tuning parameter"),
        ("MEMRA_F16G_DIRECT", "grouped visitor program selector"),
        (
            "MEMRA_F16G_GU_FUSE",
            "gate-only research arm; its gate-only status is why the two split-K \
             doors were deleted rather than served (memra #458, #461)",
        ),
        ("MEMRA_F16G_SK", "split-K width parameter"),
        ("MEMRA_F16G_SK_CROSS", "split-K crossover parameter"),
        ("MEMRA_F16G_TAIL", "grouped visitor tail form selector"),
        ("MEMRA_MOE_F16G", "grouped visitor mode selector"),
    ];

    fn declared_env_names() -> Vec<&'static str> {
        DSV4_DOORS.iter().map(|row| row.env).collect()
    }

    fn engine_door_names() -> std::collections::BTreeSet<String> {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let files = door_name_source_files(root);
        assert!(
            files.len() >= 10,
            "the source derivation found almost no files, so coverage would be vacuous: {files:?}"
        );
        let mut names = std::collections::BTreeSet::new();
        for file in files {
            let text = std::fs::read_to_string(&file).unwrap_or_else(|e| panic!("{file:?}: {e}"));
            names.append(&mut door_names_in_source(&text));
        }
        assert!(
            names.len() > 30,
            "the name scan found almost nothing, so coverage would be vacuous: {names:?}"
        );
        names
    }

    /// Non-vacuity, coverage: every DSV4/F16G environment name the engine reads
    /// is either a declared door or an explicitly exempt one with a reason. The
    /// name set is DERIVED from the tree (every `src/dsv4*` and `cu/dsv4*` file
    /// plus the grouped expert kernels, by directory read), so a door added in a
    /// new file is covered the day it lands and nobody has to remember to extend
    /// a list.
    #[test]
    fn every_dsv4_door_name_in_the_engine_is_declared_or_exempt() {
        let names = engine_door_names();
        let undeclared = undeclared_door_names(&names, &declared_env_names(), EXEMPT);
        assert!(
            undeclared.is_empty(),
            "DSV4 environment names the engine reads with no door row and no exemption: \
             {undeclared:?}\nAdd a DSV4_DOORS row with its declared default and served-path \
             reach, or an EXEMPT entry saying why it is not a door."
        );
        let stale = stale_exemptions(&names, EXEMPT);
        assert!(
            stale.is_empty(),
            "exemptions for names the engine no longer reads: {stale:?}"
        );
    }

    /// Red arm for the coverage derivation, and the answer to "does this gate
    /// stop covering doors added later?". Add a door to a source, change NOTHING
    /// in the gate, and the gate fails naming the new variable.
    #[test]
    fn a_door_added_to_the_engine_is_caught_without_touching_the_gate() {
        let mut names = engine_door_names();
        let clean = undeclared_door_names(&names, &declared_env_names(), EXEMPT);
        assert!(clean.is_empty(), "{clean:?}");
        // Exactly what a future lane's diff looks like to the scanner.
        let source = r#"
            /// A brand new door nobody declared.
            pub fn dsv4_tomorrows_door_on() -> bool {
                std::env::var("MEMRA_DSV4_TOMORROWS_DOOR").as_deref() == Ok("1")
            }
        "#;
        let scanned = door_names_in_source(source);
        assert_eq!(
            scanned.iter().map(String::as_str).collect::<Vec<_>>(),
            vec!["MEMRA_DSV4_TOMORROWS_DOOR"]
        );
        names.extend(scanned);
        let undeclared = undeclared_door_names(&names, &declared_env_names(), EXEMPT);
        assert_eq!(undeclared, vec!["MEMRA_DSV4_TOMORROWS_DOOR".to_string()]);
        // And the same scan ignores prose that merely mentions the prefix.
        assert!(door_names_in_source("we read MEMRA_DSV4_ names here").is_empty());
    }

    /// The inert doors must sort into their own futures, not into one
    /// population: a door whose program cannot serve is removable on its own
    /// evidence, a door whose program CAN serve waits on that program's verdict,
    /// and a gate instrument's input is neither.
    #[test]
    fn each_inert_door_carries_its_own_disposition() {
        let disposition = |door: &str| {
            DSV4_DOORS
                .iter()
                .find(|row| row.name == door)
                .map(|row| served_disposition(row, &PROGRAM_FACTS))
                .expect(door)
        };
        // Its own FLAGS row says no serving request arms full-token replay.
        const RECLASSIFY: &[&str] = &["replay cadence"];
        // TP/EP cannot serve at all, so a door admitted only there waits on
        // nothing. EMPTY since the norm doors left the registry: the PP-2 port
        // took them out of this list and the served ABBA then deleted them. See
        // the required synthetic subject below, which is what keeps the
        // permanence disposition asserted when this list holds nobody.
        const PERMANENTLY_UNREACHABLE: &[&str] = &[];
        const ENGAGED: &[&str] = &[
            "dense exact-tail transport",
            "dense-fast",
            "HC dot split S16",
        ];

        // COVERAGE, and it is the half that was missing. Until 2026-09-11 this
        // test named doors and nothing checked that it named ALL of them, so
        // `graph split-K` and `split-K-fast` sat in the registry with NO
        // disposition assertion at all: not a vacuity, a blind spot, and the
        // kind that grows by one every time a door lands. The flip deletes those
        // two, which resolves this instance; this assertion is what stops the
        // next one. Found by the norm-fuse PP-2 port lane's coverage check.
        //
        // Keyed to the registry rather than to a count, so a door landing or
        // leaving fails HERE with its own name rather than shifting a number.
        let covered: std::collections::BTreeSet<&str> = RECLASSIFY
            .iter()
            .chain(PERMANENTLY_UNREACHABLE)
            .chain(ENGAGED)
            .copied()
            .collect();
        let registry: std::collections::BTreeSet<&str> =
            DSV4_DOORS.iter().map(|row| row.name).collect();
        assert!(!registry.is_empty(), "an empty registry covers vacuously");
        assert_eq!(
            covered, registry,
            "every registry door needs a disposition assertion here"
        );
        // The three sets must also be disjoint, or a door could be "covered" by
        // being asserted two incompatible ways.
        assert_eq!(
            covered.len(),
            RECLASSIFY.len() + PERMANENTLY_UNREACHABLE.len() + ENGAGED.len(),
            "a door is asserted in more than one disposition set"
        );

        for door in RECLASSIFY {
            assert_eq!(
                disposition(door),
                DoorDisposition::ReclassifyAsGateInput,
                "{door}"
            );
        }
        // EMPTY since the PP-2 port, and empty by DECISION rather than by
        // omission: a `for` over an empty list passes, so this list cannot be
        // the only subject carrying the permanence rule here. The required
        // subject below is, and it is synthetic, so nobody can repair it away.
        for door in PERMANENTLY_UNREACHABLE {
            assert_eq!(
                disposition(door),
                DoorDisposition::PermanentlyUnreachable,
                "{door}"
            );
        }
        assert_eq!(
            served_disposition(&SYNTHETIC_TP_EP_ONLY_ROW, &PROGRAM_FACTS),
            DoorDisposition::PermanentlyUnreachable,
            "the permanence disposition must still be asserted with no real \
             TP/EP-only row left in the registry"
        );
        for door in ENGAGED {
            assert_eq!(disposition(door), DoorDisposition::Engaged, "{door}");
        }
    }

    /// Red arm for the permanence claim, which is the claim doing the most work
    /// in the recommendation. "Permanently unreachable" is keyed to a FACT about
    /// the program (memra #457: TP/EP refuses a batched prime and refuses the
    /// drafter), not hard-coded per door. Teach TP/EP to serve and the same three
    /// doors stop being removable on their own evidence and become an ordinary
    /// admission question.
    #[test]
    fn permanence_is_keyed_to_the_program_fact_not_to_the_door() {
        let if_tp_ep_could_serve = ProgramFacts {
            tp_ep_can_serve: true,
            ..PROGRAM_FACTS
        };
        // This arm ran over the three real TP/EP-only doors until 2026-09-11.
        // The rule is about ANY door admitted only by a program that cannot
        // serve, and it must keep its teeth when no such door happens to exist
        // this week -- which the PP-2 norm port is about to make the case. So it
        // runs on the synthetic stand-in, which nobody can repair.
        const TP_EP_ONLY: DoorRow = DoorRow {
            name: "a future TP/EP-only door",
            env: "MEMRA_DSV4_REPLAY_CADENCE",
            merged: "#374",
            declared_default: DeclaredDefault::On,
            declared_served: DoorState::OffProgram(SYNTHETIC_TP_EP_ONLY),
            declared_bench: DoorState::On(DoorShape::AllRoutedShapes),
            resolve: resolve_synthetic_tp_ep_only,
            admitted_by: AdmittingProgram::TpEpOnly,
            merged_gain_pct: (0.0, 0.0),
            measured_on: "synthetic: the permanence arm, not a claim about any box",
        };
        assert!(
            !(TP_EP_ONLY.resolve)(&SERVED_PROGRAM).engaged(),
            "the arm is vacuous unless the stand-in is inert on the served program"
        );
        assert_eq!(
            served_disposition(&TP_EP_ONLY, &PROGRAM_FACTS),
            DoorDisposition::PermanentlyUnreachable
        );
        assert_eq!(
            served_disposition(&TP_EP_ONLY, &if_tp_ep_could_serve),
            DoorDisposition::PortAdmissionOrRedeclare
        );
    }

    /// The mirror of the adoption. The matrix program is the served one now, so
    /// the interesting program is the one a CLASS gate arms: the scalar
    /// reference executor. Nothing in the registry may depend on the expert
    /// program any more, because the two doors that did are deleted, and this
    /// test is what says so rather than a comment.
    #[test]
    fn the_reference_arm_resolves_every_remaining_door_the_same_way() {
        let reference_served = Dsv4Program {
            name: "served-with-reference",
            matrix_moe: false,
            ..SERVED_PROGRAM
        };
        for row in DSV4_DOORS {
            assert_eq!(
                (row.resolve)(&reference_served),
                (row.resolve)(&SERVED_PROGRAM),
                "{} still resolves on the expert program",
                row.name
            );
        }
        // And the non-vacuity: "same on both" above must be a fact about the
        // EXPERT PROGRAM rather than about a dead resolver, so something must
        // still resolve differently on a different axis.
        //
        // This anchor was `norm-fuse`, named literally, and the PP-2 norm port
        // makes that door resolve identically on both programs, which would turn
        // the assertion red for a reason that has nothing to do with what it
        // checks. Same lesson as the red arms above, one level up. Two anchors
        // now, neither of them a door name.
        assert!(!DSV4_DOORS.is_empty(), "an empty registry loops zero times");
        assert_eq!(DSV4_DOORS.len(), 4);
        // 1. The machinery is alive and program-sensitive, by construction and
        //    permanently: the synthetic stand-in cannot be fixed.
        assert_ne!(
            resolve_synthetic_tp_ep_only(&SERVED_PROGRAM),
            resolve_synthetic_tp_ep_only(&TUNED_BENCH_PROGRAM)
        );
        // 2. And the REGISTRY still contains at least one such row, computed
        //    rather than named, so this tracks the registry through door churn
        //    instead of pinning whichever door happens to differ this month.
        assert!(
            DSV4_DOORS
                .iter()
                .any(|row| (row.resolve)(&SERVED_PROGRAM) != (row.resolve)(&TUNED_BENCH_PROGRAM)),
            "no registry row resolves differently on the two programs, so the \
             expert-program claim above is vacuous"
        );
    }

    /// The honest scope of what the merged receipts establish. The dev pair
    /// drifts about 5% within one arm across a run, so a door merged at under
    /// about 2% is below that instrument's resolution. This is not a
    /// re-litigation of any merge, it is a list, and the list rots if nobody
    /// writes it down.
    #[test]
    fn the_doors_merged_below_the_dev_pair_instrument_floor_are_named() {
        let below: Vec<_> = DSV4_DOORS
            .iter()
            .filter(|row| below_instrument_floor(row, DEV_PAIR_INSTRUMENT_FLOOR_PCT))
            .map(|row| row.name)
            .collect();
        assert_eq!(
            below,
            vec![
                "replay cadence",
                "dense exact-tail transport",
                "dense-fast",
            ]
        );
        // The doors ABOVE the floor used to be named literally here, which is
        // the same disease as
        // the red arms one level up: an assertion pinned to a door NAME to prove
        // something about the MECHANISM. Deleting or renaming either door would
        // have broken it for a reason unrelated to what it checks. Stated as the
        // partition property instead, which tracks the registry through churn.
        let above: Vec<_> = DSV4_DOORS
            .iter()
            .filter(|row| !below_instrument_floor(row, DEV_PAIR_INSTRUMENT_FLOOR_PCT))
            .map(|row| row.name)
            .collect();
        // Non-vacuity in BOTH directions: a floor that caught everything, or
        // nothing, would make one of these lists empty and the split
        // meaningless.
        assert!(!below.is_empty(), "no door is below the floor");
        assert!(!above.is_empty(), "no door is above the floor");
        assert_eq!(
            below.len() + above.len(),
            DSV4_DOORS.len(),
            "below and above must partition the registry"
        );
    }

    /// The strongest removal case: no evidence for the default in EITHER
    /// direction. Inert on the served path, so the default buys nothing, AND
    /// merged below the floor of the pair it was qualified on, so the number
    /// that justified it is not one that pair can resolve.
    #[test]
    fn doors_with_no_evidence_in_either_direction_are_named() {
        assert_eq!(
            doors_without_evidence_in_either_direction(
                DSV4_DOORS,
                &PROGRAM_FACTS,
                DEV_PAIR_INSTRUMENT_FLOOR_PCT
            ),
            // norm-fuse and norm-fuse2 left this list with the PP-2 port: they
            // engage on the served path now, so the "no evidence in either
            // direction" complaint no longer applies to them. What remains is
            // the weaker floor complaint, asserted separately below.
            vec!["replay cadence"]
        );
        // Engaged, but on evidence the pair cannot resolve: a different and
        // weaker complaint, kept separate so it cannot be quoted as the first.
        assert_eq!(
            engaged_but_below_the_floor(DSV4_DOORS, DEV_PAIR_INSTRUMENT_FLOOR_PCT),
            vec!["dense exact-tail transport", "dense-fast"]
        );
        // Red arm: the sets are computed from the floor, not asserted. Drop the
        // floor below every merged magnitude and both sets must empty out.
        assert!(
            doors_without_evidence_in_either_direction(DSV4_DOORS, &PROGRAM_FACTS, 0.1).is_empty()
        );
        assert!(engaged_but_below_the_floor(DSV4_DOORS, 0.1).is_empty());
        // And a floor above everything catches every inert door, and ONLY the
        // default-ON ones: a default-OFF door is off by decision, so it is not
        // a door with no evidence for its default.
        // Both sides are DERIVED, so equality between two empty lists would pass
        // while proving nothing (memra #482 audit, the empty-subject vacuity
        // class). This lane's port takes the inert list from four rows to one,
        // so the day it reaches zero this assertion must stop passing quietly.
        let at_any_floor =
            doors_without_evidence_in_either_direction(DSV4_DOORS, &PROGRAM_FACTS, 100.0);
        assert!(
            !at_any_floor.is_empty(),
            "empty subject set: a floor above every magnitude must still catch the inert doors"
        );
        assert_eq!(at_any_floor, inert_default_on_doors(DSV4_DOORS));
        // Red arm for the default-ON filter itself, and the reason it is a
        // synthetic row rather than a registry one: under the owner's
        // 2026-09-10 ruling a same-class win with a clean receipt becomes the
        // code, so the registry is not expected to carry a default-OFF door for
        // long, and a check that only has teeth while one happens to exist is a
        // check that quietly stops having them. `MEMRA_DSV4_DENSE_TILE` was the
        // door that exposed the defect and it is gone (memra #471); this row
        // stands in for the next one, inert and below any floor.
        const OFF_DOOR: &[DoorRow] = &[DoorRow {
            name: "a default-OFF door, inert and unmeasurable",
            env: "MEMRA_DSV4_REPLAY_CADENCE",
            merged: "#374",
            declared_default: DeclaredDefault::Off,
            declared_served: DoorState::OffProgram(SYNTHETIC_TP_EP_ONLY),
            declared_bench: DoorState::On(DoorShape::AllRoutedShapes),
            resolve: resolve_synthetic_tp_ep_only,
            admitted_by: AdmittingProgram::TpEpOnly,
            merged_gain_pct: (0.0, 0.0),
            measured_on: "synthetic: the floor arm, not a claim about any box",
        }];
        assert!(
            !(OFF_DOOR[0].resolve)(&SERVED_PROGRAM).engaged()
                && below_instrument_floor(&OFF_DOOR[0], DEV_PAIR_INSTRUMENT_FLOOR_PCT),
            "the arm is vacuous unless the stand-in row is both inert and below the floor"
        );
        assert!(
            doors_without_evidence_in_either_direction(
                OFF_DOOR,
                &PROGRAM_FACTS,
                DEV_PAIR_INSTRUMENT_FLOOR_PCT
            )
            .is_empty(),
            "a door that is OFF BY DECISION is not a door with no evidence for its default"
        );
        // And the same row defaulting ON is caught, so the filter is what
        // excluded it rather than some other term.
        const ON_DOOR: &[DoorRow] = &[DoorRow {
            name: "the same row, defaulting ON",
            env: "MEMRA_DSV4_REPLAY_CADENCE",
            merged: "#374",
            declared_default: DeclaredDefault::On,
            declared_served: DoorState::OffProgram(SYNTHETIC_TP_EP_ONLY),
            declared_bench: DoorState::On(DoorShape::AllRoutedShapes),
            resolve: resolve_synthetic_tp_ep_only,
            admitted_by: AdmittingProgram::TpEpOnly,
            merged_gain_pct: (0.0, 0.0),
            measured_on: "synthetic: the floor arm, not a claim about any box",
        }];
        assert_eq!(
            doors_without_evidence_in_either_direction(
                ON_DOOR,
                &PROGRAM_FACTS,
                DEV_PAIR_INSTRUMENT_FLOOR_PCT
            )
            .len(),
            1
        );
    }
}
