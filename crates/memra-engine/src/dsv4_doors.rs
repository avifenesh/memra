//! DSV4 door reach: what a door's declared default actually resolves to on the
//! program a request runs under.
//!
//! Why this module exists (memra #454, #458). Every DSV4 door merged since #374
//! was measured honestly, on the tuned bench program every `dsv4_*_gate` binary
//! pins: attention TP/EP, `MEMRA_DSV4_MOE_PROGRAM=matrix`, the drafter off, the
//! small-kernel diet on, and the gate-only fused-GU arm turned on AFTER load
//! through `Dsv4Gpu::set_grouped_gu_fuse_for_gate`. What was never checked is
//! whether that program is the one a customer request takes. It is not: the
//! served program is PP-2 with the reference expert program, a resident DSpark
//! drafter and chunked prefill, and the engine refuses the union of the two in
//! two independent places: the TP/EP topology guard refuses MTP/DSpark state,
//! and `prefill_with_cache_chunked` refuses a batched prime at all under
//! `topology.is_tp_ep()`, "DSV4 TP/EP vertical slice currently admits only a
//! single-token prime; batched replicated cache hydration is not wired". So
//! TP/EP can neither chunk nor serve, and either refusal alone makes the two
//! programs disjoint. Both are cited by FUNCTION and refusal text on purpose:
//! line numbers in this file move under every lane that touches it. Six of the nine
//! merged default-ON doors, including the two largest wins, cannot engage there,
//! and they go inert SILENTLY because an unset value resolves to `Ok(admitted)`
//! with `admitted == false`.
//!
//! So this module holds three things, and the load path uses the same code the
//! gate does:
//!
//! 1. `Dsv4Program`, a description of a running program, with the two constants
//!    that matter: `SERVED_PROGRAM` (what a customer request takes today) and
//!    `TUNED_BENCH_PROGRAM` (what every DSV4 gate binary pins).
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

/// What the door's state means for what to DO about it. Printed in the load
/// receipt and pinned per door by the gate, so the three cases cannot collapse
/// into one recommendation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DoorDisposition {
    /// Engaged. Nothing to decide.
    Engaged,
    /// The default is a claim this program does not honour: port the admission
    /// predicate to this program, or re-declare the default to match reality.
    PortAdmissionOrRedeclare,
    /// The door belongs to another program. It resolves itself if that program
    /// becomes the default, and leaves with it if it does not.
    FollowsTheProgramDecision,
    /// Nothing about a serving program can make this door engage: it is a gate
    /// instrument's input filed as a product door.
    ReclassifyAsGateInput,
    /// Refused at load. The configuration is illegal, not silently degraded.
    RefusedConfiguration,
}

impl DoorState {
    pub fn engaged(self) -> bool {
        matches!(self, DoorState::On(_))
    }

    pub fn disposition(self) -> DoorDisposition {
        match self {
            DoorState::On(_) => DoorDisposition::Engaged,
            DoorState::Off => DoorDisposition::PortAdmissionOrRedeclare,
            DoorState::RefusedAtLoad => DoorDisposition::RefusedConfiguration,
            DoorState::OffProgram(_) => DoorDisposition::FollowsTheProgramDecision,
            DoorState::NoServingCaller(_) => DoorDisposition::ReclassifyAsGateInput,
        }
    }
}

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
    /// `MEMRA_DSV4_MOE_PROGRAM=matrix`: the matrix expert executor
    /// (`dsv4_grouped.rs`). The reference program never enters that file.
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
}

/// What a customer request runs today: PP-2 (`Dsv4TopologyPlan::pp_ep`, so
/// `is_tp_ep()` is false), the reference expert program, DSpark resident,
/// chunked prefill, and no gate-only arms.
pub const SERVED_PROGRAM: Dsv4Program = Dsv4Program {
    name: "served",
    tp_ep: false,
    chains_f32: true,
    dots_f32: true,
    matrix_moe: false,
    drafter_resident: true,
    gate_armed_gu_fuse: false,
    hc_geometry_24x16384: true,
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
};

// ---------------------------------------------------------------------------
// Policy functions. `Dsv4Gpu::load` calls these; so does the registry below.
// ---------------------------------------------------------------------------

/// The TP/EP f32x admission term the norm doors share. One function so the load
/// site and the door registry cannot disagree about what "admitted" means.
pub fn norm_admitted(tp_ep: bool, chains_f32: bool) -> bool {
    tp_ep && chains_f32
}

/// Default ON only in the qualified TP/EP f32 domain. An explicit unsupported ON
/// request still refuses instead of silently admitting an unqualified path.
pub fn norm_fuse_environment_policy(
    value: Result<&str, &std::env::VarError>,
    admitted: bool,
) -> Res<bool> {
    match value {
        Err(std::env::VarError::NotPresent) => Ok(admitted),
        Ok("0") => Ok(false),
        Ok("1") if admitted => Ok(true),
        Ok("1") => Err("norm fusion requires TP/EP f32x".into()),
        _ => Err("MEMRA_DSV4_NORM_FUSE requires 0 or 1".into()),
    }
}

pub fn norm_fuse2_environment_policy(
    value: Result<&str, &std::env::VarError>,
    admitted: bool,
) -> Res<bool> {
    match value {
        Err(std::env::VarError::NotPresent) => Ok(admitted),
        Ok("0") => Ok(false),
        Ok("1") if admitted => Ok(true),
        Ok("1") => Err("norm fusion2 requires TP/EP f32x".into()),
        _ => Err("MEMRA_DSV4_NORM_FUSE2 requires 0 or 1".into()),
    }
}

/// Default ON under an admitted norm2 door since the 2026-09-10 model campaign;
/// an explicit `0` is the rollback seam to the single-CTA kernel. Without the
/// norm2 door the pack has no call site, so an unset value degrades to OFF
/// rather than refusing every composed-off launch, while an explicit `1` without
/// the door stays a configuration error instead of a silent no-op. `admitted` is
/// the norm2 door's own resolved value.
pub fn norm2_wide_environment_policy(
    value: Result<&str, &std::env::VarError>,
    admitted: bool,
) -> Res<bool> {
    match value {
        Err(std::env::VarError::NotPresent) => Ok(admitted),
        Ok("0") => Ok(false),
        Ok("1") if admitted => Ok(true),
        Ok("1") => Err("norm2 wide pack requires MEMRA_DSV4_NORM_FUSE2=1".into()),
        _ => Err("MEMRA_DSV4_NORM2_WIDE requires 0 or 1".into()),
    }
}

/// memra #458. The matrix program's plain gate/up split-K arm computes gate and
/// up in ONE fused launch, so it requires the fused-GU arm; `gate_up` refuses
/// with "split-K requires plain fused GU" when it is not on. That arm is
/// gate-only: `MEMRA_F16G_GU_FUSE` is OFF and research-only, and the only way to
/// turn it on is `Dsv4Gpu::set_grouped_gu_fuse_for_gate`, which every gate
/// binary calls after load and no serving process calls at all. Split-K has been
/// default ON since #392, so `MEMRA_DSV4_MOE_PROGRAM=matrix` on a serving stack
/// boots CLEAN and then fails 100% of requests as `engine_error` 500s.
///
/// This function is the load-time refusal that replaces that outage. It joins
/// the four sibling refusals in `validate_matrix_program`. The per-request check
/// in `dsv4_grouped::gate_up` stays where it is as the backstop; it is no longer
/// the thing an operator finds out from.
pub fn matrix_splitk_admission(matrix_moe: bool, splitk_on: bool, gate_armed: bool) -> Res<()> {
    if matrix_moe && splitk_on && !gate_armed {
        return Err(
            "MEMRA_DSV4_MOE_PROGRAM=matrix with split-K ON (MEMRA_DSV4_MOE_M1_SPLITK unset or \
             graph, default ON since #392) requires the gate-only fused-GU arm that only a gate \
             binary arms: set MEMRA_DSV4_MOE_M1_SPLITK=0 to serve the matrix program, or call \
             memra_engine::arm_matrix_splitk_door_for_gate() before load in a bench process"
                .into(),
        );
    }
    Ok(())
}

/// Set by a gate binary BEFORE `Dsv4Gpu::load`, and by nothing else. It is what
/// keeps the matrix split-K combination a bench arm in the strong sense: a
/// serving process that inherits the default-ON door with the matrix program
/// REFUSES to load instead of failing every request. There is no environment
/// variable that sets this.
static MATRIX_SPLITK_ARMED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Permit this process to run the matrix expert program with split-K ON. Gate
/// binaries call it before load, after which they arm the fused-GU seam the arm
/// needs; no serving path calls it.
pub fn arm_matrix_splitk_door_for_gate() {
    MATRIX_SPLITK_ARMED.store(true, std::sync::atomic::Ordering::SeqCst);
}

pub fn matrix_splitk_door_armed() -> bool {
    MATRIX_SPLITK_ARMED.load(std::sync::atomic::Ordering::SeqCst)
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
}

static NOT_PRESENT: std::env::VarError = std::env::VarError::NotPresent;

fn unset() -> Result<&'static str, &'static std::env::VarError> {
    Err(&NOT_PRESENT)
}

fn resolve_norm_fuse(p: &Dsv4Program) -> DoorState {
    match norm_fuse_environment_policy(unset(), norm_admitted(p.tp_ep, p.chains_f32)) {
        Ok(true) => DoorState::On(DoorShape::AllRoutedShapes),
        Ok(false) => DoorState::Off,
        Err(_) => DoorState::RefusedAtLoad,
    }
}

fn resolve_norm_fuse2(p: &Dsv4Program) -> DoorState {
    match norm_fuse2_environment_policy(unset(), norm_admitted(p.tp_ep, p.chains_f32)) {
        Ok(true) => DoorState::On(DoorShape::AllRoutedShapes),
        Ok(false) => DoorState::Off,
        Err(_) => DoorState::RefusedAtLoad,
    }
}

fn resolve_norm2_wide(p: &Dsv4Program) -> DoorState {
    let fuse2 = resolve_norm_fuse2(p).engaged();
    match norm2_wide_environment_policy(unset(), fuse2) {
        Ok(true) => DoorState::On(DoorShape::AllRoutedShapes),
        Ok(false) => DoorState::Off,
        Err(_) => DoorState::RefusedAtLoad,
    }
}

const MATRIX_EXECUTOR_ONLY: &str = "the arm lives in the matrix expert executor (dsv4_grouped.rs); the reference expert \
     program the served path runs never enters that file";

fn resolve_graph_splitk(p: &Dsv4Program) -> DoorState {
    if !p.matrix_moe {
        return DoorState::OffProgram(MATRIX_EXECUTOR_ONLY);
    }
    // Default ON since #392, and then #458: the plain gate/up arm needs the
    // gate-only fused-GU seam, so an unarmed matrix process refuses at load.
    match matrix_splitk_admission(true, true, p.gate_armed_gu_fuse) {
        Ok(()) => DoorState::On(DoorShape::DecodeOnlyM1),
        Err(_) => DoorState::RefusedAtLoad,
    }
}

fn resolve_splitk_fast(p: &Dsv4Program) -> DoorState {
    // The paired-fetch entries are a choice of split-K entry family: no split-K,
    // no call site. Same reachability as its parent, never a wider one.
    resolve_graph_splitk(p)
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
/// registries claim for each.
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
    },
    DoorRow {
        name: "dense exact-tail transport",
        env: "MEMRA_DSV4_DENSE_EXACT_TAIL",
        merged: "#374",
        declared_default: DeclaredDefault::On,
        declared_served: DoorState::On(DoorShape::DecodeOnlyM1),
        declared_bench: DoorState::On(DoorShape::DecodeOnlyM1),
        resolve: resolve_dense_exact_tail,
    },
    DoorRow {
        name: "graph split-K",
        env: "MEMRA_DSV4_MOE_M1_SPLITK",
        merged: "#392",
        declared_default: DeclaredDefault::On,
        declared_served: DoorState::OffProgram(MATRIX_EXECUTOR_ONLY),
        declared_bench: DoorState::On(DoorShape::DecodeOnlyM1),
        resolve: resolve_graph_splitk,
    },
    DoorRow {
        name: "dense-fast",
        env: "MEMRA_DSV4_DENSE_FAST",
        merged: "#404",
        declared_default: DeclaredDefault::On,
        declared_served: DoorState::On(DoorShape::DecodeOnlyM1),
        declared_bench: DoorState::On(DoorShape::DecodeOnlyM1),
        resolve: resolve_dense_fast,
    },
    DoorRow {
        name: "norm-fuse",
        env: "MEMRA_DSV4_NORM_FUSE",
        merged: "#404",
        declared_default: DeclaredDefault::On,
        declared_served: DoorState::Off,
        declared_bench: DoorState::On(DoorShape::AllRoutedShapes),
        resolve: resolve_norm_fuse,
    },
    DoorRow {
        name: "HC dot split S16",
        env: "MEMRA_DSV4_HC_DOT_SPLIT",
        merged: "#418",
        declared_default: DeclaredDefault::On,
        declared_served: DoorState::On(DoorShape::AllRoutedShapes),
        declared_bench: DoorState::On(DoorShape::AllRoutedShapes),
        resolve: resolve_hc_dot_split,
    },
    DoorRow {
        name: "split-K-fast",
        env: "MEMRA_DSV4_SPLITK_FAST",
        merged: "#425",
        declared_default: DeclaredDefault::On,
        declared_served: DoorState::OffProgram(MATRIX_EXECUTOR_ONLY),
        declared_bench: DoorState::On(DoorShape::DecodeOnlyM1),
        resolve: resolve_splitk_fast,
    },
    DoorRow {
        name: "norm-fuse2",
        env: "MEMRA_DSV4_NORM_FUSE2",
        merged: "#426",
        declared_default: DeclaredDefault::On,
        declared_served: DoorState::Off,
        declared_bench: DoorState::On(DoorShape::AllRoutedShapes),
        resolve: resolve_norm_fuse2,
    },
    DoorRow {
        name: "norm2-wide",
        env: "MEMRA_DSV4_NORM2_WIDE",
        merged: "#430",
        declared_default: DeclaredDefault::On,
        declared_served: DoorState::Off,
        declared_bench: DoorState::On(DoorShape::AllRoutedShapes),
        resolve: resolve_norm2_wide,
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
/// program. This is the number memra #454 is about; it is six today, and the
/// gate pins it so it cannot grow silently.
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
                (row.resolve)(program).disposition()
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
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
            name: "norm2-wide, claiming the served path",
            env: "MEMRA_DSV4_NORM2_WIDE",
            merged: "#430",
            declared_default: DeclaredDefault::On,
            // The lie: this door cannot engage on PP-2 without norm-fuse2.
            declared_served: DoorState::On(DoorShape::AllRoutedShapes),
            declared_bench: DoorState::On(DoorShape::AllRoutedShapes),
            resolve: resolve_norm2_wide,
        }];
        let violations = declaration_violations(LIAR);
        assert_eq!(violations.len(), 1, "{violations:?}");
        assert_eq!(violations[0].program, "served");
        assert_eq!(violations[0].resolved, DoorState::Off);
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
        let differing: Vec<_> = DSV4_DOORS
            .iter()
            .filter(|row| (row.resolve)(&SERVED_PROGRAM) != (row.resolve)(&TUNED_BENCH_PROGRAM))
            .map(|row| row.name)
            .collect();
        assert_eq!(differing.len(), 6, "{differing:?}");
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
    /// this number, and all three are decisions that must be written down.
    #[test]
    fn six_default_on_doors_are_inert_on_the_served_path() {
        assert_eq!(
            inert_default_on_doors(DSV4_DOORS),
            vec![
                "replay cadence",
                "graph split-K",
                "norm-fuse",
                "split-K-fast",
                "norm-fuse2",
                "norm2-wide",
            ]
        );
    }

    /// memra #458, red arm: the combination that failed 100% of requests on the
    /// prod-candidate box must refuse at load, and must keep being admitted for
    /// the armed bench process that measured the +10.2% the door merged on.
    #[test]
    fn matrix_plus_default_on_splitk_refuses_at_load_unless_a_gate_armed_it() {
        let refusal = matrix_splitk_admission(true, true, false).unwrap_err();
        assert!(refusal.contains("MEMRA_DSV4_MOE_M1_SPLITK=0"), "{refusal}");
        assert!(
            refusal.contains("arm_matrix_splitk_door_for_gate"),
            "{refusal}"
        );
        // The documented workaround, and the bench program, both still load.
        assert_eq!(matrix_splitk_admission(true, false, false), Ok(()));
        assert_eq!(matrix_splitk_admission(true, true, true), Ok(()));
        // The reference expert program never had the problem.
        assert_eq!(matrix_splitk_admission(false, true, false), Ok(()));
    }

    /// Every door in the registry owes a `docs/FLAGS.md` row. A door added to the
    /// engine with no registry entry is caught by
    /// `every_dsv4_door_name_in_the_engine_is_declared_or_exempt` below; this is
    /// the other half, so a registry row cannot document itself.
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
        (
            "MEMRA_DSV4_MOE_PROGRAM",
            "program selector: the disjointness itself (memra #461)",
        ),
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
            "gate-only research arm; memra #458 is what its gate-only status costs",
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

    /// The three not-engaged cases must stay apart: they have three different
    /// fixes, and collapsing them is how six doors get one recommendation.
    #[test]
    fn each_inert_door_carries_its_own_disposition() {
        let disposition = |door: &str| {
            DSV4_DOORS
                .iter()
                .find(|row| row.name == door)
                .map(|row| (row.resolve)(&SERVED_PROGRAM).disposition())
                .expect(door)
        };
        // Its own FLAGS row says no serving request arms full-token replay.
        assert_eq!(
            disposition("replay cadence"),
            DoorDisposition::ReclassifyAsGateInput
        );
        // These live in the matrix executor: memra #461 decides them.
        assert_eq!(
            disposition("graph split-K"),
            DoorDisposition::FollowsTheProgramDecision
        );
        assert_eq!(
            disposition("split-K-fast"),
            DoorDisposition::FollowsTheProgramDecision
        );
        // These have call sites on the served program and an admission predicate
        // that says no: porting or re-declaring is the fix.
        for door in ["norm-fuse", "norm-fuse2", "norm2-wide"] {
            assert_eq!(
                disposition(door),
                DoorDisposition::PortAdmissionOrRedeclare,
                "{door}"
            );
        }
        // If the matrix program becomes the served expert program, the two
        // FollowsTheProgramDecision doors resolve themselves, and the other four
        // do not move. That is the whole reason the cases are kept apart.
        let matrix_served = Dsv4Program {
            name: "served-with-matrix",
            matrix_moe: true,
            ..SERVED_PROGRAM
        };
        assert_eq!(
            (DSV4_DOORS
                .iter()
                .find(|row| row.name == "graph split-K")
                .unwrap()
                .resolve)(&matrix_served),
            // Still refused while the fused-GU arm stays gate-only (memra #458),
            // which is a configuration error an operator can see, not silence.
            DoorState::RefusedAtLoad
        );
        assert_eq!(
            (DSV4_DOORS
                .iter()
                .find(|row| row.name == "norm2-wide")
                .unwrap()
                .resolve)(&matrix_served),
            DoorState::Off
        );
    }
}
