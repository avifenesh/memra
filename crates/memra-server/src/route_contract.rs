//! Serve routes are registered implementations of the server's policy contract (memra#504).
//!
//! The `Cmd::Generate -> Event` contract is genuinely shared between routes: billing, auth, the
//! ledger, tool parsing and every API dialect ride `PromptUsage` + `Done` and are correct on the
//! central worker and on the DSv4 thread alike. Every policy that instead reads a SIDE CHANNEL
//! (`worker::Metrics`, the health beat and the prime odometer, memory admission, the rewrite
//! bundle, prime fairness, the lane cap mirror) has exactly one writer, and a second route is not
//! it: each of those surfaces became a silent no-op on DSv4 and was found separately, by
//! accident, weeks apart (#449, #500, #501, #502, #503).
//!
//! The rule this module enforces: routing traffic to a path that publishes no liveness, no
//! metrics and no admission cost is a boot-time refusal, not a runtime surprise. Every route
//! registers a [`RouteContract`] that declares EVERY [`PolicySurface`] as either implemented
//! (with a call-site token the wiring test greps in the route's source) or refused by name (with
//! the issue that owns the gap). A surface left undeclared fails [`RouteRegistry::check`], which
//! runs before `ready_tx` fires, so the process exits `FATAL: worker init failed` instead of
//! serving. A refused surface whose policy an operator has actually armed (today:
//! `MEMRA_REWRITE_BUNDLE` beside a DSv4 route, #449) is the same refusal with the route named.
//!
//! What this module does NOT do: implement the DSv4 policies. Each issue flips its declaration
//! from `Refused` to `Implemented` where the route gains the call site: #500 (progress), #501
//! (occupancy, service metrics) and #503 (memory cost) have; #449 (rewrite bundle) has not, and
//! the registry names it at every boot.

use std::collections::BTreeMap;

/// Every server policy that reads a side channel a route must write. Adding a variant here
/// forces every registered contract to declare it (the exhaustiveness check walks `ALL`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PolicySurface {
    /// How many requests the route runs concurrently (the admission cap mirror, `lane_cap`).
    Capacity,
    /// In-flight and queued counts a shed / retry estimate can read.
    Occupancy,
    /// Liveness attestation while working: the health beat or the prime odometer.
    Progress,
    /// Who catches a per-request or per-route failure and what the client sees.
    FaultOwnership,
    /// How the route drains and stops on SIGTERM / worker shutdown.
    ShutdownOwnership,
    /// Per-request memory cost published before allocation (`admit_memory`, the admission book).
    MemoryCost,
    /// Whether an installed rewrite bundle (`MEMRA_REWRITE_BUNDLE`) governs this route's programs.
    RewriteQualification,
    /// Cooperative prime service (the walker's yield/resume) for this route's sessions.
    PrimeFairness,
    /// `worker::Metrics` rows (`completed`, `tokens_out`, `step_p50_ms`) the queue estimator reads.
    ServiceMetrics,
}

impl PolicySurface {
    pub const ALL: [PolicySurface; 9] = [
        PolicySurface::Capacity,
        PolicySurface::Occupancy,
        PolicySurface::Progress,
        PolicySurface::FaultOwnership,
        PolicySurface::ShutdownOwnership,
        PolicySurface::MemoryCost,
        PolicySurface::RewriteQualification,
        PolicySurface::PrimeFairness,
        PolicySurface::ServiceMetrics,
    ];

    pub fn name(self) -> &'static str {
        match self {
            PolicySurface::Capacity => "capacity",
            PolicySurface::Occupancy => "occupancy",
            PolicySurface::Progress => "progress",
            PolicySurface::FaultOwnership => "fault-ownership",
            PolicySurface::ShutdownOwnership => "shutdown-ownership",
            PolicySurface::MemoryCost => "memory-cost",
            PolicySurface::RewriteQualification => "rewrite-qualification",
            PolicySurface::PrimeFairness => "prime-fairness",
            PolicySurface::ServiceMetrics => "service-metrics",
        }
    }

    /// The environment knob that ARMS this policy, when one exists. A refused surface whose knob
    /// is set is a boot refusal: the operator asked for a policy the route cannot honor.
    pub fn arming_env(self) -> Option<&'static str> {
        match self {
            PolicySurface::RewriteQualification => Some("MEMRA_REWRITE_BUNDLE"),
            _ => None,
        }
    }
}

/// What a route says about one surface.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolicyDecl {
    /// The route writes the side channel. `evidence` is a token that must appear, outside
    /// comments, in the route's source file; the wiring test greps it so a declaration cannot
    /// outlive its call site.
    Implemented { evidence: &'static str },
    /// The route does not write it and says so, naming the issue that owns the gap.
    Refused { reason: &'static str },
}

/// The route's concurrency, the number the admission cap mirror must read (#501, #502).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouteCapacity {
    /// One request at a time (a FIFO channel is the queue).
    Serial,
    /// Up to this many concurrent sessions.
    Sessions(usize),
}

impl RouteCapacity {
    pub fn concurrency(self) -> usize {
        match self {
            RouteCapacity::Serial => 1,
            RouteCapacity::Sessions(n) => n.max(1),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RouteContract {
    /// The served model name this route answers for.
    pub model: String,
    /// The route implementation: `hybrid-worker` or `dsv4-thread`.
    pub kind: &'static str,
    /// The source file whose comment-stripped text must contain every `Implemented` evidence.
    pub source_file: &'static str,
    pub capacity: RouteCapacity,
    decls: BTreeMap<PolicySurface, PolicyDecl>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RouteContractError {
    /// A surface the contract never declared: the omission the class of bugs lived in.
    Undeclared {
        model: String,
        kind: &'static str,
        surfaces: Vec<PolicySurface>,
    },
    /// A refused surface whose arming knob is set.
    ArmedButRefused {
        model: String,
        kind: &'static str,
        surface: PolicySurface,
        env: &'static str,
        reason: &'static str,
    },
}

impl std::fmt::Display for RouteContractError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Undeclared {
                model,
                kind,
                surfaces,
            } => write!(
                f,
                "route contract for model {model:?} ({kind}) declares nothing for {:?}; every \
                 serve route must implement or refuse every policy surface by name before it may \
                 take traffic (memra#504)",
                surfaces.iter().map(|s| s.name()).collect::<Vec<_>>()
            ),
            Self::ArmedButRefused {
                model,
                kind,
                surface,
                env,
                reason,
            } => write!(
                f,
                "{env} is set but no route in this process honors the {} policy; route \
                 {model:?} ({kind}) refuses it: {reason}. Unset {env} or serve a model on a \
                 route that honors it; a policy no route can read is refused at boot, not \
                 ignored at runtime (memra#504)",
                surface.name()
            ),
        }
    }
}

impl std::error::Error for RouteContractError {}

impl RouteContract {
    pub fn new(
        model: impl Into<String>,
        kind: &'static str,
        source_file: &'static str,
        capacity: RouteCapacity,
    ) -> Self {
        Self {
            model: model.into(),
            kind,
            source_file,
            capacity,
            decls: BTreeMap::new(),
        }
    }

    pub fn implemented(mut self, surface: PolicySurface, evidence: &'static str) -> Self {
        self.decls
            .insert(surface, PolicyDecl::Implemented { evidence });
        self
    }

    pub fn refused(mut self, surface: PolicySurface, reason: &'static str) -> Self {
        self.decls.insert(surface, PolicyDecl::Refused { reason });
        self
    }

    pub fn decl(&self, surface: PolicySurface) -> Option<&PolicyDecl> {
        self.decls.get(&surface)
    }

    pub fn declarations(&self) -> impl Iterator<Item = (PolicySurface, &PolicyDecl)> {
        self.decls.iter().map(|(s, d)| (*s, d))
    }

    /// Every surface declared (the omission the class of bugs lived in).
    pub fn check_declared(&self) -> Result<(), RouteContractError> {
        let undeclared: Vec<PolicySurface> = PolicySurface::ALL
            .iter()
            .copied()
            .filter(|s| !self.decls.contains_key(s))
            .collect();
        if !undeclared.is_empty() {
            return Err(RouteContractError::Undeclared {
                model: self.model.clone(),
                kind: self.kind,
                surfaces: undeclared,
            });
        }
        Ok(())
    }

    /// One boot line per route: what it implements, what it refuses and why.
    pub fn describe(&self) -> String {
        let implemented: Vec<&str> = self
            .decls
            .iter()
            .filter_map(|(s, d)| matches!(d, PolicyDecl::Implemented { .. }).then_some(s.name()))
            .collect();
        let refused: Vec<String> = self
            .decls
            .iter()
            .filter_map(|(s, d)| match d {
                PolicyDecl::Refused { reason } => Some(format!("{}: {reason}", s.name())),
                _ => None,
            })
            .collect();
        format!(
            "[route-contract] model={:?} route={} capacity={} implemented=[{}] refused=[{}]",
            self.model,
            self.kind,
            match self.capacity {
                RouteCapacity::Serial => "serial".to_string(),
                RouteCapacity::Sessions(n) => format!("{n} sessions"),
            },
            implemented.join(","),
            refused.join("; ")
        )
    }

    /// The central worker route (`worker.rs` run loop): every surface implemented. The evidence
    /// tokens are the call sites; `route_contract::tests` greps them in comment-stripped source.
    pub fn hybrid_worker(model: impl Into<String>, sessions: usize) -> Self {
        Self::new(
            model,
            "hybrid-worker",
            "worker.rs",
            RouteCapacity::Sessions(sessions),
        )
        .implemented(PolicySurface::Capacity, "MEMRA_MAX_SESSIONS")
        .implemented(PolicySurface::Occupancy, "m.admission_inflight = ")
        .implemented(PolicySurface::Progress, "health.beat_busy()")
        .implemented(PolicySurface::FaultOwnership, "request_fault_guard(")
        .implemented(
            PolicySurface::ShutdownOwnership,
            "host_capture_drain_at_shutdown(",
        )
        .implemented(PolicySurface::MemoryCost, "admit_memory::decide(")
        .implemented(
            PolicySurface::RewriteQualification,
            "install_rewrite_bundle(",
        )
        .implemented(PolicySurface::PrimeFairness, "PrimeService::default()")
        .implemented(PolicySurface::ServiceMetrics, "m.step_p50_ms = ")
    }
}

impl RouteContract {
    /// The DSv4 serving thread (`dsv4_serve.rs`): one request at a time on a FIFO channel. Every
    /// surface the thread does not write is refused by name with the issue that owns it, so the
    /// registry prints it at boot and refuses an operator who arms one of those policies.
    /// Flipping a line from `refused` to `implemented` is the registration side of each issue
    /// (#500, #501 and #503 done; #449 open). Declared here and not in `dsv4_serve.rs` so the
    /// wiring test's grep of that file cannot be satisfied by the declaration text itself.
    pub fn dsv4_thread(model: impl Into<String>) -> Self {
        Self::dsv4_thread_sessions(model, 1)
    }

    /// The DSv4 thread with `sessions` pipelined serving lanes (memra #667, `MEMRA_DSV4_SESSIONS`):
    /// one lane is the serial route above; more lanes share the queue and the launch turn.
    pub fn dsv4_thread_sessions(model: impl Into<String>, sessions: usize) -> Self {
        let capacity = if sessions > 1 {
            RouteCapacity::Sessions(sessions)
        } else {
            RouteCapacity::Serial
        };
        Self::new(model, "dsv4-thread", "dsv4_serve.rs", capacity)
            .implemented(
                PolicySurface::Capacity,
                "std::sync::mpsc::channel::<Box<Request>>()",
            )
            // memra#501: `running` rises at dequeue on the route's own book
            // (route_telemetry::RouteLoad); the HTTP layer prices waits and the trio from it.
            .implemented(PolicySurface::Occupancy, "load.begin()")
            // memra#500: the thread's prime odometer stamps route to its own RouteHealth.
            .implemented(
                PolicySurface::Progress,
                "memra_engine::progress::ProgressSinkScope::install(",
            )
            .implemented(PolicySurface::FaultOwnership, "std::panic::catch_unwind(")
            .implemented(
                PolicySurface::ShutdownOwnership,
                "let Ok(mut req) = next else { break };",
            )
            // memra#503: the route's own per-device charge through the shared decision rule,
            // before any allocation (dsv4_admit).
            .implemented(PolicySurface::MemoryCost, "dsv4_admit::admit_session(")
            .refused(
                PolicySurface::RewriteQualification,
                "the route consumes no MEMRA_REWRITE_BUNDLE and none of its programs is a plan rewrite (memra#449)",
            )
            .refused(
                PolicySurface::PrimeFairness,
                "no worker Session exists for a dsv4 request; the prime is one synchronous chunked call on the serving thread (memra#535 P4)",
            )
            // memra#501: the served counters and the service-time window the route's estimate reads.
            .implemented(PolicySurface::ServiceMetrics, "run.finish(stats)")
    }
}

/// The interactive session cap of the central worker, as ONE pure derivation: `max_sessions`
/// (`MEMRA_MAX_SESSIONS`, default 64) when `batching_on`, else `legacy_max_active`. The worker's
/// admission gate, `lib.rs::lane_cap` and the registry all call this, so the number the gate
/// enforces, the number the HTTP layer sheds on and the number the registry publishes are the
/// same number by construction (memra#502 was two copies of this arithmetic disagreeing). No
/// floor: `MEMRA_MAX_SESSIONS=0` is 0 everywhere, as the gate has always read it.
pub fn interactive_cap(
    batching_on: bool,
    max_sessions: Option<usize>,
    legacy_max_active: usize,
) -> usize {
    if batching_on {
        max_sessions.unwrap_or(64)
    } else {
        legacy_max_active
    }
}

/// `interactive_cap` from the process environment with the legacy `MAX_ACTIVE`. The worker's gate
/// passes its own `max_active` (which the confidence-trace door lowers to 1) instead.
pub fn hybrid_interactive_cap() -> usize {
    let batching_on = std::env::var("MEMRA_SERVE_BATCH")
        .map(|v| v != "0")
        .unwrap_or(true);
    let max_sessions = std::env::var("MEMRA_MAX_SESSIONS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok());
    interactive_cap(batching_on, max_sessions, crate::worker::MAX_ACTIVE)
}

/// The enumerable route set of one process.
#[derive(Debug, Default, Clone)]
pub struct RouteRegistry {
    routes: Vec<RouteContract>,
}

impl RouteRegistry {
    pub fn register(&mut self, contract: RouteContract) {
        self.routes.push(contract);
    }

    pub fn routes(&self) -> &[RouteContract] {
        &self.routes
    }

    pub fn contract_for(&self, model: &str) -> Option<&RouteContract> {
        self.routes.iter().find(|r| r.model == model)
    }

    /// The route's declared concurrency, the number the admission cap mirror should read.
    pub fn capacity_for(&self, model: &str) -> Option<RouteCapacity> {
        self.contract_for(model).map(|r| r.capacity)
    }

    /// The boot gate. Every registered route must declare every surface (an omission is the
    /// first error). An ARMED policy is refused only when no registered route can honor it: a
    /// mixed process (hybrid models beside a dsv4 checkpoint) with `MEMRA_REWRITE_BUNDLE` keeps
    /// booting, the bundle governs the hybrid route and the dsv4 line names its refusal; a
    /// dsv4-only process with the bundle set has armed a policy nobody reads and refuses.
    pub fn check(&self, env: &dyn Fn(&str) -> Option<String>) -> Result<(), RouteContractError> {
        for route in &self.routes {
            route.check_declared()?;
        }
        for surface in PolicySurface::ALL {
            let Some(var) = surface.arming_env() else {
                continue;
            };
            if !env(var).is_some_and(|v| !v.is_empty()) {
                continue;
            }
            let honored = self
                .routes
                .iter()
                .any(|r| matches!(r.decl(surface), Some(PolicyDecl::Implemented { .. })));
            if honored || self.routes.is_empty() {
                continue;
            }
            let first = &self.routes[0];
            if let Some(PolicyDecl::Refused { reason }) = first.decl(surface) {
                return Err(RouteContractError::ArmedButRefused {
                    model: first.model.clone(),
                    kind: first.kind,
                    surface,
                    env: var,
                    reason,
                });
            }
        }
        Ok(())
    }

    /// `check` against the process environment.
    pub fn check_process_env(&self) -> Result<(), RouteContractError> {
        self.check(&|var| std::env::var(var).ok())
    }
}

/// Comment-stripped source text: the same rule the odometer wiring gate uses (a doc mention of a
/// call is not an invocation).
pub fn code_only(src: &str) -> String {
    src.lines()
        .map(|l| l.trim_start())
        .filter(|l| !l.starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn no_env(_: &str) -> Option<String> {
        None
    }

    /// The red arm the issue demands: a route that satisfies nothing, registered, must fail the
    /// same gate the production routes pass.
    #[test]
    fn a_stub_route_that_declares_nothing_fails_the_gate() {
        let mut reg = RouteRegistry::default();
        reg.register(RouteContract::hybrid_worker("gate", 4));
        reg.register(RouteContract::new(
            "stub",
            "stub-route",
            "stub.rs",
            RouteCapacity::Serial,
        ));
        let err = reg.check(&no_env).unwrap_err();
        match &err {
            RouteContractError::Undeclared {
                model, surfaces, ..
            } => {
                assert_eq!(model, "stub");
                assert_eq!(surfaces.len(), PolicySurface::ALL.len());
            }
            other => panic!("expected Undeclared, got {other:?}"),
        }
        assert!(err.to_string().contains("memra#504"));
        assert!(err.to_string().contains("service-metrics"));
    }

    #[test]
    fn a_partially_declared_route_names_exactly_what_it_omitted() {
        let route = RouteContract::new("half", "stub-route", "stub.rs", RouteCapacity::Serial)
            .implemented(PolicySurface::Capacity, "x")
            .refused(PolicySurface::Progress, "not yet");
        match route.check_declared().unwrap_err() {
            RouteContractError::Undeclared { surfaces, .. } => {
                assert_eq!(surfaces.len(), PolicySurface::ALL.len() - 2);
                assert!(!surfaces.contains(&PolicySurface::Capacity));
                assert!(!surfaces.contains(&PolicySurface::Progress));
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn the_hybrid_worker_contract_implements_every_surface() {
        let route = RouteContract::hybrid_worker("gate", 64);
        route.check_declared().unwrap();
        for s in PolicySurface::ALL {
            assert!(
                matches!(route.decl(s), Some(PolicyDecl::Implemented { .. })),
                "{s:?}"
            );
        }
        assert_eq!(route.capacity.concurrency(), 64);
        assert!(route.describe().contains("refused=[]"));
    }

    #[test]
    fn the_dsv4_contract_declares_its_serving_lanes() {
        assert_eq!(
            RouteContract::dsv4_thread("ds").capacity,
            RouteCapacity::Serial
        );
        let lanes = RouteContract::dsv4_thread_sessions("ds", 2);
        assert_eq!(lanes.capacity, RouteCapacity::Sessions(2));
        assert_eq!(lanes.capacity.concurrency(), 2);
        lanes.check_declared().unwrap();
        assert_eq!(
            RouteContract::dsv4_thread_sessions("ds", 1).capacity,
            RouteCapacity::Serial
        );
    }

    #[test]
    fn the_dsv4_contract_refuses_its_open_gaps_by_issue() {
        let route = RouteContract::dsv4_thread("ds");
        route.check_declared().unwrap();
        let refused: Vec<PolicySurface> = route
            .declarations()
            .filter_map(|(s, d)| matches!(d, PolicyDecl::Refused { .. }).then_some(s))
            .collect();
        assert_eq!(
            refused,
            vec![
                PolicySurface::RewriteQualification,
                PolicySurface::PrimeFairness,
            ]
        );
        assert!(matches!(
            route.decl(PolicySurface::MemoryCost),
            Some(PolicyDecl::Implemented { .. })
        ));
        for (s, issue) in [
            (PolicySurface::RewriteQualification, "#449"),
            (PolicySurface::PrimeFairness, "#535"),
        ] {
            match route.decl(s) {
                Some(PolicyDecl::Refused { reason }) => {
                    assert!(reason.contains(issue), "{s:?}: {reason}")
                }
                other => panic!("{s:?}: {other:?}"),
            }
        }
        assert_eq!(route.capacity, RouteCapacity::Serial);
        assert!(route.describe().contains("route=dsv4-thread"));
    }

    /// The #449 minimum: a rewrite bundle armed in a process where no route can consume it
    /// refuses at boot with the refusing route named, instead of no-oping past the `continue`.
    /// A mixed process keeps booting: the bundle governs the hybrid route (revuto round 2).
    #[test]
    fn an_armed_rewrite_bundle_refuses_only_when_no_route_honors_it() {
        let env = |var: &str| (var == "MEMRA_REWRITE_BUNDLE").then(|| "/tmp/bundle".to_string());
        // dsv4-only: nobody reads the bundle -> refused, by route and issue
        let mut dsv4_only = RouteRegistry::default();
        dsv4_only.register(RouteContract::dsv4_thread("ds"));
        dsv4_only.check(&no_env).unwrap();
        let text = dsv4_only.check(&env).unwrap_err().to_string();
        assert!(text.contains("MEMRA_REWRITE_BUNDLE is set but no route in this process honors the rewrite-qualification policy"), "{text}");
        assert!(text.contains("\"ds\" (dsv4-thread)"), "{text}");
        assert!(text.contains("#449"), "{text}");
        // an empty value is unset
        let empty = |var: &str| (var == "MEMRA_REWRITE_BUNDLE").then(String::new);
        dsv4_only.check(&empty).unwrap();
        // mixed: the hybrid route honors it, the process boots, the dsv4 line still refuses
        let mut mixed = RouteRegistry::default();
        mixed.register(RouteContract::hybrid_worker("gate", 4));
        mixed.register(RouteContract::dsv4_thread("ds"));
        mixed.check(&env).unwrap();
        assert!(
            mixed
                .contract_for("ds")
                .unwrap()
                .describe()
                .contains("rewrite-qualification:")
        );
        // hybrid-only with the bundle armed is the ordinary case
        let mut hybrid_only = RouteRegistry::default();
        hybrid_only.register(RouteContract::hybrid_worker("gate", 4));
        hybrid_only.check(&env).unwrap();
    }

    #[test]
    fn the_interactive_cap_is_one_derivation_with_both_branches() {
        assert_eq!(interactive_cap(true, None, 4), 64);
        assert_eq!(interactive_cap(true, Some(8), 4), 8);
        assert_eq!(
            interactive_cap(true, Some(0), 4),
            0,
            "no floor: the gate reads 0 as 0"
        );
        assert_eq!(
            interactive_cap(false, Some(8), 4),
            4,
            "batching off is the legacy cap"
        );
        assert_eq!(
            interactive_cap(false, None, 1),
            1,
            "the confidence-trace door lowers max_active to 1"
        );
    }

    #[test]
    fn capacity_is_readable_per_model() {
        let mut reg = RouteRegistry::default();
        reg.register(RouteContract::hybrid_worker("gate", 8));
        reg.register(RouteContract::dsv4_thread("ds"));
        assert_eq!(reg.capacity_for("gate"), Some(RouteCapacity::Sessions(8)));
        assert_eq!(reg.capacity_for("ds"), Some(RouteCapacity::Serial));
        assert_eq!(reg.capacity_for("ds").unwrap().concurrency(), 1);
        assert_eq!(reg.capacity_for("nope"), None);
    }

    /// WIRING GATE, generalized from `progress::tests::the_prime_walks_actually_call_the_odometer`
    /// (which greps one engine file): every `Implemented` declaration of every production route
    /// names a call-site token, and that token must exist outside comments in the route's own
    /// source. A declaration that outlives its call site fails here; a new route is born with
    /// this test already reading its file.
    #[test]
    fn every_implemented_declaration_has_its_call_site_in_the_routes_source() {
        let sources: &[(&str, &str)] = &[
            ("worker.rs", include_str!("worker.rs")),
            ("dsv4_serve.rs", include_str!("dsv4_serve.rs")),
        ];
        // The declarations live in THIS file, never in a route's source, so a grep of the route's
        // file cannot be satisfied by the declaration text (revuto round 1 on the dsv4 contract).
        for route in [
            RouteContract::hybrid_worker("gate", 4),
            RouteContract::dsv4_thread("ds"),
        ] {
            assert!(
                !sources.iter().any(|(name, _)| *name == "route_contract.rs"),
                "the registry must never grep its own file"
            );
            let _ = route;
        }
        let mut reg = RouteRegistry::default();
        reg.register(RouteContract::hybrid_worker("gate", 4));
        reg.register(RouteContract::dsv4_thread("ds"));
        for route in reg.routes() {
            let (_, src) = sources
                .iter()
                .find(|(name, _)| *name == route.source_file)
                .unwrap_or_else(|| {
                    panic!(
                        "route {} names an unknown source {}",
                        route.kind, route.source_file
                    )
                });
            let code = code_only(src);
            for (surface, decl) in route.declarations() {
                if let PolicyDecl::Implemented { evidence } = decl {
                    assert!(
                        code.contains(evidence),
                        "route {} declares {} implemented with evidence {evidence:?}, but {} has no such call outside comments",
                        route.kind,
                        surface.name(),
                        route.source_file
                    );
                }
            }
        }
    }
}
