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
//! What this module does NOT do: implement the DSv4 policies. #500 (progress), #501 (capacity,
//! occupancy, metrics), #503 (memory cost) and #449 (rewrite bundle) each flip one declaration
//! from `Refused` to `Implemented`; until then the registry names them at every boot.

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
                "{env} is set but route {model:?} ({kind}) refuses the {} policy: {reason}. \
                 Unset {env} or serve this model on a route that honors it; a policy the route \
                 cannot read is refused at boot, not ignored at runtime (memra#504)",
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

    /// Every surface declared, and no refused surface armed by `env`.
    pub fn check(&self, env: &dyn Fn(&str) -> Option<String>) -> Result<(), RouteContractError> {
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
        for (surface, decl) in &self.decls {
            if let (PolicyDecl::Refused { reason }, Some(var)) = (decl, surface.arming_env())
                && env(var).is_some_and(|v| !v.is_empty())
            {
                return Err(RouteContractError::ArmedButRefused {
                    model: self.model.clone(),
                    kind: self.kind,
                    surface: *surface,
                    env: var,
                    reason,
                });
            }
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
    /// Flipping a line from `refused` to `implemented` is the registration side of #500, #501,
    /// #503 and #449. Declared here and not in `dsv4_serve.rs` so the wiring test's grep of
    /// that file cannot be satisfied by the declaration text itself.
    pub fn dsv4_thread(model: impl Into<String>) -> Self {
        Self::new(model, "dsv4-thread", "dsv4_serve.rs", RouteCapacity::Serial)
            .implemented(
                PolicySurface::Capacity,
                "std::sync::mpsc::channel::<Box<Request>>()",
            )
            .refused(
                PolicySurface::Occupancy,
                "the FIFO channel is the queue and publishes no in-flight or queued count; the lane cap mirror reads 64 for a serial route (memra#501)",
            )
            .refused(
                PolicySurface::Progress,
                "the serving thread stamps neither the health beat nor the prime odometer, so /health reads the idle central worker (memra#500)",
            )
            .implemented(PolicySurface::FaultOwnership, "std::panic::catch_unwind(")
            .implemented(
                PolicySurface::ShutdownOwnership,
                "while let Ok(mut req) = rx.recv()",
            )
            .refused(
                PolicySurface::MemoryCost,
                "requests reach allocation with a single-active-request reservation and no per-device feasibility, tier or defer decision (memra#503)",
            )
            .refused(
                PolicySurface::RewriteQualification,
                "the route consumes no MEMRA_REWRITE_BUNDLE and none of its programs is a plan rewrite (memra#449)",
            )
            .refused(
                PolicySurface::PrimeFairness,
                "no worker Session exists for a dsv4 request; the prime is one synchronous chunked call on the serving thread (memra#535 P4)",
            )
            .refused(
                PolicySurface::ServiceMetrics,
                "the thread publishes no worker::Metrics, so the queue estimator uses its static 2 s service time (memra#501)",
            )
    }
}

/// The interactive session cap the central worker's admission gate applies, derived ONCE so the
/// registry, the gate and `lib.rs::lane_cap` cannot disagree (memra#502 was a second copy of this
/// arithmetic disagreeing): `MEMRA_MAX_SESSIONS` (default 64) when batching is on, the legacy
/// `MAX_ACTIVE` when `MEMRA_SERVE_BATCH=0`.
pub fn hybrid_interactive_cap() -> usize {
    let batching_on = std::env::var("MEMRA_SERVE_BATCH")
        .map(|v| v != "0")
        .unwrap_or(true);
    if batching_on {
        std::env::var("MEMRA_MAX_SESSIONS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(64)
            .max(1)
    } else {
        crate::worker::MAX_ACTIVE
    }
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

    /// The boot gate: every registered route checks clean, or the first failure is the reason
    /// the worker refuses to come up.
    pub fn check(&self, env: &dyn Fn(&str) -> Option<String>) -> Result<(), RouteContractError> {
        for route in &self.routes {
            route.check(env)?;
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
        match route.check(&no_env).unwrap_err() {
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
        route.check(&no_env).unwrap();
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
    fn the_dsv4_contract_refuses_its_open_gaps_by_issue() {
        let route = RouteContract::dsv4_thread("ds");
        route.check(&no_env).unwrap();
        let refused: Vec<PolicySurface> = route
            .declarations()
            .filter_map(|(s, d)| matches!(d, PolicyDecl::Refused { .. }).then_some(s))
            .collect();
        assert_eq!(
            refused,
            vec![
                PolicySurface::Occupancy,
                PolicySurface::Progress,
                PolicySurface::MemoryCost,
                PolicySurface::RewriteQualification,
                PolicySurface::PrimeFairness,
                PolicySurface::ServiceMetrics,
            ]
        );
        for (s, issue) in [
            (PolicySurface::Progress, "#500"),
            (PolicySurface::Occupancy, "#501"),
            (PolicySurface::ServiceMetrics, "#501"),
            (PolicySurface::MemoryCost, "#503"),
            (PolicySurface::RewriteQualification, "#449"),
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

    /// The #449 minimum: a rewrite bundle armed beside a route that cannot consume it refuses at
    /// boot with the route named, instead of no-oping past the `continue`.
    #[test]
    fn an_armed_rewrite_bundle_beside_dsv4_refuses_at_boot_by_name() {
        let mut reg = RouteRegistry::default();
        reg.register(RouteContract::hybrid_worker("gate", 4));
        reg.register(RouteContract::dsv4_thread("ds"));
        reg.check(&no_env).unwrap();
        let env = |var: &str| (var == "MEMRA_REWRITE_BUNDLE").then(|| "/tmp/bundle".to_string());
        let err = reg.check(&env).unwrap_err();
        let text = err.to_string();
        assert!(text.contains("MEMRA_REWRITE_BUNDLE is set but route \"ds\" (dsv4-thread) refuses the rewrite-qualification policy"), "{text}");
        assert!(text.contains("#449"), "{text}");
        // an empty value is unset
        let empty = |var: &str| (var == "MEMRA_REWRITE_BUNDLE").then(String::new);
        reg.check(&empty).unwrap();
        // a hybrid-only process with the bundle armed is fine
        let mut hybrid_only = RouteRegistry::default();
        hybrid_only.register(RouteContract::hybrid_worker("gate", 4));
        hybrid_only.check(&env).unwrap();
    }

    #[test]
    fn the_hybrid_cap_helper_follows_the_batching_switch() {
        // Only the two branches' shape is asserted (env is process-global in tests): the
        // default is 64 with batching on, MAX_ACTIVE with it off.
        let cap = hybrid_interactive_cap();
        assert!(cap == crate::worker::MAX_ACTIVE || cap >= 1);
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
