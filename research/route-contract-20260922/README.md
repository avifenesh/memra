# Route policy contract (memra#504)

Verdict: the serve route set is enumerable (`route_contract::RouteRegistry`), every route declares
each of nine policy surfaces as implemented (with a call-site token) or refused by name (with the
owning issue), the registry is checked before the ready handoff so an omission or an armed-but-
refused policy is `FATAL: worker init failed`, and a stub route that declares nothing fails the same
gate the production routes pass.

## The class (from the issue)

`worker.rs` and `dsv4_serve.rs` are two serve routes sharing the `Cmd::Generate -> Event`
contract. Every side-channel policy had one writer, so on DSv4 `worker::Metrics`, the health beat
and prime odometer, `admit_memory`/`admit_predict`, the lane-cap mirror, `MEMRA_REWRITE_BUNDLE`
and prime fairness were silent no-ops, found one at a time (#449, #500, #501, #502, #503). The
odometer wiring gate was green because it greps one engine file.

## The contract (`crates/memra-server/src/route_contract.rs`)

- `PolicySurface::ALL`: capacity, occupancy, progress, fault-ownership, shutdown-ownership,
  memory-cost, rewrite-qualification, prime-fairness, service-metrics. Adding a variant forces
  every contract to declare it.
- `PolicyDecl::Implemented { evidence }` names a call-site token that must appear outside comments
  in the route's `source_file`; `Refused { reason }` names the issue.
- `RouteContract::hybrid_worker(model, sessions)`: every surface implemented (evidence:
  `health.beat_busy()`, `m.step_p50_ms = `, `admit_memory::decide(`, `install_rewrite_bundle(`,
  `PrimeService::default()`, `request_fault_guard(`, `host_capture_drain_at_shutdown(`,
  `m.admission_inflight = `, `"MEMRA_MAX_SESSIONS"`).
- `dsv4_serve::contract(model)`: capacity `Serial`, fault and shutdown ownership implemented
  (`catch_unwind`, the `rx.recv()` loop), and six refusals: occupancy (#501), progress (#500),
  memory-cost (#503), rewrite-qualification (#449), prime-fairness (#535 P4), service-metrics
  (#501). Flipping a line to `implemented` is the registration side of each of those issues.
- `RouteRegistry::check(env)`: undeclared surface, or a refused surface whose arming knob is set
  (`MEMRA_REWRITE_BUNDLE`), fails; `worker::run` builds the registry after both model maps are
  complete and sends the error through `ready_tx` before `mark_ready`, which is the existing
  `FATAL: worker init failed` exit. One `[route-contract]` line per route prints on a clean boot.
- `RouteRegistry::capacity_for(model)` exposes the per-route concurrency for #501/#502 to consume
  in `lane_cap`; not consumed yet in this PR.

## Receipts

- `route_contract::tests` (7): the stub red arm, partial declaration naming, hybrid full
  implementation, DSv4 refusals by issue, armed rewrite bundle beside DSv4 refuses by name (and a
  hybrid-only process with the bundle armed passes), capacity per model, and the registry-driven
  wiring gate over `worker.rs` and `dsv4_serve.rs`.
- `raw/serve-9b-boot.log`: the 9B on the local 5090 prints
  `[route-contract] model="gate" route=hybrid-worker capacity=64 sessions implemented=[...] refused=[]`
  and answers a completion.
- `raw/local-ci/local-ci.log`: the full battery on the lane tree.

No DSv4 checkpoint is on this rig, so the DSv4 boot line and the `MEMRA_REWRITE_BUNDLE` refusal are
unit-tested through the same `RouteRegistry::check` the worker calls, not booted.

## What stays open

#500, #501, #503 and #449 implement the refused surfaces; each is one `refused` line becoming
`implemented` with its call site. `lane_cap` (#501/#502) should read `capacity_for(model)`.
