# D CPU capacity construction seam (day 4)

`memra_tier::peer::test_support::FakePeerCapacity<G>` accepts the caller's
`Rc<RefCell<G: BudgetGovernor>>`, device count, and current state epoch:

```rust,ignore
let mut peer = FakePeerCapacity::new(shared_governor.clone(), 2, 7);
peer.set_route(0, 1, true, true, LinkHealth::AtMaximum)?;
let lease = peer.reserve(plan)?; // frozen PeerCapacity trait, no duplicate schema
```

All directions start **unavailable**. Inject context, pool and health separately via
`set_route`; missing, unknown or downgraded directions refuse before governor charge.
Reverse direction is independent. `state` advances to test stale plans; explicit release
of old-state physically retired leases remains legal. `owners` exposes CPU DeviceOwner
retain/resolve hooks for combined B→D lifetime tests. CPU backing is zeroed and sized to
`plan.bytes`, not the old four-byte fixture. Caller supplies budget/fixtures; D does not
export or duplicate a governor. `release` refuses retained/graph/inflight ownership and
foreign/double releases. D's existing peer fake now delegates capacity to this very seam,
so the frozen v1.1 directed/capacity schedules exercise the exported implementation.

This module is explicitly public without a feature (as requested) to avoid lead-owned
Cargo amendments. It is opt-in test support, not a runtime backend or live grant. There
are no environment reads, CUDA calls, hardware claims or implicit route promotions.
B's cross-crate integration tests can import it without changing dependency features.
