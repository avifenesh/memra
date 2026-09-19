# B → D day-4 seam request

At integration `020d2047`, `crates/memra-tier/tests/peer/fake.rs::Peer` and its
constructor/controls remain private to D's test binary. B does not edit or import
that implementation. Please export a test-only construction seam (not a runtime
backend): injected shared governor, PeerCapacity reserve/release, directed
context/pool/link controls, retained DeviceOwner access, and explicit
producer/consumer/graph completion controls. Arbitrary byte lengths and epochs
must be supported, not only the four-byte fixture.

B needs lookup → reserve peer source → direct local materialization → fenced ready
→ Busy release → fence retirement → retry, including denial/downgrade between
lookup and reserve and between reserve and submission. Both directions must stay
independent. No host bounce and no remote attention operand publication.

Until that export lands, B tests its adapter against the frozen PeerCapacity trait
with a B-owned CPU fake. Those tests are NOT a B→D implementation interoperability
receipt or PCIe/GPU evidence. Shared contracts and D files remain unchanged.
