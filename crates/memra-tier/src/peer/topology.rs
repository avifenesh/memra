//! Read-only probe contract. Queries do not enable peer/pool grants or wake a GPU.
//! An owner-thread transfer gate later records actual grants and byte-path qualification.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Observation<T> {
    Known(T),
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceTopology {
    pub device: u32,
    pub numa_node: Observation<u32>,
    pub cpu_affinity: Vec<u32>,
    pub current_generation: Observation<u8>,
    pub max_generation: Observation<u8>,
    pub current_width: Observation<u8>,
    pub max_width: Observation<u8>,
    pub idle_p8: Observation<bool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkHealth {
    AtMaximum,
    IdleDeferred,
    Downgraded,
    Unknown,
}
impl DeviceTopology {
    pub fn link_health(&self) -> LinkHealth {
        use Observation::Known;
        match (
            self.current_generation,
            self.max_generation,
            self.current_width,
            self.max_width,
        ) {
            (Known(g), Known(gm), Known(w), Known(wm)) if g > 0 && gm >= g && w > 0 && wm >= w => {
                if w < wm {
                    LinkHealth::Downgraded
                } else if g == gm {
                    LinkHealth::AtMaximum
                } else if self.idle_p8 == Known(true) {
                    LinkHealth::IdleDeferred
                } else if self.idle_p8 == Known(false) {
                    LinkHealth::Downgraded
                } else {
                    LinkHealth::Unknown
                }
            }
            _ => LinkHealth::Unknown,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PeerEdge {
    pub source: u32,
    pub destination: u32,
    /// CUDA capability query is separate from successful context + pool access grants.
    pub can_access: Observation<bool>,
    pub context_granted: Observation<bool>,
    pub pool_granted: Observation<bool>,
    /// True only after an exact byte test AND evidence this route did not host-bounce.
    pub direct_copy_qualified: Observation<bool>,
    /// Root/switch sharing label derived from topology, never an assumed aggregate link rate.
    pub shared_fabric: Option<String>,
}
impl PeerEdge {
    pub fn available_for_tiering(&self) -> bool {
        self.source != self.destination
            && [
                self.can_access,
                self.context_granted,
                self.pool_granted,
                self.direct_copy_qualified,
            ]
            .into_iter()
            .all(|v| v == Observation::Known(true))
    }
}

#[derive(Debug, Clone)]
pub struct TopologySnapshot {
    pub devices: Vec<DeviceTopology>,
    /// Directed: 4 cards require 12 separate edges. Missing/unknown is unavailable.
    pub edges: Vec<PeerEdge>,
}
impl TopologySnapshot {
    pub fn peer_available(&self, source: u32, destination: u32) -> bool {
        let endpoints = [source, destination].into_iter().all(|id| {
            let matches: Vec<_> = self.devices.iter().filter(|d| d.device == id).collect();
            matches.len() == 1 && matches[0].link_health() == LinkHealth::AtMaximum
        });
        let edges: Vec<_> = self
            .edges
            .iter()
            .filter(|e| e.source == source && e.destination == destination)
            .collect();
        endpoints && edges.len() == 1 && edges[0].available_for_tiering()
    }
}

pub trait TopologyProbe {
    type Error;
    fn snapshot_read_only(&self) -> Result<TopologySnapshot, Self::Error>;
}

/// Scope supplied by the native owner on each use; never deserialized as a live grant.
/// A changed context, topology census or executable invalidates an old observation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RouteScope {
    pub source_context: u64,
    pub destination_context: u64,
    pub topology_digest: crate::contracts::Digest,
    pub binary_digest: crate::contracts::Digest,
}
#[derive(Debug)]
pub struct ScopedTopology {
    snapshot: TopologySnapshot,
    scope: RouteScope,
}
impl ScopedTopology {
    /// The adapter must populate actual context/pool grants and direct byte evidence.
    /// CPU metadata cannot certify that driver truth; this only enforces expiry.
    pub fn new(snapshot: TopologySnapshot, scope: RouteScope) -> Self {
        Self { snapshot, scope }
    }
    pub fn peer_available(&self, source: u32, destination: u32, current: &RouteScope) -> bool {
        current == &self.scope
            && current.source_context != 0
            && current.destination_context != 0
            && self.snapshot.peer_available(source, destination)
    }
}
