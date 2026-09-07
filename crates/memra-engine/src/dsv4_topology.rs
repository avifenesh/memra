//! Explicit DSV4 multi-card topology admission.
//!
//! PP+EP is the shipped correctness reference.  The TP/EP topology is a
//! separate program: every rank must own every trunk layer's attention/router
//! state and the expert adapter must return a rank-order down partial.  Keeping
//! this admission contract separate prevents a requested TP/EP experiment from
//! silently running the old PP walk.

use std::sync::atomic::{AtomicBool, Ordering};

type Res<T> = Result<T, String>;

static TP_EP_FOR_GATE: AtomicBool = AtomicBool::new(false);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Dsv4Topology {
    /// Existing layer-owner placement, optionally with whole-expert EP.
    PpEp,
    /// All trunk layers are resident on both ranks. Attention/router state is
    /// replicated; ModelOpt experts are split by the separate adapter contract.
    TpEpAllLayers,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Dsv4TopologyPlan {
    pub topology: Dsv4Topology,
    pub world: usize,
    pub layers: usize,
    pub experts: usize,
    pub hidden: usize,
    pub inter: usize,
}

impl Dsv4TopologyPlan {
    pub fn pp_ep(
        world: usize,
        layers: usize,
        experts: usize,
        hidden: usize,
        inter: usize,
    ) -> Res<Self> {
        if world != 2 || layers == 0 || experts == 0 || hidden == 0 || inter == 0 {
            return Err(format!(
                "invalid DSV4 PP/EP plan world={world} layers={layers} experts={experts} hidden={hidden} inter={inter}"
            ));
        }
        Ok(Self {
            topology: Dsv4Topology::PpEp,
            world,
            layers,
            experts,
            hidden,
            inter,
        })
    }

    pub fn tp_ep_all_layers(
        world: usize,
        layers: usize,
        experts: usize,
        hidden: usize,
        inter: usize,
    ) -> Res<Self> {
        if world != 2 {
            return Err(format!(
                "DSV4 TP/EP all-layer topology requires world=2, got {world}"
            ));
        }
        if layers == 0 || experts == 0 || !experts.is_multiple_of(world) {
            return Err(format!(
                "DSV4 TP/EP all-layer topology requires nonzero even expert count, got layers={layers} experts={experts}"
            ));
        }
        if hidden == 0
            || inter == 0
            || !hidden.is_multiple_of(128)
            || !inter.is_multiple_of(128)
            || !(hidden / world).is_multiple_of(128)
            || !(inter / world).is_multiple_of(128)
        {
            return Err(format!(
                "DSV4 TP/EP all-layer dimensions are not 128-aligned: hidden={hidden} inter={inter}"
            ));
        }
        Ok(Self {
            topology: Dsv4Topology::TpEpAllLayers,
            world,
            layers,
            experts,
            hidden,
            inter,
        })
    }

    pub const fn is_tp_ep(self) -> bool {
        matches!(self.topology, Dsv4Topology::TpEpAllLayers)
    }
}

/// Gate-only process switch.  No environment variable or serving request can
/// select the topology after load.
pub fn set_tp_ep_for_gate(enabled: bool) -> bool {
    TP_EP_FOR_GATE.swap(enabled, Ordering::SeqCst)
}

pub fn tp_ep_for_gate() -> bool {
    TP_EP_FOR_GATE.load(Ordering::Acquire)
}

#[cfg(test)]
mod tests {
    use super::{Dsv4Topology, Dsv4TopologyPlan};

    #[test]
    fn pp_and_tp_ep_are_distinct_programs() {
        let pp = Dsv4TopologyPlan::pp_ep(2, 43, 256, 4096, 2048).unwrap();
        let tp = Dsv4TopologyPlan::tp_ep_all_layers(2, 43, 256, 4096, 2048).unwrap();
        assert_eq!(pp.topology, Dsv4Topology::PpEp);
        assert_eq!(tp.topology, Dsv4Topology::TpEpAllLayers);
        assert!(!pp.is_tp_ep());
        assert!(tp.is_tp_ep());
    }

    #[test]
    fn tp_ep_admission_refuses_unsupported_shapes() {
        assert!(Dsv4TopologyPlan::tp_ep_all_layers(1, 43, 256, 4096, 2048).is_err());
        assert!(Dsv4TopologyPlan::tp_ep_all_layers(2, 43, 255, 4096, 2048).is_err());
        assert!(Dsv4TopologyPlan::tp_ep_all_layers(2, 43, 256, 4096, 2050).is_err());
    }
}
