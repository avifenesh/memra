//! Explicit DSV4 multi-card topology admission.
//!
//! TP/EP is the served two-card default since memra #710 (2026-09-25), and PP-2
//! (`MEMRA_DSV4_TOPOLOGY=pp`) is the rollback program.  The two are separate programs:
//! under TP/EP every rank must own every trunk layer's attention/router state and the
//! expert adapter must return a rank-order down partial.  Keeping this admission
//! contract separate prevents a TP/EP load from silently running the PP walk, or the
//! reverse.

type Res<T> = Result<T, String>;

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

/// Where a two-card load places the model. `Dsv4Gpu::load` takes it from every caller and has
/// no default (S06:F0, owner 2026-11-03): a gate that does not name its placement would measure
/// whatever the load picked, and before this the load picked PP-2 while serving loads TP/EP.
/// memra-server passes [`Self::SERVED`] unless `MEMRA_DSV4_TOPOLOGY=pp`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Dsv4Placement {
    /// Two pipeline stages over a layer split, optionally with whole-expert EP.
    Pp,
    /// Every trunk layer on both ranks with the expert-ID EP pair; attention is split by head
    /// (exact attention TP2) when `attention_tp`, replicated otherwise.
    TpEp { attention_tp: bool },
}

impl Dsv4Placement {
    /// What memra-server loads when `MEMRA_DSV4_TOPOLOGY` is unset (memra #710, 2026-09-25):
    /// TP/EP with exact attention TP2. Serving-qualification gates run this placement first.
    pub const SERVED: Self = Self::TpEp { attention_tp: true };

    pub const fn is_tp_ep(self) -> bool {
        matches!(self, Self::TpEp { .. })
    }

    /// The name logs, receipts and gate arguments use: `tp_ep` (the served placement),
    /// `tp_ep_replicated` (TP/EP with replicated attention, a gate-only placement) or `pp`
    /// (PP-2, the `MEMRA_DSV4_TOPOLOGY=pp` rollback).
    pub const fn name(self) -> &'static str {
        match self {
            Self::TpEp { attention_tp: true } => "tp_ep",
            Self::TpEp {
                attention_tp: false,
            } => "tp_ep_replicated",
            Self::Pp => "pp",
        }
    }

    /// The placement a [`Self::name`] names; any other text refuses.
    pub fn from_name(raw: &str) -> Res<Self> {
        match raw {
            "tp_ep" => Ok(Self::SERVED),
            "tp_ep_replicated" => Ok(Self::TpEp {
                attention_tp: false,
            }),
            "pp" => Ok(Self::Pp),
            other => Err(format!(
                "placement {other:?} unknown (tp_ep | tp_ep_replicated | pp)"
            )),
        }
    }

    /// The `--placement=<name>` argument of a gate binary that has cells on more than one
    /// placement, taken out of `args` so the gate's positional parsing never sees it. It is
    /// required: a cell names the placement it measures, so its command line does too. `runs`
    /// lists the placements the gate has cells for. The selectors this replaced refuse rather
    /// than parse into nothing: the `--tpep` switch, `MEMRA_DSV4_ATTENTION_TP_GATE` and
    /// `DSV4_ROWS_GATE_TOPOLOGY`.
    pub fn take_gate_arg(args: &mut Vec<String>, runs: &[Self]) -> Res<Self> {
        for retired in ["MEMRA_DSV4_ATTENTION_TP_GATE", "DSV4_ROWS_GATE_TOPOLOGY"] {
            if std::env::var_os(retired).is_some() {
                return Err(format!(
                    "{retired} is retired: unset it and pass --placement=<{}>",
                    Self::names(runs)
                ));
            }
        }
        Self::parse_gate_arg(args, runs)
    }

    fn names(runs: &[Self]) -> String {
        runs.iter()
            .map(|placement| placement.name())
            .collect::<Vec<_>>()
            .join("|")
    }

    fn parse_gate_arg(args: &mut Vec<String>, runs: &[Self]) -> Res<Self> {
        if args.iter().any(|arg| arg == "--tpep") {
            return Err(format!(
                "--tpep is retired: pass --placement=<{}>",
                Self::names(runs)
            ));
        }
        let mut found = None;
        let mut rest = Vec::with_capacity(args.len());
        for arg in args.drain(..) {
            match arg.strip_prefix("--placement=") {
                Some(_) if found.is_some() => {
                    return Err("--placement given twice".into());
                }
                Some(raw) => found = Some(Self::from_name(raw)?),
                None => rest.push(arg),
            }
        }
        *args = rest;
        let placement = found.ok_or_else(|| {
            format!(
                "--placement=<{}> is required: name the placement this cell measures \
                 (tp_ep is the served placement)",
                Self::names(runs)
            )
        })?;
        if !runs.contains(&placement) {
            return Err(format!(
                "this gate has no {} cell (--placement=<{}>)",
                placement.name(),
                Self::names(runs)
            ));
        }
        Ok(placement)
    }
}

#[cfg(test)]
mod tests {
    use super::{Dsv4Placement, Dsv4Topology, Dsv4TopologyPlan};

    /// The served placement is the one memra-server resolves for an unset
    /// `MEMRA_DSV4_TOPOLOGY`, and each placement has its own name.
    #[test]
    fn the_served_placement_is_tp_ep_with_attention_tp() {
        assert_eq!(
            Dsv4Placement::SERVED,
            Dsv4Placement::TpEp { attention_tp: true }
        );
        assert_eq!(Dsv4Placement::SERVED.name(), "tp_ep");
        assert_eq!(
            Dsv4Placement::TpEp {
                attention_tp: false
            }
            .name(),
            "tp_ep_replicated"
        );
        assert_eq!(Dsv4Placement::Pp.name(), "pp");
        for placement in [
            Dsv4Placement::SERVED,
            Dsv4Placement::TpEp {
                attention_tp: false,
            },
            Dsv4Placement::Pp,
        ] {
            assert_eq!(Dsv4Placement::from_name(placement.name()), Ok(placement));
        }
        for raw in ["", "tpep", "tp_ep_attn", "PP", "pp2"] {
            assert!(Dsv4Placement::from_name(raw).is_err(), "{raw}");
        }
    }

    fn args(raw: &[&str]) -> Vec<String> {
        raw.iter().map(|arg| arg.to_string()).collect()
    }

    /// A gate's placement argument is required, taken out of the positional arguments, and
    /// refused when the gate has no cell for it or when a retired selector rides along.
    #[test]
    fn a_gate_names_its_placement_on_the_command_line() {
        let both = [Dsv4Placement::SERVED, Dsv4Placement::Pp];
        let mut argv = args(&["gate", "model", "--placement=pp", "tape", "24"]);
        assert_eq!(
            Dsv4Placement::parse_gate_arg(&mut argv, &both),
            Ok(Dsv4Placement::Pp)
        );
        assert_eq!(argv, args(&["gate", "model", "tape", "24"]));
        let mut argv = args(&["gate", "model", "tape", "--placement=tp_ep"]);
        assert_eq!(
            Dsv4Placement::parse_gate_arg(&mut argv, &both),
            Ok(Dsv4Placement::SERVED)
        );
        for refused in [
            &["gate", "model", "tape"][..],
            &["gate", "--placement=tp_ep_replicated", "model"],
            &["gate", "--placement=tp_ep", "--placement=pp"],
            &["gate", "--placement=tpep"],
            &["gate", "--placement=pp", "--tpep"],
        ] {
            let mut argv = args(refused);
            assert!(
                Dsv4Placement::parse_gate_arg(&mut argv, &both).is_err(),
                "{refused:?}"
            );
        }
    }

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
