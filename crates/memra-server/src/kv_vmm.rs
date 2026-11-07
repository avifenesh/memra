//! On-demand KV planes (WP-B day 37, `research/spill-b-20260919/DAY37.md`), the serving default
//! on the card class whose deciding cell read PROMOTE-ELIGIBLE: the full-power RTX PRO 6000
//! Blackwell Workstation Edition (`docs/decisions/KV-VMM-SERVING-DEFAULT.md`). There, the
//! full-attention K/V planes of a covered session (the MTP spec route and the plain route,
//! single device, no SWA ring, no latent planes) are on-demand VMM planes: the whole capacity is
//! a reserved virtual range at a fixed address, and physical granules are mapped as the
//! session's rows grow. Every other card keeps pooled planes. Nothing numeric changes: the same
//! kernels touch the same bytes at addresses a session keeps for its life, and a session's
//! allocator is fixed at construction.
//!
//! This module holds the read-once configuration and the pure arithmetic the worker uses at its
//! ensure points (construction, the tick top, adoption); the worker owns the wiring.

use memra_engine::cache::{GrowEvent, VmmFaults, VmmGrowPlacement};

/// `SPEC_SHRINK_SLACK`: the engine's own bound on a speculative round's overshoot past the
/// request's need, and the rows an on-demand cache must be backed past its position before any
/// engine entry (`memra_engine::cache::ON_DEMAND_MIN_AHEAD_ROWS`).
pub(crate) const SLACK: usize = 64;

static CONFIG: std::sync::OnceLock<KvVmmConfig> = std::sync::OnceLock::new();

/// Install the worker's read-once configuration (worker start). A respawned worker reads the
/// same device and environment and installs the same program; a different one is refused, so
/// the allocator cannot move mid-process. The text of a VMM refusal may differ between two
/// starts; only whether the card fell back is part of the program.
pub(crate) fn install(cfg: KvVmmConfig) -> Result<(), String> {
    match CONFIG.get() {
        Some(existing) if existing.same_program(&cfg) => Ok(()),
        Some(_) => Err("kv-vmm configuration changed between worker starts".to_string()),
        None => CONFIG
            .set(cfg)
            .map_err(|_| "kv-vmm configuration raced at worker start".to_string()),
    }
}

/// Whether the on-demand planes are on in this process (false until `install`, and in every
/// unit test that builds sessions without a worker).
pub(crate) fn armed() -> bool {
    CONFIG.get().is_some_and(|c| c.armed)
}

/// Read-once configuration, built at worker start from the worker's device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct KvVmmConfig {
    /// The device is the full-power RTX PRO 6000 Blackwell Workstation Edition or its Max-Q
    /// edition (`memra_engine::parallel::HardwareTarget::is_pro6000_workstation`,
    /// `is_pro6000_maxq`), each from its own deciding cell.
    pub armed: bool,
    /// Where grows run (DAY37 1.5's rule per card class).
    pub placement: VmmGrowPlacement,
    /// Why the placement is what it is (the boot line states it).
    pub placement_source: &'static str,
    /// `MEMRA_KV_VMM_FAULT` (the fault door of the DAY37 A5 cells, `tools/health-fault-gate.sh`
    /// arm k), parsed on every card and armed only with the planes.
    pub faults: Option<VmmFaults>,
    /// The receipted card refused VMM at boot (its granularity query failed): the planes stay
    /// pooled and the boot line quotes the refusal.
    pub unsupported: Option<String>,
}

/// The per-class placement (`memra_engine::tier_transfer::kv_vmm_placement_for_device`).
pub(crate) fn placement_for_device(name: &str) -> (VmmGrowPlacement, &'static str) {
    memra_engine::tier_transfer::kv_vmm_placement_for_device(name)
}

impl KvVmmConfig {
    /// Arms the planes on the receipted card class and parses `MEMRA_KV_VMM_FAULT` on every
    /// card; an unparseable fault spec refuses the boot.
    pub(crate) fn for_device(device_name: &str) -> Result<Self, String> {
        Self::from_values(
            std::env::var("MEMRA_KV_VMM_FAULT").ok().as_deref(),
            device_name,
        )
    }

    pub(crate) fn from_values(fault: Option<&str>, device_name: &str) -> Result<Self, String> {
        use memra_engine::parallel::HardwareTarget;
        let armed = HardwareTarget::is_pro6000_workstation(device_name)
            || HardwareTarget::is_pro6000_maxq(device_name);
        let (placement, placement_source) = placement_for_device(device_name);
        let mut faults = None;
        if let Some(f) = fault.map(str::trim).filter(|f| !f.is_empty()) {
            let parsed = VmmFaults::parse(f).ok_or_else(|| {
                format!(
                    "MEMRA_KV_VMM_FAULT={f:?}: expected a comma list of build:<n>, \
                     ensure:<n>, mapper:all"
                )
            })?;
            // Under the inline placement no mapper thread starts, so `mapper:all` would inject
            // nothing while the boot line claimed a fault.
            if armed && placement == VmmGrowPlacement::Inline && parsed.mapper_all {
                return Err(format!(
                    "MEMRA_KV_VMM_FAULT={f:?}: mapper:all injects nothing under the inline grow \
                     placement this card runs (no mapper thread)"
                ));
            }
            faults = Some(parsed);
        }
        Ok(Self {
            armed,
            placement,
            placement_source,
            faults,
            unsupported: None,
        })
    }

    /// The VMM granularity query's outcome on the receipted card (`None` when the planes are not
    /// this card's default and nothing was queried). A refusal falls back to pooled planes, the
    /// program every other card runs, instead of refusing the boot: the planes are a default,
    /// not an operator's request. Returns the configuration to install and the granularity.
    pub(crate) fn with_granularity(
        mut self,
        granularity: Option<Result<usize, String>>,
    ) -> (Self, Option<usize>) {
        match granularity {
            Some(Ok(g)) if self.armed => (self, Some(g)),
            Some(Err(err)) if self.armed => {
                self.armed = false;
                self.unsupported = Some(err);
                (self, None)
            }
            _ => (self, None),
        }
    }

    /// The process-wide program: everything but the text of a VMM refusal.
    fn same_program(&self, other: &Self) -> bool {
        self.armed == other.armed
            && self.placement == other.placement
            && self.placement_source == other.placement_source
            && self.faults == other.faults
            && self.unsupported.is_some() == other.unsupported.is_some()
    }

    /// Boot receipt, one line.
    pub(crate) fn boot_line(&self, granularity: Option<usize>) -> String {
        if !self.armed {
            let inert = if self.faults.is_some() {
                "; MEMRA_KV_VMM_FAULT injects nothing on pooled planes"
            } else {
                ""
            };
            return match &self.unsupported {
                Some(err) => {
                    format!("[kv-vmm] door=OFF (VMM unsupported: {err}; pooled K/V planes{inert})")
                }
                None => format!(
                    "[kv-vmm] door=OFF (pooled K/V planes: the on-demand planes are the default \
                     on the RTX PRO 6000 Blackwell Workstation and Max-Q Workstation Editions \
                     only{inert})"
                ),
            };
        }
        format!(
            "[kv-vmm] door=ON routes=spec,plain granularity={} grow={} ({}) reap=owner-tick \
             slack_rows={SLACK} faults={}",
            granularity.map_or("unknown".to_string(), |g| g.to_string()),
            self.placement.name(),
            self.placement_source,
            self.faults
                .as_ref()
                .map_or("none".to_string(), |f| format!("{f:?}")),
        )
    }
}

/// Rows one tick can write for a session past its position: a spec burst (the burst target,
/// the draft depth, the surplus row and the verify row) or a plain step (the async chain width).
pub(crate) fn tick_rows(spec: bool, burst_t: usize, k: usize, async_chain_k: usize) -> usize {
    if spec {
        burst_t.saturating_add(k).saturating_add(2)
    } else {
        async_chain_k.max(1)
    }
}

/// `bound(s) = min(capacity, max(pos, floor) + tick_rows + SLACK)` (DAY37 1.2).
pub(crate) fn bound_rows(capacity: usize, pos: usize, floor: usize, tick: usize) -> usize {
    pos.max(floor)
        .saturating_add(tick)
        .saturating_add(SLACK)
        .min(capacity)
}

/// The construction scope's initial rows: the prompt plus one tick plus the slack, capped at the
/// cache capacity.
pub(crate) fn initial_rows(capacity: usize, prompt: usize, tick: usize) -> usize {
    bound_rows(capacity, 0, prompt, tick)
}

/// One grow receipt line.
pub(crate) fn grow_line(id: &str, label: &str, rows: usize, ev: &GrowEvent) -> String {
    format!(
        "[kv-vmm] grow id={id} plane={label} rows={rows} extent_bytes={} mapped={} reserved={} \
         owner_us={} waited={}",
        ev.bytes,
        ev.mapped,
        ev.reserved,
        ev.owner_us,
        u8::from(ev.waited)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const PRO_WS: &str = "NVIDIA RTX PRO 6000 Blackwell Workstation Edition";

    /// The planes arm on the two cards whose deciding cells read PROMOTE-ELIGIBLE (the
    /// Workstation and Max-Q Workstation editions), each with the placement its cell qualified,
    /// and nowhere else.
    #[test]
    fn the_planes_follow_the_card_class() {
        for (name, source) in [
            (PRO_WS, "pro6000-ws-receipt"),
            (
                "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",
                "pro6000-maxq-receipt",
            ),
        ] {
            let on = KvVmmConfig::from_values(None, name).unwrap();
            assert!(on.armed, "{name}");
            assert_eq!(on.placement, VmmGrowPlacement::Inline, "{name}");
            assert_eq!(on.placement_source, source, "{name}");
            assert!(on.faults.is_none(), "{name}");
        }
        for name in [
            "NVIDIA GeForce RTX 5090",
            "NVIDIA GeForce RTX 5090 Laptop GPU",
            "NVIDIA RTX PRO 6000 Blackwell Server Edition",
            "NVIDIA H100 80GB HBM3",
            "",
        ] {
            let off = KvVmmConfig::from_values(None, name).unwrap();
            assert!(!off.armed, "{name}");
            assert!(off.faults.is_none(), "{name}");
        }
    }

    #[test]
    fn the_fault_door_parses_on_every_card_and_refuses_what_cannot_fire() {
        let f = KvVmmConfig::from_values(Some("ensure:1,ensure:2,build:3"), PRO_WS)
            .unwrap()
            .faults
            .unwrap();
        assert!(!f.mapper_all);
        assert_eq!(f.ensure, vec![1, 2]);
        assert_eq!(f.build, vec![3]);
        // A malformed value refuses the boot on every card, not only where the planes arm.
        for name in [PRO_WS, "NVIDIA GeForce RTX 5090", ""] {
            assert!(
                KvVmmConfig::from_values(Some("bogus"), name).is_err(),
                "{name}"
            );
            assert!(
                KvVmmConfig::from_values(Some("grow:2"), name).is_err(),
                "{name}"
            );
            assert!(
                KvVmmConfig::from_values(Some("ensure:0"), name).is_err(),
                "{name}"
            );
        }
        // The receipted card grows inline, so no mapper thread exists for `mapper:all` to fail.
        let err = KvVmmConfig::from_values(Some("mapper:all,ensure:1"), PRO_WS).unwrap_err();
        assert!(err.contains("inline"), "{err}");
        // Where the planes are pooled the parsed spec is kept for the boot line and arms nothing.
        let off = KvVmmConfig::from_values(Some("mapper:all,ensure:1"), "NVIDIA GeForce RTX 5090")
            .unwrap();
        assert!(!off.armed);
        assert!(off.faults.as_ref().is_some_and(|f| f.mapper_all));
        assert!(
            off.boot_line(None)
                .contains("MEMRA_KV_VMM_FAULT injects nothing on pooled planes"),
            "{}",
            off.boot_line(None)
        );
    }

    /// The receipted card whose VMM query fails keeps pooled planes and says why, instead of
    /// refusing the boot; a card the planes do not arm on is never queried.
    #[test]
    fn a_vmm_refusal_on_the_receipted_card_falls_back_to_pooled_planes() {
        let on = KvVmmConfig::from_values(None, PRO_WS).unwrap();
        let (ok, g) = on.clone().with_granularity(Some(Ok(2 << 20)));
        assert!(ok.armed && ok.unsupported.is_none());
        assert_eq!(g, Some(2 << 20));

        let refusal = "REFUSED: VMM granularity query failed: CUDA_ERROR_NOT_SUPPORTED";
        let (fell, g) = on.clone().with_granularity(Some(Err(refusal.to_string())));
        assert!(!fell.armed);
        assert_eq!(g, None);
        assert_eq!(fell.unsupported.as_deref(), Some(refusal));
        assert_eq!(
            fell.boot_line(g),
            format!("[kv-vmm] door=OFF (VMM unsupported: {refusal}; pooled K/V planes)")
        );
        // A respawn that reads another refusal text runs the same program; one that reads VMM
        // support does not.
        let (again, _) = on
            .clone()
            .with_granularity(Some(Err("CUDA_ERROR_INVALID_VALUE".to_string())));
        assert!(fell.same_program(&again));
        assert!(!fell.same_program(&ok));
        // A fault spec on the fallen-back card is stated and stays inert.
        let faulted = KvVmmConfig::from_values(Some("ensure:1"), PRO_WS).unwrap();
        let (faulted, _) = faulted.with_granularity(Some(Err(refusal.to_string())));
        assert!(!faulted.armed);
        assert!(
            faulted
                .boot_line(None)
                .contains("injects nothing on pooled planes")
        );

        let off = KvVmmConfig::from_values(None, "NVIDIA GeForce RTX 5090").unwrap();
        let (off, g) = off.with_granularity(None);
        assert!(!off.armed && off.unsupported.is_none() && g.is_none());
    }

    #[test]
    fn the_bound_covers_one_tick_past_the_prompt_and_the_position() {
        // Spec: burst 32, k 4 -> 38 rows a tick.
        let spec = tick_rows(true, 32, 4, 1);
        assert_eq!(spec, 38);
        assert_eq!(tick_rows(false, 32, 4, 1), 1);
        assert_eq!(tick_rows(false, 32, 4, 0), 1);
        assert_eq!(tick_rows(false, 32, 4, 8), 8);
        // Priming: the floor (prompt) dominates the position.
        assert_eq!(bound_rows(65_536, 10, 5_000, spec), 5_000 + 38 + SLACK);
        // Decoding: the position dominates.
        assert_eq!(bound_rows(65_536, 9_000, 5_000, spec), 9_000 + 38 + SLACK);
        // Capped at the capacity.
        assert_eq!(bound_rows(9_010, 9_000, 5_000, spec), 9_010);
        assert_eq!(initial_rows(65_536, 5_000, spec), 5_000 + 38 + SLACK);
        assert_eq!(initial_rows(100, 5_000, spec), 100);
    }

    #[test]
    fn boot_line_states_the_planes_and_their_placement() {
        let off = KvVmmConfig::from_values(None, "NVIDIA GeForce RTX 5090").unwrap();
        assert!(off.boot_line(None).starts_with("[kv-vmm] door=OFF"));
        let on = KvVmmConfig::from_values(None, PRO_WS).unwrap();
        let line = on.boot_line(Some(2 << 20));
        assert!(line.starts_with(
            "[kv-vmm] door=ON routes=spec,plain granularity=2097152 grow=inline \
             (pro6000-ws-receipt)"
        ));
        assert!(line.contains("reap=owner-tick"));
        assert!(line.contains("faults=none"));
    }

    #[test]
    fn grow_line_locks_its_fields() {
        let ev = GrowEvent {
            offset: 0,
            bytes: 4 << 20,
            mapped: 4 << 20,
            reserved: 1 << 30,
            owner_us: 17,
            waited: true,
        };
        assert_eq!(
            grow_line("r1", "k3", 5_102, &ev),
            "[kv-vmm] grow id=r1 plane=k3 rows=5102 extent_bytes=4194304 mapped=4194304 \
             reserved=1073741824 owner_us=17 waited=1"
        );
    }
}
