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
/// same device and environment and installs the same value; a different value is refused, so
/// the allocator cannot move mid-process.
pub(crate) fn install(cfg: KvVmmConfig) -> Result<(), String> {
    match CONFIG.get() {
        Some(existing) if *existing == cfg => Ok(()),
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
    /// The device is the full-power RTX PRO 6000 Blackwell Workstation Edition
    /// (`memra_engine::parallel::HardwareTarget::is_pro6000_workstation`).
    pub armed: bool,
    /// Where grows run (DAY37 1.5's rule per card class).
    pub placement: VmmGrowPlacement,
    /// Why the placement is what it is (the boot line states it).
    pub placement_source: &'static str,
    /// `MEMRA_KV_VMM_FAULT` (the fault door of the DAY37 A5 cells), read only when armed.
    pub faults: Option<VmmFaults>,
}

/// The per-class placement (`memra_engine::tier_transfer::kv_vmm_placement_for_device`).
pub(crate) fn placement_for_device(name: &str) -> (VmmGrowPlacement, &'static str) {
    memra_engine::tier_transfer::kv_vmm_placement_for_device(name)
}

impl KvVmmConfig {
    /// Arms the planes on the receipted card class and reads `MEMRA_KV_VMM_FAULT` there; an
    /// unparseable fault spec refuses the boot.
    pub(crate) fn for_device(device_name: &str) -> Result<Self, String> {
        Self::from_values(
            std::env::var("MEMRA_KV_VMM_FAULT").ok().as_deref(),
            device_name,
        )
    }

    pub(crate) fn from_values(fault: Option<&str>, device_name: &str) -> Result<Self, String> {
        let armed = memra_engine::parallel::HardwareTarget::is_pro6000_workstation(device_name);
        let (placement, placement_source) = placement_for_device(device_name);
        let mut faults = None;
        if armed && let Some(f) = fault.map(str::trim).filter(|f| !f.is_empty()) {
            faults = Some(VmmFaults::parse(f).ok_or_else(|| {
                format!(
                    "MEMRA_KV_VMM_FAULT={f:?}: expected a comma list of build:<n>, \
                     ensure:<n>, mapper:all"
                )
            })?);
        }
        Ok(Self {
            armed,
            placement,
            placement_source,
            faults,
        })
    }

    /// Boot receipt, one line.
    pub(crate) fn boot_line(&self, granularity: Option<usize>) -> String {
        if !self.armed {
            return "[kv-vmm] door=OFF (pooled K/V planes: the on-demand planes are the default \
                    on the RTX PRO 6000 Blackwell Workstation Edition only)"
                .to_string();
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

    /// The planes arm on the card class whose deciding cell read PROMOTE-ELIGIBLE, with the
    /// placement that cell qualified, and nowhere else.
    #[test]
    fn the_planes_follow_the_card_class() {
        let on = KvVmmConfig::from_values(None, PRO_WS).unwrap();
        assert!(on.armed);
        assert_eq!(on.placement, VmmGrowPlacement::Inline);
        assert_eq!(on.placement_source, "pro6000-ws-receipt");
        assert!(on.faults.is_none());
        for name in [
            "NVIDIA GeForce RTX 5090",
            "NVIDIA GeForce RTX 5090 Laptop GPU",
            "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",
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
    fn the_fault_door_is_read_only_when_armed_and_refuses_bad_values() {
        // Unarmed, the fault spec cannot arm anything and is not parsed.
        let off = KvVmmConfig::from_values(Some("bogus"), "NVIDIA GeForce RTX 5090").unwrap();
        assert!(!off.armed && off.faults.is_none());
        let f = KvVmmConfig::from_values(Some("mapper:all,ensure:3,build:1"), PRO_WS)
            .unwrap()
            .faults
            .unwrap();
        assert!(f.mapper_all);
        assert_eq!(f.ensure, vec![3]);
        assert_eq!(f.build, vec![1]);
        assert!(KvVmmConfig::from_values(Some("grow:2"), PRO_WS).is_err());
        assert!(KvVmmConfig::from_values(Some("ensure:0"), PRO_WS).is_err());
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
