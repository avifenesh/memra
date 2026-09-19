//! Checked, model-agnostic payload arithmetic. No allocation or support decisions.
//! Inputs must come from a compiled plan + tensor/record census. Estimates stay labeled.

use crate::contracts::WIRE_VERSION;
pub use crate::contracts::{DeviceBudget, Endpoint, PlacementReport, RouteBudget, RouteKind};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetError {
    Overflow,
    InvalidGeometry,
    UnknownDevice,
    DuplicateOwner,
    DuplicateReplica,
}

fn add(a: u64, b: u64) -> Result<u64, BudgetError> {
    a.checked_add(b).ok_or(BudgetError::Overflow)
}
fn mul(a: u64, b: u64) -> Result<u64, BudgetError> {
    a.checked_mul(b).ok_or(BudgetError::Overflow)
}

#[derive(Debug, Clone)]
pub struct WeightAllocation {
    pub bytes: u64,
    /// None means census-exact; Some names the estimation method, never hardware evidence.
    pub estimate: Option<String>,
}

#[derive(Debug, Clone)]
pub struct DevicePlan {
    pub device: u32,
    /// Physical capacity for payload planning, or free bytes plus already-charged residency.
    /// Caller must not subtract loaded weights twice when adapting a live free-VRAM query.
    pub capacity_bytes: u64,
    pub weights: Vec<WeightAllocation>,
    pub fixed_state_bytes_per_request: u64,
    pub staging_bytes: u64,
    pub loader_bytes: u64,
    pub scratch_bytes: u64,
    pub reserve_bytes: u64,
}

#[derive(Debug, Clone)]
pub struct RecordOwner {
    pub owner: u32,
    /// Physical copies, explicitly enumerated. Semantic aliases do not add a copy.
    pub devices: Vec<u32>,
    /// Entries = floor(context_tokens / tokens_per_record). Tail bytes belong in fixed state.
    pub tokens_per_record: u64,
    /// Includes scales and physical padding, not a logical value-vector width.
    pub record_bytes: u64,
}

#[derive(Debug, Clone)]
pub struct RoutePlan {
    pub from: Endpoint,
    pub to: Endpoint,
    pub kind: RouteKind,
    /// Demand trace, NOT inferred from retained KV slope.
    pub bytes_per_generated_token: u64,
    pub generated_tokens_per_second: u64,
    pub restore_bytes: u64,
    pub restores_per_second: u64,
    /// Measured sustained route rate only; None is unknown, not nominal PCIe capacity.
    pub measured_bytes_per_second: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct PlacementPlan {
    pub devices: Vec<DevicePlan>,
    pub owners: Vec<RecordOwner>,
    pub routes: Vec<RoutePlan>,
    pub host_resident_bytes: u64,
    pub host_staging_bytes: u64,
    pub host_loader_and_os_bytes: u64,
}

/// Returns negative headroom rather than hiding oversubscription or saturating arithmetic.
/// This report is NOT an admission permit: unknown scratch and estimated weights remain unknown.
pub fn placement_report(
    plan: &PlacementPlan,
    context_tokens: u64,
    requests: u64,
) -> Result<PlacementReport, BudgetError> {
    use std::collections::BTreeSet;
    if plan.devices.is_empty() || context_tokens == 0 || requests == 0 {
        return Err(BudgetError::InvalidGeometry);
    }
    let devices: BTreeSet<_> = plan.devices.iter().map(|d| d.device).collect();
    if devices.len() != plan.devices.len() {
        return Err(BudgetError::InvalidGeometry);
    }
    let mut owners = BTreeSet::new();
    for owner in &plan.owners {
        if !owners.insert(owner.owner) {
            return Err(BudgetError::DuplicateOwner);
        }
        if owner.tokens_per_record == 0 || owner.record_bytes == 0 || owner.devices.is_empty() {
            return Err(BudgetError::InvalidGeometry);
        }
        let mut replicas = BTreeSet::new();
        for device in &owner.devices {
            if !devices.contains(device) {
                return Err(BudgetError::UnknownDevice);
            }
            if !replicas.insert(device) {
                return Err(BudgetError::DuplicateReplica);
            }
        }
    }
    let mut per_device = Vec::new();
    for device in &plan.devices {
        let weight_bytes = device.weights.iter().try_fold(0, |n, w| add(n, w.bytes))?;
        let global_kv_bytes = plan
            .owners
            .iter()
            .filter(|o| o.devices.contains(&device.device))
            .try_fold(0, |n, o| {
                add(
                    n,
                    mul(context_tokens / o.tokens_per_record, o.record_bytes)?,
                )
            })?;
        let global_kv_bytes = mul(global_kv_bytes, requests)?;
        let fixed_state_bytes = mul(device.fixed_state_bytes_per_request, requests)?;
        let total_bytes = [
            weight_bytes,
            global_kv_bytes,
            fixed_state_bytes,
            device.staging_bytes,
            device.loader_bytes,
            device.scratch_bytes,
            device.reserve_bytes,
        ]
        .into_iter()
        .try_fold(0, add)?;
        let replica_bytes = plan
            .owners
            .iter()
            .filter(|o| o.devices.iter().skip(1).any(|d| *d == device.device))
            .try_fold(0, |n, o| {
                add(
                    n,
                    mul(
                        mul(context_tokens / o.tokens_per_record, o.record_bytes)?,
                        requests,
                    )?,
                )
            })?;
        per_device.push(DeviceBudget {
            version: WIRE_VERSION,
            replica_bytes,
            device: device.device,
            weight_bytes,
            global_kv_bytes,
            fixed_state_bytes,
            staging_bytes: device.staging_bytes,
            loader_bytes: device.loader_bytes,
            scratch_bytes: device.scratch_bytes,
            reserve_bytes: device.reserve_bytes,
            total_bytes,
            remaining_bytes: i128::from(device.capacity_bytes) - i128::from(total_bytes),
            estimates: device
                .weights
                .iter()
                .filter_map(|w| w.estimate.clone())
                .collect(),
        });
    }
    let mut routes = Vec::new();
    for route in &plan.routes {
        for endpoint in [route.from, route.to] {
            if let Endpoint::Device(d) = endpoint {
                if !devices.contains(&d) {
                    return Err(BudgetError::UnknownDevice);
                }
            }
        }
        let valid = match (route.from, route.to, route.kind) {
            (Endpoint::Device(a), Endpoint::Device(b), RouteKind::Local) => a == b,
            (
                Endpoint::Device(a),
                Endpoint::Device(b),
                RouteKind::PcieP2p | RouteKind::HostBounce,
            ) => a != b,
            (Endpoint::Device(_), Endpoint::PinnedHost, RouteKind::HostDevice)
            | (Endpoint::PinnedHost, Endpoint::Device(_), RouteKind::HostDevice)
            | (Endpoint::PinnedHost, Endpoint::Nvme, RouteKind::Storage)
            | (Endpoint::Nvme, Endpoint::PinnedHost, RouteKind::Storage) => true,
            _ => false,
        };
        if !valid || route.measured_bytes_per_second == Some(0) {
            return Err(BudgetError::InvalidGeometry);
        }
        let demand = add(
            mul(
                route.bytes_per_generated_token,
                route.generated_tokens_per_second,
            )?,
            mul(route.restore_bytes, route.restores_per_second)?,
        )?;
        routes.push(RouteBudget {
            version: WIRE_VERSION,
            measured_bytes_per_second: route.measured_bytes_per_second,
            estimates: vec!["Caller-supplied demand; shared-fabric capacity unmeasured".into()],
            from: route.from,
            to: route.to,
            kind: route.kind,
            demand_bytes_per_second: demand,
            physical_bytes_per_second: mul(
                demand,
                if route.kind == RouteKind::HostBounce {
                    2
                } else {
                    1
                },
            )?,
            within_measured_envelope: route
                .measured_bytes_per_second
                .map(|rate| u128::from(demand) * 10 <= u128::from(rate) * 7),
        });
    }
    Ok(PlacementReport {
        version: WIRE_VERSION,
        estimates: vec![
            "Payload estimate only; unmeasured runtime workspace is not admission evidence".into(),
        ],
        per_device,
        routes,
        host_bytes: add(
            add(plan.host_resident_bytes, plan.host_staging_bytes)?,
            plan.host_loader_and_os_bytes,
        )?,
    })
}
