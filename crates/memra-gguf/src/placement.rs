//! Pure contiguous multi-GPU stage placement.
//!
//! The planner consumes byte costs only. It does not inspect environment variables, open tensor
//! payloads, initialize CUDA, or choose an execution strategy. Runtime wiring remains a separate
//! qualification step.

use std::fmt;
use std::ops::Range;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LayerPlacementCost {
    pub weight_bytes: u64,
    pub kv_bytes_per_token: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlacementRequest<'a> {
    pub layers: &'a [LayerPlacementCost],
    /// Fixed bytes charged to each stage, in device order. This is where callers account for
    /// embeddings, the output head, stage workspaces, or other non-layer residency.
    pub fixed_stage_bytes: &'a [u64],
    pub context_tokens: u64,
    /// Distinct process-local CUDA ordinals. Phase 1 deliberately targets two through four cards.
    pub devices: &'a [usize],
    /// Legal cut positions between layers, using the `ModelPlan::partition_boundaries` convention:
    /// every value is strictly inside `1..layers.len()`.
    pub legal_boundaries: &'a [usize],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StagePlacementCost {
    pub weight_bytes: u64,
    pub kv_bytes: u64,
    pub fixed_bytes: u64,
    pub total_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StagePlacement {
    pub device: usize,
    pub layers: Range<usize>,
    pub cost: StagePlacementCost,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlacementPlan {
    pub stages: Vec<StagePlacement>,
    pub max_stage_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlacementError {
    DeviceCount { count: usize },
    DuplicateDevice { device: usize },
    TooFewLayers { layers: usize, stages: usize },
    FixedStageCount { fixed: usize, stages: usize },
    InvalidBoundary { boundary: usize, layers: usize },
    DuplicateBoundary { boundary: usize },
    InsufficientBoundaries { available: usize, required: usize },
    CostOverflow,
    NoPlacement,
}

impl fmt::Display for PlacementError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DeviceCount { count } => {
                write!(f, "stage placement requires 2..=4 devices, got {count}")
            }
            Self::DuplicateDevice { device } => {
                write!(f, "stage placement repeats device {device}")
            }
            Self::TooFewLayers { layers, stages } => {
                write!(
                    f,
                    "cannot place {layers} layers across {stages} non-empty stages"
                )
            }
            Self::FixedStageCount { fixed, stages } => write!(
                f,
                "fixed-stage cost count {fixed} does not match stage count {stages}"
            ),
            Self::InvalidBoundary { boundary, layers } => write!(
                f,
                "partition boundary {boundary} is outside the strict range 1..{layers}"
            ),
            Self::DuplicateBoundary { boundary } => {
                write!(f, "partition boundary {boundary} is duplicated")
            }
            Self::InsufficientBoundaries {
                available,
                required,
            } => write!(
                f,
                "only {available} legal partition boundaries are available; {required} required"
            ),
            Self::CostOverflow => write!(f, "stage placement byte cost overflows u64"),
            Self::NoPlacement => write!(f, "no legal contiguous stage placement exists"),
        }
    }
}

impl std::error::Error for PlacementError {}

#[derive(Clone)]
struct Candidate {
    peak: u64,
    cuts: Vec<usize>,
}

impl Candidate {
    fn better_than(&self, other: &Self) -> bool {
        self.peak < other.peak || (self.peak == other.peak && self.cuts < other.cuts)
    }
}

fn prefix_costs(layers: &[LayerPlacementCost]) -> Result<(Vec<u64>, Vec<u64>), PlacementError> {
    let mut weights = Vec::with_capacity(layers.len() + 1);
    let mut kv = Vec::with_capacity(layers.len() + 1);
    weights.push(0u64);
    kv.push(0u64);
    for layer in layers {
        weights.push(
            weights
                .last()
                .unwrap()
                .checked_add(layer.weight_bytes)
                .ok_or(PlacementError::CostOverflow)?,
        );
        kv.push(
            kv.last()
                .unwrap()
                .checked_add(layer.kv_bytes_per_token)
                .ok_or(PlacementError::CostOverflow)?,
        );
    }
    Ok((weights, kv))
}

fn stage_cost(
    stage: usize,
    start: usize,
    end: usize,
    weights: &[u64],
    kv: &[u64],
    request: &PlacementRequest<'_>,
) -> Result<StagePlacementCost, PlacementError> {
    let weight_bytes = weights[end] - weights[start];
    let kv_bytes_per_token = kv[end] - kv[start];
    let kv_bytes = kv_bytes_per_token
        .checked_mul(request.context_tokens)
        .ok_or(PlacementError::CostOverflow)?;
    let fixed_bytes = request.fixed_stage_bytes[stage];
    let total_bytes = weight_bytes
        .checked_add(kv_bytes)
        .and_then(|bytes| bytes.checked_add(fixed_bytes))
        .ok_or(PlacementError::CostOverflow)?;
    Ok(StagePlacementCost {
        weight_bytes,
        kv_bytes,
        fixed_bytes,
        total_bytes,
    })
}

/// Minimize the maximum stage byte cost over legal contiguous cuts.
///
/// Equal-peak plans choose the lexicographically smallest cut vector. The tie-break is explicit so
/// repeated planning cannot change placement because of hash iteration or input boundary order.
pub fn plan_contiguous_stages(
    request: PlacementRequest<'_>,
) -> Result<PlacementPlan, PlacementError> {
    let stages = request.devices.len();
    if !(2..=4).contains(&stages) {
        return Err(PlacementError::DeviceCount { count: stages });
    }
    let mut devices = request.devices.to_vec();
    devices.sort_unstable();
    for pair in devices.windows(2) {
        if pair[0] == pair[1] {
            return Err(PlacementError::DuplicateDevice { device: pair[0] });
        }
    }
    if request.layers.len() < stages {
        return Err(PlacementError::TooFewLayers {
            layers: request.layers.len(),
            stages,
        });
    }
    if request.fixed_stage_bytes.len() != stages {
        return Err(PlacementError::FixedStageCount {
            fixed: request.fixed_stage_bytes.len(),
            stages,
        });
    }

    let n_layers = request.layers.len();
    let mut legal = request.legal_boundaries.to_vec();
    for &boundary in &legal {
        if boundary == 0 || boundary >= n_layers {
            return Err(PlacementError::InvalidBoundary {
                boundary,
                layers: n_layers,
            });
        }
    }
    legal.sort_unstable();
    for pair in legal.windows(2) {
        if pair[0] == pair[1] {
            return Err(PlacementError::DuplicateBoundary { boundary: pair[0] });
        }
    }
    if legal.len() < stages - 1 {
        return Err(PlacementError::InsufficientBoundaries {
            available: legal.len(),
            required: stages - 1,
        });
    }

    let (weights, kv) = prefix_costs(request.layers)?;
    let mut ends = legal.clone();
    ends.push(n_layers);
    let mut dp = vec![vec![None::<Candidate>; n_layers + 1]; stages + 1];
    dp[0][0] = Some(Candidate {
        peak: 0,
        cuts: Vec::new(),
    });

    for stage in 0..stages {
        for &end in &ends {
            if stage + 1 == stages && end != n_layers {
                continue;
            }
            if stage + 1 < stages && end == n_layers {
                continue;
            }
            for start in 0..end {
                let Some(previous) = dp[stage][start].as_ref() else {
                    continue;
                };
                let cost = stage_cost(stage, start, end, &weights, &kv, &request)?;
                let mut cuts = previous.cuts.clone();
                if end != n_layers {
                    cuts.push(end);
                }
                let candidate = Candidate {
                    peak: previous.peak.max(cost.total_bytes),
                    cuts,
                };
                if dp[stage + 1][end]
                    .as_ref()
                    .is_none_or(|best| candidate.better_than(best))
                {
                    dp[stage + 1][end] = Some(candidate);
                }
            }
        }
    }

    let best = dp[stages][n_layers]
        .take()
        .ok_or(PlacementError::NoPlacement)?;
    // The minimum-prefix candidate retained by the DP can have later cuts than another prefix
    // whose larger local peak is hidden by the final global bottleneck. Reconstruct under the
    // proven optimal peak instead: suffix feasibility plus first-valid-cut selection yields the
    // lexicographically smallest complete plan among all equal-peak plans.
    let mut feasible = vec![vec![false; n_layers + 1]; stages + 1];
    feasible[stages][n_layers] = true;
    for stage in (0..stages).rev() {
        for start in (0..n_layers).rev() {
            for &end in &ends {
                if end <= start
                    || (stage + 1 == stages && end != n_layers)
                    || (stage + 1 < stages && end == n_layers)
                {
                    continue;
                }
                if stage_cost(stage, start, end, &weights, &kv, &request)?.total_bytes <= best.peak
                    && feasible[stage + 1][end]
                {
                    feasible[stage][start] = true;
                    break;
                }
            }
        }
    }
    if !feasible[0][0] {
        return Err(PlacementError::NoPlacement);
    }
    let mut cuts = Vec::with_capacity(stages - 1);
    let mut start = 0usize;
    for stage in 0..stages - 1 {
        let end = legal
            .iter()
            .copied()
            .find(|&end| {
                end > start
                    && stage_cost(stage, start, end, &weights, &kv, &request)
                        .is_ok_and(|cost| cost.total_bytes <= best.peak)
                    && feasible[stage + 1][end]
            })
            .ok_or(PlacementError::NoPlacement)?;
        cuts.push(end);
        start = end;
    }
    let mut boundaries = Vec::with_capacity(stages + 1);
    boundaries.push(0);
    boundaries.extend(cuts);
    boundaries.push(n_layers);
    let mut placements = Vec::with_capacity(stages);
    for stage in 0..stages {
        let start = boundaries[stage];
        let end = boundaries[stage + 1];
        placements.push(StagePlacement {
            device: request.devices[stage],
            layers: start..end,
            cost: stage_cost(stage, start, end, &weights, &kv, &request)?,
        });
    }
    Ok(PlacementPlan {
        stages: placements,
        max_stage_bytes: best.peak,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cost(weight_bytes: u64, kv_bytes_per_token: u64) -> LayerPlacementCost {
        LayerPlacementCost {
            weight_bytes,
            kv_bytes_per_token,
        }
    }

    fn boundaries(layers: usize) -> Vec<usize> {
        (1..layers).collect()
    }

    #[test]
    fn dense_uniform_costs_split_evenly_and_include_fixed_and_kv_bytes() {
        let layers = vec![cost(10, 1); 8];
        let legal = boundaries(layers.len());
        let plan = plan_contiguous_stages(PlacementRequest {
            layers: &layers,
            fixed_stage_bytes: &[5, 0],
            context_tokens: 10,
            devices: &[2, 3],
            legal_boundaries: &legal,
        })
        .unwrap();
        assert_eq!(plan.stages[0].layers, 0..4);
        assert_eq!(plan.stages[1].layers, 4..8);
        assert_eq!(plan.stages[0].cost.weight_bytes, 40);
        assert_eq!(plan.stages[0].cost.kv_bytes, 40);
        assert_eq!(plan.stages[0].cost.total_bytes, 85);
        assert_eq!(plan.max_stage_bytes, 85);
    }

    #[test]
    fn moe_heavy_tail_uses_three_contiguous_stages() {
        let layers = [
            cost(10, 0),
            cost(10, 0),
            cost(90, 0),
            cost(90, 0),
            cost(90, 0),
            cost(90, 0),
        ];
        let legal = boundaries(layers.len());
        let plan = plan_contiguous_stages(PlacementRequest {
            layers: &layers,
            fixed_stage_bytes: &[0, 0, 0],
            context_tokens: 0,
            devices: &[0, 1, 2],
            legal_boundaries: &legal,
        })
        .unwrap();
        assert_eq!(
            plan.stages
                .iter()
                .map(|stage| stage.layers.clone())
                .collect::<Vec<_>>(),
            vec![0..2, 2..4, 4..6]
        );
        assert_eq!(plan.max_stage_bytes, 180);
    }

    #[test]
    fn four_stage_uniform_plan_is_stable() {
        let layers = vec![cost(7, 0); 8];
        let legal = boundaries(layers.len());
        let plan = plan_contiguous_stages(PlacementRequest {
            layers: &layers,
            fixed_stage_bytes: &[0; 4],
            context_tokens: 0,
            devices: &[4, 5, 6, 7],
            legal_boundaries: &legal,
        })
        .unwrap();
        assert_eq!(
            plan.stages
                .iter()
                .map(|stage| stage.layers.clone())
                .collect::<Vec<_>>(),
            vec![0..2, 2..4, 4..6, 6..8]
        );
        assert_eq!(plan.max_stage_bytes, 14);
    }

    #[test]
    fn equal_peak_tie_chooses_lexicographically_first_cuts() {
        let layers = vec![cost(0, 0); 4];
        let legal = vec![3, 1, 2];
        let plan = plan_contiguous_stages(PlacementRequest {
            layers: &layers,
            fixed_stage_bytes: &[0, 0],
            context_tokens: 0,
            devices: &[0, 1],
            legal_boundaries: &legal,
        })
        .unwrap();
        assert_eq!(plan.stages[0].layers, 0..1);
        assert_eq!(plan.stages[1].layers, 1..4);
    }

    #[test]
    fn illegal_and_insufficient_boundaries_refuse() {
        let layers = vec![cost(1, 0); 6];
        for legal in [vec![0, 2], vec![2, 6], vec![2, 2]] {
            assert!(
                plan_contiguous_stages(PlacementRequest {
                    layers: &layers,
                    fixed_stage_bytes: &[0, 0],
                    context_tokens: 0,
                    devices: &[0, 1],
                    legal_boundaries: &legal,
                })
                .is_err()
            );
        }
        assert_eq!(
            plan_contiguous_stages(PlacementRequest {
                layers: &layers,
                fixed_stage_bytes: &[0, 0, 0],
                context_tokens: 0,
                devices: &[0, 1, 2],
                legal_boundaries: &[3],
            }),
            Err(PlacementError::InsufficientBoundaries {
                available: 1,
                required: 2,
            })
        );
    }

    #[test]
    fn byte_cost_overflow_refuses() {
        let layers = [cost(u64::MAX, 0), cost(1, 0)];
        assert_eq!(
            plan_contiguous_stages(PlacementRequest {
                layers: &layers,
                fixed_stage_bytes: &[0, 0],
                context_tokens: 1,
                devices: &[0, 1],
                legal_boundaries: &[1],
            }),
            Err(PlacementError::CostOverflow)
        );

        let layers = [cost(0, u64::MAX), cost(0, 0)];
        assert_eq!(
            plan_contiguous_stages(PlacementRequest {
                layers: &layers,
                fixed_stage_bytes: &[0, 0],
                context_tokens: 2,
                devices: &[0, 1],
                legal_boundaries: &[1],
            }),
            Err(PlacementError::CostOverflow)
        );
    }
}
