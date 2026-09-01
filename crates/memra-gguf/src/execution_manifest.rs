//! Static operation-level manifests for tuned execution programs.
//!
//! These tables describe implemented kernel programs, not model families. A new model can use a
//! program when every operation in its compiled `ModelPlan` is present. Missing operations fail
//! closed and remain visible as capability blockers.

use crate::model_plan::{ModelPlan, OperationKind, OperationSupport, PlanCapabilities};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::path::Path;

#[derive(Clone, Copy)]
pub struct KernelManifest {
    pub name: &'static str,
    support: fn(OperationKind) -> OperationSupport,
}

impl KernelManifest {
    pub const fn new(name: &'static str, support: fn(OperationKind) -> OperationSupport) -> Self {
        Self { name, support }
    }

    pub fn support(self, operation: OperationKind) -> OperationSupport {
        (self.support)(operation)
    }

    pub fn capabilities(self, plan: &ModelPlan) -> PlanCapabilities {
        plan.derive_capabilities(|operation| self.support(operation))
    }

    pub fn trunk_capabilities(self, plan: &ModelPlan) -> PlanCapabilities {
        plan.derive_trunk_capabilities(|operation| self.support(operation))
    }

    pub fn multimodal_prefill_capabilities(self, plan: &ModelPlan) -> PlanCapabilities {
        plan.derive_multimodal_prefill_capabilities(|operation| self.support(operation))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RewriteSurface {
    CarriedPrime,
    DecodeEager,
    DecodeBatch,
    DecodeGraph,
    MtpSpec,
    /// glm5_next T-parallel speculative loop (embedded MLA-mixer MTP draft + one-walk
    /// verify over the HyperConnections trunk). A DISTINCT program from `MtpSpec` — the
    /// two manifests' operation tables are disjoint on the mixer/residual classes, so no
    /// plan is ever eligible for both.
    Glm5Spec,
    Pipeline,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RewriteQualifications {
    pub plan_sha256: String,
    passed: BTreeSet<RewriteSurface>,
}

impl RewriteQualifications {
    pub fn load(bundle: &Path, plan: &ModelPlan) -> Result<Self, String> {
        let expected = execution_rewrites(plan);
        let artifact_lock = std::fs::read(bundle.join("artifact.lock"))
            .map_err(|error| format!("read artifact.lock: {error}"))?;
        let artifact_lock_sha256 = hex_sha256(&artifact_lock);
        let index = std::fs::read_to_string(bundle.join("rewrite-receipts.tsv"))
            .map_err(|error| format!("read rewrite receipt index: {error}"))?;
        let mut passed = BTreeSet::new();
        for line in index.lines().skip(1) {
            let columns: Vec<_> = line.split('\t').collect();
            if columns.len() != 4 || columns[3] != "passed" {
                return Err(format!("malformed rewrite receipt index row {line:?}"));
            }
            let rewrite = expected
                .iter()
                .find(|rewrite| rewrite.id == columns[0])
                .ok_or_else(|| format!("receipt names unknown rewrite {}", columns[0]))?;
            if !rewrite.eligible() || columns[1] != rewrite.plan_sha256 {
                return Err(format!(
                    "receipt {} is not eligible for plan {}",
                    rewrite.id, rewrite.plan_sha256
                ));
            }
            let receipt_path = bundle
                .join("rewrite-receipts")
                .join(format!("{}.tsv", rewrite.id));
            let receipt = std::fs::read(&receipt_path)
                .map_err(|error| format!("read {}: {error}", receipt_path.display()))?;
            if hex_sha256(&receipt) != columns[2] {
                return Err(format!("rewrite receipt hash mismatch for {}", rewrite.id));
            }
            let text = std::str::from_utf8(&receipt)
                .map_err(|error| format!("rewrite receipt is not UTF-8: {error}"))?;
            for (key, value) in [
                ("status", "passed"),
                ("rewrite", rewrite.id),
                ("surface", rewrite.surface.as_str()),
                ("implementation", rewrite.implementation),
                ("plan_sha256", rewrite.plan_sha256.as_str()),
                ("artifact_lock_sha256", artifact_lock_sha256.as_str()),
                ("first_violation", "none"),
            ] {
                if !text.lines().any(|line| line == format!("{key}\t{value}")) {
                    return Err(format!(
                        "rewrite receipt {} does not bind {key}={value}",
                        rewrite.id
                    ));
                }
            }
            passed.insert(rewrite.surface);
        }
        Ok(Self {
            plan_sha256: plan_sha256(plan),
            passed,
        })
    }

    pub fn allows(&self, surface: RewriteSurface) -> bool {
        self.passed.contains(&surface)
    }

    pub fn all_eligible(&self, plan: &ModelPlan) -> bool {
        execution_rewrites(plan)
            .into_iter()
            .filter(ExecutionRewrite::eligible)
            .all(|rewrite| self.allows(rewrite.surface))
    }
}

impl RewriteSurface {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CarriedPrime => "carried-prime",
            Self::DecodeEager => "decode-eager",
            Self::DecodeBatch => "decode-batch",
            Self::DecodeGraph => "decode-graph",
            Self::MtpSpec => "mtp-spec",
            Self::Glm5Spec => "glm5-spec",
            Self::Pipeline => "pipeline",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionRewrite {
    pub id: &'static str,
    pub surface: RewriteSurface,
    pub implementation: &'static str,
    pub plan_sha256: String,
    pub canonical_operations: Vec<OperationKind>,
    pub blockers: Vec<OperationKind>,
}

impl ExecutionRewrite {
    pub fn eligible(&self) -> bool {
        self.blockers.is_empty()
    }

    pub fn verify_logits(
        &self,
        implementation_sha256: &str,
        reference: &[f32],
        candidate: &[f32],
        policy: RewriteParityPolicy,
    ) -> Result<RewriteParityReceipt, String> {
        if !self.eligible() {
            return Err(format!(
                "rewrite {} is blocked by {:?}",
                self.id, self.blockers
            ));
        }
        if !is_sha256(implementation_sha256) {
            return Err("rewrite implementation identity must be a lowercase SHA-256".into());
        }
        if reference.len() != candidate.len() || reference.is_empty() {
            return Err(format!(
                "rewrite parity requires equal non-empty streams (reference={} candidate={})",
                reference.len(),
                candidate.len()
            ));
        }
        let mut max_abs = 0.0f32;
        let mut max_rel = 0.0f32;
        let mut first_violation = None;
        for (index, (&expected, &actual)) in reference.iter().zip(candidate).enumerate() {
            if !expected.is_finite() || !actual.is_finite() {
                return Err(format!("rewrite parity has a non-finite value at {index}"));
            }
            let absolute = (expected - actual).abs();
            let relative = absolute / expected.abs().max(1e-6);
            max_abs = max_abs.max(absolute);
            max_rel = max_rel.max(relative);
            let allowed = policy.max_abs + policy.max_rel * expected.abs();
            if absolute > allowed && first_violation.is_none() {
                first_violation = Some(index);
            }
        }
        let reference_argmax = stable_argmax(reference);
        let candidate_argmax = stable_argmax(candidate);
        let passed = first_violation.is_none()
            && (!policy.require_argmax || reference_argmax == candidate_argmax);
        Ok(RewriteParityReceipt {
            rewrite_id: self.id,
            surface: self.surface,
            implementation: self.implementation,
            implementation_sha256: implementation_sha256.to_string(),
            plan_sha256: self.plan_sha256.clone(),
            artifact_lock_sha256: None,
            reference_sha256: f32_stream_sha256(reference),
            candidate_sha256: f32_stream_sha256(candidate),
            value_kind: RewriteValueKind::LogitsF32,
            values: reference.len(),
            max_abs,
            max_rel,
            reference_argmax,
            candidate_argmax,
            policy,
            passed,
            first_violation,
        })
    }

    pub fn verify_tokens(
        &self,
        implementation_sha256: &str,
        reference: &[u32],
        candidate: &[u32],
    ) -> Result<RewriteParityReceipt, String> {
        if !self.eligible() {
            return Err(format!(
                "rewrite {} is blocked by {:?}",
                self.id, self.blockers
            ));
        }
        if !is_sha256(implementation_sha256) {
            return Err("rewrite implementation identity must be a lowercase SHA-256".into());
        }
        if reference.len() != candidate.len() || reference.is_empty() {
            return Err(format!(
                "rewrite parity requires equal non-empty token streams (reference={} candidate={})",
                reference.len(),
                candidate.len()
            ));
        }
        let first_violation = reference
            .iter()
            .zip(candidate)
            .position(|(expected, actual)| expected != actual);
        let max_abs = reference
            .iter()
            .zip(candidate)
            .map(|(&expected, &actual)| expected.abs_diff(actual) as f32)
            .fold(0.0f32, f32::max);
        Ok(RewriteParityReceipt {
            rewrite_id: self.id,
            surface: self.surface,
            implementation: self.implementation,
            implementation_sha256: implementation_sha256.to_string(),
            plan_sha256: self.plan_sha256.clone(),
            artifact_lock_sha256: None,
            reference_sha256: u32_stream_sha256(reference),
            candidate_sha256: u32_stream_sha256(candidate),
            value_kind: RewriteValueKind::TokenIdsU32,
            values: reference.len(),
            max_abs,
            max_rel: 0.0,
            reference_argmax: 0,
            candidate_argmax: 0,
            policy: RewriteParityPolicy {
                max_abs: 0.0,
                max_rel: 0.0,
                require_argmax: false,
            },
            passed: first_violation.is_none(),
            first_violation,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RewriteParityPolicy {
    pub max_abs: f32,
    pub max_rel: f32,
    pub require_argmax: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RewriteParityReceipt {
    pub rewrite_id: &'static str,
    pub surface: RewriteSurface,
    pub implementation: &'static str,
    pub implementation_sha256: String,
    pub plan_sha256: String,
    pub artifact_lock_sha256: Option<String>,
    pub reference_sha256: String,
    pub candidate_sha256: String,
    pub value_kind: RewriteValueKind,
    pub values: usize,
    pub max_abs: f32,
    pub max_rel: f32,
    pub reference_argmax: usize,
    pub candidate_argmax: usize,
    pub policy: RewriteParityPolicy,
    pub passed: bool,
    pub first_violation: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RewriteValueKind {
    LogitsF32,
    TokenIdsU32,
}

impl RewriteValueKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::LogitsF32 => "logits-f32",
            Self::TokenIdsU32 => "token-ids-u32",
        }
    }
}

impl RewriteParityReceipt {
    pub fn bind_artifact_lock(mut self, artifact_lock: &[u8]) -> Self {
        self.artifact_lock_sha256 = Some(hex_sha256(artifact_lock));
        self
    }

    pub fn validate_for(&self, rewrite: &ExecutionRewrite) -> Result<(), String> {
        if self.rewrite_id != rewrite.id
            || self.surface != rewrite.surface
            || self.implementation != rewrite.implementation
            || self.plan_sha256 != rewrite.plan_sha256
        {
            return Err(format!(
                "rewrite receipt identity does not match {} for plan {}",
                rewrite.id, rewrite.plan_sha256
            ));
        }
        if !rewrite.eligible() {
            return Err(format!("rewrite {} is no longer eligible", rewrite.id));
        }
        if !self.passed {
            return Err(format!("rewrite {} parity receipt failed", rewrite.id));
        }
        Ok(())
    }

    pub fn to_tsv(&self) -> String {
        let mut output = String::new();
        writeln!(output, "format\tmemra-rewrite-parity-v1").unwrap();
        writeln!(
            output,
            "status\t{}",
            if self.passed { "passed" } else { "failed" }
        )
        .unwrap();
        writeln!(output, "rewrite\t{}", self.rewrite_id).unwrap();
        writeln!(output, "surface\t{}", self.surface.as_str()).unwrap();
        writeln!(output, "implementation\t{}", self.implementation).unwrap();
        writeln!(
            output,
            "implementation_sha256\t{}",
            self.implementation_sha256
        )
        .unwrap();
        writeln!(output, "plan_sha256\t{}", self.plan_sha256).unwrap();
        if let Some(hash) = self.artifact_lock_sha256.as_ref() {
            writeln!(output, "artifact_lock_sha256\t{hash}").unwrap();
        }
        writeln!(output, "reference_sha256\t{}", self.reference_sha256).unwrap();
        writeln!(output, "candidate_sha256\t{}", self.candidate_sha256).unwrap();
        writeln!(output, "value_kind\t{}", self.value_kind.as_str()).unwrap();
        writeln!(output, "values\t{}", self.values).unwrap();
        writeln!(output, "max_abs\t{}", self.max_abs).unwrap();
        writeln!(output, "max_rel\t{}", self.max_rel).unwrap();
        writeln!(output, "reference_argmax\t{}", self.reference_argmax).unwrap();
        writeln!(output, "candidate_argmax\t{}", self.candidate_argmax).unwrap();
        writeln!(output, "atol\t{}", self.policy.max_abs).unwrap();
        writeln!(output, "rtol\t{}", self.policy.max_rel).unwrap();
        writeln!(output, "require_argmax\t{}", self.policy.require_argmax).unwrap();
        writeln!(
            output,
            "first_violation\t{}",
            self.first_violation
                .map_or_else(|| "none".to_string(), |index| index.to_string())
        )
        .unwrap();
        output
    }
}

pub fn execution_rewrites(plan: &ModelPlan) -> Vec<ExecutionRewrite> {
    let plan_sha256 = plan_sha256(plan);
    let trunk = plan.trunk_operations();
    let mut spec_operations = plan
        .draft_operations()
        .unwrap_or_else(|| vec![OperationKind::DraftPlan]);
    spec_operations.extend(plan.trunk_operations());
    let selections = [
        (
            "carried-prime.v1",
            RewriteSurface::CarriedPrime,
            CARRIED_PRIME,
            trunk.clone(),
            CARRIED_PRIME.trunk_capabilities(plan).batch,
        ),
        (
            "decode-eager.v1",
            RewriteSurface::DecodeEager,
            NATIVE_EAGER,
            trunk.clone(),
            NATIVE_EAGER.trunk_capabilities(plan).batch,
        ),
        (
            "decode-batch.v1",
            RewriteSurface::DecodeBatch,
            DECODE_BATCH,
            trunk.clone(),
            DECODE_BATCH.trunk_capabilities(plan).batch,
        ),
        (
            "decode-graph.v1",
            RewriteSurface::DecodeGraph,
            DECODE_GRAPH,
            trunk.clone(),
            DECODE_GRAPH.trunk_capabilities(plan).cuda_graph,
        ),
        (
            "mtp-spec.v1",
            RewriteSurface::MtpSpec,
            MTP_SPEC,
            spec_operations.clone(),
            MTP_SPEC.capabilities(plan).speculative,
        ),
        (
            "glm5-spec.v1",
            RewriteSurface::Glm5Spec,
            GLM5_SPEC,
            spec_operations,
            GLM5_SPEC.capabilities(plan).speculative,
        ),
        (
            "pipeline.v1",
            RewriteSurface::Pipeline,
            PIPELINE,
            trunk,
            PIPELINE.capabilities(plan).pipeline,
        ),
    ];
    selections
        .into_iter()
        .map(
            |(id, surface, manifest, canonical_operations, capability)| ExecutionRewrite {
                id,
                surface,
                implementation: manifest.name,
                plan_sha256: plan_sha256.clone(),
                canonical_operations,
                blockers: capability.blockers,
            },
        )
        .collect()
}

fn plan_sha256(plan: &ModelPlan) -> String {
    hex_sha256(format!("{plan:#?}\n").as_bytes())
}

fn f32_stream_sha256(values: &[f32]) -> String {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    hex_sha256(&bytes)
}

fn u32_stream_sha256(values: &[u32]) -> String {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    hex_sha256(&bytes)
}

fn hex_sha256(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn stable_argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(left_index, left), (right_index, right)| {
            left.total_cmp(right)
                .then_with(|| right_index.cmp(left_index))
        })
        .map(|(index, _)| index)
        .unwrap_or(0)
}

fn carried_prime_support(operation: OperationKind) -> OperationSupport {
    let mut support = OperationSupport::none();
    support.batch = matches!(
        operation,
        OperationKind::Embedding
            | OperationKind::RmsNorm
            | OperationKind::FullAttention
            | OperationKind::GatedDeltaNet
            | OperationKind::FusedAttentionGate
            | OperationKind::DenseMlp
            | OperationKind::SiluActivation
            | OperationKind::SerialResidual
            | OperationKind::KvState
            | OperationKind::RecurrentState
            | OperationKind::LogitsMask
            | OperationKind::OutputProjection
    );
    support
}

pub const CARRIED_PRIME: KernelManifest =
    KernelManifest::new("carried-prime-batch", carried_prime_support);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeBatchProgram {
    Generic,
    SlidingGatedMoe,
    Gemma,
}

pub fn decode_batch_program(plan: &ModelPlan) -> DecodeBatchProgram {
    let operations = plan.trunk_operations();
    if operations.contains(&OperationKind::GemmaResidual)
        || operations.contains(&OperationKind::GemmaParallelMoeResidual)
    {
        DecodeBatchProgram::Gemma
    } else if operations.contains(&OperationKind::SlidingWindowAttention)
        && operations.contains(&OperationKind::SeparateAttentionGate)
        && operations.contains(&OperationKind::MoeMlp)
        && operations.contains(&OperationKind::SigmoidRouter)
    {
        DecodeBatchProgram::SlidingGatedMoe
    } else {
        DecodeBatchProgram::Generic
    }
}

/// Does this plan's residual topology have NO converted batched-decode arm?
///
/// `HyperConnections` carries `streams` parallel residual streams collapsed per layer; every
/// batched path (`decode_step_batch*`, `prime_cache_batch`, `prime_graph_new`, the graph-capture
/// and speculative entry points) runs a serial residual and refuses this topology at the engine
/// boundary rather than computing a different model. The converted set is `forward`,
/// `forward_last`, `prime_cache`, `decode_step` — all reachable from the per-session eager route.
///
/// This is the plan-level twin of the `Gemma` arm of [`decode_batch_program`]: the scheduler needs
/// to know a model has no batched arm BEFORE it places the session, so the topology is read from
/// the `ModelPlan` rather than re-derived from engine internals.
pub fn decode_batch_unconverted(plan: &ModelPlan) -> bool {
    plan.trunk_operations()
        .contains(&OperationKind::HyperConnections)
}

pub fn gdn_dspark_compatible(plan: &ModelPlan) -> bool {
    let operations = plan.trunk_operations();
    operations.contains(&OperationKind::GatedDeltaNet)
        && operations.contains(&OperationKind::FusedAttentionGate)
}

fn decode_graph_support(operation: OperationKind) -> OperationSupport {
    let mut support = OperationSupport::none();
    support.cuda_graph = matches!(
        operation,
        OperationKind::Embedding
            | OperationKind::RmsNorm
            | OperationKind::FullAttention
            | OperationKind::GatedDeltaNet
            | OperationKind::FusedAttentionGate
            | OperationKind::DenseMlp
            | OperationKind::MoeMlp
            | OperationKind::SharedMlp
            | OperationKind::SoftmaxRouter
            | OperationKind::SigmoidRouter
            | OperationKind::SiluActivation
            | OperationKind::SerialResidual
            | OperationKind::KvState
            | OperationKind::RecurrentState
            | OperationKind::LogitsMask
            | OperationKind::OutputProjection
    );
    support
}

pub const DECODE_GRAPH: KernelManifest =
    KernelManifest::new("decode-cuda-graph", decode_graph_support);

fn pipeline_support(operation: OperationKind) -> OperationSupport {
    let mut support = OperationSupport::none();
    support.pipeline = matches!(
        operation,
        OperationKind::Embedding
            | OperationKind::RmsNorm
            | OperationKind::FullAttention
            | OperationKind::SlidingWindowAttention
            | OperationKind::SeparateAttentionGate
            | OperationKind::DenseMlp
            | OperationKind::MoeMlp
            | OperationKind::SharedMlp
            | OperationKind::SigmoidRouter
            | OperationKind::SiluActivation
            | OperationKind::SwiGluClampedActivation
            | OperationKind::SerialResidual
            | OperationKind::KvState
            | OperationKind::SlidingKvState
            | OperationKind::Mtp
            | OperationKind::MtpFusion
            | OperationKind::MtpHead
            | OperationKind::LogitsMask
            | OperationKind::OutputProjection
            | OperationKind::PipelineBoundary
    );
    support
}

pub const PIPELINE: KernelManifest =
    KernelManifest::new("pipeline-state-transport", pipeline_support);

fn decode_batch_support(operation: OperationKind) -> OperationSupport {
    let mut support = OperationSupport::none();
    support.batch = matches!(
        operation,
        OperationKind::Embedding
            | OperationKind::RmsNorm
            | OperationKind::FullAttention
            | OperationKind::SlidingWindowAttention
            | OperationKind::GatedDeltaNet
            | OperationKind::FusedAttentionGate
            | OperationKind::SeparateAttentionGate
            | OperationKind::DenseMlp
            | OperationKind::MoeMlp
            | OperationKind::SharedMlp
            | OperationKind::SoftmaxRouter
            | OperationKind::SigmoidRouter
            | OperationKind::SiluActivation
            | OperationKind::GeluTanhActivation
            | OperationKind::SwiGluClampedActivation
            | OperationKind::SerialResidual
            | OperationKind::GemmaResidual
            | OperationKind::GemmaParallelMoeResidual
            | OperationKind::KvState
            | OperationKind::SlidingKvState
            | OperationKind::RecurrentState
            | OperationKind::LogitsSoftcap
            | OperationKind::LogitsMask
            | OperationKind::OutputProjection
    );
    support
}

pub const DECODE_BATCH: KernelManifest = KernelManifest::new("decode-batch", decode_batch_support);

fn native_eager_support(operation: OperationKind) -> OperationSupport {
    let mut support = OperationSupport::none();
    support.batch = matches!(
        operation,
        OperationKind::Embedding
            | OperationKind::RmsNorm
            | OperationKind::FullAttention
            | OperationKind::DenseMlp
            | OperationKind::SiluActivation
            | OperationKind::SerialResidual
            | OperationKind::KvState
            | OperationKind::LogitsSoftcap
            | OperationKind::LogitsMask
            | OperationKind::OutputProjection
    );
    support
}

pub const NATIVE_EAGER: KernelManifest = KernelManifest::new("native-eager", native_eager_support);

fn mtp_spec_support(operation: OperationKind) -> OperationSupport {
    let mut support = OperationSupport::none();
    let common = matches!(
        operation,
        OperationKind::Embedding
            | OperationKind::RmsNorm
            | OperationKind::FullAttention
            | OperationKind::SlidingWindowAttention
            | OperationKind::GatedDeltaNet
            | OperationKind::FusedAttentionGate
            | OperationKind::SeparateAttentionGate
            | OperationKind::DenseMlp
            | OperationKind::MoeMlp
            | OperationKind::SharedMlp
            | OperationKind::SoftmaxRouter
            | OperationKind::SigmoidRouter
            | OperationKind::SiluActivation
            | OperationKind::SwiGluClampedActivation
            | OperationKind::SerialResidual
            | OperationKind::KvState
            | OperationKind::SlidingKvState
            | OperationKind::RecurrentState
            | OperationKind::LogitsMask
            | OperationKind::OutputProjection
    );
    support.spec_draft = common
        || matches!(
            operation,
            OperationKind::Mtp | OperationKind::MtpFusion | OperationKind::MtpHead
        );
    support.spec_verify = common;
    support
}

pub const MTP_SPEC: KernelManifest = KernelManifest::new("mtp-spec", mtp_spec_support);

/// glm5_next T-parallel speculative loop (lane/glm5-spec-routing, 2026-08-30): the gated
/// program is `glm_spec.rs` — draft K through the embedded MLA-mixer MTP block
/// (`mtp_head_forward_mla_cached`), verify in ONE t=K+1 walk over the HyperConnections
/// trunk in the batched-decode kernel classes (`glm5_verify_rows`), greedy or
/// rejection-sampled accept, per-plane rollback (`glm5_verify_rollback`).
///
/// The table below names EXACTLY the operation classes those gates cover — the
/// glm5_next class (hc residual topology on every trunk layer + KDA/MLA(+kpool) mixers,
/// MoE/dense FFN) with a serial-residual MLA-mixer MoE MTP block. Deliberately absent, so
/// every other family stays fail-closed here (no-generic-support law; each carries its own
/// gated program or none):
///   * `FullAttention`/`SlidingWindowAttention`/`GatedDeltaNet`/`KvState`/... — the qwen35 /
///     step35 / gemma classes route through `MTP_SPEC`, never here;
///   * `CompressedMlaAttention` — the dsv4 class (its own dspark program);
///   * `SerialResidual`/`Gemma*Residual` as TRUNK residuals — `glm5_verify_rows` requires
///     the hc topology (the walk's `hyper::pre_exact`/`post` glue IS the program), so a
///     non-hc trunk must block. `SerialResidual` is spec_draft-only: the NextN block
///     carries no hc_* tensors (mtp-draft LANE.md) and is serial by plan.
///   * `DenseMlp` on the DRAFT side — `mtp_head_forward_mla_cached` refuses Dense-FFN MTP
///     blocks by name; the trunk's dense L0 is in the gated walk (`hyper_ffn_branch_batch`).
fn glm5_spec_support(operation: OperationKind) -> OperationSupport {
    let mut support = OperationSupport::none();
    let common = matches!(
        operation,
        OperationKind::Embedding
            | OperationKind::RmsNorm
            | OperationKind::LatentMlaAttention
            | OperationKind::SparseIndex
            | OperationKind::SharedSparseIndex
            | OperationKind::KimiDeltaNet
            | OperationKind::RecurrentState
            | OperationKind::LatentKvState
            | OperationKind::MoeMlp
            | OperationKind::SigmoidRouter
            | OperationKind::SharedMlp
            | OperationKind::SwiGluPreClampedActivation
            | OperationKind::OutputProjection
    );
    support.spec_verify = common
        || matches!(
            operation,
            OperationKind::HyperConnections | OperationKind::DenseMlp
        );
    support.spec_draft = common
        || matches!(
            operation,
            OperationKind::Mtp
                | OperationKind::MtpFusion
                | OperationKind::MtpHead
                | OperationKind::SerialResidual
        );
    support
}

pub const GLM5_SPEC: KernelManifest = KernelManifest::new("glm5-spec", glm5_spec_support);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{HfConfig, ModelConfig};

    fn plan(json: &str) -> ModelPlan {
        ModelPlan::compile(&ModelConfig::from_hf(&HfConfig::parse(json))).unwrap()
    }

    #[test]
    fn manifest_reports_operation_blockers_instead_of_model_names() {
        let dense = plan(
            r#"{"model_type":"qwen3","num_hidden_layers":2,"hidden_size":64,
            "num_attention_heads":2,"num_key_value_heads":1,"head_dim":32,
            "intermediate_size":128,"vocab_size":16,"max_position_embeddings":128}"#,
        );
        assert!(CARRIED_PRIME.trunk_capabilities(&dense).batch.supported);
        assert!(DECODE_BATCH.trunk_capabilities(&dense).batch.supported);
        assert!(DECODE_GRAPH.trunk_capabilities(&dense).cuda_graph.supported);

        let gemma = plan(
            r#"{"model_type":"gemma4","num_hidden_layers":2,"hidden_size":64,
            "num_attention_heads":2,"num_key_value_heads":1,"head_dim":32,
            "global_head_dim":32,"intermediate_size":128,"vocab_size":16,
            "max_position_embeddings":128,"sliding_window":64,
            "layer_types":["sliding_attention","full_attention"],
            "rope_parameters":{"full_attention":{"rope_theta":10000},
            "sliding_attention":{"rope_theta":10000}}}"#,
        );
        let capability = CARRIED_PRIME.trunk_capabilities(&gemma).batch;
        assert!(!capability.supported);
        assert!(
            capability
                .blockers
                .contains(&OperationKind::SlidingWindowAttention)
        );
        assert!(capability.blockers.contains(&OperationKind::GemmaResidual));
        assert_eq!(decode_batch_program(&gemma), DecodeBatchProgram::Gemma);
        assert!(!DECODE_GRAPH.trunk_capabilities(&gemma).cuda_graph.supported);

        let mut sliding_gated_moe = plan(
            r#"{"model_type":"qwen3_moe","num_hidden_layers":2,"hidden_size":64,
            "num_attention_heads":2,"num_key_value_heads":1,"head_dim":32,
            "intermediate_size":128,"vocab_size":16,"max_position_embeddings":128,
            "num_experts":4,"num_experts_per_tok":2,"moe_intermediate_size":32}"#,
        );
        let crate::model_plan::AttentionPlan::Full(mut attention) =
            sliding_gated_moe.layers[0].attention.clone()
        else {
            unreachable!()
        };
        attention.output_gate = crate::config::AttentionGateKind::SeparateHead;
        sliding_gated_moe.layers[0].attention = crate::model_plan::AttentionPlan::SlidingWindow {
            attention,
            window: 64,
        };
        let crate::model_plan::MlpPlan::Moe(moe) = &mut sliding_gated_moe.layers[0].mlp else {
            unreachable!()
        };
        moe.router = crate::model_plan::RouterPlan::Sigmoid {
            normalize_selected: true,
            scaling_factor: 1.0,
            selection_bias: false,
        };
        assert_eq!(
            decode_batch_program(&sliding_gated_moe),
            DecodeBatchProgram::SlidingGatedMoe
        );
        assert!(
            !DECODE_GRAPH
                .trunk_capabilities(&sliding_gated_moe)
                .cuda_graph
                .supported
        );
        assert_eq!(decode_batch_program(&dense), DecodeBatchProgram::Generic);
        assert_eq!(
            MTP_SPEC.capabilities(&dense).speculative.blockers,
            vec![OperationKind::DraftPlan]
        );

        let qwen35 = crate::model_packs::by_alias("qwen35")
            .unwrap()
            .compile_tiny_plan()
            .unwrap();
        assert!(MTP_SPEC.capabilities(&qwen35).speculative.supported);

        let dsv4 = crate::model_packs::by_alias("deepseek_v4_dspark")
            .unwrap()
            .compile_tiny_plan()
            .unwrap();
        let dsv4_batch = DECODE_BATCH.trunk_capabilities(&dsv4).batch;
        assert!(!dsv4_batch.supported);
        assert!(
            dsv4_batch
                .blockers
                .contains(&OperationKind::CompressedMlaAttention)
        );
        assert!(!MTP_SPEC.capabilities(&dsv4).speculative.supported);

        let root = std::env::temp_dir().join(format!(
            "memra-plan-backend-external-draft-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let trunk_path = root.join("trunk.gguf");
        let draft_path = root.join("draft.gguf");
        crate::micro_gguf::write_step35_meta_only(&trunk_path).unwrap();
        crate::micro_gguf::write_step35_mtp_meta_only(&draft_path).unwrap();
        let trunk_cfg = ModelConfig::from_gguf(&crate::GgufFile::open(&trunk_path).unwrap());
        let draft_cfg = ModelConfig::from_gguf(&crate::GgufFile::open(&draft_path).unwrap());
        let mut external = crate::model_packs::for_config(&trunk_cfg)
            .unwrap()
            .compile_plan(&trunk_cfg)
            .unwrap();
        let draft = crate::model_packs::for_config(&draft_cfg)
            .unwrap()
            .compile_plan(&draft_cfg)
            .unwrap();
        assert_eq!(
            external.draft_source,
            crate::model_plan::DraftSourcePlan::ExternalArtifact
        );
        external.attach_external_draft(&draft).unwrap();
        assert!(MTP_SPEC.capabilities(&external).speculative.supported);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rewrite_receipts_bind_plan_program_binary_and_outputs() {
        let dense = plan(
            r#"{"model_type":"qwen3","num_hidden_layers":2,"hidden_size":64,
            "num_attention_heads":2,"num_key_value_heads":1,"head_dim":32,
            "intermediate_size":128,"vocab_size":16,"max_position_embeddings":128}"#,
        );
        let rewrites = execution_rewrites(&dense);
        let batch = rewrites
            .iter()
            .find(|rewrite| rewrite.surface == RewriteSurface::DecodeBatch)
            .unwrap();
        assert!(batch.eligible());
        let spec = rewrites
            .iter()
            .find(|rewrite| rewrite.surface == RewriteSurface::MtpSpec)
            .unwrap();
        assert_eq!(spec.blockers, vec![OperationKind::DraftPlan]);

        let policy = RewriteParityPolicy {
            max_abs: 0.01,
            max_rel: 0.01,
            require_argmax: true,
        };
        let implementation = "00".repeat(32);
        let artifact_lock = b"format_version=2\nfamily=qwen3\n";
        let receipt = batch
            .verify_logits(
                &implementation,
                &[0.0, 1.0, -1.0],
                &[0.0, 1.001, -1.001],
                policy,
            )
            .unwrap()
            .bind_artifact_lock(artifact_lock);
        assert!(receipt.passed);
        receipt.validate_for(batch).unwrap();
        let tsv = receipt.to_tsv();
        assert!(tsv.contains("rewrite\tdecode-batch.v1"));
        assert!(tsv.contains(&format!("plan_sha256\t{}", batch.plan_sha256)));
        assert!(tsv.contains(&format!("implementation_sha256\t{implementation}")));

        let root = std::env::temp_dir().join(format!(
            "memra-rewrite-qualification-{}",
            std::process::id()
        ));
        let receipts = root.join("rewrite-receipts");
        std::fs::create_dir_all(&receipts).unwrap();
        std::fs::write(root.join("artifact.lock"), artifact_lock).unwrap();
        std::fs::write(receipts.join("decode-batch.v1.tsv"), &tsv).unwrap();
        std::fs::write(
            root.join("rewrite-receipts.tsv"),
            format!(
                "rewrite\tplan_sha256\treceipt_sha256\tstatus\n\
                 decode-batch.v1\t{}\t{}\tpassed\n",
                batch.plan_sha256,
                hex_sha256(tsv.as_bytes())
            ),
        )
        .unwrap();
        let qualifications = RewriteQualifications::load(&root, &dense).unwrap();
        assert!(qualifications.allows(RewriteSurface::DecodeBatch));
        assert!(!qualifications.allows(RewriteSurface::DecodeGraph));
        std::fs::write(receipts.join("decode-batch.v1.tsv"), "tampered").unwrap();
        assert!(RewriteQualifications::load(&root, &dense).is_err());
        std::fs::remove_dir_all(root).unwrap();

        let failed = batch
            .verify_logits(
                &implementation,
                &[0.0, 1.0, -1.0],
                &[2.0, 1.0, -1.0],
                policy,
            )
            .unwrap();
        assert!(!failed.passed);
        assert!(failed.validate_for(batch).is_err());

        let other = plan(
            r#"{"model_type":"qwen3","num_hidden_layers":3,"hidden_size":64,
            "num_attention_heads":2,"num_key_value_heads":1,"head_dim":32,
            "intermediate_size":128,"vocab_size":16,"max_position_embeddings":128}"#,
        );
        let other_batch = execution_rewrites(&other)
            .into_iter()
            .find(|rewrite| rewrite.surface == RewriteSurface::DecodeBatch)
            .unwrap();
        assert!(receipt.validate_for(&other_batch).is_err());

        let qwen35 = crate::model_packs::by_alias("qwen35")
            .unwrap()
            .compile_tiny_plan()
            .unwrap();
        let qwen35_spec = execution_rewrites(&qwen35)
            .into_iter()
            .find(|rewrite| rewrite.surface == RewriteSurface::MtpSpec)
            .unwrap();
        assert!(qwen35_spec.eligible());
    }
}

/// glm5-spec class matrix (lane/glm5-spec-routing, 2026-08-30): the GLM5_SPEC manifest must
/// declare speculative support for EXACTLY the glm5_next class with an MTP block present, and
/// stay fail-closed everywhere else — including for glm5_next under the MTP_SPEC manifest
/// (the qwen program), so the two spec programs can never both claim one plan.
#[cfg(test)]
mod glm5_spec_class_matrix {
    use super::*;
    use crate::model_plan::OperationKind;

    /// The tparallel-verify / mtp-draft gate fixture family's config (4 trunk layers:
    /// KDA + DSA(MLA+kpool) alternating, dense L0 + sigmoid noaux_tc MoE, mHC streams)
    /// plus `nextn` MTP block(s). The REAL GLM-5.3-Flash text_config carries the same
    /// operation classes (silu+swiglu_limit -> SwiGluPreClamped, sigmoid router, shared
    /// expert, dense first layers, hc residuals, 1 NextN block).
    fn glm5_plan(nextn: u32) -> crate::model_plan::ModelPlan {
        let json = format!(
            r#"{{
          "model_type": "glm5_next_text", "num_hidden_layers": 4,
          "num_nextn_predict_layers": {nextn},
          "hidden_size": 128, "intermediate_size": 64, "vocab_size": 32,
          "max_position_embeddings": 512, "rms_norm_eps": 1e-05, "hidden_act": "silu",
          "swiglu_limit": 10.0, "tie_word_embeddings": true, "hc_mult": 4, "hc_eps": 1e-06,
          "hc_sinkhorn_iters": 20, "mhc": true,
          "layer_types": ["linear_attention", "deepseek_sparse_attention",
                          "linear_attention", "deepseek_sparse_attention"],
          "mlp_layer_types": ["dense", "sparse", "sparse", "sparse"],
          "first_k_dense_replace": 1, "indexer_types": ["full", "full", "full", "full"],
          "linear_attn_config": {{"num_heads": 1, "head_dim": 128, "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0, "kda_layers": [0, 2], "full_attn_layers": [1, 3]}},
          "num_attention_heads": 2, "num_key_value_heads": 2, "q_lora_rank": 16,
          "kv_lora_rank": 16, "qk_head_dim": 16, "qk_nope_head_dim": 16, "qk_rope_head_dim": 0,
          "v_head_dim": 16, "mla_use_nope": true, "index_n_heads": 1, "index_head_dim": 8,
          "index_topk": 8, "index_kpool": 4, "index_kpool_always_select_tail": true,
          "index_kpool_compress": true, "indexer_rope_interleave": true,
          "index_share_for_mtp_iteration": true, "n_routed_experts": 4, "num_experts_per_tok": 2,
          "moe_intermediate_size": 64, "n_shared_experts": 1, "scoring_func": "sigmoid",
          "topk_method": "noaux_tc", "routed_scaling_factor": 2.5, "norm_topk_prob": true,
          "n_group": 1, "topk_group": 1, "head_dim": 0, "attention_bias": false,
          "moe_router_dtype": "float32", "dtype": "bfloat16"
        }}"#
        );
        let config = crate::config::ModelConfig::from_hf(&crate::config::HfConfig::parse(&json));
        crate::model_packs::for_config(&config)
            .unwrap()
            .compile_plan(&config)
            .unwrap()
    }

    #[test]
    fn glm5_with_mtp_block_is_supported_and_the_rewrite_row_is_eligible() {
        let plan = glm5_plan(1);
        let capability = GLM5_SPEC.capabilities(&plan).speculative;
        assert!(
            capability.supported,
            "glm5_next + MTP block must be the supported class; blockers: {:?}",
            capability.blockers
        );
        let rewrite = execution_rewrites(&plan)
            .into_iter()
            .find(|rewrite| rewrite.surface == RewriteSurface::Glm5Spec)
            .unwrap();
        assert!(
            rewrite.eligible(),
            "glm5-spec.v1 must be eligible for this plan"
        );
        // A sealed bundle for this plan now OWES a glm5-spec receipt: the surface is
        // eligible, so `RewriteQualifications::allows(Glm5Spec)` is false until the
        // real-artifact qualification lane banks one — fail-closed for production bundles.
        assert_eq!(rewrite.id, "glm5-spec.v1");
    }

    #[test]
    fn glm5_without_an_mtp_block_fails_closed_on_the_draft_plan() {
        let plan = glm5_plan(0);
        let capability = GLM5_SPEC.capabilities(&plan).speculative;
        assert!(!capability.supported);
        assert_eq!(
            capability.blockers,
            vec![OperationKind::DraftPlan],
            "no NextN block = no draft program; the blocker must name it"
        );
    }

    #[test]
    fn the_two_spec_programs_never_both_claim_one_plan() {
        // glm5_next stays UNSUPPORTED under the qwen-class MTP_SPEC manifest (its hc/KDA/MLA
        // classes are deliberately absent there) — the worker's mtp_spec_capable therefore
        // remains false for glm5 plans and the qwen spec route can never take them.
        let glm5 = glm5_plan(1);
        let mtp_spec = MTP_SPEC.capabilities(&glm5).speculative;
        assert!(!mtp_spec.supported);
        assert!(
            mtp_spec.blockers.contains(&OperationKind::HyperConnections),
            "the hc topology must be among MTP_SPEC's named blockers: {:?}",
            mtp_spec.blockers
        );

        // And the qwen35 class stays UNSUPPORTED under GLM5_SPEC (Full/GDN/KvState/Serial
        // trunk classes absent from its table) — qwen routing is unchanged by this lane.
        let qwen35 = crate::model_packs::by_alias("qwen35")
            .unwrap()
            .compile_tiny_plan()
            .unwrap();
        assert!(MTP_SPEC.capabilities(&qwen35).speculative.supported);
        let qwen_glm5 = GLM5_SPEC.capabilities(&qwen35).speculative;
        assert!(!qwen_glm5.supported);
        assert!(
            qwen_glm5.blockers.contains(&OperationKind::SerialResidual),
            "a serial-residual trunk must block the hc walk program: {:?}",
            qwen_glm5.blockers
        );
    }

    #[test]
    fn foreign_state_classes_fail_closed_under_glm5_spec() {
        // dsv4: CompressedMlaAttention class — its spec program is dspark, never this walk.
        let dsv4 = crate::model_packs::by_alias("deepseek_v4_dspark")
            .unwrap()
            .compile_tiny_plan()
            .unwrap();
        let dsv4_cap = GLM5_SPEC.capabilities(&dsv4).speculative;
        assert!(!dsv4_cap.supported);
        assert!(
            dsv4_cap
                .blockers
                .contains(&OperationKind::CompressedMlaAttention)
        );

        // Dense qwen3: no draft plan AND a foreign trunk class.
        let dense = crate::model_packs::by_alias("qwen3")
            .unwrap()
            .compile_tiny_plan()
            .unwrap();
        assert!(!GLM5_SPEC.capabilities(&dense).speculative.supported);
    }
}
