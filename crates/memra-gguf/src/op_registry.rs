//! Operation-implementation registry — the single place where the engine declares, per plan
//! operation, which tuned execution surfaces have an implementation.
//!
//! The execution manifests in [`crate::execution_manifest`] (`CARRIED_PRIME`, `NATIVE_EAGER`,
//! `DECODE_BATCH`, `DECODE_GRAPH`, `MTP_SPEC`, `GLM5_SPEC`, `PIPELINE`) DERIVE their per-operation
//! support from this table. Before this module existed each manifest carried its own
//! `matches!(operation, …)` allowlist, so the question "what can the engine do with operation X"
//! had seven answers in seven places and no place said "nothing yet" out loud. Here it has one
//! answer, per operation, and an operation with no implemented surface is an explicit row.
//!
//! Contract (memra#535, P0):
//! - One arm per `OperationKind`. The match is exhaustive, so adding a plan operation without
//!   declaring its row is a compile error — fail-closed AND visible, instead of fail-closed and
//!   silent.
//! - A surface is declared here only when the engine implements it for that operation and the
//!   implementation is on the gated program the manifest names. Declaring a surface is a claim the
//!   manifest's receipts have to back; it is not a wish list.
//! - Dedicated per-family arms that bypass the canonical programs (the `HyperConnections`
//!   batched-decode walk `decode_step_batch_hyper`, `prime_cache_hyper`, the glm5 TP walk, the
//!   dsv4 serial route) are NOT rows here until they implement the canonical surface contract;
//!   `execution_manifest::decode_batch_unconverted` keeps naming that gap until then. When a
//!   later phase moves such an arm behind the shared contract, its row flips here and the
//!   dedicated-arm predicate is deleted — one edit, one review, one receipt.
//! - `docs/EXECUTION-SURFACES.md` is rendered from this table (`render_table`) and a test pins
//!   it, the same way the perf boards pin their generated blocks: a registry change that does
//!   not regenerate the doc fails the build.

use crate::model_plan::OperationKind;

/// The tuned execution surfaces an operation may implement. Each field is one column of the
/// manifest table; the names match `execution_manifest::RewriteSurface` where a surface has a
/// rewrite receipt, with the two speculative programs split into their draft and verify halves
/// because the manifests gate those separately (`OperationSupport::{spec_draft, spec_verify}`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Surfaces {
    /// `CARRIED_PRIME`: the batched cross-request continuation prime (`prime_cache_batch`).
    pub carried_prime: bool,
    /// `NATIVE_EAGER`: the native eager decode program.
    pub decode_eager: bool,
    /// `DECODE_BATCH`: the batched decode trunk (`decode_step_batch*`).
    pub decode_batch: bool,
    /// `DECODE_GRAPH`: CUDA-graph captured decode.
    pub decode_graph: bool,
    /// `MTP_SPEC`, draft side (the MTP/NextN draft head program).
    pub mtp_spec_draft: bool,
    /// `MTP_SPEC`, verify side (the trunk walked at t=K+1).
    pub mtp_spec_verify: bool,
    /// `GLM5_SPEC`, draft side (embedded MLA-mixer MTP block).
    pub glm5_spec_draft: bool,
    /// `GLM5_SPEC`, verify side (one t=K+1 walk over the HyperConnections trunk).
    pub glm5_spec_verify: bool,
    /// `PIPELINE`: stage-split trunk with boundary state transport.
    pub pipeline: bool,
}

impl Surfaces {
    /// No surface implemented. Every row starts here and adds what the engine has.
    pub const NONE: Self = Self {
        carried_prime: false,
        decode_eager: false,
        decode_batch: false,
        decode_graph: false,
        mtp_spec_draft: false,
        mtp_spec_verify: false,
        glm5_spec_draft: false,
        glm5_spec_verify: false,
        pipeline: false,
    };

    pub const fn carried_prime(self) -> Self {
        Self {
            carried_prime: true,
            ..self
        }
    }
    pub const fn decode_eager(self) -> Self {
        Self {
            decode_eager: true,
            ..self
        }
    }
    pub const fn decode_batch(self) -> Self {
        Self {
            decode_batch: true,
            ..self
        }
    }
    pub const fn decode_graph(self) -> Self {
        Self {
            decode_graph: true,
            ..self
        }
    }
    pub const fn mtp_spec_draft(self) -> Self {
        Self {
            mtp_spec_draft: true,
            ..self
        }
    }
    pub const fn mtp_spec_verify(self) -> Self {
        Self {
            mtp_spec_verify: true,
            ..self
        }
    }
    pub const fn glm5_spec_draft(self) -> Self {
        Self {
            glm5_spec_draft: true,
            ..self
        }
    }
    pub const fn glm5_spec_verify(self) -> Self {
        Self {
            glm5_spec_verify: true,
            ..self
        }
    }
    pub const fn pipeline(self) -> Self {
        Self {
            pipeline: true,
            ..self
        }
    }

    /// True when no surface is implemented for the operation.
    pub const fn is_none(self) -> bool {
        !(self.carried_prime
            || self.decode_eager
            || self.decode_batch
            || self.decode_graph
            || self.mtp_spec_draft
            || self.mtp_spec_verify
            || self.glm5_spec_draft
            || self.glm5_spec_verify
            || self.pipeline)
    }

    /// Column order of the rendered table and of [`Surfaces::flags`].
    pub const COLUMNS: [&'static str; 9] = [
        "carried_prime",
        "decode_eager",
        "decode_batch",
        "decode_graph",
        "mtp_spec_draft",
        "mtp_spec_verify",
        "glm5_spec_draft",
        "glm5_spec_verify",
        "pipeline",
    ];

    /// The row as booleans in [`Surfaces::COLUMNS`] order.
    pub const fn flags(self) -> [bool; 9] {
        [
            self.carried_prime,
            self.decode_eager,
            self.decode_batch,
            self.decode_graph,
            self.mtp_spec_draft,
            self.mtp_spec_verify,
            self.glm5_spec_draft,
            self.glm5_spec_verify,
            self.pipeline,
        ]
    }
}

/// Which tuned surfaces the engine implements for `operation`. One arm per operation; exhaustive.
///
/// Rows with no surface are grouped in the last arm ON PURPOSE: the group is the visible list of
/// operations every tuned program refuses today (audio and vision front ends, the dsv4 compressed
/// MLA class, qwen4_exp gated residuals, dspark heads, …). Moving an operation out of that group
/// is the whole of "the engine now supports X on surface Y" and must arrive with the receipt.
pub const fn surfaces(operation: OperationKind) -> Surfaces {
    match operation {
        OperationKind::Embedding => Surfaces::NONE
            .carried_prime()
            .decode_eager()
            .decode_batch()
            .decode_graph()
            .mtp_spec_draft()
            .mtp_spec_verify()
            .glm5_spec_draft()
            .glm5_spec_verify()
            .pipeline(),
        OperationKind::RmsNorm => Surfaces::NONE
            .carried_prime()
            .decode_eager()
            .decode_batch()
            .decode_graph()
            .mtp_spec_draft()
            .mtp_spec_verify()
            .glm5_spec_draft()
            .glm5_spec_verify()
            .pipeline(),
        OperationKind::FullAttention => Surfaces::NONE
            .carried_prime()
            .decode_eager()
            .decode_batch()
            .decode_graph()
            .mtp_spec_draft()
            .mtp_spec_verify()
            .pipeline(),
        OperationKind::SlidingWindowAttention => Surfaces::NONE
            .decode_batch()
            .mtp_spec_draft()
            .mtp_spec_verify()
            .pipeline(),
        OperationKind::LatentMlaAttention => Surfaces::NONE
            .glm5_spec_draft()
            .glm5_spec_verify()
            .pipeline(),
        OperationKind::SparseIndex => Surfaces::NONE
            .glm5_spec_draft()
            .glm5_spec_verify()
            .pipeline(),
        OperationKind::SharedSparseIndex => Surfaces::NONE.glm5_spec_draft().glm5_spec_verify(),
        OperationKind::GatedDeltaNet => Surfaces::NONE
            .carried_prime()
            .decode_batch()
            .decode_graph()
            .mtp_spec_draft()
            .mtp_spec_verify(),
        OperationKind::KimiDeltaNet => Surfaces::NONE
            .glm5_spec_draft()
            .glm5_spec_verify()
            .pipeline(),
        OperationKind::FusedAttentionGate => Surfaces::NONE
            .carried_prime()
            .decode_batch()
            .decode_graph()
            .mtp_spec_draft()
            .mtp_spec_verify(),
        OperationKind::SeparateAttentionGate => Surfaces::NONE
            .decode_batch()
            .mtp_spec_draft()
            .mtp_spec_verify()
            .pipeline(),
        OperationKind::DenseMlp => Surfaces::NONE
            .carried_prime()
            .decode_eager()
            .decode_batch()
            .decode_graph()
            .mtp_spec_draft()
            .mtp_spec_verify()
            .glm5_spec_verify()
            .pipeline(),
        OperationKind::MoeMlp => Surfaces::NONE
            .decode_batch()
            .decode_graph()
            .mtp_spec_draft()
            .mtp_spec_verify()
            .glm5_spec_draft()
            .glm5_spec_verify()
            .pipeline(),
        OperationKind::SoftmaxRouter => Surfaces::NONE
            .decode_batch()
            .decode_graph()
            .mtp_spec_draft()
            .mtp_spec_verify(),
        OperationKind::SigmoidRouter => Surfaces::NONE
            .decode_batch()
            .decode_graph()
            .mtp_spec_draft()
            .mtp_spec_verify()
            .glm5_spec_draft()
            .glm5_spec_verify()
            .pipeline(),
        OperationKind::SharedMlp => Surfaces::NONE
            .decode_batch()
            .decode_graph()
            .mtp_spec_draft()
            .mtp_spec_verify()
            .glm5_spec_draft()
            .glm5_spec_verify()
            .pipeline(),
        OperationKind::SiluActivation => Surfaces::NONE
            .carried_prime()
            .decode_eager()
            .decode_batch()
            .decode_graph()
            .mtp_spec_draft()
            .mtp_spec_verify()
            .pipeline(),
        OperationKind::GeluTanhActivation => Surfaces::NONE.decode_batch(),
        OperationKind::SwiGluClampedActivation => Surfaces::NONE
            .decode_batch()
            .mtp_spec_draft()
            .mtp_spec_verify()
            .pipeline(),
        OperationKind::SwiGluPreClampedActivation => Surfaces::NONE
            .glm5_spec_draft()
            .glm5_spec_verify()
            .pipeline(),
        OperationKind::SerialResidual => Surfaces::NONE
            .carried_prime()
            .decode_eager()
            .decode_batch()
            .decode_graph()
            .mtp_spec_draft()
            .mtp_spec_verify()
            .glm5_spec_draft()
            .pipeline(),
        OperationKind::GemmaResidual => Surfaces::NONE.decode_batch(),
        OperationKind::GemmaParallelMoeResidual => Surfaces::NONE.decode_batch(),
        OperationKind::HyperConnections => Surfaces::NONE.glm5_spec_verify().pipeline(),
        OperationKind::KvState => Surfaces::NONE
            .carried_prime()
            .decode_eager()
            .decode_batch()
            .decode_graph()
            .mtp_spec_draft()
            .mtp_spec_verify()
            .pipeline(),
        OperationKind::SlidingKvState => Surfaces::NONE
            .decode_batch()
            .mtp_spec_draft()
            .mtp_spec_verify()
            .pipeline(),
        OperationKind::RecurrentState => Surfaces::NONE
            .carried_prime()
            .decode_batch()
            .decode_graph()
            .mtp_spec_draft()
            .mtp_spec_verify()
            .glm5_spec_draft()
            .glm5_spec_verify()
            .pipeline(),
        OperationKind::LatentKvState => Surfaces::NONE
            .glm5_spec_draft()
            .glm5_spec_verify()
            .pipeline(),
        OperationKind::Mtp => Surfaces::NONE.mtp_spec_draft().glm5_spec_draft().pipeline(),
        OperationKind::MtpFusion => Surfaces::NONE.mtp_spec_draft().glm5_spec_draft().pipeline(),
        OperationKind::MtpHead => Surfaces::NONE.mtp_spec_draft().glm5_spec_draft().pipeline(),
        OperationKind::PipelineBoundary => Surfaces::NONE.pipeline(),
        OperationKind::LogitsSoftcap => Surfaces::NONE.decode_eager().decode_batch(),
        OperationKind::LogitsMask => Surfaces::NONE
            .carried_prime()
            .decode_eager()
            .decode_batch()
            .decode_graph()
            .mtp_spec_draft()
            .mtp_spec_verify()
            .pipeline(),
        OperationKind::OutputProjection => Surfaces::NONE
            .carried_prime()
            .decode_eager()
            .decode_batch()
            .decode_graph()
            .mtp_spec_draft()
            .mtp_spec_verify()
            .glm5_spec_draft()
            .glm5_spec_verify()
            .pipeline(),
        // ---- no tuned surface implemented (reference executor only) ----
        OperationKind::AudioLogMel
        | OperationKind::AudioStridedConv
        | OperationKind::AudioPositionEmbedding
        | OperationKind::BiasedLayerNorm
        | OperationKind::GeluErfActivation
        | OperationKind::AudioEncoderAttention
        | OperationKind::AudioDecoderSelfAttention
        | OperationKind::AudioCrossAttention
        | OperationKind::AudioCrossKvState
        | OperationKind::AsrDeterministicDecode
        | OperationKind::VisionPatchEmbedding
        | OperationKind::VisionBidirectionalAttention
        | OperationKind::VisionMlp
        | OperationKind::VisionStandardize
        | OperationKind::VisionDownsample
        | OperationKind::VisionProjection
        | OperationKind::VisionTokenInjection
        | OperationKind::CompressedMlaAttention
        | OperationKind::KvCompressor
        | OperationKind::MicroBlockSparseIndex
        | OperationKind::SqrtSoftplusRouter
        | OperationKind::TokenHashRouter
        | OperationKind::SwiGluOaiActivation
        | OperationKind::NamedActivation
        | OperationKind::GatedResidualConnections
        | OperationKind::GatedResidualMixer
        | OperationKind::PleNgramEmbedding
        | OperationKind::CompressedAttentionState
        | OperationKind::DraftPlan
        | OperationKind::DsparkFusion
        | OperationKind::DsparkMarkovHead
        | OperationKind::DsparkConfidenceHead => Surfaces::NONE,
    }
}

/// Every `OperationKind`, in declaration order, for table rendering and coverage tests. Kept by
/// hand because the enum has no iteration derive; `surfaces` is exhaustive regardless, so a
/// variant missing here still gets a row in the manifest — only the rendered doc would lag, and
/// `registry_lists_every_operation` catches that.
pub const ALL_OPERATIONS: &[OperationKind] = &[
    OperationKind::AudioLogMel,
    OperationKind::AudioStridedConv,
    OperationKind::AudioPositionEmbedding,
    OperationKind::BiasedLayerNorm,
    OperationKind::GeluErfActivation,
    OperationKind::AudioEncoderAttention,
    OperationKind::AudioDecoderSelfAttention,
    OperationKind::AudioCrossAttention,
    OperationKind::AudioCrossKvState,
    OperationKind::AsrDeterministicDecode,
    OperationKind::Embedding,
    OperationKind::VisionPatchEmbedding,
    OperationKind::VisionBidirectionalAttention,
    OperationKind::VisionMlp,
    OperationKind::VisionStandardize,
    OperationKind::VisionDownsample,
    OperationKind::VisionProjection,
    OperationKind::VisionTokenInjection,
    OperationKind::RmsNorm,
    OperationKind::FullAttention,
    OperationKind::SlidingWindowAttention,
    OperationKind::LatentMlaAttention,
    OperationKind::CompressedMlaAttention,
    OperationKind::KvCompressor,
    OperationKind::SparseIndex,
    OperationKind::SharedSparseIndex,
    OperationKind::MicroBlockSparseIndex,
    OperationKind::GatedDeltaNet,
    OperationKind::KimiDeltaNet,
    OperationKind::FusedAttentionGate,
    OperationKind::SeparateAttentionGate,
    OperationKind::DenseMlp,
    OperationKind::MoeMlp,
    OperationKind::SoftmaxRouter,
    OperationKind::SigmoidRouter,
    OperationKind::SqrtSoftplusRouter,
    OperationKind::TokenHashRouter,
    OperationKind::SharedMlp,
    OperationKind::SiluActivation,
    OperationKind::GeluTanhActivation,
    OperationKind::SwiGluOaiActivation,
    OperationKind::SwiGluClampedActivation,
    OperationKind::SwiGluPreClampedActivation,
    OperationKind::NamedActivation,
    OperationKind::SerialResidual,
    OperationKind::GemmaResidual,
    OperationKind::GemmaParallelMoeResidual,
    OperationKind::HyperConnections,
    OperationKind::GatedResidualConnections,
    OperationKind::GatedResidualMixer,
    OperationKind::PleNgramEmbedding,
    OperationKind::KvState,
    OperationKind::SlidingKvState,
    OperationKind::RecurrentState,
    OperationKind::LatentKvState,
    OperationKind::CompressedAttentionState,
    OperationKind::Mtp,
    OperationKind::DraftPlan,
    OperationKind::MtpFusion,
    OperationKind::MtpHead,
    OperationKind::DsparkFusion,
    OperationKind::DsparkMarkovHead,
    OperationKind::DsparkConfidenceHead,
    OperationKind::PipelineBoundary,
    OperationKind::LogitsSoftcap,
    OperationKind::LogitsMask,
    OperationKind::OutputProjection,
];

/// Render the operation x surface table as Markdown. `docs/EXECUTION-SURFACES.md` carries this
/// between its `EXEC-SURFACES` markers; a test pins the two together.
pub fn render_table() -> String {
    let mut out = String::new();
    out.push_str("| operation |");
    for column in Surfaces::COLUMNS {
        out.push(' ');
        out.push_str(column);
        out.push_str(" |");
    }
    out.push('\n');
    out.push_str("|---|");
    for _ in Surfaces::COLUMNS {
        out.push_str(":-:|");
    }
    out.push('\n');
    for &operation in ALL_OPERATIONS {
        let row = surfaces(operation);
        out.push_str(&format!("| `{operation:?}` |"));
        for flag in row.flags() {
            out.push_str(if flag { " yes |" } else { " — |" });
        }
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn registry_lists_every_operation_once() {
        let unique: HashSet<_> = ALL_OPERATIONS.iter().copied().collect();
        assert_eq!(
            unique.len(),
            ALL_OPERATIONS.len(),
            "duplicate in ALL_OPERATIONS"
        );
        // The count is pinned so that adding an enum variant (which the exhaustive `surfaces`
        // match already forces you to classify) also forces this list and the rendered doc to
        // follow. Bump it in the same commit as the variant.
        assert_eq!(
            ALL_OPERATIONS.len(),
            67,
            "OperationKind variant count moved; update ALL_OPERATIONS"
        );
    }

    #[test]
    fn none_rows_are_really_none() {
        for &operation in ALL_OPERATIONS {
            let row = surfaces(operation);
            if row.is_none() {
                assert_eq!(row, Surfaces::NONE);
            }
        }
    }

    #[test]
    fn rendered_table_matches_docs_execution_surfaces() {
        let doc = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../docs/EXECUTION-SURFACES.md"
        ))
        .expect("docs/EXECUTION-SURFACES.md exists");
        let start = doc
            .find("<!-- EXEC-SURFACES:START -->")
            .expect("start marker");
        let end = doc.find("<!-- EXEC-SURFACES:END -->").expect("end marker");
        let pinned = doc[start + "<!-- EXEC-SURFACES:START -->".len()..end].trim();
        let rendered = render_table();
        assert_eq!(
            pinned,
            rendered.trim(),
            "docs/EXECUTION-SURFACES.md is stale: paste the output of \
             `cargo test -p memra-gguf -- --ignored print_execution_surfaces_table --nocapture` \
             between the EXEC-SURFACES markers"
        );
    }

    /// Not a check: prints the table for pasting into the doc. Run explicitly.
    #[test]
    #[ignore]
    fn print_execution_surfaces_table() {
        println!("{}", render_table());
    }
}
