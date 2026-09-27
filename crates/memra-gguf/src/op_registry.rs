//! Operation-implementation registry — the single place where the engine declares, per plan
//! operation, which tuned execution surfaces have an implementation.
//!
//! The execution manifests in [`crate::execution_manifest`] (`CARRIED_PRIME`, `NATIVE_EAGER`,
//! `NATIVE_GEMMA_EAGER`, `NATIVE_GDN_EAGER`, `DECODE_BATCH`, `DECODE_GRAPH`, `MTP_SPEC`, `GLM5_SPEC`, `PIPELINE`) DERIVE their per-operation
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
    /// `NATIVE_GEMMA_EAGER`: the canonical dense, non-PLE Gemma T1 program.
    /// This is a separate implementation of DecodeEager, not fresh-KV coverage.
    pub gemma_eager: bool,
    /// `NATIVE_GDN_EAGER`: dense serial GatedDeltaNet/full-attention T1 program.
    pub gdn_eager: bool,
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
    /// `CHUNKED_PRIME`: the GENERIC `prime_cache` chunked, continuation-capable prime program
    /// (`prime_chunk_ranges` splits inside a call; `cache.pos > 0` resumes across calls) is
    /// implemented for this operation AND its chunk/tick invariance is gated at model scale
    /// (`tools/chunk-invariance-gate.sh`, `concat-prime-probe tickinv`). Operations with their
    /// own prime program (HyperConnections via `prime_cache_hyper`, PLE via `gemma4_e4b_prime`)
    /// say no here, and the driver primes them monolithically. A row flips only on the model-
    /// scale receipt: `GemmaParallelMoeResidual` was MEASURED chunk-dependent first (memra#562:
    /// the gemma MoE arm routed prefill through the m-dependent cuBLAS matmul, prefill logits
    /// moved O(1) with the chunk size, first divergence at row 0) and became yes when the router
    /// moved to `router_gemv` and the 26B read EXACT on chunkinv and tickinv. Consulted today
    /// only inside the gemma family; widening it to a global predicate needs the per-op
    /// receipts first.
    pub chunked_prime: bool,
}

impl Surfaces {
    /// No surface implemented. Every row starts here and adds what the engine has.
    pub const NONE: Self = Self {
        carried_prime: false,
        decode_eager: false,
        gemma_eager: false,
        gdn_eager: false,
        decode_batch: false,
        decode_graph: false,
        mtp_spec_draft: false,
        mtp_spec_verify: false,
        glm5_spec_draft: false,
        glm5_spec_verify: false,
        pipeline: false,
        chunked_prime: false,
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
    pub const fn gemma_eager(self) -> Self {
        Self {
            gemma_eager: true,
            ..self
        }
    }
    pub const fn gdn_eager(self) -> Self {
        Self {
            gdn_eager: true,
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
    pub const fn chunked_prime(self) -> Self {
        Self {
            chunked_prime: true,
            ..self
        }
    }

    /// True when no surface is implemented for the operation.
    pub const fn is_none(self) -> bool {
        !(self.carried_prime
            || self.decode_eager
            || self.gemma_eager
            || self.gdn_eager
            || self.decode_batch
            || self.decode_graph
            || self.mtp_spec_draft
            || self.mtp_spec_verify
            || self.glm5_spec_draft
            || self.glm5_spec_verify
            || self.pipeline
            || self.chunked_prime)
    }

    /// Column order of the rendered table and of [`Surfaces::flags`].
    pub const COLUMNS: [&'static str; 12] = [
        "carried_prime",
        "decode_eager",
        "decode_batch",
        "decode_graph",
        "mtp_spec_draft",
        "mtp_spec_verify",
        "glm5_spec_draft",
        "glm5_spec_verify",
        "pipeline",
        "chunked_prime",
        "gemma_eager",
        "gdn_eager",
    ];

    /// The row as booleans in [`Surfaces::COLUMNS`] order.
    pub const fn flags(self) -> [bool; 12] {
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
            self.chunked_prime,
            self.gemma_eager,
            self.gdn_eager,
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
            .gdn_eager()
            .gemma_eager()
            .carried_prime()
            .decode_eager()
            .decode_batch()
            .decode_graph()
            .mtp_spec_draft()
            .mtp_spec_verify()
            .glm5_spec_draft()
            .glm5_spec_verify()
            .pipeline()
            .chunked_prime(),
        OperationKind::RmsNorm => Surfaces::NONE
            .gdn_eager()
            .gemma_eager()
            .carried_prime()
            .decode_eager()
            .decode_batch()
            .decode_graph()
            .mtp_spec_draft()
            .mtp_spec_verify()
            .glm5_spec_draft()
            .glm5_spec_verify()
            .pipeline()
            .chunked_prime(),
        OperationKind::FullAttention => Surfaces::NONE
            .gdn_eager()
            .gemma_eager()
            .carried_prime()
            .decode_eager()
            .decode_batch()
            .decode_graph()
            .mtp_spec_draft()
            .mtp_spec_verify()
            .pipeline()
            .chunked_prime(),
        OperationKind::SlidingWindowAttention => Surfaces::NONE
            .gemma_eager()
            .decode_batch()
            .mtp_spec_draft()
            .mtp_spec_verify()
            .pipeline()
            .chunked_prime(),
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
            .gdn_eager()
            .carried_prime()
            .decode_batch()
            .decode_graph()
            .mtp_spec_draft()
            .mtp_spec_verify()
            .chunked_prime(),
        OperationKind::KimiDeltaNet => Surfaces::NONE
            .glm5_spec_draft()
            .glm5_spec_verify()
            .pipeline(),
        OperationKind::FusedAttentionGate => Surfaces::NONE
            .gdn_eager()
            .carried_prime()
            .decode_batch()
            .decode_graph()
            .mtp_spec_draft()
            .mtp_spec_verify()
            .chunked_prime(),
        OperationKind::SeparateAttentionGate => Surfaces::NONE
            .decode_batch()
            .mtp_spec_draft()
            .mtp_spec_verify()
            .pipeline()
            .chunked_prime(),
        OperationKind::DenseMlp => Surfaces::NONE
            .gdn_eager()
            .gemma_eager()
            .carried_prime()
            .decode_eager()
            .decode_batch()
            .decode_graph()
            .mtp_spec_draft()
            .mtp_spec_verify()
            .glm5_spec_verify()
            .pipeline()
            .chunked_prime(),
        OperationKind::MoeMlp => Surfaces::NONE
            .decode_batch()
            .decode_graph()
            .mtp_spec_draft()
            .mtp_spec_verify()
            .glm5_spec_draft()
            .glm5_spec_verify()
            .pipeline()
            .chunked_prime(),
        OperationKind::RetainedExpertRouting => Surfaces::NONE,
        OperationKind::SoftmaxRouter => Surfaces::NONE
            .decode_batch()
            .decode_graph()
            .mtp_spec_draft()
            .mtp_spec_verify()
            .chunked_prime(),
        OperationKind::SigmoidRouter => Surfaces::NONE
            .decode_batch()
            .decode_graph()
            .mtp_spec_draft()
            .mtp_spec_verify()
            .glm5_spec_draft()
            .glm5_spec_verify()
            .pipeline()
            .chunked_prime(),
        OperationKind::SharedMlp => Surfaces::NONE
            .decode_batch()
            .decode_graph()
            .mtp_spec_draft()
            .mtp_spec_verify()
            .glm5_spec_draft()
            .glm5_spec_verify()
            .pipeline()
            .chunked_prime(),
        OperationKind::SiluActivation => Surfaces::NONE
            .gdn_eager()
            .carried_prime()
            .decode_eager()
            .decode_batch()
            .decode_graph()
            .mtp_spec_draft()
            .mtp_spec_verify()
            .pipeline()
            .chunked_prime(),
        OperationKind::GeluTanhActivation => {
            Surfaces::NONE.gemma_eager().decode_batch().chunked_prime()
        }
        OperationKind::SwiGluClampedActivation => Surfaces::NONE
            .decode_batch()
            .mtp_spec_draft()
            .mtp_spec_verify()
            .pipeline()
            .chunked_prime(),
        OperationKind::SwiGluPreClampedActivation => Surfaces::NONE
            .glm5_spec_draft()
            .glm5_spec_verify()
            .pipeline(),
        OperationKind::SerialResidual => Surfaces::NONE
            .gdn_eager()
            .carried_prime()
            .decode_eager()
            .decode_batch()
            .decode_graph()
            .mtp_spec_draft()
            .mtp_spec_verify()
            .glm5_spec_draft()
            .pipeline()
            .chunked_prime(),
        OperationKind::GemmaResidual => Surfaces::NONE.gemma_eager().decode_batch().chunked_prime(),
        OperationKind::GemmaParallelMoeResidual => Surfaces::NONE.decode_batch().chunked_prime(),
        OperationKind::HyperConnections => Surfaces::NONE.glm5_spec_verify().pipeline(),
        OperationKind::KvState => Surfaces::NONE
            .gdn_eager()
            .gemma_eager()
            .carried_prime()
            .decode_eager()
            .decode_batch()
            .decode_graph()
            .mtp_spec_draft()
            .mtp_spec_verify()
            .pipeline()
            .chunked_prime(),
        OperationKind::SlidingKvState => Surfaces::NONE
            .gemma_eager()
            .decode_batch()
            .mtp_spec_draft()
            .mtp_spec_verify()
            .pipeline()
            .chunked_prime(),
        OperationKind::RecurrentState => Surfaces::NONE
            .gdn_eager()
            .carried_prime()
            .decode_batch()
            .decode_graph()
            .mtp_spec_draft()
            .mtp_spec_verify()
            .glm5_spec_draft()
            .glm5_spec_verify()
            .pipeline()
            .chunked_prime(),
        OperationKind::LatentKvState => Surfaces::NONE
            .glm5_spec_draft()
            .glm5_spec_verify()
            .pipeline(),
        OperationKind::Mtp => Surfaces::NONE.mtp_spec_draft().glm5_spec_draft().pipeline(),
        OperationKind::MtpFusion => Surfaces::NONE.mtp_spec_draft().glm5_spec_draft().pipeline(),
        OperationKind::MtpHead => Surfaces::NONE.mtp_spec_draft().glm5_spec_draft().pipeline(),
        OperationKind::PipelineBoundary => Surfaces::NONE.pipeline(),
        OperationKind::LogitsSoftcap => Surfaces::NONE
            .gemma_eager()
            .decode_eager()
            .decode_batch()
            .chunked_prime(),
        OperationKind::LogitsMask => Surfaces::NONE
            .gemma_eager()
            .carried_prime()
            .decode_eager()
            .decode_batch()
            .decode_graph()
            .mtp_spec_draft()
            .mtp_spec_verify()
            .pipeline()
            .chunked_prime(),
        OperationKind::OutputProjection => Surfaces::NONE
            .gdn_eager()
            .gemma_eager()
            .carried_prime()
            .decode_eager()
            .decode_batch()
            .decode_graph()
            .mtp_spec_draft()
            .mtp_spec_verify()
            .glm5_spec_draft()
            .glm5_spec_verify()
            .pipeline()
            .chunked_prime(),
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
    OperationKind::RetainedExpertRouting,
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
            68,
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
    fn gemma_eager_declares_only_the_dense_t1_operations() {
        let expected = [
            OperationKind::Embedding,
            OperationKind::RmsNorm,
            OperationKind::FullAttention,
            OperationKind::SlidingWindowAttention,
            OperationKind::DenseMlp,
            OperationKind::GeluTanhActivation,
            OperationKind::GemmaResidual,
            OperationKind::KvState,
            OperationKind::SlidingKvState,
            OperationKind::LogitsSoftcap,
            OperationKind::LogitsMask,
            OperationKind::OutputProjection,
        ];
        for &operation in ALL_OPERATIONS {
            assert_eq!(
                surfaces(operation).gemma_eager,
                expected.contains(&operation),
                "{operation:?}"
            );
        }
        // No global change to the generic eager/fresh-KV support column.
        for operation in [
            OperationKind::SlidingWindowAttention,
            OperationKind::SlidingKvState,
            OperationKind::GeluTanhActivation,
            OperationKind::GemmaResidual,
        ] {
            assert!(!surfaces(operation).decode_eager);
        }
    }

    #[test]
    fn gdn_eager_declares_only_the_dense_serial_t1_operations() {
        let expected = [
            OperationKind::Embedding,
            OperationKind::RmsNorm,
            OperationKind::FullAttention,
            OperationKind::GatedDeltaNet,
            OperationKind::FusedAttentionGate,
            OperationKind::DenseMlp,
            OperationKind::SiluActivation,
            OperationKind::SerialResidual,
            OperationKind::KvState,
            OperationKind::RecurrentState,
            OperationKind::OutputProjection,
        ];
        for &operation in ALL_OPERATIONS {
            assert_eq!(
                surfaces(operation).gdn_eager,
                expected.contains(&operation),
                "{operation:?}"
            );
        }
        for operation in [
            OperationKind::GatedDeltaNet,
            OperationKind::RecurrentState,
            OperationKind::FusedAttentionGate,
        ] {
            assert!(
                !surfaces(operation).decode_eager,
                "generic eager/fresh-KV widened"
            );
            assert!(!surfaces(operation).gemma_eager, "Gemma eager widened");
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
