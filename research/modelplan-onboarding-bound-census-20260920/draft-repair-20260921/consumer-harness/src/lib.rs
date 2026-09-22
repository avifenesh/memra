#![cfg(test)]
use memra_gguf::bound_source::consumer::{FfnActivation, read_nvfp4_macro_scale};
use memra_gguf::bound_source::draft::PreparedExternalDraftSource;
use memra_gguf::config::{HfConfig, ModelConfig, SwigluClamp};
use memra_gguf::model_plan::{ActivationPlan, MlpPlan};
use memra_gguf::source::{GgufSource, TensorSource};
use memra_gguf::{GgmlType, GgufFile};
use std::cell::RefCell;

// Compile the production consumers verbatim. Only the terminal device calls are replaced
// by a recorder; activation selection and macro reads are the actual engine implementation.
#[path = "../../../../../crates/memra-engine/src/ffn_activation.rs"]
mod ffn_activation;
mod fixture;
#[path = "../../../../../crates/memra-engine/src/model/nvfp4_scale.rs"]
mod nvfp4_scale;

type CudaSlice<T> = Vec<T>;
type Result = std::result::Result<(), Box<dyn std::error::Error>>;
#[derive(Debug, PartialEq)]
enum Kernel {
    Silu,
    SiluScaled,
    Oai(f32, f32),
    Post(f32),
    Pre(f32),
}
#[derive(Default)]
struct Engine {
    calls: RefCell<Vec<(Kernel, f32, f32, usize)>>,
}
impl Engine {
    fn record(&self, kernel: Kernel, gs: f32, us: f32, n: usize) -> Result {
        self.calls.borrow_mut().push((kernel, gs, us, n));
        Ok(())
    }
    fn silu_mul(
        &self,
        _: &CudaSlice<f32>,
        _: &CudaSlice<f32>,
        _: &mut CudaSlice<f32>,
        n: usize,
    ) -> Result {
        self.record(Kernel::Silu, 1.0, 1.0, n)
    }
    fn silu_mul_scaled(
        &self,
        _: &CudaSlice<f32>,
        _: &CudaSlice<f32>,
        gs: f32,
        us: f32,
        _: &mut CudaSlice<f32>,
        n: usize,
    ) -> Result {
        self.record(Kernel::SiluScaled, gs, us, n)
    }
    #[allow(clippy::too_many_arguments)] // allow: exact native call signatures
    fn swigluoai_mul_scaled(
        &self,
        _: &CudaSlice<f32>,
        _: &CudaSlice<f32>,
        gs: f32,
        us: f32,
        alpha: f32,
        limit: f32,
        _: &mut CudaSlice<f32>,
        n: usize,
    ) -> Result {
        self.record(Kernel::Oai(alpha, limit), gs, us, n)
    }
    #[allow(clippy::too_many_arguments)] // allow: exact native call signatures
    fn swiglu_clamped_mul_scaled(
        &self,
        _: &CudaSlice<f32>,
        _: &CudaSlice<f32>,
        gs: f32,
        us: f32,
        limit: f32,
        _: &mut CudaSlice<f32>,
        n: usize,
    ) -> Result {
        self.record(Kernel::Post(limit), gs, us, n)
    }
    #[allow(clippy::too_many_arguments)] // allow: exact native call signatures
    fn swiglu_preclamped_mul_scaled(
        &self,
        _: &CudaSlice<f32>,
        _: &CudaSlice<f32>,
        gs: f32,
        us: f32,
        limit: f32,
        _: &mut CudaSlice<f32>,
        n: usize,
    ) -> Result {
        self.record(Kernel::Pre(limit), gs, us, n)
    }
}

fn target(width: u32, oai: Option<(f32, f32)>) -> ModelConfig {
    let extra = match oai {
        Some((alpha, limit)) => format!(
            r#", "model_type":"minimax_m3","num_local_experts":4,
            "num_experts_per_tok":2,"dense_intermediate_size":64,"shared_intermediate_size":64,
            "moe_layer_freq":[0,0],"swiglu_alpha":{alpha},"swiglu_limit":{limit},
            "rotary_dim":16,"use_gemma_norm":false"#
        ),
        None => r#", "model_type":"qwen3""#.into(),
    };
    ModelConfig::from_hf(
        &HfConfig::try_parse(&format!(
            r#"{{"num_hidden_layers":2,
        "hidden_size":{width},"num_attention_heads":2,"num_key_value_heads":1,"head_dim":{},
        "intermediate_size":64,"vocab_size":16,"max_position_embeddings":128,
        "rms_norm_eps":0.000001,"rope_theta":10000.0{extra}}}"#,
            width / 2
        ))
        .unwrap(),
    )
}

#[test]
fn actual_ffn_consumer_selects_exact_program_and_oai_parameters() {
    let gate = vec![2.0];
    let up = vec![3.0];
    let mut output = vec![0.0];
    let cfg = target(32, None);
    let engine = Engine::default();
    ffn_activation::apply(&engine, &cfg, &gate, &up, 1.0, 1.0, None, &mut output, 1).unwrap();
    ffn_activation::apply(&engine, &cfg, &gate, &up, 2.0, 3.0, None, &mut output, 1).unwrap();
    assert_eq!(
        *engine.calls.borrow(),
        [
            (Kernel::Silu, 1.0, 1.0, 1),
            (Kernel::SiluScaled, 2.0, 3.0, 1)
        ]
    );
    for (alpha, limit) in [(1.702, 7.0), (1.5, 7.0), (1.702, 6.0)] {
        let cfg = target(32, Some((alpha, limit)));
        let plan = memra_gguf::model_packs::compile_for_load(&cfg).unwrap();
        let MlpPlan::Dense(dense) = &plan.layers[0].mlp else {
            panic!("expected dense target")
        };
        let dispatch = FfnActivation::for_target(&cfg, None);
        dispatch.validate_declared(&dense.activation).unwrap();
        assert!(dispatch.validate_declared(&ActivationPlan::Silu).is_err());
        assert!(
            dispatch
                .validate_declared(&ActivationPlan::SwiGluOai {
                    alpha: alpha + 0.1,
                    limit
                })
                .is_err()
        );
        assert!(
            dispatch
                .validate_declared(&ActivationPlan::SwiGluOai {
                    alpha,
                    limit: limit + 1.0
                })
                .is_err()
        );
        let engine = Engine::default();
        ffn_activation::apply(&engine, &cfg, &gate, &up, 2.0, 3.0, None, &mut output, 1).unwrap();
        assert_eq!(
            *engine.calls.borrow(),
            [(Kernel::Oai(alpha, limit), 2.0, 3.0, 1)]
        );
    }
}

#[test]
fn actual_ffn_consumer_preserves_step_post_and_distinct_pre_clamps() {
    let cfg = target(32, None);
    let engine = Engine::default();
    for limit in [2.5, 6.5] {
        let declared = ActivationPlan::SwiGluClamped { limit };
        let clamp = memra_gguf::bound_source::consumer::step_mtp_clamp(&declared).unwrap();
        FfnActivation::for_target(&cfg, Some(SwigluClamp::Post(clamp)))
            .validate_declared(&declared)
            .unwrap();
        ffn_activation::apply(
            &engine,
            &cfg,
            &vec![2.0],
            &vec![3.0],
            1.0,
            1.0,
            Some(SwigluClamp::Post(clamp)),
            &mut vec![0.0],
            1,
        )
        .unwrap();
        assert!(
            FfnActivation::for_target(&cfg, Some(SwigluClamp::Pre(clamp)))
                .validate_declared(&declared)
                .is_err()
        );
    }
    ffn_activation::apply(
        &engine,
        &cfg,
        &vec![2.0],
        &vec![3.0],
        1.0,
        1.0,
        Some(SwigluClamp::Pre(2.5)),
        &mut vec![0.0],
        1,
    )
    .unwrap();
    assert_eq!(
        *engine.calls.borrow(),
        [
            (Kernel::Post(2.5), 1.0, 1.0, 1),
            (Kernel::Post(6.5), 1.0, 1.0, 1),
            (Kernel::Pre(2.5), 1.0, 1.0, 1)
        ]
    );
}

#[test]
fn real_draft_refuses_target_program_selected_by_the_actual_consumer() {
    let f = fixture::make(32, false, None, false);
    let file = GgufFile::open(f.0.join("draft.gguf")).unwrap();
    let source = GgufSource(&file);
    let compatible = target(32, None);
    PreparedExternalDraftSource::compile(&source, &compatible).unwrap();
    for parameters in [(1.702, 7.0), (1.5, 7.0), (1.702, 6.0)] {
        let cfg = target(32, Some(parameters));
        let engine = Engine::default();
        ffn_activation::apply(
            &engine,
            &cfg,
            &vec![2.0],
            &vec![3.0],
            1.0,
            1.0,
            None,
            &mut vec![0.0],
            1,
        )
        .unwrap();
        assert_eq!(
            engine.calls.borrow()[0].0,
            Kernel::Oai(parameters.0, parameters.1)
        );
        assert!(PreparedExternalDraftSource::compile(&source, &cfg).is_err());
    }
}

#[test]
fn selected_head_and_student_nvfp4_scales_match_actual_resident_reader() {
    for student in [false, true] {
        let width = if student { 128 } else { 64 };
        let name = if student {
            "blk.2.nextn.out_up.weight"
        } else {
            "blk.2.nextn.shared_head_head.weight"
        };
        for dtype in [GgmlType::F32, GgmlType::F16, GgmlType::BF16] {
            let f = fixture::make(width, student, Some(dtype), false);
            let file = GgufFile::open(f.0.join("draft.gguf")).unwrap();
            let source = GgufSource(&file);
            let prepared = PreparedExternalDraftSource::compile(&source, &target(width, None));
            if dtype != GgmlType::F32 {
                assert!(prepared.is_err(), "student={student}, scale={dtype:?}");
                assert!(nvfp4_scale::load(&source, name).is_err());
                continue;
            }
            let prepared = prepared.unwrap();
            prepared
                .with_runtime(|runtime| {
                    // GgufSource cannot take the direct HF-native import. load_t therefore reaches
                    // load_from_source_inner's NVFP4 resident branch and this production reader.
                    assert!(runtime.try_find_nvfp4_native(name).unwrap().is_none());
                    assert_eq!(
                        runtime.try_find(name).unwrap().unwrap().ggml_type,
                        GgmlType::NVFP4
                    );
                    assert_eq!(nvfp4_scale::load(runtime, name).unwrap(), 2.0);
                    assert_eq!(
                        nvfp4_scale::load(runtime, "blk.2.nextn.eh_proj.weight").unwrap(),
                        1.0
                    );
                })
                .unwrap();
            assert_eq!(
                prepared.artifact_identity().unwrap().opened_source_sha256,
                source.artifact_sha256().unwrap()
            );
        }
    }
}

#[test]
fn unselected_scale_bytes_stay_inventoried_and_identity_bound() {
    let f = fixture::make(64, false, Some(GgmlType::F32), true);
    let file = GgufFile::open(f.0.join("draft.gguf")).unwrap();
    let source = GgufSource(&file);
    let prepared = PreparedExternalDraftSource::compile(&source, &target(64, None)).unwrap();
    let identity = prepared.artifact_identity().unwrap();
    assert_eq!(
        source.find("output.scale").unwrap().ggml_type,
        GgmlType::BF16
    );
    prepared
        .with_runtime(|runtime| {
            assert!(runtime.try_find("output.scale").is_err());
            assert_eq!(
                nvfp4_scale::load(runtime, prepared.head_name()).unwrap(),
                2.0
            );
        })
        .unwrap();
    std::fs::rename(f.0.join("draft.gguf"), f.0.join("opened.gguf")).unwrap();
    std::fs::write(f.0.join("draft.gguf"), b"replacement").unwrap();
    assert_eq!(prepared.artifact_identity().unwrap(), identity);
}

#[test]
fn actual_macro_decoder_checks_encoding_and_width_without_reinterpreting_bytes() {
    use memra_gguf::source::TensorView;
    use std::borrow::Cow;
    let bytes = 2.0f32.to_le_bytes();
    let mut view = TensorView {
        bytes: Cow::Borrowed(&bytes),
        ne: vec![],
        ggml_type: GgmlType::F32,
    };
    assert_eq!(read_nvfp4_macro_scale(&view).unwrap(), 2.0); // existing scalar HF view
    view.bytes = Cow::Borrowed(&bytes[..2]);
    assert!(read_nvfp4_macro_scale(&view).is_err());
    for dtype in [GgmlType::F16, GgmlType::BF16] {
        view.ggml_type = dtype;
        assert!(read_nvfp4_macro_scale(&view).is_err());
    }
}
