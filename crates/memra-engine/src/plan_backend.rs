//! Runtime binding for execution manifests owned by the ModelPlan compiler.

pub use memra_gguf::execution_manifest::*;

use crate::hybrid::{Ffn, HybridModel, Mixer};
use crate::model::GpuTensor;
use std::fmt::Write as _;

mod runtime_identity;
pub use runtime_identity::running_implementation_sha256;
use runtime_identity::{LoadedLibraries, hash_parts, numeric_environment, numeric_program_sha256};
pub(crate) use runtime_identity::{
    RewriteLoadState, checked_rewrite_identity, identity_requested, install_rewrite_admission,
    refuse_external_rewrite_artifacts, rewrite_admission_allows, source_artifact_identity,
};

impl RewriteLoadState {
    // Deliberately recomputed for strict admission while public model fields can mutate.
    // This is O(model metadata) per check, including pipeline checks on token paths.
    // Moving it to a request boundary requires an immutable model/program API first;
    // strict-serving performance is pending a target-rig gate. Legacy is unaffected.
    pub(crate) fn validate(&self, model: &HybridModel) -> Result<(), String> {
        self.libraries
            .as_ref()
            .ok_or("loaded-library identity missing")?
            .validate()?;
        let environment = numeric_environment(std::env::vars_os());
        let actual = loaded_model_sha256(model);
        if self.matches_snapshot(&model.plan, &actual, &environment) {
            return Ok(());
        }
        if self.plan != model.plan {
            return Err("compiled plan changed since load".into());
        }
        if self.environment != environment {
            let keys: std::collections::BTreeSet<_> = self
                .environment
                .keys()
                .chain(environment.keys())
                .filter(|key| self.environment.get(*key) != environment.get(*key))
                .collect();
            return Err(format!(
                "numerical environment changed since load: keys={keys:?}"
            ));
        }
        Err(format!(
            "loaded tensor program changed: expected={} actual={actual}",
            self.model_sha256
        ))
    }
}

pub(crate) fn capture_rewrite_identity(
    model: &HybridModel,
    source: &dyn memra_gguf::source::TensorSource,
    artifact_sha256: String,
    load_mtp: bool,
) -> Result<(RewriteIdentity, RewriteLoadState), Box<dyn std::error::Error>> {
    refuse_external_rewrite_artifacts()?;
    if model.plan.draft_source == memra_gguf::model_plan::DraftSourcePlan::ExternalArtifact
        || model.glm5_dflash.is_some()
        || model.frspec_src_sha16.is_some()
    {
        return Err(
            "rewrite identity requires a composite identity for external draft artifacts".into(),
        );
    }
    let state = RewriteLoadState {
        plan: model.plan.clone(),
        model_sha256: loaded_model_sha256(model),
        environment: numeric_environment(std::env::vars_os()),
        libraries: Some(LoadedLibraries::capture()?),
    };
    let interpretation = format!(
        "gguf={} safetensors={} activation={:?} preserve_experts={} nvfp4={} loaded_libraries={}",
        source.gguf().is_some(),
        source.st_dir().is_some(),
        source.expert_activation_precision(),
        source.preserve_expert_encodings(),
        source.nvfp4_cache_tag(),
        state.libraries.as_ref().unwrap().sha256,
    );
    let identity = RewriteIdentity {
        artifact_sha256,
        implementation_sha256: running_implementation_sha256()?,
        numeric_program_sha256: numeric_program_sha256(
            load_mtp,
            &interpretation,
            &hardware_snapshot(&model.devices())?,
            &state.model_sha256,
            &state.environment,
        ),
    };
    identity.validate()?;
    Ok((identity, state))
}

fn hardware_snapshot(devices: &[usize]) -> Result<String, Box<dyn std::error::Error>> {
    use cudarc::driver::{result, sys};
    let mut driver = 0;
    // CUDA is already initialized by the loader; this only queries it.
    unsafe { sys::cuDriverGetVersion(&mut driver).result()? };
    let mut out = format!("driver_api={driver}\n");
    #[cfg(target_os = "linux")]
    out.push_str(
        &std::fs::read_to_string("/proc/driver/nvidia/version")
            .map_err(|e| format!("cannot bind the loaded NVIDIA driver version: {e}"))?,
    );
    for &ordinal in devices {
        let device = result::device::get(ordinal.try_into()?)?;
        writeln!(
            out,
            "device={ordinal} name={} uuid={:?}",
            result::device::get_name(device)?,
            result::device::get_uuid(device)?.bytes
        )?;
        // SAFETY: `device` is returned by cuDeviceGet above, and no context is created.
        unsafe {
            writeln!(out, "total_mem={}", result::device::total_mem(device)?)?;
            for attribute in [
                sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID,
                sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PCI_BUS_ID,
                sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID,
            ] {
                writeln!(
                    out,
                    "{attribute:?}={}",
                    result::device::get_attribute(device, attribute)?
                )?;
            }
        }
    }
    Ok(out)
}

// Only host metadata is read here. Never serialize CUDA pointers, scratch buffers or
// free-memory readings: those change without changing the program. The actual selected
// mirrors ARE included because free memory can select their numerical class at load.
fn tensor_program(out: &mut String, tensor: &GpuTensor) {
    write!(out, "device={} ", tensor.ordinal()).unwrap();
    match tensor {
        GpuTensor::Quant {
            qtype,
            row_bytes,
            ne,
            scale,
            rp,
            fp8,
            rp4,
            blk,
            f16,
            a4,
            ..
        } => {
            writeln!(out, "quant={qtype} rows={row_bytes} shape={ne:?} scale={:08x} rp={rp} rp4={} fp8={:?} blk={:?} f16={} a4={a4:?}",
                scale.to_bits(), rp4.is_some(),
                fp8.as_ref().map(|w| (w.scale.to_bits(), w.blk.as_ref().map(|b| (b.rows, b.cols)))),
                blk.as_ref().map(|b| (b.rows, b.cols)), f16.is_some()).unwrap();
            #[cfg(memra_cutlass)]
            if let GpuTensor::Quant { cutlass, .. } = tensor {
                writeln!(out, "cutlass={}", cutlass.is_some()).unwrap();
            }
        }
        GpuTensor::Float { ne, .. } => writeln!(out, "f32={ne:?}").unwrap(),
        GpuTensor::FloatBf16 { ne, .. } => writeln!(out, "bf16={ne:?}").unwrap(),
    }
}

fn mixer_program(out: &mut String, mixer: &Mixer) {
    match mixer {
        Mixer::Full(a) => {
            writeln!(
                out,
                "full tp={:?}",
                a.step_tp_qkv
                    .as_ref()
                    .map(|tp| (&tp.devices, tp.attention.is_some()))
            )
            .unwrap();
            for t in [&a.wq, &a.wk, &a.wv, &a.wo]
                .into_iter()
                .chain(a.wo_pqs.iter())
                .chain(a.q_norm.iter())
                .chain(a.k_norm.iter())
                .chain(a.attn_gate.iter())
            {
                tensor_program(out, t);
            }
        }
        Mixer::Linear(a) => {
            writeln!(out, "linear={:?}", a.geometry).unwrap();
            for t in [
                &a.wqkv,
                &a.wqkv_gate,
                &a.ssm_beta,
                &a.ssm_alpha,
                &a.ssm_a,
                &a.ssm_dt,
                &a.ssm_conv1d,
                &a.ssm_norm,
                &a.ssm_out,
            ] {
                tensor_program(out, t);
            }
        }
        Mixer::Mla(a) => {
            writeln!(
                out,
                "mla={:?} bf16={:?} tp={} shard={}",
                a.geom,
                (a.wk_b16.is_some(), a.wv_b16.is_some()),
                a.tp.is_some(),
                a.tp_shard
            )
            .unwrap();
            for t in [
                &a.wq_a,
                &a.q_a_norm,
                &a.wq_b,
                &a.wkv_a,
                &a.kv_a_norm,
                &a.wk_b,
                &a.wv_b,
                &a.wo,
            ] {
                tensor_program(out, t);
            }
            if let Some(index) = &a.index {
                writeln!(out, "index={:?}", index.geom).unwrap();
                for t in [
                    &index.wq_b,
                    &index.wk,
                    &index.k_norm_w,
                    &index.k_norm_b,
                    &index.weights_proj,
                    &index.kpool_gate,
                    &index.kpool_ape,
                ] {
                    tensor_program(out, t);
                }
            }
        }
        Mixer::Kda(a) => {
            writeln!(out, "kda={:?} tp={}", a.plan, a.tp.is_some()).unwrap();
            for t in [
                &a.wq, &a.wk, &a.wv, &a.f_a, &a.f_b, &a.g_a, &a.g_b, &a.b_proj, &a.wo, &a.a_log,
                &a.dt_bias, &a.o_norm,
            ] {
                tensor_program(out, t);
            }
        }
    }
}

fn ffn_program(out: &mut String, ffn: &Ffn) {
    match ffn {
        Ffn::Dense {
            ffn_gate,
            ffn_up,
            ffn_down,
            ffn_down_pqs,
        } => {
            out.push_str("dense\n");
            for t in [ffn_gate, ffn_up, ffn_down]
                .into_iter()
                .chain(ffn_down_pqs.iter())
            {
                tensor_program(out, t);
            }
        }
        Ffn::Moe(m) => {
            writeln!(out, "moe active={:?} bias={:?} bf16={} macros={} dev={:?} step_ep={:?} step_tp={:?} glm_ep={} glm_tp={}",
                m.active_experts, m.exp_probs_b, m.w4a16_bf16_activations, m.has_macros,
                m.dev_exps.as_ref().map(|d| (d.dev, d.rp, d.fp8_blk.is_some())),
                m.step_ep.as_ref().map(|e| (&e.devices, e.configured_by_tp, e.activation_limit, e.nvfp4_device_routes)),
                m.step_tp.as_ref().map(|e| (&e.devices, e.activation_limit)), m.glm5_ep.is_some(), m.glm5_tp_split.is_some()).unwrap();
            for t in [&m.gate_inp]
                .into_iter()
                .chain(m.gate_inp_shexp.iter())
                .chain(m.gate_shexp.iter())
                .chain(m.up_shexp.iter())
                .chain(m.down_shexp.iter())
            {
                tensor_program(out, t);
            }
            for exps in [&m.gate_exps, &m.up_exps, &m.down_exps] {
                writeln!(out, "experts qt={} in={} out={} count={} row={} stride={} layouts={:?} macros={:?} fp8={:?}",
                    exps.qtype, exps.in_f, exps.out_f, exps.n_expert, exps.row_bytes, exps.expert_stride, exps.layouts, exps.macros,
                    exps.fp8_blk.as_ref().map(|b| (b.rows, b.cols, b.expert_stride))).unwrap();
            }
        }
    }
}

fn loaded_model_sha256(model: &HybridModel) -> String {
    let mut out = format!(
        "config={:?}\nplan={:?}\ndevices={:?}\nembedding={:?}/{}\nhyper={:?}\n",
        model.cfg,
        model.plan,
        model.devices(),
        model.embd.ggml_type,
        model.embd.n_embd,
        model.hyper
    );
    tensor_program(&mut out, &model.output_norm);
    tensor_program(&mut out, &model.output);
    for (i, layer) in model.layers.iter().enumerate() {
        writeln!(out, "layer={i} tp_glue={}", layer.tp_glue.len()).unwrap();
        tensor_program(&mut out, &layer.attn_norm);
        tensor_program(&mut out, &layer.post_attn_norm);
        mixer_program(&mut out, &layer.mixer);
        ffn_program(&mut out, &layer.ffn);
        if let Some(g) = &layer.gemma4 {
            writeln!(out, "gemma scale={:08x}", g.layer_scale.to_bits()).unwrap();
            tensor_program(&mut out, &g.ffn_norm);
            tensor_program(&mut out, &g.post_ffw_norm);
            if let Some(m) = &g.moe_bits {
                for t in [
                    &m.post_ffw_norm_1,
                    &m.pre_ffw_norm_2,
                    &m.post_ffw_norm_2,
                    &m.shared_gate,
                    &m.shared_up,
                    &m.shared_down,
                ] {
                    tensor_program(&mut out, t);
                }
            }
            if let Some(e) = &g.e4b {
                writeln!(out, "kv_share={:?}", e.kv_share).unwrap();
                for t in [&e.inp_gate, &e.proj, &e.post_norm]
                    .into_iter()
                    .chain(e.qkv_cat.iter())
                {
                    tensor_program(&mut out, t);
                }
            }
        }
    }
    writeln!(
        out,
        "mtp={} extra={}",
        model.mtp.is_some(),
        model.mtp_extra.len()
    )
    .unwrap();
    for head in model.mtp.iter().chain(&model.mtp_extra) {
        writeln!(
            out,
            "head geom={:?} step={:?} d2t={:?} target={}",
            head.geom
                .as_ref()
                .map(|g| (g.d_inner, g.n_head, g.n_head_kv)),
            head.step35,
            head.d2t,
            head.d2t_from_target_head
        )
        .unwrap();
        for t in [
            &head.enorm,
            &head.hnorm,
            &head.eh_proj,
            &head.attn_norm,
            &head.post_attn_norm,
        ]
        .into_iter()
        .chain(head.shared_head_norm.iter())
        .chain(head.shared_head_head.iter())
        .chain(head.geom.iter().map(|g| &g.out_up))
        {
            tensor_program(&mut out, t);
        }
        mixer_program(&mut out, &head.mixer);
        ffn_program(&mut out, &head.ffn);
    }
    writeln!(
        out,
        "external={} trim={:?}",
        model.glm5_dflash.is_some(),
        model.frspec_src_sha16
    )
    .unwrap();
    hash_parts([out])
}

pub fn bind_rewrite_artifact(
    model: &HybridModel,
    receipt: RewriteParityReceipt,
) -> Result<RewriteParityReceipt, Box<dyn std::error::Error>> {
    let identity = model.rewrite_identity()?;
    let path = std::env::var_os("MEMRA_ARTIFACT_LOCK")
        .ok_or("rewrite receipt requires MEMRA_ARTIFACT_LOCK")?;
    // The lock links inspection evidence only. It cannot supply any runtime identity.
    Ok(receipt
        .bind_runtime_identity(identity)?
        .bind_artifact_lock(&std::fs::read(path)?))
}
