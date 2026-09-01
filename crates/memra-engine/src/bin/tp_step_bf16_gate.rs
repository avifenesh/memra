//! Official Step-3.7 BF16 projection gate for native tensor-parallel collectives.
//!
//! The existing host-staged projection is the exact oracle. This gate proves that one activation
//! upload, native peer broadcast, rank-local BF16 projection, and native peer gather preserve its
//! token-major output exactly. The final root readback remains intentional; this is not distributed
//! attention or product-throughput evidence.

use memra_engine::Engine;
use memra_engine::parallel::{
    HardwareTarget, ModelParallelContract, TopologyRequest, validate_step_fp8_checkpoint,
};
use memra_engine::tp::{Bf16Matrix, TpE4m3HostBounce, step_tp_f32_mirror_enabled};
use memra_gguf::GgmlType;
use memra_gguf::source::{SafetensorsSource, TensorSource};

fn devices() -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    let raw = std::env::var("MEMRA_TP_DEVICES").unwrap_or_else(|_| "0,1".to_string());
    let devices = raw
        .split(',')
        .map(|part| part.trim().parse::<usize>())
        .collect::<Result<Vec<_>, _>>()?;
    if !(2..=8).contains(&devices.len()) {
        return Err(
            format!("Step BF16 TP gate requires 2..=8 ranks, MEMRA_TP_DEVICES={raw:?}").into(),
        );
    }
    Ok(devices)
}

fn activations(tokens: usize, width: usize) -> Vec<f32> {
    (0..tokens * width)
        .map(|index| {
            let mixed = index.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((mixed % 8191) as f32 - 4095.0) / 2048.0
        })
        .collect()
}

fn compare_exact(label: &str, expected: &[f32], actual: &[f32]) -> Result<(), String> {
    if expected.len() != actual.len() {
        return Err(format!(
            "{label}: output lengths differ, {} != {}",
            expected.len(),
            actual.len()
        ));
    }
    let mut mismatches = 0usize;
    let mut max_abs = 0.0f32;
    let mut peak = 0.0f32;
    let mut first = None;
    for (index, (&left, &right)) in expected.iter().zip(actual).enumerate() {
        if !left.is_finite() || !right.is_finite() {
            return Err(format!("{label}: non-finite value at output {index}"));
        }
        max_abs = max_abs.max((left - right).abs());
        peak = peak.max(left.abs());
        if left.to_bits() != right.to_bits() {
            mismatches += 1;
            first.get_or_insert((index, left, right));
        }
    }
    let relative_to_peak = max_abs / peak.max(1.0);
    if let Some((index, left, right)) = first {
        println!(
            "TP_BF16_EXACT label={label} bit_mismatches={mismatches}/{} \
             max_abs={max_abs:.9e} relative_to_peak={relative_to_peak:.9e} \
             first_index={index} expected={left:.9e} actual={right:.9e} \
             expected_bits={:08x} actual_bits={:08x}",
            expected.len(),
            left.to_bits(),
            right.to_bits(),
        );
    } else {
        println!(
            "TP_BF16_EXACT label={label} bit_mismatches=0/{} \
             max_abs=0.000000000e0 relative_to_peak=0.000000000e0",
            expected.len()
        );
    }
    if mismatches != 0 {
        return Err(format!(
            "{label}: BF16 output differs from its exact oracle"
        ));
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = std::env::args().nth(1).expect(
        "usage: tp-step-bf16-gate <official-step-safetensors-dir> \
             [layer] [q|k|v] [ambient-device]",
    );
    let layer = std::env::args()
        .nth(2)
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(3);
    let projection = std::env::args().nth(3).unwrap_or_else(|| "q".to_string());
    let ambient_device = std::env::args()
        .nth(4)
        .map(|value| value.parse::<usize>())
        .transpose()?;
    let tokens = std::env::var("MEMRA_TP_TOKENS")
        .ok()
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(2);
    let f32_mirror = step_tp_f32_mirror_enabled()?;
    if tokens == 0 {
        return Err("MEMRA_TP_TOKENS must be nonzero".into());
    }

    let source = SafetensorsSource::open(std::path::Path::new(&model))?;
    let contract = ModelParallelContract::from_model(&source.config())?;
    if layer >= contract.trunk_layers {
        return Err(format!(
            "Step BF16 TP layer {layer} is outside trunk layers 0..{}",
            contract.trunk_layers
        )
        .into());
    }
    let qualified = validate_step_fp8_checkpoint(&source, &contract)?;
    let devices = devices()?;
    let plan = contract.plan(TopologyRequest {
        pipeline: 1,
        tensor: devices.len(),
        // Step's 1,280-wide routed projection supports tensor-sharded experts only at TP2.
        // TP4/TP8 pair legal attention TP with expert ownership; this gate still exercises
        // only Q/K/V column sharding and never labels expert parallelism as tensor parallelism.
        expert_parallel: devices.len() > 2,
        available_devices: devices.len(),
        hardware: HardwareTarget::RtxPro6000Blackwell,
    })?;

    let (suffix, expected_out) = match projection.as_str() {
        "q" => (
            "attn_q.weight",
            contract.query_heads[layer] * contract.head_dim,
        ),
        "k" => (
            "attn_k.weight",
            contract.kv_heads[layer] * contract.head_dim,
        ),
        "v" => (
            "attn_v.weight",
            contract.kv_heads[layer] * contract.head_dim,
        ),
        _ => return Err(format!("projection must be q, k, or v, got {projection:?}").into()),
    };
    let name = format!("blk.{layer}.{suffix}");
    let tensor = source
        .find(&name)
        .ok_or_else(|| format!("official Step checkpoint is missing {name}"))?;
    if tensor.ggml_type != GgmlType::BF16 {
        return Err(format!(
            "official Step projection {name} must preserve BF16 bytes, got {:?}",
            tensor.ggml_type
        )
        .into());
    }
    if tensor.ne.len() != 2 {
        return Err(format!("{name} must be a 2-D matrix, got {:?}", tensor.ne).into());
    }
    let matrix = Bf16Matrix {
        bytes: tensor.bytes.as_ref(),
        in_features: tensor.ne[0] as usize,
        out_features: tensor.ne[1] as usize,
    };
    matrix.validate()?;
    if matrix.in_features != contract.hidden_size || matrix.out_features != expected_out {
        return Err(format!(
            "{name} shape {}x{} != registered {expected_out}x{}",
            matrix.out_features, matrix.in_features, contract.hidden_size
        )
        .into());
    }

    let runtime = TpE4m3HostBounce::new_native_p2p(&devices)?;
    let names = runtime.device_names()?;
    if names
        .iter()
        .any(|name| !name.contains("RTX PRO 6000") || !name.contains("Blackwell"))
    {
        return Err(format!("unqualified TP hardware: {names:?}").into());
    }
    let resident = if f32_mirror {
        runtime.upload_step_bf16_column_parallel_f32_mirror(matrix)?
    } else {
        runtime.upload_step_bf16_column_parallel(matrix)?
    };
    let input = activations(tokens, matrix.in_features);
    let oracle = runtime.bf16_column_parallel_resident(&resident, &input, tokens)?;
    let canonical_runtime = TpE4m3HostBounce::new_single_rank_oracle(devices[0])?;
    let canonical_resident = canonical_runtime.upload_step_bf16_column_parallel(matrix)?;
    let canonical =
        canonical_runtime.bf16_column_parallel_resident(&canonical_resident, &input, tokens)?;

    let ambient_engine = if let Some(device) = ambient_device {
        if device == devices[0] {
            return Err(format!(
                "ambient-device {device} must differ from TP root device {}",
                devices[0]
            )
            .into());
        }
        println!(
            "TP_BF16_AMBIENT_CONTEXT device={device} tp_root={} purpose=pp-scope-regression",
            devices[0]
        );
        Some(Engine::new(device)?)
    } else {
        None
    };
    let run_native = || -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let _ambient = ambient_engine
            .as_ref()
            .map(|engine| engine.gpu.enter_main())
            .transpose()?;
        runtime.bf16_column_parallel_resident_native(&resident, &input, tokens)
    };
    let native_first = run_native()?;
    let native_second = run_native()?;
    compare_exact("host_oracle_vs_native", &oracle.gathered, &native_first)?;
    compare_exact("native_repeat", &native_first, &native_second)?;
    compare_exact(
        "canonical_tp1_vs_host_tp",
        &canonical.gathered,
        &oracle.gathered,
    )?;
    compare_exact(
        "canonical_tp1_vs_native",
        &canonical.gathered,
        &native_first,
    )?;

    if f32_mirror {
        println!(
            "STEP_TP_BF16_GATE_PASS layer={layer} projection={projection} tokens={tokens} \
             tp={} checkpoint_fp8_projections={qualified} resident_weights=true \
             bf16_residency=f32-mirror canonical_tp1_residency=bf16 \
             canonical_tp1_tp_exact=true host_oracle_native_exact=true \
             native_repeat_exact=true devices={devices:?} names={names:?} \
             transport={} native_p2p={} activation=host-canonical output=root-readback \
             performance_claim=false",
            plan.request.tensor,
            runtime.transport_label(),
            runtime.native_p2p(),
        );
    } else {
        println!(
            "STEP_TP_BF16_GATE_PASS layer={layer} projection={projection} tokens={tokens} \
             tp={} checkpoint_fp8_projections={qualified} resident_weights=true \
             canonical_tp1_tp_exact=true host_oracle_native_exact=true \
             native_repeat_exact=true devices={devices:?} names={names:?} \
             transport={} native_p2p={} activation=host-canonical output=root-readback \
             performance_claim=false",
            plan.request.tensor,
            runtime.transport_label(),
            runtime.native_p2p(),
        );
    }
    Ok(())
}
