//! Official Step-3.7 checkpoint gate for the first real TP execution slice.
//!
//! This executes one routed expert from the checkpoint's native stacked E4M3 banks on two
//! independent CUDA devices. Gate/up use column parallelism; down uses canonical checkpoint-block
//! reduction. Host bounce is the default oracle; `MEMRA_STEP_TP_NATIVE_P2P=1` exercises the same
//! program with peer collectives. Neither mode is product-throughput evidence.

use memra_engine::Engine;
use memra_engine::parallel::{
    HardwareTarget, ModelParallelContract, TopologyRequest, validate_step_fp8_checkpoint,
};
use memra_engine::tp::{
    E4m3BlockMatrix, E4m3ExpertBank, TpE4m3HostBounce, step_tp_native_p2p_enabled,
};
use memra_gguf::source::{Fp8StackedNative, SafetensorsSource, TensorSource};

fn devices() -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    let raw = std::env::var("MEMRA_TP_DEVICES").unwrap_or_else(|_| "0,1".to_string());
    let devices = raw
        .split(',')
        .map(|part| part.trim().parse::<usize>())
        .collect::<Result<Vec<_>, _>>()?;
    if devices.len() != 2 {
        return Err(format!(
            "first Step TP runtime gate requires exactly two ranks, MEMRA_TP_DEVICES={raw:?}"
        )
        .into());
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

fn expert_matrix<'a>(
    bank: &'a Fp8StackedNative<'a>,
    expert: usize,
) -> Result<E4m3BlockMatrix<'a>, Box<dyn std::error::Error>> {
    if expert >= bank.n_expert {
        return Err(format!("expert {expert} outside 0..{}", bank.n_expert).into());
    }
    let code_stride = bank.out_f * bank.in_f;
    let scale_stride = bank.scale_rows * bank.scale_cols;
    Ok(E4m3BlockMatrix {
        codes: &bank.bytes[expert * code_stride..(expert + 1) * code_stride],
        scales: &bank.scales[expert * scale_stride..(expert + 1) * scale_stride],
        out_features: bank.out_f,
        in_features: bank.in_f,
    })
}

fn expert_bank<'a>(bank: &'a Fp8StackedNative<'a>) -> E4m3ExpertBank<'a> {
    E4m3ExpertBank {
        codes: bank.bytes,
        scales: &bank.scales,
        expert_count: bank.n_expert,
        out_features: bank.out_f,
        in_features: bank.in_f,
    }
}

fn compare_exact(label: &str, expected: &[f32], actual: &[f32]) -> Result<(), String> {
    let mismatches = expected
        .iter()
        .zip(actual)
        .filter(|(left, right)| left.to_bits() != right.to_bits())
        .count();
    println!(
        "TP_EXACT label={label} bit_mismatches={mismatches}/{}",
        expected.len()
    );
    if mismatches != 0 {
        return Err(format!(
            "{label}: column-sharded output differs from unsharded output"
        ));
    }
    Ok(())
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(index, _)| index)
        .unwrap()
}

fn compare_reduced(
    label: &str,
    expected: &[f32],
    actual: &[f32],
    tokens: usize,
    width: usize,
) -> Result<(), String> {
    let mut max_abs = 0.0f32;
    let mut peak = 0.0f32;
    for (&left, &right) in expected.iter().zip(actual) {
        if !left.is_finite() || !right.is_finite() {
            return Err(format!("{label}: non-finite row-parallel result"));
        }
        max_abs = max_abs.max((left - right).abs());
        peak = peak.max(left.abs());
    }
    let mut argmax_mismatches = 0;
    for token in 0..tokens {
        let range = token * width..(token + 1) * width;
        if argmax(&expected[range.clone()]) != argmax(&actual[range]) {
            argmax_mismatches += 1;
        }
    }
    let relative_to_peak = max_abs / peak.max(1.0);
    println!(
        "TP_REDUCE label={label} max_abs={max_abs:.6e} \
         relative_to_peak={relative_to_peak:.6e} argmax_mismatches={argmax_mismatches}"
    );
    if argmax_mismatches != 0 || relative_to_peak > 1.0e-5 {
        return Err(format!(
            "{label}: row-parallel reduction exceeds the correctness bound"
        ));
    }
    Ok(())
}

fn compare_repeat(label: &str, first: &[f32], second: &[f32]) -> Result<(), String> {
    let mismatches = first
        .iter()
        .zip(second)
        .filter(|(left, right)| left.to_bits() != right.to_bits())
        .count();
    println!(
        "TP_RESIDENT_REPEAT label={label} bit_mismatches={mismatches}/{}",
        first.len()
    );
    if mismatches != 0 {
        return Err(format!(
            "{label}: repeated resident execution is not bit-identical"
        ));
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = std::env::args().nth(1).expect(
        "usage: tp-step-fp8-gate <official-step-safetensors-dir> \
             [layer] [expert] [ambient-device]",
    );
    let layer = std::env::args()
        .nth(2)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(3);
    let expert = std::env::args()
        .nth(3)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);
    let ambient_device = std::env::args()
        .nth(4)
        .map(|value| value.parse::<usize>())
        .transpose()?;
    let tokens = std::env::var("MEMRA_TP_TOKENS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1);

    let source = SafetensorsSource::open(std::path::Path::new(&model))?;
    let contract = ModelParallelContract::from_model(&source.config())?;
    let qualified = validate_step_fp8_checkpoint(&source, &contract)?;
    let devices = devices()?;
    let plan = contract.plan(TopologyRequest {
        pipeline: 1,
        tensor: 2,
        expert_parallel: false,
        available_devices: devices.len(),
        hardware: HardwareTarget::RtxPro6000Blackwell,
    })?;
    let native_p2p = step_tp_native_p2p_enabled()?;
    let runtime = if native_p2p {
        TpE4m3HostBounce::new_native_p2p(&devices)?
    } else {
        TpE4m3HostBounce::new(&devices)?
    };
    let names = runtime.device_names()?;
    if names
        .iter()
        .any(|name| !name.contains("RTX PRO 6000") || !name.contains("Blackwell"))
    {
        return Err(format!("unqualified TP hardware: {names:?}").into());
    }
    println!(
        "TP_CONTRACT variant={} checkpoint_fp8_projections={qualified} devices={devices:?} \
         names={names:?} world={} transport={} native_p2p={} performance_claim=false",
        contract.variant,
        plan.world_size,
        runtime.transport_label(),
        runtime.native_p2p(),
    );

    let name = |projection: &str| format!("blk.{layer}.ffn_{projection}_exps.weight");
    let gate_bank = source
        .find_fp8_stacked_native(&name("gate"))
        .ok_or_else(|| format!("missing native E4M3 {}", name("gate")))?;
    let up_bank = source
        .find_fp8_stacked_native(&name("up"))
        .ok_or_else(|| format!("missing native E4M3 {}", name("up")))?;
    let down_bank = source
        .find_fp8_stacked_native(&name("down"))
        .ok_or_else(|| format!("missing native E4M3 {}", name("down")))?;
    let gate = expert_matrix(&gate_bank, expert)?;
    let up = expert_matrix(&up_bank, expert)?;
    let down = expert_matrix(&down_bank, expert)?;
    let input = activations(tokens, gate.in_features);

    let gate_full = runtime.full(gate, &input, tokens)?;
    let gate_tp = runtime.column_parallel(gate, &input, tokens)?;
    compare_exact("expert_gate", &gate_full, &gate_tp.gathered)?;

    let up_full = runtime.full(up, &input, tokens)?;
    let up_tp = runtime.column_parallel(up, &input, tokens)?;
    compare_exact("expert_up", &up_full, &up_tp.gathered)?;

    let down_input: Vec<f32> = gate_full
        .iter()
        .zip(&up_full)
        .map(|(&gate, &up)| gate / (1.0 + (-gate).exp()) * up)
        .collect();
    let down_full = runtime.full(down, &down_input, tokens)?;
    let down_tp = runtime.row_parallel(down, &down_input, tokens)?;
    compare_reduced(
        "expert_down",
        &down_full,
        &down_tp.reduced,
        tokens,
        down.out_features,
    )?;

    let resident = runtime.upload_expert(gate, up, down)?;
    let resident_first = runtime.run_expert(&resident, &input, tokens)?;
    let resident_second = runtime.run_expert(&resident, &input, tokens)?;
    compare_reduced(
        "resident_expert",
        &down_full,
        &resident_first,
        tokens,
        down.out_features,
    )?;
    compare_repeat("resident_expert", &resident_first, &resident_second)?;

    let resident_bank = runtime.upload_tensor_parallel(
        expert_bank(&gate_bank),
        expert_bank(&up_bank),
        expert_bank(&down_bank),
    )?;
    let canonical_runtime = TpE4m3HostBounce::new_single_rank_oracle(devices[0])?;
    let canonical_bank = canonical_runtime.upload_tensor_parallel(
        expert_bank(&gate_bank),
        expert_bank(&up_bank),
        expert_bank(&down_bank),
    )?;
    let selected = vec![expert; tokens];
    let route_weights = vec![1.0f32; tokens];
    let canonical_first = canonical_runtime.run_tensor_parallel_routes(
        &canonical_bank,
        &input,
        tokens,
        &selected,
        &route_weights,
        1,
    )?;
    let canonical_second = canonical_runtime.run_tensor_parallel_routes(
        &canonical_bank,
        &input,
        tokens,
        &selected,
        &route_weights,
        1,
    )?;
    let ambient_engine = if let Some(device) = ambient_device {
        if device == devices[0] {
            return Err(format!(
                "ambient-device {device} must differ from TP root device {}",
                devices[0]
            )
            .into());
        }
        println!(
            "TP_AMBIENT_CONTEXT device={device} tp_root={} purpose=pp-scope-regression",
            devices[0]
        );
        Some(Engine::new(device)?)
    } else {
        None
    };
    let run_resident_bank = || -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let _ambient = ambient_engine
            .as_ref()
            .map(|engine| engine.gpu.enter_main())
            .transpose()?;
        runtime.run_tensor_parallel_routes(
            &resident_bank,
            &input,
            tokens,
            &selected,
            &route_weights,
            1,
        )
    };
    let resident_bank_first = run_resident_bank()?;
    let resident_bank_second = run_resident_bank()?;
    compare_reduced(
        "legacy_full_vs_canonical_block",
        &down_full,
        &canonical_first,
        tokens,
        down.out_features,
    )?;
    compare_exact(
        "canonical_tp1_vs_tp2",
        &canonical_first,
        &resident_bank_first,
    )?;
    compare_repeat("canonical_tp1", &canonical_first, &canonical_second)?;
    compare_repeat("canonical_tp2", &resident_bank_first, &resident_bank_second)?;

    println!(
        "STEP_TP2_FP8_GATE_PASS layer={layer} expert={expert} tokens={tokens} \
         resident_weights=true canonical_tp1_tp2_exact=true \
         legacy_full_compatibility_only=true transport={} native_p2p={} \
         performance_claim=false",
        runtime.transport_label(),
        runtime.native_p2p(),
    );
    Ok(())
}
