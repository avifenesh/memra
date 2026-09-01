//! Official Step-3.7-Flash-NVFP4 checkpoint gate for the first NVFP4 TP execution slice.
//!
//! This executes one routed expert from the checkpoint's native stacked modelopt NVFP4 banks on
//! two independent CUDA devices. Gate/up use column parallelism; down uses 64-superblock-aligned
//! row parallelism. Rank compute repacks modelopt bytes into memra block_nvfp4 rows (value-exact
//! nibble reorder) and runs the proven `qmatvec_nvfp4_fast` dp4a kernel; the per-expert
//! `weight_scale_2` macro applies once post-gather/post-reduce (canonical order). A host f32
//! dequant reference pins the modelopt decode semantics before any TP comparison. Host bounce is
//! the only transport in this increment. Nothing here is product-throughput evidence.

use memra_engine::parallel::{
    HardwareTarget, ModelParallelContract, TopologyRequest, validate_step_nvfp4_checkpoint,
};
use memra_engine::tp::{Nvfp4BlockMatrix, Nvfp4ExpertBank, TpE4m3HostBounce};
use memra_gguf::source::{Nvfp4StackedNative, SafetensorsSource, TensorSource};

fn devices() -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    let raw = std::env::var("MEMRA_TP_DEVICES").unwrap_or_else(|_| "0,1".to_string());
    let devices = raw
        .split(',')
        .map(|part| part.trim().parse::<usize>())
        .collect::<Result<Vec<_>, _>>()?;
    if devices.len() != 2 {
        return Err(format!(
            "first Step NVFP4 TP runtime gate requires exactly two ranks, MEMRA_TP_DEVICES={raw:?}"
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
    bank: &'a Nvfp4StackedNative<'a>,
    expert: usize,
) -> Result<Nvfp4BlockMatrix<'a>, Box<dyn std::error::Error>> {
    if expert >= bank.n_expert {
        return Err(format!("expert {expert} outside 0..{}", bank.n_expert).into());
    }
    let code_stride = bank.out_f * bank.in_f / 2;
    let scale_stride = bank.out_f * bank.in_f / 16;
    Ok(Nvfp4BlockMatrix {
        codes: &bank.codes[expert * code_stride..(expert + 1) * code_stride],
        scales: &bank.scales[expert * scale_stride..(expert + 1) * scale_stride],
        macro_scale: bank.macros[expert],
        out_features: bank.out_f,
        in_features: bank.in_f,
    })
}

fn expert_bank<'a>(bank: &'a Nvfp4StackedNative<'a>) -> Nvfp4ExpertBank<'a> {
    Nvfp4ExpertBank {
        codes: bank.codes,
        scales: bank.scales,
        macros: &bank.macros,
        expert_count: bank.n_expert,
        out_features: bank.out_f,
        in_features: bank.in_f,
    }
}

/// Host f32 dequant of one output row: sum over 16-element groups of
/// `e2m1(code) * ue4m3(scale)`, then x-weighted, then the per-expert macro. This is the modelopt
/// semantic the kernel path must reproduce (within its q8_1 activation quantization).
fn host_row_reference(matrix: Nvfp4BlockMatrix<'_>, row: usize, input: &[f32]) -> f32 {
    let row_values = memra_gguf::nvfp4_repack::dequant_modelopt_row(
        &matrix.codes[row * matrix.in_features / 2..(row + 1) * matrix.in_features / 2],
        &matrix.scales[row * matrix.in_features / 16..(row + 1) * matrix.in_features / 16],
        matrix.in_features,
    );
    row_values
        .iter()
        .zip(input)
        .map(|(&weight, &x)| weight * x)
        .sum::<f32>()
        * matrix.macro_scale
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
    let model = std::env::args()
        .nth(1)
        .expect("usage: tp_step_nvfp4_gate <official-step-nvfp4-safetensors-dir> [layer] [expert]");
    let layer = std::env::args()
        .nth(2)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(3);
    let expert = std::env::args()
        .nth(3)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);
    let tokens = std::env::var("MEMRA_TP_TOKENS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1);

    let source = SafetensorsSource::open(std::path::Path::new(&model))?;
    let contract = ModelParallelContract::from_model(&source.config())?;
    let qualified = validate_step_nvfp4_checkpoint(&source, &contract)?;
    let devices = devices()?;
    let plan = contract.plan(TopologyRequest {
        pipeline: 1,
        tensor: 2,
        expert_parallel: false,
        available_devices: devices.len(),
        hardware: HardwareTarget::RtxPro6000Blackwell,
    })?;
    let runtime = TpE4m3HostBounce::new(&devices)?;
    let names = runtime.device_names()?;
    if names
        .iter()
        .any(|name| !name.contains("RTX PRO 6000") || !name.contains("Blackwell"))
    {
        return Err(format!("unqualified TP hardware: {names:?}").into());
    }
    println!(
        "TP_CONTRACT variant={} checkpoint_nvfp4_projections={qualified} devices={devices:?} \
         names={names:?} world={} transport=host-bounce native_p2p=false \
         performance_claim=false",
        contract.variant, plan.world_size,
    );

    let name = |projection: &str| format!("blk.{layer}.ffn_{projection}_exps.weight");
    let gate_bank = source
        .find_nvfp4_stacked_native(&name("gate"))
        .ok_or_else(|| format!("missing native NVFP4 {}", name("gate")))?;
    let up_bank = source
        .find_nvfp4_stacked_native(&name("up"))
        .ok_or_else(|| format!("missing native NVFP4 {}", name("up")))?;
    let down_bank = source
        .find_nvfp4_stacked_native(&name("down"))
        .ok_or_else(|| format!("missing native NVFP4 {}", name("down")))?;
    let gate = expert_matrix(&gate_bank, expert)?;
    let up = expert_matrix(&up_bank, expert)?;
    let down = expert_matrix(&down_bank, expert)?;
    let input = activations(tokens, gate.in_features);

    // Decode-semantics pin: the kernel's first-token row 0 must match the host modelopt dequant
    // within the q8_1 activation-quantization envelope. Catches nibble-order, scale-grid, and
    // dropped-macro bugs decisively before any TP comparison.
    let gate_full = runtime.full_nvfp4(gate, &input, tokens)?;
    let reference = host_row_reference(gate, 0, &input[..gate.in_features]);
    let kernel = gate_full[0];
    let denom = reference.abs().max(1.0e-6);
    let rel = (kernel - reference).abs() / denom;
    println!(
        "NVFP4_DECODE_PIN layer={layer} expert={expert} host_ref={reference:.6e} \
         kernel={kernel:.6e} rel={rel:.6e}"
    );
    if !kernel.is_finite() || rel > 5.0e-2 {
        return Err(format!(
            "NVFP4 decode pin failed: kernel {kernel:.6e} vs host modelopt dequant \
             {reference:.6e} (rel {rel:.3e}) — decode semantics broken, not a TP question"
        )
        .into());
    }

    let gate_tp = runtime.column_parallel_nvfp4(gate, &input, tokens)?;
    compare_exact("expert_gate", &gate_full, &gate_tp.gathered)?;

    let up_full = runtime.full_nvfp4(up, &input, tokens)?;
    let up_tp = runtime.column_parallel_nvfp4(up, &input, tokens)?;
    compare_exact("expert_up", &up_full, &up_tp.gathered)?;

    let down_input: Vec<f32> = gate_full
        .iter()
        .zip(&up_full)
        .map(|(&gate, &up)| gate / (1.0 + (-gate).exp()) * up)
        .collect();
    let down_full = runtime.full_nvfp4(down, &down_input, tokens)?;
    let down_tp = runtime.row_parallel_nvfp4(down, &down_input, tokens)?;
    compare_reduced(
        "expert_down",
        &down_full,
        &down_tp.reduced,
        tokens,
        down.out_features,
    )?;

    let resident = runtime.upload_expert_nvfp4(gate, up, down)?;
    let resident_first = runtime.run_expert_nvfp4(&resident, &input, tokens)?;
    let resident_second = runtime.run_expert_nvfp4(&resident, &input, tokens)?;
    compare_reduced(
        "resident_expert",
        &down_full,
        &resident_first,
        tokens,
        down.out_features,
    )?;
    compare_repeat("resident_expert", &resident_first, &resident_second)?;

    let resident_bank = runtime.upload_tensor_parallel_nvfp4(
        expert_bank(&gate_bank),
        expert_bank(&up_bank),
        expert_bank(&down_bank),
    )?;
    let canonical_runtime = TpE4m3HostBounce::new_single_rank_oracle(devices[0])?;
    let canonical_bank = canonical_runtime.upload_tensor_parallel_nvfp4(
        expert_bank(&gate_bank),
        expert_bank(&up_bank),
        expert_bank(&down_bank),
    )?;
    let selected = vec![expert; tokens];
    let route_weights = vec![1.0f32; tokens];
    let canonical_first = canonical_runtime.run_tensor_parallel_routes_nvfp4(
        &canonical_bank,
        &input,
        tokens,
        &selected,
        &route_weights,
        1,
        None,
    )?;
    let canonical_second = canonical_runtime.run_tensor_parallel_routes_nvfp4(
        &canonical_bank,
        &input,
        tokens,
        &selected,
        &route_weights,
        1,
        None,
    )?;
    let resident_bank_first = runtime.run_tensor_parallel_routes_nvfp4(
        &resident_bank,
        &input,
        tokens,
        &selected,
        &route_weights,
        1,
        None,
    )?;
    let resident_bank_second = runtime.run_tensor_parallel_routes_nvfp4(
        &resident_bank,
        &input,
        tokens,
        &selected,
        &route_weights,
        1,
        None,
    )?;
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
        "STEP_TP2_NVFP4_GATE_PASS layer={layer} expert={expert} tokens={tokens} \
         resident_weights=true canonical_tp1_tp2_exact=true macro_fold=post_reduce_once \
         transport=host-bounce native_p2p=false performance_claim=false",
    );
    Ok(())
}
