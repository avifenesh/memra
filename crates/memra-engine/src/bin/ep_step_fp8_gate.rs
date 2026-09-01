//! Official Step-3.7 checkpoint gate for persistent expert-parallel ownership.
//!
//! Product ranks (two, four, or eight) own disjoint slices of the routed-expert bank. A strict
//! transport-only mode additionally admits other 2..=8 rank counts for physical-link evidence.
//! Selected experts execute on their owner CUDA context. This is the expert-ownership half of
//! Step's TP-attention/EP-expert product layouts, not TP or throughput evidence by itself.

use cudarc::driver::CudaSlice;
use memra_engine::Engine;
use memra_engine::hybrid::HybridModel;
use memra_engine::model::GpuTensor;
use memra_engine::parallel::{
    HardwareTarget, ModelParallelContract, TopologyRequest, validate_step_fp8_checkpoint,
};
use memra_engine::tp::{
    Bf16Matrix, E4m3BlockMatrix, E4m3ExpertBank, ResidentBf16ColumnParallel,
    ResidentExpertParallel, ResidentReplicatedDeviceRows, ResidentStepBf16RowParallel,
    StepGroupedFp8ProjectionOutput, TpE4m3HostBounce, moe_residual_host,
    step_expert_activation_host,
};
use memra_gguf::GgmlType;
use memra_gguf::config::{LayerGeometry, SwigluClamp};
use memra_gguf::source::{Fp8StackedNative, SafetensorsSource, TensorSource};
use std::hint::black_box;
use std::time::Instant;

unsafe extern "C" {
    fn cudaProfilerStart() -> i32;
    fn cudaProfilerStop() -> i32;
}

fn strict_bool(name: &str) -> Result<bool, Box<dyn std::error::Error>> {
    Ok(match std::env::var(name).ok().as_deref() {
        None | Some("") | Some("0") => false,
        Some("1") => true,
        Some(value) => return Err(format!("{name}={value:?} is invalid; expected 0 or 1").into()),
    })
}

fn devices(transport_only: bool) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    let raw = std::env::var("MEMRA_TP_DEVICES").unwrap_or_else(|_| "0,1".to_string());
    let devices = raw
        .split(',')
        .map(|part| part.trim().parse::<usize>())
        .collect::<Result<Vec<_>, _>>()?;
    if !(2..=8).contains(&devices.len()) || (!transport_only && !matches!(devices.len(), 2 | 4 | 8))
    {
        return Err(format!(
            "Step EP {} gate rejects {} ranks, MEMRA_TP_DEVICES={raw:?}",
            if transport_only {
                "transport-only"
            } else {
                "product"
            },
            devices.len(),
        )
        .into());
    }
    let mut unique = devices.clone();
    unique.sort_unstable();
    unique.dedup();
    if unique.len() != devices.len() {
        return Err(format!("Step EP devices must be distinct, got {devices:?}").into());
    }
    Ok(devices)
}

fn activations(tokens: usize, width: usize) -> Vec<f32> {
    activations_with_salt(tokens, width, 0)
}

fn activations_with_salt(tokens: usize, width: usize, salt: usize) -> Vec<f32> {
    (0..tokens * width)
        .map(|index| {
            let mixed = index
                .wrapping_add(salt.wrapping_mul(104_729))
                .wrapping_mul(1_664_525)
                .wrapping_add(1_013_904_223);
            ((mixed % 8191) as f32 - 4095.0) / 2048.0
        })
        .collect()
}

struct OwnedBf16Matrix {
    bytes: Vec<u8>,
    in_features: usize,
    out_features: usize,
}

impl OwnedBf16Matrix {
    fn view(&self) -> Bf16Matrix<'_> {
        Bf16Matrix {
            bytes: &self.bytes,
            in_features: self.in_features,
            out_features: self.out_features,
        }
    }
}

fn load_bf16_matrix(
    source: &dyn TensorSource,
    name: &str,
    expected_in: usize,
    expected_out: usize,
) -> Result<OwnedBf16Matrix, Box<dyn std::error::Error>> {
    let tensor = source
        .find(name)
        .ok_or_else(|| format!("official Step checkpoint is missing {name}"))?;
    if tensor.ggml_type != GgmlType::BF16 || tensor.ne.len() != 2 {
        return Err(format!(
            "{name} must be a 2-D BF16 matrix, got {:?} {:?}",
            tensor.ggml_type, tensor.ne
        )
        .into());
    }
    let matrix = OwnedBf16Matrix {
        bytes: tensor.bytes.into_owned(),
        in_features: tensor.ne[0] as usize,
        out_features: tensor.ne[1] as usize,
    };
    matrix.view().validate()?;
    if matrix.in_features != expected_in || matrix.out_features != expected_out {
        return Err(format!(
            "{name} shape {}x{} != registered {expected_out}x{expected_in}",
            matrix.out_features, matrix.in_features
        )
        .into());
    }
    Ok(matrix)
}

fn load_f32_vector(
    engine: &Engine,
    source: &dyn TensorSource,
    name: &str,
    expected: usize,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let tensor = GpuTensor::load_from_source(engine, source, name)?;
    let values = engine.dtoh(tensor.float_data())?;
    if values.len() != expected || !values.iter().all(|value| value.is_finite()) {
        return Err(format!(
            "{name} values {} != {expected} or contain a non-finite value",
            values.len()
        )
        .into());
    }
    Ok(values)
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

fn reference_expert(
    runtime: &TpE4m3HostBounce,
    gate_bank: &Fp8StackedNative<'_>,
    up_bank: &Fp8StackedNative<'_>,
    down_bank: &Fp8StackedNative<'_>,
    expert: usize,
    input: &[f32],
    activation_limit: Option<f32>,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let gate = runtime.full(expert_matrix(gate_bank, expert)?, input, 1)?;
    let up = runtime.full(expert_matrix(up_bank, expert)?, input, 1)?;
    let activated: Vec<f32> = gate
        .iter()
        .zip(&up)
        .map(|(&gate, &up)| step_expert_activation_host(gate, up, activation_limit))
        .collect();
    runtime.full(expert_matrix(down_bank, expert)?, &activated, 1)
}

const STEP_GROUPED_TOP_K: usize = 8;

fn grouped_selected_routes(tokens: usize, experts: usize) -> Result<Vec<usize>, String> {
    if tokens == 0 || experts < STEP_GROUPED_TOP_K || experts % STEP_GROUPED_TOP_K != 0 {
        return Err(format!(
            "grouped Step route fixture requires tokens > 0 and expert count divisible by \
             {STEP_GROUPED_TOP_K}, got tokens={tokens} experts={experts}"
        ));
    }
    let stride = experts / STEP_GROUPED_TOP_K;
    Ok((0..tokens)
        .flat_map(|token| {
            (0..STEP_GROUPED_TOP_K).map(move |slot| (slot * stride + token) % experts)
        })
        .collect())
}

fn dynamic_grouped_selected_routes(
    tokens: usize,
    experts: usize,
    ranks: usize,
) -> Result<Vec<usize>, String> {
    if tokens == 0 || !matches!(ranks, 2 | 4 | 8) || experts % ranks != 0 {
        return Err(format!(
            "dynamic grouped Step route fixture requires tokens > 0, 2/4/8 ranks, and divisible \
             experts, got tokens={tokens} experts={experts} ranks={ranks}"
        ));
    }
    let owner_by_slot: &[usize; STEP_GROUPED_TOP_K] = match ranks {
        2 => &[0, 0, 0, 0, 1, 1, 1, 1],
        4 => &[0, 1, 1, 2, 2, 2, 3, 3],
        8 => &[0, 1, 2, 3, 4, 5, 6, 7],
        _ => unreachable!(),
    };
    let per_rank = experts / ranks;
    let mut selected = Vec::with_capacity(tokens * STEP_GROUPED_TOP_K);
    for token in 0..tokens {
        for (slot, &rank) in owner_by_slot.iter().enumerate() {
            let local = (token * 11 + slot * 7 + 1) % per_rank;
            selected.push(rank * per_rank + local);
        }
    }
    Ok(selected)
}

fn grouped_route_weights(tokens: usize, shift: usize) -> Vec<f32> {
    (0..tokens)
        .flat_map(|token| {
            (0..STEP_GROUPED_TOP_K).map(move |slot| {
                let numerator = (slot + token + shift) % STEP_GROUPED_TOP_K + 1;
                numerator as f32 / 36.0
            })
        })
        .collect()
}

fn combine_down_rows_host(
    down: &[f32],
    route_weights: &[f32],
    tokens: usize,
    width: usize,
) -> Result<Vec<f32>, String> {
    let pairs = tokens
        .checked_mul(STEP_GROUPED_TOP_K)
        .ok_or("host grouped combine pair count overflow")?;
    let down_values = pairs
        .checked_mul(width)
        .ok_or("host grouped combine value count overflow")?;
    if route_weights.len() != pairs || down.len() != down_values {
        return Err(format!(
            "host grouped combine down/weights {}/{} != pairs {pairs} x width {width}",
            down.len(),
            route_weights.len()
        ));
    }
    let mut output = vec![0.0f32; tokens * width];
    for token in 0..tokens {
        for slot in 0..STEP_GROUPED_TOP_K {
            let pair = token * STEP_GROUPED_TOP_K + slot;
            let weight = route_weights[pair];
            let source = &down[pair * width..(pair + 1) * width];
            let destination = &mut output[token * width..(token + 1) * width];
            for (sum, &value) in destination.iter_mut().zip(source) {
                let product = weight * value;
                *sum += product;
            }
        }
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn reference_grouped_projections(
    runtime: &TpE4m3HostBounce,
    gate_bank: &Fp8StackedNative<'_>,
    up_bank: &Fp8StackedNative<'_>,
    down_bank: &Fp8StackedNative<'_>,
    input: &[f32],
    tokens: usize,
    selected: &[usize],
    activation_limit: Option<f32>,
) -> Result<StepGroupedFp8ProjectionOutput, Box<dyn std::error::Error>> {
    let pairs = tokens
        .checked_mul(STEP_GROUPED_TOP_K)
        .ok_or("grouped Step oracle route count overflow")?;
    if selected.len() != pairs {
        return Err(format!(
            "grouped Step oracle selected routes {} != {tokens}x{STEP_GROUPED_TOP_K} ({pairs})",
            selected.len()
        )
        .into());
    }
    let mut gate_rows = Vec::with_capacity(pairs * gate_bank.out_f);
    let mut up_rows = Vec::with_capacity(pairs * up_bank.out_f);
    let mut down_rows = Vec::with_capacity(pairs * down_bank.out_f);
    for (pair, &expert) in selected.iter().enumerate() {
        let token = pair / STEP_GROUPED_TOP_K;
        let input_row = &input[token * gate_bank.in_f..(token + 1) * gate_bank.in_f];
        let gate = runtime.full(expert_matrix(gate_bank, expert)?, input_row, 1)?;
        let up = runtime.full(expert_matrix(up_bank, expert)?, input_row, 1)?;
        let activated = gate
            .iter()
            .zip(&up)
            .map(|(&gate, &up)| step_expert_activation_host(gate, up, activation_limit))
            .collect::<Vec<_>>();
        let down = runtime.full(expert_matrix(down_bank, expert)?, &activated, 1)?;
        gate_rows.extend(gate);
        up_rows.extend(up);
        down_rows.extend(down);
    }
    Ok(StepGroupedFp8ProjectionOutput {
        gate: gate_rows,
        up: up_rows,
        down: down_rows,
    })
}

fn elapsed_us(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1_000_000.0
}

#[allow(clippy::too_many_arguments)]
fn run_grouped_fp8_gate(
    runtime: &TpE4m3HostBounce,
    gate_bank: &Fp8StackedNative<'_>,
    up_bank: &Fp8StackedNative<'_>,
    down_bank: &Fp8StackedNative<'_>,
    input: &[f32],
    tokens: usize,
    layer: usize,
    activation_limit: Option<f32>,
) -> Result<(), Box<dyn std::error::Error>> {
    let selected = grouped_selected_routes(tokens, gate_bank.n_expert)?;

    let start = Instant::now();
    let oracle = reference_grouped_projections(
        runtime,
        gate_bank,
        up_bank,
        down_bank,
        input,
        tokens,
        &selected,
        activation_limit,
    )?;
    let oracle_first_us = elapsed_us(start);
    let start = Instant::now();
    let oracle_repeat = reference_grouped_projections(
        runtime,
        gate_bank,
        up_bank,
        down_bank,
        input,
        tokens,
        &selected,
        activation_limit,
    )?;
    let oracle_repeat_us = elapsed_us(start);
    compare_exact(
        "grouped_oracle_gate_repeat",
        &oracle.gate,
        &oracle_repeat.gate,
    )?;
    compare_exact("grouped_oracle_up_repeat", &oracle.up, &oracle_repeat.up)?;
    compare_exact(
        "grouped_oracle_down_repeat",
        &oracle.down,
        &oracle_repeat.down,
    )?;

    let start = Instant::now();
    let mut prepared = runtime.prepare_step_grouped_fp8_gate(
        expert_bank(gate_bank),
        expert_bank(up_bank),
        expert_bank(down_bank),
        input,
        tokens,
        &selected,
        activation_limit,
    )?;
    let prepare_us = elapsed_us(start);
    let profile = strict_bool("MEMRA_STEP_EP_GROUPED_FP8_PROFILE")?;
    if profile {
        let result = unsafe { cudaProfilerStart() };
        if result != 0 {
            return Err(format!("cudaProfilerStart failed with CUDA error {result}").into());
        }
    }
    let grouped_runs = (|| {
        let start = Instant::now();
        let first = runtime.run_step_grouped_fp8_gate(&mut prepared)?;
        let grouped_first_us = elapsed_us(start);
        let start = Instant::now();
        let second = runtime.run_step_grouped_fp8_gate(&mut prepared)?;
        let grouped_repeat_us = elapsed_us(start);
        Ok::<_, Box<dyn std::error::Error>>((first, grouped_first_us, second, grouped_repeat_us))
    })();
    if profile {
        let result = unsafe { cudaProfilerStop() };
        if result != 0 {
            return Err(format!("cudaProfilerStop failed with CUDA error {result}").into());
        }
    }
    let (first, grouped_first_us, second, grouped_repeat_us) = grouped_runs?;

    compare_exact("grouped_gate_oracle", &oracle.gate, &first.gate)?;
    compare_exact("grouped_up_oracle", &oracle.up, &first.up)?;
    compare_exact("grouped_down_oracle", &oracle.down, &first.down)?;
    compare_exact("grouped_gate_repeat", &first.gate, &second.gate)?;
    compare_exact("grouped_up_repeat", &first.up, &second.up)?;
    compare_exact("grouped_down_repeat", &first.down, &second.down)?;

    let oracle_projection_launches = prepared.pairs() * 3 * 2;
    let oracle_best_us = oracle_first_us.min(oracle_repeat_us);
    let grouped_best_us = grouped_first_us.min(grouped_repeat_us);
    println!(
        "STEP_GROUPED_FP8_GATE_PASS layer={layer} tokens={} pairs={} experts_per_token={} \
         activation_limit={} csr_uploads=2 workspace_sets=3 preparation_count=1 executions=2 \
         hot_device_allocations=0 \
         oracle_projection_kernel_launches={oracle_projection_launches} \
         grouped_projection_kernel_launches=6 grouped_activation_kernel_launches=2 \
         prepare_us={prepare_us:.3} oracle_first_us={oracle_first_us:.3} \
         oracle_repeat_us={oracle_repeat_us:.3} grouped_first_us={grouped_first_us:.3} \
         grouped_repeat_us={grouped_repeat_us:.3} best_speedup={:.3} \
         raw_bit_gate=true raw_bit_up=true raw_bit_down=true production_path=false \
         routing=false combine=false tensor_parallel_claim=false",
        prepared.tokens(),
        prepared.pairs(),
        STEP_GROUPED_TOP_K,
        activation_limit.unwrap_or(0.0),
        oracle_best_us / grouped_best_us,
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_owner_grouped_fp8_gate(
    runtime: &TpE4m3HostBounce,
    resident: &ResidentExpertParallel,
    gate_bank: &Fp8StackedNative<'_>,
    up_bank: &Fp8StackedNative<'_>,
    down_bank: &Fp8StackedNative<'_>,
    input: &[f32],
    tokens: usize,
    layer: usize,
    activation_limit: Option<f32>,
) -> Result<(), Box<dyn std::error::Error>> {
    let selected = grouped_selected_routes(tokens, gate_bank.n_expert)?;

    let start = Instant::now();
    let oracle = reference_grouped_projections(
        runtime,
        gate_bank,
        up_bank,
        down_bank,
        input,
        tokens,
        &selected,
        activation_limit,
    )?;
    let oracle_first_us = elapsed_us(start);
    let start = Instant::now();
    let oracle_repeat = reference_grouped_projections(
        runtime,
        gate_bank,
        up_bank,
        down_bank,
        input,
        tokens,
        &selected,
        activation_limit,
    )?;
    let oracle_repeat_us = elapsed_us(start);
    compare_exact(
        "owner_grouped_oracle_gate_repeat",
        &oracle.gate,
        &oracle_repeat.gate,
    )?;
    compare_exact(
        "owner_grouped_oracle_up_repeat",
        &oracle.up,
        &oracle_repeat.up,
    )?;
    compare_exact(
        "owner_grouped_oracle_down_repeat",
        &oracle.down,
        &oracle_repeat.down,
    )?;

    let start = Instant::now();
    let mut prepared = runtime.prepare_step_grouped_expert_parallel_gate(
        resident,
        input,
        tokens,
        &selected,
        activation_limit,
    )?;
    let prepare_us = elapsed_us(start);
    if prepared.active_owners() != runtime.devices().len() {
        return Err(format!(
            "official owner-grouped fixture reached {} of {} ranks",
            prepared.active_owners(),
            runtime.devices().len()
        )
        .into());
    }
    let owner_pair_counts = prepared.owner_pair_counts();
    let profile = strict_bool("MEMRA_STEP_EP_OWNER_GROUPED_FP8_PROFILE")?;
    if profile {
        let result = unsafe { cudaProfilerStart() };
        if result != 0 {
            return Err(format!("cudaProfilerStart failed with CUDA error {result}").into());
        }
    }
    let grouped_runs = (|| {
        let start = Instant::now();
        let first = runtime.run_step_grouped_expert_parallel_gate(resident, &mut prepared)?;
        let grouped_first_us = elapsed_us(start);
        let start = Instant::now();
        let second = runtime.run_step_grouped_expert_parallel_gate(resident, &mut prepared)?;
        let grouped_repeat_us = elapsed_us(start);
        Ok::<_, Box<dyn std::error::Error>>((first, grouped_first_us, second, grouped_repeat_us))
    })();
    if profile {
        let result = unsafe { cudaProfilerStop() };
        if result != 0 {
            return Err(format!("cudaProfilerStop failed with CUDA error {result}").into());
        }
    }
    let (first, grouped_first_us, second, grouped_repeat_us) = grouped_runs?;

    compare_exact("owner_grouped_gate_oracle", &oracle.gate, &first.gate)?;
    compare_exact("owner_grouped_up_oracle", &oracle.up, &first.up)?;
    compare_exact("owner_grouped_down_oracle", &oracle.down, &first.down)?;
    compare_exact("owner_grouped_gate_repeat", &first.gate, &second.gate)?;
    compare_exact("owner_grouped_up_repeat", &first.up, &second.up)?;
    compare_exact("owner_grouped_down_repeat", &first.down, &second.down)?;

    let executions = 2;
    let projection_launches = prepared.active_owners() * 3 * executions;
    let quantize_launches = projection_launches;
    let activation_launches = prepared.active_owners() * executions;
    let oracle_projection_launches = prepared.pairs() * 3 * executions;
    let oracle_best_us = oracle_first_us.min(oracle_repeat_us);
    let grouped_best_us = grouped_first_us.min(grouped_repeat_us);
    println!(
        "STEP_OWNER_GROUPED_FP8_GATE_PASS layer={layer} tokens={} pairs={} \
         experts_per_token={} owners={} owner_pair_counts={owner_pair_counts:?} \
         activation_limit={} csr_uploads={} workspace_sets={} preparation_count=1 executions=2 \
         hot_device_allocations=0 \
         oracle_projection_kernel_launches={oracle_projection_launches} \
         grouped_quantize_kernel_launches={quantize_launches} \
         grouped_projection_kernel_launches={projection_launches} \
         grouped_activation_kernel_launches={activation_launches} \
         prepare_us={prepare_us:.3} oracle_first_us={oracle_first_us:.3} \
         oracle_repeat_us={oracle_repeat_us:.3} grouped_first_us={grouped_first_us:.3} \
         grouped_repeat_us={grouped_repeat_us:.3} best_speedup={:.3} \
         raw_bit_gate=true raw_bit_up=true raw_bit_down=true \
         dispatch=native-p2p expert_layout=expert-parallel output=canonical-pair-order \
         production_path=false routing=false combine=false tensor_parallel_claim=false",
        prepared.tokens(),
        prepared.pairs(),
        STEP_GROUPED_TOP_K,
        prepared.active_owners(),
        activation_limit.unwrap_or(0.0),
        prepared.active_owners() * 2,
        prepared.active_owners() * 3,
        oracle_best_us / grouped_best_us,
    );
    Ok(())
}

fn compare_projection_exact(
    label: &str,
    expected: &StepGroupedFp8ProjectionOutput,
    actual: &StepGroupedFp8ProjectionOutput,
) -> Result<(), String> {
    compare_exact(&format!("{label}_gate"), &expected.gate, &actual.gate)?;
    compare_exact(&format!("{label}_up"), &expected.up, &actual.up)?;
    compare_exact(&format!("{label}_down"), &expected.down, &actual.down)
}

#[allow(clippy::too_many_arguments)]
fn run_dynamic_owner_grouped_fp8_gate(
    runtime: &TpE4m3HostBounce,
    resident: &ResidentExpertParallel,
    gate_bank: &Fp8StackedNative<'_>,
    up_bank: &Fp8StackedNative<'_>,
    down_bank: &Fp8StackedNative<'_>,
    initial_input: &[f32],
    initial_tokens: usize,
    layer: usize,
    activation_limit: Option<f32>,
) -> Result<(), Box<dyn std::error::Error>> {
    let default_capacity = initial_tokens
        .checked_add(2)
        .ok_or("dynamic owner-grouped token capacity overflow")?;
    let capacity_tokens = std::env::var("MEMRA_STEP_EP_DYNAMIC_CAPACITY_TOKENS")
        .ok()
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(default_capacity);
    if capacity_tokens <= initial_tokens {
        return Err(format!(
            "dynamic owner-grouped capacity {capacity_tokens} must exceed initial tokens \
             {initial_tokens}"
        )
        .into());
    }
    let initial_selected = grouped_selected_routes(initial_tokens, gate_bank.n_expert)?;
    let dynamic_tokens = capacity_tokens;
    let dynamic_input = activations_with_salt(dynamic_tokens, gate_bank.in_f, layer + 17);
    let dynamic_selected = dynamic_grouped_selected_routes(
        dynamic_tokens,
        gate_bank.n_expert,
        runtime.devices().len(),
    )?;

    let start = Instant::now();
    let initial_oracle = reference_grouped_projections(
        runtime,
        gate_bank,
        up_bank,
        down_bank,
        initial_input,
        initial_tokens,
        &initial_selected,
        activation_limit,
    )?;
    let initial_oracle_us = elapsed_us(start);
    let start = Instant::now();
    let dynamic_oracle = reference_grouped_projections(
        runtime,
        gate_bank,
        up_bank,
        down_bank,
        &dynamic_input,
        dynamic_tokens,
        &dynamic_selected,
        activation_limit,
    )?;
    let dynamic_oracle_us = elapsed_us(start);

    let start = Instant::now();
    let mut prepared = runtime.prepare_step_grouped_expert_parallel_gate_with_capacity(
        resident,
        initial_input,
        initial_tokens,
        &initial_selected,
        activation_limit,
        capacity_tokens,
    )?;
    let prepare_us = elapsed_us(start);
    if prepared.active_owners() != runtime.devices().len() {
        return Err(format!(
            "dynamic owner-grouped initial fixture reached {} of {} ranks",
            prepared.active_owners(),
            runtime.devices().len()
        )
        .into());
    }
    let initial_owner_pair_counts = prepared.owner_pair_counts();
    let start = Instant::now();
    let initial_first = runtime.run_step_grouped_expert_parallel_gate(resident, &mut prepared)?;
    let initial_first_us = elapsed_us(start);
    compare_projection_exact(
        "dynamic_owner_initial_oracle",
        &initial_oracle,
        &initial_first,
    )?;

    let profile = strict_bool("MEMRA_STEP_EP_DYNAMIC_OWNER_GROUPED_FP8_PROFILE")?;
    if profile {
        let result = unsafe { cudaProfilerStart() };
        if result != 0 {
            return Err(format!("cudaProfilerStart failed with CUDA error {result}").into());
        }
    }
    let dynamic_runs = (|| {
        let start = Instant::now();
        runtime.refresh_step_grouped_expert_parallel_gate(
            resident,
            &mut prepared,
            &dynamic_input,
            dynamic_tokens,
            &dynamic_selected,
        )?;
        let dynamic_refresh_us = elapsed_us(start);
        if prepared.active_owners() != runtime.devices().len() {
            return Err(format!(
                "dynamic owner-grouped changed fixture reached {} of {} ranks",
                prepared.active_owners(),
                runtime.devices().len()
            )
            .into());
        }
        let dynamic_owner_pair_counts = prepared.owner_pair_counts();
        let start = Instant::now();
        let dynamic_first =
            runtime.run_step_grouped_expert_parallel_gate(resident, &mut prepared)?;
        let dynamic_first_us = elapsed_us(start);
        let start = Instant::now();
        let dynamic_repeat =
            runtime.run_step_grouped_expert_parallel_gate(resident, &mut prepared)?;
        let dynamic_repeat_us = elapsed_us(start);

        let start = Instant::now();
        runtime.refresh_step_grouped_expert_parallel_gate(
            resident,
            &mut prepared,
            initial_input,
            initial_tokens,
            &initial_selected,
        )?;
        let initial_refresh_us = elapsed_us(start);
        if prepared.active_owners() != runtime.devices().len() {
            return Err(format!(
                "dynamic owner-grouped restored fixture reached {} of {} ranks",
                prepared.active_owners(),
                runtime.devices().len()
            )
            .into());
        }
        let start = Instant::now();
        let initial_restored =
            runtime.run_step_grouped_expert_parallel_gate(resident, &mut prepared)?;
        let initial_restored_us = elapsed_us(start);
        Ok::<_, Box<dyn std::error::Error>>((
            dynamic_refresh_us,
            dynamic_owner_pair_counts,
            dynamic_first,
            dynamic_first_us,
            dynamic_repeat,
            dynamic_repeat_us,
            initial_refresh_us,
            initial_restored,
            initial_restored_us,
        ))
    })();
    if profile {
        let result = unsafe { cudaProfilerStop() };
        if result != 0 {
            return Err(format!("cudaProfilerStop failed with CUDA error {result}").into());
        }
    }
    let (
        dynamic_refresh_us,
        dynamic_owner_pair_counts,
        dynamic_first,
        dynamic_first_us,
        dynamic_repeat,
        dynamic_repeat_us,
        initial_refresh_us,
        initial_restored,
        initial_restored_us,
    ) = dynamic_runs?;

    compare_projection_exact(
        "dynamic_owner_changed_oracle",
        &dynamic_oracle,
        &dynamic_first,
    )?;
    compare_projection_exact(
        "dynamic_owner_changed_repeat",
        &dynamic_first,
        &dynamic_repeat,
    )?;
    compare_projection_exact(
        "dynamic_owner_restored_oracle",
        &initial_oracle,
        &initial_restored,
    )?;
    compare_projection_exact(
        "dynamic_owner_restored_repeat",
        &initial_first,
        &initial_restored,
    )?;

    let ranks = runtime.devices().len();
    let profiled_owner_executions = ranks * 3;
    let profiled_projection_launches = profiled_owner_executions * 3;
    let profiled_quantize_launches = profiled_projection_launches;
    let profiled_activation_launches = profiled_owner_executions;
    let profiled_total_kernels =
        profiled_projection_launches + profiled_quantize_launches + profiled_activation_launches;
    let initial_best_us = initial_first_us.min(initial_restored_us);
    let dynamic_best_us = dynamic_first_us.min(dynamic_repeat_us);
    println!(
        "STEP_DYNAMIC_OWNER_GROUPED_FP8_GATE_PASS layer={layer} initial_tokens={initial_tokens} \
         dynamic_tokens={dynamic_tokens} capacity_tokens={} initial_pairs={} dynamic_pairs={} \
         experts_per_token={} owners={ranks} \
         initial_owner_pair_counts={initial_owner_pair_counts:?} \
         dynamic_owner_pair_counts={dynamic_owner_pair_counts:?} \
         activation_limit={} preparation_count=1 input_refreshes=2 route_refreshes=2 \
         profiled_executions=3 total_executions=4 hot_device_allocations=0 \
         profiled_grouped_quantize_kernel_launches={profiled_quantize_launches} \
         profiled_grouped_projection_kernel_launches={profiled_projection_launches} \
         profiled_grouped_activation_kernel_launches={profiled_activation_launches} \
         profiled_total_kernel_launches={profiled_total_kernels} \
         prepare_us={prepare_us:.3} dynamic_refresh_us={dynamic_refresh_us:.3} \
         initial_refresh_us={initial_refresh_us:.3} initial_oracle_us={initial_oracle_us:.3} \
         dynamic_oracle_us={dynamic_oracle_us:.3} initial_first_us={initial_first_us:.3} \
         initial_restored_us={initial_restored_us:.3} dynamic_first_us={dynamic_first_us:.3} \
         dynamic_repeat_us={dynamic_repeat_us:.3} initial_best_speedup={:.3} \
         dynamic_best_speedup={:.3} raw_bit_gate=true raw_bit_up=true raw_bit_down=true \
         dispatch=native-p2p expert_layout=expert-parallel output=canonical-pair-order \
         production_path=false production_routing=false combine=false tensor_parallel_claim=false",
        prepared.max_tokens(),
        initial_tokens * STEP_GROUPED_TOP_K,
        dynamic_tokens * STEP_GROUPED_TOP_K,
        STEP_GROUPED_TOP_K,
        activation_limit.unwrap_or(0.0),
        initial_oracle_us / initial_best_us,
        dynamic_oracle_us / dynamic_best_us,
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_dynamic_owner_grouped_fp8_combine_gate(
    runtime: &TpE4m3HostBounce,
    resident: &ResidentExpertParallel,
    gate_bank: &Fp8StackedNative<'_>,
    up_bank: &Fp8StackedNative<'_>,
    down_bank: &Fp8StackedNative<'_>,
    initial_input: &[f32],
    initial_tokens: usize,
    layer: usize,
    activation_limit: Option<f32>,
) -> Result<(), Box<dyn std::error::Error>> {
    let default_capacity = initial_tokens
        .checked_add(2)
        .ok_or("dynamic owner-grouped combine token capacity overflow")?;
    let capacity_tokens = std::env::var("MEMRA_STEP_EP_DYNAMIC_CAPACITY_TOKENS")
        .ok()
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(default_capacity);
    if capacity_tokens <= initial_tokens {
        return Err(format!(
            "dynamic owner-grouped combine capacity {capacity_tokens} must exceed initial tokens \
             {initial_tokens}"
        )
        .into());
    }
    let initial_selected = grouped_selected_routes(initial_tokens, gate_bank.n_expert)?;
    let initial_weights = grouped_route_weights(initial_tokens, 0);
    let dynamic_tokens = capacity_tokens;
    let dynamic_input = activations_with_salt(dynamic_tokens, gate_bank.in_f, layer + 17);
    let dynamic_selected = dynamic_grouped_selected_routes(
        dynamic_tokens,
        gate_bank.n_expert,
        runtime.devices().len(),
    )?;
    let dynamic_weights = grouped_route_weights(dynamic_tokens, 3);

    let start = Instant::now();
    let initial_oracle_projection = reference_grouped_projections(
        runtime,
        gate_bank,
        up_bank,
        down_bank,
        initial_input,
        initial_tokens,
        &initial_selected,
        activation_limit,
    )?;
    let initial_oracle = combine_down_rows_host(
        &initial_oracle_projection.down,
        &initial_weights,
        initial_tokens,
        gate_bank.in_f,
    )?;
    let initial_oracle_us = elapsed_us(start);
    let start = Instant::now();
    let dynamic_oracle_projection = reference_grouped_projections(
        runtime,
        gate_bank,
        up_bank,
        down_bank,
        &dynamic_input,
        dynamic_tokens,
        &dynamic_selected,
        activation_limit,
    )?;
    let dynamic_oracle = combine_down_rows_host(
        &dynamic_oracle_projection.down,
        &dynamic_weights,
        dynamic_tokens,
        gate_bank.in_f,
    )?;
    let dynamic_oracle_us = elapsed_us(start);

    let start = Instant::now();
    let mut prepared = runtime.prepare_step_grouped_expert_parallel_gate_with_capacity(
        resident,
        initial_input,
        initial_tokens,
        &initial_selected,
        activation_limit,
        capacity_tokens,
    )?;
    let mut combine =
        runtime.prepare_step_grouped_expert_parallel_combine(&prepared, &initial_weights)?;
    let prepare_us = elapsed_us(start);
    if prepared.active_owners() != runtime.devices().len()
        || combine.owner_pair_counts() != prepared.owner_pair_counts()
    {
        return Err("dynamic owner-grouped combine initial owner geometry differs".into());
    }
    let initial_owner_pair_counts = prepared.owner_pair_counts();
    let start = Instant::now();
    let initial_projection =
        runtime.run_step_grouped_expert_parallel_gate(resident, &mut prepared)?;
    let initial_projection_us = elapsed_us(start);
    compare_projection_exact(
        "dynamic_combine_initial_projection",
        &initial_oracle_projection,
        &initial_projection,
    )?;
    let start = Instant::now();
    let initial_first =
        runtime.run_step_grouped_expert_parallel_combine(&prepared, &mut combine)?;
    let initial_combine_first_us = elapsed_us(start);
    let start = Instant::now();
    let initial_repeat =
        runtime.run_step_grouped_expert_parallel_combine(&prepared, &mut combine)?;
    let initial_combine_repeat_us = elapsed_us(start);
    compare_exact(
        "dynamic_combine_initial_oracle",
        &initial_oracle,
        &initial_first,
    )?;
    compare_exact(
        "dynamic_combine_initial_repeat",
        &initial_first,
        &initial_repeat,
    )?;

    let profile = strict_bool("MEMRA_STEP_EP_DYNAMIC_OWNER_GROUPED_FP8_COMBINE_PROFILE")?;
    if profile {
        let result = unsafe { cudaProfilerStart() };
        if result != 0 {
            return Err(format!("cudaProfilerStart failed with CUDA error {result}").into());
        }
    }
    let dynamic_runs = (|| {
        let start = Instant::now();
        runtime.refresh_step_grouped_expert_parallel_gate(
            resident,
            &mut prepared,
            &dynamic_input,
            dynamic_tokens,
            &dynamic_selected,
        )?;
        let dynamic_projection_refresh_us = elapsed_us(start);
        let start = Instant::now();
        runtime.refresh_step_grouped_expert_parallel_combine(
            &prepared,
            &mut combine,
            &dynamic_weights,
        )?;
        let dynamic_combine_refresh_us = elapsed_us(start);
        if prepared.active_owners() != runtime.devices().len()
            || combine.owner_pair_counts() != prepared.owner_pair_counts()
        {
            return Err("dynamic owner-grouped combine changed owner geometry differs".into());
        }
        let dynamic_owner_pair_counts = prepared.owner_pair_counts();
        let start = Instant::now();
        let dynamic_projection =
            runtime.run_step_grouped_expert_parallel_gate(resident, &mut prepared)?;
        let dynamic_projection_us = elapsed_us(start);
        let start = Instant::now();
        let dynamic_first =
            runtime.run_step_grouped_expert_parallel_combine(&prepared, &mut combine)?;
        let dynamic_combine_first_us = elapsed_us(start);
        let start = Instant::now();
        let dynamic_repeat =
            runtime.run_step_grouped_expert_parallel_combine(&prepared, &mut combine)?;
        let dynamic_combine_repeat_us = elapsed_us(start);

        let start = Instant::now();
        runtime.refresh_step_grouped_expert_parallel_gate(
            resident,
            &mut prepared,
            initial_input,
            initial_tokens,
            &initial_selected,
        )?;
        let initial_projection_refresh_us = elapsed_us(start);
        let start = Instant::now();
        runtime.refresh_step_grouped_expert_parallel_combine(
            &prepared,
            &mut combine,
            &initial_weights,
        )?;
        let initial_combine_refresh_us = elapsed_us(start);
        let start = Instant::now();
        let initial_restored_projection =
            runtime.run_step_grouped_expert_parallel_gate(resident, &mut prepared)?;
        let initial_restored_projection_us = elapsed_us(start);
        let start = Instant::now();
        let initial_restored =
            runtime.run_step_grouped_expert_parallel_combine(&prepared, &mut combine)?;
        let initial_restored_combine_us = elapsed_us(start);
        Ok::<_, Box<dyn std::error::Error>>((
            dynamic_projection_refresh_us,
            dynamic_combine_refresh_us,
            dynamic_owner_pair_counts,
            dynamic_projection,
            dynamic_projection_us,
            dynamic_first,
            dynamic_combine_first_us,
            dynamic_repeat,
            dynamic_combine_repeat_us,
            initial_projection_refresh_us,
            initial_combine_refresh_us,
            initial_restored_projection,
            initial_restored_projection_us,
            initial_restored,
            initial_restored_combine_us,
        ))
    })();
    if profile {
        let result = unsafe { cudaProfilerStop() };
        if result != 0 {
            return Err(format!("cudaProfilerStop failed with CUDA error {result}").into());
        }
    }
    let (
        dynamic_projection_refresh_us,
        dynamic_combine_refresh_us,
        dynamic_owner_pair_counts,
        dynamic_projection,
        dynamic_projection_us,
        dynamic_first,
        dynamic_combine_first_us,
        dynamic_repeat,
        dynamic_combine_repeat_us,
        initial_projection_refresh_us,
        initial_combine_refresh_us,
        initial_restored_projection,
        initial_restored_projection_us,
        initial_restored,
        initial_restored_combine_us,
    ) = dynamic_runs?;

    compare_projection_exact(
        "dynamic_combine_changed_projection",
        &dynamic_oracle_projection,
        &dynamic_projection,
    )?;
    compare_exact(
        "dynamic_combine_changed_oracle",
        &dynamic_oracle,
        &dynamic_first,
    )?;
    compare_exact(
        "dynamic_combine_changed_repeat",
        &dynamic_first,
        &dynamic_repeat,
    )?;
    compare_projection_exact(
        "dynamic_combine_restored_projection",
        &initial_oracle_projection,
        &initial_restored_projection,
    )?;
    compare_exact(
        "dynamic_combine_restored_oracle",
        &initial_oracle,
        &initial_restored,
    )?;
    compare_exact(
        "dynamic_combine_restored_repeat",
        &initial_first,
        &initial_restored,
    )?;

    let ranks = runtime.devices().len();
    let profiled_projection_executions = 2;
    let profiled_owner_projection_executions = ranks * profiled_projection_executions;
    let profiled_projection_launches = profiled_owner_projection_executions * 3;
    let profiled_quantize_launches = profiled_projection_launches;
    let profiled_activation_launches = profiled_owner_projection_executions;
    let profiled_combine_executions = 3;
    let profiled_scatter_launches = ranks * profiled_combine_executions;
    let profiled_reduce_launches = profiled_combine_executions;
    let profiled_total_kernels = profiled_projection_launches
        + profiled_quantize_launches
        + profiled_activation_launches
        + profiled_scatter_launches
        + profiled_reduce_launches;
    let initial_combine_best_us = initial_combine_first_us.min(initial_combine_repeat_us);
    let dynamic_combine_best_us = dynamic_combine_first_us.min(dynamic_combine_repeat_us);
    let initial_grouped_best_us = initial_projection_us + initial_combine_best_us;
    let dynamic_grouped_best_us = dynamic_projection_us + dynamic_combine_best_us;
    println!(
        "STEP_DYNAMIC_OWNER_GROUPED_FP8_COMBINE_GATE_PASS layer={layer} \
         initial_tokens={initial_tokens} dynamic_tokens={dynamic_tokens} capacity_tokens={} \
         initial_pairs={} dynamic_pairs={} experts_per_token={} owners={ranks} \
         initial_owner_pair_counts={initial_owner_pair_counts:?} \
         dynamic_owner_pair_counts={dynamic_owner_pair_counts:?} activation_limit={} \
         projection_preparation_count=1 combine_preparation_count=1 projection_refreshes=2 \
         combine_refreshes=2 profiled_projection_executions={profiled_projection_executions} \
         profiled_combine_executions={profiled_combine_executions} hot_device_allocations=0 \
         profiled_grouped_quantize_kernel_launches={profiled_quantize_launches} \
         profiled_grouped_projection_kernel_launches={profiled_projection_launches} \
         profiled_grouped_activation_kernel_launches={profiled_activation_launches} \
         profiled_combine_scatter_kernel_launches={profiled_scatter_launches} \
         profiled_combine_reduce_kernel_launches={profiled_reduce_launches} \
         profiled_total_kernel_launches={profiled_total_kernels} prepare_us={prepare_us:.3} \
         dynamic_projection_refresh_us={dynamic_projection_refresh_us:.3} \
         dynamic_combine_refresh_us={dynamic_combine_refresh_us:.3} \
         initial_projection_refresh_us={initial_projection_refresh_us:.3} \
         initial_combine_refresh_us={initial_combine_refresh_us:.3} \
         initial_oracle_us={initial_oracle_us:.3} dynamic_oracle_us={dynamic_oracle_us:.3} \
         initial_projection_us={initial_projection_us:.3} \
         initial_combine_first_us={initial_combine_first_us:.3} \
         initial_combine_repeat_us={initial_combine_repeat_us:.3} \
         dynamic_projection_us={dynamic_projection_us:.3} \
         dynamic_combine_first_us={dynamic_combine_first_us:.3} \
         dynamic_combine_repeat_us={dynamic_combine_repeat_us:.3} \
         initial_restored_projection_us={initial_restored_projection_us:.3} \
         initial_restored_combine_us={initial_restored_combine_us:.3} \
         initial_best_speedup={:.3} dynamic_best_speedup={:.3} \
         raw_bit_gate=true raw_bit_up=true raw_bit_down=true raw_bit_combine=true \
         combine_numeric=separate-rn-mul-add combine_order=canonical-top8 \
         dispatch=native-p2p expert_layout=expert-parallel \
         combine_output=root-device-readback \
         production_path=false production_routing=false tensor_parallel_claim=false",
        prepared.max_tokens(),
        initial_tokens * STEP_GROUPED_TOP_K,
        dynamic_tokens * STEP_GROUPED_TOP_K,
        STEP_GROUPED_TOP_K,
        activation_limit.unwrap_or(0.0),
        initial_oracle_us / initial_grouped_best_us,
        dynamic_oracle_us / dynamic_grouped_best_us,
    );
    Ok(())
}

fn execute_owner_grouped_fp8_device_handoff(
    runtime: &TpE4m3HostBounce,
    resident: &ResidentExpertParallel,
    prepared: &mut memra_engine::tp::PreparedStepGroupedExpertParallelGate,
    combine: &mut memra_engine::tp::PreparedPeerWeightedRouteCombine,
) -> Result<(), Box<dyn std::error::Error>> {
    runtime.execute_step_grouped_expert_parallel_gate(resident, prepared)?;
    runtime.execute_step_grouped_expert_parallel_combine(prepared, combine)?;
    runtime.broadcast_step_grouped_expert_parallel_combine(prepared, combine)
}

fn compare_distributed_exact(
    label: &str,
    expected: &[f32],
    outputs: &[Vec<f32>],
) -> Result<(), String> {
    if outputs.is_empty() {
        return Err(format!("{label}: no distributed rank outputs"));
    }
    for (rank, output) in outputs.iter().enumerate() {
        compare_exact(&format!("{label}_rank{rank}"), expected, output)?;
    }
    Ok(())
}

struct ResidentTransitionAttention {
    q: ResidentBf16ColumnParallel,
    k: ResidentBf16ColumnParallel,
    v: ResidentBf16ColumnParallel,
    gate: ResidentBf16ColumnParallel,
    o: ResidentStepBf16RowParallel,
}

struct AttentionToExpertTransition {
    attention: Vec<f32>,
    residual: ResidentReplicatedDeviceRows,
    post_attention_norm: ResidentReplicatedDeviceRows,
}

struct PreparedAttentionTransitionGate {
    canonical_runtime: TpE4m3HostBounce,
    resident: ResidentTransitionAttention,
    canonical_resident: ResidentTransitionAttention,
    input_norm: Vec<f32>,
    post_attention_norm: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    rope_factors: Option<Vec<f32>>,
    positions: Vec<i32>,
    geometry: LayerGeometry,
    rms_eps: f32,
}

fn prepare_attention_transition_gate(
    source: &SafetensorsSource,
    runtime: &TpE4m3HostBounce,
    hidden: usize,
    tokens: usize,
    layer: usize,
) -> Result<PreparedAttentionTransitionGate, Box<dyn std::error::Error>> {
    let config = source.config();
    let geometry = config
        .layer_geometry(layer as u32)
        .ok_or_else(|| format!("Step layer {layer} has no attention geometry"))?;
    let q_width = geometry.n_head as usize * geometry.head_dim_k as usize;
    let kv_width = geometry.n_head_kv as usize * geometry.head_dim_k as usize;
    let prefix = format!("blk.{layer}");
    let q_matrix = load_bf16_matrix(source, &format!("{prefix}.attn_q.weight"), hidden, q_width)?;
    let k_matrix = load_bf16_matrix(source, &format!("{prefix}.attn_k.weight"), hidden, kv_width)?;
    let v_matrix = load_bf16_matrix(source, &format!("{prefix}.attn_v.weight"), hidden, kv_width)?;
    let gate_matrix = load_bf16_matrix(
        source,
        &format!("{prefix}.attn_gate.weight"),
        hidden,
        geometry.n_head as usize,
    )?;
    let o_matrix = load_bf16_matrix(
        source,
        &format!("{prefix}.attn_output.weight"),
        q_width,
        hidden,
    )?;
    let canonical_runtime = TpE4m3HostBounce::new_single_rank_oracle(runtime.devices()[0])?;
    let resident = ResidentTransitionAttention {
        q: runtime.upload_step_bf16_column_parallel(q_matrix.view())?,
        k: runtime.upload_step_bf16_column_parallel(k_matrix.view())?,
        v: runtime.upload_step_bf16_column_parallel(v_matrix.view())?,
        gate: runtime.upload_step_bf16_column_parallel(gate_matrix.view())?,
        o: runtime.upload_step_bf16_row_parallel(o_matrix.view())?,
    };
    let canonical_resident = ResidentTransitionAttention {
        q: canonical_runtime.upload_step_bf16_column_parallel(q_matrix.view())?,
        k: canonical_runtime.upload_step_bf16_column_parallel(k_matrix.view())?,
        v: canonical_runtime.upload_step_bf16_column_parallel(v_matrix.view())?,
        gate: canonical_runtime.upload_step_bf16_column_parallel(gate_matrix.view())?,
        o: canonical_runtime.upload_step_bf16_row_parallel(o_matrix.view())?,
    };
    let root = runtime
        .rank_engine(0)
        .ok_or("attention transition runtime has no root rank")?;
    let input_norm = load_f32_vector(root, source, &format!("{prefix}.attn_norm.weight"), hidden)?;
    let post_attention_norm =
        load_f32_vector(root, source, &format!("{prefix}.ffn_norm.weight"), hidden)?;
    let q_norm = load_f32_vector(
        root,
        source,
        &format!("{prefix}.attn_q_norm.weight"),
        geometry.head_dim_k as usize,
    )?;
    let k_norm = load_f32_vector(
        root,
        source,
        &format!("{prefix}.attn_k_norm.weight"),
        geometry.head_dim_k as usize,
    )?;
    let rope_factors = if geometry.rope_factors && source.find("rope_freqs.weight").is_some() {
        Some(load_f32_vector(
            root,
            source,
            "rope_freqs.weight",
            geometry.n_rot as usize / 2,
        )?)
    } else {
        None
    };
    let positions = (0..tokens)
        .map(i32::try_from)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(PreparedAttentionTransitionGate {
        canonical_runtime,
        resident,
        canonical_resident,
        input_norm,
        post_attention_norm,
        q_norm,
        k_norm,
        rope_factors,
        positions,
        geometry,
        rms_eps: config.rms_eps,
    })
}

fn run_attention_transition_pair(
    runtime: &TpE4m3HostBounce,
    prepared: &PreparedAttentionTransitionGate,
    input: &[f32],
    tokens: usize,
    hidden: usize,
) -> Result<(AttentionToExpertTransition, AttentionToExpertTransition), Box<dyn std::error::Error>>
{
    let canonical_hidden = prepared
        .canonical_runtime
        .upload_replicated_device_rows(input, tokens, hidden)?;
    let canonical = run_attention_to_expert_transition(
        &prepared.canonical_runtime,
        &prepared.canonical_resident,
        &canonical_hidden,
        &prepared.input_norm,
        &prepared.post_attention_norm,
        &prepared.q_norm,
        &prepared.k_norm,
        prepared.rope_factors.as_deref(),
        &prepared.positions,
        prepared.geometry,
        prepared.rms_eps,
    )?;
    let native_hidden = runtime.upload_replicated_device_rows(input, tokens, hidden)?;
    let native = run_attention_to_expert_transition(
        runtime,
        &prepared.resident,
        &native_hidden,
        &prepared.input_norm,
        &prepared.post_attention_norm,
        &prepared.q_norm,
        &prepared.k_norm,
        prepared.rope_factors.as_deref(),
        &prepared.positions,
        prepared.geometry,
        prepared.rms_eps,
    )?;
    Ok((canonical, native))
}

#[allow(clippy::too_many_arguments)]
fn run_attention_to_expert_transition(
    runtime: &TpE4m3HostBounce,
    resident: &ResidentTransitionAttention,
    hidden: &ResidentReplicatedDeviceRows,
    input_norm: &[f32],
    post_attention_norm: &[f32],
    q_norm: &[f32],
    k_norm: &[f32],
    rope_factors: Option<&[f32]>,
    positions: &[i32],
    geometry: LayerGeometry,
    rms_eps: f32,
) -> Result<AttentionToExpertTransition, Box<dyn std::error::Error>> {
    let tp = runtime.devices().len();
    let heads = geometry.n_head as usize;
    let kv_heads = geometry.n_head_kv as usize;
    let head_dim = geometry.head_dim_k as usize;
    let tokens = hidden.tokens();
    if hidden.ranks() != tp
        || hidden.width() != input_norm.len()
        || post_attention_norm.len() != hidden.width()
        || positions.len() != tokens
        || heads % tp != 0
        || kv_heads % tp != 0
        || q_norm.len() != head_dim
        || k_norm.len() != head_dim
    {
        return Err(format!(
            "attention-to-expert geometry hidden={}x{} ranks={} heads={heads}/{kv_heads} \
             head_dim={head_dim} positions={} q/k_norm={}/{} tp={tp}",
            hidden.tokens(),
            hidden.width(),
            hidden.ranks(),
            positions.len(),
            q_norm.len(),
            k_norm.len(),
        )
        .into());
    }
    let local_heads = heads / tp;
    let local_kv_heads = kv_heads / tp;
    let normalized = runtime.rms_norm_replicated_device_rows(hidden, input_norm, rms_eps)?;
    let q_shards =
        runtime.bf16_column_parallel_resident_replicated_device_shards(&resident.q, &normalized)?;
    let k_shards =
        runtime.bf16_column_parallel_resident_replicated_device_shards(&resident.k, &normalized)?;
    let v_shards =
        runtime.bf16_column_parallel_resident_replicated_device_shards(&resident.v, &normalized)?;
    let gate_shards = runtime
        .bf16_column_parallel_resident_replicated_device_shards(&resident.gate, &normalized)?;

    let mut attention_shards = Vec::<CudaSlice<f32>>::with_capacity(tp);
    for rank in 0..tp {
        let engine = runtime
            .rank_engine(rank)
            .ok_or_else(|| format!("attention-to-expert TP rank {rank} has no engine"))?;
        let _main = engine.gpu.enter_main()?;
        let q_norm_weight = engine.htod(q_norm)?;
        let k_norm_weight = engine.htod(k_norm)?;
        let positions = engine.htod_i32(positions)?;
        let rope_factors = rope_factors
            .map(|factors| engine.htod(factors))
            .transpose()?;
        let mut q = engine.uninit(tokens * local_heads * head_dim)?;
        engine.rms_norm(
            &q_shards[rank],
            &q_norm_weight,
            &mut q,
            head_dim,
            local_heads * tokens,
            rms_eps,
        )?;
        let mut k = engine.uninit(tokens * local_kv_heads * head_dim)?;
        engine.rms_norm(
            &k_shards[rank],
            &k_norm_weight,
            &mut k,
            head_dim,
            local_kv_heads * tokens,
            rms_eps,
        )?;
        engine.rope_neox2(
            &mut q,
            &mut k,
            &positions,
            head_dim,
            geometry.n_rot as usize,
            local_heads,
            local_kv_heads,
            tokens,
            geometry.rope_base,
            1.0,
            rope_factors.as_ref(),
        )?;
        let mut attention = engine.uninit(tokens * local_heads * head_dim)?;
        let window = geometry.window.map(|window| window as usize);
        if let Some(window) = window.filter(|&window| tokens > window) {
            engine.sdpa_naive_w(
                &q,
                &k,
                &v_shards[rank],
                &mut attention,
                head_dim,
                local_heads,
                local_kv_heads,
                tokens,
                tokens,
                geometry.attention_scale(),
                true,
                window,
            )?;
        } else {
            engine.fa_prefill(
                &q,
                &k,
                &v_shards[rank],
                &mut attention,
                head_dim,
                local_heads,
                local_kv_heads,
                tokens,
                tokens,
                geometry.attention_scale(),
                true,
            )?;
        }
        let mut gated = engine.uninit(tokens * local_heads * head_dim)?;
        engine.attn_head_gate(
            &attention,
            &gate_shards[rank],
            &mut gated,
            None,
            head_dim,
            local_heads,
            tokens,
        )?;
        attention_shards.push(gated);
    }

    let o = runtime.step_bf16_row_parallel_resident_replicated_device(
        &resident.o,
        &attention_shards,
        tokens,
    )?;
    let attention =
        runtime.gather_native_column_shards(&attention_shards, tokens, local_heads * head_dim)?;
    let (residual, post_attention_norm) =
        runtime.add_rms_norm_replicated_device_rows(hidden, &o, post_attention_norm, rms_eps)?;
    Ok(AttentionToExpertTransition {
        attention,
        residual,
        post_attention_norm,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_attention_to_routed_expert_gate(
    source: &SafetensorsSource,
    runtime: &TpE4m3HostBounce,
    resident_experts: &ResidentExpertParallel,
    gate_bank: &Fp8StackedNative<'_>,
    up_bank: &Fp8StackedNative<'_>,
    down_bank: &Fp8StackedNative<'_>,
    input: &[f32],
    tokens: usize,
    layer: usize,
    activation_limit: Option<f32>,
) -> Result<(), Box<dyn std::error::Error>> {
    let hidden = gate_bank.in_f;
    let transition = prepare_attention_transition_gate(source, runtime, hidden, tokens, layer)?;
    let (canonical, native) =
        run_attention_transition_pair(runtime, &transition, input, tokens, hidden)?;
    let canonical_residual = transition
        .canonical_runtime
        .collect_replicated_device_rows(&canonical.residual)?
        .remove(0);
    let canonical_post_norm = transition
        .canonical_runtime
        .collect_replicated_device_rows(&canonical.post_attention_norm)?
        .remove(0);
    compare_exact(
        "attention_to_ep_canonical_attention",
        &canonical.attention,
        &native.attention,
    )?;
    let native_residual = runtime.collect_replicated_device_rows(&native.residual)?;
    compare_distributed_exact(
        "attention_to_ep_residual",
        &canonical_residual,
        &native_residual,
    )?;
    let native_post_norm = runtime.collect_replicated_device_rows(&native.post_attention_norm)?;
    compare_distributed_exact(
        "attention_to_ep_post_norm",
        &canonical_post_norm,
        &native_post_norm,
    )?;

    let selected = grouped_selected_routes(tokens, gate_bank.n_expert)?;
    let route_weights = grouped_route_weights(tokens, layer);
    let oracle_projection = reference_grouped_projections(
        runtime,
        gate_bank,
        up_bank,
        down_bank,
        &canonical_post_norm,
        tokens,
        &selected,
        activation_limit,
    )?;
    let oracle = combine_down_rows_host(&oracle_projection.down, &route_weights, tokens, hidden)?;
    let mut prepared = runtime.prepare_step_grouped_expert_parallel_gate(
        resident_experts,
        &canonical_post_norm,
        tokens,
        &selected,
        activation_limit,
    )?;
    let prepared_generation = prepared.generation();
    let mut combine =
        runtime.prepare_step_grouped_expert_parallel_combine(&prepared, &route_weights)?;
    runtime.refresh_step_grouped_expert_parallel_inputs_from_replicated(
        resident_experts,
        &mut prepared,
        &native.post_attention_norm,
    )?;
    if prepared.generation() <= prepared_generation {
        return Err("attention-to-expert device input generation did not advance".into());
    }
    runtime.refresh_step_grouped_expert_parallel_combine(
        &prepared,
        &mut combine,
        &route_weights,
    )?;
    runtime.execute_step_grouped_expert_parallel_gate(resident_experts, &mut prepared)?;
    let projection = runtime.collect_step_grouped_expert_parallel_gate(&prepared)?;
    compare_projection_exact(
        "attention_to_ep_routed_projection",
        &oracle_projection,
        &projection,
    )?;
    runtime.execute_step_grouped_expert_parallel_combine(&prepared, &mut combine)?;
    runtime.broadcast_step_grouped_expert_parallel_combine(&prepared, &mut combine)?;
    let outputs = runtime.collect_step_grouped_expert_parallel_broadcast(&prepared, &combine)?;
    compare_distributed_exact("attention_to_ep_routed_combine", &oracle, &outputs)?;

    println!(
        "STEP_ATTENTION_TO_ROUTED_EP_GATE_PASS layer={layer} tokens={tokens} tp={} \
         query_heads={} kv_heads={} head_dim={} attention_input_norm=rank-local \
         qkv_tensor_parallel=true head_gate_tensor_parallel=true qk_norm_rank_local=true \
         rope_rank_local=true attention_tensor_parallel=true o_tensor_parallel=true \
         o_reduction=global-tp8-block-order post_attention_residual=rank-local \
         post_attention_norm=rank-local ep_input_transport=device-to-device \
         routed_expert_layout=expert-parallel owners={} routes=host-fixture \
         raw_bit_attention=true raw_bit_residual=true raw_bit_post_norm=true \
         raw_bit_gate=true raw_bit_up=true raw_bit_down=true raw_bit_combine=true \
         official_order=attention-then-routed-expert router_included=false \
         shared_expert_included=false final_residual_included=false \
         full_layer_claim=false tensor_parallel_claim=attention-only \
        production_path=false performance_claim=false",
        runtime.devices().len(),
        transition.geometry.n_head,
        transition.geometry.n_head_kv,
        transition.geometry.head_dim_k,
        runtime.devices().len(),
    );
    Ok(())
}

fn compare_route_indices(label: &str, expected: &[u32], actual: &[u32]) -> Result<(), String> {
    if expected.len() != actual.len() {
        return Err(format!(
            "{label}: route count {} != {}",
            actual.len(),
            expected.len()
        ));
    }
    let mismatches = expected.iter().zip(actual).filter(|(a, b)| a != b).count();
    println!(
        "EP_ROUTE_EXACT label={label} index_mismatches={mismatches}/{}",
        expected.len()
    );
    if mismatches != 0 {
        return Err(format!(
            "{label}: selected experts differ from the reference"
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_full_decoder_layer_gate(
    source: &SafetensorsSource,
    runtime: &TpE4m3HostBounce,
    resident_experts: &ResidentExpertParallel,
    gate_bank: &Fp8StackedNative<'_>,
    up_bank: &Fp8StackedNative<'_>,
    down_bank: &Fp8StackedNative<'_>,
    input: &[f32],
    tokens: usize,
    layer: usize,
    routed_activation_limit: Option<f32>,
) -> Result<(), Box<dyn std::error::Error>> {
    let config = source.config();
    let moe = config
        .moe
        .as_ref()
        .ok_or("official Step full-layer gate requires MoE geometry")?;
    let expert_count = moe.expert_count as usize;
    let experts_per_token = moe.expert_used_count as usize;
    if expert_count != gate_bank.n_expert
        || experts_per_token != STEP_GROUPED_TOP_K
        || gate_bank.in_f != up_bank.in_f
        || gate_bank.out_f != up_bank.out_f
        || down_bank.in_f != gate_bank.out_f
        || down_bank.out_f != gate_bank.in_f
    {
        return Err(format!(
            "official Step full-layer MoE geometry experts={expert_count} top_k={experts_per_token} \
             gate={}x{} up={}x{} down={}x{}",
            gate_bank.out_f,
            gate_bank.in_f,
            up_bank.out_f,
            up_bank.in_f,
            down_bank.out_f,
            down_bank.in_f,
        )
        .into());
    }
    let (router_scale, route_norm) = config
        .sigmoid_router()
        .ok_or("official Step full-layer gate requires sigmoid routing")?;
    let hidden = gate_bank.in_f;
    let shared_width = moe.expert_shared_ff_length as usize;
    let prefix = format!("blk.{layer}");
    let router_matrix = load_bf16_matrix(
        source,
        &format!("{prefix}.ffn_gate_inp.weight"),
        hidden,
        expert_count,
    )?;
    let shared_gate = load_bf16_matrix(
        source,
        &format!("{prefix}.ffn_gate_shexp.weight"),
        hidden,
        shared_width,
    )?;
    let shared_up = load_bf16_matrix(
        source,
        &format!("{prefix}.ffn_up_shexp.weight"),
        hidden,
        shared_width,
    )?;
    let shared_down = load_bf16_matrix(
        source,
        &format!("{prefix}.ffn_down_shexp.weight"),
        shared_width,
        hidden,
    )?;
    let root = runtime
        .rank_engine(0)
        .ok_or("official Step full-layer runtime has no root rank")?;
    let router_bias = load_f32_vector(
        root,
        source,
        &format!("{prefix}.exp_probs_b.bias"),
        expert_count,
    )?;
    // This binary is the step35 EP/TP cell; its expert kernels encode step35's POST clamp.
    // A PRE-clamped arch (glm5_next) has no arm here — refuse by name, never substitute.
    let shared_activation_limit = post_clamp_only(config.clamp_shexp_at(layer as u32), layer)?;

    let transition = prepare_attention_transition_gate(source, runtime, hidden, tokens, layer)?;
    let router = runtime.upload_sigmoid_topk_router(
        router_matrix.view(),
        &router_bias,
        None,
        experts_per_token,
        router_scale,
        route_norm,
    )?;
    let canonical_router = transition.canonical_runtime.upload_sigmoid_topk_router(
        router_matrix.view(),
        &router_bias,
        None,
        experts_per_token,
        router_scale,
        route_norm,
    )?;
    let shared = runtime.upload_replicated_bf16_swiglu(
        shared_gate.view(),
        shared_up.view(),
        shared_down.view(),
    )?;
    let canonical_shared = transition.canonical_runtime.upload_replicated_bf16_swiglu(
        shared_gate.view(),
        shared_up.view(),
        shared_down.view(),
    )?;

    let (canonical, native) =
        run_attention_transition_pair(runtime, &transition, input, tokens, hidden)?;
    compare_exact(
        "full_layer_attention",
        &canonical.attention,
        &native.attention,
    )?;
    let canonical_residual = transition
        .canonical_runtime
        .collect_replicated_device_rows(&canonical.residual)?
        .remove(0);
    let native_residual = runtime.collect_replicated_device_rows(&native.residual)?;
    compare_distributed_exact(
        "full_layer_attention_residual",
        &canonical_residual,
        &native_residual,
    )?;
    let canonical_post_norm = transition
        .canonical_runtime
        .collect_replicated_device_rows(&canonical.post_attention_norm)?
        .remove(0);
    let native_post_norm = runtime.collect_replicated_device_rows(&native.post_attention_norm)?;
    compare_distributed_exact(
        "full_layer_post_attention_norm",
        &canonical_post_norm,
        &native_post_norm,
    )?;

    let canonical_route = transition
        .canonical_runtime
        .sigmoid_topk_replicated_device_rows_host(
            &canonical_router,
            &canonical.post_attention_norm,
        )?;
    let native_route =
        runtime.sigmoid_topk_replicated_device_rows_host(&router, &native.post_attention_norm)?;
    compare_exact(
        "full_layer_router_logits",
        &canonical_route.logits,
        &native_route.logits,
    )?;
    compare_route_indices(
        "full_layer_router_selected_tp1_tp4",
        &canonical_route.selected,
        &native_route.selected,
    )?;
    compare_exact(
        "full_layer_router_weights_tp1_tp4",
        &canonical_route.weights,
        &native_route.weights,
    )?;
    let (host_selected, host_weights) = HybridModel::moe_route_sigmoid_host_public(
        &canonical_route.logits,
        tokens,
        expert_count,
        experts_per_token,
        Some(&router_bias),
        router_scale,
        route_norm,
        None,
    )?;
    compare_route_indices(
        "full_layer_router_selected_host",
        &host_selected,
        &canonical_route.selected,
    )?;
    compare_exact(
        "full_layer_router_weights_host",
        &host_weights,
        &canonical_route.weights,
    )?;
    let selected = canonical_route
        .selected
        .iter()
        .map(|&expert| usize::try_from(expert))
        .collect::<Result<Vec<_>, _>>()?;

    let oracle_projection = reference_grouped_projections(
        runtime,
        gate_bank,
        up_bank,
        down_bank,
        &canonical_post_norm,
        tokens,
        &selected,
        routed_activation_limit,
    )?;
    let routed_oracle = combine_down_rows_host(
        &oracle_projection.down,
        &canonical_route.weights,
        tokens,
        hidden,
    )?;
    let canonical_shared_output = transition
        .canonical_runtime
        .replicated_bf16_swiglu_resident_device(
            &canonical_shared,
            &canonical.post_attention_norm,
            shared_activation_limit,
        )?;
    let canonical_shared_output = transition
        .canonical_runtime
        .collect_replicated_device_rows(&canonical_shared_output)?
        .remove(0);
    let native_shared_output = runtime.replicated_bf16_swiglu_resident_device(
        &shared,
        &native.post_attention_norm,
        shared_activation_limit,
    )?;
    let native_shared_host = runtime.collect_replicated_device_rows(&native_shared_output)?;
    compare_distributed_exact(
        "full_layer_shared_expert",
        &canonical_shared_output,
        &native_shared_host,
    )?;
    let final_oracle = moe_residual_host(
        &canonical_residual,
        &routed_oracle,
        &canonical_shared_output,
    )?;

    let mut prepared = runtime.prepare_step_grouped_expert_parallel_gate(
        resident_experts,
        &canonical_post_norm,
        tokens,
        &selected,
        routed_activation_limit,
    )?;
    let generation = prepared.generation();
    let mut combine = runtime
        .prepare_step_grouped_expert_parallel_combine(&prepared, &canonical_route.weights)?;
    runtime.refresh_step_grouped_expert_parallel_inputs_from_replicated(
        resident_experts,
        &mut prepared,
        &native.post_attention_norm,
    )?;
    if prepared.generation() <= generation {
        return Err("full-layer routed input generation did not advance".into());
    }
    runtime.refresh_step_grouped_expert_parallel_combine(
        &prepared,
        &mut combine,
        &canonical_route.weights,
    )?;
    runtime.execute_step_grouped_expert_parallel_gate(resident_experts, &mut prepared)?;
    let projection = runtime.collect_step_grouped_expert_parallel_gate(&prepared)?;
    compare_projection_exact(
        "full_layer_routed_projection",
        &oracle_projection,
        &projection,
    )?;
    runtime.execute_step_grouped_expert_parallel_combine(&prepared, &mut combine)?;
    runtime.broadcast_step_grouped_expert_parallel_combine(&prepared, &mut combine)?;
    let routed_outputs =
        runtime.collect_step_grouped_expert_parallel_broadcast(&prepared, &combine)?;
    compare_distributed_exact("full_layer_routed_combine", &routed_oracle, &routed_outputs)?;
    let full_layer_output = runtime.finish_step_grouped_expert_parallel_layer(
        &prepared,
        &combine,
        &native_shared_output,
        &native.residual,
    )?;
    let full_layer_outputs = runtime.collect_replicated_device_rows(&full_layer_output)?;
    compare_distributed_exact("full_layer_output", &final_oracle, &full_layer_outputs)?;

    println!(
        "STEP_FULL_DECODER_LAYER_GATE_PASS layer={layer} tokens={tokens} ranks={} \
         query_heads={} kv_heads={} head_dim={} router=sigmoid-topk-device \
         router_control=host-visible router_bias=true router_scale={router_scale} \
         router_norm={route_norm} experts={expert_count} experts_per_token={experts_per_token} \
         routed_expert_layout=expert-parallel active_owners={} \
         routed_activation_limit={} shared_expert_layout=replicated \
         shared_width={shared_width} shared_activation_limit={} \
         final_add_order=routed-plus-shared-then-residual \
         raw_bit_attention=true raw_bit_router_logits=true raw_bit_router_selected=true \
         raw_bit_router_weights=true raw_bit_routed_gate=true raw_bit_routed_up=true \
         raw_bit_routed_down=true raw_bit_routed_combine=true raw_bit_shared_expert=true \
         raw_bit_final_output=true full_layer_claim=true \
         tensor_parallel_scope=attention expert_parallel_scope=routed-experts \
         production_path=false performance_claim=false",
        runtime.devices().len(),
        transition.geometry.n_head,
        transition.geometry.n_head_kv,
        transition.geometry.head_dim_k,
        prepared.active_owners(),
        routed_activation_limit.unwrap_or(0.0),
        shared_activation_limit.unwrap_or(0.0),
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_dynamic_owner_grouped_fp8_device_handoff_gate(
    runtime: &TpE4m3HostBounce,
    resident: &ResidentExpertParallel,
    gate_bank: &Fp8StackedNative<'_>,
    up_bank: &Fp8StackedNative<'_>,
    down_bank: &Fp8StackedNative<'_>,
    initial_input: &[f32],
    initial_tokens: usize,
    layer: usize,
    activation_limit: Option<f32>,
) -> Result<(), Box<dyn std::error::Error>> {
    let default_capacity = initial_tokens
        .checked_add(2)
        .ok_or("device handoff token capacity overflow")?;
    let capacity_tokens = std::env::var("MEMRA_STEP_EP_DYNAMIC_CAPACITY_TOKENS")
        .ok()
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(default_capacity);
    if capacity_tokens <= initial_tokens {
        return Err(format!(
            "device handoff capacity {capacity_tokens} must exceed initial tokens \
             {initial_tokens}"
        )
        .into());
    }
    let initial_selected = grouped_selected_routes(initial_tokens, gate_bank.n_expert)?;
    let initial_weights = grouped_route_weights(initial_tokens, 0);
    let dynamic_tokens = capacity_tokens;
    let dynamic_input = activations_with_salt(dynamic_tokens, gate_bank.in_f, layer + 17);
    let dynamic_selected = dynamic_grouped_selected_routes(
        dynamic_tokens,
        gate_bank.n_expert,
        runtime.devices().len(),
    )?;
    let dynamic_weights = grouped_route_weights(dynamic_tokens, 3);

    let start = Instant::now();
    let initial_oracle_projection = reference_grouped_projections(
        runtime,
        gate_bank,
        up_bank,
        down_bank,
        initial_input,
        initial_tokens,
        &initial_selected,
        activation_limit,
    )?;
    let initial_oracle = combine_down_rows_host(
        &initial_oracle_projection.down,
        &initial_weights,
        initial_tokens,
        gate_bank.in_f,
    )?;
    let initial_oracle_us = elapsed_us(start);
    let start = Instant::now();
    let dynamic_oracle_projection = reference_grouped_projections(
        runtime,
        gate_bank,
        up_bank,
        down_bank,
        &dynamic_input,
        dynamic_tokens,
        &dynamic_selected,
        activation_limit,
    )?;
    let dynamic_oracle = combine_down_rows_host(
        &dynamic_oracle_projection.down,
        &dynamic_weights,
        dynamic_tokens,
        gate_bank.in_f,
    )?;
    let dynamic_oracle_us = elapsed_us(start);

    let start = Instant::now();
    let mut prepared = runtime.prepare_step_grouped_expert_parallel_gate_with_capacity(
        resident,
        initial_input,
        initial_tokens,
        &initial_selected,
        activation_limit,
        capacity_tokens,
    )?;
    let mut combine =
        runtime.prepare_step_grouped_expert_parallel_combine(&prepared, &initial_weights)?;
    let prepare_us = elapsed_us(start);
    if prepared.active_owners() != runtime.devices().len()
        || combine.distributed_ranks() != runtime.devices().len()
        || combine.owner_pair_counts() != prepared.owner_pair_counts()
    {
        return Err("device handoff initial owner/rank geometry differs".into());
    }
    let initial_owner_pair_counts = prepared.owner_pair_counts();
    let initial_generation = prepared.generation();

    runtime.execute_step_grouped_expert_parallel_gate(resident, &mut prepared)?;
    let initial_projection = runtime.collect_step_grouped_expert_parallel_gate(&prepared)?;
    compare_projection_exact(
        "device_handoff_initial_projection",
        &initial_oracle_projection,
        &initial_projection,
    )?;
    runtime.execute_step_grouped_expert_parallel_combine(&prepared, &mut combine)?;
    runtime.broadcast_step_grouped_expert_parallel_combine(&prepared, &mut combine)?;
    let initial_outputs =
        runtime.collect_step_grouped_expert_parallel_broadcast(&prepared, &combine)?;
    compare_distributed_exact("device_handoff_initial", &initial_oracle, &initial_outputs)?;

    let start = Instant::now();
    execute_owner_grouped_fp8_device_handoff(runtime, resident, &mut prepared, &mut combine)?;
    let initial_repeat =
        runtime.collect_step_grouped_expert_parallel_broadcast(&prepared, &combine)?;
    let initial_device_us = elapsed_us(start);
    compare_distributed_exact(
        "device_handoff_initial_repeat",
        &initial_oracle,
        &initial_repeat,
    )?;

    let start = Instant::now();
    runtime.refresh_step_grouped_expert_parallel_gate(
        resident,
        &mut prepared,
        &dynamic_input,
        dynamic_tokens,
        &dynamic_selected,
    )?;
    runtime.refresh_step_grouped_expert_parallel_combine(
        &prepared,
        &mut combine,
        &dynamic_weights,
    )?;
    let dynamic_refresh_us = elapsed_us(start);
    if prepared.active_owners() != runtime.devices().len()
        || combine.distributed_ranks() != runtime.devices().len()
        || combine.owner_pair_counts() != prepared.owner_pair_counts()
    {
        return Err("device handoff dynamic owner/rank geometry differs".into());
    }
    let dynamic_owner_pair_counts = prepared.owner_pair_counts();
    let dynamic_generation = prepared.generation();
    if dynamic_generation <= initial_generation {
        return Err("device handoff projection generation did not advance".into());
    }
    runtime.execute_step_grouped_expert_parallel_gate(resident, &mut prepared)?;
    let dynamic_projection = runtime.collect_step_grouped_expert_parallel_gate(&prepared)?;
    compare_projection_exact(
        "device_handoff_dynamic_projection",
        &dynamic_oracle_projection,
        &dynamic_projection,
    )?;
    runtime.execute_step_grouped_expert_parallel_combine(&prepared, &mut combine)?;
    runtime.broadcast_step_grouped_expert_parallel_combine(&prepared, &mut combine)?;
    let dynamic_outputs =
        runtime.collect_step_grouped_expert_parallel_broadcast(&prepared, &combine)?;
    compare_distributed_exact("device_handoff_dynamic", &dynamic_oracle, &dynamic_outputs)?;

    let start = Instant::now();
    execute_owner_grouped_fp8_device_handoff(runtime, resident, &mut prepared, &mut combine)?;
    let dynamic_repeat =
        runtime.collect_step_grouped_expert_parallel_broadcast(&prepared, &combine)?;
    let dynamic_device_us = elapsed_us(start);
    compare_distributed_exact(
        "device_handoff_dynamic_repeat",
        &dynamic_oracle,
        &dynamic_repeat,
    )?;

    let profile = strict_bool("MEMRA_STEP_EP_DYNAMIC_OWNER_GROUPED_FP8_DEVICE_HANDOFF_PROFILE")?;
    if profile {
        let result = unsafe { cudaProfilerStart() };
        if result != 0 {
            return Err(format!("cudaProfilerStart failed with CUDA error {result}").into());
        }
    }
    let profile_runs = (|| {
        let start = Instant::now();
        runtime.refresh_step_grouped_expert_parallel_gate(
            resident,
            &mut prepared,
            initial_input,
            initial_tokens,
            &initial_selected,
        )?;
        runtime.refresh_step_grouped_expert_parallel_combine(
            &prepared,
            &mut combine,
            &initial_weights,
        )?;
        let profiled_initial_refresh_us = elapsed_us(start);
        let start = Instant::now();
        execute_owner_grouped_fp8_device_handoff(runtime, resident, &mut prepared, &mut combine)?;
        let profiled_initial_handoff_us = elapsed_us(start);

        let start = Instant::now();
        runtime.refresh_step_grouped_expert_parallel_gate(
            resident,
            &mut prepared,
            &dynamic_input,
            dynamic_tokens,
            &dynamic_selected,
        )?;
        runtime.refresh_step_grouped_expert_parallel_combine(
            &prepared,
            &mut combine,
            &dynamic_weights,
        )?;
        let profiled_dynamic_refresh_us = elapsed_us(start);
        let start = Instant::now();
        execute_owner_grouped_fp8_device_handoff(runtime, resident, &mut prepared, &mut combine)?;
        let profiled_dynamic_handoff_us = elapsed_us(start);
        Ok::<_, Box<dyn std::error::Error>>((
            profiled_initial_refresh_us,
            profiled_initial_handoff_us,
            profiled_dynamic_refresh_us,
            profiled_dynamic_handoff_us,
        ))
    })();
    if profile {
        let result = unsafe { cudaProfilerStop() };
        if result != 0 {
            return Err(format!("cudaProfilerStop failed with CUDA error {result}").into());
        }
    }
    let (
        profiled_initial_refresh_us,
        profiled_initial_handoff_us,
        profiled_dynamic_refresh_us,
        profiled_dynamic_handoff_us,
    ) = profile_runs?;
    let profiled_dynamic_outputs =
        runtime.collect_step_grouped_expert_parallel_broadcast(&prepared, &combine)?;
    compare_distributed_exact(
        "device_handoff_profiled_dynamic",
        &dynamic_oracle,
        &profiled_dynamic_outputs,
    )?;

    runtime.refresh_step_grouped_expert_parallel_gate(
        resident,
        &mut prepared,
        initial_input,
        initial_tokens,
        &initial_selected,
    )?;
    runtime.refresh_step_grouped_expert_parallel_combine(
        &prepared,
        &mut combine,
        &initial_weights,
    )?;
    execute_owner_grouped_fp8_device_handoff(runtime, resident, &mut prepared, &mut combine)?;
    let restored_outputs =
        runtime.collect_step_grouped_expert_parallel_broadcast(&prepared, &combine)?;
    compare_distributed_exact(
        "device_handoff_restored",
        &initial_oracle,
        &restored_outputs,
    )?;

    let ranks = runtime.devices().len();
    let profiled_projection_executions = 2;
    let profiled_owner_projection_executions = ranks * profiled_projection_executions;
    let profiled_projection_launches = profiled_owner_projection_executions * 3;
    let profiled_quantize_launches = profiled_projection_launches;
    let profiled_activation_launches = profiled_owner_projection_executions;
    let profiled_combine_executions = 2;
    let profiled_scatter_launches = ranks * profiled_combine_executions;
    let profiled_reduce_launches = profiled_combine_executions;
    let profiled_broadcast_executions = 2;
    let profiled_peer_broadcast_calls = (ranks - 1) * profiled_broadcast_executions;
    let profiled_total_kernels = profiled_projection_launches
        + profiled_quantize_launches
        + profiled_activation_launches
        + profiled_scatter_launches
        + profiled_reduce_launches;
    println!(
        "STEP_DYNAMIC_OWNER_GROUPED_FP8_DEVICE_HANDOFF_GATE_PASS layer={layer} \
         initial_tokens={initial_tokens} dynamic_tokens={dynamic_tokens} capacity_tokens={} \
         initial_pairs={} dynamic_pairs={} experts_per_token={} owners={ranks} \
         distributed_ranks={} initial_owner_pair_counts={initial_owner_pair_counts:?} \
         dynamic_owner_pair_counts={dynamic_owner_pair_counts:?} activation_limit={} \
         initial_generation={initial_generation} dynamic_generation={dynamic_generation} \
         generation_monotonic=true projection_preparation_count=1 combine_preparation_count=1 \
         profiled_projection_refreshes=2 profiled_combine_refreshes=2 \
         profiled_projection_executions={profiled_projection_executions} \
         profiled_combine_executions={profiled_combine_executions} \
         profiled_broadcast_executions={profiled_broadcast_executions} \
         profiled_intermediate_dtoh_calls=0 hot_device_allocations=0 \
         profiled_grouped_quantize_kernel_launches={profiled_quantize_launches} \
         profiled_grouped_projection_kernel_launches={profiled_projection_launches} \
         profiled_grouped_activation_kernel_launches={profiled_activation_launches} \
         profiled_combine_scatter_kernel_launches={profiled_scatter_launches} \
         profiled_combine_reduce_kernel_launches={profiled_reduce_launches} \
         profiled_peer_broadcast_calls={profiled_peer_broadcast_calls} \
         profiled_total_kernel_launches={profiled_total_kernels} prepare_us={prepare_us:.3} \
         dynamic_refresh_us={dynamic_refresh_us:.3} \
         profiled_initial_refresh_us={profiled_initial_refresh_us:.3} \
         profiled_initial_handoff_us={profiled_initial_handoff_us:.3} \
         profiled_dynamic_refresh_us={profiled_dynamic_refresh_us:.3} \
         profiled_dynamic_handoff_us={profiled_dynamic_handoff_us:.3} \
         initial_oracle_us={initial_oracle_us:.3} dynamic_oracle_us={dynamic_oracle_us:.3} \
         initial_device_handoff_us={initial_device_us:.3} \
         dynamic_device_handoff_us={dynamic_device_us:.3} initial_speedup={:.3} \
         dynamic_speedup={:.3} raw_bit_gate=true raw_bit_up=true raw_bit_down=true \
         raw_bit_combine=true rank_outputs_exact=true combine_numeric=separate-rn-mul-add \
         combine_order=canonical-top8 dispatch=native-p2p expert_layout=expert-parallel \
         attention_handoff=replicated-hidden-state production_path=false \
         production_routing=false attention_tensor_parallel_ready=false \
         tensor_parallel_claim=false",
        prepared.max_tokens(),
        initial_tokens * STEP_GROUPED_TOP_K,
        dynamic_tokens * STEP_GROUPED_TOP_K,
        STEP_GROUPED_TOP_K,
        combine.distributed_ranks(),
        activation_limit.unwrap_or(0.0),
        initial_oracle_us / initial_device_us,
        dynamic_oracle_us / dynamic_device_us,
    );
    Ok(())
}

fn compare_exact(label: &str, expected: &[f32], actual: &[f32]) -> Result<(), String> {
    let mismatches = bit_mismatches(expected, actual);
    println!(
        "EP_EXACT label={label} bit_mismatches={mismatches}/{}",
        expected.len()
    );
    if mismatches != 0 {
        return Err(format!("{label}: EP output differs from the reference"));
    }
    Ok(())
}

fn bit_mismatches(expected: &[f32], actual: &[f32]) -> usize {
    expected
        .iter()
        .zip(actual)
        .filter(|(left, right)| left.to_bits() != right.to_bits())
        .count()
}

fn benchmark_config() -> Result<Option<(usize, usize)>, Box<dyn std::error::Error>> {
    let raw = match std::env::var("MEMRA_STEP_EP_BENCH_ITERS") {
        Ok(raw) if !raw.is_empty() && raw != "0" => raw,
        Ok(_) | Err(std::env::VarError::NotPresent) => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    let iterations = raw.parse::<usize>().map_err(|_| {
        format!("MEMRA_STEP_EP_BENCH_ITERS={raw:?} is invalid; expected an integer")
    })?;
    if !(1..=1000).contains(&iterations) {
        return Err(format!(
            "MEMRA_STEP_EP_BENCH_ITERS={iterations} is invalid; expected 1..=1000"
        )
        .into());
    }
    let warmup = match std::env::var("MEMRA_STEP_EP_BENCH_WARMUP") {
        Ok(raw) => raw.parse::<usize>().map_err(|_| {
            format!("MEMRA_STEP_EP_BENCH_WARMUP={raw:?} is invalid; expected an integer")
        })?,
        Err(std::env::VarError::NotPresent) => 5,
        Err(error) => return Err(error.into()),
    };
    if warmup > 1000 {
        return Err(
            format!("MEMRA_STEP_EP_BENCH_WARMUP={warmup} is invalid; expected 0..=1000").into(),
        );
    }
    Ok(Some((warmup, iterations)))
}

fn nearest_rank_ns(sorted: &[u128], percentile: usize) -> u128 {
    debug_assert!(!sorted.is_empty());
    debug_assert!((1..=100).contains(&percentile));
    let rank = percentile
        .checked_mul(sorted.len())
        .expect("benchmark percentile rank overflow")
        .div_ceil(100);
    sorted[rank.saturating_sub(1).min(sorted.len() - 1)]
}

fn output_checksum(values: &[f32]) -> u64 {
    values.iter().fold(0x9e37_79b9_7f4a_7c15, |state, value| {
        state.rotate_left(7) ^ u64::from(value.to_bits())
    })
}

#[allow(clippy::too_many_arguments)]
fn benchmark_routed_experts(
    runtime: &TpE4m3HostBounce,
    resident: &ResidentExpertParallel,
    input: &[f32],
    tokens: usize,
    selected: &[usize],
    route_weights: &[f32],
    experts_per_token: usize,
    expected: &[f32],
    warmup: usize,
    iterations: usize,
    activation_limit: Option<f32>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mode = runtime.expert_activation_label();
    for iteration in 0..warmup {
        let output = runtime.run_routed_experts(
            resident,
            input,
            tokens,
            selected,
            route_weights,
            experts_per_token,
            activation_limit,
        )?;
        let mismatches = bit_mismatches(expected, &output);
        if mismatches != 0 {
            return Err(format!(
                "EP benchmark mode={mode} warmup={iteration} differs from reference: \
                 {mismatches}/{}",
                expected.len()
            )
            .into());
        }
        black_box(output);
    }

    let mut samples_ns = Vec::with_capacity(iterations);
    let mut checksum = None;
    for iteration in 0..iterations {
        let start = Instant::now();
        let output = runtime.run_routed_experts(
            resident,
            input,
            tokens,
            selected,
            route_weights,
            experts_per_token,
            activation_limit,
        )?;
        let elapsed_ns = start.elapsed().as_nanos();
        let mismatches = bit_mismatches(expected, &output);
        if mismatches != 0 {
            return Err(format!(
                "EP benchmark mode={mode} iteration={iteration} differs from reference: \
                 {mismatches}/{}",
                expected.len()
            )
            .into());
        }
        let sample_checksum = output_checksum(&output);
        match checksum {
            Some(expected_checksum) if expected_checksum != sample_checksum => {
                return Err(format!(
                    "EP benchmark mode={mode} checksum changed at iteration {iteration}"
                )
                .into());
            }
            None => checksum = Some(sample_checksum),
            _ => {}
        }
        samples_ns.push(elapsed_ns);
        println!(
            "EP_BENCH_SAMPLE mode={mode} iteration={iteration} \
             elapsed_us={:.3} bit_mismatches=0/{} checksum={sample_checksum:016x}",
            elapsed_ns as f64 / 1000.0,
            expected.len()
        );
        black_box(output);
    }

    let mut sorted = samples_ns.clone();
    sorted.sort_unstable();
    let total_ns = samples_ns.iter().copied().sum::<u128>();
    let mean_us = total_ns as f64 / iterations as f64 / 1000.0;
    let routes_per_iteration = tokens * experts_per_token;
    let routes_per_second = routes_per_iteration as f64 * 1_000_000.0 / mean_us;
    println!(
        "EP_BENCH_SUMMARY mode={mode} warmup={warmup} iterations={iterations} \
         tokens={tokens} routes_per_iteration={routes_per_iteration} \
         mean_us={mean_us:.3} min_us={:.3} p50_us={:.3} p90_us={:.3} \
         p95_us={:.3} p99_us={:.3} max_us={:.3} \
         routes_per_second={routes_per_second:.3} checksum={:016x} \
         output={} product_topology=false tensor_parallel_claim=false \
         performance_claim=diagnostic",
        sorted[0] as f64 / 1000.0,
        nearest_rank_ns(&sorted, 50) as f64 / 1000.0,
        nearest_rank_ns(&sorted, 90) as f64 / 1000.0,
        nearest_rank_ns(&sorted, 95) as f64 / 1000.0,
        nearest_rank_ns(&sorted, 99) as f64 / 1000.0,
        sorted[sorted.len() - 1] as f64 / 1000.0,
        checksum.unwrap_or_default(),
        runtime.expert_output_label(),
    );
    Ok(())
}

#[cfg(test)]
mod benchmark_tests {
    use super::{
        STEP_GROUPED_TOP_K, combine_down_rows_host, dynamic_grouped_selected_routes,
        grouped_route_weights, grouped_selected_routes, nearest_rank_ns, output_checksum,
    };

    #[test]
    fn nearest_rank_percentiles_use_observed_samples() {
        let samples = [10, 20, 30, 40, 50];
        assert_eq!(nearest_rank_ns(&samples, 50), 30);
        assert_eq!(nearest_rank_ns(&samples, 90), 50);
        assert_eq!(nearest_rank_ns(&samples, 95), 50);
        assert_eq!(nearest_rank_ns(&samples, 99), 50);
    }

    #[test]
    fn output_checksum_is_bit_sensitive_and_repeat_stable() {
        let values = [0.0, -0.0, 1.0, -2.5];
        assert_eq!(output_checksum(&values), output_checksum(&values));
        assert_ne!(
            output_checksum(&values),
            output_checksum(&[0.0, 0.0, 1.0, -2.5])
        );
    }

    #[test]
    fn grouped_route_fixture_is_top8_unique_and_expert_spanning() {
        let routes = grouped_selected_routes(2, 288).unwrap();
        assert_eq!(routes.len(), 2 * STEP_GROUPED_TOP_K);
        for token_routes in routes.chunks_exact(STEP_GROUPED_TOP_K) {
            let mut unique = token_routes.to_vec();
            unique.sort_unstable();
            unique.dedup();
            assert_eq!(unique.len(), STEP_GROUPED_TOP_K);
        }
        assert_eq!(
            &routes[..STEP_GROUPED_TOP_K],
            &[0, 36, 72, 108, 144, 180, 216, 252]
        );
        assert_eq!(
            &routes[STEP_GROUPED_TOP_K..],
            &[1, 37, 73, 109, 145, 181, 217, 253]
        );
    }

    #[test]
    fn dynamic_grouped_fixture_changes_owner_load_without_duplicate_routes() {
        let routes = dynamic_grouped_selected_routes(3, 288, 4).unwrap();
        assert_eq!(routes.len(), 3 * STEP_GROUPED_TOP_K);
        let mut owner_counts = [0usize; 4];
        for token_routes in routes.chunks_exact(STEP_GROUPED_TOP_K) {
            let mut unique = token_routes.to_vec();
            unique.sort_unstable();
            unique.dedup();
            assert_eq!(unique.len(), STEP_GROUPED_TOP_K);
            for &expert in token_routes {
                owner_counts[expert / 72] += 1;
            }
        }
        assert_eq!(owner_counts, [3, 6, 9, 6]);
        assert!(dynamic_grouped_selected_routes(3, 288, 3).is_err());
    }

    #[test]
    fn grouped_route_weights_are_shifted_normalized_top8_permutations() {
        let weights = grouped_route_weights(2, 3);
        assert_eq!(weights.len(), 2 * STEP_GROUPED_TOP_K);
        let expected = (1..=STEP_GROUPED_TOP_K)
            .map(|value| value as f32 / 36.0)
            .collect::<Vec<_>>();
        for token_weights in weights.chunks_exact(STEP_GROUPED_TOP_K) {
            let mut sorted = token_weights.to_vec();
            sorted.sort_by(f32::total_cmp);
            assert_eq!(sorted, expected);
            assert!((token_weights.iter().sum::<f32>() - 1.0).abs() <= f32::EPSILON);
        }
        assert_ne!(
            &weights[..STEP_GROUPED_TOP_K],
            &weights[STEP_GROUPED_TOP_K..]
        );
    }

    #[test]
    fn host_combine_uses_canonical_slot_weights() {
        let mut down = vec![0.0f32; STEP_GROUPED_TOP_K * 2];
        down[..6].copy_from_slice(&[1.0, 10.0, 2.0, 20.0, -4.0, 8.0]);
        let weights = [0.5, 0.25, 0.125, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert_eq!(
            combine_down_rows_host(&down, &weights, 1, 2).unwrap(),
            [0.5, 11.0]
        );
        assert!(combine_down_rows_host(&down[..down.len() - 1], &weights, 1, 2).is_err());
    }
}

#[derive(Default)]
struct NumericDelta {
    mismatches: usize,
    max_abs: f32,
    max_ulp: u32,
    first: Option<(usize, u32, u32)>,
}

fn ordered_float_bits(value: f32) -> u32 {
    let bits = value.to_bits();
    if bits & 0x8000_0000 == 0 {
        bits | 0x8000_0000
    } else {
        !bits
    }
}

fn numeric_delta(expected: &[f32], actual: &[f32]) -> NumericDelta {
    assert_eq!(expected.len(), actual.len());
    let mut delta = NumericDelta::default();
    for (index, (&left, &right)) in expected.iter().zip(actual).enumerate() {
        if left.to_bits() == right.to_bits() {
            continue;
        }
        delta.mismatches += 1;
        delta.max_abs = delta.max_abs.max((left - right).abs());
        delta.max_ulp = delta
            .max_ulp
            .max(ordered_float_bits(left).abs_diff(ordered_float_bits(right)));
        delta
            .first
            .get_or_insert((index, left.to_bits(), right.to_bits()));
    }
    delta
}

/// This binary is the step35 expert-parallel cell end to end: its FP8 banks, grouped gates and
/// reference expert all encode step35's POST clamp (`min(silu(gate), l) * clamp(up, +-l)`).
/// glm5_next's PRE form is a different program with no arm here, so it is refused by name — a
/// bare limit handed to a POST kernel would run and report a passing gate on wrong arithmetic.
fn post_clamp_only(
    clamp: Option<SwigluClamp>,
    layer: usize,
) -> Result<Option<f32>, Box<dyn std::error::Error>> {
    match clamp {
        None => Ok(None),
        Some(SwigluClamp::Post(l)) => Ok(Some(l)),
        Some(SwigluClamp::Pre(_)) => Err(format!(
            "layer {layer}: PRE-clamped SwiGLU (glm5_next) has no expert-parallel arm in \
             ep_step_fp8_gate; this cell is step35 POST-clamp only"
        )
        .into()),
    }
}

fn per_iteration_us(start: Instant, iterations: usize) -> f64 {
    start.elapsed().as_secs_f64() * 1_000_000.0 / iterations as f64
}

#[allow(clippy::too_many_arguments)]
fn probe_device_numerics(
    runtime: &TpE4m3HostBounce,
    gate_bank: &Fp8StackedNative<'_>,
    up_bank: &Fp8StackedNative<'_>,
    down_bank: &Fp8StackedNative<'_>,
    experts: &[usize],
    weights: &[f32],
    input: &[f32],
    activation_limit: Option<f32>,
) -> Result<bool, Box<dyn std::error::Error>> {
    let iterations = std::env::var("MEMRA_STEP_EP_NUMERIC_ITERS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(50);
    if iterations == 0 {
        return Err("MEMRA_STEP_EP_NUMERIC_ITERS must be positive".into());
    }
    let input_row = input
        .get(..gate_bank.in_f)
        .ok_or("numeric probe input has no complete token row")?;
    let engine = runtime
        .rank_engine(0)
        .ok_or("numeric probe runtime has no rank-zero engine")?;
    let mut all_cuda_expf_activation_exact = true;
    let mut all_host_exact_activation_exact = true;
    let mut down_rows = Vec::with_capacity(experts.len());
    let mut timing_gate = None;
    let mut timing_up = None;

    for &expert in experts {
        let gate = runtime.full(expert_matrix(gate_bank, expert)?, input_row, 1)?;
        let up = runtime.full(expert_matrix(up_bank, expert)?, input_row, 1)?;
        let host_activation = gate
            .iter()
            .zip(&up)
            .map(|(&gate, &up)| step_expert_activation_host(gate, up, activation_limit))
            .collect::<Vec<_>>();
        let (device_activation, host_exact_activation, host_exact_activation_repeat) = {
            let _main = engine.gpu.enter_main()?;
            let gate_d = engine.htod(&gate)?;
            let up_d = engine.htod(&up)?;
            let mut activation_d = engine.zeros(gate.len())?;
            if let Some(limit) = activation_limit {
                engine.swiglu_clamped_mul_scaled(
                    &gate_d,
                    &up_d,
                    1.0,
                    1.0,
                    limit,
                    &mut activation_d,
                    host_activation.len(),
                )?;
            } else {
                engine.silu_mul(&gate_d, &up_d, &mut activation_d, host_activation.len())?;
            }
            let result = engine.dtoh(&activation_d)?;
            let mut host_exact_d = engine.zeros(gate.len())?;
            if let Some(limit) = activation_limit {
                engine.silu_clamped_mul_host_expf(
                    &gate_d,
                    &up_d,
                    limit,
                    &mut host_exact_d,
                    host_activation.len(),
                )?;
            } else {
                engine.silu_mul_host_expf(
                    &gate_d,
                    &up_d,
                    &mut host_exact_d,
                    host_activation.len(),
                )?;
            }
            let host_exact = engine.dtoh(&host_exact_d)?;
            let mut host_exact_repeat_d = engine.zeros(gate.len())?;
            if let Some(limit) = activation_limit {
                engine.silu_clamped_mul_host_expf(
                    &gate_d,
                    &up_d,
                    limit,
                    &mut host_exact_repeat_d,
                    host_activation.len(),
                )?;
            } else {
                engine.silu_mul_host_expf(
                    &gate_d,
                    &up_d,
                    &mut host_exact_repeat_d,
                    host_activation.len(),
                )?;
            }
            let host_exact_repeat = engine.dtoh(&host_exact_repeat_d)?;
            if timing_gate.is_none() {
                timing_gate = Some(gate_d);
                timing_up = Some(up_d);
            }
            (result, host_exact, host_exact_repeat)
        };
        let delta = numeric_delta(&host_activation, &device_activation);
        all_cuda_expf_activation_exact &= delta.mismatches == 0;
        let first = delta
            .first
            .map(|(index, host, device)| {
                format!("index={index} host=0x{host:08x} device=0x{device:08x}")
            })
            .unwrap_or_else(|| "none".to_string());
        println!(
            "EP_DEVICE_NUMERIC kind=activation expert={expert} \
             bit_mismatches={}/{} max_abs={:.9e} max_ulp={} first=\"{}\"",
            delta.mismatches,
            host_activation.len(),
            delta.max_abs,
            delta.max_ulp,
            first,
        );
        let host_exact_delta = numeric_delta(&host_activation, &host_exact_activation);
        let host_exact_repeat_delta =
            numeric_delta(&host_exact_activation, &host_exact_activation_repeat);
        all_host_exact_activation_exact &=
            host_exact_delta.mismatches == 0 && host_exact_repeat_delta.mismatches == 0;
        let first = host_exact_delta
            .first
            .map(|(index, host, device)| {
                format!("index={index} host=0x{host:08x} device=0x{device:08x}")
            })
            .unwrap_or_else(|| "none".to_string());
        println!(
            "EP_DEVICE_NUMERIC kind=activation_host_exact expert={expert} \
             bit_mismatches={}/{} max_abs={:.9e} max_ulp={} first=\"{}\" \
             repeat_mismatches={}/{}",
            host_exact_delta.mismatches,
            host_activation.len(),
            host_exact_delta.max_abs,
            host_exact_delta.max_ulp,
            first,
            host_exact_repeat_delta.mismatches,
            host_activation.len(),
        );
        down_rows.push(runtime.full(expert_matrix(down_bank, expert)?, &host_activation, 1)?);
    }

    let mut host_accumulation = vec![0.0f32; gate_bank.in_f];
    for (&weight, row) in weights.iter().zip(&down_rows) {
        for (sum, &value) in host_accumulation.iter_mut().zip(row) {
            *sum += weight * value;
        }
    }
    let down_device = {
        let _main = engine.gpu.enter_main()?;
        down_rows
            .iter()
            .map(|row| engine.htod(row))
            .collect::<Result<Vec<_>, _>>()?
    };
    let device_accumulation = {
        let _main = engine.gpu.enter_main()?;
        let mut output = engine.zeros(gate_bank.in_f)?;
        for (&weight, row) in weights.iter().zip(&down_device) {
            let mut view = output.slice_mut(0..gate_bank.in_f);
            engine.axpy_into(row, weight, &mut view, gate_bank.in_f)?;
        }
        engine.dtoh(&output)?
    };
    let device_accumulation_repeat = {
        let _main = engine.gpu.enter_main()?;
        let mut output = engine.zeros(gate_bank.in_f)?;
        for (&weight, row) in weights.iter().zip(&down_device) {
            let mut view = output.slice_mut(0..gate_bank.in_f);
            engine.axpy_into(row, weight, &mut view, gate_bank.in_f)?;
        }
        engine.dtoh(&output)?
    };
    let host_exact_accumulation = {
        let _main = engine.gpu.enter_main()?;
        let mut output = engine.zeros(gate_bank.in_f)?;
        for (&weight, row) in weights.iter().zip(&down_device) {
            let mut view = output.slice_mut(0..gate_bank.in_f);
            engine.axpy_host_into(&row.slice(0..row.len()), weight, &mut view, gate_bank.in_f)?;
        }
        engine.dtoh(&output)?
    };
    let host_exact_accumulation_repeat = {
        let _main = engine.gpu.enter_main()?;
        let mut output = engine.zeros(gate_bank.in_f)?;
        for (&weight, row) in weights.iter().zip(&down_device) {
            let mut view = output.slice_mut(0..gate_bank.in_f);
            engine.axpy_host_into(&row.slice(0..row.len()), weight, &mut view, gate_bank.in_f)?;
        }
        engine.dtoh(&output)?
    };
    let accumulation_delta = numeric_delta(&host_accumulation, &device_accumulation);
    let accumulation_repeat_delta =
        numeric_delta(&device_accumulation, &device_accumulation_repeat);
    let first = accumulation_delta
        .first
        .map(|(index, host, device)| {
            format!("index={index} host=0x{host:08x} device=0x{device:08x}")
        })
        .unwrap_or_else(|| "none".to_string());
    println!(
        "EP_DEVICE_NUMERIC kind=accumulation bit_mismatches={}/{} max_abs={:.9e} \
         max_ulp={} first=\"{}\" repeat_mismatches={}/{}",
        accumulation_delta.mismatches,
        host_accumulation.len(),
        accumulation_delta.max_abs,
        accumulation_delta.max_ulp,
        first,
        accumulation_repeat_delta.mismatches,
        host_accumulation.len(),
    );
    let host_exact_accumulation_delta = numeric_delta(&host_accumulation, &host_exact_accumulation);
    let host_exact_accumulation_repeat_delta =
        numeric_delta(&host_exact_accumulation, &host_exact_accumulation_repeat);
    let first = host_exact_accumulation_delta
        .first
        .map(|(index, host, device)| {
            format!("index={index} host=0x{host:08x} device=0x{device:08x}")
        })
        .unwrap_or_else(|| "none".to_string());
    println!(
        "EP_DEVICE_NUMERIC kind=accumulation_host_exact bit_mismatches={}/{} \
         max_abs={:.9e} max_ulp={} first=\"{}\" repeat_mismatches={}/{}",
        host_exact_accumulation_delta.mismatches,
        host_accumulation.len(),
        host_exact_accumulation_delta.max_abs,
        host_exact_accumulation_delta.max_ulp,
        first,
        host_exact_accumulation_repeat_delta.mismatches,
        host_accumulation.len(),
    );

    let gate_d = timing_gate.ok_or("numeric probe has no timing gate")?;
    let up_d = timing_up.ok_or("numeric probe has no timing up")?;
    let host_activation_us = {
        let _main = engine.gpu.enter_main()?;
        let start = Instant::now();
        let mut checksum = 0u32;
        for _ in 0..iterations {
            let gate = engine.dtoh(&gate_d)?;
            let up = engine.dtoh(&up_d)?;
            let activation = gate
                .iter()
                .zip(&up)
                .map(|(&gate, &up)| step_expert_activation_host(gate, up, activation_limit))
                .collect::<Vec<_>>();
            checksum ^= activation[activation.len() / 2].to_bits();
            let activation_d = engine.htod(&activation)?;
            engine.stream().synchronize()?;
            black_box(activation_d.len());
        }
        black_box(checksum);
        per_iteration_us(start, iterations)
    };
    let device_activation_us = {
        let _main = engine.gpu.enter_main()?;
        let mut activation = engine.zeros(gate_d.len())?;
        if let Some(limit) = activation_limit {
            engine.swiglu_clamped_mul_scaled(
                &gate_d,
                &up_d,
                1.0,
                1.0,
                limit,
                &mut activation,
                gate_d.len(),
            )?;
        } else {
            engine.silu_mul(&gate_d, &up_d, &mut activation, gate_d.len())?;
        }
        engine.stream().synchronize()?;
        let start = Instant::now();
        for _ in 0..iterations {
            if let Some(limit) = activation_limit {
                engine.swiglu_clamped_mul_scaled(
                    &gate_d,
                    &up_d,
                    1.0,
                    1.0,
                    limit,
                    &mut activation,
                    gate_d.len(),
                )?;
            } else {
                engine.silu_mul(&gate_d, &up_d, &mut activation, gate_d.len())?;
            }
            engine.stream().synchronize()?;
        }
        black_box(&activation);
        per_iteration_us(start, iterations)
    };
    let host_exact_activation_us = {
        let _main = engine.gpu.enter_main()?;
        let mut activation = engine.zeros(gate_d.len())?;
        if let Some(limit) = activation_limit {
            engine.silu_clamped_mul_host_expf(
                &gate_d,
                &up_d,
                limit,
                &mut activation,
                gate_d.len(),
            )?;
        } else {
            engine.silu_mul_host_expf(&gate_d, &up_d, &mut activation, gate_d.len())?;
        }
        engine.stream().synchronize()?;
        let start = Instant::now();
        for _ in 0..iterations {
            if let Some(limit) = activation_limit {
                engine.silu_clamped_mul_host_expf(
                    &gate_d,
                    &up_d,
                    limit,
                    &mut activation,
                    gate_d.len(),
                )?;
            } else {
                engine.silu_mul_host_expf(&gate_d, &up_d, &mut activation, gate_d.len())?;
            }
            engine.stream().synchronize()?;
        }
        black_box(&activation);
        per_iteration_us(start, iterations)
    };
    let host_accumulation_us = {
        let _main = engine.gpu.enter_main()?;
        let start = Instant::now();
        let mut checksum = 0u32;
        for _ in 0..iterations {
            let mut output = vec![0.0f32; gate_bank.in_f];
            for (&weight, row) in weights.iter().zip(&down_device) {
                let row = engine.dtoh(row)?;
                for (sum, value) in output.iter_mut().zip(row) {
                    *sum += weight * value;
                }
            }
            checksum ^= output[output.len() / 2].to_bits();
        }
        black_box(checksum);
        per_iteration_us(start, iterations)
    };
    let device_accumulation_us = {
        let _main = engine.gpu.enter_main()?;
        let zero = vec![0.0f32; gate_bank.in_f];
        let start = Instant::now();
        for _ in 0..iterations {
            let mut output = engine.htod(&zero)?;
            for (&weight, row) in weights.iter().zip(&down_device) {
                let mut view = output.slice_mut(0..gate_bank.in_f);
                engine.axpy_into(row, weight, &mut view, gate_bank.in_f)?;
            }
            engine.stream().synchronize()?;
            black_box(&output);
        }
        per_iteration_us(start, iterations)
    };
    let host_exact_accumulation_us = {
        let _main = engine.gpu.enter_main()?;
        let zero = vec![0.0f32; gate_bank.in_f];
        let start = Instant::now();
        for _ in 0..iterations {
            let mut output = engine.htod(&zero)?;
            for (&weight, row) in weights.iter().zip(&down_device) {
                let mut view = output.slice_mut(0..gate_bank.in_f);
                engine.axpy_host_into(
                    &row.slice(0..row.len()),
                    weight,
                    &mut view,
                    gate_bank.in_f,
                )?;
            }
            engine.stream().synchronize()?;
            black_box(&output);
        }
        per_iteration_us(start, iterations)
    };
    let cuda_fma_accumulation_exact =
        accumulation_delta.mismatches == 0 && accumulation_repeat_delta.mismatches == 0;
    let activation_exact = all_host_exact_activation_exact;
    let accumulation_exact = host_exact_accumulation_delta.mismatches == 0
        && host_exact_accumulation_repeat_delta.mismatches == 0;
    println!(
        "STEP_EP_DEVICE_NUMERIC_PROBE \
         cuda_expf_activation_exact={all_cuda_expf_activation_exact} \
         cuda_fma_accumulation_exact={cuda_fma_accumulation_exact} \
         activation_exact={activation_exact} accumulation_exact={accumulation_exact} \
         iterations={iterations} \
         activation_host_stage_us={host_activation_us:.3} \
         activation_device_kernel_us={device_activation_us:.3} \
         activation_host_exact_kernel_us={host_exact_activation_us:.3} \
         accumulation_host_stage_us={host_accumulation_us:.3} \
         accumulation_device_kernel_us={device_accumulation_us:.3} \
         accumulation_host_exact_kernel_us={host_exact_accumulation_us:.3} \
         performance_claim=false"
    );
    Ok(activation_exact && accumulation_exact)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = std::env::args()
        .nth(1)
        .expect("usage: ep-step-fp8-gate <official-step-safetensors-dir> [layer]");
    let layer = std::env::args()
        .nth(2)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(3);
    let tokens = std::env::var("MEMRA_TP_TOKENS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1);

    let source = SafetensorsSource::open(std::path::Path::new(&model))?;
    let config = source.config();
    let contract = ModelParallelContract::from_model(&config)?;
    let qualified = validate_step_fp8_checkpoint(&source, &contract)?;
    let activation_limit = post_clamp_only(config.clamp_exp_at(layer as u32), layer)?;
    let transport_only = strict_bool("MEMRA_STEP_EP_TRANSPORT_ONLY")?;
    let grouped_fp8_gate = strict_bool("MEMRA_STEP_EP_GROUPED_FP8_GATE")?;
    let owner_grouped_fp8_gate = strict_bool("MEMRA_STEP_EP_OWNER_GROUPED_FP8_GATE")?;
    let dynamic_owner_grouped_fp8_gate =
        strict_bool("MEMRA_STEP_EP_DYNAMIC_OWNER_GROUPED_FP8_GATE")?;
    let dynamic_owner_grouped_fp8_combine_gate =
        strict_bool("MEMRA_STEP_EP_DYNAMIC_OWNER_GROUPED_FP8_COMBINE_GATE")?;
    let dynamic_owner_grouped_fp8_device_handoff_gate =
        strict_bool("MEMRA_STEP_EP_DYNAMIC_OWNER_GROUPED_FP8_DEVICE_HANDOFF_GATE")?;
    let attention_to_routed_ep_gate = strict_bool("MEMRA_STEP_ATTENTION_TO_ROUTED_EP_GATE")?;
    let full_decoder_layer_gate = strict_bool("MEMRA_STEP_FULL_DECODER_LAYER_GATE")?;
    let native_p2p = memra_engine::tp::step_tp_native_p2p_enabled()?;
    let device_arithmetic = memra_engine::tp::step_ep_device_arithmetic_enabled()?;
    if device_arithmetic && !native_p2p {
        return Err("MEMRA_STEP_EP_DEVICE_ARITHMETIC=1 requires MEMRA_STEP_TP_NATIVE_P2P=1".into());
    }
    if (owner_grouped_fp8_gate
        || dynamic_owner_grouped_fp8_gate
        || dynamic_owner_grouped_fp8_combine_gate
        || dynamic_owner_grouped_fp8_device_handoff_gate
        || attention_to_routed_ep_gate
        || full_decoder_layer_gate)
        && (!native_p2p || !device_arithmetic)
    {
        return Err("Step owner-grouped FP8 gates require native P2P and \
             MEMRA_STEP_EP_DEVICE_ARITHMETIC=1"
            .into());
    }
    let devices = devices(transport_only)?;
    let world = if transport_only && !matches!(devices.len(), 2 | 4 | 8) {
        devices.len()
    } else {
        contract
            .plan(TopologyRequest {
                pipeline: 1,
                tensor: devices.len(),
                expert_parallel: true,
                available_devices: devices.len(),
                hardware: HardwareTarget::RtxPro6000Blackwell,
            })?
            .world_size
    };
    let runtime = if device_arithmetic {
        TpE4m3HostBounce::new_native_p2p_device_arithmetic(&devices)?
    } else if native_p2p {
        TpE4m3HostBounce::new_native_p2p(&devices)?
    } else {
        TpE4m3HostBounce::new(&devices)?
    };
    let names = runtime.device_names()?;
    if names
        .iter()
        .any(|name| !name.contains("RTX PRO 6000") || !name.contains("Blackwell"))
    {
        return Err(format!("unqualified EP hardware: {names:?}").into());
    }

    let name = |projection: &str| format!("blk.{layer}.ffn_{projection}_exps.weight");
    let gate = source
        .find_fp8_stacked_native(&name("gate"))
        .ok_or_else(|| format!("missing native E4M3 {}", name("gate")))?;
    let up = source
        .find_fp8_stacked_native(&name("up"))
        .ok_or_else(|| format!("missing native E4M3 {}", name("up")))?;
    let down = source
        .find_fp8_stacked_native(&name("down"))
        .ok_or_else(|| format!("missing native E4M3 {}", name("down")))?;
    let resident =
        runtime.upload_expert_parallel(expert_bank(&gate), expert_bank(&up), expert_bank(&down))?;
    let input = activations(tokens, gate.in_f);
    let per_rank = gate.n_expert / devices.len();
    let experts = (0..devices.len())
        .map(|owner| owner * per_rank)
        .collect::<Vec<_>>();
    let weight_sum = (1..=devices.len()).sum::<usize>() as f32;
    let weights = (1..=devices.len())
        .map(|weight| weight as f32 / weight_sum)
        .collect::<Vec<_>>();
    let mut expected = vec![0.0f32; tokens * gate.in_f];
    let mut selected = Vec::with_capacity(tokens * experts.len());
    let mut route_weights = Vec::with_capacity(tokens * experts.len());
    for token in 0..tokens {
        let input_row = &input[token * gate.in_f..(token + 1) * gate.in_f];
        for (&expert, &weight) in experts.iter().zip(&weights) {
            let result = reference_expert(
                &runtime,
                &gate,
                &up,
                &down,
                expert,
                input_row,
                activation_limit,
            )?;
            for (sum, value) in expected[token * gate.in_f..(token + 1) * gate.in_f]
                .iter_mut()
                .zip(result)
            {
                *sum += weight * value;
            }
            selected.push(expert);
            route_weights.push(weight);
        }
    }
    let first = runtime.run_routed_experts(
        &resident,
        &input,
        tokens,
        &selected,
        &route_weights,
        experts.len(),
        activation_limit,
    )?;
    let second = runtime.run_routed_experts(
        &resident,
        &input,
        tokens,
        &selected,
        &route_weights,
        experts.len(),
        activation_limit,
    )?;
    compare_exact("reference", &expected, &first)?;
    compare_exact("resident_repeat", &first, &second)?;
    if grouped_fp8_gate {
        run_grouped_fp8_gate(
            &runtime,
            &gate,
            &up,
            &down,
            &input,
            tokens,
            layer,
            activation_limit,
        )?;
    }
    if owner_grouped_fp8_gate {
        run_owner_grouped_fp8_gate(
            &runtime,
            &resident,
            &gate,
            &up,
            &down,
            &input,
            tokens,
            layer,
            activation_limit,
        )?;
    }
    if dynamic_owner_grouped_fp8_gate {
        run_dynamic_owner_grouped_fp8_gate(
            &runtime,
            &resident,
            &gate,
            &up,
            &down,
            &input,
            tokens,
            layer,
            activation_limit,
        )?;
    }
    if dynamic_owner_grouped_fp8_combine_gate {
        run_dynamic_owner_grouped_fp8_combine_gate(
            &runtime,
            &resident,
            &gate,
            &up,
            &down,
            &input,
            tokens,
            layer,
            activation_limit,
        )?;
    }
    if dynamic_owner_grouped_fp8_device_handoff_gate {
        run_dynamic_owner_grouped_fp8_device_handoff_gate(
            &runtime,
            &resident,
            &gate,
            &up,
            &down,
            &input,
            tokens,
            layer,
            activation_limit,
        )?;
    }
    if attention_to_routed_ep_gate {
        run_attention_to_routed_expert_gate(
            &source,
            &runtime,
            &resident,
            &gate,
            &up,
            &down,
            &input,
            tokens,
            layer,
            activation_limit,
        )?;
    }
    if full_decoder_layer_gate {
        run_full_decoder_layer_gate(
            &source,
            &runtime,
            &resident,
            &gate,
            &up,
            &down,
            &input,
            tokens,
            layer,
            activation_limit,
        )?;
    }
    if strict_bool("MEMRA_STEP_EP_DEVICE_NUMERIC_PROBE")?
        && !probe_device_numerics(
            &runtime,
            &gate,
            &up,
            &down,
            &experts,
            &weights,
            &input,
            activation_limit,
        )?
    {
        return Err("device-resident EP numerics differ from the host-canonical oracle".into());
    }
    if let Some((warmup, iterations)) = benchmark_config()? {
        benchmark_routed_experts(
            &runtime,
            &resident,
            &input,
            tokens,
            &selected,
            &route_weights,
            experts.len(),
            &expected,
            warmup,
            iterations,
            activation_limit,
        )?;
    }

    println!(
        "STEP_EP{}_FP8_{}_PASS variant={} checkpoint_fp8_projections={qualified} \
         layer={layer} tokens={tokens} experts={experts:?} owners=0..{} world={} \
         expert_layout=expert-parallel resident_weights=true expert_transport={} \
         native_p2p={} activation={} accumulation={} output={} \
         routed_clamp={} \
         product_topology={} tensor_parallel_claim=false performance_claim=false",
        devices.len(),
        if transport_only && !matches!(devices.len(), 2 | 4 | 8) {
            "TRANSPORT"
        } else {
            "GATE"
        },
        contract.variant,
        devices.len(),
        world,
        runtime.transport_label(),
        runtime.native_p2p(),
        runtime.expert_activation_label(),
        runtime.expert_accumulation_label(),
        runtime.expert_output_label(),
        activation_limit.unwrap_or(0.0),
        matches!(devices.len(), 2 | 4 | 8),
    );
    Ok(())
}
