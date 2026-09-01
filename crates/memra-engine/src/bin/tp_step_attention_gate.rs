//! Official Step-3.7 attention and KV-cache tensor-parallel correctness gate.
//!
//! Q/K/V projections remain resident on their owning ranks through QK norm, RoPE, head-local
//! attention, and head-wise gating. Rank outputs are gathered only after attention, then the
//! already-qualified canonical Step O reduction runs. The stateful arm appends rank-local K/V
//! shards into persistent quantized cache planes over multiple calls and reconstructs their bytes
//! only for an exact TP1 oracle. The gate projection and small metadata remain host-staged.

use cudarc::driver::CudaSlice;
use memra_engine::model::GpuTensor;
use memra_engine::parallel::{
    HardwareTarget, ModelParallelContract, TopologyRequest, validate_step_fp8_checkpoint,
};
use memra_engine::tp::{
    Bf16Matrix, ResidentBf16ColumnParallel, ResidentStepBf16RowParallel, ResidentTpKvCache,
    TpE4m3HostBounce, TpKvTransaction, step_tp_f32_mirror_enabled,
};
use memra_engine::{Engine, kv_cache_formats};
use memra_gguf::GgmlType;
use memra_gguf::config::LayerGeometry;
use memra_gguf::source::{SafetensorsSource, TensorSource};

fn devices() -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    let raw = std::env::var("MEMRA_TP_DEVICES").unwrap_or_else(|_| "0,1".to_string());
    let devices = raw
        .split(',')
        .map(|part| part.trim().parse::<usize>())
        .collect::<Result<Vec<_>, _>>()?;
    if !(2..=8).contains(&devices.len()) {
        return Err(format!("Step attention TP gate requires 2..=8 ranks, got {raw:?}").into());
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
            "TP_ATTENTION_EXACT label={label} bit_mismatches={mismatches}/{} \
             max_abs={max_abs:.9e} relative_to_peak={relative_to_peak:.9e} \
             first_index={index} expected={left:.9e} actual={right:.9e} \
             expected_bits={:08x} actual_bits={:08x}",
            expected.len(),
            left.to_bits(),
            right.to_bits(),
        );
    } else {
        println!(
            "TP_ATTENTION_EXACT label={label} bit_mismatches=0/{} \
             max_abs=0.000000000e0 relative_to_peak=0.000000000e0",
            expected.len()
        );
    }
    if mismatches != 0 {
        return Err(format!(
            "{label}: attention output differs from its exact oracle"
        ));
    }
    Ok(())
}

fn compare_bytes_exact(label: &str, expected: &[u8], actual: &[u8]) -> Result<(), String> {
    if expected.len() != actual.len() {
        return Err(format!(
            "{label}: cache lengths differ, {} != {}",
            expected.len(),
            actual.len()
        ));
    }
    let first = expected
        .iter()
        .zip(actual)
        .enumerate()
        .find(|(_, (left, right))| left != right);
    if let Some((index, (&left, &right))) = first {
        let mismatches = expected
            .iter()
            .zip(actual)
            .filter(|(left, right)| left != right)
            .count();
        println!(
            "TP_CACHE_BYTES_EXACT label={label} byte_mismatches={mismatches}/{} \
             first_index={index} expected={left:02x} actual={right:02x}",
            expected.len()
        );
        return Err(format!("{label}: quantized cache bytes differ"));
    }
    println!(
        "TP_CACHE_BYTES_EXACT label={label} byte_mismatches=0/{}",
        expected.len()
    );
    Ok(())
}

struct ResidentAttention {
    q: ResidentBf16ColumnParallel,
    k: ResidentBf16ColumnParallel,
    v: ResidentBf16ColumnParallel,
    o: ResidentStepBf16RowParallel,
}

struct AttentionInputs<'a> {
    activations: &'a [f32],
    gate: &'a [f32],
    q_norm: &'a [f32],
    k_norm: &'a [f32],
    rope_factors: Option<&'a [f32]>,
    positions: &'a [i32],
    tokens: usize,
    geometry: LayerGeometry,
    rms_eps: f32,
}

fn gate_shard(gate: &[f32], tokens: usize, heads: usize, tp: usize, rank: usize) -> Vec<f32> {
    let local_heads = heads / tp;
    let mut shard = Vec::with_capacity(tokens * local_heads);
    for token in 0..tokens {
        let start = token * heads + rank * local_heads;
        shard.extend_from_slice(&gate[start..start + local_heads]);
    }
    shard
}

#[allow(clippy::manual_is_multiple_of)] // allow: divisor is runtime-derived; the modulo form keeps a zero divisor loud (a panic), where is_multiple_of would return false silently
fn run_cacheless_attention(
    runtime: &TpE4m3HostBounce,
    resident: &ResidentAttention,
    input: &AttentionInputs<'_>,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let tp = runtime.devices().len();
    let heads = input.geometry.n_head as usize;
    let kv_heads = input.geometry.n_head_kv as usize;
    let head_dim = input.geometry.head_dim_k as usize;
    if heads % tp != 0 || kv_heads % tp != 0 {
        return Err(format!(
            "attention heads q={heads} kv={kv_heads} are not divisible by TP={tp}"
        )
        .into());
    }
    let local_heads = heads / tp;
    let local_kv_heads = kv_heads / tp;
    let q_shards = runtime.bf16_column_parallel_resident_device_shards(
        &resident.q,
        input.activations,
        input.tokens,
    )?;
    let k_shards = runtime.bf16_column_parallel_resident_device_shards(
        &resident.k,
        input.activations,
        input.tokens,
    )?;
    let v_shards = runtime.bf16_column_parallel_resident_device_shards(
        &resident.v,
        input.activations,
        input.tokens,
    )?;
    let mut attention_shards = Vec::<CudaSlice<f32>>::with_capacity(tp);
    for rank in 0..tp {
        let engine = runtime
            .rank_engine(rank)
            .ok_or_else(|| format!("TP rank {rank} has no engine"))?;
        let _main = engine.gpu.enter_main()?;
        let q_norm_weight = engine.htod(input.q_norm)?;
        let k_norm_weight = engine.htod(input.k_norm)?;
        let positions = engine.htod_i32(input.positions)?;
        let rope_factors = input
            .rope_factors
            .map(|factors| engine.htod(factors))
            .transpose()?;
        let mut q = engine.uninit(input.tokens * local_heads * head_dim)?;
        engine.rms_norm(
            &q_shards[rank],
            &q_norm_weight,
            &mut q,
            head_dim,
            local_heads * input.tokens,
            input.rms_eps,
        )?;
        let mut k = engine.uninit(input.tokens * local_kv_heads * head_dim)?;
        engine.rms_norm(
            &k_shards[rank],
            &k_norm_weight,
            &mut k,
            head_dim,
            local_kv_heads * input.tokens,
            input.rms_eps,
        )?;
        engine.rope_neox2(
            &mut q,
            &mut k,
            &positions,
            head_dim,
            input.geometry.n_rot as usize,
            local_heads,
            local_kv_heads,
            input.tokens,
            input.geometry.rope_base,
            1.0,
            rope_factors.as_ref(),
        )?;
        let mut attention = engine.uninit(input.tokens * local_heads * head_dim)?;
        let window = input.geometry.window.map(|window| window as usize);
        if let Some(window) = window.filter(|&window| input.tokens > window) {
            engine.sdpa_naive_w(
                &q,
                &k,
                &v_shards[rank],
                &mut attention,
                head_dim,
                local_heads,
                local_kv_heads,
                input.tokens,
                input.tokens,
                input.geometry.attention_scale(),
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
                input.tokens,
                input.tokens,
                input.geometry.attention_scale(),
                true,
            )?;
        }
        let gate = engine.htod(&gate_shard(input.gate, input.tokens, heads, tp, rank))?;
        let mut gated = engine.uninit(input.tokens * local_heads * head_dim)?;
        engine.attn_head_gate(
            &attention,
            &gate,
            &mut gated,
            None,
            head_dim,
            local_heads,
            input.tokens,
        )?;
        attention_shards.push(gated);
    }
    runtime.gather_native_column_shards(&attention_shards, input.tokens, local_heads * head_dim)
}

struct CacheBytes {
    k: Vec<u8>,
    v: Vec<u8>,
}

struct CacheGrowRun {
    cache: CacheBytes,
    rows: usize,
    source_capacity: usize,
    target_capacity: usize,
    next_generation: u64,
}

struct CachedAttentionRun {
    attention: Vec<f32>,
    output: Vec<f32>,
    speculative_attention: Vec<f32>,
    rollback_attention: Vec<f32>,
    recommit_attention: Vec<f32>,
    speculative_output: Vec<f32>,
    rollback_output: Vec<f32>,
    recommit_output: Vec<f32>,
    partial_cache: CacheBytes,
    rollback_cache: CacheBytes,
    final_cache: CacheBytes,
    k_tok_bytes: usize,
    v_tok_bytes: usize,
    logical_capacity: usize,
    ring_window: Option<usize>,
    physical_rows: usize,
    grow: Option<CacheGrowRun>,
}

fn gather_cache_rows(
    rank_rows: &[Vec<u8>],
    tokens: usize,
    local_tok_bytes: usize,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut gathered = Vec::with_capacity(tokens * local_tok_bytes * rank_rows.len());
    for token in 0..tokens {
        let start = token * local_tok_bytes;
        let end = start + local_tok_bytes;
        for (rank, rows) in rank_rows.iter().enumerate() {
            if rows.len() != tokens * local_tok_bytes {
                return Err(format!(
                    "rank {rank} cache bytes {} != {tokens}x{local_tok_bytes}",
                    rows.len()
                )
                .into());
            }
            gathered.extend_from_slice(&rows[start..end]);
        }
    }
    Ok(gathered)
}

fn read_cache_bytes(
    runtime: &TpE4m3HostBounce,
    cache: &ResidentTpKvCache,
    tokens: usize,
) -> Result<CacheBytes, Box<dyn std::error::Error>> {
    if tokens > cache.committed_len() {
        return Err(format!(
            "cache read requests {tokens} committed rows from length {}",
            cache.committed_len()
        )
        .into());
    }
    let mut k_rank_rows = Vec::with_capacity(runtime.devices().len());
    let mut v_rank_rows = Vec::with_capacity(runtime.devices().len());
    for rank in 0..runtime.devices().len() {
        let engine = runtime
            .rank_engine(rank)
            .ok_or_else(|| format!("TP rank {rank} has no engine"))?;
        let rank_cache = cache
            .rank(rank)
            .ok_or_else(|| format!("TP rank {rank} has no KV cache"))?;
        let _main = engine.gpu.enter_main()?;
        let physical = cache.physical_range(0, tokens)?;
        let k = engine.view_u8_range(
            rank_cache.k(),
            physical.start * cache.k_tok_bytes(),
            physical.end * cache.k_tok_bytes(),
        );
        let v = engine.view_u8_range(
            rank_cache.v(),
            physical.start * cache.v_tok_bytes(),
            physical.end * cache.v_tok_bytes(),
        );
        k_rank_rows.push(engine.dtoh_u8_view(&k)?);
        v_rank_rows.push(engine.dtoh_u8_view(&v)?);
    }
    Ok(CacheBytes {
        k: gather_cache_rows(&k_rank_rows, tokens, cache.k_tok_bytes())?,
        v: gather_cache_rows(&v_rank_rows, tokens, cache.v_tok_bytes())?,
    })
}

fn check_cache_state(
    label: &str,
    runtime: &TpE4m3HostBounce,
    cache: &ResidentTpKvCache,
    committed: usize,
    staged: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    if cache.committed_len() != committed || cache.staged_len() != staged {
        return Err(format!(
            "{label}: host cache state committed/staged={}/{} != {committed}/{staged}",
            cache.committed_len(),
            cache.staged_len()
        )
        .into());
    }
    let lengths = runtime.tp_kv_device_lengths(cache)?;
    let expected = i32::try_from(staged)?;
    if lengths.iter().any(|&length| length != expected) {
        return Err(
            format!("{label}: device cache lengths {lengths:?} != staged length {staged}").into(),
        );
    }
    println!(
        "TP_CACHE_TX_STATE label={label} committed={committed} staged={staged} \
         device_lengths={lengths:?}"
    );
    Ok(())
}

fn run_cached_token(
    runtime: &TpE4m3HostBounce,
    resident: &ResidentAttention,
    input: &AttentionInputs<'_>,
    token: usize,
    cache: &mut ResidentTpKvCache,
    transaction: TpKvTransaction,
) -> Result<(Vec<f32>, Vec<f32>), Box<dyn std::error::Error>> {
    let tp = runtime.devices().len();
    let heads = input.geometry.n_head as usize;
    let kv_heads = input.geometry.n_head_kv as usize;
    let head_dim = input.geometry.head_dim_k as usize;
    let local_heads = heads / tp;
    let local_kv_heads = kv_heads / tp;
    let local_kv_dim = local_kv_heads * head_dim;
    let hidden = input.activations.len() / input.tokens;
    let activation_start = token * hidden;
    let activation_end = activation_start + hidden;
    let activation = &input.activations[activation_start..activation_end];
    let q_shards =
        runtime.bf16_column_parallel_resident_device_shards(&resident.q, activation, 1)?;
    let k_shards =
        runtime.bf16_column_parallel_resident_device_shards(&resident.k, activation, 1)?;
    let v_shards =
        runtime.bf16_column_parallel_resident_device_shards(&resident.v, activation, 1)?;
    let mut q_normed = Vec::with_capacity(tp);
    let mut k_normed = Vec::with_capacity(tp);
    for rank in 0..tp {
        let engine = runtime
            .rank_engine(rank)
            .ok_or_else(|| format!("TP rank {rank} has no engine"))?;
        let _main = engine.gpu.enter_main()?;
        let q_norm_weight = engine.htod(input.q_norm)?;
        let k_norm_weight = engine.htod(input.k_norm)?;
        let position = engine.htod_i32(&[input.positions[token]])?;
        let rope_factors = input
            .rope_factors
            .map(|factors| engine.htod(factors))
            .transpose()?;
        let mut q = engine.uninit(local_heads * head_dim)?;
        engine.rms_norm(
            &q_shards[rank],
            &q_norm_weight,
            &mut q,
            head_dim,
            local_heads,
            input.rms_eps,
        )?;
        let mut k = engine.uninit(local_kv_dim)?;
        engine.rms_norm(
            &k_shards[rank],
            &k_norm_weight,
            &mut k,
            head_dim,
            local_kv_heads,
            input.rms_eps,
        )?;
        engine.rope_neox2(
            &mut q,
            &mut k,
            &position,
            head_dim,
            input.geometry.n_rot as usize,
            local_heads,
            local_kv_heads,
            1,
            input.geometry.rope_base,
            1.0,
            rope_factors.as_ref(),
        )?;
        q_normed.push(q);
        k_normed.push(k);
    }
    runtime.append_tp_kv_transaction(cache, transaction, &k_normed, &v_shards, 1)?;
    let view_start = cache
        .ring_window()
        .map(|window| cache.staged_len().saturating_sub(window))
        .unwrap_or(0);
    let physical = cache.physical_range(view_start, cache.staged_len())?;
    let t_kv = cache.staged_len() - view_start;

    let mut attention_shards = Vec::<CudaSlice<f32>>::with_capacity(tp);
    #[allow(clippy::needless_range_loop)]
    // allow: the explicit index loop keeps the offset arithmetic visible and aligned with the device-side indexing
    for rank in 0..tp {
        let engine = runtime
            .rank_engine(rank)
            .ok_or_else(|| format!("TP rank {rank} has no engine"))?;
        let rank_cache = cache
            .rank(rank)
            .ok_or_else(|| format!("TP rank {rank} has no KV cache"))?;
        let _main = engine.gpu.enter_main()?;
        let k_view = engine.view_u8_range(
            rank_cache.k(),
            physical.start * cache.k_tok_bytes(),
            physical.end * cache.k_tok_bytes(),
        );
        let v_view = engine.view_u8_range(
            rank_cache.v(),
            physical.start * cache.v_tok_bytes(),
            physical.end * cache.v_tok_bytes(),
        );
        let mut attention = engine.uninit(local_heads * head_dim)?;
        engine.fa_decode_kvmod(
            &q_normed[rank],
            &k_view,
            &v_view,
            &mut attention,
            head_dim,
            local_heads,
            local_kv_heads,
            t_kv,
            input.geometry.attention_scale(),
            cache.k_tok_bytes(),
            cache.v_tok_bytes(),
            false,
        )?;
        let gate_start = token * heads + rank * local_heads;
        let gate = engine.htod(&input.gate[gate_start..gate_start + local_heads])?;
        let mut gated = engine.uninit(local_heads * head_dim)?;
        engine.attn_head_gate(
            &attention,
            &gate,
            &mut gated,
            None,
            head_dim,
            local_heads,
            1,
        )?;
        attention_shards.push(gated);
    }
    let attention =
        runtime.gather_native_column_shards(&attention_shards, 1, local_heads * head_dim)?;
    let output = runtime.step_bf16_row_parallel_resident_native(&resident.o, &attention, 1)?;
    Ok((attention, output))
}

#[allow(clippy::manual_is_multiple_of)] // allow: divisor is runtime-derived; the modulo form keeps a zero divisor loud (a panic), where is_multiple_of would return false silently
fn run_cached_attention_sequence(
    runtime: &TpE4m3HostBounce,
    resident: &ResidentAttention,
    input: &AttentionInputs<'_>,
    grow: bool,
    swa: bool,
) -> Result<CachedAttentionRun, Box<dyn std::error::Error>> {
    let tp = runtime.devices().len();
    let heads = input.geometry.n_head as usize;
    let kv_heads = input.geometry.n_head_kv as usize;
    let head_dim = input.geometry.head_dim_k as usize;
    if heads % tp != 0 || kv_heads % tp != 0 {
        return Err(format!(
            "attention heads q={heads} kv={kv_heads} are not divisible by TP={tp}"
        )
        .into());
    }
    if input.tokens != 3 || input.activations.len() % input.tokens != 0 {
        return Err(format!(
            "transactional cache gate requires exactly 3 tokens, got {}",
            input.tokens
        )
        .into());
    }
    let hidden = input.activations.len() / input.tokens;
    let (k_format, v_format) = kv_cache_formats();
    if (k_format, v_format) != ("q8_0", "q5_1") || Engine::kv_fp8_on() {
        return Err(format!(
            "cache gate requires q8_0/q5_1 without FP8 cache, got {k_format}/{v_format} fp8={}",
            Engine::kv_fp8_on()
        )
        .into());
    }
    let cache_capacity = if swa {
        let window = input
            .geometry
            .window
            .ok_or("SWA cache gate requires an official sliding-window layer")?
            as usize;
        window
            .checked_add(memra_engine::cache::PRIME_CHUNK_MAX_TOKENS + 64)
            .ok_or("SWA cache gate capacity overflow")?
    } else {
        input.tokens
    };
    let mut cache = if swa {
        runtime.allocate_tp_swa_kv_cache(
            kv_heads * head_dim,
            kv_heads * head_dim,
            cache_capacity,
            input.geometry.window.unwrap() as usize,
        )?
    } else {
        runtime.allocate_tp_kv_cache(kv_heads * head_dim, kv_heads * head_dim, cache_capacity)?
    };

    let first = cache.begin_transaction()?;
    let (attention0, output0) = run_cached_token(runtime, resident, input, 0, &mut cache, first)?;
    runtime.commit_tp_kv_transaction(&mut cache, first, 1)?;
    check_cache_state("first-commit", runtime, &cache, 1, 1)?;
    if runtime
        .commit_tp_kv_transaction(&mut cache, first, 1)
        .is_ok()
    {
        return Err("double-finalized TP KV transaction unexpectedly committed".into());
    }

    let partial = cache.begin_transaction()?;
    if cache.begin_transaction().is_ok() {
        return Err("nested TP KV transaction unexpectedly started".into());
    }
    let (attention1, output1) = run_cached_token(runtime, resident, input, 1, &mut cache, partial)?;
    let (speculative_attention, speculative_output) =
        run_cached_token(runtime, resident, input, 2, &mut cache, partial)?;
    check_cache_state("partial-staged", runtime, &cache, 1, 3)?;
    runtime.commit_tp_kv_transaction(&mut cache, partial, 1)?;
    check_cache_state("partial-commit", runtime, &cache, 2, 2)?;
    let partial_cache = read_cache_bytes(runtime, &cache, 2)?;

    let rollback = cache.begin_transaction()?;
    let (rollback_attention, rollback_output) =
        run_cached_token(runtime, resident, input, 2, &mut cache, rollback)?;
    check_cache_state("rollback-staged", runtime, &cache, 2, 3)?;
    runtime.rollback_tp_kv_transaction(&mut cache, rollback)?;
    check_cache_state("rollback-restored", runtime, &cache, 2, 2)?;
    if runtime
        .rollback_tp_kv_transaction(&mut cache, rollback)
        .is_ok()
    {
        return Err("double-finalized TP KV transaction unexpectedly rolled back".into());
    }
    let rollback_cache = read_cache_bytes(runtime, &cache, 2)?;

    let recommit = cache.begin_transaction()?;
    let (recommit_attention, recommit_output) =
        run_cached_token(runtime, resident, input, 2, &mut cache, recommit)?;
    check_cache_state("recommit-staged", runtime, &cache, 2, 3)?;
    runtime.commit_tp_kv_transaction(&mut cache, recommit, 1)?;
    check_cache_state("recommit-final", runtime, &cache, 3, 3)?;
    let final_cache = read_cache_bytes(runtime, &cache, 3)?;

    let generations = [
        first.generation(),
        partial.generation(),
        rollback.generation(),
        recommit.generation(),
    ];
    if generations != [1, 2, 3, 4] {
        return Err(
            format!("TP KV transaction generations {generations:?} are not monotonic").into(),
        );
    }
    println!(
        "TP_CACHE_TX_FAIL_CLOSED nested=true stale_commit=true stale_rollback=true \
         generations={generations:?}"
    );

    let grow = if grow {
        let rows = cache.committed_len();
        let source_capacity = cache.capacity();
        let target_capacity = source_capacity
            .checked_add(5)
            .ok_or("TP KV gate grow capacity overflow")?;
        let mut grown = runtime.grow_tp_kv_cache(&cache, target_capacity, rows)?;
        check_cache_state("grow-restored", runtime, &grown, rows, rows)?;
        let grown_cache = read_cache_bytes(runtime, &grown, rows)?;
        compare_bytes_exact(
            "tp_kv_grow_source_vs_target_k",
            &final_cache.k,
            &grown_cache.k,
        )?;
        compare_bytes_exact(
            "tp_kv_grow_source_vs_target_v",
            &final_cache.v,
            &grown_cache.v,
        )?;

        let next = grown.begin_transaction()?;
        let expected_generation = recommit
            .generation()
            .checked_add(1)
            .ok_or("TP KV gate generation overflow")?;
        if next.generation() != expected_generation || next.base_len() != rows {
            return Err(format!(
                "grown TP KV transaction generation/base={}/{} != {expected_generation}/{rows}",
                next.generation(),
                next.base_len()
            )
            .into());
        }
        runtime.rollback_tp_kv_transaction(&mut grown, next)?;
        check_cache_state("grow-generation-check", runtime, &grown, rows, rows)?;
        Some(CacheGrowRun {
            cache: grown_cache,
            rows,
            source_capacity,
            target_capacity,
            next_generation: next.generation(),
        })
    } else {
        None
    };

    let mut all_attention = Vec::with_capacity(input.tokens * heads * head_dim);
    all_attention.extend_from_slice(&attention0);
    all_attention.extend_from_slice(&attention1);
    all_attention.extend_from_slice(&recommit_attention);
    let mut all_output = Vec::with_capacity(input.tokens * hidden);
    all_output.extend_from_slice(&output0);
    all_output.extend_from_slice(&output1);
    all_output.extend_from_slice(&recommit_output);
    Ok(CachedAttentionRun {
        attention: all_attention,
        output: all_output,
        speculative_attention,
        rollback_attention,
        recommit_attention,
        speculative_output,
        rollback_output,
        recommit_output,
        partial_cache,
        rollback_cache,
        final_cache,
        k_tok_bytes: cache.k_tok_bytes() * tp,
        v_tok_bytes: cache.v_tok_bytes() * tp,
        logical_capacity: cache.capacity(),
        ring_window: cache.ring_window(),
        physical_rows: cache.physical_capacity(),
        grow,
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = std::env::args().nth(1).expect(
        "usage: tp-step-attention-gate <official-step-safetensors-dir> \
         [layer] [ambient-device]",
    );
    let layer = std::env::args()
        .nth(2)
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(3);
    let ambient_device = std::env::args()
        .nth(3)
        .map(|value| value.parse::<usize>())
        .transpose()?;
    let tokens = std::env::var("MEMRA_TP_TOKENS")
        .ok()
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(3);
    let f32_mirror = step_tp_f32_mirror_enabled()?;
    if tokens != 3 {
        return Err("MEMRA_TP_TOKENS must be 3 for the transaction gate".into());
    }
    let grow = match std::env::var("MEMRA_TP_GATE_GROW").ok().as_deref() {
        None | Some("") | Some("0") => false,
        Some("1") => true,
        Some(value) => {
            return Err(format!("MEMRA_TP_GATE_GROW={value:?} is invalid; expected 0 or 1").into());
        }
    };
    let swa = match std::env::var("MEMRA_TP_GATE_SWA").ok().as_deref() {
        None | Some("") | Some("0") => false,
        Some("1") => true,
        Some(value) => {
            return Err(format!("MEMRA_TP_GATE_SWA={value:?} is invalid; expected 0 or 1").into());
        }
    };

    let source = SafetensorsSource::open(std::path::Path::new(&model))?;
    let config = source.config();
    let contract = ModelParallelContract::from_model(&config)?;
    if layer >= contract.trunk_layers {
        return Err(format!(
            "Step attention TP layer {layer} is outside trunk layers 0..{}",
            contract.trunk_layers
        )
        .into());
    }
    let geometry = config
        .layer_geometry(layer as u32)
        .ok_or_else(|| format!("Step layer {layer} has no registered geometry"))?;
    if swa && geometry.window.is_none() {
        return Err(format!(
            "MEMRA_TP_GATE_SWA=1 requires an official sliding-window layer, got layer {layer}"
        )
        .into());
    }
    let qualified = validate_step_fp8_checkpoint(&source, &contract)?;
    let devices = devices()?;
    let plan = contract.plan(TopologyRequest {
        pipeline: 1,
        tensor: devices.len(),
        expert_parallel: devices.len() > 2,
        available_devices: devices.len(),
        hardware: HardwareTarget::RtxPro6000Blackwell,
    })?;
    for rank in 0..devices.len() {
        plan.query_head_range(layer, rank)
            .ok_or_else(|| format!("layer {layer} has no query-head range for rank {rank}"))?;
        plan.kv_head_range(layer, rank)
            .ok_or_else(|| format!("layer {layer} has no KV-head range for rank {rank}"))?;
    }

    let hidden = contract.hidden_size;
    let q_width = contract.query_heads[layer] * contract.head_dim;
    let kv_width = contract.kv_heads[layer] * contract.head_dim;
    let prefix = format!("blk.{layer}");
    let q_matrix = load_bf16_matrix(&source, &format!("{prefix}.attn_q.weight"), hidden, q_width)?;
    let k_matrix = load_bf16_matrix(
        &source,
        &format!("{prefix}.attn_k.weight"),
        hidden,
        kv_width,
    )?;
    let v_matrix = load_bf16_matrix(
        &source,
        &format!("{prefix}.attn_v.weight"),
        hidden,
        kv_width,
    )?;
    let o_matrix = load_bf16_matrix(
        &source,
        &format!("{prefix}.attn_output.weight"),
        q_width,
        hidden,
    )?;

    let runtime = TpE4m3HostBounce::new_native_p2p(&devices)?;
    let canonical_runtime = TpE4m3HostBounce::new_single_rank_oracle(devices[0])?;
    let names = runtime.device_names()?;
    if names
        .iter()
        .any(|name| !name.contains("RTX PRO 6000") || !name.contains("Blackwell"))
    {
        return Err(format!("unqualified TP hardware: {names:?}").into());
    }
    let resident = if f32_mirror {
        ResidentAttention {
            q: runtime.upload_step_bf16_column_parallel_f32_mirror(q_matrix.view())?,
            k: runtime.upload_step_bf16_column_parallel_f32_mirror(k_matrix.view())?,
            v: runtime.upload_step_bf16_column_parallel_f32_mirror(v_matrix.view())?,
            o: runtime.upload_step_bf16_row_parallel_f32_mirror(o_matrix.view())?,
        }
    } else {
        ResidentAttention {
            q: runtime.upload_step_bf16_column_parallel(q_matrix.view())?,
            k: runtime.upload_step_bf16_column_parallel(k_matrix.view())?,
            v: runtime.upload_step_bf16_column_parallel(v_matrix.view())?,
            o: runtime.upload_step_bf16_row_parallel(o_matrix.view())?,
        }
    };
    let canonical_resident = ResidentAttention {
        q: canonical_runtime.upload_step_bf16_column_parallel(q_matrix.view())?,
        k: canonical_runtime.upload_step_bf16_column_parallel(k_matrix.view())?,
        v: canonical_runtime.upload_step_bf16_column_parallel(v_matrix.view())?,
        o: canonical_runtime.upload_step_bf16_row_parallel(o_matrix.view())?,
    };

    let root = runtime
        .rank_engine(0)
        .ok_or("native TP runtime has no root rank")?;
    let input = activations(tokens, hidden);
    let root_input = root.htod(&input)?;
    let gate_weight =
        GpuTensor::load_from_source(root, &source, &format!("{prefix}.attn_gate.weight"))?;
    let gate = root.matmul(&gate_weight, &root_input, tokens)?;
    let gate = root.dtoh(&gate)?;
    if gate.len() != tokens * geometry.n_head as usize {
        return Err(format!(
            "attention gate output {} != {}x{}",
            gate.len(),
            tokens,
            geometry.n_head
        )
        .into());
    }
    let q_norm =
        GpuTensor::load_from_source(root, &source, &format!("{prefix}.attn_q_norm.weight"))?;
    let k_norm =
        GpuTensor::load_from_source(root, &source, &format!("{prefix}.attn_k_norm.weight"))?;
    let q_norm = root.dtoh(q_norm.float_data())?;
    let k_norm = root.dtoh(k_norm.float_data())?;
    let rope_factors = if geometry.rope_factors && source.find("rope_freqs.weight").is_some() {
        let factors = GpuTensor::load_from_source(root, &source, "rope_freqs.weight")?;
        Some(root.dtoh(factors.float_data())?)
    } else {
        None
    };
    let positions = (0..tokens)
        .map(i32::try_from)
        .collect::<Result<Vec<_>, _>>()?;
    let inputs = AttentionInputs {
        activations: &input,
        gate: &gate,
        q_norm: &q_norm,
        k_norm: &k_norm,
        rope_factors: rope_factors.as_deref(),
        positions: &positions,
        tokens,
        geometry,
        rms_eps: config.rms_eps,
    };

    let canonical_attention =
        run_cacheless_attention(&canonical_runtime, &canonical_resident, &inputs)?;
    let canonical_output = canonical_runtime.step_bf16_row_parallel_resident_native(
        &canonical_resident.o,
        &canonical_attention,
        tokens,
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
            "TP_ATTENTION_AMBIENT_CONTEXT device={device} tp_root={} purpose=pp-scope-regression",
            devices[0]
        );
        Some(Engine::new(device)?)
    } else {
        None
    };
    let run_native = || -> Result<(Vec<f32>, Vec<f32>), Box<dyn std::error::Error>> {
        let _ambient = ambient_engine
            .as_ref()
            .map(|engine| engine.gpu.enter_main())
            .transpose()?;
        let attention = run_cacheless_attention(&runtime, &resident, &inputs)?;
        let output =
            runtime.step_bf16_row_parallel_resident_native(&resident.o, &attention, tokens)?;
        Ok((attention, output))
    };
    let (native_attention, native_output) = run_native()?;
    let (native_attention_repeat, native_output_repeat) = run_native()?;
    compare_exact(
        "canonical_tp1_vs_native_tp_attention",
        &canonical_attention,
        &native_attention,
    )?;
    compare_exact(
        "native_tp_attention_repeat",
        &native_attention,
        &native_attention_repeat,
    )?;
    compare_exact(
        "canonical_tp1_vs_native_tp_output",
        &canonical_output,
        &native_output,
    )?;
    compare_exact(
        "native_tp_output_repeat",
        &native_output,
        &native_output_repeat,
    )?;
    let canonical_cached =
        run_cached_attention_sequence(&canonical_runtime, &canonical_resident, &inputs, grow, swa)?;
    let run_cached_native = || -> Result<CachedAttentionRun, Box<dyn std::error::Error>> {
        let _ambient = ambient_engine
            .as_ref()
            .map(|engine| engine.gpu.enter_main())
            .transpose()?;
        run_cached_attention_sequence(&runtime, &resident, &inputs, grow, swa)
    };
    let native_cached = run_cached_native()?;
    let native_cached_repeat = run_cached_native()?;
    compare_exact(
        "canonical_tp1_vs_native_tp_cached_attention",
        &canonical_cached.attention,
        &native_cached.attention,
    )?;
    compare_exact(
        "native_tp_cached_attention_repeat",
        &native_cached.attention,
        &native_cached_repeat.attention,
    )?;
    compare_exact(
        "canonical_tp1_vs_native_tp_cached_output",
        &canonical_cached.output,
        &native_cached.output,
    )?;
    compare_exact(
        "native_tp_cached_output_repeat",
        &native_cached.output,
        &native_cached_repeat.output,
    )?;
    compare_exact(
        "canonical_tp1_vs_native_tp_speculative_attention",
        &canonical_cached.speculative_attention,
        &native_cached.speculative_attention,
    )?;
    compare_exact(
        "native_tp_speculative_vs_rollback_attention",
        &native_cached.speculative_attention,
        &native_cached.rollback_attention,
    )?;
    compare_exact(
        "native_tp_rollback_vs_recommit_attention",
        &native_cached.rollback_attention,
        &native_cached.recommit_attention,
    )?;
    compare_exact(
        "canonical_tp1_vs_native_tp_speculative_output",
        &canonical_cached.speculative_output,
        &native_cached.speculative_output,
    )?;
    compare_exact(
        "native_tp_speculative_vs_rollback_output",
        &native_cached.speculative_output,
        &native_cached.rollback_output,
    )?;
    compare_exact(
        "native_tp_rollback_vs_recommit_output",
        &native_cached.rollback_output,
        &native_cached.recommit_output,
    )?;
    compare_bytes_exact(
        "canonical_tp1_vs_native_tp_partial_k_cache",
        &canonical_cached.partial_cache.k,
        &native_cached.partial_cache.k,
    )?;
    compare_bytes_exact(
        "canonical_tp1_vs_native_tp_rollback_k_cache",
        &canonical_cached.rollback_cache.k,
        &native_cached.rollback_cache.k,
    )?;
    compare_bytes_exact(
        "native_tp_partial_vs_rollback_k_cache",
        &native_cached.partial_cache.k,
        &native_cached.rollback_cache.k,
    )?;
    compare_bytes_exact(
        "canonical_tp1_vs_native_tp_final_k_cache",
        &canonical_cached.final_cache.k,
        &native_cached.final_cache.k,
    )?;
    compare_bytes_exact(
        "native_tp_final_k_cache_repeat",
        &native_cached.final_cache.k,
        &native_cached_repeat.final_cache.k,
    )?;
    compare_bytes_exact(
        "canonical_tp1_vs_native_tp_partial_v_cache",
        &canonical_cached.partial_cache.v,
        &native_cached.partial_cache.v,
    )?;
    compare_bytes_exact(
        "canonical_tp1_vs_native_tp_rollback_v_cache",
        &canonical_cached.rollback_cache.v,
        &native_cached.rollback_cache.v,
    )?;
    compare_bytes_exact(
        "native_tp_partial_vs_rollback_v_cache",
        &native_cached.partial_cache.v,
        &native_cached.rollback_cache.v,
    )?;
    compare_bytes_exact(
        "canonical_tp1_vs_native_tp_final_v_cache",
        &canonical_cached.final_cache.v,
        &native_cached.final_cache.v,
    )?;
    compare_bytes_exact(
        "native_tp_final_v_cache_repeat",
        &native_cached.final_cache.v,
        &native_cached_repeat.final_cache.v,
    )?;
    if canonical_cached.k_tok_bytes != native_cached.k_tok_bytes
        || canonical_cached.v_tok_bytes != native_cached.v_tok_bytes
        || canonical_cached.logical_capacity != native_cached.logical_capacity
        || canonical_cached.ring_window != native_cached.ring_window
        || canonical_cached.physical_rows != native_cached.physical_rows
    {
        return Err(format!(
            "canonical cache geometry k/v={}/{} logical={} ring={:?}/{} != native \
             {}/{} logical={} ring={:?}/{}",
            canonical_cached.k_tok_bytes,
            canonical_cached.v_tok_bytes,
            canonical_cached.logical_capacity,
            canonical_cached.ring_window,
            canonical_cached.physical_rows,
            native_cached.k_tok_bytes,
            native_cached.v_tok_bytes,
            native_cached.logical_capacity,
            native_cached.ring_window,
            native_cached.physical_rows,
        )
        .into());
    }
    match (
        &canonical_cached.grow,
        &native_cached.grow,
        &native_cached_repeat.grow,
    ) {
        (Some(canonical), Some(native), Some(repeat)) if grow => {
            compare_bytes_exact(
                "canonical_tp1_vs_native_tp_grown_k_cache",
                &canonical.cache.k,
                &native.cache.k,
            )?;
            compare_bytes_exact(
                "native_tp_grown_k_cache_repeat",
                &native.cache.k,
                &repeat.cache.k,
            )?;
            compare_bytes_exact(
                "canonical_tp1_vs_native_tp_grown_v_cache",
                &canonical.cache.v,
                &native.cache.v,
            )?;
            compare_bytes_exact(
                "native_tp_grown_v_cache_repeat",
                &native.cache.v,
                &repeat.cache.v,
            )?;
            let expected = (
                canonical.rows,
                canonical.source_capacity,
                canonical.target_capacity,
                canonical.next_generation,
            );
            let native_state = (
                native.rows,
                native.source_capacity,
                native.target_capacity,
                native.next_generation,
            );
            let repeat_state = (
                repeat.rows,
                repeat.source_capacity,
                repeat.target_capacity,
                repeat.next_generation,
            );
            if native_state != expected || repeat_state != expected {
                return Err(format!(
                    "TP KV grow state canonical={expected:?} native={native_state:?} \
                     repeat={repeat_state:?}"
                )
                .into());
            }
            println!(
                "STEP_TP_KV_GROW_GATE_PASS layer={layer} tp={} rows={} \
                 source_capacity={} target_capacity={} next_generation={} \
                 cache_bytes_exact=true device_lengths_exact=true \
                 generation_preserved=true rank_streams_synchronized=true \
                 transport={} native_p2p={} devices={devices:?} \
                 performance_claim=false",
                plan.request.tensor,
                native.rows,
                native.source_capacity,
                native.target_capacity,
                native.next_generation,
                runtime.transport_label(),
                runtime.native_p2p(),
            );
        }
        (None, None, None) if !grow => {}
        _ => return Err("TP KV grow gate state is inconsistent across runtimes".into()),
    }

    println!(
        "STEP_TP_ATTENTION_GATE_PASS layer={layer} tokens={tokens} tp={} \
         query_heads={} kv_heads={} head_dim={} checkpoint_fp8_projections={qualified} \
         qkv_tensor_parallel=true qk_norm_rank_local=true rope_rank_local=true \
         attention_tensor_parallel=true attention_cacheless=true kv_cache_distributed=false \
         gate_tensor_parallel=false gate_shards=host-canonical o_tensor_parallel=true \
         reduction=global-tp8-block-order transport={} native_p2p={} \
         canonical_tp1_tp_exact=true native_repeat_exact=true devices={devices:?} \
         names={names:?} performance_claim=false",
        plan.request.tensor,
        geometry.n_head,
        geometry.n_head_kv,
        geometry.head_dim_k,
        runtime.transport_label(),
        runtime.native_p2p(),
    );
    println!(
        "STEP_TP_KV_CACHE_GATE_PASS layer={layer} steps={tokens} tp={} \
         query_heads={} kv_heads={} head_dim={} cache_formats=q8_0/q5_1 \
         k_tok_bytes={} v_tok_bytes={} qkv_tensor_parallel=true \
         qk_norm_rank_local=true rope_rank_local=true attention_tensor_parallel=true \
         kv_cache_distributed=true cache_state_advanced=true cache_bytes_exact=true \
         cache_append_exact=true cache_partial_commit_exact=true \
         cache_rollback_exact=true cache_recommit_exact=true \
         cache_length_mirrors_exact=true cache_commit_semantics_tested=true \
         cache_rollback_tested=true transaction_fail_closed=true gate_tensor_parallel=false \
         gate_shards=host-canonical o_tensor_parallel=true \
         reduction=global-tp8-block-order transport={} native_p2p={} \
         canonical_tp1_tp_exact=true native_repeat_exact=true devices={devices:?} \
         names={names:?} performance_claim=false",
        plan.request.tensor,
        geometry.n_head,
        geometry.n_head_kv,
        geometry.head_dim_k,
        native_cached.k_tok_bytes,
        native_cached.v_tok_bytes,
        runtime.transport_label(),
        runtime.native_p2p(),
    );
    if f32_mirror {
        println!(
            "STEP_TP_F32_MIRROR_GATE_PASS layer={layer} tp={} projections=qkvo \
             bf16_residency=f32-mirror canonical_tp1_residency=bf16 \
             cache_formats=q8_0/q5_1 attention_tensor_parallel=true \
             canonical_tp1_tp_exact=true native_repeat_exact=true \
             transport={} native_p2p={} devices={devices:?} performance_claim=false",
            plan.request.tensor,
            runtime.transport_label(),
            runtime.native_p2p(),
        );
    }
    if swa {
        println!(
            "STEP_TP_SWA_RING_GATE_PASS layer={layer} tp={} window={} \
             logical_capacity={} physical_rows={} cache_formats=q8_0/q5_1 \
             hydration=aligned-live-prefix append=shared-absolute-ring \
             attention_view=contiguous-last-window rollback_window_preserved=true \
             grow_live_prefix_only={} canonical_tp1_tp_exact=true \
             native_repeat_exact=true transport={} native_p2p={} \
             devices={devices:?} performance_claim=false",
            plan.request.tensor,
            native_cached.ring_window.unwrap(),
            native_cached.logical_capacity,
            native_cached.physical_rows,
            grow,
            runtime.transport_label(),
            runtime.native_p2p(),
        );
    }
    Ok(())
}
