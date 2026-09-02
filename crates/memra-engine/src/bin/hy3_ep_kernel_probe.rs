//! Exact-geometry kernel probe for HY3's automatic-EP W4A8 decode chain.
//!
//! This is a profiler instrument, not a serving path. It creates two local NVFP4 experts and
//! eight global route slots, of which exactly two are owned by this rank, matching the expected
//! EP4 c1 occupancy. The three modes isolate the shipped paired gate/up, q8 activation, and down
//! kernels; `chain` queues all three. Use NCU kernel-name filters over a small iteration count.

use cudarc::driver::{CudaSlice, DevicePtr};
use memra_engine::Engine;
use memra_engine::tp::{TpE4m3HostBounce, TpKvVerifiedLayer};

const HIDDEN: usize = 4096;
const EXPERT_WIDTH: usize = 1536;
const PAIRS: usize = 8;
const LOCAL_EXPERTS: usize = 2;
const TP_KV_LAYERS: usize = 80;
const TP_K_BYTES: usize = 544;
const TP_V_BYTES: usize = 384;
const TP_K_SRC_STRIDE: usize = 2 * TP_K_BYTES;
const TP_V_SRC_STRIDE: usize = 2 * TP_V_BYTES;
const TP_KV_ROWS: usize = 2;
const TP_KV_LEN: i32 = 130;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn byte(&mut self) -> u8 {
        (self.next() >> 24) as u8
    }

    fn f32(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u32 << 24) as f32) - 0.5
    }
}

fn random_nvfp4_bank(seed: u64, experts: usize, out_f: usize, in_f: usize) -> Vec<u8> {
    assert!(in_f.is_multiple_of(64));
    let mut rng = Rng(seed);
    let mut bytes = Vec::with_capacity(experts * out_f * (in_f / 64) * 36);
    for _ in 0..experts * out_f {
        for _ in 0..in_f / 64 {
            for _ in 0..4 {
                bytes.push(0x38 + (rng.byte() & 0x0f));
            }
            for _ in 0..32 {
                bytes.push(rng.byte());
            }
        }
    }
    bytes
}

fn constant_bf16_bytes(elements: usize, value: f32) -> Vec<u8> {
    let bits = ((value.to_bits() >> 16) as u16).to_le_bytes();
    let mut bytes = Vec::with_capacity(2 * elements);
    for _ in 0..elements {
        bytes.extend_from_slice(&bits);
    }
    bytes
}

struct Probe {
    engine: Engine,
    gate_bank: CudaSlice<u8>,
    up_bank: CudaSlice<u8>,
    down_bank: CudaSlice<u8>,
    selected: CudaSlice<i32>,
    input_q: CudaSlice<i8>,
    input_d: CudaSlice<f32>,
    gate_out: CudaSlice<f32>,
    up_out: CudaSlice<f32>,
    gate_macros: CudaSlice<f32>,
    up_macros: CudaSlice<f32>,
    down_macros: CudaSlice<f32>,
    activation_q: CudaSlice<i8>,
    activation_d: CudaSlice<f32>,
    slot_rows: CudaSlice<f32>,
    slot_rows_raw: u64,
    shared_gate: CudaSlice<u8>,
    shared_up: CudaSlice<u8>,
    shared_down: CudaSlice<u8>,
    shared_input: CudaSlice<f32>,
    shared_act: CudaSlice<f32>,
    shared_out: CudaSlice<f32>,
}

impl Probe {
    fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let engine = Engine::new(0)?;
        let gate_row_bytes = HIDDEN / 64 * 36;
        let down_row_bytes = EXPERT_WIDTH / 64 * 36;
        let gate_bank = engine.htod_bytes(&random_nvfp4_bank(
            0x243f_6a88_85a3_08d3,
            LOCAL_EXPERTS,
            EXPERT_WIDTH,
            HIDDEN,
        ))?;
        let up_bank = engine.htod_bytes(&random_nvfp4_bank(
            0x1319_8a2e_0370_7344,
            LOCAL_EXPERTS,
            EXPERT_WIDTH,
            HIDDEN,
        ))?;
        let down_bank = engine.htod_bytes(&random_nvfp4_bank(
            0xa409_3822_299f_31d0,
            LOCAL_EXPERTS,
            HIDDEN,
            EXPERT_WIDTH,
        ))?;
        assert_eq!(
            gate_bank.len(),
            LOCAL_EXPERTS * EXPERT_WIDTH * gate_row_bytes
        );
        assert_eq!(up_bank.len(), LOCAL_EXPERTS * EXPERT_WIDTH * gate_row_bytes);
        assert_eq!(down_bank.len(), LOCAL_EXPERTS * HIDDEN * down_row_bytes);

        let selected = engine.htod_i32(&[0, 1, 48, 49, 96, 97, 144, 145])?;
        let mut rng = Rng(0x082e_fa98_ec4e_6c89);
        let input = (0..HIDDEN).map(|_| rng.f32()).collect::<Vec<_>>();
        let input = engine.htod(&input)?;
        let mut input_q = engine.alloc_i8_uninit(HIDDEN)?;
        let mut input_d = engine.uninit(HIDDEN / 32)?;
        engine.quantize_q8_1_into(&input, 1, HIDDEN, &mut input_q, &mut input_d)?;

        let gate_values = (0..PAIRS * EXPERT_WIDTH)
            .map(|_| rng.f32())
            .collect::<Vec<_>>();
        let up_values = (0..PAIRS * EXPERT_WIDTH)
            .map(|_| rng.f32())
            .collect::<Vec<_>>();
        let gate_out = engine.htod(&gate_values)?;
        let up_out = engine.htod(&up_values)?;
        let gate_macros = engine.htod(&[1.0, 0.75])?;
        let up_macros = engine.htod(&[0.875, 1.0])?;
        let down_macros = engine.htod(&[1.0, 0.625])?;
        let activation_q = engine.htod_i8(
            &(0..PAIRS * EXPERT_WIDTH)
                .map(|index| (index as u8).wrapping_mul(29) as i8)
                .collect::<Vec<_>>(),
        )?;
        let activation_d = engine.htod(&vec![0.01; PAIRS * EXPERT_WIDTH / 32])?;
        let slot_rows = engine.zeros(PAIRS * HIDDEN)?;
        let slot_rows_raw = {
            let stream = engine.stream();
            let (pointer, _guard) = slot_rows.device_ptr(&stream);
            pointer
        };
        let shared_gate = engine.htod_bytes(&constant_bf16_bytes(HIDDEN * EXPERT_WIDTH, 0.01))?;
        let shared_up = engine.htod_bytes(&constant_bf16_bytes(HIDDEN * EXPERT_WIDTH, 0.015))?;
        let shared_down = engine.htod_bytes(&constant_bf16_bytes(EXPERT_WIDTH * HIDDEN, -0.005))?;
        let shared_input = engine.htod(&vec![0.01; HIDDEN])?;
        let shared_act = engine.zeros(EXPERT_WIDTH)?;
        let shared_out = engine.zeros(HIDDEN)?;
        engine.stream().synchronize()?;
        Ok(Self {
            engine,
            gate_bank,
            up_bank,
            down_bank,
            selected,
            input_q,
            input_d,
            gate_out,
            up_out,
            gate_macros,
            up_macros,
            down_macros,
            activation_q,
            activation_d,
            slot_rows,
            slot_rows_raw,
            shared_gate,
            shared_up,
            shared_down,
            shared_input,
            shared_act,
            shared_out,
        })
    }

    fn paired(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.engine.qmatvec_nvfp4_q8_ep_paired_slots_into(
            &self.gate_bank,
            &self.up_bank,
            &self.selected,
            &self.input_q,
            &self.input_d,
            &mut self.gate_out,
            &mut self.up_out,
            PAIRS,
            PAIRS,
            HIDDEN,
            EXPERT_WIDTH,
            0,
            LOCAL_EXPERTS,
            HIDDEN / 64 * 36,
            EXPERT_WIDTH * (HIDDEN / 64 * 36),
        )
    }

    fn activation(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.engine.silu_mul_scaled_host_expf_q8_ep_slots_into(
            &self.gate_out,
            &self.up_out,
            &self.gate_macros,
            &self.up_macros,
            &self.selected,
            0,
            LOCAL_EXPERTS,
            None,
            &mut self.activation_q,
            &mut self.activation_d,
            EXPERT_WIDTH,
            PAIRS,
        )
    }

    fn down(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.engine.qmatvec_nvfp4_q8_ep_down_slots_raw(
            &self.down_bank,
            &self.selected,
            &self.activation_q,
            &self.activation_d,
            &self.down_macros,
            self.slot_rows_raw,
            PAIRS,
            EXPERT_WIDTH,
            HIDDEN,
            0,
            LOCAL_EXPERTS,
            EXPERT_WIDTH / 64 * 36,
            HIDDEN * (EXPERT_WIDTH / 64 * 36),
        )
    }

    fn shared_gate_up(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.engine.matvec_bf16_dual_silu_into(
            &self.shared_gate,
            &self.shared_up,
            &self.shared_input,
            &mut self.shared_act,
            HIDDEN,
            EXPERT_WIDTH,
            None,
        )
    }

    fn shared_down(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.engine.matvec_bf16_into(
            &self.shared_down,
            &self.shared_act,
            &mut self.shared_out,
            EXPERT_WIDTH,
            HIDDEN,
        )
    }

    fn run(&mut self, mode: &str) -> Result<(), Box<dyn std::error::Error>> {
        match mode {
            "paired" => self.paired(),
            "activation" => self.activation(),
            "down" => self.down(),
            "shared_gate_up" => self.shared_gate_up(),
            "shared_down" => self.shared_down(),
            "shared_chain" => {
                self.shared_gate_up()?;
                self.shared_down()
            }
            "chain" => {
                self.paired()?;
                self.activation()?;
                self.down()
            }
            _ => Err(format!(
                "unknown mode {mode:?}; expected paired, activation, down, chain, \
                     shared_gate_up, shared_down, shared_chain"
            )
            .into()),
        }
    }
}

struct KvRestoreProbe {
    engine: Engine,
    k_src: Vec<CudaSlice<u8>>,
    v_src: Vec<CudaSlice<u8>>,
    k_dst: Vec<CudaSlice<u8>>,
    v_dst: Vec<CudaSlice<u8>>,
    lens: Vec<CudaSlice<i32>>,
    table: CudaSlice<u64>,
}

impl KvRestoreProbe {
    fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let engine = Engine::new(0)?;
        let mut k_src = Vec::with_capacity(TP_KV_LAYERS);
        let mut v_src = Vec::with_capacity(TP_KV_LAYERS);
        let mut k_dst = Vec::with_capacity(TP_KV_LAYERS);
        let mut v_dst = Vec::with_capacity(TP_KV_LAYERS);
        let mut lens = Vec::with_capacity(TP_KV_LAYERS);
        for layer in 0..TP_KV_LAYERS {
            let k = (0..TP_KV_ROWS * TP_K_SRC_STRIDE)
                .map(|index| (layer as u8).wrapping_mul(17).wrapping_add(index as u8))
                .collect::<Vec<_>>();
            let v = (0..TP_KV_ROWS * TP_V_SRC_STRIDE)
                .map(|index| (layer as u8).wrapping_mul(29).wrapping_add(index as u8))
                .collect::<Vec<_>>();
            k_src.push(engine.htod_bytes(&k)?);
            v_src.push(engine.htod_bytes(&v)?);
            k_dst.push(engine.alloc_u8(TP_KV_ROWS * TP_K_BYTES)?);
            v_dst.push(engine.alloc_u8(TP_KV_ROWS * TP_V_BYTES)?);
            lens.push(engine.htod_i32(&[0])?);
        }
        let stream = engine.stream();
        let mut table = vec![0u64; 5 * TP_KV_LAYERS];
        for layer in 0..TP_KV_LAYERS {
            table[layer] = k_src[layer].device_ptr(&stream).0;
            table[TP_KV_LAYERS + layer] = v_src[layer].device_ptr(&stream).0;
            table[2 * TP_KV_LAYERS + layer] = k_dst[layer].device_ptr(&stream).0;
            table[3 * TP_KV_LAYERS + layer] = v_dst[layer].device_ptr(&stream).0;
            table[4 * TP_KV_LAYERS + layer] = lens[layer].device_ptr(&stream).0;
        }
        let table = engine.htod_u64(&table)?;
        engine.stream().synchronize()?;
        Ok(Self {
            engine,
            k_src,
            v_src,
            k_dst,
            v_dst,
            lens,
            table,
        })
    }

    fn baseline(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        for layer in 0..TP_KV_LAYERS {
            for row in 0..TP_KV_ROWS {
                self.engine.copy_u8_range_into(
                    &mut self.k_dst[layer],
                    row * TP_K_BYTES,
                    &self.k_src[layer],
                    row * TP_K_SRC_STRIDE,
                    TP_K_BYTES,
                )?;
                self.engine.copy_u8_range_into(
                    &mut self.v_dst[layer],
                    row * TP_V_BYTES,
                    &self.v_src[layer],
                    row * TP_V_SRC_STRIDE,
                    TP_V_BYTES,
                )?;
            }
            self.engine.set_i32_one(&mut self.lens[layer], TP_KV_LEN)?;
        }
        Ok(())
    }

    fn batched(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.engine.copy_batch_uniform_kv_u8_set_len(
            &self.table,
            TP_KV_LAYERS,
            TP_KV_ROWS,
            TP_K_BYTES,
            TP_V_BYTES,
            TP_K_SRC_STRIDE,
            TP_V_SRC_STRIDE,
            TP_KV_LEN as usize,
        )
    }

    fn validate(&self) -> Result<(), Box<dyn std::error::Error>> {
        for layer in 0..TP_KV_LAYERS {
            let k_source = self.engine.dtoh_u8(&self.k_src[layer])?;
            let k_expected = (0..TP_KV_ROWS)
                .flat_map(|row| {
                    k_source[row * TP_K_SRC_STRIDE..row * TP_K_SRC_STRIDE + TP_K_BYTES]
                        .iter()
                        .copied()
                })
                .collect::<Vec<_>>();
            if k_expected != self.engine.dtoh_u8(&self.k_dst[layer])? {
                return Err(format!("TP KV K mismatch at layer {layer}").into());
            }
            let v_source = self.engine.dtoh_u8(&self.v_src[layer])?;
            let v_expected = (0..TP_KV_ROWS)
                .flat_map(|row| {
                    v_source[row * TP_V_SRC_STRIDE..row * TP_V_SRC_STRIDE + TP_V_BYTES]
                        .iter()
                        .copied()
                })
                .collect::<Vec<_>>();
            if v_expected != self.engine.dtoh_u8(&self.v_dst[layer])? {
                return Err(format!("TP KV V mismatch at layer {layer}").into());
            }
            if self.engine.dtoh_i32_one(&self.lens[layer])? != TP_KV_LEN {
                return Err(format!("TP KV length mismatch at layer {layer}").into());
            }
        }
        Ok(())
    }
}

struct KvRestoreP2pProbe {
    engine: Engine,
    runtime: TpE4m3HostBounce,
    caches: Vec<memra_engine::cache::ResidentTpKvCache>,
    k_src: Vec<CudaSlice<u8>>,
    v_src: Vec<CudaSlice<u8>>,
}

impl KvRestoreP2pProbe {
    fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let engine = Engine::new(0)?;
        let runtime = TpE4m3HostBounce::new_native_p2p(&[0, 1])?;
        let mut caches = Vec::with_capacity(TP_KV_LAYERS);
        let mut k_src = Vec::with_capacity(TP_KV_LAYERS);
        let mut v_src = Vec::with_capacity(TP_KV_LAYERS);
        let zeros_k = vec![0u8; 128 * TP_K_SRC_STRIDE];
        let zeros_v = vec![0u8; 128 * TP_V_SRC_STRIDE];
        for layer in 0..TP_KV_LAYERS {
            let mut cache = runtime.allocate_tp_kv_cache(1024, 1024, 256)?;
            runtime.hydrate_tp_kv_cache(&mut cache, 128, &zeros_k, &zeros_v)?;
            caches.push(cache);
            let k = (0..TP_KV_ROWS * TP_K_SRC_STRIDE)
                .map(|index| (layer as u8).wrapping_mul(17).wrapping_add(index as u8))
                .collect::<Vec<_>>();
            let v = (0..TP_KV_ROWS * TP_V_SRC_STRIDE)
                .map(|index| (layer as u8).wrapping_mul(29).wrapping_add(index as u8))
                .collect::<Vec<_>>();
            k_src.push(engine.htod_bytes(&k)?);
            v_src.push(engine.htod_bytes(&v)?);
        }
        engine.stream().synchronize()?;
        for rank in 0..2 {
            runtime
                .rank_engine(rank)
                .ok_or("TP KV P2P probe lost a rank engine")?
                .stream()
                .synchronize()?;
        }
        Ok(Self {
            engine,
            runtime,
            caches,
            k_src,
            v_src,
        })
    }

    fn run(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let stream = self.engine.stream();
        let mut layers = self
            .caches
            .iter_mut()
            .zip(&self.k_src)
            .zip(&self.v_src)
            .map(|((cache, k), v)| TpKvVerifiedLayer {
                cache,
                start: 128,
                logical_len: 130,
                source_k_raw: k.device_ptr(&stream).0,
                source_v_raw: v.device_ptr(&stream).0,
                source_k_tok_bytes: TP_K_SRC_STRIDE,
                source_v_tok_bytes: TP_V_SRC_STRIDE,
            })
            .collect::<Vec<_>>();
        if !self.runtime.restore_tp_kv_layers_from_device(&mut layers)? {
            return Err("TP KV P2P batch unexpectedly declined exact geometry".into());
        }
        Ok(())
    }

    fn run_baseline(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let stream = self.engine.stream();
        for layer in 0..TP_KV_LAYERS {
            self.runtime.restore_tp_kv_rows_from_device(
                &mut self.caches[layer],
                128,
                130,
                self.k_src[layer].device_ptr(&stream).0,
                self.v_src[layer].device_ptr(&stream).0,
                TP_K_SRC_STRIDE,
                TP_V_SRC_STRIDE,
            )?;
        }
        Ok(())
    }

    fn synchronize(&self) -> Result<(), Box<dyn std::error::Error>> {
        for rank in 0..2 {
            self.runtime
                .rank_engine(rank)
                .ok_or("TP KV P2P probe lost a rank engine")?
                .stream()
                .synchronize()?;
        }
        Ok(())
    }

    fn validate(&self) -> Result<(), Box<dyn std::error::Error>> {
        for layer in 0..TP_KV_LAYERS {
            let k_source = self.engine.dtoh_u8(&self.k_src[layer])?;
            let v_source = self.engine.dtoh_u8(&self.v_src[layer])?;
            let lengths = self.runtime.tp_kv_device_lengths(&self.caches[layer])?;
            if lengths != [TP_KV_LEN, TP_KV_LEN]
                || self.caches[layer].committed_len() != TP_KV_LEN as usize
            {
                return Err(format!("TP KV P2P length mismatch at layer {layer}").into());
            }
            for rank in 0..2 {
                let rank_cache = self.caches[layer]
                    .rank(rank)
                    .ok_or_else(|| format!("TP KV P2P cache lost rank {rank}"))?;
                let engine = self
                    .runtime
                    .rank_engine(rank)
                    .ok_or_else(|| format!("TP KV P2P runtime lost rank {rank}"))?;
                let k_actual = engine.dtoh_u8(rank_cache.k())?;
                let v_actual = engine.dtoh_u8(rank_cache.v())?;
                for row in 0..TP_KV_ROWS {
                    let k_expected = &k_source[row * TP_K_SRC_STRIDE + rank * TP_K_BYTES
                        ..row * TP_K_SRC_STRIDE + (rank + 1) * TP_K_BYTES];
                    let k_begin = (128 + row) * TP_K_BYTES;
                    if &k_actual[k_begin..k_begin + TP_K_BYTES] != k_expected {
                        return Err(format!(
                            "TP KV P2P K mismatch at layer {layer} rank {rank} row {row}"
                        )
                        .into());
                    }
                    let v_expected = &v_source[row * TP_V_SRC_STRIDE + rank * TP_V_BYTES
                        ..row * TP_V_SRC_STRIDE + (rank + 1) * TP_V_BYTES];
                    let v_begin = (128 + row) * TP_V_BYTES;
                    if &v_actual[v_begin..v_begin + TP_V_BYTES] != v_expected {
                        return Err(format!(
                            "TP KV P2P V mismatch at layer {layer} rank {rank} row {row}"
                        )
                        .into());
                    }
                }
            }
        }
        Ok(())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mode = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "chain".to_string());
    let iterations = std::env::args()
        .nth(2)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1000);
    if iterations == 0 {
        return Err("iterations must be nonzero".into());
    }
    if mode == "kv_restore_baseline" || mode == "kv_restore_batch" {
        let mut probe = KvRestoreProbe::new()?;
        for _ in 0..5 {
            if mode == "kv_restore_baseline" {
                probe.baseline()?;
            } else {
                probe.batched()?;
            }
        }
        probe.engine.stream().synchronize()?;
        let started = std::time::Instant::now();
        for _ in 0..iterations {
            if mode == "kv_restore_baseline" {
                probe.baseline()?;
            } else {
                probe.batched()?;
            }
        }
        probe.engine.stream().synchronize()?;
        let microseconds = started.elapsed().as_secs_f64() * 1.0e6 / iterations as f64;
        probe.validate()?;
        println!(
            "HY3_TP_KV_RESTORE_PROBE mode={mode} iterations={iterations} \
             us_per_round={microseconds:.3} layers={TP_KV_LAYERS} k_bytes={TP_K_BYTES} \
             v_bytes={TP_V_BYTES} rows={TP_KV_ROWS} logical_len={TP_KV_LEN} exact=true"
        );
        return Ok(());
    }
    if mode == "kv_restore_p2p" || mode == "kv_restore_p2p_baseline" {
        let mut probe = KvRestoreP2pProbe::new()?;
        for _ in 0..5 {
            if mode == "kv_restore_p2p" {
                probe.run()?;
            } else {
                probe.run_baseline()?;
            }
        }
        probe.synchronize()?;
        let started = std::time::Instant::now();
        for _ in 0..iterations {
            if mode == "kv_restore_p2p" {
                probe.run()?;
            } else {
                probe.run_baseline()?;
            }
        }
        probe.synchronize()?;
        let microseconds = started.elapsed().as_secs_f64() * 1.0e6 / iterations as f64;
        probe.validate()?;
        println!(
            "HY3_TP_KV_RESTORE_P2P_PROBE mode={mode} iterations={iterations} \
             us_per_round={microseconds:.3} layers={TP_KV_LAYERS} ranks=2 rows={TP_KV_ROWS} \
             exact=true"
        );
        return Ok(());
    }
    let mut probe = Probe::new()?;
    for _ in 0..20 {
        probe.run(&mode)?;
    }
    probe.engine.stream().synchronize()?;
    let started = std::time::Instant::now();
    for _ in 0..iterations {
        probe.run(&mode)?;
    }
    probe.engine.stream().synchronize()?;
    let microseconds = started.elapsed().as_secs_f64() * 1.0e6 / iterations as f64;
    let checksum = if mode == "shared_gate_up" {
        probe.engine.dtoh(&probe.shared_act)?
    } else if mode.starts_with("shared_") {
        probe.engine.dtoh(&probe.shared_out)?
    } else {
        probe.engine.dtoh(&probe.slot_rows)?
    };
    let finite = checksum.iter().all(|value| value.is_finite());
    println!(
        "HY3_EP_KERNEL_PROBE mode={mode} iterations={iterations} us_per_call={microseconds:.3} \
         hidden={HIDDEN} expert_width={EXPERT_WIDTH} pairs={PAIRS} owned=2 finite={finite}"
    );
    if !finite {
        return Err("probe output contains a non-finite value".into());
    }
    Ok(())
}
