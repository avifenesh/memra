//! Exact-geometry kernel probe for HY3's automatic-EP W4A8 decode chain.
//!
//! This is a profiler instrument, not a serving path. It creates two local NVFP4 experts and
//! eight global route slots, of which exactly two are owned by this rank, matching the expected
//! EP4 c1 occupancy. The three modes isolate the shipped paired gate/up, q8 activation, and down
//! kernels; `chain` queues all three. Use NCU kernel-name filters over a small iteration count.

use cudarc::driver::{CudaSlice, DevicePtr};
use memra_engine::Engine;

const HIDDEN: usize = 4096;
const EXPERT_WIDTH: usize = 1536;
const PAIRS: usize = 8;
const LOCAL_EXPERTS: usize = 2;

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
