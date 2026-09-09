//! Exact carried-prime capture. Mutable session addresses are read from a device
//! table refreshed at every chunk, including after a park/restore or state swap.
use crate::{
    Engine,
    cache::Cache,
    hybrid::{Ffn, HybridModel, Mixer},
};
use cudarc::driver::{CudaGraph, CudaSlice, DevicePtr, LaunchConfig, PushKernelArg, sys};
use std::cell::Cell;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const STRIDE: usize = 12;
thread_local! {
    static ACTIVE: Cell<Option<(u64, usize, usize)>> = const { Cell::new(None) };
}

pub(crate) fn current() -> Option<(u64, usize)> {
    ACTIVE.with(|a| a.get().map(|(p, n, il)| (p + (il * STRIDE * 8) as u64, n)))
}

pub(crate) fn set_layer(il: usize) {
    ACTIVE.with(|a| {
        if let Some((p, n, _)) = a.get() {
            a.set(Some((p, n, il)));
        }
    });
}

struct CaptureScope;
impl Drop for CaptureScope {
    fn drop(&mut self) {
        ACTIVE.with(|a| a.set(None));
    }
}

struct EventTracking<'a> {
    engine: &'a Engine,
    restore: bool,
}
impl Drop for EventTracking<'_> {
    fn drop(&mut self) {
        if self.restore {
            unsafe {
                self.engine.ctx().enable_event_tracking();
            }
        }
    }
}

struct F16Scope<'a> {
    engine: &'a Engine,
    previous: Option<crate::f16_ffi::F16Scratch>,
    owned: &'a mut Option<crate::f16_ffi::F16Scratch>,
}
impl Drop for F16Scope<'_> {
    fn drop(&mut self) {
        *self.owned = self.engine.f16_scratch_swap(self.previous.take());
    }
}

// Graph destruction precedes storage destruction. No pointer into a session is
// held here: all state-bearing kernel entries resolve the device table.
struct ChunkGraph {
    graph: Option<CudaGraph>,
    context: std::sync::Arc<cudarc::driver::CudaContext>,
    private_f16: Option<crate::f16_ffi::F16Scratch>,
    table: CudaSlice<u64>,
    input: CudaSlice<f32>,
    output: CudaSlice<f32>,
    positions: CudaSlice<i32>,
    rows: usize,
    capacity: usize,
    slab_pointer: u64,
    dequant_pointer: u64,
    tap_layers: Vec<usize>,
    replays: usize,
}

impl Drop for ChunkGraph {
    fn drop(&mut self) {
        // Includes cancellation/error teardown, not only the normal prime epilogue.
        // Do not release graph-owned addresses until queued work is known complete.
        if let Err(error) = self.context.synchronize() {
            eprintln!("FATAL prime graph teardown completion unproven: {error}");
            std::process::abort();
        }
        drop(self.graph.take());
        let result = unsafe { sys::cuDeviceGraphMemTrim(self.context.ordinal() as i32) };
        if result != sys::CUresult::CUDA_SUCCESS {
            eprintln!("prime graph memory trim failed: {result:?}");
        }
    }
}

// The GPU worker serializes this cache and binds its context before use.
unsafe impl Send for ChunkGraph {}

pub(crate) fn supported(m: &HybridModel, e: &Engine) -> bool {
    env!("MEMRA_BUILT_CUDA_ARCH") == "120a"
        && e.sm_count() == 170
        && m.cfg.n_embd == 5120
        && m.layers.len() == 64
        && m.layers.iter().enumerate().all(|(il, l)| {
            let valid = match &l.mixer {
                Mixer::Full(_) => {
                    let g = m.cfg.full_attention_geometry_at(il as u32);
                    g.n_head == 24 && g.n_head_kv == 4 && g.head_dim_k == 256
                }
                Mixer::Linear(la) => {
                    la.geometry.key_head_dim == 128 && la.geometry.value_head_dim == 128
                }
                _ => false,
            };
            valid && matches!(l.ffn, Ffn::Dense { .. })
        })
}

pub(crate) fn eligible(
    m: &HybridModel,
    e: &Engine,
    cache: &Cache,
    t: usize,
    base: usize,
    seq_end: usize,
) -> bool {
    let reusable_shape = cache
        .qwen_prime_graph
        .as_ref()
        .and_then(|p| p.downcast_ref::<ChunkGraph>())
        .is_some_and(|p| p.rows == t && p.capacity >= seq_end);
    supported(m, e)
        && crate::spec::graph_launch_headroom_ok(e)
        // A one-off restored suffix or boundary tail cannot amortize capture.
        // It executes the identical eager chunk rather than padding/reusing a graph.
        && (reusable_shape || seq_end.saturating_sub(base) >= t.saturating_mul(4))
        && base > 0
        && (128..=1024).contains(&t)
        && Engine::gdn_chunked_enabled()
        && Engine::gdn_chunk_size() == 32
        && e.gdn_mma_enabled(32)
        && !e.gdn_wgmma_on(32)
        && std::env::var("MEMRA_PRIME_DEQW").as_deref() != Ok("0")
        && std::env::var("MEMRA_PRIME_DEQW_DB").as_deref() != Ok("0")
        && std::env::var("MEMRA_NOFA").is_err()
        && std::env::var("MEMRA_PRIME_APPEND_LOOP").is_err()
        && std::env::var("MEMRA_PRIME_SEG").as_deref() != Ok("1")
        && std::env::var("MEMRA_PRIME_SLABS").as_deref() != Ok("0")
        && std::env::var("MEMRA_PRIME_ANATOMY").as_deref() != Ok("1")
}

fn table(e: &Engine, cache: &Cache, rows: usize) -> Vec<u64> {
    let mut out = vec![0; cache.kv.len() * STRIDE];
    let stream = e.stream();
    for il in 0..cache.kv.len() {
        let a = &mut out[il * STRIDE..(il + 1) * STRIDE];
        if let Some(kv) = &cache.kv[il] {
            a[0] = kv.k.device_ptr(&stream).0;
            a[1] = kv.v.device_ptr(&stream).0;
            a[2] = kv.len_d.device_ptr(&stream).0;
            a[6] = kv.len as u64;
            a[7] = (kv.len + rows) as u64;
        }
        if let Some(r) = &cache.recur[il] {
            a[3] = r.conv_state.device_ptr(&stream).0;
            a[4] = r.ssm_state.device_ptr(&stream).0;
            a[5] = r.ssm_state_alt.device_ptr(&stream).0;
        }
        if let Some(tap) = &cache.dflash_taps
            && let Some(slot) = tap.layer_ids.iter().position(|&l| l == il)
        {
            assert!(tap.base + rows <= tap.t);
            a[8] = tap.buf.device_ptr(&stream).0;
            a[9] = ((tap.base * tap.layer_ids.len() + slot) * tap.hidden) as u64;
            a[10] = (tap.layer_ids.len() * tap.hidden) as u64;
        }
    }
    out
}

fn advance_host(cache: &mut Cache, rows: usize, layers: usize) {
    for kv in cache.kv.iter_mut().take(layers).flatten() {
        kv.len += rows;
    }
    for r in cache.recur.iter_mut().take(layers).flatten() {
        std::mem::swap(&mut r.ssm_state, &mut r.ssm_state_alt);
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run(
    m: &HybridModel,
    e: &Engine,
    input: CudaSlice<f32>,
    positions: &CudaSlice<i32>,
    rows: usize,
    base: usize,
    cache: &mut Cache,
    seq_end: usize,
) -> Result<CudaSlice<f32>> {
    let hidden = m.cfg.n_embd as usize;
    let nff = m
        .layers
        .iter()
        .map(|l| match &l.ffn {
            Ffn::Dense { ffn_gate, .. } => ffn_gate.out_features(),
            _ => hidden,
        })
        .max()
        .unwrap();
    let slab = m.prime_slabs_get(e, rows, hidden, nff)?;
    let slab_pointer = slab.lock().unwrap().xa.device_ptr(&e.stream()).0;
    let dequant_pointer = e
        .prime_deqw_ws
        .lock()
        .unwrap()
        .as_ref()
        .map_or(0, |(k, _)| k.device_ptr(&e.stream()).0);
    let tap_layers = cache
        .dflash_taps
        .as_ref()
        .map_or_else(Vec::new, |t| t.layer_ids.clone());
    let old = cache
        .qwen_prime_graph
        .take()
        .map(|p| p.downcast::<ChunkGraph>().expect("prime graph type"));
    let mut pool = match old {
        Some(p)
            if p.rows == rows
                && p.capacity >= seq_end
                && p.slab_pointer == slab_pointer
                && p.dequant_pointer == dequant_pointer
                && p.tap_layers == tap_layers =>
        {
            p
        }
        old => {
            e.stream().synchronize()?;
            drop(old);
            e.trim_device_graph_mem()?;
            e.prepare_prime_graph(seq_end)?;
            Box::new(ChunkGraph {
                graph: None,
                context: e.ctx().clone(),
                private_f16: Some(crate::f16_ffi::F16Scratch::with_capacity(
                    e,
                    rows * nff * 2,
                )?),
                table: e.htod_u64(&vec![0; cache.kv.len() * STRIDE])?,
                input: e.uninit(rows * hidden)?,
                output: e.uninit(rows * hidden)?,
                positions: e.uninit_i32(rows)?,
                rows,
                capacity: seq_end,
                slab_pointer,
                tap_layers,
                dequant_pointer: e
                    .prime_deqw_ws
                    .lock()
                    .unwrap()
                    .as_ref()
                    .unwrap()
                    .0
                    .device_ptr(&e.stream())
                    .0,
                replays: 0,
            })
        }
    };
    e.htod_u64_into(&table(e, cache, rows), &mut pool.table)?;
    e.copy_into(&mut pool.input, 0, &input, rows * hidden)?;
    e.stream().memcpy_dtod(positions, &mut pool.positions)?;
    let previous = e.f16_scratch_swap(pool.private_f16.take());
    let scratch_scope = F16Scope {
        engine: e,
        previous,
        owned: &mut pool.private_f16,
    };
    if pool.graph.is_none() {
        let started = std::time::Instant::now();
        e.stream().synchronize()?;
        let tracking = e.ctx().is_event_tracking();
        if tracking {
            unsafe {
                e.ctx().disable_event_tracking();
            }
        }
        let tracking_guard = EventTracking {
            engine: e,
            restore: tracking,
        };
        let ptr = pool.table.device_ptr(&e.stream()).0;
        ACTIVE.with(|a| {
            assert!(a.get().is_none());
            a.set(Some((ptr, pool.capacity, 0)));
        });
        let scope = CaptureScope;
        e.stream()
            .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)?;
        let before_state: Vec<_> = cache
            .recur
            .iter()
            .map(|r| r.as_ref().map(|r| r.ssm_state.device_ptr(&e.stream()).0))
            .collect();
        let before: Vec<_> = cache.kv.iter().map(|k| k.as_ref().map(|k| k.len)).collect();
        let result = (|| {
            // input clone is a device allocation/copy wholly inside the graph.
            let x = m.prime_layers(
                e,
                pool.input.clone(),
                0,
                m.layers.len(),
                &pool.positions,
                rows,
                base,
                cache,
                seq_end,
            )?;
            e.copy_into(&mut pool.output, 0, &x, rows * hidden)
        })();
        let graph = e.stream().end_capture(
            sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
        );
        drop(scope);
        drop(tracking_guard);
        // Capture records device work but performs host bookkeeping. Restore each
        // layer that actually advanced, including a partially refused capture.
        for (kv, old) in cache.kv.iter_mut().zip(before) {
            if let (Some(kv), Some(len)) = (kv, old) {
                kv.len = len;
            }
        }
        for (r, old) in cache.recur.iter_mut().zip(before_state) {
            if let (Some(r), Some(ptr)) = (r, old)
                && r.ssm_state.device_ptr(&e.stream()).0 != ptr
            {
                std::mem::swap(&mut r.ssm_state, &mut r.ssm_state_alt);
            }
        }
        result?;
        let graph = graph?.ok_or("prime capture returned no graph")?;
        let safety = crate::glm5_decode_graph::graph_census(&graph)?;
        if safety.host_memcpys != 0 {
            return Err(format!(
                "prime graph captured {} host memcpy endpoints",
                safety.host_memcpys
            )
            .into());
        }
        let census = crate::graph_update::node_census(&graph)?;
        eprintln!(
            "[prime-graph] captured rows={rows} capacity={} capture_ms={:.3} nodes={census:?}",
            pool.capacity,
            started.elapsed().as_secs_f64() * 1000.0
        );
        pool.graph = Some(graph);
    }
    pool.graph.as_ref().unwrap().launch()?;
    advance_host(cache, rows, m.layers.len());
    pool.replays += 1;
    eprintln!(
        "[prime-graph] replay rows={rows} base={base} count={}",
        pool.replays
    );
    let mut output = e.uninit(rows * hidden)?;
    e.copy_into(&mut output, 0, &pool.output, rows * hidden)?;
    drop(scratch_scope);
    cache.qwen_prime_graph = Some(pool);
    Ok(output)
}

impl Engine {
    fn prepare_prime_graph(&self, capacity: usize) -> Result<()> {
        for name in [
            "append_quantize_kv_q8_0_q5_1_rows_prime_table",
            "fa_dequant_kv_ws_bf16_prime_table",
            "fa_prefill_qw_db_prime_table",
            "fa_prefill_qw_t3_prime_table",
            "ssm_conv1d_gdn_state_f32_prime_table",
            "ssm_conv_ring_update_f32_prime_table",
            "gdn_chunk_state_mma_prime_table",
            "prime_tap_table",
        ] {
            self.func(name);
        }
        let mut ws = self.prime_deqw_ws.lock().unwrap();
        let bytes = capacity * 4 * 256 * 2;
        if ws
            .as_ref()
            .is_none_or(|(k, v)| k.len() < bytes || v.len() < bytes)
        {
            *ws = Some((self.alloc_u8_uninit(bytes)?, self.alloc_u8_uninit(bytes)?));
        }
        Ok(())
    }

    pub(crate) fn trim_device_graph_mem(&self) -> Result<()> {
        unsafe {
            sys::cuDeviceGraphMemTrim(self.ctx().ordinal() as i32).result()?;
        }
        Ok(())
    }

    pub(crate) fn prime_tap_table(
        &self,
        x: &CudaSlice<f32>,
        table: u64,
        hidden: usize,
        rows: usize,
    ) -> Result<()> {
        let f = self.func("prime_tap_table");
        let (h, t) = (hidden as i32, rows as i32);
        let stream = self.stream();
        let mut b = stream.launch_builder(&f);
        b.arg(x).arg(&table).arg(&h).arg(&t);
        unsafe {
            b.launch(LaunchConfig::for_num_elems((rows * hidden) as u32))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires CUDA; run on an isolated GPU under /tmp/memra-gpu.lock"]
    fn tap_replay_follows_relocated_destination_and_stride() -> Result<()> {
        let e = Engine::new(0)?;
        let tracking = e.ctx().is_event_tracking();
        if tracking {
            unsafe { e.ctx().disable_event_tracking() };
        }
        let _tracking = EventTracking {
            engine: &e,
            restore: tracking,
        };
        // A copy must preserve every bit, including NaN payloads and signed zero.
        let bits = [
            0x8000_0000,
            0x7fc0_0011,
            0x3f80_0000,
            0xbf80_0000,
            0x7f80_0000,
            1,
        ];
        let input = e.htod(&bits.map(f32::from_bits))?;
        let sentinel = 0x7fc0_0022;
        let a = e.htod(&[f32::from_bits(sentinel); 18])?;
        let b = e.htod(&[f32::from_bits(sentinel); 18])?;
        let mut table = e.htod_u64(&[0; STRIDE])?;
        let pointer = table.device_ptr(&e.stream()).0;
        e.func("prime_tap_table");
        e.stream().synchronize()?;
        e.stream()
            .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)?;
        e.prime_tap_table(&input, pointer, 3, 2)?;
        let graph = e
            .stream()
            .end_capture(sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_UPLOAD)?
            .ok_or("tap test capture returned no graph")?;
        let mut expected_a = vec![sentinel; 18];
        for (destination, offset, stride) in [(&a, 2, 5), (&b, 1, 6)] {
            let mut words = [0; STRIDE];
            words[8] = destination.device_ptr(&e.stream()).0;
            words[9] = offset as u64;
            words[10] = stride as u64;
            e.htod_u64_into(&words, &mut table)?;
            graph.launch()?;
            let mut expected = vec![sentinel; 18];
            for row in 0..2 {
                expected[offset + row * stride..offset + row * stride + 3]
                    .copy_from_slice(&bits[row * 3..row * 3 + 3]);
            }
            assert_eq!(
                e.dtoh(destination)?
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                expected
            );
            if offset == 2 {
                expected_a = expected;
            }
        }
        // The second replay must not write its old destination.
        assert_eq!(
            e.dtoh(&a)?.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            expected_a
        );
        Ok(())
    }
}
