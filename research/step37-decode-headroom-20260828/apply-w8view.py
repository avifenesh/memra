#!/usr/bin/env python3
"""Apply MEMRA_W8_VIEW to crates/memra-engine/src/lib.rs by ANCHOR, not by diff context.

The box's lib.rs is a different tree from the lane branch AND carries a sibling agent's
uncommitted edits, so a context diff would fuzz or fail. Every edit here is keyed on a string
that exists in both trees, each one is asserted, and the script refuses a second application.
Usage: apply.py <path-to-lib.rs>   (writes in place; prints APPLIED or refuses loudly)
"""
import sys

p = sys.argv[1]
s = open(p).read()
if "w8_view_on" in s:
    print("REFUSED: already applied (w8_view_on present)"); sys.exit(2)

def sub(old, new, count=1):
    global s
    n = s.count(old)
    assert n == count, "anchor count %d != %d for %r" % (n, count, old[:80])
    s = s.replace(old, new)

# 1. mirror map keyed on (ptr, in_f, out_f): a row-range VIEW starting at row 0 carries the
#    PARENT slab's pointer, so a pointer-only key would hand the head-split lo half the full
#    head's mirror and read past the rows it owns.
sub("    w8_mirrors: Mutex<std::collections::HashMap<u64, CudaSlice<u8>>>,",
    """    /// KEYED ON (pointer, in_f, out_f), not on the pointer alone: a row-range VIEW of a slab
    /// carries the PARENT's base pointer when the range starts at row 0, so a pointer-only key
    /// would hand the head-split lo half (4096 x 64448) the full head's mirror (4096 x 128896)
    /// and read 2x past the rows it owns. The shape is part of the identity of a mirror.
    w8_mirrors: Mutex<std::collections::HashMap<(u64, u32, u32), CudaSlice<u8>>>,""")
sub("""        let key = {
            let s = self.gpu.stream();
            let (p, _g) = data.device_ptr(&s);
            p as u64
        };""",
    """        let key = {
            let s = self.gpu.stream();
            let (p, _g) = data.device_ptr(&s);
            (p as u64, in_f as u32, out_f as u32)
        };""", count=2)

# 2. view twin of the q8_0 encoder (same kernel, same geometry, different operand type)
enc_anchor = "    /// Fused q8_0 QKV against a q8_1 activation (MEMRA_STEP_TP_W8): one launch over the"
sub(enc_anchor, '''    /// ROW-RANGE-VIEW twin of `encode_q8_0_from_bf16`. Identical kernel, identical launch
    /// geometry, identical per-row program: only the operand type differs, because the split
    /// decode paths hold their rows as a `CudaView` of the resident slab, not as an owned slab.
    pub fn encode_q8_0_from_bf16_view(
        &self,
        w_bf16: &cudarc::driver::CudaView<'_, u8>,
        out: &mut CudaSlice<u8>,
        in_f: usize,
        out_f: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if in_f % 32 != 0
            || w_bf16.len() < in_f * out_f * 2
            || out.len() < out_f * Self::q8_0_row_bytes(in_f)
        {
            return Err(format!(
                "encode_q8_0_from_bf16_view geometry in={in_f} out={out_f} src={} dst={}",
                w_bf16.len(),
                out.len()
            )
            .into());
        }
        let f = self.func("encode_q8_0_rows_from_bf16");
        const PAIRS_PER_BLOCK: u32 = 4;
        let pairs = (out_f * (in_f / 32)) as u64;
        let cfg = LaunchConfig {
            grid_dim: ((pairs.div_ceil(PAIRS_PER_BLOCK as u64)) as u32, 1, 1),
            block_dim: (32, PAIRS_PER_BLOCK, 1),
            shared_mem_bytes: 0,
        };
        let (ini, outi) = (in_f as i32, out_f as i32);
        let __s_b = self.gpu.stream();
        let mut b = __s_b.launch_builder(&f);
        b.arg(w_bf16).arg(out).arg(&ini).arg(&outi);
        unsafe {
            b.launch(cfg)?;
        }
        Ok(())
    }

''' + enc_anchor)

# 3. the view mirror GEMV
sub("""    pub fn matvec_bf16_into(
        &self,
        data: &CudaSlice<u8>,""",
'''    /// MEMRA_W8_VIEW: the q8_0 mirror for a bf16 GEMV whose weight is a ROW-RANGE VIEW.
    /// `MEMRA_W8_HYBRID` hangs off `matvec_bf16_into`, and the two split decode paths pinned in
    /// the step37 serving env send only their HI half there: HEAD_SPLIT runs
    /// `rank1.matvec_bf16_into(head_hi)` beside `e.matvec_bf16_view_into(head_lo)`, and
    /// SHEXP_OVERLAP does the same with the shared-expert down rows. The view launcher had no
    /// mirror, so the lo half kept streaming 2 B/w while its twin ran at 1.0625, and because the
    /// halves execute CONCURRENTLY on the two cards the critical path is the SLOW half.
    /// NUMERIC CLASS: identical to the rest of `MEMRA_STEP_TP_W8`, so it carries that argmax
    /// acceptance and that maxdiff class, not a new one. Default OFF until measured.
    fn matvec_bf16_view_via_q8_mirror(
        &self,
        data: &cudarc::driver::CudaView<'_, u8>,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        in_f: usize,
        out_f: usize,
    ) -> Result<Option<()>, Box<dyn std::error::Error>> {
        use cudarc::driver::DevicePtr;
        let key = {
            let s = self.gpu.stream();
            let (p, _g) = data.device_ptr(&s);
            (p as u64, in_f as u32, out_f as u32)
        };
        {
            let mut mirrors = self
                .w8_mirrors
                .lock()
                .map_err(|_| "w8 mirror map is poisoned")?;
            if !mirrors.contains_key(&key) {
                let mut interleaved = self.alloc_u8_uninit(out_f * Self::q8_0_row_bytes(in_f))?;
                self.encode_q8_0_from_bf16_view(data, &mut interleaved, in_f, out_f)?;
                let planar = self.build_q8_rp4_raw(&interleaved, in_f, out_f)?;
                mirrors.insert(key, planar);
                // Unconditional, once per distinct shape: a door with no announce cannot be read
                // in BOTH directions, and this lane was already burned once by a sweep that
                // inferred "never engages" from a log line that did not exist in the tree.
                eprintln!("[w8-view] mirror built in_f={in_f} out_f={out_f}");
            }
        }
        let nblk = in_f / 32;
        {
            let mut act = self.w8_act.lock().map_err(|_| "w8 act map is poisoned")?;
            if !act.contains_key(&in_f) {
                let aq = self.alloc_uninit::<i8>(in_f)?;
                let ad = self.alloc_uninit::<f32>(nblk)?;
                act.insert(in_f, (aq, ad));
            }
            let (aq, ad) = act.get_mut(&in_f).expect("just inserted");
            self.quantize_q8_1_into(x, 1, in_f, aq, ad)?;
        }
        let mirrors = self
            .w8_mirrors
            .lock()
            .map_err(|_| "w8 mirror map is poisoned")?;
        let act = self.w8_act.lock().map_err(|_| "w8 act map is poisoned")?;
        let mirror = mirrors.get(&key).expect("built above");
        let (aq, ad) = act.get(&in_f).expect("built above");
        self.qmatvec_mmvq_into(
            mirror,
            aq,
            ad,
            1,
            in_f,
            out_f,
            QT_Q8_0,
            Self::q8_0_row_bytes(in_f),
            1.0,
            true,
            y,
        )?;
        Ok(Some(()))
    }

    pub fn matvec_bf16_into(
        &self,
        data: &CudaSlice<u8>,''')

# 4. route matvec_bf16_view_into through it (the tail AFTER that fn's own geometry check)
tail = """        let f = self.func("matvec_bf16_f32acc");
        let cfg = LaunchConfig {
            grid_dim: (out_f as u32, 1, 1),
            block_dim: (mmv_block(), 1, 1),
            shared_mem_bytes: 0,
        };
        let ini = in_f as i32;
        let __s_b = self.gpu.stream();
        let mut b = __s_b.launch_builder(&f);
        b.arg(data).arg(x).arg(y).arg(&ini);
        unsafe {
            b.launch(cfg)?;
        }
        Ok(())
    }
"""
gate = """        if w8_view_on() && step_tp_w8_on() && w8_hybrid_on() && in_f % 32 == 0 && out_f >= 64 {
            if let Some(()) = self.matvec_bf16_view_via_q8_mirror(data, x, y, in_f, out_f)? {
                return Ok(());
            }
        }
"""
marker = "matvec_bf16_view_into geometry bytes={} x={} y={} in={in_f} out={out_f}"
i = s.index(marker)
j = s.index(tail, i)
s = s[:j] + gate + tail + s[j + len(tail):]

# 5. the door itself
sub("""pub(crate) fn step_tp_w8_on() -> bool {
    static ENV: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
    step37_door(&ENV, "MEMRA_STEP_TP_W8")
}""",
"""pub(crate) fn step_tp_w8_on() -> bool {
    static ENV: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
    step37_door(&ENV, "MEMRA_STEP_TP_W8")
}

/// MEMRA_W8_VIEW=1: extend the W8 hybrid half to the ROW-RANGE-VIEW GEMVs, i.e. the lo halves
/// that `MEMRA_HEAD_SPLIT` and `MEMRA_SHEXP_OVERLAP` keep on rank 0. NOT a step37 family door
/// and NOT armed by `arm_step37_serving_defaults`: it stays off until it carries its own
/// interleaved speed rows and its own argmax gate. Unset or `=0` is the rollback seam.
pub(crate) fn w8_view_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MEMRA_W8_VIEW").as_deref() == Ok("1"))
}""")

open(p, "w").write(s)
print("APPLIED")
