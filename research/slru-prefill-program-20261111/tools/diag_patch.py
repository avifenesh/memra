import pathlib
import sys

p = pathlib.Path(sys.argv[1]) / "crates/memra-engine/src/hybrid_forward.rs"
s = p.read_text()


def once(old, new, count=1):
    global s
    n = s.count(old)
    assert n == count, (n, old[:80])
    s = s.replace(old, new, 1)


once(
    """        Self::moe_ffn_sequential_zq8(e, m, z, zq8, t, cfg, il, max_block, vrows)
    }
""",
    """        DIAG1217_ARM.with(|c| c.set("per-token-loop"));
        let out = Self::moe_ffn_sequential_zq8(e, m, z, zq8, t, cfg, il, max_block, vrows)?;
        if let Some(dir) = std::env::var_os("MEMRA_DIAG1217_DIR")
            && t > MOE_DEV_MAX_T
            && (il < 2 || il == u16::MAX)
        {
            let arm = DIAG1217_ARM.with(|c| c.get());
            let dir = std::path::PathBuf::from(dir);
            let zin = e.dtoh(z)?;
            let o = e.dtoh(&out)?;
            let b = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
            let tag = format!("il{il}-t{t}");
            std::fs::write(dir.join(format!("{tag}-in.bin")), b(&zin))?;
            std::fs::write(dir.join(format!("{tag}-out.bin")), b(&o))?;
            eprintln!("[diag1217] il={il} t={t} arm={arm}");
        }
        Ok(out)
    }
""",
)
once(
    """            return Self::moe_ffn_pairs(e, m, z, &logits, t, cfg);
        }
""",
    """            DIAG1217_ARM.with(|c| c.set("pairs"));
            return Self::moe_ffn_pairs(e, m, z, &logits, t, cfg);
        }
""",
)
once(
    """            && !observe_routes
        {
            return Self::moe_ffn_dev(e, m, z, zq8, &logits, t, cfg, il, max_block);
        }""",
    """            && !observe_routes
        {
            DIAG1217_ARM.with(|c| c.set("dev-slab"));
            return Self::moe_ffn_dev(e, m, z, zq8, &logits, t, cfg, il, max_block);
        }""",
)
once(
    """            if row_ok {
                return Self::moe_ffn_dev(e, m, z, zq8, &logits, t, cfg, il, max_block);""",
    """            if row_ok {
                DIAG1217_ARM.with(|c| c.set("dev-cache-row"));
                return Self::moe_ffn_dev(e, m, z, zq8, &logits, t, cfg, il, max_block);""",
)
once(
    """            let y_down = if f16g {
                // CSR order end-to-end""",
    """            DIAG1217_ARM.with(|c| c.set(if f16g { "pairs-f16g" } else { "pairs-mma" }));
            let y_down = if f16g {
                // CSR order end-to-end""",
)
once(
    """        let (zq, zd) = e.quantize_q8_1(z, t, n_embd)?;
        let gate = matvec(
            0,
            &exi,""",
    """        DIAG1217_ARM.with(|c| c.set("pairs-q8"));
        let (zq, zd) = e.quantize_q8_1(z, t, n_embd)?;
        let gate = matvec(
            0,
            &exi,""",
)
once(
    "pub const PRIME_MIN_T: usize = 16;",
    """pub const PRIME_MIN_T: usize = 16;
thread_local! {
    static DIAG1217_ARM: std::cell::Cell<&'static str> = const { std::cell::Cell::new("") };
}""",
)
p.write_text(s)
print("patched")
