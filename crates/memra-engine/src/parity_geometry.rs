//! Geometry-first parity: assert the SHAPE against config before believing any byte.
//!
//! WHY THIS EXISTS (GATE-INTEGRITY-20260819 §5, A/B shapes). An elementwise value compare
//! implicitly catches most tensor LAYOUT errors, because reordered values stop matching. It
//! catches **nothing** about geometry that lives in config scalars and never touches the bytes
//! — `n_rot`, `rope_base`/`rope_theta`, `sliding_window`, `head_dim` splits, tap sets. On
//! 2026-08-19 an `n_rot` defect (rotary width wrong by 4x) survived a byte-parity gate that ran
//! every single run at `maxdiff=0.0e0`, on 13/13 tensors, and the gate reported `ok`.
//!
//! Two of the parity binaries were one degree worse than blind: they took their geometry FROM
//! the bytes.
//!
//! ```ignore
//! let ctx = th.len() / (c.target_layer_ids.len() * c.hidden);   // dflash_parity.rs:33
//! let ctx = taps.len() / (n_taps * h);                          // dspark_q38_parity.rs
//! ```
//!
//! No remainder check, and no assertion that the dump was produced under the same config as the
//! checkpoint under test. A reference regenerated under a different `hidden`, a different tap
//! set or a different `block_size` is INDISTINGUISHABLE from a correct one: `ctx` silently
//! becomes another number and the comparison proceeds against a reinterpreted buffer. Every
//! structural check in those files was a PRODUCT — `ne.len() == block_size * hidden`,
//! `got.len() == want.len()`, `base.len() == (b-1) * v` — and a product is blind to any
//! factorisation that multiplies out the same.
//!
//! The shape to copy is `nv27b_probe` / `m3_probe` in memra-gguf: assert `ne` AND `ggml_type`
//! per tensor before believing anything. Geometry first, then bytes.
//!
//! So: the oracle that produces a dump also writes a GEOMETRY MANIFEST next to it, and the
//! parity gate refuses to interpret a dump whose manifest disagrees with the checkpoint — or is
//! missing. A missing manifest is a REFUSAL, not a warning and not a skip: "we cannot tell which
//! program produced these bytes" is the exact state this module exists to make impossible.
//!
//! Deliberately dependency-free (`std` only, no serde) so it can be included by
//! `#[path = "../parity_geometry.rs"] mod parity_geometry;` from a bin without becoming part of
//! the library's public surface, and so its own tests run under a bare
//! `rustc --test crates/memra-engine/src/parity_geometry.rs` on a box with no CUDA at all.
//! That matters: both parity binaries need a GPU, so this is the only part of the fix that can
//! be proven without one.

use std::collections::BTreeMap;

/// A geometry manifest: flat `key=value` lines, `#` comments, blank lines ignored.
///
/// Flat text rather than JSON on purpose — no dependency, and a receipt a human greps.
#[derive(Debug, Clone)]
pub struct Geometry {
    path: String,
    map: BTreeMap<String, String>,
}

/// The manifest filename that must sit beside a dump family, e.g. `dflash` -> `dflash-geometry.txt`.
pub fn manifest_path(cache_dir: &str, family: &str) -> String {
    format!("{cache_dir}/{family}-geometry.txt")
}

impl Geometry {
    /// Load a manifest, or return the refusal message naming the regeneration command.
    pub fn load(path: &str, regen_hint: &str) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| {
            format!(
                "GEOMETRY MANIFEST MISSING: {path}: {e}\n  \
                 This gate compares bytes against a reference dump. Without the manifest there \
                 is no way to assert the dump was produced under the same config as the \
                 checkpoint under test, and a reference regenerated under a different hidden \
                 size, tap set or block size is indistinguishable from a correct one \
                 (GATE-INTEGRITY-20260819 §5). Refusing rather than guessing.\n  \
                 Regenerate the reference (and its manifest) with:\n    {regen_hint}"
            )
        })?;
        let mut map = BTreeMap::new();
        for (i, raw) in text.lines().enumerate() {
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let (k, v) = line
                .split_once('=')
                .ok_or_else(|| format!("{path}:{}: not a key=value line: {raw:?}", i + 1))?;
            if map
                .insert(k.trim().to_string(), v.trim().to_string())
                .is_some()
            {
                return Err(format!("{path}:{}: duplicate key {:?}", i + 1, k.trim()));
            }
        }
        if map.is_empty() {
            return Err(format!(
                "{path}: manifest is EMPTY — an empty manifest asserts nothing and must not \
                 read as agreement"
            ));
        }
        Ok(Self {
            path: path.to_string(),
            map,
        })
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.map.get(key).map(String::as_str)
    }

    /// Required numeric field.
    pub fn need_usize(&self, key: &str) -> Result<usize, String> {
        let raw = self
            .get(key)
            .ok_or_else(|| format!("{}: missing required key {key}", self.path))?;
        raw.parse::<usize>()
            .map_err(|e| format!("{}: {key}={raw:?} is not a usize: {e}", self.path))
    }

    /// The manifest's value for `key` must equal the checkpoint's `want`.
    ///
    /// This is the inversion that matters: the config is the authority and the dump is the
    /// claimant, never the other way round.
    pub fn expect_usize(&self, key: &str, want: usize) -> Result<(), String> {
        let got = self.need_usize(key)?;
        if got == want {
            return Ok(());
        }
        Err(format!(
            "GEOMETRY MISMATCH {key}: reference dump was produced with {got}, the checkpoint \
             under test has {want}.\n  The bytes would still compare — a different {key} \
             reinterprets the same buffer — which is why this is asserted BEFORE any value \
             compare. Manifest: {}",
            self.path
        ))
    }

    /// Same, for a string-valued field (tap sets, dtype, per-layer flag vectors).
    pub fn expect_str(&self, key: &str, want: &str) -> Result<(), String> {
        let got = self
            .get(key)
            .ok_or_else(|| format!("{}: missing required key {key}", self.path))?;
        if got == want {
            return Ok(());
        }
        Err(format!(
            "GEOMETRY MISMATCH {key}: reference dump was produced with {got:?}, the checkpoint \
             under test has {want:?}. Manifest: {}",
            self.path
        ))
    }

    /// Numeric field with a tolerance, for the float scalars that are pure config and never
    /// reach the bytes: `rope_theta` is the one that motivated this module's existence in a
    /// sibling gate (identical bytes under a different rotary base is still a different model).
    pub fn expect_f64_near(&self, key: &str, want: f64, rel: f64) -> Result<(), String> {
        let raw = self
            .get(key)
            .ok_or_else(|| format!("{}: missing required key {key}", self.path))?;
        let got: f64 = raw
            .parse()
            .map_err(|e| format!("{}: {key}={raw:?} is not a number: {e}", self.path))?;
        let denom = want.abs().max(f64::MIN_POSITIVE);
        if ((got - want).abs() / denom) <= rel {
            return Ok(());
        }
        Err(format!(
            "GEOMETRY MISMATCH {key}: reference dump was produced with {got}, the checkpoint \
             under test has {want} (rel tol {rel}). Manifest: {}",
            self.path
        ))
    }

    /// Optional field: asserted when the producer wrote it, absent-tolerated otherwise.
    ///
    /// For fields an upstream config legitimately leaves unset (`sliding_window` is `None` on
    /// non-SWA models). Absence is reported so a silently-thinning manifest is visible.
    pub fn expect_usize_if_present(&self, key: &str, want: usize) -> Result<bool, String> {
        if self.get(key).is_none() {
            return Ok(false);
        }
        self.expect_usize(key, want)?;
        Ok(true)
    }

    /// Same, string-valued. Used for per-layer flag vectors whose upstream representation
    /// (`layer_types`) an export may nest or omit.
    pub fn expect_str_if_present(&self, key: &str, want: &str) -> Result<bool, String> {
        if self.get(key).is_none() {
            return Ok(false);
        }
        self.expect_str(key, want)?;
        Ok(true)
    }
}

/// A dump's element count must EQUAL the product the config predicts — not merely divide by it.
///
/// The replaced idiom was `let ctx = th.len() / (n_taps * hidden);`. Integer division silently
/// accepts a remainder, and an inferred `ctx` cannot disagree with anything.
pub fn expect_len(dump: &str, got: usize, want: usize, how: &str) -> Result<(), String> {
    if got == want {
        return Ok(());
    }
    Err(format!(
        "GEOMETRY MISMATCH {dump}: {got} elements, config predicts {want} ({how}).\n  \
         The old form DERIVED a dimension from this length by integer division, so a dump of the \
         wrong size became a comparison against a reinterpreted buffer instead of a failure."
    ))
}

/// Divide with a remainder check, for the one place a dimension genuinely has to be read out of
/// a length. Returns an error naming the remainder rather than truncating it away.
///
/// Currently no CALLER, deliberately: both parity gates now take every dimension from the
/// manifest and assert lengths with [`expect_len`], which is strictly stronger. It stays here,
/// tested, as the replacement for the `len / per` idiom the next gate will reach for — the
/// remaining known site is `crates/memra-gguf/examples/dequant_oracle_diff.rs`, in another crate
/// (GATE-INTEGRITY-20260819 §5).
#[allow(dead_code)]
pub fn exact_div(dump: &str, len: usize, per: usize, per_desc: &str) -> Result<usize, String> {
    if per == 0 {
        return Err(format!(
            "{dump}: cannot divide by {per_desc}=0 — the config is degenerate, not the dump"
        ));
    }
    if len % per != 0 {
        return Err(format!(
            "GEOMETRY MISMATCH {dump}: {len} elements is not a multiple of {per_desc}={per} \
             (remainder {}). The old form used integer division and silently truncated this.",
            len % per
        ));
    }
    if len == 0 {
        return Err(format!("{dump}: empty dump"));
    }
    Ok(len / per)
}

/// Format a usize slice the way the oracles write it: comma-separated, no spaces.
pub fn join_usize(v: &[usize]) -> String {
    v.iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

/// Format a bool slice as `1`/`0`, for per-layer flag vectors (`layer_sliding`).
pub fn join_bool(v: &[bool]) -> String {
    v.iter()
        .map(|b| if *b { "1" } else { "0" })
        .collect::<Vec<_>>()
        .join(",")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(body: &str) -> String {
        let p = std::env::temp_dir().join(format!(
            "memra_parity_geometry_{}_{:?}.txt",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&p, body).unwrap();
        p.to_string_lossy().into_owned()
    }

    const GOOD: &str = "\
# produced by tools/dflash_oracle.py
producer=tools/dflash_oracle.py
dtype=f32
hidden=2560
n_layer=5
block_size=8
ctx=8
n_taps=5
target_layer_ids=13,26,39,52,61
head_dim=128
n_head=20
n_head_kv=4
rope_theta=1000000.0
sliding_window=2048
layer_sliding=1,1,1,1,0
";

    #[test]
    fn a_missing_manifest_is_a_refusal_that_names_the_remedy() {
        let err = Geometry::load("/nonexistent/dflash-geometry.txt", "python tools/x.py A B")
            .expect_err("a missing manifest must not load");
        assert!(err.contains("GEOMETRY MANIFEST MISSING"), "{err}");
        assert!(err.contains("python tools/x.py A B"), "{err}");
    }

    #[test]
    fn an_empty_manifest_is_not_agreement() {
        let p = write("\n# only comments\n\n");
        let err = Geometry::load(&p, "regen").expect_err("empty manifest must not load");
        assert!(err.contains("EMPTY"), "{err}");
        std::fs::remove_file(p).ok();
    }

    #[test]
    fn a_matching_manifest_passes_every_field() {
        let p = write(GOOD);
        let g = Geometry::load(&p, "regen").unwrap();
        g.expect_str("dtype", "f32").unwrap();
        g.expect_usize("hidden", 2560).unwrap();
        g.expect_usize("block_size", 8).unwrap();
        g.expect_usize("n_taps", 5).unwrap();
        g.expect_str("target_layer_ids", &join_usize(&[13, 26, 39, 52, 61]))
            .unwrap();
        g.expect_str(
            "layer_sliding",
            &join_bool(&[true, true, true, true, false]),
        )
        .unwrap();
        g.expect_f64_near("rope_theta", 1e6, 1e-9).unwrap();
        assert!(g.expect_usize_if_present("sliding_window", 2048).unwrap());
        assert_eq!(g.need_usize("ctx").unwrap(), 8);
        std::fs::remove_file(p).ok();
    }

    /// THE DEFECT CLASS. A reference regenerated with hidden=1280 and twice the taps has the
    /// IDENTICAL total byte count: 8 * 5 * 2560 == 8 * 10 * 1280. The old code divided that
    /// length by (n_taps * hidden), got a `ctx` that made the arithmetic close, and compared
    /// away. Products are blind to their own factorisations; the manifest is not.
    #[test]
    fn a_refactorised_geometry_with_the_same_product_is_caught() {
        let p = write(
            &GOOD
                .replace("hidden=2560", "hidden=1280")
                .replace("n_taps=5", "n_taps=10"),
        );
        let g = Geometry::load(&p, "regen").unwrap();
        let err = g
            .expect_usize("hidden", 2560)
            .expect_err("must catch hidden");
        assert!(err.contains("GEOMETRY MISMATCH hidden"), "{err}");
        assert!(err.contains("2560") && err.contains("1280"), "{err}");
        // And the product itself still matches, which is exactly why the old check passed.
        let old_total = 8 * 5 * 2560;
        let new_total = 8 * 10 * 1280;
        assert_eq!(old_total, new_total, "the fixture's premise");
        std::fs::remove_file(p).ok();
    }

    /// The scalars that never touch a byte: a 4x rotary base with byte-identical dumps.
    #[test]
    fn a_config_only_scalar_is_caught_even_though_no_byte_moves() {
        let p = write(&GOOD.replace("rope_theta=1000000.0", "rope_theta=250000.0"));
        let g = Geometry::load(&p, "regen").unwrap();
        let err = g
            .expect_f64_near("rope_theta", 1e6, 1e-6)
            .expect_err("must catch rope_theta");
        assert!(err.contains("GEOMETRY MISMATCH rope_theta"), "{err}");
        // head_dim is the same class (the n_rot family) and is asserted the same way.
        let p2 = write(&GOOD.replace("head_dim=128", "head_dim=64"));
        let g2 = Geometry::load(&p2, "regen").unwrap();
        assert!(
            g2.expect_usize("head_dim", 128)
                .expect_err("must catch head_dim")
                .contains("GEOMETRY MISMATCH head_dim")
        );
        std::fs::remove_file(p).ok();
        std::fs::remove_file(p2).ok();
    }

    #[test]
    fn a_tap_set_permutation_is_caught_though_the_count_agrees() {
        let p = write(&GOOD.replace("13,26,39,52,61", "13,26,39,52,60"));
        let g = Geometry::load(&p, "regen").unwrap();
        g.expect_usize("n_taps", 5).expect("count still agrees");
        let err = g
            .expect_str("target_layer_ids", &join_usize(&[13, 26, 39, 52, 61]))
            .expect_err("the tap SET must be asserted, not just its length");
        assert!(err.contains("GEOMETRY MISMATCH target_layer_ids"), "{err}");
        std::fs::remove_file(p).ok();
    }

    #[test]
    fn a_dropped_optional_field_is_reported_not_silently_satisfied() {
        let p = write(&GOOD.replace("sliding_window=2048\n", ""));
        let g = Geometry::load(&p, "regen").unwrap();
        assert!(
            !g.expect_usize_if_present("sliding_window", 2048).unwrap(),
            "absence must be reported as `false`, never as agreement"
        );
        std::fs::remove_file(p).ok();
    }

    #[test]
    fn expect_len_replaces_the_derive_and_a_wrong_length_fails() {
        expect_len(
            "dflash-target_hidden.f32",
            8 * 5 * 2560,
            8 * 5 * 2560,
            "ctx*n_taps*hidden",
        )
        .unwrap();
        let err = expect_len(
            "dflash-target_hidden.f32",
            8 * 5 * 2560 + 2560,
            8 * 5 * 2560,
            "ctx*n_taps*hidden",
        )
        .expect_err("a longer dump must fail, not redefine ctx");
        assert!(
            err.contains("GEOMETRY MISMATCH dflash-target_hidden.f32"),
            "{err}"
        );
    }

    #[test]
    fn exact_div_refuses_a_remainder_instead_of_truncating() {
        assert_eq!(exact_div("d", 80, 10, "n_taps*hidden").unwrap(), 8);
        // The old `len / per` returned 8 here too, quietly discarding 3 elements.
        let err = exact_div("d", 83, 10, "n_taps*hidden").expect_err("remainder must fail");
        assert!(err.contains("remainder 3"), "{err}");
        assert!(exact_div("d", 0, 10, "x").is_err(), "empty dump must fail");
        assert!(exact_div("d", 80, 0, "x").is_err(), "div by zero must fail");
    }

    #[test]
    fn a_malformed_or_duplicated_key_is_an_error_not_a_shrug() {
        let p = write("hidden=2560\nnot a kv line\n");
        assert!(Geometry::load(&p, "r").is_err());
        std::fs::remove_file(p).ok();
        let p = write("hidden=2560\nhidden=1280\n");
        let err = Geometry::load(&p, "r").expect_err("duplicate key must fail");
        assert!(err.contains("duplicate key"), "{err}");
        std::fs::remove_file(p).ok();
    }

    #[test]
    fn a_missing_required_key_names_itself() {
        let p = write("hidden=2560\n");
        let g = Geometry::load(&p, "r").unwrap();
        assert!(
            g.need_usize("ctx")
                .unwrap_err()
                .contains("missing required key ctx")
        );
        assert!(
            g.expect_str("dtype", "f32")
                .unwrap_err()
                .contains("missing required key dtype")
        );
        std::fs::remove_file(p).ok();
    }
}
