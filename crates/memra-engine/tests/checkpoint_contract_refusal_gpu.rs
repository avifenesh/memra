//! memra#541 on a real device: the hybrid loader binds the canonical tensor contract BEFORE any
//! upload. A byte-tampered copy of the glm-dsa micro fixture (one trunk tensor renamed in the
//! GGUF header, same length so every offset stays valid) is refused with the pack named and no
//! model built; the untampered fixture loads and logs the binding.
//!
//! GPU-gated: `#[ignore]` by default (CI is compile-only). Run on the rig:
//!   flock /tmp/memra-5090.lock cargo test -p memra-engine --test checkpoint_contract_refusal_gpu -- --ignored

use memra_engine::Engine;
use memra_engine::hybrid::HybridModel;
use memra_gguf::GgufFile;
use memra_gguf::micro_gguf::write_glm_dsa_micro;

fn tampered_copy(src: &std::path::Path, from: &str, to: &str) -> std::path::PathBuf {
    assert_eq!(from.len(), to.len());
    let mut bytes = std::fs::read(src).unwrap();
    let needle = from.as_bytes();
    let at = bytes
        .windows(needle.len())
        .position(|w| w == needle)
        .expect("tensor name present in header");
    bytes[at..at + needle.len()].copy_from_slice(to.as_bytes());
    let out = src.with_extension("tampered.gguf");
    std::fs::write(&out, bytes).unwrap();
    out
}

#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn gpu_loader_refuses_a_tampered_census_before_upload_and_loads_the_clean_one() {
    let p = std::env::temp_dir().join(format!("memra-541-gpu-{}.gguf", std::process::id()));
    write_glm_dsa_micro(&p, 0x6_10AD_0802).unwrap();
    let e = Engine::new(0).expect("CUDA device 0");

    // 1. renamed trunk tensor: missing under its contract name, extra under the new one
    let renamed = tampered_copy(&p, "blk.0.attn_norm.weight", "blk.0.attn_nOrm.weight");
    let g = GgufFile::open(&renamed).unwrap();
    let err = match HybridModel::load(&e, &g) {
        Ok(_) => panic!("a renamed trunk tensor must not load"),
        Err(err) => err.to_string(),
    };
    std::fs::remove_file(&renamed).ok();
    assert!(
        err.contains("checkpoint refused before upload (pack glm_dsa, Gguf"),
        "{err}"
    );
    assert!(err.contains("blk.0.attn_norm.weight"), "{err}");

    // 2. absent output head under a SeparateHead pack: refused, the embedding is not substituted
    let headless = tampered_copy(&p, "output.weight", "outpuT.weight");
    let g = GgufFile::open(&headless).unwrap();
    let err = match HybridModel::load(&e, &g) {
        Ok(_) => panic!("an absent head under a SeparateHead pack must not load"),
        Err(err) => err.to_string(),
    };
    std::fs::remove_file(&headless).ok();
    assert!(err.contains("output head ownership"), "{err}");
    assert!(err.contains("the embedding is not substituted"), "{err}");

    // 3. the clean fixture still loads through the same boundary
    let g = GgufFile::open(&p).unwrap();
    let model = HybridModel::load(&e, &g).expect("clean glm-dsa micro fixture loads");
    std::fs::remove_file(&p).ok();
    assert!(!model.layers.is_empty());
}
