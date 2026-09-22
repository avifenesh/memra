//! Actual dense/hybrid root loading of pinned, real tiny checkpoint containers.
//! Launch one case per process under the existing physical-GPU lease wrapper.
//! This gate reports unqualified fixture loading/readback, never rewrite qualification.
use memra_engine::{
    Engine,
    hybrid::HybridModel,
    model::{GpuTensor, Model},
};
use memra_gguf::{
    GgufFile,
    source::{GgufSource, SafetensorsSource, TensorSource},
};
use sha2::{Digest, Sha256};
use std::{error::Error, path::PathBuf};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
fn setting(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("missing {name}"))
}
fn lease() -> Vec<u8> {
    assert!(
        std::env::var_os("DOCS_RS").is_none(),
        "documentation stubs are not native"
    );
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../research/modelplan-onboarding-rewrite-identity-20260920/qualify-native.py");
    let out = std::process::Command::new("python3").args(["-B", "-c", "import json,runpy,sys; m=runpy.run_path(sys.argv[1]); print(json.dumps(m['verify_lease'](),sort_keys=True))"]).arg(script).output().expect("existing lease verifier");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    out.stdout
}
fn read_head(e: &Engine, t: &GpuTensor) -> Result<(Vec<u8>, String)> {
    match t {
        GpuTensor::Float { data, ne } => {
            assert_eq!(ne, &[256, 64]);
            Ok((
                e.dtoh(data)?
                    .into_iter()
                    .flat_map(f32::to_le_bytes)
                    .collect(),
                "Float".into(),
            ))
        }
        GpuTensor::Quant {
            bytes,
            ne,
            qtype,
            row_bytes,
            scale,
            rp,
            ..
        } => {
            assert_eq!(ne, &[256, 64]);
            assert_eq!(*qtype, memra_engine::QT_Q5_K);
            assert_eq!(*row_bytes, 176);
            assert_eq!(scale.to_bits(), 1f32.to_bits());
            assert!(!rp);
            Ok((e.dtoh_u8(bytes)?, "Quant".into()))
        }
        _ => Err("unexpected native output representation for declared fixture".into()),
    }
}
fn query(e: &Engine, t: &GpuTensor) -> Result<()> {
    if setting("MEMRA_BOUND_NVFP4_QUERY") != "1" {
        return Ok(());
    }
    let GpuTensor::Quant {
        bytes,
        ne,
        qtype,
        scale,
        ..
    } = t
    else {
        return Err("NVFP4 query lost native representation".into());
    };
    assert_eq!(ne, &[256, 256]);
    assert_eq!(*qtype, memra_engine::QT_NVFP4);
    assert_eq!(scale.to_bits(), 2f32.to_bits());
    let raw = e.dtoh_u8(bytes)?;
    assert_eq!(raw.len(), 36864);
    assert_eq!(raw.iter().filter(|&&b| b == 0x38).count(), 4096);
    assert_eq!(raw.iter().filter(|&&b| b == 0x22).count(), 32768);
    std::fs::write(
        PathBuf::from(setting("MEMRA_BOUND_OUT")).join("query.bytes"),
        raw,
    )?;
    Ok(())
}
fn root(e: &Engine, s: &dyn TensorSource, which: &str) -> Result<(Vec<u8>, String)> {
    match which {
        "dense" => {
            let m = Model::load_dense_from_source(e, s)?;
            query(e, &m.layers[0].wq)?;
            read_head(e, &m.output)
        }
        "hybrid" => {
            let m = HybridModel::load_from_source_without_mtp(e, s)?;
            assert!(
                !m.rewrite_is_qualified(),
                "fixture must not claim qualification"
            );
            if let memra_engine::hybrid::Mixer::Full(fa) = &m.layers[0].mixer {
                query(e, &fa.wq)?;
            } else {
                return Err("fixture changed mixer".into());
            }
            read_head(e, &m.output)
        }
        _ => Err("unknown root".into()),
    }
}
#[test]
#[ignore = "requires native build, pinned fixtures and coordinator-assigned GPU lease"]
fn native_bound_loader_case() {
    assert!(std::env::var_os("MEMRA_REWRITE_BUNDLE").is_none());
    assert!(std::env::var_os("MEMRA_ARTIFACT_LOCK").is_none());
    assert!(
        memra_engine::alloc_trace_on(),
        "pre-upload observation requires allocation trace"
    );
    let lease_before = lease();
    let e = Engine::new(0).expect("native CUDA Engine");
    let case = setting("MEMRA_BOUND_CASE");
    let which = setting("MEMRA_BOUND_ROOT");
    let path = PathBuf::from(setting("MEMRA_BOUND_CASE_PATH"));
    let format = setting("MEMRA_BOUND_FORMAT");
    let out = PathBuf::from(setting("MEMRA_BOUND_OUT"));
    std::fs::create_dir(&out).unwrap();
    eprintln!("BOUND_LOADER_BEGIN {case} {which} unqualified_fixture");
    let result = match format.as_str() {
        "gguf" => GgufFile::open(&path)
            .map_err(|x| -> Box<dyn Error> { Box::new(x) })
            .and_then(|g| root(&e, &GgufSource(&g), &which)),
        "hf" => SafetensorsSource::open(&path)
            .map_err(|x| -> Box<dyn Error> { Box::new(x) })
            .and_then(|s| root(&e, &s, &which)),
        _ => panic!("unknown format"),
    };
    e.stream().synchronize().unwrap();
    let expected_error = setting("MEMRA_BOUND_EXPECT_ERROR");
    if expected_error.is_empty() {
        let (bytes, kind) = result.unwrap();
        assert_eq!(kind, setting("MEMRA_BOUND_OUTPUT_KIND"));
        assert_eq!(
            format!("{:x}", Sha256::digest(&bytes)),
            setting("MEMRA_BOUND_OUTPUT_SHA256")
        );
        std::fs::write(out.join("output.bytes"), bytes).unwrap();
        eprintln!("BOUND_LOADER_END {case} {which} accepted_unqualified_fixture");
    } else {
        let err = result
            .expect_err("malformed checkpoint must refuse")
            .to_string();
        assert!(err.contains(&expected_error), "wrong refusal: {err}");
        std::fs::write(out.join("error.txt"), err).unwrap();
        eprintln!("BOUND_LOADER_END {case} {which} contextual_refusal");
    }
    assert_eq!(lease_before, lease());
}
