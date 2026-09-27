//! Small real safetensors artifacts and evidence helpers; no mock TensorSource or repacker.
use memra_engine::Engine;
use memra_engine::model::HostExps;
use memra_gguf::source::{SafetensorsSource, TensorSource};
use sha2::{Digest, Sha256};
use std::fs::{File, OpenOptions, Permissions};
use std::io::{Read, Write};
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

pub const EXPERTS: usize = 2;
pub const ROWS: usize = 512;
pub const COLS: usize = 128;
pub const ROW_BYTES: usize = COLS / 64 * 36;
pub const STRIDE: usize = ROWS * ROW_BYTES;
pub const MACROS: [f32; EXPERTS] = [0.25, 0.5];

#[derive(Clone, Copy)]
pub enum Layout {
    Stacked,
    PerExpert,
}

impl Layout {
    pub fn name(self) -> &'static str {
        match self {
            Self::Stacked => "native_nvfp4_stacked_cache_lifecycle",
            Self::PerExpert => "native_nvfp4_per_expert_cache_lifecycle",
        }
    }

    fn tensor_name(self) -> &'static str {
        match self {
            Self::Stacked => "blk.1.ffn_gate_exps.weight",
            Self::PerExpert => "blk.0.ffn_gate_exps.weight",
        }
    }

    pub fn load(self, engine: &Engine, source: &SafetensorsSource) -> HostExps {
        match self {
            Self::Stacked => {
                assert!(
                    source
                        .find_nvfp4_stacked_native(self.tensor_name())
                        .is_some()
                );
                HostExps::load_stacked_from_source(engine, source, self.tensor_name())
            }
            Self::PerExpert => {
                for expert in 0..EXPERTS {
                    assert!(
                        source
                            .find_nvfp4_native(&format!("blk.0.ffn_gate_exps.{expert}.weight"))
                            .is_some()
                    );
                }
                HostExps::load_from_source(engine, source, self.tensor_name(), EXPERTS)
            }
        }
        .expect("public HostExps native disk loader must succeed")
    }

    pub fn cache(self, source: &SafetensorsSource, artifact: &Path) -> PathBuf {
        let prefix = match self {
            Self::Stacked => format!("{}-stacked-", self.tensor_name().replace('.', "-")),
            Self::PerExpert => "blk0-gate-".to_owned(),
        };
        artifact.join(".memra-repack").join(format!(
            "{prefix}{EXPERTS}x{ROWS}x{COLS}{}.nvfp4",
            source.nvfp4_cache_tag()
        ))
    }
}

/// Reuse the parent lane's ancestor/physical-UUID/FLOCK verifier. It performs no GPU calls.
/// The external memra-gpu-run wrapper owns the lock throughout this process's lifetime.
pub fn native_lease() -> Vec<u8> {
    assert!(
        std::env::var_os("MEMRA_ARTIFACT_LOCK").is_some()
            || std::env::var_os("MEMRA_REWRITE_BUNDLE").is_some(),
        "launch with MEMRA_ARTIFACT_LOCK or MEMRA_REWRITE_BUNDLE present; never set it in-test"
    );
    assert_ne!(std::env::var("MEMRA_ST_REPACK_DISK").as_deref(), Ok("0"));
    assert_ne!(std::env::var("MEMRA_ST_PINNED").as_deref(), Ok("1"));
    let device = std::env::var("CUDA_VISIBLE_DEVICES")
        .expect("launch with exactly one assigned CUDA_VISIBLE_DEVICES device");
    assert!(!device.trim().is_empty() && device != "-1" && !device.contains(','));
    assert!(
        std::env::var_os("DOCS_RS").is_none(),
        "DOCS_RS is compile-only"
    );
    let verifier = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../research/modelplan-onboarding-rewrite-identity-20260920/qualify-native.py");
    let output = std::process::Command::new("python3")
        .args([
            "-B",
            "-c",
            "import json, runpy, sys; module = runpy.run_path(sys.argv[1]); print(json.dumps(module['verify_lease'](), sort_keys=True))",
        ])
        .arg(verifier)
        .output()
        .expect("python3 and the parent native lease verifier must be available");
    assert!(
        output.status.success(),
        "assigned native lease verification failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        !output.stdout.is_empty(),
        "lease verifier returned no evidence"
    );
    output.stdout
}

pub fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub fn file_sha256(path: &Path) -> String {
    let mut file = File::open(path).unwrap();
    let mut hash = Sha256::new();
    let mut buf = [0; 64 * 1024];
    loop {
        let n = file.read(&mut buf).unwrap();
        if n == 0 {
            break;
        }
        hash.update(&buf[..n]);
    }
    format!("{hash:x}", hash = hash.finalize())
}

pub fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

pub fn input() -> Vec<f32> {
    [1.0, 0.5, -0.25, 2.0]
        .into_iter()
        .cycle()
        .take(COLS)
        .collect()
}

fn code(expert: usize, row: usize) -> u8 {
    // Equal nibbles deliberately make the expected interleave independent of production repack.
    let nibble = 2 + expert as u8 * 2 + if row.is_multiple_of(2) { 0 } else { 8 };
    nibble | (nibble << 4)
}

fn scale(expert: usize) -> u8 {
    [0x38, 0x40][expert] // E4M3 1 and 2; macro-scales remain separate.
}

pub fn expected_bytes() -> Vec<u8> {
    let mut bytes = Vec::with_capacity(EXPERTS * STRIDE);
    for expert in 0..EXPERTS {
        for row in 0..ROWS {
            for _ in 0..COLS / 64 {
                bytes.extend([scale(expert); 4]);
                bytes.extend([code(expert, row); 32]);
            }
        }
    }
    bytes
}

pub fn expected_output(expert: usize, scaled: bool) -> Vec<f32> {
    // Input sums to 104 exactly. E2M1 magnitudes 1 and 2 times E4M3 scales 1 and 2
    // give weights +/-1 and +/-4. All products/reductions are exact binary fractions.
    let magnitude = [104.0, 416.0][expert] * if scaled { MACROS[expert] } else { 1.0 };
    (0..ROWS)
        .map(|row| {
            if row.is_multiple_of(2) {
                magnitude
            } else {
                -magnitude
            }
        })
        .collect()
}

pub struct Evidence {
    pub root: PathBuf,
    pub artifact: PathBuf,
    hashes: Vec<String>,
    source_hashes: [String; 2],
}

impl Evidence {
    pub fn new(layout: Layout) -> Self {
        let out = std::env::var_os("REWRITE_REPACK_OUT")
            .filter(|value| !value.is_empty())
            .expect("REWRITE_REPACK_OUT must name a persistent evidence directory");
        let out = PathBuf::from(out);
        assert!(out.is_absolute(), "REWRITE_REPACK_OUT must be absolute");
        std::fs::create_dir_all(&out).unwrap();
        let root = out.join(layout.name());
        std::fs::create_dir(&root).expect("use a fresh evidence directory for each run");
        let artifact = root.join("artifact");
        std::fs::create_dir(&artifact).unwrap();
        let mut evidence = Self {
            root,
            artifact,
            hashes: Vec::new(),
            source_hashes: Default::default(),
        };
        let config = match layout {
            Layout::Stacked => {
                r#"{
              "model_type":"step3p5","num_hidden_layers":2,"hidden_size":128,
              "intermediate_size":512,"num_attention_heads":1,"num_attention_groups":1,
              "head_dim":128,"vocab_size":64,"max_position_embeddings":2048,
              "moe_num_experts":2,"moe_top_k":1,"moe_intermediate_size":512,
              "share_expert_dim":128,"moe_layers_enum":"1","moe_router_activation":"sigmoid",
              "layer_types":["full_attention","sliding_attention"],
              "rope_theta":[5000000,10000],"partial_rotary_factors":[0.5,1],"sliding_window":512,
              "attention_other_setting":{"num_attention_heads":1,"num_attention_groups":1},
              "swiglu_limits":[0,0],"swiglu_limits_shared":[0,0]
            }"#
            }
            Layout::PerExpert => {
                r#"{
              "model_type":"qwen3_moe","num_hidden_layers":1,"hidden_size":128,
              "num_attention_heads":2,"intermediate_size":512,"vocab_size":64,
              "max_position_embeddings":2048,"num_key_value_heads":2,"head_dim":64,
              "num_experts":2,"num_experts_per_tok":1,"moe_intermediate_size":512
            }"#
            }
        };
        evidence.save("artifact/config.json", config.as_bytes());
        evidence.save("artifact/model.safetensors", &safetensors(layout));
        for (index, name) in ["config.json", "model.safetensors"].iter().enumerate() {
            let path = evidence.artifact.join(name);
            std::fs::set_permissions(&path, Permissions::from_mode(0o400)).unwrap();
            evidence.source_hashes[index] = file_sha256(&path);
        }
        let manifest = format!(
            "{}  config.json\n{}  model.safetensors\n",
            evidence.source_hashes[0], evidence.source_hashes[1]
        );
        evidence.save("artifact/SHA256SUMS", manifest.as_bytes());
        evidence.save("expected.nvfp4", &expected_bytes());
        evidence.save("input.f32le", &f32_bytes(&input()));
        for expert in 0..EXPERTS {
            evidence.save(
                &format!("expected-{expert}-raw.f32le"),
                &f32_bytes(&expected_output(expert, false)),
            );
            evidence.save(
                &format!("expected-{expert}-scaled.f32le"),
                &f32_bytes(&expected_output(expert, true)),
            );
        }
        evidence
    }

    pub fn save(&mut self, name: &str, bytes: &[u8]) {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(self.root.join(name))
            .unwrap();
        file.write_all(bytes).unwrap();
        file.sync_all().unwrap();
        self.hashes.push(format!("{}  {name}\n", sha256(bytes)));
    }

    pub fn verify_source(&self) {
        for (index, name) in ["config.json", "model.safetensors"].iter().enumerate() {
            assert_eq!(
                file_sha256(&self.artifact.join(name)),
                self.source_hashes[index],
                "immutable fixture {name} changed"
            );
        }
    }

    pub fn finish(mut self, receipt: &str) {
        self.verify_source();
        self.save("receipt.txt", receipt.as_bytes());
        let manifest = self.hashes.concat();
        self.save("SHA256SUMS", manifest.as_bytes());
        println!(
            "PASS evidence={} manifest_sha256={}",
            self.root.display(),
            sha256(manifest.as_bytes())
        );
    }
}

fn safetensors(layout: Layout) -> Vec<u8> {
    let mut header = Vec::new();
    let mut data = Vec::new();
    let mut tensor = |name: &str, dtype: &str, shape: &[usize], bytes: &[u8]| {
        let start = data.len();
        data.extend_from_slice(bytes);
        header.push(format!("\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":{shape:?},\"data_offsets\":[{start},{}]}}", data.len()));
    };
    let codes = |expert| {
        (0..ROWS)
            .flat_map(move |row| std::iter::repeat_n(code(expert, row), COLS / 2))
            .collect::<Vec<_>>()
    };
    match layout {
        Layout::Stacked => {
            tensor(
                "model.layers.1.moe.gate_proj.weight",
                "U8",
                &[EXPERTS, ROWS, COLS / 2],
                &(0..EXPERTS).flat_map(codes).collect::<Vec<_>>(),
            );
            tensor(
                "model.layers.1.moe.gate_proj.weight_scale",
                "F8_E4M3",
                &[EXPERTS, ROWS, COLS / 16],
                &(0..EXPERTS)
                    .flat_map(|expert| std::iter::repeat_n(scale(expert), ROWS * COLS / 16))
                    .collect::<Vec<_>>(),
            );
            tensor(
                "model.layers.1.moe.gate_proj.weight_scale_2",
                "F32",
                &[EXPERTS],
                &f32_bytes(&MACROS),
            );
        }
        Layout::PerExpert => {
            for (expert, macro_scale) in MACROS.iter().enumerate() {
                let stem = format!("model.layers.0.mlp.experts.{expert}.gate_proj");
                tensor(
                    &format!("{stem}.weight"),
                    "U8",
                    &[ROWS, COLS / 2],
                    &codes(expert),
                );
                tensor(
                    &format!("{stem}.weight_scale"),
                    "F8_E4M3",
                    &[ROWS, COLS / 16],
                    &vec![scale(expert); ROWS * COLS / 16],
                );
                tensor(
                    &format!("{stem}.weight_scale_2"),
                    "F32",
                    &[1],
                    &macro_scale.to_le_bytes(),
                );
            }
        }
    }
    let mut header = format!("{{{}}}", header.join(",")).into_bytes();
    header.resize(header.len().next_multiple_of(8), b' ');
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.extend(data);
    bytes
}
