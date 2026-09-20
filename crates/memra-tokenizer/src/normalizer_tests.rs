//! Expected IDs were captured independently with HF tokenizers 0.22.2 before implementation.
use crate::{GGUF_INPUT_PROGRAM_KEY, Tokenizer, json, normalizer::NormalizationProgram};
use memra_gguf::GgufFile;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

struct FixtureDir(PathBuf);

impl FixtureDir {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "memra-nfc-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
}

impl Drop for FixtureDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

// Test-only JSON/GGUF framing. Neither expected IDs nor normalization are implemented here.
fn json_text(value: &json::Value) -> String {
    match value {
        json::Value::Null => "null".into(),
        json::Value::Bool(v) => v.to_string(),
        json::Value::Num(v) => v.to_string(),
        json::Value::Str(v) => {
            use std::fmt::Write;
            let mut result = String::from("\"");
            for c in v.chars() {
                match c {
                    '\\' => result.push_str("\\\\"),
                    '"' => result.push_str("\\\""),
                    c if c < '\u{20}' => write!(&mut result, "\\u{:04x}", c as u32).unwrap(),
                    c => result.push(c),
                }
            }
            result.push('"');
            result
        }
        json::Value::Arr(v) => format!(
            "[{}]",
            v.iter().map(json_text).collect::<Vec<_>>().join(",")
        ),
        json::Value::Obj(v) => {
            let mut entries: Vec<_> = v.iter().collect();
            entries.sort_by_key(|(key, _)| *key);
            format!(
                "{{{}}}",
                entries
                    .into_iter()
                    .map(|(key, value)| format!(
                        "{}:{}",
                        json_text(&json::Value::Str(key.clone())),
                        json_text(value)
                    ))
                    .collect::<Vec<_>>()
                    .join(",")
            )
        }
    }
}

fn string_bytes(text: &str) -> Vec<u8> {
    let mut result = (text.len() as u64).to_le_bytes().to_vec();
    result.extend_from_slice(text.as_bytes());
    result
}

fn string_array(values: &[String]) -> Vec<u8> {
    let mut bytes = 8u32.to_le_bytes().to_vec();
    bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
    for value in values {
        bytes.extend(string_bytes(value));
    }
    bytes
}

fn imported_program(tokenizer: &json::Value) -> json::Value {
    json::Value::Obj(
        [
            ("version".into(), json::Value::Num(1.0)),
            (
                "normalizer".into(),
                tokenizer
                    .get("normalizer")
                    .cloned()
                    .unwrap_or(json::Value::Null),
            ),
            (
                "added_tokens".into(),
                tokenizer.get("added_tokens").cloned().unwrap(),
            ),
        ]
        .into_iter()
        .collect(),
    )
}

fn gguf(tokenizer: &json::Value, pre: &str, program: Option<&json::Value>) -> Vec<u8> {
    let model = tokenizer.get("model").unwrap();
    let vocab = model.get("vocab").unwrap().as_obj().unwrap();
    let added = tokenizer.get("added_tokens").unwrap().as_arr().unwrap();
    let n = vocab
        .values()
        .chain(added.iter().map(|token| token.get("id").unwrap()))
        .map(|id| id.as_u64().unwrap() as usize)
        .max()
        .unwrap()
        + 1;
    let mut pieces = vec![String::new(); n];
    let mut types = vec![1i32; n];
    for (piece, id) in vocab {
        pieces[id.as_u64().unwrap() as usize] = piece.clone();
    }
    for token in added {
        let id = token.get("id").unwrap().as_u64().unwrap() as usize;
        pieces[id] = token.get("content").unwrap().as_str().unwrap().to_owned();
        types[id] = if token.get("special").unwrap().as_bool().unwrap() {
            3
        } else {
            4
        };
    }
    let merges: Vec<_> = model
        .get("merges")
        .unwrap()
        .as_arr()
        .unwrap()
        .iter()
        .map(|pair| {
            let pair = pair.as_arr().unwrap();
            format!(
                "{} {}",
                pair[0].as_str().unwrap(),
                pair[1].as_str().unwrap()
            )
        })
        .collect();
    let mut type_bytes = 5u32.to_le_bytes().to_vec();
    type_bytes.extend_from_slice(&(n as u64).to_le_bytes());
    for kind in types {
        type_bytes.extend_from_slice(&kind.to_le_bytes());
    }
    let mut metadata = vec![
        ("tokenizer.ggml.model", 8u32, string_bytes("gpt2")),
        ("tokenizer.ggml.pre", 8, string_bytes(pre)),
        ("tokenizer.ggml.tokens", 9, string_array(&pieces)),
        ("tokenizer.ggml.token_type", 9, type_bytes),
        ("tokenizer.ggml.merges", 9, string_array(&merges)),
        (
            "tokenizer.ggml.eos_token_id",
            4,
            519u32.to_le_bytes().to_vec(),
        ),
        ("tokenizer.ggml.add_bos_token", 7, vec![0]),
    ];
    if let Some(program) = program {
        metadata.push((GGUF_INPUT_PROGRAM_KEY, 8, string_bytes(&json_text(program))));
    }
    let mut bytes = b"GGUF".to_vec();
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
    for (key, kind, value) in metadata {
        bytes.extend(string_bytes(key));
        bytes.extend_from_slice(&kind.to_le_bytes());
        bytes.extend(value);
    }
    bytes.resize(bytes.len().div_ceil(32) * 32, 0);
    bytes
}

fn load_gguf(bytes: &[u8]) -> Result<Tokenizer, String> {
    let dir = FixtureDir::new();
    let path = dir.0.join("tokenizer.gguf");
    std::fs::write(&path, bytes).unwrap();
    let file = GgufFile::open(&path).unwrap();
    Tokenizer::from_gguf(&file)
}

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/nfc")
}

fn data(name: &str) -> json::Value {
    json::parse(&std::fs::read_to_string(root().join(name)).unwrap()).unwrap()
}

fn ids(value: &json::Value) -> Vec<u32> {
    value
        .as_arr()
        .unwrap()
        .iter()
        .map(|id| u32::try_from(id.as_u64().unwrap()).unwrap())
        .collect()
}

#[test]
fn whitespace_consumed_added_tokens_match_independent_oracle() {
    let fixture = data("whitespace-oracle.json");
    let mut cases = 0;
    for variant in fixture.get("variants").unwrap().as_arr().unwrap() {
        let name = variant.get("name").unwrap().as_str().unwrap();
        let mut source = data("tiny-base-tokenizer.json");
        let json::Value::Obj(ref mut fields) = source else {
            unreachable!()
        };
        for field in ["normalizer", "added_tokens"] {
            fields.insert(field.into(), variant.get(field).unwrap().clone());
        }
        if let Some(additions) = variant.get("vocab_additions") {
            let json::Value::Obj(model) = fields.get_mut("model").unwrap() else {
                unreachable!()
            };
            let json::Value::Obj(vocab) = model.get_mut("vocab").unwrap() else {
                unreachable!()
            };
            vocab.extend(additions.as_obj().unwrap().clone());
        }
        let dir = FixtureDir::new();
        std::fs::write(dir.0.join("tokenizer.json"), json_text(&source)).unwrap();
        std::fs::write(
            dir.0.join("generation_config.json"),
            r#"{"eos_token_id":519}"#,
        )
        .unwrap();
        let hf = Tokenizer::from_hf_dir(&dir.0).unwrap();
        let imported =
            load_gguf(&gguf(&source, "qwen2", Some(&imported_program(&source)))).unwrap();
        for case in variant.get("cases").unwrap().as_arr().unwrap() {
            cases += 1;
            let text = case.get("text").unwrap().as_str().unwrap();
            for expectation in case.get("expected").unwrap().as_arr().unwrap() {
                let parse = expectation.get("parse_special").unwrap().as_bool().unwrap();
                let add = expectation.get("add_special").unwrap().as_bool().unwrap();
                let expected = ids(expectation.get("ids").unwrap());
                for (loader, tokenizer) in [("HF", &hf), ("GGUF", &imported)] {
                    assert_eq!(
                        tokenizer.encode_special(text, add, parse),
                        expected,
                        "{name} {loader}: {text:?}, parse={parse}, add={add}"
                    );
                }
            }
        }
    }
    assert_eq!(cases, 48);
}

#[test]
fn inverted_whitespace_spans_do_not_reemit_consumed_tokens() {
    // HF 0.22.2 panics on these inverted spans, so this is a native robustness
    // contract, separate from the oracle-defined parity cases above. <X>'s rstrip
    // already consumed the entire suffix; later adjusted tab spans emit no IDs.
    let fixture = data("whitespace-oracle.json");
    let mut variants = 0;
    for variant in fixture.get("variants").unwrap().as_arr().unwrap() {
        let name = variant.get("name").unwrap().as_str().unwrap();
        if !name.starts_with("consumed-tab-") {
            continue;
        }
        variants += 1;
        let mut source = data("tiny-base-tokenizer.json");
        let json::Value::Obj(ref mut fields) = source else {
            unreachable!()
        };
        for field in ["normalizer", "added_tokens"] {
            fields.insert(field.into(), variant.get(field).unwrap().clone());
        }
        let dir = FixtureDir::new();
        std::fs::write(dir.0.join("tokenizer.json"), json_text(&source)).unwrap();
        std::fs::write(
            dir.0.join("generation_config.json"),
            r#"{"eos_token_id":519}"#,
        )
        .unwrap();
        let hf = Tokenizer::from_hf_dir(&dir.0).unwrap();
        let imported =
            load_gguf(&gguf(&source, "qwen2", Some(&imported_program(&source)))).unwrap();
        for text in ["<X>\t\t", "<X>\t \t", "<X>\t\t\t"] {
            for add in [false, true] {
                for parse in [false, true] {
                    for (loader, tokenizer) in [("HF", &hf), ("GGUF", &imported)] {
                        assert_eq!(
                            tokenizer.encode_special(text, add, parse),
                            vec![520],
                            "{name} {loader}: {text:?}, add={add}, parse={parse}"
                        );
                    }
                }
            }
        }
    }
    assert_eq!(variants, 4);
}

#[test]
fn staged_nfc_and_added_tokens_match_independent_oracle() {
    let fixture = data("fixtures.json");
    let variants = fixture.get("variants").unwrap().as_arr().unwrap();
    assert_eq!(variants.len(), 31);
    let mut count = 0;
    for variant in variants {
        let name = variant.get("name").unwrap().as_str().unwrap();
        let path = root().join(variant.get("tokenizer_json").unwrap().as_str().unwrap());
        let tokenizer = Tokenizer::from_hf_dir(path.parent().unwrap())
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let source = json::parse(&std::fs::read_to_string(path).unwrap()).unwrap();
        let pre = match name {
            "qwen35-identity-control" => "qwen35",
            "glm-identity-control" => "glm4",
            _ => "qwen2",
        };
        let imported = load_gguf(&gguf(&source, pre, Some(&imported_program(&source))))
            .unwrap_or_else(|error| panic!("GGUF {name}: {error}"));
        for case in variant.get("cases").unwrap().as_arr().unwrap() {
            count += 1;
            let text = case.get("raw").unwrap().as_str().unwrap();
            for add_special in [false, true] {
                let expected = case
                    .get("expected")
                    .unwrap()
                    .get(if add_special { "true" } else { "false" })
                    .unwrap()
                    .get("ids")
                    .unwrap();
                assert_eq!(
                    tokenizer.encode(text, add_special),
                    ids(expected),
                    "variant={name}, text={text:?}, add_special={add_special}"
                );
                assert_eq!(
                    imported.encode(text, add_special),
                    ids(expected),
                    "GGUF variant={name}, text={text:?}, add_special={add_special}"
                );
            }
        }
    }
    assert_eq!(count, 315);
}

#[test]
fn gguf_requires_an_explicit_consistent_input_program() {
    let source = data("variants/nfc-unicode/tokenizer.json");
    let plain = load_gguf(&gguf(&source, "qwen2", None)).unwrap();
    assert_eq!(
        plain.normalization_program(),
        &NormalizationProgram::Identity
    );
    assert_eq!(plain.encode("e\u{301}", false), vec![101, 274]);
    let program = imported_program(&source);
    let imported = load_gguf(&gguf(&source, "qwen2", Some(&program))).unwrap();
    assert_eq!(imported.encode("e\u{301}", false), vec![195, 169]);

    let json::Value::Obj(mut bad) = program.clone() else {
        unreachable!()
    };
    bad.insert("version".into(), json::Value::Num(2.0));
    assert!(load_gguf(&gguf(&source, "qwen2", Some(&json::Value::Obj(bad)))).is_err());

    let json::Value::Obj(mut bad) = program.clone() else {
        unreachable!()
    };
    bad.insert(
        "normalizer".into(),
        json::parse(r#"{"type":"Lowercase"}"#).unwrap(),
    );
    assert!(load_gguf(&gguf(&source, "qwen2", Some(&json::Value::Obj(bad)))).is_err());

    let json::Value::Obj(mut bad) = program.clone() else {
        unreachable!()
    };
    bad.insert("added_tokens".into(), json::Value::Arr(vec![]));
    assert!(load_gguf(&gguf(&source, "qwen2", Some(&json::Value::Obj(bad)))).is_err());

    for (field, value) in [
        ("id", json::Value::Num(0.0)),
        ("content", json::Value::Str("mismatched content".into())),
        ("special", json::Value::Bool(false)),
    ] {
        let mut bad = program.clone();
        let json::Value::Obj(ref mut obj) = bad else {
            unreachable!()
        };
        let json::Value::Arr(tokens) = obj.get_mut("added_tokens").unwrap() else {
            unreachable!()
        };
        let json::Value::Obj(token) = &mut tokens[0] else {
            unreachable!()
        };
        token.insert(field.into(), value);
        assert!(
            load_gguf(&gguf(&source, "qwen2", Some(&bad))).is_err(),
            "{field}"
        );
    }
}

#[test]
fn special_recognition_and_postprocessing_are_independent() {
    let fixture = data("special-mode-matrix.json");
    let mut count = 0;
    for variant in fixture.get("variants").unwrap().as_arr().unwrap() {
        // The ordinary offline suite uses the small self-contained fixture. The pinned
        // real-vocabulary counterpart is also run by the artifact qualification command.
        let path = variant.get("tokenizer_json").unwrap().as_str().unwrap();
        if path.starts_with("pinned/") {
            continue;
        }
        let path = root().join(path);
        let tokenizer = Tokenizer::from_hf_dir(path.parent().unwrap()).unwrap();
        for case in variant.get("cases").unwrap().as_arr().unwrap() {
            count += 1;
            let text = case.get("raw").unwrap().as_str().unwrap();
            for parse_special in [false, true] {
                for add_special in [false, true] {
                    let expected = case
                        .get("expected_by_parse_special")
                        .unwrap()
                        .get(if parse_special { "true" } else { "false" })
                        .unwrap()
                        .get(if add_special { "true" } else { "false" })
                        .unwrap()
                        .get("ids")
                        .unwrap();
                    assert_eq!(
                        tokenizer.encode_special(text, add_special, parse_special),
                        ids(expected),
                        "{text:?}, parse_special={parse_special}, add_special={add_special}"
                    );
                }
            }
        }
    }
    assert_eq!(
        count, 11,
        "the four pinned-real cases are qualified separately"
    );
}

#[test]
fn declared_programs_are_preserved_and_unsupported_imports_refuse() {
    let fixture = data("normalizer-import-contract.json");
    let cases = fixture.get("cases").unwrap().as_arr().unwrap();
    assert_eq!(cases.len(), 22);
    for case in cases {
        let name = case.get("name").unwrap().as_str().unwrap();
        let result = Tokenizer::from_hf_dir(&root().join("imports").join(name));
        match name {
            "missing" | "null" => assert_eq!(
                result.unwrap().normalization_program(),
                &NormalizationProgram::Identity
            ),
            "nfc" => assert_eq!(
                result.unwrap().normalization_program(),
                &NormalizationProgram::Nfc
            ),
            "sequence-nfc" => assert_eq!(
                result.unwrap().normalization_program(),
                &NormalizationProgram::Sequence(vec![NormalizationProgram::Nfc])
            ),
            "sequence-empty" => assert_eq!(
                result.unwrap().normalization_program(),
                &NormalizationProgram::Sequence(vec![])
            ),
            _ => {
                let error = match result {
                    Ok(_) => panic!("unsupported import {name} was accepted"),
                    Err(error) => error,
                };
                assert!(error.contains("normalizer"), "{name}: {error}");
            }
        }
    }
}
