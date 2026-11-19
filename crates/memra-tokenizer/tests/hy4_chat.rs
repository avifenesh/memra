//! Hy4-flash (`hy_v4_flash`) chat dialect and tokenizer gates against the pinned artifact.
//!
//! Oracles (research/modelplan-onboarding-hy4flash-20261119/chat/gen_fixtures.py):
//! transformers 5.23.0 `render_jinja_template` over the artifact's `chat_template.jinja` for the
//! renders, and the `tokenizers` library over its `tokenizer.json` for the ids. Byte and id
//! parity are the only acceptance.
//!
//! 1. every render fixture through `apply_chat_template_tools_ex` with the artifact template
//!    (the dispatch the serve path uses), byte for byte; every template refusal refused;
//! 2. the ids of every render under the artifact tokenizer (`add_special_tokens=False`, as
//!    `apply_chat_template` tokenizes);
//! 3. a raw corpus (the deepseek-v3 split's adversarial cases, the Hy4 control tokens, every
//!    render) in both encode modes.

use std::path::{Path, PathBuf};

use memra_tokenizer::Tokenizer;
use memra_tokenizer::chat::{
    ThinkMode, ToolCall, Turn, Val, apply_chat_template_tools_ex, template_is_hy4,
};
use serde_json::Value;

fn lane() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../research/modelplan-onboarding-hy4flash-20261119/chat")
}

fn read(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"))
}

fn val(v: &Value) -> Val {
    match v {
        Value::Null => Val::Null,
        Value::Bool(b) => Val::Bool(*b),
        Value::Number(n) => Val::Num(n.to_string()),
        Value::String(s) => Val::Str(s.clone()),
        Value::Array(a) => Val::Arr(a.iter().map(val).collect()),
        Value::Object(o) => Val::Obj(o.iter().map(|(k, v)| (k.clone(), val(v))).collect()),
    }
}

/// Python truthiness on an optional string field, as the template tests it.
fn nonempty(v: Option<&Value>) -> Option<String> {
    v.and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(String::from)
}

fn turn(msg: &Value) -> Turn {
    let tool_calls = msg
        .get("tool_calls")
        .and_then(Value::as_array)
        .map(|calls| {
            calls
                .iter()
                .map(|tc| {
                    let f = &tc["function"];
                    ToolCall {
                        name: f["name"].as_str().unwrap().to_string(),
                        args: f["arguments"]
                            .as_object()
                            .unwrap()
                            .iter()
                            .map(|(k, v)| (k.clone(), val(v)))
                            .collect(),
                        id: nonempty(tc.get("id")).or_else(|| nonempty(tc.get("tool_call_id"))),
                        ..Default::default()
                    }
                })
                .collect()
        })
        .unwrap_or_default();
    Turn {
        role: msg["role"].as_str().unwrap().to_string(),
        content: msg["content"].as_str().unwrap_or_default().to_string(),
        tool_calls,
        // `reasoning` takes precedence; `reasoning_content` is the fallback when it is empty.
        reasoning: nonempty(msg.get("reasoning"))
            .or_else(|| nonempty(msg.get("reasoning_content"))),
        tool_call_id: msg
            .get("tool_call_id")
            .and_then(Value::as_str)
            .map(String::from),
        ..Default::default()
    }
}

fn renders() -> Vec<Value> {
    let text = read(&lane().join("fixtures/renders.json"));
    serde_json::from_str::<Value>(&text)
        .unwrap()
        .as_array()
        .unwrap()
        .clone()
}

fn render(template: &str, case: &Value) -> Result<String, String> {
    let turns: Vec<Turn> = case["messages"]
        .as_array()
        .unwrap()
        .iter()
        .map(turn)
        .collect();
    let tools: Vec<Val> = case["tools"]
        .as_array()
        .map(|t| t.iter().map(val).collect())
        .unwrap_or_default();
    let tools_json: Vec<String> = case["tools"]
        .as_array()
        .map(|t| t.iter().map(|tool| tool.to_string()).collect())
        .unwrap_or_default();
    apply_chat_template_tools_ex(
        Some(template),
        &turns,
        case["add_generation_prompt"].as_bool().unwrap(),
        &tools_json,
        &tools,
        ThinkMode::Default,
        case["reasoning_effort"].as_str(),
        None,
    )
}

fn tokenizer() -> Tokenizer {
    Tokenizer::from_hf_dir(&lane().join("artifact")).unwrap()
}

#[test]
fn every_render_matches_the_artifact_template() {
    let template = read(&lane().join("artifact/chat_template.jinja"));
    assert!(template_is_hy4(&template));
    let cases = renders();
    assert!(cases.len() >= 24, "fixture set shrank: {}", cases.len());
    let mut refused = 0;
    for case in &cases {
        let name = case["name"].as_str().unwrap();
        let got = render(&template, case);
        match case.get("expected").and_then(Value::as_str) {
            Some(expected) => {
                let got = got.unwrap_or_else(|e| panic!("{name}: refused: {e}"));
                if got != expected {
                    let at = got
                        .bytes()
                        .zip(expected.bytes())
                        .position(|(a, b)| a != b)
                        .unwrap_or(got.len().min(expected.len()));
                    panic!(
                        "{name}: render differs at byte {at}\n  got: {:?}\n want: {:?}",
                        &got[at.saturating_sub(40)..(at + 80).min(got.len())],
                        &expected[at.saturating_sub(40)..(at + 80).min(expected.len())]
                    );
                }
            }
            None => {
                let error = got.expect_err(name);
                assert!(
                    error.contains("should be no_think/high/max"),
                    "{name}: {error}"
                );
                refused += 1;
            }
        }
    }
    assert_eq!(refused, 2, "the two invalid-effort fixtures");
}

#[test]
fn every_render_tokenizes_to_the_reference_ids() {
    let tok = tokenizer();
    let mut checked = 0;
    for case in renders() {
        let Some(text) = case.get("expected").and_then(Value::as_str) else {
            continue;
        };
        checked += 1;
        let want: Vec<u32> = case["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|id| id.as_u64().unwrap() as u32)
            .collect();
        assert_eq!(tok.encode(text, false), want, "{}", case["name"]);
    }
    assert_eq!(checked, 22, "every render carries its ids");
}

#[test]
fn the_raw_corpus_tokenizes_to_the_reference_ids_in_both_modes() {
    let tok = tokenizer();
    let unhex = |s: &str| -> String {
        let bytes: Vec<u8> = (0..s.len() / 2)
            .map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap())
            .collect();
        String::from_utf8(bytes).unwrap()
    };
    let ids = |s: &str| -> Vec<u32> {
        if s.is_empty() {
            Vec::new()
        } else {
            s.split(',').map(|x| x.parse().unwrap()).collect()
        }
    };
    let corpus = read(&lane().join("corpus.tsv"));
    let reference = read(&lane().join("ref-ids.tsv"));
    let mut cases = 0;
    for (line, ref_line) in corpus.lines().zip(reference.lines()) {
        let (name, hexed) = line.split_once('\t').unwrap();
        let cols: Vec<&str> = ref_line.split('\t').collect();
        assert_eq!(cols[0], name);
        let text = unhex(hexed);
        assert_eq!(
            tok.encode(&text, true),
            ids(cols[1]),
            "{name} add_special=true"
        );
        assert_eq!(
            tok.encode(&text, false),
            ids(cols[2]),
            "{name} add_special=false"
        );
        cases += 1;
    }
    assert_eq!(cases, corpus.lines().count());
    assert!(cases >= 130, "corpus shrank: {cases}");
}

/// The Hy4 detection does not capture the real Hy3 template (which shares the
/// `<｜reasoning_mode{}｜>` format string), and the Hy3 marker does not occur in the Hy4 one: the
/// two dialects never dispatch to each other's renderer.
#[test]
fn hy3_and_hy4_templates_stay_apart() {
    let hy4 = read(&lane().join("artifact/chat_template.jinja"));
    assert!(!hy4.contains("hy_User"));
    let hy3 = read(&lane().join("../../step-sku-20260807/templates/hy3.chat_template.jinja"));
    assert!(
        hy3.contains("<\u{ff5c}reasoning_mode{}\u{ff5c}>"),
        "the shared marker"
    );
    assert!(!template_is_hy4(&hy3));
}

/// A thinking-off request renders `no_think` whatever effort it names; an explicit effort wins
/// otherwise; the template's `high` is the default; the efforts it raises on are refused.
#[test]
fn the_effort_ladder_resolves_as_the_template_does() {
    use memra_tokenizer::chat::hy4_reasoning_effort;
    assert_eq!(
        hy4_reasoning_effort(ThinkMode::NoThink, Some("max")),
        Ok("no_think")
    );
    assert_eq!(
        hy4_reasoning_effort(ThinkMode::NoThink, None),
        Ok("no_think")
    );
    assert_eq!(
        hy4_reasoning_effort(ThinkMode::Think, Some("max")),
        Ok("max")
    );
    assert_eq!(
        hy4_reasoning_effort(ThinkMode::Default, Some("no_think")),
        Ok("no_think")
    );
    assert_eq!(hy4_reasoning_effort(ThinkMode::Default, None), Ok("high"));
    assert!(hy4_reasoning_effort(ThinkMode::NoThink, Some("low")).is_err());
}

/// The template writes a tool definition's keys raw between quotes; a key with a quote in it
/// is not escaped (the GLM macro, which escapes, is a different program).
#[test]
fn tool_definition_keys_are_written_raw() {
    let template = read(&lane().join("artifact/chat_template.jinja"));
    let tool = Val::Obj(vec![
        ("name".into(), Val::Str("f".into())),
        ("x\"y".into(), Val::Num("1".into())),
    ]);
    let turns = [Turn {
        role: "user".into(),
        content: "q".into(),
        ..Default::default()
    }];
    let out = apply_chat_template_tools_ex(
        Some(&template),
        &turns,
        true,
        &["{}".into()],
        &[tool],
        ThinkMode::Default,
        None,
        None,
    )
    .unwrap();
    assert!(out.contains(r#"{"name": "f", "x"y": 1}"#), "{out}");
}
