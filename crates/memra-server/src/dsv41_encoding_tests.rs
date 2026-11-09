//! deepseek-v4.1 (`encoding_dsv41`) gates: the renderer, the token ids and the parse side against
//! the vendor's own `encoding/encoding.py`.
//!
//! The oracle is DeepSeek-V4.1-Flash-1017's encoding.py at its pinned revision. Its fixtures live
//! in research/modelplan-onboarding-dsv41-20261109/encoding/ (`gen_fixtures.py` runs that file
//! over every branch, the vendor's `encoding/tests` pairs and `test_encoding.py` cases included).
//! Byte parity is the only acceptance. Four gates read them:
//!
//! 1. every render fixture through the renderer entry and the dispatch the tokenizer uses, its
//!    bytes and its image order, and every refusal fixture refused;
//! 2. the token ids of every render under the artifact's own tokenizer.json, rendered through the
//!    `Tokenizer` loaded from the artifact files (the census-detected encoding, no template);
//! 3. the fixtures memra's OpenAI surface can express, through `build_chat_request`;
//! 4. the vendor's parse of each completion, through the serve path's stream parser.

use super::*;
use memra_tokenizer::chat::{ContentPart, Dsv4Encoding, Val};
use std::path::{Path, PathBuf};

const EOS: &str = "<\u{ff5c}end\u{2581}of\u{2581}sentence\u{ff5c}>";

fn base() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../research/modelplan-onboarding-dsv41-20261109/encoding")
}

fn read_json(path: &Path) -> serde_json::Value {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse {path:?}: {e}"))
}

struct Fixture {
    name: String,
    input: serde_json::Value,
    expected: Option<String>,
}

fn fixtures() -> Vec<Fixture> {
    let dir = base().join("fixtures");
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("read {dir:?}: {e}"))
        .map(|e| e.unwrap().path())
        .filter(|p| p.is_dir())
        .collect();
    dirs.sort();
    dirs.into_iter()
        .map(|d| Fixture {
            name: d.file_name().unwrap().to_string_lossy().into_owned(),
            input: read_json(&d.join("input.json")),
            expected: std::fs::read_to_string(d.join("expected.txt")).ok(),
        })
        .collect()
}

/// A tool's `function` object with a tool-level `namespace` folded in: encoding.py reads the tool
/// level first (`tools_from_openai_format`), and a null there is no namespace.
fn tool_function(tool: &serde_json::Value) -> Val {
    let mut function = tool["function"].clone();
    if let Some(ns) = tool.get("namespace").filter(|ns| !ns.is_null()) {
        function["namespace"] = ns.clone();
    }
    json_to_val(&function)
}

fn tool_functions(tools: Option<&serde_json::Value>) -> Vec<Val> {
    tools
        .and_then(|t| t.as_array())
        .map(|a| a.iter().map(tool_function).collect())
        .unwrap_or_default()
}

/// encoding.py `encode_arguments_to_dsml` reads arguments with a tolerance the serve surface does
/// not share (it 400s anything that is not a JSON object): a dict as is, a JSON string decoded up
/// to twice, anything else wrapped as `{"arguments": <the original value>}`.
fn vendor_arguments(raw: &serde_json::Value) -> Vec<(String, Val)> {
    let mut value = raw.clone();
    for _ in 0..2 {
        let serde_json::Value::String(s) = &value else {
            break;
        };
        match serde_json::from_str::<serde_json::Value>(s) {
            Ok(decoded) => value = decoded,
            Err(_) => break,
        }
    }
    match value {
        serde_json::Value::Object(o) => {
            o.iter().map(|(k, v)| (k.clone(), json_to_val(v))).collect()
        }
        _ => vec![("arguments".to_string(), json_to_val(raw))],
    }
}

/// Python truthiness on an optional string field (`tool_call.get("namespace") or ...`).
fn truthy_str(v: Option<&serde_json::Value>) -> Option<String> {
    v.and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .map(String::from)
}

/// One OpenAI-shape message (with the vendor's extra fields) as a renderer turn. Image parts take
/// the next index into `images`, which records each image's url.
fn v41_turn(msg: &serde_json::Value, images: &mut Vec<String>) -> TmplTurn {
    let role = msg["role"].as_str().unwrap().to_string();
    let (content, parts) = match msg.get("content") {
        Some(serde_json::Value::String(s)) => (s.clone(), None),
        Some(serde_json::Value::Array(blocks)) => {
            let parts = blocks
                .iter()
                .map(|b| match b["type"].as_str().unwrap() {
                    "text" => ContentPart::Text(b["text"].as_str().unwrap_or_default().to_string()),
                    "image_url" => {
                        let url = match &b["image_url"] {
                            serde_json::Value::String(u) => u.clone(),
                            other => other["url"].as_str().unwrap().to_string(),
                        };
                        images.push(url);
                        ContentPart::Image(images.len() - 1)
                    }
                    "image" => {
                        let url = ["url", "source", "data"]
                            .iter()
                            .find_map(|k| b.get(*k).and_then(|v| v.as_str()))
                            .unwrap()
                            .to_string();
                        images.push(url);
                        ContentPart::Image(images.len() - 1)
                    }
                    other => panic!("fixture content part type {other:?}"),
                })
                .collect();
            (String::new(), Some(parts))
        }
        _ => (String::new(), None),
    };
    let tool_calls = msg
        .get("tool_calls")
        .and_then(|a| a.as_array())
        .map(|calls| {
            calls
                .iter()
                .map(|tc| {
                    let f = &tc["function"];
                    TmplToolCall {
                        name: f["name"].as_str().unwrap().to_string(),
                        args: vendor_arguments(&f["arguments"]),
                        id: truthy_str(tc.get("id")).or_else(|| truthy_str(f.get("id"))),
                        namespace: truthy_str(tc.get("namespace")).or_else(|| {
                            f.get("namespace")
                                .and_then(|v| v.as_str())
                                .map(String::from)
                        }),
                        ..Default::default()
                    }
                })
                .collect()
        })
        .unwrap_or_default();
    TmplTurn {
        role,
        content,
        parts,
        tool_calls,
        reasoning: msg
            .get("reasoning_content")
            .and_then(|r| r.as_str())
            .filter(|r| !r.is_empty())
            .map(String::from),
        tool_call_id: msg
            .get("tool_call_id")
            .and_then(|s| s.as_str())
            .map(String::from),
        task: msg.get("task").and_then(|s| s.as_str()).map(String::from),
        tools: tool_functions(msg.get("tools")),
        response_format: msg.get("response_format").map(json_to_val),
        wo_eos: msg.get("wo_eos").and_then(|v| v.as_bool()).unwrap_or(false),
        ..Default::default()
    }
}

struct Rendered {
    turns: Vec<TmplTurn>,
    images: Vec<String>,
    req_tools: Vec<Val>,
    think: ThinkMode,
    effort: Option<String>,
}

fn renderer_inputs(input: &serde_json::Value) -> Rendered {
    let mut images = Vec::new();
    let turns = input["messages"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| v41_turn(m, &mut images))
        .collect();
    Rendered {
        turns,
        images,
        req_tools: tool_functions(input.get("req_tools")),
        think: match input["think"].as_str().unwrap() {
            "chat" => ThinkMode::NoThink,
            "thinking" => ThinkMode::Think,
            other => panic!("think {other:?}"),
        },
        effort: input["reasoning_effort"].as_str().map(String::from),
    }
}

/// Gate 1: every vendor render, byte for byte, with its image order; every vendor refusal refused.
/// Thinking fixtures also render through `ThinkMode::Default`, the serve default, which is
/// thinking for this dialect as for encoding_dsv4.
#[test]
fn dsv41_fixtures_match_the_vendor_encoding() {
    let all = fixtures();
    let (mut renders, mut refusals) = (0, 0);
    for f in &all {
        let r = renderer_inputs(&f.input);
        let modes: &[ThinkMode] = if r.think == ThinkMode::Think {
            &[ThinkMode::Think, ThinkMode::Default]
        } else {
            &[ThinkMode::NoThink]
        };
        for &think in modes {
            let got = chat::apply_dsv41_template_media(
                &r.turns,
                true,
                &r.req_tools,
                think,
                r.effort.as_deref(),
            );
            let dispatched = chat::apply_chat_template_tools_ex(
                None,
                &r.turns,
                true,
                &[],
                &r.req_tools,
                think,
                r.effort.as_deref(),
                Some(Dsv4Encoding::V41),
            );
            match &f.expected {
                Some(expected) => {
                    let (prompt, order) = got.unwrap_or_else(|e| {
                        panic!("{}: the vendor renders it, memra: {e}", f.name)
                    });
                    assert_eq!(&prompt, expected, "{} diverged from encoding.py", f.name);
                    let order: Vec<&str> = order.iter().map(|&i| r.images[i].as_str()).collect();
                    let want: Vec<&str> = f.input["expected_images"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_str().unwrap())
                        .collect();
                    assert_eq!(order, want, "{}: image order", f.name);
                    assert_eq!(dispatched.as_ref(), Ok(&prompt), "{}: dispatch", f.name);
                }
                None => {
                    assert!(
                        got.is_err() && dispatched.is_err(),
                        "{}: encoding.py refuses ({}), memra rendered it",
                        f.name,
                        f.input["error"]
                    );
                }
            }
        }
        if f.expected.is_some() {
            renders += 1;
        } else {
            refusals += 1;
        }
    }
    assert!(
        renders >= 160 && refusals >= 15,
        "{renders} renders, {refusals} refusals"
    );
}

/// The vendor's five `encoding/tests` pairs are in the fixture set under their own bytes (the
/// generator asserts it reproduces them), so gate 1 is the vendor's own golden test as well.
#[test]
fn dsv41_vendor_goldens_are_fixtures_verbatim() {
    for n in 1..=5 {
        let golden =
            std::fs::read_to_string(base().join(format!("ref/encoding/tests/test_output_{n}.txt")))
                .unwrap();
        let fixture = std::fs::read_to_string(
            base().join(format!("fixtures/vendor-golden-{n}/expected.txt")),
        )
        .unwrap();
        assert_eq!(fixture, golden, "vendor golden {n}");
    }
}

/// Gate 2: the ids the artifact's tokenizer.json gives each render (banked with transformers'
/// AutoTokenizer, as the vendor's generate.py encodes). Rendered through the `Tokenizer` loaded
/// from the artifact's files, so the census (config.json `model_type`), the template-less
/// dispatch and the deepseek-v3 pre-tokenizer are all under test, and every dialect marker must
/// land on its one id.
#[test]
fn dsv41_token_ids_match_the_artifact_tokenizer() {
    let tok = memra_tokenizer::Tokenizer::from_hf_dir(&base().join("ref"))
        .expect("load the V4.1 tokenizer from ref/");
    assert_eq!(tok.dsv4_encoding(), Some(Dsv4Encoding::V41));
    assert_eq!(tok.pre(), "deepseek-v3");
    assert!(
        tok.chat_template().is_none(),
        "the artifact ships no template"
    );
    for marker in chat::EngineDialect::Dsv41.markers() {
        let ids = tok.encode(marker, false);
        assert_eq!(ids.len(), 1, "{marker} encodes as {ids:?}");
        assert_eq!(Some(ids[0]), tok.id_of(marker), "{marker}");
    }
    let banked = read_json(&base().join("fixtures/tokenization.json"));
    let banked = banked.as_object().unwrap();
    let all = fixtures();
    assert_eq!(
        banked.len(),
        all.iter().filter(|f| f.expected.is_some()).count(),
        "every render fixture has banked ids"
    );
    for f in all.iter().filter(|f| f.expected.is_some()) {
        let r = renderer_inputs(&f.input);
        let rendered = tok
            .apply_chat_template_tools_ex(
                &r.turns,
                true,
                &[],
                &r.req_tools,
                r.think,
                r.effort.as_deref(),
            )
            .unwrap();
        assert_eq!(Some(&rendered), f.expected.as_ref(), "{}", f.name);
        let want: Vec<u32> = banked[&f.name]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        assert_eq!(tok.encode(&rendered, true), want, "{}: token ids", f.name);
    }
}

fn dsv41_caps() -> ModelCaps {
    ModelCaps {
        chat_ok: true,
        dsv4: true,
        dsv41: true,
        tools_branch: true,
        context_length: 1_048_576,
        ..Default::default()
    }
}

fn wire_plan(body: serde_json::Value) -> Result<ChatPlan, String> {
    let req: ChatCompletionReq = serde_json::from_value(body).unwrap();
    let (tx, _rx) = worker::event_channel();
    build_chat_request(req, Some(&dsv41_caps()), tx, lanes::Lane::Interactive, None)
}

fn render_plan(plan: &ChatPlan) -> String {
    chat::apply_chat_template_tools_ex(
        None,
        &plan.request.chat_turns,
        true,
        &plan.request.tools_json,
        &plan.request.tools_struct,
        plan.request.think,
        plan.request.reasoning_effort.as_deref(),
        Some(Dsv4Encoding::V41),
    )
    .unwrap()
}

/// Gate 3: the fixtures the OpenAI chat surface can express (the generator marks them `wire`),
/// from the wire through `build_chat_request`: role mapping, request tools, reasoning switches,
/// text content parts (joined by `"\n\n"`, not concatenated), assistant reasoning and calls.
#[test]
fn dsv41_wire_fixtures_render_through_the_serve_pipeline() {
    let mut n = 0;
    for f in fixtures() {
        if f.input.get("wire").and_then(|w| w.as_bool()) != Some(true) {
            continue;
        }
        let mut body = json!({"model": "dsv41", "messages": f.input["messages"]});
        if let Some(tools) = f.input["req_tools"].as_array() {
            body["tools"] = json!(tools);
        }
        match (
            f.input["think"].as_str(),
            f.input["reasoning_effort"].as_str(),
        ) {
            (Some("chat"), _) => body["reasoning_effort"] = json!("none"),
            (_, Some(level)) => body["reasoning_effort"] = json!(level),
            _ => {}
        }
        let plan = wire_plan(body).unwrap_or_else(|e| panic!("{}: {e}", f.name));
        assert_eq!(
            Some(render_plan(&plan)),
            f.expected,
            "{} diverged from encoding.py through the serve pipeline",
            f.name
        );
        // the dsv4 arm's law: a parser on a think-open prompt or a tools request, none otherwise.
        let armed =
            f.input["think"].as_str() == Some("thinking") || f.input["req_tools"].is_array();
        assert_eq!(plan.parser.is_some(), armed, "{}: parser arming", f.name);
        n += 1;
    }
    assert!(n >= 60, "only {n} wire fixtures");
}

/// The surface refuses what memra cannot serve on V4.1, by name: an image part (no vision
/// program), and tool-call arguments that are not a JSON object (encoding.py would wrap them).
#[test]
fn dsv41_wire_refuses_images_and_non_object_arguments() {
    let err = wire_plan(
        json!({"model": "dsv41", "messages": [{"role": "user", "content": [
        {"type": "text", "text": "what is this?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}]}]}),
    )
    .err()
    .unwrap();
    assert!(
        err.contains("image_url parts are not served on deepseek_v41"),
        "{err}"
    );
    let err = wire_plan(json!({"model": "dsv41", "messages": [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c", "type": "function",
            "function": {"name": "f", "arguments": "not json"}}]}]}))
    .err()
    .unwrap();
    assert!(err.contains("not valid JSON"), "{err}");
}

/// The V4.1 parser, the `</｜DSML｜ calls>` stop and the dsv4 arm stay apart: a V4.1 tools request
/// stops on the spaced close, a DSv4 one on its own, and only V4.1 joins text parts.
#[test]
fn dsv41_caps_select_the_spaced_dsml_stop_and_keep_dsv4_unchanged() {
    let tools = json!([{"type": "function", "function": {"name": "f", "parameters": {}}}]);
    let body = json!({"model": "m", "tools": tools, "messages": [
        {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}]});
    let plan = wire_plan(body.clone()).unwrap();
    assert!(
        plan.request
            .stop_strings
            .contains(&"</\u{ff5c}DSML\u{ff5c} calls>".to_string()),
        "{:?}",
        plan.request.stop_strings
    );
    assert!(render_plan(&plan).contains("<\u{ff5c}User\u{ff5c}>a\n\nb<\u{ff5c}Assistant\u{ff5c}>"));

    let dsv4 = ModelCaps {
        dsv41: false,
        ..dsv41_caps()
    };
    let req: ChatCompletionReq = serde_json::from_value(body).unwrap();
    let (tx, _rx) = worker::event_channel();
    let plan = build_chat_request(req, Some(&dsv4), tx, lanes::Lane::Interactive, None).unwrap();
    assert!(
        plan.request
            .stop_strings
            .contains(&"</\u{ff5c}DSML\u{ff5c}tool_calls>".to_string())
    );
    assert!(
        plan.request.chat_turns[0].parts.is_none(),
        "DSv4 keeps the flattened text"
    );
    assert_eq!(plan.request.chat_turns[0].content, "ab");
}

fn parse_all(
    parser: &mut toolcall::ToolStreamParser,
    chunks: &[&str],
) -> (String, String, Vec<ParsedToolCall>) {
    let mut pieces = Vec::new();
    for c in chunks {
        pieces.extend(parser.push(c));
    }
    pieces.extend(parser.finish());
    let (mut content, mut reasoning, mut calls) = (String::new(), String::new(), Vec::new());
    for p in pieces {
        match p {
            Piece::Content(t) => content.push_str(&t),
            Piece::Reasoning(t) => reasoning.push_str(&t),
            Piece::Call(c) => calls.push(c),
        }
    }
    (content, reasoning, calls)
}

/// Gate 4: the vendor's parse of each completion, through `ToolStreamParser::dsv41` whole and
/// one char at a time. The worker never streams the end-of-sentence id, so it is cut first. Call
/// names keep the `ns::name` spelling the vendor splits into name and namespace; arguments compare
/// as JSON (memra writes them compact). Where the vendor refuses, the house policy holds instead:
/// no call is fabricated and every byte comes back as content or reasoning.
#[test]
fn dsv41_parser_matches_the_vendor_parse() {
    let dir = base().join("parse");
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    files.sort();
    assert!(files.len() >= 15, "{} parse fixtures", files.len());
    for path in files {
        let fx = read_json(&path);
        let completion = fx["completion"].as_str().unwrap();
        let text = completion.strip_suffix(EOS).unwrap_or(completion);
        let thinking = fx["think"].as_str() == Some("thinking");
        let chars: Vec<String> = text.chars().map(String::from).collect();
        let chars: Vec<&str> = chars.iter().map(String::as_str).collect();
        for chunks in [vec![text], chars] {
            let mut parser = ToolStreamParser::dsv41(thinking);
            let (content, reasoning, calls) = parse_all(&mut parser, &chunks);
            if let Some(expected) = fx.get("expected") {
                assert_eq!(content, expected["content"].as_str().unwrap(), "{path:?}");
                assert_eq!(
                    reasoning,
                    expected["reasoning_content"].as_str().unwrap(),
                    "{path:?}"
                );
                let want = expected["tool_calls"].as_array().unwrap();
                assert_eq!(calls.len(), want.len(), "{path:?}");
                for (got, want) in calls.iter().zip(want) {
                    let name = match want["namespace"].as_str() {
                        Some(ns) => format!("{ns}::{}", want["name"].as_str().unwrap()),
                        None => want["name"].as_str().unwrap().to_string(),
                    };
                    assert_eq!(got.name, name, "{path:?}");
                    let args: serde_json::Value = serde_json::from_str(&got.arguments).unwrap();
                    assert_eq!(args, want["arguments"], "{path:?}");
                }
            } else {
                assert!(
                    calls.is_empty(),
                    "{path:?}: a refused completion yielded {calls:?}"
                );
                match text.find("<\u{ff5c}DSML") {
                    // the block surfaces verbatim, tags included (the `\n\n` before an open is
                    // wire syntax and stays stripped, as on the DSv4 arm).
                    Some(i) => assert!(content.ends_with(&text[i..]), "{path:?}: {content:?}"),
                    // a think segment that never closes is reasoning, all of it.
                    None => assert_eq!(
                        (reasoning.as_str(), content.as_str()),
                        (text, ""),
                        "{path:?}"
                    ),
                }
            }
        }
    }
}

/// The agent law the DSv4 arm pins (`an_agent_tool_call_turn_rerenders_to_what_the_model_
/// generated`), on V4.1: the next turn's render is the parked prompt plus exactly what the model
/// generated, reasoning, split and spaced-DSML calls included, then the tool results.
#[test]
fn dsv41_agent_tool_call_turn_rerenders_to_what_the_model_generated() {
    const D: &str = "<\u{ff5c}DSML\u{ff5c}";
    const DE: &str = "</\u{ff5c}DSML\u{ff5c}";
    let tools = json!([{"type": "function", "function": {
        "name": "grep_source",
        "description": "Search the staged source corpus for a pattern.",
        "parameters": {"type": "object",
                       "properties": {"pattern": {"type": "string"}, "limit": {"type": "integer"}},
                       "required": ["pattern"]}}}]);
    let render = |messages: serde_json::Value| -> String {
        render_plan(
            &wire_plan(json!({"model": "dsv41", "messages": messages, "tools": tools})).unwrap(),
        )
    };
    let user = "Find the function that merges tool messages.";
    let turn1 = render(json!([{"role": "user", "content": user}]));
    assert!(turn1.ends_with("<think>"), "{turn1:?}");
    let generated = [
        "The user wants the merge function.</think>Searching.".to_string(),
        String::new(),
        format!("{D} calls>"),
        format!("{D} invoke name=\"grep_source\">"),
        format!("{D} parameter name=\"pattern\" string=\"true\">merge_tool{DE} parameter>"),
        format!("{D} parameter name=\"limit\" string=\"false\">5{DE} parameter>"),
        format!("{DE} invoke>"),
        format!("{D} invoke name=\"grep_source\">"),
        format!("{D} parameter name=\"pattern\" string=\"true\">fn merge{DE} parameter>"),
        format!("{DE} invoke>"),
        format!("{DE} calls>"),
    ]
    .join("\n");
    let mut parser = ToolStreamParser::dsv41(true);
    let (content, reasoning, calls) = parse_all(&mut parser, &[&generated]);
    assert_eq!(calls.len(), 2);
    assert_eq!(reasoning, "The user wants the merge function.");
    assert_eq!(content, "Searching.");
    let mut messages = vec![
        json!({"role": "user", "content": user}),
        json!({"role": "assistant", "content": content, "reasoning_content": reasoning,
               "tool_calls": calls.iter().map(|c| json!({"id": c.id, "type": "function",
                   "function": {"name": c.name, "arguments": c.arguments}})).collect::<Vec<_>>()}),
    ];
    for c in &calls {
        messages.push(
            json!({"role": "tool", "tool_call_id": c.id, "content": "fn merge_tool_messages"}),
        );
    }
    let turn2 = render(serde_json::Value::Array(messages));
    let parked = format!("{turn1}{generated}{EOS}");
    assert!(
        turn2.starts_with(&parked),
        "turn 2 stopped extending the generation:\n  parked: {parked:?}\n  turn 2: {turn2:?}"
    );
    assert!(turn2.len() > parked.len());
}

/// The pause predicate and the tag sets: the V4.1 close ends a tool-call tail, and neither
/// dialect's parser takes the other's tags as a call.
#[test]
fn dsv41_and_dsv4_tags_do_not_cross() {
    assert!(toolcall::tail_ends_with_tool_call(
        "x</\u{ff5c}DSML\u{ff5c} calls>\n"
    ));
    let v4_block = "\n\n<\u{ff5c}DSML\u{ff5c}tool_calls>\n<\u{ff5c}DSML\u{ff5c}invoke name=\"f\">\n\
                    </\u{ff5c}DSML\u{ff5c}invoke>\n</\u{ff5c}DSML\u{ff5c}tool_calls>";
    let v41_block = "\n\n<\u{ff5c}DSML\u{ff5c} calls>\n<\u{ff5c}DSML\u{ff5c} invoke name=\"f\">\n\
                     </\u{ff5c}DSML\u{ff5c} invoke>\n</\u{ff5c}DSML\u{ff5c} calls>";
    let (_, _, calls) = parse_all(&mut ToolStreamParser::dsv41(false), &[v4_block]);
    assert!(calls.is_empty(), "V4.1 parsed a V4 block");
    let (_, _, calls) = parse_all(&mut ToolStreamParser::dsv4(false), &[v41_block]);
    assert!(calls.is_empty(), "V4 parsed a V4.1 block");
    let (content, _, calls) = parse_all(&mut ToolStreamParser::dsv41(false), &[v41_block]);
    assert_eq!((content.as_str(), calls.len()), ("", 1));
    assert_eq!(calls[0].arguments, "{}");
}
