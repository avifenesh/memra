//! Inkling 2 chat dialect and tokenizer gates against the pinned artifact
//! (thinkingmachines/Inkling-2-Mini@ed40e7da625f).
//!
//! Oracles (research/modelplan-onboarding-inkling2mini-20261121/chat/gen_fixtures.py):
//! transformers 5.23.0 `render_jinja_template` over the artifact's `chat_template.jinja` for the
//! renders, and the `tokenizers` library over its `tokenizer.json` for the ids. Byte and id
//! parity are the only acceptance.
//!
//! 1. every render fixture through the dispatch the serve path uses, byte for byte, and every
//!    template refusal refused with the template's own message;
//! 2. the ids of every render (`add_special_tokens=False`, as `apply_chat_template` tokenizes);
//! 3. a raw corpus in both encode modes: code, scripts, numbers, emoji, whitespace runs, every
//!    added token, and both ends of every run of one class of the split regex;
//! 4. (ignored by default, minutes in a debug build) every Unicode scalar value, by digest.

use std::path::{Path, PathBuf};

use memra_tokenizer::chat::{
    ContentPart, INKLING2_DEFAULT_EFFORT, ThinkMode, ToolCall, Turn, Val, apply_chat_template_str,
    apply_chat_template_tools_ex, apply_inkling2_template, inkling2_effort, inkling2_effort_number,
    template_is_inkling2,
};
use memra_tokenizer::{PreSplit, Tokenizer};

fn lane() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../research/modelplan-onboarding-inkling2mini-20261121/chat")
}

fn read(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"))
}

/// The fixtures as `Val`: objects keep their key order and numbers their text, both of which
/// the template's output depends on and the crate's own `json` reader drops.
struct Json<'a> {
    s: &'a [u8],
    i: usize,
}

impl Json<'_> {
    fn parse(text: &str) -> Val {
        let mut p = Json {
            s: text.as_bytes(),
            i: 0,
        };
        let v = p.value();
        p.ws();
        assert_eq!(p.i, p.s.len(), "trailing bytes");
        v
    }
    fn ws(&mut self) {
        while self.i < self.s.len() && self.s[self.i].is_ascii_whitespace() {
            self.i += 1;
        }
    }
    fn eat(&mut self, lit: &str) -> bool {
        let hit = self.s[self.i..].starts_with(lit.as_bytes());
        if hit {
            self.i += lit.len();
        }
        hit
    }
    fn value(&mut self) -> Val {
        self.ws();
        let first = self.s[self.i];
        match first {
            b'{' => {
                self.i += 1;
                let mut pairs = Vec::new();
                loop {
                    self.ws();
                    if self.eat("}") {
                        break;
                    }
                    self.eat(",");
                    self.ws();
                    let key = self.string();
                    self.ws();
                    assert!(self.eat(":"));
                    pairs.push((key, self.value()));
                }
                Val::Obj(pairs)
            }
            b'[' => {
                self.i += 1;
                let mut items = Vec::new();
                loop {
                    self.ws();
                    if self.eat("]") {
                        break;
                    }
                    self.eat(",");
                    items.push(self.value());
                }
                Val::Arr(items)
            }
            b'"' => Val::Str(self.string()),
            _ if self.eat("null") => Val::Null,
            _ if self.eat("true") => Val::Bool(true),
            _ if self.eat("false") => Val::Bool(false),
            _ => {
                let from = self.i;
                while self.i < self.s.len() && b"+-.eE0123456789".contains(&self.s[self.i]) {
                    self.i += 1;
                }
                assert!(self.i > from, "unexpected byte at {from}");
                Val::Num(String::from_utf8(self.s[from..self.i].to_vec()).unwrap())
            }
        }
    }
    fn hex4(&mut self) -> u32 {
        let h = std::str::from_utf8(&self.s[self.i..self.i + 4]).unwrap();
        self.i += 4;
        u32::from_str_radix(h, 16).unwrap()
    }
    fn string(&mut self) -> String {
        assert_eq!(self.s[self.i], b'"');
        self.i += 1;
        let mut out: Vec<u8> = Vec::new();
        loop {
            let b = self.s[self.i];
            self.i += 1;
            match b {
                b'"' => break,
                b'\\' => {
                    let e = self.s[self.i];
                    self.i += 1;
                    let c = match e {
                        b'n' => '\n',
                        b'r' => '\r',
                        b't' => '\t',
                        b'b' => '\u{8}',
                        b'f' => '\u{c}',
                        b'u' => {
                            let mut cp = self.hex4();
                            if (0xD800..0xDC00).contains(&cp) {
                                assert!(self.eat("\\u"));
                                cp = 0x10000 + ((cp - 0xD800) << 10) + (self.hex4() - 0xDC00);
                            }
                            char::from_u32(cp).unwrap()
                        }
                        other => other as char,
                    };
                    let mut buf = [0u8; 4];
                    out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
                }
                raw => out.push(raw),
            }
        }
        String::from_utf8(out).unwrap()
    }
}

fn get<'a>(v: &'a Val, key: &str) -> Option<&'a Val> {
    match v {
        Val::Obj(pairs) => pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v),
        _ => None,
    }
}

fn text<'a>(v: &'a Val, key: &str) -> Option<&'a str> {
    match get(v, key) {
        Some(Val::Str(s)) => Some(s),
        _ => None,
    }
}

fn list<'a>(v: &'a Val, key: &str) -> &'a [Val] {
    match get(v, key) {
        Some(Val::Arr(a)) => a,
        _ => &[],
    }
}

/// A fixture message as a `Turn`. `Err` names what the typed request surface cannot carry (the
/// template refuses those from the raw JSON; the HTTP layer refuses them before a `Turn` exists).
fn turn(msg: &Val) -> Result<Turn, &'static str> {
    let mut tool_calls = Vec::new();
    for tc in list(msg, "tool_calls") {
        let function = get(tc, "function").unwrap();
        let args = match get(function, "arguments") {
            None => Vec::new(),
            Some(Val::Obj(pairs)) => pairs.clone(),
            Some(_) => return Err("tool call arguments that are not an object"),
        };
        tool_calls.push(ToolCall {
            name: text(function, "name").unwrap().to_string(),
            args,
            id: text(tc, "id").map(String::from),
            ..Default::default()
        });
    }
    let no_content = matches!(get(msg, "content"), None | Some(Val::Null));
    let (content, parts) = match get(msg, "content") {
        Some(Val::Str(s)) => (s.clone(), None),
        Some(Val::Arr(items)) => {
            let mut parts = Vec::new();
            for (i, part) in items.iter().enumerate() {
                parts.push(match text(part, "type") {
                    None | Some("text" | "input_text") => {
                        ContentPart::Text(text(part, "text").unwrap_or_default().to_string())
                    }
                    Some("image" | "input_image" | "image_url") => ContentPart::Image(i),
                    Some(_) => return Err("a content part that is neither text nor image"),
                });
            }
            (String::new(), Some(parts))
        }
        _ => (String::new(), None),
    };
    Ok(Turn {
        role: text(msg, "role").unwrap().to_string(),
        content,
        no_content,
        parts,
        tool_calls,
        reasoning: text(msg, "reasoning_content").map(String::from),
        tool_call_id: text(msg, "tool_call_id").map(String::from),
        tool_name: text(msg, "name").map(String::from),
        ..Default::default()
    })
}

fn renders() -> Vec<Val> {
    match Json::parse(&read(&lane().join("fixtures/renders.json"))) {
        Val::Arr(cases) => cases,
        _ => panic!("renders.json is an array"),
    }
}

enum Rendered {
    Text(String),
    Refused(String),
    Unrepresentable(&'static str),
}

fn render(template: &str, case: &Val) -> Rendered {
    let mut turns = Vec::new();
    for msg in list(case, "messages") {
        match turn(msg) {
            Ok(turn) => turns.push(turn),
            Err(why) => return Rendered::Unrepresentable(why),
        }
    }
    let tools = list(case, "tools");
    let tools_json: Vec<String> = tools.iter().map(|_| "{}".to_string()).collect();
    let agp = get(case, "add_generation_prompt") == Some(&Val::Bool(true));
    let out = match get(case, "reasoning_effort") {
        // a numeric effort has its own entry: the shared one takes the OpenAI string
        Some(Val::Num(n)) => inkling2_effort_number(n.parse().unwrap())
            .and_then(|effort| apply_inkling2_template(&turns, tools, effort)),
        effort => apply_chat_template_tools_ex(
            Some(template),
            &turns,
            agp,
            &tools_json,
            tools,
            ThinkMode::Default,
            match effort {
                Some(Val::Str(s)) => Some(s.as_str()),
                _ => None,
            },
            None,
        ),
    };
    match out {
        Ok(text) => Rendered::Text(text),
        Err(error) => Rendered::Refused(error),
    }
}

fn template() -> String {
    read(&lane().join("artifact/chat_template.jinja"))
}

fn tokenizer() -> Tokenizer {
    Tokenizer::from_hf_dir(&lane().join("artifact")).unwrap()
}

#[test]
fn every_render_matches_the_artifact_template() {
    let template = template();
    assert!(template_is_inkling2(&template));
    let cases = renders();
    assert!(cases.len() >= 70, "fixture set shrank: {}", cases.len());
    let (mut matched, mut refused, mut images, mut unrepresentable) = (0, 0, 0, 0);
    for case in &cases {
        let name = text(case, "name").unwrap();
        let got = render(&template, case);
        if text(case, "memra") == Some("refuses-image") {
            // the template renders a placeholder; this engine has no vision program
            match got {
                Rendered::Refused(error) => {
                    assert!(error.contains("image parts"), "{name}: {error}")
                }
                _ => panic!("{name}: an image part must be refused"),
            }
            images += 1;
            continue;
        }
        match (text(case, "expected"), got) {
            (Some(expected), Rendered::Text(got)) => {
                if got != expected {
                    let at = got
                        .bytes()
                        .zip(expected.bytes())
                        .position(|(a, b)| a != b)
                        .unwrap_or(got.len().min(expected.len()));
                    let from = (0..=at.saturating_sub(40))
                        .rev()
                        .find(|&i| got.is_char_boundary(i) && expected.is_char_boundary(i))
                        .unwrap();
                    panic!(
                        "{name}: render differs at byte {at}\n  got: {:?}\n want: {:?}",
                        &got[from..],
                        &expected[from..]
                    );
                }
                matched += 1;
            }
            (Some(_), Rendered::Refused(error)) => panic!("{name}: refused: {error}"),
            (Some(_), Rendered::Unrepresentable(why)) => panic!("{name}: {why}"),
            (None, Rendered::Text(got)) => panic!("{name}: the template refuses, got {got:?}"),
            (None, Rendered::Refused(error)) => {
                let vendor = text(case, "error").unwrap();
                // an image in an assistant turn: the template's reason is the role, ours the part
                assert!(
                    vendor.ends_with(&error) || error.contains("image parts"),
                    "{name}: {error:?} is not the template's {vendor:?}"
                );
                refused += 1;
            }
            (None, Rendered::Unrepresentable(_)) => unrepresentable += 1,
        }
    }
    assert_eq!(
        (matched, refused, images, unrepresentable),
        (56, 10, 1, 3),
        "renders, refusals, image refusals, cases the typed surface cannot carry"
    );
}

#[test]
fn every_render_tokenizes_to_the_reference_ids() {
    let tok = tokenizer();
    let mut checked = 0;
    for case in renders() {
        let Some(expected) = text(&case, "expected") else {
            continue;
        };
        let want: Vec<u32> = list(&case, "ids")
            .iter()
            .map(|id| match id {
                Val::Num(n) => n.parse().unwrap(),
                _ => panic!("ids are numbers"),
            })
            .collect();
        assert_eq!(
            tok.encode(expected, false),
            want,
            "{:?}",
            text(&case, "name")
        );
        checked += 1;
    }
    assert_eq!(checked, 57, "every render carries its ids");
}

fn unhex(s: &str) -> String {
    let bytes: Vec<u8> = (0..s.len() / 2)
        .map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap())
        .collect();
    String::from_utf8(bytes).unwrap()
}

#[test]
fn the_raw_corpus_tokenizes_to_the_reference_ids_in_both_modes() {
    let tok = tokenizer();
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
    let mut failed = Vec::new();
    for (line, ref_line) in corpus.lines().zip(reference.lines()) {
        let (name, hexed) = line.split_once('\t').unwrap();
        let cols: Vec<&str> = ref_line.split('\t').collect();
        assert_eq!(cols[0], name);
        let text = unhex(hexed);
        if tok.encode(&text, true) != ids(cols[1]) || tok.encode(&text, false) != ids(cols[2]) {
            failed.push(name);
        }
        cases += 1;
    }
    assert!(
        failed.is_empty(),
        "{} of {cases} differ: {failed:?}",
        failed.len()
    );
    assert_eq!(cases, reference.lines().count());
    assert!(cases >= 999, "corpus shrank: {cases}");
}

/// The contexts of gen_fixtures.py `contexts`, the positions the split regex tells apart.
fn contexts(c: char) -> String {
    format!("x{c}y X{c}Y {c}{c}'S 1{c}2 a'{c}e a'{c}l a'r{c} .{c}/\n{c}\r\n {c}")
}

/// Every Unicode scalar value in those contexts, 4096 scalars a chunk, against the digest of
/// the reference ids. Run with `--ignored` (release: about a minute).
#[test]
#[ignore = "every scalar value: minutes in a debug build"]
fn every_scalar_value_tokenizes_to_the_reference_digest() {
    use sha2::{Digest, Sha256};
    let tok = tokenizer();
    let scalars: Vec<char> = (1..=0x10FFFFu32).filter_map(char::from_u32).collect();
    let digests = read(&lane().join("fixtures/scalar-digests.tsv"));
    let mut failed = Vec::new();
    let mut chunks = 0;
    for (chunk, line) in scalars.chunks(4096).zip(digests.lines()) {
        let cols: Vec<&str> = line.split('\t').collect();
        assert_eq!(u32::from_str_radix(cols[0], 16).unwrap(), chunk[0] as u32);
        let text = chunk
            .iter()
            .map(|&c| contexts(c))
            .collect::<Vec<_>>()
            .join(" ");
        let ids = tok.encode(&text, false);
        let mut hash = Sha256::new();
        for id in &ids {
            hash.update(id.to_le_bytes());
        }
        let hex: String = hash.finalize().iter().map(|b| format!("{b:02x}")).collect();
        if ids.len().to_string() != cols[2] || hex != cols[3] {
            failed.push(cols[0].to_string());
        }
        chunks += 1;
    }
    assert_eq!(chunks, digests.lines().count());
    assert_eq!(chunks, 272);
    assert!(failed.is_empty(), "chunks starting at {failed:?} differ");
}

/// What the artifact's files resolve to at load: the gpt-4o split, the one end-of-generation
/// id, and each marker the dialect writes as a single id.
#[test]
fn the_artifact_loads_as_the_gpt4o_split_with_its_stop_and_markers() {
    let tok = tokenizer();
    assert_eq!(tok.split(), PreSplit::Gpt4o);
    assert_eq!(tok.pre(), "inkling2");
    assert_eq!(tok.eos_id(), 200006);
    assert_eq!(tok.eos_ids(), &[200006]);
    assert_eq!(tok.eog_ids(), vec![200006]);
    for (piece, id) in [
        ("<|endoftext|>", 199999),
        ("<|message_user|>", 200000),
        ("<|message_model|>", 200001),
        ("<|message_system|>", 200002),
        ("<|message_tool|>", 200003),
        ("<|content_text|>", 200004),
        ("<|content_image|>", 200005),
        ("<|content_model_end_sampling|>", 200006),
        ("<|content_thinking|>", 200008),
        ("<|end_message|>", 200010),
        ("<|content_tool_error|>", 200022),
        ("<|content_xml|>", 200024),
        ("<|content_invoke_tool_json|>", 200049),
        ("<|content_invoke_tool_text|>", 200057),
    ] {
        assert_eq!(tok.id_of(piece), Some(id), "{piece}");
        assert_eq!(tok.encode(piece, false), vec![id], "{piece}");
        // a control token: text only when the caller asks for specials
        assert_eq!(tok.decode_special(&[id], true), piece);
        assert_eq!(tok.decode_special(&[id], false), "");
    }
    // the stop id ends a rendered assistant turn and nothing else in a prompt
    let prompt = tok.apply_chat_template(
        &[("user", "Hi"), ("assistant", "Hello."), ("user", "Bye")],
        true,
    );
    let ids = tok.encode(&prompt, false);
    assert_eq!(ids.iter().filter(|&&id| id == 200006).count(), 1);
    assert_eq!(ids.last(), Some(&200010));
}

/// `model.ignore_merges` is load-bearing on this vocabulary: 337 pieces are words the merge
/// table does not build (the corpus's `unmerged-*` cases), and each encodes as its one id only
/// because a whole word is looked up before any merge runs.
#[test]
fn a_word_that_is_one_piece_encodes_without_the_merge_table() {
    let tok = tokenizer();
    assert_eq!(tok.encode("kwargs", false), vec![87550]);
    assert_eq!(tok.encode("Aleksandr", false), vec![49919]);
}

#[test]
fn the_effort_resolves_as_the_template_does() {
    assert_eq!(
        inkling2_effort(ThinkMode::Default, None),
        Ok(INKLING2_DEFAULT_EFFORT)
    );
    assert_eq!(inkling2_effort(ThinkMode::Think, None), Ok(0.9));
    for (name, level) in [
        ("none", 0.0),
        ("minimal", 0.1),
        ("low", 0.2),
        ("medium", 0.7),
        ("high", 0.9),
        ("xhigh", 0.99),
        ("max", 0.99),
        (" medium\t", 0.7),
    ] {
        assert_eq!(
            inkling2_effort(ThinkMode::Default, Some(name)),
            Ok(level),
            "{name}"
        );
    }
    // thinking off is level 0 whatever the request names; a name the template refuses still is
    assert_eq!(inkling2_effort(ThinkMode::NoThink, None), Ok(0.0));
    assert_eq!(inkling2_effort(ThinkMode::NoThink, Some("max")), Ok(0.0));
    assert_eq!(
        inkling2_effort(ThinkMode::NoThink, Some("ultra")),
        Err("Unknown reasoning_effort: ultra".into())
    );
    assert_eq!(
        inkling2_effort(ThinkMode::Default, Some("0.5")),
        Err("Unknown reasoning_effort: 0.5".into())
    );
    assert!(inkling2_effort_number(f64::NAN).is_err());
    assert!(inkling2_effort_number(0.990001).is_err());
    assert_eq!(inkling2_effort_number(0.99), Ok(0.99));
}

/// Thinking off writes level 0 through the shared entry, and the plain entry writes the
/// template's default: the same bytes the tools entry writes for a request with no effort.
#[test]
fn the_shared_entries_dispatch_to_the_dialect() {
    let template = template();
    let turns = [Turn {
        role: "user".into(),
        content: "Hi".into(),
        ..Default::default()
    }];
    let render = |think, effort| {
        apply_chat_template_tools_ex(Some(&template), &turns, true, &[], &[], think, effort, None)
            .unwrap()
    };
    let user = "<|message_user|><|content_text|>Hi<|end_message|>";
    let line = |n: &str| {
        format!("<|message_system|><|content_text|>Thinking effort level: {n}<|end_message|>")
    };
    assert_eq!(
        render(ThinkMode::Default, None),
        format!("{}{user}", line("0.9"))
    );
    assert_eq!(
        render(ThinkMode::NoThink, Some("high")),
        format!("{}{user}", line("0"))
    );
    assert_eq!(
        render(ThinkMode::Think, Some("xhigh")),
        format!("{}{user}", line("0.99"))
    );
    assert_eq!(
        apply_chat_template_str(Some(&template), &[("user", "Hi")], true),
        render(ThinkMode::Default, None)
    );
    // tool definitions as strings alone cannot be re-serialized the way the template does
    let error = apply_chat_template_tools_ex(
        Some(&template),
        &turns,
        true,
        &["{}".into()],
        &[],
        ThinkMode::Default,
        None,
        None,
    )
    .unwrap_err();
    assert!(error.contains("tools_struct"), "{error}");
}

/// A request's JSON may spell a number any way JSON allows; the template re-serializes it
/// through Python, so the prompt carries Python's spelling. Every float of the number fixture
/// is respelled the way Rust (and serde_json) print it, `1e-7` for `1e-07`, and the render
/// must not move.
#[test]
fn tool_json_numbers_render_as_python_prints_them_whatever_their_spelling() {
    fn respell(v: &Val, changed: &mut usize) -> Val {
        match v {
            Val::Num(text) if text.contains(['.', 'e', 'E']) => {
                let other = format!("{:?}", text.parse::<f64>().unwrap());
                if &other != text {
                    *changed += 1;
                }
                Val::Num(other)
            }
            Val::Arr(items) => Val::Arr(items.iter().map(|x| respell(x, changed)).collect()),
            Val::Obj(pairs) => Val::Obj(
                pairs
                    .iter()
                    .map(|(k, x)| (k.clone(), respell(x, changed)))
                    .collect(),
            ),
            other => other.clone(),
        }
    }
    let template = template();
    let cases = renders();
    let case = cases
        .iter()
        .find(|case| text(case, "name") == Some("tool-json-numbers"))
        .unwrap();
    let expected = text(case, "expected").unwrap();
    for needle in [
        "1e-07",
        "1e+16",
        "123456789012345678901234567890",
        "5e-324",
        "-0.0",
    ] {
        assert!(expected.contains(needle), "{needle}");
    }
    let mut changed = 0;
    let respelled = respell(case, &mut changed);
    assert!(changed >= 8, "the respelling moved only {changed} numbers");
    match render(&template, &respelled) {
        Rendered::Text(got) => assert_eq!(got, expected),
        _ => panic!("the number fixture renders"),
    }
}

/// The detector reads two literals no other committed dialect writes.
#[test]
fn the_detector_does_not_capture_other_templates() {
    let hy3 = read(&lane().join("../../step-sku-20260807/templates/hy3.chat_template.jinja"));
    assert!(!template_is_inkling2(&hy3));
    assert!(!template_is_inkling2(
        "<|im_start|>user\n{{ content }}<|im_end|>\n<think>\n"
    ));
}
