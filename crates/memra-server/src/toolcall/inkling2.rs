//! Inkling 2 (`thinkingmachines/Inkling-2-Mini`) response parser: the `response_template` of the
//! artifact's `tokenizer_config.json`, as a machine over marker events.
//!
//! The model writes its turn as messages,
//! `<|message_model|>{name}{content marker}{body}<|end_message|>`, and ends it with
//! `<|content_model_end_sampling|>` (the checkpoint's one end-of-generation id). A message's
//! content marker says what its body is: thinking is `reasoning`, text (and the tool-error and
//! tool-text kinds) is content, `<|content_invoke_tool_json|>` is one call,
//! `{"name": .., "args": {..}}`. Text outside a message is not output. Inside a body only the
//! end marker means anything: every other marker is body text, as in the template, which
//! watches for nothing but a region's close while the region is open.
//!
//! The markers are control tokens, one id each, and the machine consumes them as events. There
//! are two front ends:
//!
//! - [`Machine::push_token`], keyed by token id ([`Inkling2Markers`]). A marker is a marker
//!   only when its own id arrives. Text is never searched, so a body that spells
//!   `<|end_message|>` and a call marker out in ordinary tokens stays text and cannot produce
//!   a call. This is the serve path's entry.
//! - [`Machine::push_text`], which finds the markers by spelling in rendered text. The vendor
//!   parser (transformers `parse_response`) reads text, so this is the entry its fixtures gate
//!   (research/modelplan-onboarding-inkling2mini-20261121/chat/fixtures/parses.json). It has
//!   the vendor's blind spot: it cannot tell a control token from its spelling.
//!
//! Both feed the same machine, so they agree on every stream in which no text spells a marker.
//!
//! Arguments keep the model's numbers: the call body is read by a JSON reader that holds a
//! number as its literal, so an integer of any size stays exact and `1E400` stays `1E400`.
//!
//! No route serves the dialect yet: the serve step wires [`ToolStreamParser::inkling2`],
//! [`ToolStreamParser::push_token`] and [`inkling2_schema_types`], and the `dead_code` allows
//! here go with it. Until then the fixture gates below are their only callers.
#![allow(dead_code)]

use std::collections::HashMap;

use super::{ParsedToolCall, Piece, emit_content, fnv1a64, partial_suffix_len};

const MODEL: &str = "<|message_model|>";
const END: &str = "<|end_message|>";
const CALL: &str = "<|content_invoke_tool_json|>";
const THINKING: &str = "<|content_thinking|>";
const TEXT: &str = "<|content_text|>";
const TOOL_ERROR: &str = "<|content_tool_error|>";
const TOOL_TEXT: &str = "<|content_invoke_tool_text|>";

/// Bound on the header (the text between `<|message_model|>` and the content marker) kept
/// while waiting for that marker. A header is only ever output as part of a malformed call
/// surfaced verbatim, and a real one is a tool name; one past the bound is not kept.
const HEADER_CAP: usize = 4096;
/// Nesting bound of the call-body reader, which recurses over model output.
const MAX_DEPTH: usize = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Body {
    Thinking,
    Text,
    Call,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Marker {
    Model,
    End,
    Open(Body, &'static str),
}

impl Marker {
    fn spelling(self) -> &'static str {
        match self {
            Marker::Model => MODEL,
            Marker::End => END,
            Marker::Open(_, spelling) => spelling,
        }
    }
}

/// The markers the text front end looks for outside a message.
const OUTSIDE: [(&str, Marker); 6] = [
    (MODEL, Marker::Model),
    (THINKING, Marker::Open(Body::Thinking, THINKING)),
    (TEXT, Marker::Open(Body::Text, TEXT)),
    (TOOL_ERROR, Marker::Open(Body::Text, TOOL_ERROR)),
    (TOOL_TEXT, Marker::Open(Body::Text, TOOL_TEXT)),
    (CALL, Marker::Open(Body::Call, CALL)),
];
/// Inside a body, only its close.
const INSIDE: [(&str, Marker); 1] = [(END, Marker::End)];

/// The token ids of the framing markers in one checkpoint's vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Inkling2Markers {
    model: u32,
    end: u32,
    thinking: u32,
    text: u32,
    tool_error: u32,
    tool_text: u32,
    call: u32,
}

impl Inkling2Markers {
    /// Look every marker up by its piece (`Tokenizer::id_of`). A vocabulary missing one is not
    /// this dialect's.
    pub fn resolve(id_of: impl Fn(&str) -> Option<u32>) -> Result<Self, String> {
        let id = |piece: &str| {
            id_of(piece).ok_or_else(|| format!("inkling2: the vocabulary has no {piece} token"))
        };
        Ok(Self {
            model: id(MODEL)?,
            end: id(END)?,
            thinking: id(THINKING)?,
            text: id(TEXT)?,
            tool_error: id(TOOL_ERROR)?,
            tool_text: id(TOOL_TEXT)?,
            call: id(CALL)?,
        })
    }

    fn marker(&self, id: u32) -> Option<Marker> {
        Some(match id {
            x if x == self.model => Marker::Model,
            x if x == self.end => Marker::End,
            x if x == self.thinking => Marker::Open(Body::Thinking, THINKING),
            x if x == self.text => Marker::Open(Body::Text, TEXT),
            x if x == self.tool_error => Marker::Open(Body::Text, TOOL_ERROR),
            x if x == self.tool_text => Marker::Open(Body::Text, TOOL_TEXT),
            x if x == self.call => Marker::Open(Body::Call, CALL),
            _ => return None,
        })
    }
}

pub(super) struct Machine {
    /// None = outside a message.
    body: Option<Body>,
    /// Outside a message, after `<|message_model|>`: the header text so far. None when no
    /// header is open, or the open one took a `<` or passed the cap.
    header: Option<String>,
    /// The call body being collected.
    call: String,
    /// Text front end only: rendered text not yet split into events (a marker cut short).
    pending: String,
    types: HashMap<String, HashMap<String, Vec<String>>>,
    n_calls: usize,
}

impl Machine {
    pub(super) fn new(types: HashMap<String, HashMap<String, Vec<String>>>) -> Self {
        Self {
            body: None,
            header: None,
            call: String::new(),
            pending: String::new(),
            types,
            n_calls: 0,
        }
    }

    pub(super) fn fresh(&self) -> Self {
        Self::new(self.types.clone())
    }

    pub(super) fn n_calls(&self) -> usize {
        self.n_calls
    }

    /// One generated token: its id, and its text with control tokens rendered.
    pub(super) fn push_token(
        &mut self,
        markers: &Inkling2Markers,
        id: u32,
        text: &str,
    ) -> Vec<Piece> {
        let mut out = Vec::new();
        // a stream is fed through one front end; text left by the other is text
        if !self.pending.is_empty() {
            let pending = std::mem::take(&mut self.pending);
            self.text(&pending, &mut out);
        }
        match markers.marker(id) {
            Some(marker) => self.marker(marker, &mut out),
            None => self.text(text, &mut out),
        }
        out
    }

    /// Rendered text, markers found by their spelling.
    pub(super) fn push_text(&mut self, text: &str) -> Vec<Piece> {
        self.pending.push_str(text);
        let mut out = Vec::new();
        loop {
            let watched: &[(&str, Marker)] = if self.body.is_some() {
                &INSIDE
            } else {
                &OUTSIDE
            };
            let hit = watched
                .iter()
                .filter_map(|(spelling, marker)| {
                    self.pending.find(spelling).map(|at| (at, *marker))
                })
                .min_by_key(|(at, _)| *at);
            let Some((at, marker)) = hit else {
                let keep = watched
                    .iter()
                    .map(|(spelling, _)| partial_suffix_len(&self.pending, spelling))
                    .max()
                    .unwrap_or(0);
                let upto = self.pending.len() - keep;
                let ready: String = self.pending.drain(..upto).collect();
                self.text(&ready, &mut out);
                break;
            };
            let before: String = self.pending.drain(..at).collect();
            self.text(&before, &mut out);
            self.pending.drain(..marker.spelling().len());
            self.marker(marker, &mut out);
        }
        out
    }

    /// End of the turn: a body still open ends here. A call body that is whole parses; one
    /// that is not, an empty one included, surfaces raw.
    pub(super) fn finish(&mut self) -> Vec<Piece> {
        let mut out = Vec::new();
        if !self.pending.is_empty() {
            let pending = std::mem::take(&mut self.pending);
            self.text(&pending, &mut out);
        }
        if self.body == Some(Body::Call) {
            let inner = std::mem::take(&mut self.call);
            match self.parse_call(&inner) {
                Some(call) => out.push(Piece::Call(call)),
                None => emit_content(&mut out, format!("{}{CALL}{inner}", self.header_text())),
            }
        }
        self.body = None;
        self.header = None;
        out
    }

    fn header_text(&self) -> String {
        match &self.header {
            Some(name) => format!("{MODEL}{name}"),
            None => String::new(),
        }
    }

    fn marker(&mut self, marker: Marker, out: &mut Vec<Piece>) {
        match (self.body, marker) {
            (None, Marker::Model) => self.header = Some(String::new()),
            (None, Marker::End) => self.header = None,
            (None, Marker::Open(body, _)) => {
                self.body = Some(body);
                self.call.clear();
            }
            (Some(Body::Call), Marker::End) => {
                let inner = std::mem::take(&mut self.call);
                match self.parse_call(&inner) {
                    Some(call) => out.push(Piece::Call(call)),
                    // malformed: surfaced verbatim, header and markers included.
                    None => emit_content(out, format!("{}{CALL}{inner}{END}", self.header_text())),
                }
                self.body = None;
                self.header = None;
            }
            (Some(_), Marker::End) => {
                self.body = None;
                self.header = None;
            }
            // inside a body only its close is a marker
            (Some(_), other) => self.text(other.spelling(), out),
        }
    }

    fn text(&mut self, text: &str, out: &mut Vec<Piece>) {
        if text.is_empty() {
            return;
        }
        match self.body {
            None => {
                // text outside a message is not output; it is a header only while it could be one
                if let Some(header) = &mut self.header {
                    header.push_str(text);
                    if header.contains('<') || header.len() > HEADER_CAP {
                        self.header = None;
                    }
                }
            }
            Some(Body::Thinking) => match out.last_mut() {
                Some(Piece::Reasoning(prev)) => prev.push_str(text),
                _ => out.push(Piece::Reasoning(text.to_string())),
            },
            Some(Body::Text) => emit_content(out, text.to_string()),
            Some(Body::Call) => self.call.push_str(text),
        }
    }

    /// One call body: `{"name": string, "args": object}`, other keys ignored (the response
    /// template's transform reads those two). None = malformed. String arguments are cast to
    /// the parameter's declared types when the tool declares any.
    fn parse_call(&mut self, inner: &str) -> Option<ParsedToolCall> {
        let Json::Obj(body) = Json::parse(inner)? else {
            return None;
        };
        let field = |key: &str| body.iter().find(|(k, _)| k == key).map(|(_, v)| v);
        let Json::Str(name) = field("name")? else {
            return None;
        };
        let Json::Obj(args) = field("args")? else {
            return None;
        };
        let mut args = args.clone();
        if let Some(params) = self.types.get(name).filter(|params| !params.is_empty()) {
            for (key, value) in args.iter_mut() {
                let Some(types) = params.get(key) else {
                    continue;
                };
                match value {
                    Json::Str(raw) => *value = coerce(raw, types),
                    // the vendor casts the string items of a list the schema does not call an
                    // array one by one
                    Json::Arr(items) if !types.iter().any(|t| t == "array") => {
                        for item in items.iter_mut() {
                            if let Json::Str(raw) = item {
                                *item = coerce(raw, types);
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        let mut arguments = String::new();
        Json::Obj(args).write(&mut arguments);
        // Deterministic id (greedy serve receipts stay hashable): FNV-1a over index+name+args.
        let id = format!(
            "call_{:016x}",
            fnv1a64(&[
                &self.n_calls.to_le_bytes(),
                name.as_bytes(),
                arguments.as_bytes(),
            ])
        );
        self.n_calls += 1;
        Some(ParsedToolCall {
            id,
            name: name.clone(),
            arguments,
        })
    }
}

/// JSON with every number held as the literal it was written as. `serde_json::Value` reads a
/// number into an f64 or a 64-bit integer: an integer past u64 turns into a float, and `1E400`
/// fails the whole document. A call's arguments are the model's, so they are carried as written.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum Json {
    Null,
    Bool(bool),
    Num(String),
    Str(String),
    Arr(Vec<Json>),
    /// In the order written; a repeated key keeps its first place and its last value, as
    /// Python's `json.loads` and a `serde_json` map do.
    Obj(Vec<(String, Json)>),
}

impl Json {
    /// A whole JSON document (RFC 8259: no `NaN`, no `Infinity`, no lone surrogate escape),
    /// nested at most [`MAX_DEPTH`] deep. None = not one.
    pub(super) fn parse(text: &str) -> Option<Json> {
        let mut reader = Reader {
            s: text.as_bytes(),
            i: 0,
        };
        let value = reader.value(0)?;
        reader.ws();
        (reader.i == reader.s.len()).then_some(value)
    }

    /// Compact JSON. Non-ASCII text is written raw; numbers as their literals.
    pub(super) fn write(&self, out: &mut String) {
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Json::Num(literal) => out.push_str(literal),
            Json::Str(s) => write_string(s, out),
            Json::Arr(items) => {
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    item.write(out);
                }
                out.push(']');
            }
            Json::Obj(pairs) => {
                out.push('{');
                for (i, (key, value)) in pairs.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    write_string(key, out);
                    out.push(':');
                    value.write(out);
                }
                out.push('}');
            }
        }
    }
}

fn write_string(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

struct Reader<'a> {
    s: &'a [u8],
    i: usize,
}

impl Reader<'_> {
    fn ws(&mut self) {
        while matches!(self.s.get(self.i), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.i += 1;
        }
    }

    fn eat(&mut self, literal: &str) -> bool {
        let hit = self.s[self.i..].starts_with(literal.as_bytes());
        if hit {
            self.i += literal.len();
        }
        hit
    }

    fn value(&mut self, depth: usize) -> Option<Json> {
        if depth > MAX_DEPTH {
            return None;
        }
        self.ws();
        match *self.s.get(self.i)? {
            b'{' => {
                self.i += 1;
                let mut pairs: Vec<(String, Json)> = Vec::new();
                self.ws();
                if self.eat("}") {
                    return Some(Json::Obj(pairs));
                }
                loop {
                    self.ws();
                    let key = self.string()?;
                    self.ws();
                    if !self.eat(":") {
                        return None;
                    }
                    let value = self.value(depth + 1)?;
                    match pairs.iter_mut().find(|(k, _)| *k == key) {
                        Some(pair) => pair.1 = value,
                        None => pairs.push((key, value)),
                    }
                    self.ws();
                    if self.eat("}") {
                        return Some(Json::Obj(pairs));
                    }
                    if !self.eat(",") {
                        return None;
                    }
                }
            }
            b'[' => {
                self.i += 1;
                let mut items = Vec::new();
                self.ws();
                if self.eat("]") {
                    return Some(Json::Arr(items));
                }
                loop {
                    items.push(self.value(depth + 1)?);
                    self.ws();
                    if self.eat("]") {
                        return Some(Json::Arr(items));
                    }
                    if !self.eat(",") {
                        return None;
                    }
                }
            }
            b'"' => self.string().map(Json::Str),
            _ if self.eat("null") => Some(Json::Null),
            _ if self.eat("true") => Some(Json::Bool(true)),
            _ if self.eat("false") => Some(Json::Bool(false)),
            _ => self.number().map(Json::Num),
        }
    }

    /// `-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?`
    fn number(&mut self) -> Option<String> {
        let from = self.i;
        let digits = |r: &mut Self| {
            let start = r.i;
            while matches!(r.s.get(r.i), Some(b'0'..=b'9')) {
                r.i += 1;
            }
            r.i - start
        };
        self.eat("-");
        if !self.eat("0") && digits(self) == 0 {
            return None;
        }
        if self.eat(".") && digits(self) == 0 {
            return None;
        }
        if self.eat("e") || self.eat("E") {
            let _ = self.eat("+") || self.eat("-");
            if digits(self) == 0 {
                return None;
            }
        }
        Some(String::from_utf8_lossy(&self.s[from..self.i]).into_owned())
    }

    fn hex4(&mut self) -> Option<u32> {
        let hex = std::str::from_utf8(self.s.get(self.i..self.i + 4)?).ok()?;
        if !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
            return None;
        }
        self.i += 4;
        u32::from_str_radix(hex, 16).ok()
    }

    fn string(&mut self) -> Option<String> {
        if !self.eat("\"") {
            return None;
        }
        let mut out: Vec<u8> = Vec::new();
        loop {
            let byte = *self.s.get(self.i)?;
            self.i += 1;
            match byte {
                b'"' => break,
                b'\\' => {
                    let escape = *self.s.get(self.i)?;
                    self.i += 1;
                    let c = match escape {
                        b'"' => '"',
                        b'\\' => '\\',
                        b'/' => '/',
                        b'n' => '\n',
                        b'r' => '\r',
                        b't' => '\t',
                        b'b' => '\u{8}',
                        b'f' => '\u{c}',
                        b'u' => {
                            let mut code = self.hex4()?;
                            if (0xD800..0xDC00).contains(&code) {
                                if !self.eat("\\u") {
                                    return None;
                                }
                                let low = self.hex4()?;
                                if !(0xDC00..0xE000).contains(&low) {
                                    return None;
                                }
                                code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
                            }
                            char::from_u32(code)?
                        }
                        _ => return None,
                    };
                    let mut buf = [0u8; 4];
                    out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
                }
                0..=0x1f => return None,
                raw => out.push(raw),
            }
        }
        String::from_utf8(out).ok()
    }
}

/// The JSON-schema types of every declared tool parameter, as the vendor response parser reads
/// them (transformers `ResponseParser`, `_schema_types`): `type` as a string or a list, the
/// branches of `anyOf` / `oneOf`, and `null` for a truthy `nullable`. `tools` are the request's
/// tools, a tool wrapping a `function` object or the function object itself. The map arms
/// [`super::ToolStreamParser::inkling2`].
pub fn inkling2_schema_types(
    tools: &[serde_json::Value],
) -> HashMap<String, HashMap<String, Vec<String>>> {
    // Python truthiness, which is what `schema.get("nullable")` is tested with
    fn truthy(value: &serde_json::Value) -> bool {
        match value {
            serde_json::Value::Null => false,
            serde_json::Value::Bool(b) => *b,
            serde_json::Value::Number(n) => n.as_f64() != Some(0.0),
            serde_json::Value::String(s) => !s.is_empty(),
            serde_json::Value::Array(a) => !a.is_empty(),
            serde_json::Value::Object(o) => !o.is_empty(),
        }
    }
    fn types(schema: &serde_json::Value, out: &mut Vec<String>) {
        let Some(schema) = schema.as_object() else {
            return;
        };
        match schema.get("type") {
            Some(serde_json::Value::String(t)) => out.push(t.clone()),
            Some(serde_json::Value::Array(list)) => {
                out.extend(list.iter().filter_map(|t| t.as_str().map(String::from)));
            }
            _ => {}
        }
        for union in ["anyOf", "oneOf"] {
            for choice in schema
                .get(union)
                .and_then(|c| c.as_array())
                .into_iter()
                .flatten()
            {
                types(choice, out);
            }
        }
        if schema.get("nullable").is_some_and(truthy) && !out.iter().any(|t| t == "null") {
            out.push("null".into());
        }
    }
    let mut map = HashMap::new();
    for tool in tools {
        let function = tool.get("function").unwrap_or(tool);
        let Some(name) = function.get("name").and_then(|n| n.as_str()) else {
            continue;
        };
        let mut params = HashMap::new();
        if let Some(properties) = function
            .get("parameters")
            .and_then(|p| p.get("properties"))
            .and_then(|p| p.as_object())
        {
            for (param, schema) in properties {
                let mut declared = Vec::new();
                types(schema, &mut declared);
                if !declared.is_empty() {
                    params.insert(param.clone(), declared);
                }
            }
        }
        map.insert(name.to_string(), params);
    }
    map
}

/// `[+-]?[0-9]+` as a JSON integer literal: no plus sign, no leading zeros, no negative zero.
fn integer_literal(text: &str) -> Option<String> {
    let (negative, digits) = match text.as_bytes().first()? {
        b'-' => (true, &text[1..]),
        b'+' => (false, &text[1..]),
        _ => (false, text),
    };
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let digits = digits.trim_start_matches('0');
    Some(match (digits.is_empty(), negative) {
        (true, _) => "0".into(),
        (false, true) => format!("-{digits}"),
        (false, false) => digits.to_string(),
    })
}

/// A finite whole float as the integer it is, every digit (`1e20` is `100000000000000000000`).
fn whole_float(value: f64) -> Option<String> {
    (value.is_finite() && value.fract() == 0.0)
        .then(|| integer_literal(&format!("{value:.0}")))
        .flatten()
}

/// The vendor parser's cast of a string argument to its parameter's declared types
/// (transformers `_coerce`): the first type the text reads as wins, and a text no type reads
/// stays the string it is. `string` and unknown types read nothing.
///
/// Two places leave the vendor, both recorded in the fixtures. Python's `int()` and `float()`
/// accept digit separators (`1_000`) and non-ASCII digits; those are Python numerals, not
/// numbers of JSON or JSON Schema, and stay strings here. And a digit string cast to `number`
/// keeps every digit, where the vendor rounds it through a float.
fn coerce(raw: &str, types: &[String]) -> Json {
    for declared in types {
        let text = raw.trim();
        let float = || text.parse::<f64>().ok().filter(|n| n.is_finite());
        let value = match declared.as_str() {
            "integer" => integer_literal(text)
                .or_else(|| float().and_then(whole_float))
                .map(Json::Num),
            "number" => integer_literal(text)
                .or_else(|| {
                    let number = float()?;
                    if number.fract() == 0.0 && !raw.contains('.') {
                        whole_float(number)
                    } else {
                        Some(format!("{number:?}"))
                    }
                })
                .map(Json::Num),
            "boolean" => match text.to_lowercase().as_str() {
                "true" | "1" => Some(Json::Bool(true)),
                "false" | "0" => Some(Json::Bool(false)),
                _ => None,
            },
            "null" => matches!(text, "null" | "None").then_some(Json::Null),
            "object" => Json::parse(raw).filter(|v| matches!(v, Json::Obj(_))),
            "array" => Json::parse(raw).filter(|v| matches!(v, Json::Arr(_))),
            _ => None,
        };
        if let Some(value) = value {
            return value;
        }
    }
    Json::Str(raw.to_string())
}

#[cfg(test)]
mod tests {
    use super::super::ToolStreamParser;
    use super::*;

    const EOS: &str = "<|content_model_end_sampling|>";

    fn fixtures() -> Vec<Json> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(
            "../../research/modelplan-onboarding-inkling2mini-20261121/chat/fixtures/parses.json",
        );
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
        match Json::parse(&text) {
            Some(Json::Arr(cases)) => cases,
            _ => {
                // name where the reader stopped: the furthest element it could read
                let mut reader = Reader {
                    s: text.as_bytes(),
                    i: 1,
                };
                let mut read = 0;
                while reader.value(1).is_some() {
                    read += 1;
                    reader.ws();
                    reader.eat(",");
                }
                panic!("parses.json: element {read} is not JSON this reader takes");
            }
        }
    }

    fn get<'a>(value: &'a Json, key: &str) -> Option<&'a Json> {
        match value {
            Json::Obj(pairs) => pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    fn text<'a>(value: &'a Json, key: &str) -> &'a str {
        match get(value, key) {
            Some(Json::Str(s)) => s,
            _ => "",
        }
    }

    fn list<'a>(value: &'a Json, key: &str) -> &'a [Json] {
        match get(value, key) {
            Some(Json::Arr(items)) => items,
            _ => &[],
        }
    }

    fn parser(case: &Json) -> ToolStreamParser {
        let tools: Vec<serde_json::Value> = list(case, "tools")
            .iter()
            .map(|tool| {
                let mut json = String::new();
                tool.write(&mut json);
                serde_json::from_str(&json).unwrap()
            })
            .collect();
        ToolStreamParser::inkling2(inkling2_schema_types(&tools))
    }

    /// Equal JSON values: objects whatever their key order, integers digit for digit, any
    /// other number by value.
    fn same(a: &Json, b: &Json) -> bool {
        match (a, b) {
            (Json::Num(x), Json::Num(y)) => {
                let integer = |n: &str| !n.contains(['.', 'e', 'E']);
                if integer(x) && integer(y) {
                    x == y
                } else {
                    x.parse::<f64>().unwrap() == y.parse::<f64>().unwrap()
                }
            }
            (Json::Arr(x), Json::Arr(y)) => {
                x.len() == y.len() && x.iter().zip(y).all(|(p, q)| same(p, q))
            }
            (Json::Obj(x), Json::Obj(y)) => {
                x.len() == y.len()
                    && x.iter()
                        .all(|(k, p)| y.iter().any(|(l, q)| k == l && same(p, q)))
            }
            _ => a == b,
        }
    }

    /// The pieces against a message as the vendor parser returns it: joined reasoning, joined
    /// content (absent when empty) and the calls in order. An expectation that pins a call's
    /// `arguments_json` wants that exact text.
    fn check(pieces: &[Piece], want: &Json, what: &str) {
        let (mut content, mut reasoning, mut calls) = (String::new(), String::new(), Vec::new());
        for piece in pieces {
            match piece {
                Piece::Content(text) => content.push_str(text),
                Piece::Reasoning(text) => reasoning.push_str(text),
                Piece::Call(call) => calls.push(call),
            }
        }
        assert_eq!(content, text(want, "content"), "{what}: content");
        assert_eq!(
            reasoning,
            text(want, "reasoning_content"),
            "{what}: reasoning"
        );
        let wanted = list(want, "tool_calls");
        assert_eq!(calls.len(), wanted.len(), "{what}: calls {calls:?}");
        for (call, wanted) in calls.iter().zip(wanted) {
            let function = get(wanted, "function").unwrap();
            assert_eq!(call.name, text(function, "name"), "{what}: name");
            match get(function, "arguments_json") {
                Some(Json::Str(exact)) => assert_eq!(&call.arguments, exact, "{what}: arguments"),
                _ => {
                    let got = Json::parse(&call.arguments).expect("arguments are JSON");
                    let want = get(function, "arguments").unwrap();
                    assert!(
                        same(&got, want),
                        "{what}: arguments {} vs {want:?}",
                        call.arguments
                    );
                }
            }
        }
    }

    /// The ids a real tokenizer gives the framing: every control token its own id.
    fn markers() -> Inkling2Markers {
        Inkling2Markers::resolve(|piece| {
            [MODEL, END, THINKING, TEXT, TOOL_ERROR, TOOL_TEXT, CALL]
                .iter()
                .position(|m| *m == piece)
                .map(|at| 200_000 + at as u32)
        })
        .unwrap()
    }

    /// Rendered text as the token stream it decodes from: each control-token spelling one
    /// token with its id, the text between them ordinary tokens of at most `chunk` chars.
    fn tokens(output: &str, chunk: usize) -> Vec<(u32, String)> {
        let specials = [MODEL, END, THINKING, TEXT, TOOL_ERROR, TOOL_TEXT, CALL, EOS];
        let mut out = Vec::new();
        let mut rest = output;
        while !rest.is_empty() {
            let hit = specials
                .iter()
                .enumerate()
                .filter_map(|(at, s)| rest.find(s).map(|i| (i, at, *s)))
                .min_by_key(|(i, _, _)| *i);
            let (plain, special) = match hit {
                Some((i, at, s)) => (&rest[..i], Some((200_000 + at as u32, s))),
                None => (rest, None),
            };
            let chars: Vec<char> = plain.chars().collect();
            for piece in chars.chunks(chunk) {
                out.push((7, piece.iter().collect()));
            }
            rest = &rest[plain.len()..];
            if let Some((id, s)) = special {
                out.push((id, s.to_string()));
                rest = &rest[s.len()..];
            }
        }
        out
    }

    /// Every fixture parses to the message transformers' `parse_response` returns for it under
    /// the artifact's `response_template`, or, where the fixture records that this engine
    /// leaves the vendor, to the recorded message. Four ways: the text whole, the text one char
    /// at a time (a held-back marker or header must not change the result), and the token
    /// stream by id with whole and with one-char text tokens.
    #[test]
    fn inkling2_parses_match_the_response_template() {
        let cases = fixtures();
        assert!(cases.len() >= 46, "fixture set shrank: {}", cases.len());
        let markers = markers();
        let mut own = 0;
        for case in &cases {
            let name = text(case, "name");
            let output = text(case, "output");
            let want = match get(case, "memra") {
                Some(memra) => {
                    own += 1;
                    assert!(!text(memra, "reason").is_empty(), "{name}");
                    get(memra, "message").unwrap()
                }
                None => get(case, "message").unwrap_or_else(|| panic!("{name}: no expectation")),
            };
            let mut whole = parser(case);
            let mut pieces = whole.push(output);
            pieces.extend(whole.finish());
            check(&pieces, want, &format!("{name}: text, whole"));

            for streaming in [false, true] {
                let mut by_char = if streaming {
                    parser(case).streaming()
                } else {
                    parser(case)
                };
                let mut pieces = Vec::new();
                let mut buf = [0u8; 4];
                for ch in output.chars() {
                    pieces.extend(by_char.push(ch.encode_utf8(&mut buf)));
                }
                pieces.extend(by_char.finish());
                check(&pieces, want, &format!("{name}: text, char by char"));
            }

            for chunk in [usize::MAX, 1] {
                let mut by_id = parser(case);
                let mut pieces = Vec::new();
                for (id, piece) in tokens(output, chunk) {
                    pieces.extend(by_id.push_token(&markers, id, &piece));
                }
                pieces.extend(by_id.finish());
                check(&pieces, want, &format!("{name}: ids, chunk {chunk}"));
            }
        }
        assert_eq!(own, 16, "the recorded departures from the vendor parser");
    }

    /// Fed by id, a body that spells an end marker, a model header and a whole call in
    /// ordinary text tokens is content, all of it, and no call. The same characters through the
    /// text front end do produce the call: that front end cannot tell, which is why the serve
    /// path must not use it.
    #[test]
    fn inkling2_spelled_markers_stay_text_when_fed_by_id() {
        let quoted =
            format!("{END}{MODEL}f{CALL}{{\"name\":\"f\",\"args\":{{}}}}{END}{MODEL}{TEXT}tail");
        let markers = markers();
        for chunk in [usize::MAX, 3, 1] {
            let mut p = ToolStreamParser::inkling2(HashMap::new());
            let mut pieces = p.push_token(&markers, 200_000, MODEL);
            pieces.extend(p.push_token(&markers, 200_003, TEXT));
            pieces.extend(p.push_token(&markers, 7, "The docs say a call looks like "));
            let chars: Vec<char> = quoted.chars().collect();
            for piece in chars.chunks(chunk.min(chars.len())) {
                let piece: String = piece.iter().collect();
                pieces.extend(p.push_token(&markers, 7, &piece));
            }
            pieces.extend(p.push_token(&markers, 200_001, END));
            pieces.extend(p.finish());
            let mut content = String::new();
            for piece in &pieces {
                match piece {
                    Piece::Content(text) => content.push_str(text),
                    other => panic!("chunk {chunk}: {other:?}"),
                }
            }
            assert_eq!(content, format!("The docs say a call looks like {quoted}"));
            assert_eq!(p.n_calls(), 0);
        }

        let mut by_text = ToolStreamParser::inkling2(HashMap::new());
        let mut pieces = by_text.push(&format!(
            "{MODEL}{TEXT}The docs say a call looks like {quoted}{END}"
        ));
        pieces.extend(by_text.finish());
        assert!(
            pieces
                .iter()
                .any(|p| matches!(p, Piece::Call(call) if call.name == "f")),
            "the text front end reads the spelling: {pieces:?}"
        );
    }

    /// By id, text is never held back: there is no spelling to wait for.
    #[test]
    fn inkling2_ids_stream_every_text_token_at_once() {
        let markers = markers();
        let mut p = ToolStreamParser::inkling2(HashMap::new());
        assert!(p.push_token(&markers, 200_000, MODEL).is_empty());
        assert!(p.push_token(&markers, 200_002, THINKING).is_empty());
        assert_eq!(
            p.push_token(&markers, 7, "Let me think <|end_mess"),
            vec![Piece::Reasoning("Let me think <|end_mess".into())]
        );
        assert!(p.push_token(&markers, 200_001, END).is_empty());
        assert!(p.push_token(&markers, 7, " stray ").is_empty());
        assert!(p.push_token(&markers, 200_003, TEXT).is_empty());
        assert_eq!(
            p.push_token(&markers, 7, "Hi <"),
            vec![Piece::Content("Hi <".into())]
        );
        // inside a body only the end marker's id closes it; another marker's id is its text
        assert_eq!(
            p.push_token(&markers, 200_002, THINKING),
            vec![Piece::Content(THINKING.into())]
        );
        assert!(p.push_token(&markers, 200_001, END).is_empty());
        assert!(p.push_token(&markers, 9, EOS).is_empty());
        assert!(p.finish().is_empty());
    }

    /// Content and reasoning stream as text arrives: only a possible end marker is held back.
    #[test]
    fn inkling2_text_streams_bodies_without_waiting_for_the_close() {
        let mut p = ToolStreamParser::inkling2(HashMap::new());
        assert!(p.push("<|message_model|><|content_thin").is_empty());
        assert_eq!(
            p.push("king|>Let me"),
            vec![Piece::Reasoning("Let me".into())]
        );
        assert_eq!(
            p.push(" think<|end_"),
            vec![Piece::Reasoning(" think".into())]
        );
        assert!(p.push("message|><|message_model|>").is_empty());
        assert_eq!(
            p.push("<|content_text|>Hi <"),
            vec![Piece::Content("Hi ".into())]
        );
        assert_eq!(p.push("b>"), vec![Piece::Content("<b>".into())]);
        assert!(
            p.push("<|end_message|><|content_model_end_sampling|>")
                .is_empty()
        );
        assert!(p.finish().is_empty());
        assert_eq!(p.n_calls(), 0);
    }

    /// A call is held until its message closes, takes a deterministic id, and a parser made
    /// `fresh` from a used one starts outside a message with the same types.
    #[test]
    fn inkling2_calls_are_numbered_and_fresh_restarts() {
        let tools = [
            serde_json::json!({"type": "function", "function": {"name": "f",
            "parameters": {"type": "object", "properties": {"n": {"type": "integer"}}}}}),
        ];
        let mut p = ToolStreamParser::inkling2(inkling2_schema_types(&tools));
        let call =
            "<|message_model|>f<|content_invoke_tool_json|>{\"name\":\"f\",\"args\":{\"n\":\"4\"}}";
        assert!(p.push(call).is_empty());
        let pieces = p.push("<|end_message|>");
        let [Piece::Call(first)] = pieces.as_slice() else {
            panic!("{pieces:?}");
        };
        assert_eq!(
            (first.name.as_str(), first.arguments.as_str()),
            ("f", "{\"n\":4}")
        );
        assert!(first.id.starts_with("call_"));
        assert_eq!(p.n_calls(), 1);
        assert_eq!(p.push("<|message_model|><|content_text|>mid").len(), 1);

        let mut again = p.fresh();
        assert_eq!(again.n_calls(), 0);
        let mut pieces = again.push(&format!("stray{call}<|end_message|>"));
        pieces.extend(again.finish());
        let [Piece::Call(second)] = pieces.as_slice() else {
            panic!("{pieces:?}");
        };
        assert_eq!(second, first);
    }

    /// The other dialects' markers are text to this parser, and its markers are text to theirs.
    #[test]
    fn inkling2_and_the_think_tag_dialects_do_not_cross() {
        let mut p = ToolStreamParser::inkling2(HashMap::new());
        let mut pieces =
            p.push("<|content_text|><think>x</think><tool_call>y</tool_call><|end_message|>");
        pieces.extend(p.finish());
        assert_eq!(
            pieces,
            vec![Piece::Content(
                "<think>x</think><tool_call>y</tool_call>".into()
            )]
        );
        let mut qwen = ToolStreamParser::new(HashMap::new(), false);
        let text = "<|message_model|><|content_invoke_tool_json|>{\"name\":\"f\",\"args\":{}}<|end_message|>";
        let mut pieces = qwen.push(text);
        pieces.extend(qwen.finish());
        assert_eq!(pieces, vec![Piece::Content(text.into())]);
    }

    /// The reader takes JSON and nothing else, and writes numbers back as it read them.
    #[test]
    fn inkling2_json_reader_keeps_literals_and_refuses_what_is_not_json() {
        let round = |text: &str| {
            let mut out = String::new();
            Json::parse(text).unwrap().write(&mut out);
            out
        };
        assert_eq!(
            round(
                " {\"a\" : [1E400, -0, 123456789012345678901234567890, 2.50, 1e-07], \"b\":\"\\u00e9\\ud83d\\ude42\\n\"} "
            ),
            "{\"a\":[1E400,-0,123456789012345678901234567890,2.50,1e-07],\"b\":\"é🙂\\n\"}"
        );
        assert_eq!(round("{\"k\":1,\"j\":2,\"k\":3}"), "{\"k\":3,\"j\":2}");
        for bad in [
            "",
            "{",
            "[1,]",
            "{\"a\":1,}",
            "01",
            "1.",
            ".5",
            "+1",
            "1e",
            "NaN",
            "Infinity",
            "-Infinity",
            "'a'",
            "\"\\ud800\"",
            "\"\\udc00\\ud800\"",
            "\"\\x41\"",
            "\"a\nb\"",
            "tru",
            "{\"a\" 1}",
            "[1 2]",
            "1 2",
        ] {
            assert_eq!(Json::parse(bad), None, "{bad:?}");
        }
        let deep = |n: usize| format!("{}{}", "[".repeat(n), "]".repeat(n));
        assert!(Json::parse(&deep(MAX_DEPTH)).is_some());
        assert_eq!(Json::parse(&deep(MAX_DEPTH + 2)), None);
        assert_eq!(Json::parse(&deep(100_000)), None);
    }

    /// A vocabulary without one of the markers is not this dialect's.
    #[test]
    fn inkling2_markers_need_every_marker() {
        let error = Inkling2Markers::resolve(|piece| (piece != CALL).then_some(1)).unwrap_err();
        assert!(error.contains(CALL), "{error}");
    }
}
