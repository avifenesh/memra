//! Streaming parser for template-law tool-call emissions (serve-tools lane, 2026-08-02).
//!
//! The qwen3.5/3.6-class templates instruct the model to emit
//!
//! ```text
//! optional prose...
//! <tool_call>
//! <function=get_weather>
//! <parameter=city>
//! Paris
//! </parameter>
//! </function>
//! </tool_call>
//! ```
//!
//! This module turns that text stream into OpenAI-shape `tool_calls` while passing everything
//! else through as content. It is PARSING ONLY — it sits between the worker's token stream and
//! the HTTP response and never touches generation. It is constructed ONLY for requests that
//! rendered a `<tools>` block (non-tools traffic bypasses it entirely: byte-identical streams,
//! including chunk boundaries — the isolation contract).
//!
//! MALFORMED-EMISSION POLICY (gate c): a `<tool_call>...</tool_call>` block that does not parse
//! (missing/garbled `<function=`, unpaired `<parameter=`) is surfaced VERBATIM as content —
//! tags included — and the stream continues; an unterminated `<tool_call>` at end-of-generation
//! flushes raw. Never an error, never dropped bytes: content + parsed calls always reassemble
//! to the exact generated text.
//!
//! THINK GATE: when the rendered prompt ended with an open `<think>\n` tail (the template
//! default), everything up to and including `</think>` passes through as content unscanned —
//! a `<tool_call>` mentioned while reasoning is not a call.

use std::collections::HashMap;

/// One parsed call, OpenAI-shape: `arguments` is a compact JSON object STRING.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Piece {
    Content(String),
    /// Think-segment text (serve-compat lane, 2026-08-03; gap-scan F13): the OpenRouter
    /// `reasoning` response field. Emitted while the prompt's open `<think>` tail is live;
    /// the `</think>` tag itself and its trailing `\n\n` separator are syntax, not output.
    Reasoning(String),
    Call(ParsedToolCall),
}

enum State {
    /// Prompt ended with an open `<think>` — text routes to `reasoning` until `</think>`.
    Prethink,
    /// Just past `</think>`: swallow the (up to two) separator newlines, then Scan.
    PostThink,
    /// Scanning content for `<tool_call>`.
    Scan,
    /// Inside a `<tool_call>` block, buffering until `</tool_call>`.
    InCall,
    /// gemma dialect: just consumed `<|channel>` — the channel-name line (`thought\n`) is
    /// syntax; swallow through its newline, then GemmaThought.
    GemmaLabel,
    /// gemma dialect: inside a thought channel — text routes to `reasoning` until
    /// `<channel|>` (whose preceding syntax `\n` is also swallowed).
    GemmaThought,
    /// gemma tooluse dialect: inside a `<|tool_call>...` span, buffering until `<tool_call|>`.
    GemmaCall,
    /// deepseek-v4: inside a `<｜DSML｜tool_calls>...` block, buffering until
    /// `</｜DSML｜tool_calls>`.
    Dsv4Call,
}

const OPEN: &str = "<tool_call>";
const CLOSE: &str = "</tool_call>";
const THINK_END: &str = "</think>";
/// gemma4 thought-channel dialect (lane/gemma4-serve-gaps, 2026-08-07): the template's
/// `strip_thinking` law — `<|channel>thought\n{text}\n<channel|>` — is what the model emits;
/// the serve stream must apply the same split, thought -> `reasoning`, tags + label + the
/// bracketing newlines are syntax. Channels may open ANYWHERE in the stream (the template
/// strips them from any position in history), so the gemma scanner runs the whole stream,
/// unlike the qwen prompt-open-tail Prethink.
const GEMMA_OPEN: &str = "<|channel>";
const GEMMA_CLOSE: &str = "<channel|>";
/// gemma tooluse dialect (lane/gemma4-tools, 2026-08-18): the served trunk (official Google
/// tooluse template) emits `<|tool_call>call:NAME{args}<tool_call|>`. The args are the compact
/// non-JSON dialect (bare keys, `<|"|>`-wrapped strings, bare numbers/true/false/None, nested
/// {}/[]) — parsed back into an OpenAI arguments JSON string. Spans NEVER leak into content;
/// generation stops when `<tool_call|>` completes (the serve path adds it to the request's stop
/// set), so a request yields one call per turn.
const GEMMA_CALL_OPEN: &str = "<|tool_call>";
const GEMMA_CALL_CLOSE: &str = "<tool_call|>";
/// gemma dialect string marker (`<|"|>`): its `\n`-free special-token nature means string
/// content never contains it, so it delimits string values unambiguously.
const GEMMA_DQ: &str = "<|\"|>";
/// deepseek-v4 (encoding_dsv4) tool-call block markers. The assistant emits
/// `\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name="N">\n<params>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>`
/// (multiple invokes per block). The serve path adds `</｜DSML｜tool_calls>` to the stop set
/// (scoped to dsv4 tool requests, never global), and the close stays in the stream so this
/// parser closes the span. `｜` is U+FF5C. The `\n\n` before the open is wire syntax, stripped
/// from content. Multiple invokes -> multiple OpenAI tool_calls.
const DSV4_OPEN: &str = "<\u{ff5c}DSML\u{ff5c}tool_calls>";
const DSV4_CLOSE: &str = "</\u{ff5c}DSML\u{ff5c}tool_calls>";
const DSV4_INVOKE: &str = "<\u{ff5c}DSML\u{ff5c}invoke name=\"";
const DSV4_PARAM: &str = "<\u{ff5c}DSML\u{ff5c}parameter name=\"";
const DSV4_PARAM_END: &str = "</\u{ff5c}DSML\u{ff5c}parameter>";
const DSV4_INVOKE_END: &str = "</\u{ff5c}DSML\u{ff5c}invoke>";

pub struct ToolStreamParser {
    state: State,
    /// Held-back text: in Prethink/Scan at most a partial tag suffix; in InCall the block body.
    buf: String,
    /// Declared JSON-schema `type` per (function, parameter) — drives argument coercion.
    schemas: HashMap<String, HashMap<String, String>>,
    n_calls: usize,
    /// false = reasoning-only mode (non-tools chat on a think-class model): post-think text
    /// is pure content, never scanned for `<tool_call>` (no holdback, byte-identical stream).
    scan_tools: bool,
    /// gemma4 thought-channel dialect: Scan watches for `<|channel>` instead of tool tags.
    gemma: bool,
    /// gemma4 tooluse dialect: Scan watches for BOTH `<|channel>` (thought) and `<|tool_call>`
    /// (call) spans; everything else is content.
    gemma_tools: bool,
    /// deepseek-v4 dialect: reasoning routes to `</think>` (NO separator-newline swallow —
    /// dsv4 content starts immediately after `</think>`), then Scan watches for
    /// `<｜DSML｜tool_calls>` (the `\n\n` before it is wire syntax).
    dsv4: bool,
    /// Separator-newline budget right after `</think>` (the template emits `</think>\n\n`).
    postthink_nl: u8,
}

/// Length of the longest PROPER prefix of `tag` that `s` ends with. NOTE: byte-indexed —
/// callers holding back `keep` bytes must only do so on ASCII tags (always a char
/// boundary) or re-check boundaries (the stop-scrubber truncates on char_indices).
pub fn partial_suffix_len(s: &str, tag: &str) -> usize {
    let max = (tag.len() - 1).min(s.len());
    for k in (1..=max).rev() {
        // `tag[..k]` must slice at a char boundary — the dsv4 markers carry the multibyte
        // `｜` (U+FF5C, 3 bytes), so a byte-index k inside it is not a valid slice. Skip
        // non-boundary k (a no-op for the ASCII qwen/gemma tags, where every index is one).
        if tag.is_char_boundary(k) && s.ends_with(&tag[..k]) {
            return k;
        }
    }
    0
}

impl ToolStreamParser {
    /// `schemas`: function name -> parameter -> declared JSON-schema type string.
    /// `skip_think`: true when the rendered prompt ends with an open `<think>\n` tail.
    pub fn new(schemas: HashMap<String, HashMap<String, String>>, skip_think: bool) -> Self {
        Self {
            state: if skip_think {
                State::Prethink
            } else {
                State::Scan
            },
            buf: String::new(),
            schemas,
            n_calls: 0,
            scan_tools: true,
            gemma: false,
            gemma_tools: false,
            dsv4: false,
            postthink_nl: 0,
        }
    }

    /// deepseek-v4 (encoding_dsv4) parser (lane/dsv4-template): reasoning routes to `</think>`
    /// on a thinking-mode prompt (`skip_think`; NO `</think>\n\n` separator, unlike qwen —
    /// dsv4 content begins immediately), then content until `\n\n<｜DSML｜tool_calls>`, whose
    /// block (one or more `<｜DSML｜invoke>`) parses to OpenAI `tool_calls`; the `\n\n` prefix
    /// is stripped. `schemas` unused — the DSML wire is self-describing (`string="true|false"`
    /// per param). Malformed span surfaces VERBATIM (house policy; the oracle raises).
    pub fn dsv4(skip_think: bool) -> Self {
        let mut p = Self::new(HashMap::new(), skip_think);
        p.scan_tools = false;
        p.dsv4 = true;
        p
    }

    /// gemma4 tooluse parser (lane/gemma4-tools): splits `<|channel>thought…<channel|>` to
    /// `reasoning` and `<|tool_call>call:NAME{…}<tool_call|>` to OpenAI `tool_calls`; everything
    /// else is content. Channels/calls may open at any stream position (the template's own
    /// strip_thinking law + a call after content). Reasoning-vs-content, never a tool span in
    /// content. `schemas` is unused here — the gemma call dialect is self-describing.
    pub fn gemma_tools() -> Self {
        let mut p = Self::new(HashMap::new(), false);
        p.scan_tools = false;
        p.gemma_tools = true;
        p
    }

    /// Reasoning-only parser for NON-tools chat on a think-open model (gap-scan F13):
    /// think text -> `reasoning`, everything after `</think>` passes through as content
    /// unscanned (a `<tool_call>` in plain prose is prose).
    pub fn reasoning_only() -> Self {
        let mut p = Self::new(HashMap::new(), true);
        p.scan_tools = false;
        p
    }

    /// gemma4 thought-channel splitter (lane/gemma4-serve-gaps, 2026-08-07): the model's
    /// `<|channel>thought\n{text}\n<channel|>` blocks route to `reasoning` (tags, the
    /// channel label line and the bracketing newlines are syntax); everything outside a
    /// channel is content. Channels can open at any stream position, matching the
    /// template's own `strip_thinking` law. gemma4 templates have no `<tools>` branch,
    /// so this is reasoning-only by construction.
    pub fn gemma_thought() -> Self {
        let mut p = Self::new(HashMap::new(), false);
        p.scan_tools = false;
        p.gemma = true;
        p
    }

    pub fn push(&mut self, text: &str) -> Vec<Piece> {
        self.buf.push_str(text);
        let mut out = Vec::new();
        loop {
            match self.state {
                State::Prethink => {
                    if let Some(i) = self.buf.find(THINK_END) {
                        // think text -> reasoning; the tag itself is syntax, not output.
                        self.emit_reasoning(&mut out, self.buf[..i].to_string());
                        self.buf.drain(..i + THINK_END.len());
                        // dsv4 content starts IMMEDIATELY after `</think>` (no separator
                        // newlines, unlike the qwen `</think>\n\n`); go straight to Scan.
                        if self.dsv4 {
                            self.state = State::Scan;
                        } else {
                            self.state = State::PostThink;
                            self.postthink_nl = 2;
                        }
                        continue;
                    }
                    let keep = partial_suffix_len(&self.buf, THINK_END);
                    let emit_to = self.buf.len() - keep;
                    if emit_to > 0 {
                        self.emit_reasoning(&mut out, self.buf[..emit_to].to_string());
                        self.buf.drain(..emit_to);
                    }
                    break;
                }
                State::PostThink => {
                    // swallow the template's `</think>\n\n` separator newlines (syntax).
                    while self.postthink_nl > 0 && self.buf.starts_with('\n') {
                        self.buf.drain(..1);
                        self.postthink_nl -= 1;
                    }
                    if self.postthink_nl > 0 && self.buf.is_empty() {
                        break; // more separator may still arrive
                    }
                    self.state = State::Scan;
                    continue;
                }
                State::Scan => {
                    if self.gemma_tools {
                        // content until the EARLIER of a `<|channel>` (thought) or a
                        // `<|tool_call>` (call). Both start with `<|`; a partial suffix of
                        // either is held back so a split tag never leaks as content.
                        let ch = self.buf.find(GEMMA_OPEN);
                        let cl = self.buf.find(GEMMA_CALL_OPEN);
                        let pick = match (ch, cl) {
                            (Some(a), Some(b)) if a <= b => Some((a, true)),
                            (Some(_), Some(b)) => Some((b, false)),
                            (Some(a), None) => Some((a, true)),
                            (None, Some(b)) => Some((b, false)),
                            (None, None) => None,
                        };
                        if let Some((i, is_channel)) = pick {
                            if i > 0 {
                                emit_content(&mut out, self.buf[..i].to_string());
                            }
                            if is_channel {
                                self.buf.drain(..i + GEMMA_OPEN.len());
                                self.state = State::GemmaLabel;
                            } else {
                                self.buf.drain(..i + GEMMA_CALL_OPEN.len());
                                self.state = State::GemmaCall;
                            }
                            continue;
                        }
                        let keep = partial_suffix_len(&self.buf, GEMMA_OPEN)
                            .max(partial_suffix_len(&self.buf, GEMMA_CALL_OPEN));
                        let emit_to = self.buf.len() - keep;
                        if emit_to > 0 {
                            emit_content(&mut out, self.buf[..emit_to].to_string());
                            self.buf.drain(..emit_to);
                        }
                        break;
                    }
                    if self.gemma {
                        // gemma dialect: content until a `<|channel>` opens a thought.
                        if let Some(i) = self.buf.find(GEMMA_OPEN) {
                            if i > 0 {
                                emit_content(&mut out, self.buf[..i].to_string());
                            }
                            self.buf.drain(..i + GEMMA_OPEN.len());
                            self.state = State::GemmaLabel;
                            continue;
                        }
                        let keep = partial_suffix_len(&self.buf, GEMMA_OPEN);
                        let emit_to = self.buf.len() - keep;
                        if emit_to > 0 {
                            emit_content(&mut out, self.buf[..emit_to].to_string());
                            self.buf.drain(..emit_to);
                        }
                        break;
                    }
                    if self.dsv4 {
                        // content until `<｜DSML｜tool_calls>`; the `\n\n` before it is wire
                        // syntax (encoding_dsv4 tool_calls_start_token, E:710) and is stripped.
                        if let Some(i) = self.buf.find(DSV4_OPEN) {
                            let mut end = i;
                            // strip up to two `\n` immediately before the open (the `\n\n`).
                            for _ in 0..2 {
                                if end > 0 && self.buf.as_bytes()[end - 1] == b'\n' {
                                    end -= 1;
                                } else {
                                    break;
                                }
                            }
                            if end > 0 {
                                emit_content(&mut out, self.buf[..end].to_string());
                            }
                            self.buf.drain(..i + DSV4_OPEN.len());
                            self.state = State::Dsv4Call;
                            continue;
                        }
                        // hold back a partial `<｜DSML｜tool_calls>` suffix, plus up to two
                        // trailing newlines that may be its `\n\n` prefix (stripped on match).
                        let mut keep = partial_suffix_len(&self.buf, DSV4_OPEN);
                        let mut nl = 0;
                        while nl < 2
                            && keep < self.buf.len()
                            && self.buf[..self.buf.len() - keep].ends_with('\n')
                        {
                            keep += 1;
                            nl += 1;
                        }
                        let emit_to = self.buf.len() - keep;
                        if emit_to > 0 {
                            emit_content(&mut out, self.buf[..emit_to].to_string());
                            self.buf.drain(..emit_to);
                        }
                        break;
                    }
                    if !self.scan_tools {
                        // reasoning-only mode: post-think text is pure content, unscanned.
                        if !self.buf.is_empty() {
                            emit_content(&mut out, std::mem::take(&mut self.buf));
                        }
                        break;
                    }
                    if let Some(i) = self.buf.find(OPEN) {
                        if i > 0 {
                            emit_content(&mut out, self.buf[..i].to_string());
                        }
                        self.buf.drain(..i + OPEN.len());
                        self.state = State::InCall;
                        continue;
                    }
                    let keep = partial_suffix_len(&self.buf, OPEN);
                    let emit_to = self.buf.len() - keep;
                    if emit_to > 0 {
                        emit_content(&mut out, self.buf[..emit_to].to_string());
                        self.buf.drain(..emit_to);
                    }
                    break;
                }
                State::InCall => {
                    let Some(i) = self.buf.find(CLOSE) else { break };
                    let inner: String = self.buf[..i].to_string();
                    self.buf.drain(..i + CLOSE.len());
                    self.state = State::Scan;
                    match self.parse_block(&inner) {
                        Some(call) => out.push(Piece::Call(call)),
                        // malformed: surfaced verbatim, tags included, stream continues.
                        None => emit_content(&mut out, format!("{OPEN}{inner}{CLOSE}")),
                    }
                    continue;
                }
                State::GemmaLabel => {
                    // the channel-name line (`thought\n`) is syntax — swallow through the
                    // newline. Held back until the newline arrives (label is short).
                    let Some(i) = self.buf.find('\n') else { break };
                    self.buf.drain(..i + 1);
                    self.state = State::GemmaThought;
                    continue;
                }
                State::GemmaThought => {
                    if let Some(i) = self.buf.find(GEMMA_CLOSE) {
                        // thought -> reasoning; the tag and its preceding syntax `\n` are
                        // not output (the template renders `{text}\n<channel|>`).
                        let text = self.buf[..i].strip_suffix('\n').unwrap_or(&self.buf[..i]);
                        self.emit_reasoning(&mut out, text.to_string());
                        self.buf.drain(..i + GEMMA_CLOSE.len());
                        self.state = State::Scan;
                        continue;
                    }
                    // Hold back a partial `<channel|>` suffix, plus the newline right
                    // before it (or a bare trailing newline) — it may be the close tag's
                    // syntax `\n`; if prose follows instead, it flushes with the next push.
                    let mut keep = partial_suffix_len(&self.buf, GEMMA_CLOSE);
                    if self.buf[..self.buf.len() - keep].ends_with('\n') {
                        keep += 1;
                    }
                    let emit_to = self.buf.len() - keep;
                    if emit_to > 0 {
                        self.emit_reasoning(&mut out, self.buf[..emit_to].to_string());
                        self.buf.drain(..emit_to);
                    }
                    break;
                }
                State::GemmaCall => {
                    let Some(i) = self.buf.find(GEMMA_CALL_CLOSE) else {
                        break;
                    };
                    let inner: String = self.buf[..i].to_string();
                    self.buf.drain(..i + GEMMA_CALL_CLOSE.len());
                    self.state = State::Scan;
                    match self.parse_gemma_call(&inner) {
                        Some(call) => out.push(Piece::Call(call)),
                        // malformed: surfaced verbatim, tags included, stream continues.
                        None => emit_content(
                            &mut out,
                            format!("{GEMMA_CALL_OPEN}{inner}{GEMMA_CALL_CLOSE}"),
                        ),
                    }
                    continue;
                }
                State::Dsv4Call => {
                    let Some(i) = self.buf.find(DSV4_CLOSE) else {
                        break;
                    };
                    let inner: String = self.buf[..i].to_string();
                    self.buf.drain(..i + DSV4_CLOSE.len());
                    self.state = State::Scan;
                    match self.parse_dsv4_calls(&inner) {
                        // one `<｜DSML｜tool_calls>` block yields one-or-more OpenAI calls.
                        Some(calls) if !calls.is_empty() => {
                            for c in calls {
                                out.push(Piece::Call(c));
                            }
                        }
                        // malformed / empty: surfaced verbatim, tags included, stream continues.
                        _ => emit_content(&mut out, format!("{DSV4_OPEN}{inner}{DSV4_CLOSE}")),
                    }
                    continue;
                }
            }
        }
        out
    }

    /// End of generation: flush any held-back text. An unterminated `<tool_call>` block is
    /// surfaced raw (opening tag restored) — same malformed policy. A generation that ended
    /// still inside the think segment flushes the tail as reasoning (never-closed `</think>`).
    pub fn finish(&mut self) -> Vec<Piece> {
        let mut out = Vec::new();
        if !self.buf.is_empty() {
            let tail = std::mem::take(&mut self.buf);
            match self.state {
                State::Prethink => self.emit_reasoning(&mut out, tail),
                State::InCall => emit_content(&mut out, format!("{OPEN}{tail}")),
                // generation died inside a thought channel: the tail (incl. a held-back
                // syntax newline) is reasoning, never content.
                State::GemmaThought => {
                    let t = tail.strip_suffix('\n').unwrap_or(&tail);
                    self.emit_reasoning(&mut out, t.to_string());
                }
                // died mid-label: the partial channel name is syntax, not output.
                State::GemmaLabel => {}
                // unterminated call span at end-of-generation: surfaced raw (opening tag
                // restored), same malformed policy as the qwen arm.
                State::GemmaCall => emit_content(&mut out, format!("{GEMMA_CALL_OPEN}{tail}")),
                // dsv4 unterminated `<｜DSML｜tool_calls>` block: surfaced raw (open restored).
                State::Dsv4Call => emit_content(&mut out, format!("{DSV4_OPEN}{tail}")),
                _ => emit_content(&mut out, tail),
            }
        }
        self.state = State::Scan;
        out
    }

    pub fn n_calls(&self) -> usize {
        self.n_calls
    }

    /// Think-segment text -> a Reasoning piece. ALWAYS delivered.
    ///
    /// This parser used to carry an `include_reasoning` flag that dropped the separated think
    /// text on the floor. It is gone by owner ruling (2026-08-23): reasoning tokens are output
    /// tokens and are billed as output, so a request that generated them and then withheld them
    /// charged the customer for output we never sent. The two flags that reached this drop —
    /// `include_reasoning:false` and `reasoning.exclude:true` — now turn reasoning OFF upstream
    /// in `parse_think` instead, so the cheaper request the caller asked for is the one they get.
    /// The capability is DELETED rather than left unreachable so it cannot be rewired.
    fn emit_reasoning(&self, out: &mut Vec<Piece>, text: String) {
        if text.is_empty() {
            return;
        }
        if let Some(Piece::Reasoning(prev)) = out.last_mut() {
            prev.push_str(&text);
            return;
        }
        out.push(Piece::Reasoning(text));
    }

    /// Parse one block body (the text between the `<tool_call>` tags). None = malformed.
    fn parse_block(&mut self, inner: &str) -> Option<ParsedToolCall> {
        let s = inner.trim();
        let rest = s.strip_prefix("<function=")?;
        let gt = rest.find('>')?;
        let name = &rest[..gt];
        if name.is_empty() || name.contains(['<', '>', '\n']) {
            return None;
        }
        let mut body = rest[gt + 1..].strip_suffix("</function>")?;
        let mut args = serde_json::Map::new();
        loop {
            let t = body.trim_start();
            if t.is_empty() {
                break;
            }
            let r = t.strip_prefix("<parameter=")?;
            let gt = r.find('>')?;
            let key = &r[..gt];
            if key.is_empty() || key.contains(['<', '>', '\n']) {
                return None;
            }
            // rendered form is `<parameter=k>\n{value}\n</parameter>` — the delimiter
            // newlines belong to the syntax, inner newlines belong to the value.
            let after = &r[gt + 1..];
            let after = after.strip_prefix('\n').unwrap_or(after);
            let end = after.find("</parameter>")?;
            let raw = after[..end].strip_suffix('\n').unwrap_or(&after[..end]);
            args.insert(key.to_string(), self.coerce(name, key, raw));
            body = &after[end + "</parameter>".len()..];
        }
        let arguments = serde_json::to_string(&serde_json::Value::Object(args)).ok()?;
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
            name: name.to_string(),
            arguments,
        })
    }

    /// Parse one gemma tooluse call span body (`call:NAME{args}`) into an OpenAI call. None =
    /// malformed (surfaced verbatim). The `{args}` are the compact gemma dialect (bare keys,
    /// `<|"|>`-wrapped strings, bare numbers/true/false/None, nested {}/[]).
    fn parse_gemma_call(&mut self, inner: &str) -> Option<ParsedToolCall> {
        let s = inner.trim();
        let rest = s.strip_prefix("call:")?;
        let brace = rest.find('{')?;
        let name = rest[..brace].trim();
        if name.is_empty() || name.contains(['<', '>', '\n', '{', '}']) {
            return None;
        }
        let (value, consumed) = parse_gemma_value(&rest[brace..])?;
        // trailing bytes after the object mean a malformed span.
        if rest[brace..][consumed..].trim() != "" {
            return None;
        }
        let obj = match value {
            serde_json::Value::Object(_) => value,
            _ => return None,
        };
        let arguments = serde_json::to_string(&obj).ok()?;
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
            name: name.to_string(),
            arguments,
        })
    }

    /// Parse one deepseek-v4 `<｜DSML｜tool_calls>` block body (between the tool_calls tags)
    /// into one-or-more OpenAI calls — a strict port of encoding_dsv4 `parse_tool_calls`
    /// (E:630-684). Each `<｜DSML｜invoke name="N">` carries `<｜DSML｜parameter name="K"
    /// string="true|false">V</｜DSML｜parameter>` lines: `string="true"` values are raw strings,
    /// `string="false"` values are JSON (number/bool/array/object, embedded raw). None =
    /// malformed (surfaced verbatim). `schemas` is unused — the wire is self-describing.
    fn parse_dsv4_calls(&mut self, inner: &str) -> Option<Vec<ParsedToolCall>> {
        let mut calls = Vec::new();
        let mut rest = inner;
        loop {
            let Some(iv) = rest.find(DSV4_INVOKE) else {
                // no more invokes: the remainder must be the wrapper's whitespace only.
                if rest.trim().is_empty() {
                    break;
                }
                return None;
            };
            // text before the invoke open must be whitespace (the joining/leading `\n`).
            if !rest[..iv].trim().is_empty() {
                return None;
            }
            rest = &rest[iv + DSV4_INVOKE.len()..];
            let name_end = rest.find("\">")?;
            let name = rest[..name_end].to_string();
            if name.is_empty() || name.contains(['<', '>', '\n']) {
                return None;
            }
            rest = &rest[name_end + 2..]; // past `">`
            let mut args = serde_json::Map::new();
            loop {
                let t = rest.trim_start_matches('\n');
                if let Some(after) = t.strip_prefix(DSV4_INVOKE_END) {
                    rest = after;
                    break;
                }
                let p = t.strip_prefix(DSV4_PARAM)?;
                let n_end = p.find("\" string=\"")?;
                let key = p[..n_end].to_string();
                if key.is_empty() || key.contains(['<', '>', '\n']) {
                    return None;
                }
                let after_key = &p[n_end + "\" string=\"".len()..];
                let flag_end = after_key.find("\">")?;
                let flag = &after_key[..flag_end];
                let after_flag = &after_key[flag_end + 2..];
                let v_end = after_flag.find(DSV4_PARAM_END)?;
                let value = &after_flag[..v_end];
                let coerced = match flag {
                    "true" => serde_json::Value::String(value.to_string()),
                    "false" => serde_json::from_str::<serde_json::Value>(value)
                        .unwrap_or_else(|_| serde_json::Value::String(value.to_string())),
                    _ => return None,
                };
                if args.contains_key(&key) {
                    return None; // duplicate parameter name (E:673-674)
                }
                args.insert(key, coerced);
                rest = &after_flag[v_end + DSV4_PARAM_END.len()..];
            }
            let arguments = serde_json::to_string(&serde_json::Value::Object(args)).ok()?;
            let id = format!(
                "call_{:016x}",
                fnv1a64(&[
                    &self.n_calls.to_le_bytes(),
                    name.as_bytes(),
                    arguments.as_bytes(),
                ])
            );
            self.n_calls += 1;
            calls.push(ParsedToolCall {
                id,
                name,
                arguments,
            });
        }
        if calls.is_empty() {
            return None;
        }
        Some(calls)
    }

    /// Coercion law: a parameter whose declared schema type is non-"string" is parsed as
    /// JSON (integer/number/boolean/object/array); parse failure or a declared/unknown
    /// string type keeps the raw text.
    fn coerce(&self, func: &str, param: &str, raw: &str) -> serde_json::Value {
        let declared = self
            .schemas
            .get(func)
            .and_then(|m| m.get(param))
            .map(String::as_str);
        match declared {
            Some("string") | None => serde_json::Value::String(raw.to_string()),
            // Qwen sometimes spells booleans the way Python does (`True` / `False`) even
            // though the tool template asks for JSON. OpenRouter's Draft-7 validator then
            // sees a string because the generic JSON parse below correctly rejects that
            // spelling. The declared schema removes any ambiguity: normalize only a
            // boolean-declared parameter, while still leaving every other failed coercion
            // visible as a string for downstream validation.
            Some("boolean") if raw.trim().eq_ignore_ascii_case("true") => {
                serde_json::Value::Bool(true)
            }
            Some("boolean") if raw.trim().eq_ignore_ascii_case("false") => {
                serde_json::Value::Bool(false)
            }
            Some(_) => serde_json::from_str::<serde_json::Value>(raw.trim())
                .unwrap_or_else(|_| serde_json::Value::String(raw.to_string())),
        }
    }
}

/// Coalesce adjacent content pieces (chunk boundaries are not part of any contract, but
/// fewer SSE events is strictly kinder to clients).
fn emit_content(out: &mut Vec<Piece>, text: String) {
    if text.is_empty() {
        return;
    }
    if let Some(Piece::Content(prev)) = out.last_mut() {
        prev.push_str(&text);
        return;
    }
    out.push(Piece::Content(text));
}

/// Nesting ceiling for the gemma dialect value grammar. The three parse functions below
/// are MUTUALLY RECURSIVE over MODEL OUTPUT: without a cap, a model emitting `[[[[…` (one
/// stack frame per byte) overflows the thread stack and aborts the whole process — a
/// remote crash reachable through any gemma-tools request (hermes finding, fixed
/// 2026-08-19). 64 is far past any real tool schema (observed calls nest 2-3 deep) and
/// far under any stack limit. Over-depth parses as None = the standard malformed-span
/// policy: the block surfaces VERBATIM as content, the stream continues, nothing crashes.
const GEMMA_MAX_DEPTH: usize = 64;

/// Parse one gemma dialect value at the start of `s`; returns (value, bytes consumed).
/// Grammar: `<|"|>...<|"|>` string · `{k:v,...}` object (bare keys) · `[v,...]` array ·
/// bare `true`/`false`/`None`/number, else a bare string.
fn parse_gemma_value(s: &str) -> Option<(serde_json::Value, usize)> {
    parse_gemma_value_at(s, 0)
}

fn parse_gemma_value_at(s: &str, depth: usize) -> Option<(serde_json::Value, usize)> {
    if depth >= GEMMA_MAX_DEPTH {
        return None; // over-depth = malformed: surfaced verbatim, never a stack overflow
    }
    if let Some(rest) = s.strip_prefix(GEMMA_DQ) {
        let close = rest.find(GEMMA_DQ)?;
        let consumed = GEMMA_DQ.len() + close + GEMMA_DQ.len();
        return Some((
            serde_json::Value::String(rest[..close].to_string()),
            consumed,
        ));
    }
    match s.as_bytes().first()? {
        b'{' => parse_gemma_object(s, depth),
        b'[' => parse_gemma_array(s, depth),
        _ => parse_gemma_bare(s),
    }
}

fn parse_gemma_object(s: &str, depth: usize) -> Option<(serde_json::Value, usize)> {
    let mut map = serde_json::Map::new();
    let mut i = 1; // past '{'
    if s.get(i..)?.starts_with('}') {
        return Some((serde_json::Value::Object(map), i + 1));
    }
    loop {
        let colon = s.get(i..)?.find(':')? + i;
        let key = s[i..colon].trim();
        if key.is_empty() || key.contains(['{', '}', '[', ']', ',']) {
            return None;
        }
        i = colon + 1;
        let (val, c) = parse_gemma_value_at(s.get(i..)?, depth + 1)?;
        i += c;
        map.insert(key.to_string(), val);
        match s.as_bytes().get(i)? {
            b',' => i += 1,
            b'}' => return Some((serde_json::Value::Object(map), i + 1)),
            _ => return None,
        }
    }
}

fn parse_gemma_array(s: &str, depth: usize) -> Option<(serde_json::Value, usize)> {
    let mut arr = Vec::new();
    let mut i = 1; // past '['
    if s.get(i..)?.starts_with(']') {
        return Some((serde_json::Value::Array(arr), i + 1));
    }
    loop {
        let (val, c) = parse_gemma_value_at(s.get(i..)?, depth + 1)?;
        i += c;
        arr.push(val);
        match s.as_bytes().get(i)? {
            b',' => i += 1,
            b']' => return Some((serde_json::Value::Array(arr), i + 1)),
            _ => return None,
        }
    }
}

/// A bare token runs to the next structural delimiter (`,`/`}`/`]`); numbers parse as JSON,
/// `true`/`false`/`None` map to bool/null, anything else stays a string.
fn parse_gemma_bare(s: &str) -> Option<(serde_json::Value, usize)> {
    let end = s.find([',', '}', ']']).unwrap_or(s.len());
    let token = s[..end].trim();
    let value = match token {
        "true" => serde_json::Value::Bool(true),
        "false" => serde_json::Value::Bool(false),
        "None" | "null" => serde_json::Value::Null,
        _ => serde_json::from_str::<serde_json::Value>(token)
            .ok()
            .filter(serde_json::Value::is_number)
            .unwrap_or_else(|| serde_json::Value::String(token.to_string())),
    };
    Some((value, end))
}

fn fnv1a64(parts: &[&[u8]]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for part in parts {
        for &b in *part {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    fn weather_schema() -> HashMap<String, HashMap<String, String>> {
        let mut params = HashMap::new();
        params.insert("city".to_string(), "string".to_string());
        params.insert("days".to_string(), "integer".to_string());
        params.insert("metric".to_string(), "boolean".to_string());
        let mut m = HashMap::new();
        m.insert("get_weather".to_string(), params);
        m
    }

    const EMISSION: &str = "I'll check.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\n\
Paris\n</parameter>\n<parameter=days>\n3\n</parameter>\n<parameter=metric>\ntrue\n</parameter>\n\
</function>\n</tool_call>";

    fn reassemble(pieces: &[Piece]) -> (String, Vec<ParsedToolCall>) {
        let (content, reasoning, calls) = reassemble3(pieces);
        assert!(reasoning.is_empty(), "unexpected reasoning: {reasoning:?}");
        (content, calls)
    }

    fn reassemble3(pieces: &[Piece]) -> (String, String, Vec<ParsedToolCall>) {
        let mut content = String::new();
        let mut reasoning = String::new();
        let mut calls = Vec::new();
        for p in pieces {
            match p {
                Piece::Content(t) => content.push_str(t),
                Piece::Reasoning(t) => reasoning.push_str(t),
                Piece::Call(c) => calls.push(c.clone()),
            }
        }
        (content, reasoning, calls)
    }

    #[test]
    fn parses_call_with_schema_coercion() {
        let mut p = ToolStreamParser::new(weather_schema(), false);
        let mut pieces = p.push(EMISSION);
        pieces.extend(p.finish());
        let (content, calls) = reassemble(&pieces);
        assert_eq!(content, "I'll check.\n\n");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(
            calls[0].arguments,
            r#"{"city":"Paris","days":3,"metric":true}"#
        );
        assert!(calls[0].id.starts_with("call_"));
    }

    #[test]
    fn boolean_schema_normalizes_python_style_model_literals() {
        let text = "<tool_call>\n<function=get_weather>\n<parameter=metric>\nTrue\n\
</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=get_weather>\n\
<parameter=metric>\nFalse\n</parameter>\n</function>\n</tool_call>";
        let mut parser = ToolStreamParser::new(weather_schema(), false);
        let mut pieces = parser.push(text);
        pieces.extend(parser.finish());
        let (_, calls) = reassemble(&pieces);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].arguments, r#"{"metric":true}"#);
        assert_eq!(calls[1].arguments, r#"{"metric":false}"#);
    }

    #[test]
    fn char_by_char_deltas_produce_the_same_result() {
        let mut p = ToolStreamParser::new(weather_schema(), false);
        let mut pieces: Vec<Piece> = Vec::new();
        for ch in EMISSION.chars() {
            pieces.extend(p.push(&ch.to_string()));
        }
        pieces.extend(p.finish());
        let (content, calls) = reassemble(&pieces);
        assert_eq!(content, "I'll check.\n\n");
        assert_eq!(calls.len(), 1);
        assert_eq!(
            calls[0].arguments,
            r#"{"city":"Paris","days":3,"metric":true}"#
        );
    }

    #[test]
    fn think_gate_routes_think_text_to_reasoning_not_content() {
        // gap-scan F13: think-segment text is the REASONING field, never content — a
        // `<tool_call>` mentioned while reasoning is not a call, the tag + separator
        // newlines are syntax, and post-think calls still parse.
        let mut p = ToolStreamParser::new(weather_schema(), true);
        let text = "planning a <tool_call> here...</think>\n\n<tool_call>\n\
<function=get_weather>\n<parameter=city>\nOslo\n</parameter>\n</function>\n</tool_call>";
        let mut pieces = p.push(text);
        pieces.extend(p.finish());
        let (content, reasoning, calls) = reassemble3(&pieces);
        assert_eq!(reasoning, "planning a <tool_call> here...");
        assert_eq!(content, "");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments, r#"{"city":"Oslo"}"#);
    }

    #[test]
    fn reasoning_only_mode_splits_think_from_content_char_by_char() {
        // non-tools chat on a think-open model: reasoning/content split, post-think
        // text NEVER scanned for tool tags.
        let text = "step one\nstep two</think>\n\nAnswer with a <tool_call> literal.";
        for chunked in [false, true] {
            let mut p = ToolStreamParser::reasoning_only();
            let mut pieces = Vec::new();
            if chunked {
                for ch in text.chars() {
                    pieces.extend(p.push(&ch.to_string()));
                }
            } else {
                pieces.extend(p.push(text));
            }
            pieces.extend(p.finish());
            let (content, reasoning, calls) = reassemble3(&pieces);
            assert_eq!(reasoning, "step one\nstep two", "chunked={chunked}");
            assert_eq!(
                content, "Answer with a <tool_call> literal.",
                "chunked={chunked}"
            );
            assert!(calls.is_empty());
        }
    }

    #[test]
    fn reasoning_is_always_delivered_there_is_no_suppression_path() {
        // OWNER RULING 2026-08-23: reasoning tokens are output tokens, billed as output, so
        // withholding them charged for output we never sent. The parser's `include_reasoning`
        // drop is DELETED, not merely unreachable — every dialect that separates reasoning must
        // hand it to the caller. The two flags that used to reach the drop
        // (`include_reasoning:false`, `reasoning.exclude:true`) now turn reasoning OFF in
        // `parse_think`, so a caller who does not want to pay for it does not generate it.
        //
        // This test replaces `include_reasoning_false_drops_think_text` and
        // `dsv4_include_reasoning_false_drops_think_text`, which asserted the banned behaviour.
        let mut p = ToolStreamParser::reasoning_only();
        let mut pieces = p.push("a plan</think>\n\nvisible answer");
        pieces.extend(p.finish());
        let (content, reasoning, calls) = reassemble3(&pieces);
        assert_eq!(reasoning, "a plan", "reasoning must reach the caller");
        assert_eq!(content, "visible answer");
        assert!(calls.is_empty());
        // same for the dsv4 and gemma splitters: no dialect may hide it.
        let mut p = ToolStreamParser::dsv4(true);
        let mut pieces = p.push("a plan</think>visible answer");
        pieces.extend(p.finish());
        let (content, reasoning, _) = reassemble3(&pieces);
        assert_eq!(reasoning, "a plan");
        assert_eq!(content, "visible answer");
        let mut p = ToolStreamParser::gemma_thought();
        let mut pieces = p.push("<|channel>thought\na plan\n<channel|>visible");
        pieces.extend(p.finish());
        let (content, reasoning, _) = reassemble3(&pieces);
        assert_eq!(reasoning, "a plan");
        assert_eq!(content, "visible");
    }

    #[test]
    fn unclosed_think_flushes_as_reasoning() {
        // generation died inside the think segment: the tail is reasoning, not content.
        let mut p = ToolStreamParser::reasoning_only();
        let mut pieces = p.push("half a thought");
        pieces.extend(p.finish());
        let (content, reasoning, _) = reassemble3(&pieces);
        assert_eq!(reasoning, "half a thought");
        assert_eq!(content, "");
    }

    #[test]
    fn malformed_block_is_surfaced_verbatim() {
        // broken JSON-ish emission: no <function= wrapper at all.
        let text =
            "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {broken\n</tool_call>done";
        let mut p = ToolStreamParser::new(weather_schema(), false);
        let mut pieces = p.push(text);
        pieces.extend(p.finish());
        let (content, calls) = reassemble(&pieces);
        assert_eq!(content, text); // byte-exact surfacing, tags included
        assert!(calls.is_empty());
    }

    #[test]
    fn unterminated_block_flushes_raw_on_finish() {
        let mut p = ToolStreamParser::new(weather_schema(), false);
        let mut pieces = p.push("<tool_call>\n<function=get_weather>\n<parameter=city>\nParis");
        pieces.extend(p.finish());
        let (content, calls) = reassemble(&pieces);
        assert_eq!(
            content,
            "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis"
        );
        assert!(calls.is_empty());
    }

    #[test]
    fn two_calls_and_multiline_string_values() {
        let text = "<tool_call>\n<function=get_weather>\n<parameter=city>\nline one\nline two\n\
</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=get_weather>\n<parameter=days>\n\
not-a-number\n</parameter>\n</function>\n</tool_call>";
        let mut p = ToolStreamParser::new(weather_schema(), false);
        let mut pieces = p.push(text);
        pieces.extend(p.finish());
        let (content, calls) = reassemble(&pieces);
        assert_eq!(content, "\n"); // the separator newline between the two blocks
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].arguments, r#"{"city":"line one\nline two"}"#);
        // integer-declared param that fails JSON parse falls back to the raw string.
        assert_eq!(calls[1].arguments, r#"{"days":"not-a-number"}"#);
        assert_ne!(calls[0].id, calls[1].id);
    }

    #[test]
    fn gemma_thought_channel_splits_reasoning_from_content_char_by_char() {
        // the gemma4 dialect (lane/gemma4-serve-gaps): `<|channel>thought\n{t}\n<channel|>`
        // routes to reasoning; tags/label/bracketing newlines are syntax; content follows
        // directly. Char-by-char must agree with one-shot (streaming holdback law).
        let text = "<|channel>thought\nThe user wants ok.\nSo reply ok.\n<channel|>ok";
        for chunked in [false, true] {
            let mut p = ToolStreamParser::gemma_thought();
            let mut pieces = Vec::new();
            if chunked {
                for ch in text.chars() {
                    pieces.extend(p.push(&ch.to_string()));
                }
            } else {
                pieces.extend(p.push(text));
            }
            pieces.extend(p.finish());
            let (content, reasoning, calls) = reassemble3(&pieces);
            assert_eq!(
                reasoning, "The user wants ok.\nSo reply ok.",
                "chunked={chunked}"
            );
            assert_eq!(content, "ok", "chunked={chunked}");
            assert!(calls.is_empty());
        }
    }

    #[test]
    fn gemma_content_before_and_between_channels() {
        // channels can open at ANY stream position (the template's strip_thinking law) —
        // the closed-channel prompt still lets the model open one mid-stream (observed
        // live on the 12B QAT: think-smoke receipt, content='ok<turn|>…thought…').
        let text = "ok<|channel>thought\nreconsidering\n<channel|> more";
        let mut p = ToolStreamParser::gemma_thought();
        let mut pieces = p.push(text);
        pieces.extend(p.finish());
        let (content, reasoning, _) = reassemble3(&pieces);
        assert_eq!(reasoning, "reconsidering");
        assert_eq!(content, "ok more");
    }

    #[test]
    fn gemma_unclosed_thought_flushes_as_reasoning_and_excludes_reasoning_drops() {
        // budget died inside the channel: tail is reasoning, never content.
        let mut p = ToolStreamParser::gemma_thought();
        let mut pieces = p.push("<|channel>thought\nhalf a tho");
        pieces.extend(p.finish());
        let (content, reasoning, _) = reassemble3(&pieces);
        assert_eq!(reasoning, "half a tho");
        assert_eq!(content, "");
    }

    #[test]
    fn gemma_partial_open_tag_holdback_never_loses_bytes() {
        // a `<|chan` that never becomes the tag must still be emitted as content.
        let mut p = ToolStreamParser::gemma_thought();
        let mut pieces = p.push("a <|chan");
        pieces.extend(p.push("nel of prose"));
        pieces.extend(p.finish());
        let (content, reasoning, _) = reassemble3(&pieces);
        assert_eq!(content, "a <|channel of prose");
        assert_eq!(reasoning, "");
    }

    // ---- gemma4 tooluse dialect parser (lane/gemma4-tools) --------------------------------

    #[test]
    fn gemma_tools_parses_a_call_and_never_leaks_the_span() {
        let text = "<|tool_call>call:get_weather{location:<|\"|>Paris<|\"|>,\
unit:<|\"|>celsius<|\"|>}<tool_call|>";
        for chunked in [false, true] {
            let mut p = ToolStreamParser::gemma_tools();
            let mut pieces = Vec::new();
            if chunked {
                for ch in text.chars() {
                    pieces.extend(p.push(&ch.to_string()));
                }
            } else {
                pieces.extend(p.push(text));
            }
            pieces.extend(p.finish());
            let (content, reasoning, calls) = reassemble3(&pieces);
            assert_eq!(content, "", "chunked={chunked}");
            assert_eq!(reasoning, "", "chunked={chunked}");
            assert_eq!(calls.len(), 1, "chunked={chunked}");
            assert_eq!(calls[0].name, "get_weather");
            assert_eq!(
                calls[0].arguments, r#"{"location":"Paris","unit":"celsius"}"#,
                "chunked={chunked}"
            );
            assert!(calls[0].id.starts_with("call_"));
            assert_eq!(p.n_calls(), 1);
        }
    }

    #[test]
    fn gemma_tools_splits_thought_content_and_call() {
        // a thought channel, then visible content, then a call — the three routes must
        // separate, and the tags must never appear anywhere.
        let text = "<|channel>thought\nplanning the call\n<channel|>On it.\
<|tool_call>call:shell{command:[<|\"|>echo<|\"|>,<|\"|>hi<|\"|>],timeout_ms:5000}<tool_call|>";
        for chunked in [false, true] {
            let mut p = ToolStreamParser::gemma_tools();
            let mut pieces = Vec::new();
            if chunked {
                for ch in text.chars() {
                    pieces.extend(p.push(&ch.to_string()));
                }
            } else {
                pieces.extend(p.push(text));
            }
            pieces.extend(p.finish());
            let (content, reasoning, calls) = reassemble3(&pieces);
            assert_eq!(reasoning, "planning the call", "chunked={chunked}");
            assert_eq!(content, "On it.", "chunked={chunked}");
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0].name, "shell");
            assert_eq!(
                calls[0].arguments,
                r#"{"command":["echo","hi"],"timeout_ms":5000}"#
            );
        }
    }

    #[test]
    fn gemma_tools_coerces_typed_arguments() {
        // nested object + bool + null (None) + a string carrying braces/commas/colons.
        let text = "<|tool_call>call:book{traveler:{name:<|\"|>Avi<|\"|>,age:30},\
flexible:true,note:<|\"|>a{b,c}:d<|\"|>,workdir:None}<tool_call|>";
        let mut p = ToolStreamParser::gemma_tools();
        let mut pieces = p.push(text);
        pieces.extend(p.finish());
        let (_c, _r, calls) = reassemble3(&pieces);
        assert_eq!(calls.len(), 1);
        assert_eq!(
            calls[0].arguments,
            r#"{"traveler":{"name":"Avi","age":30},"flexible":true,"note":"a{b,c}:d","workdir":null}"#
        );
    }

    #[test]
    fn gemma_deep_nesting_degrades_to_content_without_stack_growth() {
        // A model emitting `[[[[…` used to recurse one stack frame per byte through
        // parse_gemma_value ↔ parse_gemma_array and abort the process. Run the
        // pathological input on a deliberately SMALL stack (1 MiB — the un-capped parse
        // needed ~2 debug frames per byte, tens of MiB for these inputs) to pin that
        // recursion is depth-capped, and assert the over-depth span degrades to the
        // malformed policy: verbatim content, no call, no crash.
        let handle = std::thread::Builder::new()
            .stack_size(1024 * 1024)
            .spawn(|| {
                for payload in [
                    "[".repeat(100_000),
                    "{".repeat(100_000),
                    "{a:[{b:[".repeat(25_000),
                ] {
                    let text = format!("<|tool_call>call:f{{x:{payload}}}<tool_call|>");
                    let mut p = ToolStreamParser::gemma_tools();
                    let mut pieces = p.push(&text);
                    pieces.extend(p.finish());
                    let (content, _r, calls) = reassemble3(&pieces);
                    assert!(calls.is_empty());
                    assert_eq!(content, text); // tags included, byte-exact
                }
            })
            .expect("spawn");
        handle
            .join()
            .expect("deep-nest parse must not overflow the stack");
    }

    #[test]
    fn gemma_nesting_below_the_cap_still_parses() {
        // 8 levels — beyond any observed tool schema, comfortably under GEMMA_MAX_DEPTH.
        let inner = "[[[[[[[[1]]]]]]]]";
        let text = format!("<|tool_call>call:f{{x:{inner}}}<tool_call|>");
        let mut p = ToolStreamParser::gemma_tools();
        let mut pieces = p.push(&text);
        pieces.extend(p.finish());
        let (_c, _r, calls) = reassemble3(&pieces);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments, r#"{"x":[[[[[[[[1]]]]]]]]}"#);
    }

    #[test]
    fn gemma_tools_malformed_span_surfaces_verbatim() {
        let text = "<|tool_call>not a real call<tool_call|>done";
        let mut p = ToolStreamParser::gemma_tools();
        let mut pieces = p.push(text);
        pieces.extend(p.finish());
        let (content, _r, calls) = reassemble3(&pieces);
        assert!(calls.is_empty());
        assert_eq!(content, text); // tags included, byte-exact
    }

    #[test]
    fn gemma_tools_unterminated_call_flushes_raw() {
        let mut p = ToolStreamParser::gemma_tools();
        let mut pieces = p.push("<|tool_call>call:get_weather{location:<|\"|>Par");
        pieces.extend(p.finish());
        let (content, _r, calls) = reassemble3(&pieces);
        assert!(calls.is_empty());
        assert_eq!(content, "<|tool_call>call:get_weather{location:<|\"|>Par");
    }

    #[test]
    fn gemma_tools_plain_content_passes_through() {
        // a request that never calls a tool: pure content, no holdback loss, no false call.
        let mut p = ToolStreamParser::gemma_tools();
        let mut pieces = p.push("The weather in Paris is 21C and clear.");
        pieces.extend(p.finish());
        let (content, reasoning, calls) = reassemble3(&pieces);
        assert_eq!(content, "The weather in Paris is 21C and clear.");
        assert!(reasoning.is_empty());
        assert!(calls.is_empty());
    }

    #[test]
    fn gemma_tools_partial_open_tag_holdback_never_loses_bytes() {
        // a `<|to` that never becomes a tag must still surface as content.
        let mut p = ToolStreamParser::gemma_tools();
        let mut pieces = p.push("cost <|to");
        pieces.extend(p.push("ken budget"));
        pieces.extend(p.finish());
        let (content, _r, calls) = reassemble3(&pieces);
        assert_eq!(content, "cost <|token budget");
        assert!(calls.is_empty());
    }

    #[test]
    fn partial_tag_holdback_never_loses_bytes() {
        // a "<tool" that never becomes a tag must still be emitted.
        let mut p = ToolStreamParser::new(HashMap::new(), false);
        let mut pieces = p.push("a <tool");
        pieces.extend(p.push("box holds bytes"));
        pieces.extend(p.finish());
        let (content, calls) = reassemble(&pieces);
        assert_eq!(content, "a <toolbox holds bytes");
        assert!(calls.is_empty());
    }

    // ---- deepseek-v4 (encoding_dsv4) dialect parser (lane/dsv4-template) ------------------
    // The DSML wire: `{reasoning}</think>{content}\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke
    // name="N">\n<｜DSML｜parameter name="K" string="true|false">V</｜DSML｜parameter>\n
    // </｜DSML｜invoke>\n</｜DSML｜tool_calls>`. `｜` is U+FF5C.

    const DS_OPEN: &str = "<\u{ff5c}DSML\u{ff5c}tool_calls>";
    const DS_CLOSE: &str = "</\u{ff5c}DSML\u{ff5c}tool_calls>";
    const DS_INV: &str = "<\u{ff5c}DSML\u{ff5c}invoke name=\"";
    const DS_PAR: &str = "<\u{ff5c}DSML\u{ff5c}parameter name=\"";
    const DS_PAR_END: &str = "</\u{ff5c}DSML\u{ff5c}parameter>";
    const DS_INV_END: &str = "</\u{ff5c}DSML\u{ff5c}invoke>";

    /// Build the exact DSML block for one call with string params (matches encoding_dsv4).
    fn ds_call_block(name: &str, params: &[(&str, &str, bool)]) -> String {
        let mut s = String::from(DS_OPEN);
        s.push_str(&format!("\n{DS_INV}{name}\">\n"));
        for (i, (k, v, is_str)) in params.iter().enumerate() {
            if i > 0 {
                s.push('\n');
            }
            s.push_str(&format!(
                "{DS_PAR}{k}\" string=\"{}\">{v}{DS_PAR_END}",
                if *is_str { "true" } else { "false" }
            ));
        }
        s.push_str(&format!("\n{DS_INV_END}\n{DS_CLOSE}"));
        s
    }

    #[test]
    fn dsv4_parses_a_call_in_thinking_mode_split_from_reasoning() {
        // real wire from artifact test_output_1: reasoning then a get_weather call. `</think>`
        // routes reasoning; content is empty; the `\n\n` before the block is syntax.
        let block = ds_call_block(
            "get_weather",
            &[("location", "Beijing", true), ("unit", "celsius", true)],
        );
        let text = format!(
            "The user wants the weather in Beijing. I should use get_weather.</think>\n\n{block}"
        );
        for chunked in [false, true] {
            let mut p = ToolStreamParser::dsv4(true);
            let mut pieces = Vec::new();
            if chunked {
                for ch in text.chars() {
                    pieces.extend(p.push(&ch.to_string()));
                }
            } else {
                pieces.extend(p.push(&text));
            }
            pieces.extend(p.finish());
            let (content, reasoning, calls) = reassemble3(&pieces);
            assert_eq!(
                reasoning, "The user wants the weather in Beijing. I should use get_weather.",
                "chunked={chunked}"
            );
            assert_eq!(content, "", "chunked={chunked}");
            assert_eq!(calls.len(), 1, "chunked={chunked}");
            assert_eq!(calls[0].name, "get_weather");
            assert_eq!(
                calls[0].arguments, r#"{"location":"Beijing","unit":"celsius"}"#,
                "chunked={chunked}"
            );
            assert!(calls[0].id.starts_with("call_"));
        }
    }

    #[test]
    fn dsv4_content_then_call_no_reasoning_in_chat_mode() {
        // chat mode (skip_think=false): no reasoning; content precedes the call and the `\n\n`
        // separator is stripped.
        let block = ds_call_block("get_weather", &[("location", "Oslo", true)]);
        let text = format!("Let me check.\n\n{block}");
        let mut p = ToolStreamParser::dsv4(false);
        let mut pieces = p.push(&text);
        pieces.extend(p.finish());
        let (content, reasoning, calls) = reassemble3(&pieces);
        assert_eq!(content, "Let me check.");
        assert!(reasoning.is_empty());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments, r#"{"location":"Oslo"}"#);
    }

    #[test]
    fn dsv4_multiple_invokes_in_one_block_yield_multiple_calls() {
        // one `<｜DSML｜tool_calls>` block with two invokes -> two OpenAI calls, distinct ids.
        let mut block = String::from(DS_OPEN);
        block.push_str(&format!(
            "\n{DS_INV}a\">\n{DS_PAR}x\" string=\"false\">1{DS_PAR_END}\n{DS_INV_END}"
        ));
        block.push_str(&format!(
            "\n{DS_INV}b\">\n{DS_PAR}y\" string=\"true\">hi{DS_PAR_END}\n{DS_INV_END}"
        ));
        block.push_str(&format!("\n{DS_CLOSE}"));
        let text = format!("</think>{block}");
        let mut p = ToolStreamParser::dsv4(true);
        let mut pieces = p.push(&text);
        pieces.extend(p.finish());
        let (_c, _r, calls) = reassemble3(&pieces);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "a");
        assert_eq!(calls[0].arguments, r#"{"x":1}"#);
        assert_eq!(calls[1].name, "b");
        assert_eq!(calls[1].arguments, r#"{"y":"hi"}"#);
        assert_ne!(calls[0].id, calls[1].id);
    }

    #[test]
    fn dsv4_typed_args_string_false_coerces_json() {
        // string="false" values are JSON: number, bool, array, object embedded raw.
        let mut block = String::from(DS_OPEN);
        block.push_str(&format!("\n{DS_INV}f\">"));
        block.push_str(&format!("\n{DS_PAR}n\" string=\"false\">3{DS_PAR_END}"));
        block.push_str(&format!("\n{DS_PAR}b\" string=\"false\">true{DS_PAR_END}"));
        block.push_str(&format!(
            "\n{DS_PAR}arr\" string=\"false\">[1, 2]{DS_PAR_END}"
        ));
        block.push_str(&format!("\n{DS_PAR}s\" string=\"true\">plain{DS_PAR_END}"));
        block.push_str(&format!("\n{DS_INV_END}\n{DS_CLOSE}"));
        let mut p = ToolStreamParser::dsv4(false);
        let mut pieces = p.push(&block);
        pieces.extend(p.finish());
        let (_c, _r, calls) = reassemble3(&pieces);
        assert_eq!(calls.len(), 1);
        assert_eq!(
            calls[0].arguments,
            r#"{"n":3,"b":true,"arr":[1,2],"s":"plain"}"#
        );
    }

    #[test]
    fn dsv4_multiline_string_value_kept_verbatim() {
        // a string="true" value may span lines (the `<` of the close tag terminates it).
        let block = ds_call_block("note", &[("text", "line one\nline two", true)]);
        let mut p = ToolStreamParser::dsv4(false);
        let mut pieces = p.push(&block);
        pieces.extend(p.finish());
        let (_c, _r, calls) = reassemble3(&pieces);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments, r#"{"text":"line one\nline two"}"#);
    }

    #[test]
    fn dsv4_malformed_block_surfaces_verbatim() {
        // a tool_calls block with no valid invoke: surfaced VERBATIM (house policy; the oracle
        // raises), tags included, stream continues.
        let text = format!("</think>{DS_OPEN}\nnot a real invoke\n{DS_CLOSE}done");
        let mut p = ToolStreamParser::dsv4(true);
        let mut pieces = p.push(&text);
        pieces.extend(p.finish());
        let (content, _r, calls) = reassemble3(&pieces);
        assert!(calls.is_empty());
        assert_eq!(
            content,
            format!("{DS_OPEN}\nnot a real invoke\n{DS_CLOSE}done")
        );
    }

    #[test]
    fn dsv4_unterminated_block_flushes_raw() {
        let text = format!(
            "</think>{DS_OPEN}\n{DS_INV}get_weather\">\n{DS_PAR}location\" string=\"true\">Par"
        );
        let mut p = ToolStreamParser::dsv4(true);
        let mut pieces = p.push(&text);
        pieces.extend(p.finish());
        let (content, _r, calls) = reassemble3(&pieces);
        assert!(calls.is_empty());
        assert!(
            content.starts_with(DS_OPEN),
            "open tag restored: {content:?}"
        );
    }

    #[test]
    fn dsv4_plain_content_passes_through_no_false_call() {
        // a request that never calls a tool: pure content after </think>, no holdback loss.
        let mut p = ToolStreamParser::dsv4(true);
        let mut pieces = p.push("reasoning here</think>The weather in Paris is 21C.");
        pieces.extend(p.finish());
        let (content, reasoning, calls) = reassemble3(&pieces);
        assert_eq!(reasoning, "reasoning here");
        assert_eq!(content, "The weather in Paris is 21C.");
        assert!(calls.is_empty());
    }

    #[test]
    fn dsv4_partial_dsml_open_holdback_never_loses_bytes() {
        // a partial `<｜DSML｜tool` that turns out to be prose must still surface as content,
        // and legitimate `\n\n` that is NOT a block prefix must not be dropped.
        let mut p = ToolStreamParser::dsv4(false);
        let mut pieces = p.push("cost note\n\n<\u{ff5c}DSML\u{ff5c}too");
        pieces.extend(p.push("k a while"));
        pieces.extend(p.finish());
        let (content, _r, calls) = reassemble3(&pieces);
        assert_eq!(content, "cost note\n\n<\u{ff5c}DSML\u{ff5c}took a while");
        assert!(calls.is_empty());
    }
}
