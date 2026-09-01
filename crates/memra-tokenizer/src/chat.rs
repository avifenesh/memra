//! Minimal chat-template renderer for the Qwen3.5 / ChatML format.
//!
//! The model's GGUF `tokenizer.chat_template` is a large jinja template covering
//! tools, vision, and multi-step reasoning. We do NOT ship a jinja engine; instead
//! we reproduce the text-only system/user/assistant path of that template exactly,
//! which is the path memra's text-in/text-out CLI uses. The reproduced behavior
//! (verified against the dumped template):
//!
//!   - a leading `system` turn renders `<|im_start|>system\n{content}<|im_end|>\n`
//!   - `user`      -> `<|im_start|>user\n{content}<|im_end|>\n`
//!   - `assistant` -> `<|im_start|>assistant\n{content}<|im_end|>\n`
//!   - with `add_generation_prompt`, Qwen3.5 appends `<|im_start|>assistant\n<think>\n`
//!     (its default, since `enable_thinking` is undefined => the else-branch fires).
//!
//! `content` is trimmed (the template applies `|trim`). If the GGUF has no template
//! we fall back to plain ChatML (no `<think>` tail).
//!
//! Non-qwen dialects each get their own arm, dispatched by a marker substring in the raw
//! template: Tencent Hy3 (`hy_User`), gemma4 (`<|turn>`), and StepFun Step-3.7-Flash /
//! arch `step35` (`render_message_content`). The step35 check must come BEFORE the qwen
//! `<think>`-tail detection — its template contains every qwen marker, so the qwen arm would
//! render the right generation tail on the wrong turn bodies.

/// A serde-free JSON value tree, built by the server (which owns serde_json) and handed to
/// the gemma4 tools arm. The compact gemma dialect needs argument/schema TYPE fidelity that a
/// pre-rendered string cannot carry — a string `"21"` and a number `21` render differently
/// (`<|"|>21<|"|>` vs `21`), a bool is `true`/`false`, a null is `None`, and mappings/sequences
/// recurse. `Num` keeps the exact numeric text (serde_json `Number::to_string()`) so the
/// rendered bytes match jinja's `{{ number }}` (Python `str()`), which this crate cannot
/// reproduce from an f64 alone. qwen/step arms ignore this; they use `ToolCall::params`.
#[derive(Debug, Clone, PartialEq)]
pub enum Val {
    Null,
    Bool(bool),
    Num(String),
    Str(String),
    Arr(Vec<Val>),
    /// Insertion-ordered object; the gemma dialect `dictsort`s keys (case-insensitive, stable)
    /// at render time, so ties keep this insertion order — matching jinja's `| dictsort`.
    Obj(Vec<(String, Val)>),
}

/// One tool call attached to a prior assistant turn.
/// `params` values are pre-rendered strings for the qwen/step/HY3 arms (string arguments raw,
/// everything else JSON-rendered by the caller). `args`/`id` carry the gemma4 arm's typed
/// arguments and the OpenAI `tool_calls[].id` used to resolve tool-response names.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ToolCall {
    pub name: String,
    pub params: Vec<(String, String)>,
    /// gemma4: typed arguments, dictsorted and dialect-rendered by the gemma arm.
    pub args: Vec<(String, Val)>,
    /// gemma4: the call id, matched against a following tool turn's `tool_call_id`.
    pub id: Option<String>,
}

/// One chat turn for the tools-capable renderer (`apply_chat_template_tools`).
/// The `reasoning` field is consumed by gemma4 and HY3; `tool_call_id`/`tool_name`/
/// `tool_responses` are gemma4-only. The qwen/step arms use role/content/tool_calls.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Turn {
    pub role: String,
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
    /// gemma4: assistant reasoning re-rendered as a `<|channel>thought` span (only for a
    /// tool_calls-carrying assistant after the last user message — the template's guard).
    pub reasoning: Option<String>,
    /// gemma4: on a role:"tool" turn, the OpenAI `tool_call_id` used to resolve the response
    /// name against the preceding assistant's `tool_calls[].id`.
    pub tool_call_id: Option<String>,
    /// gemma4: on a role:"tool" turn, the message's own `name` field (fallback when the id
    /// does not resolve).
    pub tool_name: Option<String>,
    /// gemma4 native (Google) responses embedded on an assistant turn: (name, response value).
    /// OpenAI histories leave this empty and use role:"tool" turns instead.
    pub tool_responses: Vec<(String, Val)>,
    /// deepseek-v4 quick-instruction task token (`action`/`query`/`authority`/`domain`/
    /// `title`/`read_url`, encoding_dsv4 DS_TASK_SP_TOKENS). Set only by the dsv4 fixture
    /// harness (the internal-classification heads); the OpenAI serve surface has no `task`
    /// field, so every serve request leaves this None and every other dialect ignores it.
    pub task: Option<String>,
    /// deepseek-v4 per-turn tool `function` objects (encoding_dsv4 renders the tool
    /// declaration on the message carrying them — system on the serve surface, or a developer
    /// message in the search-pipeline fixtures). The serve path also passes request-level
    /// tools via `tools_struct`, which the dsv4 arm folds onto the leading system turn.
    /// Every other dialect ignores this.
    pub tools: Vec<Val>,
}

/// Thinking control (owner directive 2026-08-07: every supported model is a thinking model,
/// one serve surface maps to each arch's native mechanism).
///
/// - `Default` = the template's OWN default, byte-identical to the pre-surface render:
///   qwen class opens `<think>\n` (thinking ON), gemma4 renders the CLOSED thought channel
///   (its `enable_thinking | default(false)`), hy3 renders `reasoning_effort:no_think`.
/// - `NoThink` = thinking OFF via the arch's native off-switch: qwen
///   `enable_thinking=false` (closed `<think>\n\n</think>\n\n`), gemma4 closed thought
///   channel, hy3 `no_think`. On step35 — whose `<think>` tail is unconditional — it clamps
///   to the lowest effort level instead (`Reasoning: low`).
/// - `Think` = thinking explicitly ON: qwen open `<think>\n` (same bytes as its default),
///   gemma4 `<|think|>\n` injected into the system turn + an OPEN generation turn, hy3
///   an open `<think:opensource>` channel at the requested effort.
///
/// On templates with no switch at all the non-native direction is a graceful no-op.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkMode {
    Default,
    NoThink,
    Think,
}

/// Which `encoding_dsv4.py` revision governs the deepseek-v4 REASONING-EFFORT ladder
/// (0731 re-gate, 2026-08-18 — research/dsv4-template-20260818/ENCODING-DIFF.md).
///
/// The two shipped encodings differ ONLY here; every other rendering law (roles, tool
/// blocks, transitions, special tokens, think-mode prefixes, parsing) is byte-identical:
///
/// | `reasoning_effort` | `Preview` (base repo @ 60d8d707)     | `V0731` (0731 @ 7872f01b)          |
/// |--------------------|--------------------------------------|-------------------------------------|
/// | None               | no prefix                            | no prefix (None == "low" default)   |
/// | "low"              | INVALID upstream (assert) — renders as no prefix here | no prefix          |
/// | "high"             | documented NO-OP (== None)           | `DS_EFFORT_ABSOLUTE_MAX` prefix     |
/// | "max"              | `DS_EFFORT_ABSOLUTE_MAX` prefix      | `DS_EFFORT_BEYOND_MAX` prefix       |
///
/// The prefix (when non-empty) is injected once, before the first rendered message, in
/// thinking mode only; chat mode never renders a prefix under either encoding.
///
/// DETECTION IS CONFIG-KEYED, never filename-keyed: the 0731 checkpoint added exactly four
/// `dspark_*` keys to config.json in the same revision that remapped the ladder
/// (`dspark_block_size`, `dspark_markov_rank`, `dspark_noise_token_id`,
/// `dspark_target_layer_ids`); tokenizer/template files are byte-identical across the two
/// checkpoints, so config.json is the artifact's only encoding marker. `Tokenizer::from_hf_dir`
/// performs the census (all four -> `V0731`, none -> `Preview`, a partial set refuses to load).
/// Callers that cannot know the revision pass `None`; rendering then REFUSES exactly the
/// (thinking, "high"/"max") requests whose bytes differ between revisions and stays
/// infallible everywhere the two encodings agree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dsv4Encoding {
    /// deepseek-ai/DeepSeek-V4-Flash (preview) law: {None,"high"} no-op, "max" -> absolute.
    Preview,
    /// DeepSeek-V4-Flash-0731 law: None/"low" no prefix, "high" -> absolute, "max" -> beyond.
    V0731,
}

/// Render messages into the prompt string.
///
/// `template` is the raw GGUF chat_template (used only to decide qwen3.5-vs-plain
/// chatml behavior — we detect the `<think>` generation tail by substring). When
/// `None`, plain ChatML is produced.
/// ds4f rung-3 serve finding (2026-08-22): the REAL dsv4 artifacts ship their chat
/// dialect as CODE (`encoding/encoding_dsv4.py`) — tokenizer_config.json carries NO
/// `chat_template` string and no chat_template.jinja exists. Every template-STRING
/// keyed dispatch therefore never fires on the artifact we serve, and the serve-st
/// honesty gate 400s a model whose dialect is fully defined. The artifact-level truth
/// is the config `dspark_*` census (`Dsv4Encoding`, already detected at tokenizer
/// load): when it is present, the dsv4 renderer IS the model's template. This entry is
/// the plain-path dispatch on that truth; `apply_chat_template_str` keeps its exact
/// legacy bytes for every other family.
pub fn apply_chat_template_enc(
    template: Option<&str>,
    messages: &[(&str, &str)],
    add_generation_prompt: bool,
    dsv4_encoding: Option<Dsv4Encoding>,
) -> Result<String, String> {
    if dsv4_encoding.is_some() && !template.is_some_and(template_is_dsv4) {
        let msgs: Vec<Turn> = messages
            .iter()
            .map(|(r, c)| Turn {
                role: r.to_string(),
                content: c.to_string(),
                ..Default::default()
            })
            .collect();
        return apply_dsv4_template(
            &msgs,
            add_generation_prompt,
            &[],
            ThinkMode::Default,
            None,
            dsv4_encoding,
        );
    }
    Ok(apply_chat_template_str(
        template,
        messages,
        add_generation_prompt,
    ))
}

pub fn apply_chat_template_str(
    template: Option<&str>,
    messages: &[(&str, &str)],
    add_generation_prompt: bool,
) -> String {
    // Tencent Hy3 (`hy_v3`): a completely different special-token dialect (no ChatML).
    // Detected by its `hy_User` token literal; rendered by the dedicated arm below.
    // Legacy path = the template's own default ("no_think") — byte-identical to history.
    if template.is_some_and(|t| t.contains("hy_User")) {
        return apply_hy3_template(messages, add_generation_prompt, "no_think");
    }
    // StepFun Step-3.7-Flash (arch `step35`): a ChatML *dialect* — same `<|im_start|>` framing,
    // different everything else (see `apply_step35_template`). Detected by its
    // `render_message_content` macro, which no other committed template defines. This check MUST
    // precede the qwen `<think>`-tail detection below: the step35 template contains both markers,
    // so the qwen arm would produce the right generation tail with the wrong turn bodies.
    if template.is_some_and(|t| t.contains("render_message_content")) {
        let turns: Vec<Turn> = messages
            .iter()
            .map(|(r, c)| Turn {
                role: r.to_string(),
                content: c.to_string(),
                tool_calls: Vec::new(),
                ..Default::default()
            })
            .collect();
        return apply_step35_template(&turns, add_generation_prompt, &[], None);
    }
    // deepseek-v4 (`encoding_dsv4`): `<｜User｜>`/`<｜Assistant｜>` turn dialect with three
    // think modes + DSML tool calls. Detected by its two structural markers (`<｜Assistant｜>`
    // AND `｜DSML｜`). MUST precede the qwen `<think>`-tail check: a faithful dsv4 template
    // mentions `<think>` in its tools block, and the qwen detector would otherwise fire.
    // Legacy path = the model's own default thinking mode (thinking; see ThinkMode docs); BOS
    // IS emitted here (encoding_dsv4 owns the BOS — tokenizer_config add_bos_token is false).
    if template.is_some_and(template_is_dsv4) {
        let msgs: Vec<Turn> = messages
            .iter()
            .map(|(r, c)| Turn {
                role: r.to_string(),
                content: c.to_string(),
                ..Default::default()
            })
            .collect();
        return apply_dsv4_template(
            &msgs,
            add_generation_prompt,
            &[],
            ThinkMode::Default,
            None,
            None,
        )
        .expect("dsv4 render without reasoning_effort is encoding-independent");
    }
    // GLM-5.3-Flash (`glm5_next`): `[gMASK]<sop>` + `<|user|>`/`<|assistant|>`/`<|observation|>`
    // turn dialect with an always-open `<think>` tail and an always-rendered reasoning-effort
    // system line. Detected by its two structural markers. MUST precede the qwen `<think>`-tail
    // check below: this template contains BOTH `<think>` and `add_generation_prompt`, so the
    // qwen detector fires on it and used to render every GLM prompt as ChatML.
    // Legacy path = the template's own effort default (`max`) — the only default it has.
    if template.is_some_and(template_is_glm5) {
        let turns: Vec<Turn> = messages
            .iter()
            .map(|(r, c)| Turn {
                role: r.to_string(),
                content: c.to_string(),
                ..Default::default()
            })
            .collect();
        return apply_glm5_template(&turns, add_generation_prompt, &[], None)
            .expect("glm5 render without reasoning_effort takes the template's own default");
    }
    // gemma4: `<|turn>role\n{content}<turn|>\n` dialect; generation prompt appends
    // `<|turn>model\n` + the CLOSED thought channel (`<|channel>thought\n<channel|>` — the
    // template's enable_thinking-false default). bos comes from encode(add_special) — the
    // template's `{{ bos_token }}` is NOT re-emitted here (double-BOS trap).
    // Legacy path = thinking OFF (the template's `default(false)`) — byte-identical to history.
    if template.is_some_and(|t| t.contains("<|turn>")) {
        return apply_gemma4_template(messages, add_generation_prompt, false);
    }
    // qwen3.5 template emits a `<think>\n` tail on the generation prompt by default.
    let qwen_think = template
        .map(|t| t.contains("<think>") && t.contains("add_generation_prompt"))
        .unwrap_or(false);

    let mut out = String::new();
    for (i, (role, content)) in messages.iter().enumerate() {
        let content = content.trim();
        match *role {
            "system" => {
                // template requires system at the beginning; we render it wherever
                // it appears at index 0 (the common case).
                let _ = i;
                out.push_str("<|im_start|>system\n");
                out.push_str(content);
                out.push_str("<|im_end|>\n");
            }
            "user" => {
                out.push_str("<|im_start|>user\n");
                out.push_str(content);
                out.push_str("<|im_end|>\n");
            }
            "assistant" => {
                out.push_str("<|im_start|>assistant\n");
                out.push_str(content);
                out.push_str("<|im_end|>\n");
            }
            other => {
                // unsupported role in this minimal renderer; emit as a generic turn.
                out.push_str("<|im_start|>");
                out.push_str(other);
                out.push('\n');
                out.push_str(content);
                out.push_str("<|im_end|>\n");
            }
        }
    }

    if add_generation_prompt {
        out.push_str("<|im_start|>assistant\n");
        if qwen_think {
            out.push_str("<think>\n");
        }
    }

    out
}

/// The fixed tool-calling instruction block of the qwen3.5/3.6-class templates. Byte-for-byte
/// the string literal shared by ornith9b / agentworld / ref-qwen36-35b
/// (research/onboard-ornith-20260801/templates/*.jinja) and the deployed GGUF dumps.
const QWEN_TOOLS_INSTRUCTION: &str = "\n\nIf you choose to call a function ONLY reply in the \
following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n\
<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\n\
This is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n\
</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified \
format: an inner <function=...></function> block must be nested within <tool_call></tool_call> \
XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for \
your function call in natural language BEFORE the function call, but NOT after\n- If there is \
no function call available, answer the question like normal with your current knowledge and do \
not tell the user about function calls\n</IMPORTANT>";

/// Qwen3.8's REASONING-EFFORT LADDER — the two instruction sentences its chat template injects
/// at the head of the system turn, reproduced byte-for-byte out of the shipped template
/// (`research/reasoning-schema-20260823/qwen38-27b.chat_template.jinja`, == the served GGUF's
/// own `tokenizer.chat_template`; the BF16 and NVFP4-Q5K mints carry the identical 9993-byte
/// string).
///
/// THE DEFECT THIS EXISTS TO CLOSE (lane/reasoning-schema-20260823): the template's ladder is
/// `reasoning_effort|default('xhigh')` over `xhigh|medium|low` (with `high` aliased to
/// `xhigh`), but `ModelCaps::effort_levels` probed for the substring `reasoning_effort is
/// defined` — which this template does not contain. So the level never reached the render,
/// `reasoning_effort: low|medium|high` was accepted-and-ignored on every qwen3.8 request, AND
/// the template's own `xhigh` default never rendered either.
///
/// Note which rungs carry a sentence: `xhigh` and `low` do; **`medium` deliberately renders
/// NOTHING** (the template sets no `reasoning_instructions` for it), so `medium` is the
/// template's own "no steering" rung, not a missing case.
const QWEN38_EFFORT_XHIGH: &str = "Reasoning effort is set to xhigh. Please think carefully \
through the task, validate key assumptions, consider plausible alternatives, and prioritize \
correctness, consistency, and clarity in the final answer.";
const QWEN38_EFFORT_LOW: &str = "Reasoning effort is set to low. Keep your thinking brief and \
focused, moving directly to the conclusion without unnecessary elaboration.";

/// Does this template carry the Qwen3.8 reasoning-effort ladder?
///
/// Keyed on the two instruction SENTENCES this renderer reproduces, not on the jinja control
/// flow around them. That is the strongest form of the house's template-marker law: the probe
/// passes only when the literal we are about to emit is the literal the template emits, so a
/// vendor or mint that reworded a rung fails the probe and falls back to the plain qwen arm
/// (byte-identical prompts) instead of silently rendering a sentence that model never saw.
pub fn template_has_qwen_effort(template: &str) -> bool {
    template.contains(QWEN38_EFFORT_XHIGH) && template.contains(QWEN38_EFFORT_LOW)
}

/// Resolve the Qwen3.8 ladder: `(think, level)` -> the instruction sentence to inject.
///
/// Faithful to the template's own arithmetic, in its order:
///   1. thinking OFF (`enable_thinking is false`) => the whole `reasoning_instructions` block
///      is skipped, so NO sentence — a thinking-off prompt carries no effort steering.
///   2. `resolved = level | default('xhigh')`, then `high -> xhigh`.
///   3. `xhigh -> XHIGH sentence`, `low -> LOW sentence`, `medium -> '' (no sentence)`.
///   4. anything else => the template calls `raise_exception`, so we refuse too rather than
///      render a rung this model was never trained on.
fn qwen38_effort_instructions(
    think: ThinkMode,
    reasoning_effort: Option<&str>,
) -> Result<&'static str, String> {
    if think == ThinkMode::NoThink {
        return Ok("");
    }
    match reasoning_effort {
        // `None` is the template's own `default('xhigh')`; `high` is aliased to `xhigh` by the
        // template itself, and the server's canonical table already folds xhigh/max/ultra into
        // `high`, so these three are one rung by the model's own definition.
        None | Some("high") | Some("xhigh") => Ok(QWEN38_EFFORT_XHIGH),
        Some("medium") => Ok(""),
        Some("low") => Ok(QWEN38_EFFORT_LOW),
        Some(other) => Err(format!(
            "reasoning effort {other:?} is not a level this chat template defines \
             (low|medium|high; the template's own ladder is xhigh|medium|low with high \
             aliased to xhigh)"
        )),
    }
}

/// Tools-capable chat rendering (serve-tools lane, 2026-08-02). Reproduces the TOOLS branch of
/// the qwen3.5/3.6-class ChatML templates exactly (verified against the committed dumps AND the
/// deployed GGUFs' embedded templates, byte-identical):
///
///   - tools present  -> `<|im_start|>system\n# Tools\n\nYou have access to the following
///     functions:\n\n<tools>` + `\n{tool json}` each + `\n</tools>` + the fixed instruction
///     block; a leading system turn's trimmed content is appended after `\n\n`; `<|im_end|>\n`.
///   - assistant turns with `tool_calls` -> content then `<tool_call>\n<function=NAME>\n`
///     (+`\n\n` separator when content is non-empty; later calls separated by `\n`),
///     `<parameter=K>\nV\n</parameter>\n` each, `</function>\n</tool_call>`, then `<|im_end|>\n`.
///   - `tool` turns -> grouped into ONE user turn: `<|im_start|>user` opens a run of
///     consecutive tool messages, each `\n<tool_response>\n{content}\n</tool_response>`,
///     `<|im_end|>\n` closes the run.
///   - generation prompt -> `<|im_start|>assistant\n` + `<think>\n` (template default) or
///     `<think>\n\n</think>\n\n` (`ThinkMode::NoThink` = the template's `enable_thinking=false`
///     switch; ignored when the template has no `enable_thinking`).
///
/// The no-tools/no-tool-turns/`Default`-think case renders byte-identically to
/// `apply_chat_template_str` (pinned by `tools_renderer_matches_legacy_when_plain`); callers
/// that want the hard isolation guarantee keep calling the legacy function on that path.
/// Errors (never on the plain path): tools/tool turns on a template without a tools branch
/// (hy3 / gemma4 / bare ChatML).
///
/// `reasoning_effort` is a per-dialect level STRING, never a think switch: step35 renders
/// `Reasoning: {low|medium|high}` into the system turn (see `apply_step35_template`); hy3
/// consumes `no_think|low|high` (medium clamps to low); deepseek-v4 resolves it through the
/// artifact's encoding revision into the effort prompt prefix (see `Dsv4Encoding` — 0731
/// ladder low/high/max, preview "max" only). Every other dialect ignores it (their templates
/// have no `reasoning_effort` input), and `None` is each template's own default. The server
/// only supplies `Some` for models whose template consumes it (`ModelCaps::effort_levels`
/// or `ModelCaps::dsv4`), so other prompts stay byte-identical by construction, not by luck.
pub fn apply_chat_template_tools(
    template: Option<&str>,
    turns: &[Turn],
    add_generation_prompt: bool,
    tools_json: &[String],
    think: ThinkMode,
    reasoning_effort: Option<&str>,
) -> Result<String, String> {
    // Compat entry (no structured tools, no dsv4 encoding revision): CLI bins +
    // qwen/step/hy3 tests. The gemma4 arm needs typed tool DEFINITIONS and the dsv4 arm an
    // encoding revision for the effort ladder, so the serve path calls `_ex` with them
    // (a dsv4 "high"/"max" request through THIS entry refuses on the unknown revision).
    apply_chat_template_tools_ex(
        template,
        turns,
        add_generation_prompt,
        tools_json,
        &[],
        think,
        reasoning_effort,
        None,
    )
}

/// `apply_chat_template_tools` plus the gemma4 arm's structured tool `function` objects
/// (`tools_struct`) and the dsv4 arm's encoding revision (`dsv4_encoding` — the effort
/// ladder differs between the preview and 0731 checkpoints; see `Dsv4Encoding`). Every
/// non-gemma dialect ignores `tools_struct`; every non-dsv4 dialect ignores `dsv4_encoding`.
#[allow(clippy::too_many_arguments)]
pub fn apply_chat_template_tools_ex(
    template: Option<&str>,
    turns: &[Turn],
    add_generation_prompt: bool,
    tools_json: &[String],
    tools_struct: &[Val],
    think: ThinkMode,
    reasoning_effort: Option<&str>,
    dsv4_encoding: Option<Dsv4Encoding>,
) -> Result<String, String> {
    let has_tool_features = !tools_json.is_empty()
        || turns
            .iter()
            .any(|t| t.role == "tool" || !t.tool_calls.is_empty());
    // deepseek-v4 is template-STRING-less on the real artifacts (dialect ships as
    // encoding code) — the detected encoding revision is the dispatch truth there.
    let is_dsv4 = dsv4_encoding.is_some() || template.is_some_and(template_is_dsv4);
    // A template "has a tools branch" if it carries the qwen/step `<tools>` block OR the
    // gemma4 tooluse dialect (`<|turn>` turn framing AND the `<|tool>` declaration marker)
    // OR it is the dsv4 dialect (DSML defines a full tool protocol).
    let tools_branch = is_dsv4 || template.is_some_and(template_has_tools_branch);
    if has_tool_features && !tools_branch {
        return Err("model chat template has no tools branch".into());
    }
    // deepseek-v4 (`encoding_dsv4`): its own dialect all the way through, tools included.
    // Detected by its two structural markers; MUST precede the qwen/step marker checks
    // (a faithful dsv4 template mentions `<think>` in its tools block). Renders tool
    // DEFINITIONS (into the system turn), assistant DSML tool_calls, and role:"tool" turns
    // merged into user `<tool_result>` blocks. ThinkMode maps onto encoding_dsv4's
    // thinking_mode + reasoning_effort (see `apply_dsv4_template`).
    if is_dsv4 {
        return apply_dsv4_template(
            turns,
            add_generation_prompt,
            tools_struct,
            think,
            reasoning_effort,
            dsv4_encoding,
        );
    }
    // step35: its own dialect all the way through, tools included. Must precede the
    // qwen arm: the step35 template contains `<tools>`, `<think>` and `add_generation_prompt`,
    // so every qwen marker check below matches it. `ThinkMode` is ignored (no `enable_thinking`
    // in this template => `think_switch` is false => NoThink is already a documented no-op);
    // `reasoning_effort` is this dialect's own control and is honored here.
    if template.is_some_and(|t| t.contains("render_message_content")) {
        return Ok(apply_step35_template(
            turns,
            add_generation_prompt,
            tools_json,
            reasoning_effort,
        ));
    }
    // GLM-5.3-Flash (`glm5_next`): its own dialect all the way through, tools included. Must
    // precede the qwen arm below, which every one of this template's `<think>` /
    // `add_generation_prompt` / `<tools>` markers would otherwise match. `ThinkMode` is ignored
    // (the template has no off switch at all — `think_switch` is false and an explicit client
    // off-request is refused upstream); `reasoning_effort` is this dialect's own control and is
    // honored here. Tool DEFINITIONS come from `tools_struct` (the unwrapped `function`
    // objects), not the qwen `tools_json` strings, because the template renders the function
    // object alone and drops its `defer_loading`/`strict` keys.
    if template.is_some_and(template_is_glm5) {
        // A caller that has tools but no `tools_struct` reached the compat entry
        // (`apply_chat_template_tools`, which passes `&[]`). Rendering the prompt WITHOUT the
        // tools block there would be a silent downgrade — the model would be asked to call a
        // function it was never shown. Name it instead; the serve path always calls `_ex`.
        if !tools_json.is_empty() && tools_struct.is_empty() {
            return Err(
                "glm5 tool definitions need the structured `tools_struct` (the unwrapped \
                 `function` objects) — call apply_chat_template_tools_ex"
                    .into(),
            );
        }
        return apply_glm5_template(turns, add_generation_prompt, tools_struct, reasoning_effort);
    }
    // Tencent HY3: the pinned shipping template has a complete tools branch using suffixed
    // special tokens (`<tool_calls:opensource>`, `<arg_key:opensource>`, ...). It also owns
    // the no_think/low/high reasoning ladder. Reproduce the one template for plain, reasoning,
    // declarations, assistant calls and tool-result history so those surfaces cannot drift.
    if template.is_some_and(|t| t.contains("hy_User")) {
        let effort = match (think, reasoning_effort) {
            (ThinkMode::Think, Some("high")) => "high",
            (ThinkMode::Think, _) => "low",
            _ => "no_think",
        };
        return Ok(apply_hy3_template_tools(
            turns,
            add_generation_prompt,
            tools_json,
            effort,
        ));
    }
    // gemma4 TOOLUSE dialect (`<|turn>` turn framing + the `<|tool>` declaration marker):
    // the official Google tooluse template is the rendering LAW (research/gemma4-tools-20260817
    // /official-tooluse-template.jinja). Engages for tool DEFINITIONS, tool_calls, tool-role
    // turns AND plain/thinking requests on this trunk. A `<|turn>` template WITHOUT `<|tool>`
    // has no committed tools reference and falls through to the reject/plain arm below.
    // Must precede the plain `<|turn>` arm and the qwen marker checks (the tooluse template
    // carries no `<tools>`, so it would not match those).
    if template.is_some_and(|t| t.contains("<|turn>") && t.contains("<|tool>")) {
        // QAT-trunk variant emits a CLOSED thought channel on the thinking-off generation
        // prompt; the official served trunk emits a bare `<|turn>model\n`. Keyed on the exact
        // gen-prompt literal, which is present only in the QAT template's tail (verified:
        // research/gemma4-tools-20260817 template diff).
        let closed_tail = template.is_some_and(|t| t.contains("<|channel>thought\\n<channel|>"));
        return Ok(apply_gemma4_tools_template(
            turns,
            add_generation_prompt,
            tools_struct,
            think == ThinkMode::Think,
            closed_tail,
        ));
    }
    if template.is_some_and(|t| t.contains("<|turn>")) {
        // Plain-gemma4 dialect: no committed tools rendering reference. ThinkMode maps to the
        // arch's native mechanism (thinking goldens, render-thinking-goldens.py):
        //   gemma4 -> enable_thinking: default(false) = Default/NoThink;
        //             Think = <|think|> system token + open generation turn.
        if has_tool_features {
            return Err("tools are not supported on this model's chat-template dialect".into());
        }
        let messages: Vec<(&str, &str)> = turns
            .iter()
            .map(|t| (t.role.as_str(), t.content.as_str()))
            .collect();
        return Ok(apply_gemma4_template(
            &messages,
            add_generation_prompt,
            think == ThinkMode::Think,
        ));
    }
    let qwen_think = template
        .map(|t| t.contains("<think>") && t.contains("add_generation_prompt"))
        .unwrap_or(false);
    let think_switch = template.is_some_and(|t| t.contains("enable_thinking"));
    // Qwen3.8's reasoning-effort ladder. Gated on the template carrying the two instruction
    // sentences this renderer reproduces, so every OTHER qwen-class template (ornith15,
    // agentworld, ref-qwen36 — binary `enable_thinking` and no ladder) renders byte-identically
    // to before, by construction rather than by luck.
    let effort_ladder = template.is_some_and(template_has_qwen_effort);
    let effort_instructions = if effort_ladder {
        qwen38_effort_instructions(think, reasoning_effort)?
    } else {
        ""
    };

    // LEADING SYSTEM RUN. The qwen3.8 template MERGES the whole leading run of system/developer
    // turns into ONE system turn, joining trimmed non-empty contents with `\n`, and its body loop
    // then refuses a system message that appears later (`System message must be at the
    // beginning.`). memra's historical qwen arm emits one `<|im_start|>system` turn PER message,
    // which diverges from that the moment a request carries two — a shape this server produces
    // itself, since it normalizes OpenAI's `developer` role to `system`.
    //
    // The merge is scoped to LADDER templates (`qwen_effort`) on purpose, and the scope is
    // measured rather than assumed: rendering `[system, system, user]` through the shipped jinja
    // gives one merged turn on qwen3.8 and `raise_exception` on ornith15, so the two dialects do
    // NOT share this law. Every non-ladder template therefore keeps its exact historical bytes.
    let merge_leading_system = effort_ladder;
    let n_leading_system = if merge_leading_system {
        turns
            .iter()
            .take_while(|t| t.role == "system" || t.role == "developer")
            .count()
    } else {
        usize::from(!tools_json.is_empty() && turns.first().is_some_and(|t| t.role == "system"))
    };
    let merged_system = if merge_leading_system {
        turns[..n_leading_system]
            .iter()
            .map(|t| t.content.trim())
            .filter(|c| !c.is_empty())
            .collect::<Vec<_>>()
            .join("\n")
    } else {
        turns
            .first()
            .filter(|_| n_leading_system > 0)
            .map(|t| t.content.trim().to_string())
            .unwrap_or_default()
    };

    let mut out = String::new();
    // NO-TOOLS placement of the effort instruction (template law, verified against the shipped
    // jinja): the sentence is PREPENDED to the merged system turn across a blank line; when the
    // request carries no leading system content the sentence becomes a system turn of its own.
    // Emitted here, ahead of the message loop, because that is where the template emits it.
    if tools_json.is_empty()
        && merge_leading_system
        && (!effort_instructions.is_empty() || !merged_system.is_empty())
    {
        out.push_str("<|im_start|>system\n");
        if !effort_instructions.is_empty() {
            out.push_str(effort_instructions);
            if !merged_system.is_empty() {
                out.push_str("\n\n");
            }
        }
        out.push_str(&merged_system);
        out.push_str("<|im_end|>\n");
    }
    // Tools system header replaces the plain system turn (template law: the leading system
    // turn's content is folded INTO the tools block).
    if !tools_json.is_empty() {
        out.push_str("<|im_start|>system\n");
        // TOOLS placement: the effort sentence precedes the `# Tools` header inside the one
        // system turn (template law: `reasoning_instructions + '\n\n'` then the header).
        if !effort_instructions.is_empty() {
            out.push_str(effort_instructions);
            out.push_str("\n\n");
        }
        out.push_str("# Tools\n\nYou have access to the following functions:\n\n<tools>");
        for tool in tools_json {
            out.push('\n');
            out.push_str(tool);
        }
        out.push_str("\n</tools>");
        out.push_str(QWEN_TOOLS_INSTRUCTION);
        if !merged_system.is_empty() {
            out.push_str("\n\n");
            out.push_str(&merged_system);
        }
        out.push_str("<|im_end|>\n");
    }

    for (i, turn) in turns.iter().enumerate() {
        // The leading system run was already emitted (merged, or folded into the tools header).
        if i < n_leading_system {
            continue;
        }
        let content = turn.content.trim();
        match turn.role.as_str() {
            // A ladder template's leading system run never reaches here (merged above), so this
            // arm is the unchanged historical path for every other dialect — and for a system
            // message that appears AFTER a user turn, which the vendor jinja refuses outright and
            // this renderer still passes through (pre-existing, out of this lane's scope).
            "system" => {
                out.push_str("<|im_start|>system\n");
                out.push_str(content);
                out.push_str("<|im_end|>\n");
            }
            "user" => {
                out.push_str("<|im_start|>user\n");
                out.push_str(content);
                out.push_str("<|im_end|>\n");
            }
            "assistant" => {
                out.push_str("<|im_start|>assistant\n");
                // LADDER templates replay the prior turn's `<think>` block (vendor law:
                // `preserve_thinking is undefined or preserve_thinking is true` — the ABSENT
                // default is replay, `reasoning_content|trim` inside, EMPTY when the client
                // sent none). memra historically rendered assistant turns as content only,
                // a named gap off the vendor's bytes (see the server's preserve_thinking
                // kwarg doc) — and the byte that kept every multi-turn conversation from
                // ever matching a parked session's stream: the generation prompt ends in a
                // `<think>` block, so the live stream carries it while the re-render did
                // not. Scoped to `effort_ladder` so every other qwen-class template keeps
                // its exact historical bytes, by construction.
                if effort_ladder {
                    out.push_str("<think>\n");
                    out.push_str(turn.reasoning.as_deref().map(str::trim).unwrap_or(""));
                    out.push_str("\n</think>\n\n");
                }
                out.push_str(content);
                for (k, call) in turn.tool_calls.iter().enumerate() {
                    if k == 0 {
                        if !content.is_empty() {
                            out.push_str("\n\n");
                        }
                    } else {
                        out.push('\n');
                    }
                    out.push_str("<tool_call>\n<function=");
                    out.push_str(&call.name);
                    out.push_str(">\n");
                    for (key, value) in &call.params {
                        out.push_str("<parameter=");
                        out.push_str(key);
                        out.push_str(">\n");
                        out.push_str(value);
                        out.push_str("\n</parameter>\n");
                    }
                    out.push_str("</function>\n</tool_call>");
                }
                out.push_str("<|im_end|>\n");
            }
            "tool" => {
                if i == 0 || turns[i - 1].role != "tool" {
                    out.push_str("<|im_start|>user");
                }
                out.push_str("\n<tool_response>\n");
                out.push_str(content);
                out.push_str("\n</tool_response>");
                if i + 1 >= turns.len() || turns[i + 1].role != "tool" {
                    out.push_str("<|im_end|>\n");
                }
            }
            other => {
                // parity with the legacy renderer's generic-turn arm.
                out.push_str("<|im_start|>");
                out.push_str(other);
                out.push('\n');
                out.push_str(content);
                out.push_str("<|im_end|>\n");
            }
        }
    }

    if add_generation_prompt {
        out.push_str("<|im_start|>assistant\n");
        if qwen_think {
            if think == ThinkMode::NoThink && think_switch {
                out.push_str("<think>\n\n</think>\n\n");
            } else {
                out.push_str("<think>\n");
            }
        }
    }
    Ok(out)
}

/// The fixed tool-calling instruction block of the StepFun `step35` template. NOT the same
/// string as `QWEN_TOOLS_INSTRUCTION` — three differences, all load-bearing: the header says
/// "in JSONSchema format", the nesting reminder carries literal `\n...\n` inside the
/// `<function=...>` / `<tool_call>` examples, and the Reminder list has 2 bullets instead of 4
/// (no "optional reasoning BEFORE the call" and no "answer normally if no function is
/// available"). Copied byte-for-byte out of the shipped template
/// (`research/step37-bringup-20260802/raw/chat_template.jinja`, == the GGUF's own
/// `tokenizer.chat_template`).
const STEP35_TOOLS_INSTRUCTION: &str = "\n\nIf you choose to call a function ONLY reply in the \
following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n\
<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\n\
This is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n\
</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified \
format: an inner <function=...>\n...\n</function> block must be nested within <tool_call>\n\
...\n</tool_call> XML tags\n- Required parameters MUST be specified\n</IMPORTANT>";

/// StepFun Step-3.7-Flash (GGUF arch `step35`) chat template.
///
/// A ChatML *dialect*, not ChatML: it shares the `<|im_start|>role\n…<|im_end|>\n` frame and
/// nothing else. Reproduced from the shipped jinja, and pinned test-by-test against goldens
/// rendered from that jinja under jinja2 with `trim_blocks`/`lstrip_blocks` — the settings HF
/// transformers and llama.cpp's minja both parse chat templates with
/// (`research/step37-p2-20260806/render_step35_template.py`, goldens committed under `raw/`).
///
/// Where it differs from the qwen3.5/3.6 arms above — every one of these silently corrupts the
/// prompt if the qwen arm is reused:
///
/// | | qwen3.5/3.6 | step35 |
/// |---|---|---|
/// | reasoning level | `enable_thinking` bool | `Reasoning: {low,medium,high}\n\n` prefix inside the system turn |
/// | `<think>` tail | switchable | **unconditional** — no `enable_thinking`, so `ThinkMode::NoThink` is a no-op |
/// | prior assistant turns | content only | turns AFTER the last real user query also carry `<think>\n{reasoning}\n</think>\n` |
/// | tool results | grouped into a `user` turn, `\n<tool_response>\n…\n</tool_response>` | own **`tool_response`** role, `<tool_response>…</tool_response>` with NO inner newlines |
/// | content | `\|trim`med | **not** trimmed |
/// | tools header | `following functions:` | `following functions in JSONSchema format:` |
/// | call separators | `\n\n` after content, `\n` between calls | **none** |
/// | leading system + tools | appended AFTER the instruction block | folded in BEFORE `# Tools` |
///
/// `reasoning_effort` is the model's headline three-level control (low/medium/high per the
/// StepFun model card). It is a parameter here rather than a `ThinkMode`: the value is a
/// *string in the system turn*, so a bool cannot carry it. The serve path supplies it through
/// `apply_chat_template_tools` (worker `Request::reasoning_effort`, mapped from the OpenAI
/// `reasoning_effort` body field when `ModelCaps::effort_levels` is set); `None` — the
/// legacy-str path and every non-step35 model — renders the template's own default
/// (no `Reasoning:` line at all).
///
/// BOS is NOT emitted (the jinja's `{{bos_token}}` is dropped): memra's `encode(add_special)`
/// prepends it from `tokenizer.ggml.add_bos_token`/`bos_token_id` — the same double-BOS trap the
/// gemma4 arm documents.
///
/// ONE deliberate divergence: the jinja's body loop has no `else`, so a role outside
/// {system, user, assistant, tool} renders as **nothing at all** — the turn silently vanishes
/// from the prompt. memra renders it as a generic `<|im_start|>{role}\n{content}<|im_end|>\n`
/// turn instead, matching the other arms here. A dropped turn is the worse failure, and this
/// branch cannot fire on the serve surface: OpenAI roles are exactly system/user/assistant/tool,
/// all four of which are reproduced byte-for-byte.
///
/// Not reproduced (needs data `Turn` does not carry, tracked, cannot fire from an OpenAI client):
/// the `name == "observation"` alias that renames a non-leading `system` turn's role to
/// `observation`. The `<im_patch>` image-content path is handled UPSTREAM of this template
/// (lane/step37-vision, 2026-08-30): when the step vision seam is armed, the server's
/// content walker (`content_to_text_vision_step`) renders each image part's full pad-token
/// expansion and the template macro's text-separator law into the turn content string, so
/// the content arrives here as literal text and passes through verbatim.
fn apply_step35_template(
    turns: &[Turn],
    add_generation_prompt: bool,
    tools_json: &[String],
    reasoning_effort: Option<&str>,
) -> String {
    let mut out = String::new();
    let leading_system = turns.first().filter(|t| t.role == "system");

    // --- system header. Two branches in the jinja, and the ORDER differs between them.
    if !tools_json.is_empty() {
        out.push_str("<|im_start|>system\n");
        if let Some(effort) = reasoning_effort {
            out.push_str("Reasoning: ");
            out.push_str(effort);
            out.push_str("\n\n");
        }
        if let Some(sys) = leading_system {
            // unconditional `content + '\n\n'` — no emptiness check, unlike the qwen arm.
            out.push_str(&sys.content);
            out.push_str("\n\n");
        }
        out.push_str(
            "# Tools\n\nYou have access to the following functions in JSONSchema \
                      format:\n\n<tools>",
        );
        for tool in tools_json {
            out.push('\n');
            out.push_str(tool);
        }
        out.push_str("\n</tools>");
        out.push_str(STEP35_TOOLS_INSTRUCTION);
        out.push_str("<|im_end|>\n");
    } else if let Some(sys) = leading_system {
        out.push_str("<|im_start|>system\n");
        if let Some(effort) = reasoning_effort {
            out.push_str("Reasoning: ");
            out.push_str(effort);
            out.push_str("\n\n");
        }
        out.push_str(&sys.content);
        out.push_str("<|im_end|>\n");
    } else if let Some(effort) = reasoning_effort {
        out.push_str("<|im_start|>system\nReasoning: ");
        out.push_str(effort);
        out.push_str("\n\n<|im_end|>\n");
    }

    // --- last_query_index: the index of the LAST `user` turn that is a real query, i.e. whose
    // content is not itself a `<tool_response>…</tool_response>` wrapper (a client replaying tool
    // output as a user turn must not reset the reasoning boundary). Default len-1 when there is
    // no such turn, exactly as the jinja's namespace initializer does.
    let last_query_index = turns
        .iter()
        .enumerate()
        .rev()
        .find(|(_, t)| {
            t.role == "user"
                && !(t.content.starts_with("<tool_response>")
                    && t.content.ends_with("</tool_response>"))
        })
        .map(|(i, _)| i)
        .unwrap_or(turns.len().saturating_sub(1));

    for (i, turn) in turns.iter().enumerate() {
        let content = &turn.content; // NOT trimmed: this template applies no `|trim`
        match turn.role.as_str() {
            // the leading system turn lives in the header above; later ones are body turns.
            "system" if i == 0 => {}
            "system" | "user" => {
                out.push_str("<|im_start|>");
                out.push_str(&turn.role);
                out.push('\n');
                out.push_str(content);
                out.push_str("<|im_end|>\n");
            }
            "assistant" => {
                // Split an inline `<think>…</think>` out of content, mirroring the jinja's
                // string surgery exactly: reasoning = text before the FIRST `</think>`, with
                // trailing newlines stripped, then everything after the LAST `<think>` in that
                // prefix, with leading newlines stripped; body = after the LAST `</think>`,
                // leading newlines stripped.
                let (reasoning, body): (String, &str) = match content.find("</think>") {
                    Some(first) => {
                        let pre = content[..first].trim_end_matches('\n');
                        let pre = match pre.rfind("<think>") {
                            Some(o) => &pre[o + "<think>".len()..],
                            None => pre,
                        };
                        let last = content.rfind("</think>").unwrap();
                        (
                            pre.trim_start_matches('\n').to_string(),
                            content[last + "</think>".len()..].trim_start_matches('\n'),
                        )
                    }
                    None => (String::new(), content.as_str()),
                };
                out.push_str("<|im_start|>assistant\n");
                if i > last_query_index {
                    out.push_str("<think>\n");
                    out.push_str(&reasoning);
                    out.push_str("\n</think>\n");
                }
                out.push_str(body);
                // NO separator before or between calls (the qwen arm's `\n\n`/`\n` would corrupt).
                for call in &turn.tool_calls {
                    out.push_str("<tool_call>\n<function=");
                    out.push_str(&call.name);
                    out.push_str(">\n");
                    for (key, value) in &call.params {
                        out.push_str("<parameter=");
                        out.push_str(key);
                        out.push_str(">\n");
                        out.push_str(value);
                        out.push_str("\n</parameter>\n");
                    }
                    out.push_str("</function>\n</tool_call>");
                }
                out.push_str("<|im_end|>\n");
            }
            "tool" => {
                // own role, and consecutive tool turns share ONE `tool_response` turn.
                if i == 0 || turns[i - 1].role != "tool" {
                    out.push_str("<|im_start|>tool_response\n");
                }
                out.push_str("<tool_response>");
                out.push_str(content);
                out.push_str("</tool_response>");
                if i + 1 >= turns.len() || turns[i + 1].role != "tool" {
                    out.push_str("<|im_end|>\n");
                }
            }
            other => {
                // the jinja drops this turn entirely; see the divergence note above.
                out.push_str("<|im_start|>");
                out.push_str(other);
                out.push('\n');
                out.push_str(content);
                out.push_str("<|im_end|>\n");
            }
        }
    }

    if add_generation_prompt {
        out.push_str("<|im_start|>assistant\n<think>\n");
    }
    out
}

/// Text-only compatibility entry for the Hy3 `chat_template.jinja`.
/// `effort` is the template's own `reasoning_effort` input — `"no_think"` / `"low"` /
/// `"high"`, its full accepted set (the jinja `raise_exception`s on anything else; undefined
/// defaults to `'no_think'`, so callers with no opinion pass `"no_think"`):
///   - `{bos}{system…}<｜reasoning_mode:opensource｜>reasoning_effort:{effort}` header
///     (system turns concatenate into the header, before any user turn);
///   - `user`      -> `<｜hy_User:opensource｜>{content}`
///   - `assistant` -> `<｜hy_Assistant:opensource｜><think:opensource></think:opensource>{content}<｜hy_eos:opensource｜>`
///     (non-last turns; history turns render CLOSED think at every effort — the template
///     opens only turns past `last_user_index`, and OpenAI history carries no reasoning);
///   - generation prompt: `<｜hy_Assistant:opensource｜><think:opensource></think:opensource>`
///     at no_think, `…<think:opensource>` (OPEN think) at low/high.
///     Content is NOT trimmed (the Hy3 template applies no `|trim`). Goldens: rendered from the
///     pinned tencent/Hy3 template (sha 7fc351fe…, snapshot 716aa724) by
///     `research/step-sku-20260807/render-thinking-goldens.py`.
fn apply_hy3_template(
    messages: &[(&str, &str)],
    add_generation_prompt: bool,
    effort: &str,
) -> String {
    let turns: Vec<Turn> = messages
        .iter()
        .map(|(role, content)| Turn {
            role: (*role).to_string(),
            content: (*content).to_string(),
            ..Default::default()
        })
        .collect();
    apply_hy3_template_tools(&turns, add_generation_prompt, &[], effort)
}

/// Exact text/tools reproduction of Tencent HY3's pinned shipping template
/// (`chat_template.jinja` SHA-256 7fc351fe...). `tools_json` entries are the request's
/// function objects serialized by the HTTP layer in client order, matching jinja `tojson`.
fn apply_hy3_template_tools(
    turns: &[Turn],
    add_generation_prompt: bool,
    tools_json: &[String],
    effort: &str,
) -> String {
    const BOS: &str = "<\u{ff5c}hy_begin_of_sentence:opensource\u{ff5c}>";
    const USER: &str = "<\u{ff5c}hy_User:opensource\u{ff5c}>";
    const ASSISTANT: &str = "<\u{ff5c}hy_Assistant:opensource\u{ff5c}>";
    const EOS: &str = "<\u{ff5c}hy_eos:opensource\u{ff5c}>";
    const REASONING: &str = "<\u{ff5c}reasoning_mode:opensource\u{ff5c}>";
    const THINK_BEGIN: &str = "<think:opensource>";
    const THINK_END: &str = "</think:opensource>";
    const TOOLCALLS_BEGIN: &str = "<tool_calls:opensource>";
    const TOOLCALLS_END: &str = "</tool_calls:opensource>";
    const TOOLCALL_BEGIN: &str = "<tool_call:opensource>";
    const TOOLCALL_END: &str = "</tool_call:opensource>";
    const TOOL_SEP: &str = "<tool_sep:opensource>";
    const ARGKEY_BEGIN: &str = "<arg_key:opensource>";
    const ARGKEY_END: &str = "</arg_key:opensource>";
    const ARGVALUE_BEGIN: &str = "<arg_value:opensource>";
    const ARGVALUE_END: &str = "</arg_value:opensource>";
    const TOOLRESPONSES_BEGIN: &str = "<tool_responses:opensource>";
    const TOOLRESPONSES_END: &str = "</tool_responses:opensource>";
    const TOOLRESPONSE_BEGIN: &str = "<tool_response:opensource>";
    const TOOLRESPONSE_END: &str = "</tool_response:opensource>";

    debug_assert!(
        matches!(effort, "no_think" | "low" | "high"),
        "hy3 reasoning_effort must be no_think|low|high, got {effort:?}"
    );
    let mut out = String::from(BOS);
    let mut system_prompt = String::new();
    for turn in turns.iter().filter(|turn| turn.role == "system") {
        system_prompt.push_str(&turn.content);
    }
    out.push_str(&system_prompt);
    if tools_json.is_empty() {
        out.push_str(REASONING);
        out.push_str("reasoning_effort:");
        out.push_str(effort);
    } else {
        if !system_prompt.is_empty() {
            out.push_str(
                "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.",
            );
        } else {
            out.push_str(
                "# Tools\n\nYou may call one or more functions to assist with the user query.",
            );
        }
        out.push_str(
            "\n\nYou are provided with function signatures within <tools></tools> XML tags:",
        );
        out.push_str("\n<tools>\n");
        for (index, tool) in tools_json.iter().enumerate() {
            if index > 0 {
                out.push('\n');
            }
            out.push_str(tool);
        }
        out.push_str("\n</tools>\n\n");
        out.push_str("For function call returns, you should first print ");
        out.push_str(TOOLCALLS_BEGIN);
        out.push('\n');
        out.push_str("For each function call, you should return object like:\n");
        out.push_str(TOOLCALL_BEGIN);
        out.push_str("{function-name}");
        out.push_str(TOOL_SEP);
        out.push('\n');
        out.push_str(ARGKEY_BEGIN);
        out.push_str("{arg-key-1}");
        out.push_str(ARGKEY_END);
        out.push('\n');
        out.push_str(ARGVALUE_BEGIN);
        out.push_str("{arg-value-1}");
        out.push_str(ARGVALUE_END);
        out.push('\n');
        out.push_str(ARGKEY_BEGIN);
        out.push_str("{arg-key-2}");
        out.push_str(ARGKEY_END);
        out.push('\n');
        out.push_str(ARGVALUE_BEGIN);
        out.push_str("{arg-value-2}");
        out.push_str(ARGVALUE_END);
        out.push_str("\n...\n");
        out.push_str(TOOLCALL_END);
        out.push('\n');
        out.push_str("At the end of function call returns, you should print ");
        out.push_str(TOOLCALLS_END);
        out.push_str(REASONING);
        out.push_str("reasoning_effort:");
        out.push_str(effort);
    }

    let last_user = turns.iter().rposition(|turn| turn.role == "user");
    let preserve_thinking = !tools_json.is_empty();
    let mut previous_is_tool = false;
    let mut tool_run_first = true;
    for (index, turn) in turns.iter().enumerate() {
        match turn.role.as_str() {
            "user" => {
                if previous_is_tool {
                    out.push_str(TOOLRESPONSES_END);
                }
                out.push_str(USER);
                out.push_str(&turn.content);
                previous_is_tool = false;
            }
            "assistant" => {
                if previous_is_tool {
                    out.push_str(TOOLRESPONSES_END);
                }
                let keep_reasoning = preserve_thinking || last_user.is_none_or(|last| index > last);
                out.push_str(ASSISTANT);
                out.push_str(THINK_BEGIN);
                if keep_reasoning && let Some(reasoning) = turn.reasoning.as_deref() {
                    out.push_str(reasoning);
                }
                out.push_str(THINK_END);
                out.push_str(&turn.content);
                if turn.tool_calls.is_empty() {
                    if index + 1 < turns.len() {
                        out.push_str(EOS);
                    }
                } else {
                    tool_run_first = true;
                    out.push_str(TOOLCALLS_BEGIN);
                    out.push('\n');
                    for call in &turn.tool_calls {
                        out.push_str(TOOLCALL_BEGIN);
                        out.push_str(&call.name);
                        out.push_str(TOOL_SEP);
                        out.push('\n');
                        for (key, value) in &call.params {
                            out.push_str(ARGKEY_BEGIN);
                            out.push_str(key);
                            out.push_str(ARGKEY_END);
                            out.push('\n');
                            out.push_str(ARGVALUE_BEGIN);
                            out.push_str(value);
                            out.push_str(ARGVALUE_END);
                            out.push('\n');
                        }
                        out.push_str(TOOLCALL_END);
                        out.push('\n');
                    }
                    out.push_str(TOOLCALLS_END);
                    out.push_str(EOS);
                }
                previous_is_tool = false;
            }
            "tool" => {
                previous_is_tool = true;
                if tool_run_first {
                    out.push_str(TOOLRESPONSES_BEGIN);
                    out.push('\n');
                    tool_run_first = false;
                }
                out.push_str(TOOLRESPONSE_BEGIN);
                out.push('\n');
                out.push_str(&turn.content);
                out.push('\n');
                out.push_str(TOOLRESPONSE_END);
                out.push('\n');
            }
            _ => {} // system handled in the header; unknown roles are ignored by the template
        }
    }
    if previous_is_tool {
        out.push_str(TOOLRESPONSES_END);
    }
    let last_is_assistant = turns.last().is_some_and(|turn| turn.role == "assistant");
    if add_generation_prompt && !last_is_assistant {
        out.push_str(ASSISTANT);
        out.push_str(THINK_BEGIN);
        if effort == "no_think" {
            out.push_str(THINK_END); // low/high leave the think channel OPEN (the golden)
        }
    }
    out
}

/// gemma4 turn dialect (text-only path of the GGUF template, verified against the dumped
/// jinja — sha 36e3a42e…, goldens `research/step-sku-20260807/raw/thinking-goldens.txt`):
/// roles map assistant->model; each turn = `<|turn>{role}\n{content|trim}<turn|>\n`.
///
/// THINKING is `enable_thinking`, and its default is OFF (`enable_thinking | default(false)`)
/// — the inverse of the qwen class:
///   - thinking OFF (default): generation prompt = `<|turn>model\n<|channel>thought\n<channel|>`
///     (the CLOSED thought channel — the model may not think);
///   - thinking ON: a `<|think|>\n` token is injected at the very top of the FIRST system
///     turn (a system turn is CREATED if the request has none), and the generation prompt is
///     the bare `<|turn>model\n` — the thought channel is left to the model.
fn apply_gemma4_template(
    messages: &[(&str, &str)],
    add_generation_prompt: bool,
    thinking: bool,
) -> String {
    let mut out = String::new();
    let mut msgs = messages;
    // System header block: fires when thinking is on OR a leading system turn exists.
    let leading_system = msgs.first().filter(|(r, _)| *r == "system");
    if thinking || leading_system.is_some() {
        out.push_str("<|turn>system\n");
        if thinking {
            out.push_str("<|think|>\n");
        }
        if let Some((_, content)) = leading_system {
            out.push_str(content.trim());
            msgs = &msgs[1..];
        }
        out.push_str("<turn|>\n");
    }
    for (role, content) in msgs {
        let role = if *role == "assistant" { "model" } else { role };
        out.push_str("<|turn>");
        out.push_str(role);
        out.push('\n');
        out.push_str(content.trim());
        out.push_str("<turn|>\n");
    }
    if add_generation_prompt {
        out.push_str("<|turn>model\n");
        if !thinking {
            out.push_str("<|channel>thought\n<channel|>");
        }
    }
    out
}

// ---- GLM-5.3-Flash (`glm5_next`) dialect ---------------------------------------------------
// A port of the checkpoint's own chat_template.jinja, banked byte-identical at
// research/glm53-flash-bringup-20260827/chat_template.jinja. The jinja is the LAW; byte parity
// is pinned by research/glm53-flash-bringup-20260827/surface-fixtures (the
// `glm5_fixtures_match_the_vendor_jinja` test in memra-server renders the vendor jinja under
// jinja2 and asserts equality, the same oracle discipline the gemma4/dsv4 arms carry).
//
// NOTHING about this dialect is ChatML. Before this arm existed, the template's `<think>` +
// `add_generation_prompt` markers matched the qwen detector and every GLM chat request rendered
// `<|im_start|>` turns — tokens that are not in this checkpoint's special vocabulary at all
// (its extra_special_tokens are `[gMASK] <sop> <|system|> <|user|> <|assistant|>
// <|observation|>` …), so the frame tokenized as ordinary text and the prompt was off the
// model's distribution end to end. It "worked" only because GLM follows the qwen tool-format
// instruction it was handed in-context: the GGUF-template-mint failure mode exactly — fluent,
// and invisible without a byte oracle.

/// GLM-5.3-Flash template detector: the `[gMASK]<sop>` sequence head AND the `<|observation|>`
/// tool-result turn prefix. Both are unique to the GLM dialect among every committed template
/// (`rg` over research/**/*.jinja finds them only in the glm53 lane), and neither can appear in
/// a ChatML/gemma/hy3/dsv4 template by accident. Shared by the renderer dispatch, the tools
/// probe and the worker caps, so one law keys all three.
pub fn template_is_glm5(t: &str) -> bool {
    t.contains("[gMASK]<sop>") && t.contains("<|observation|>")
}

/// The GLM-5.3-Flash REASONING-EFFORT ladder, resolved exactly as the template resolves it:
///
/// ```jinja
/// {%- set effective_reasoning_effort = reasoning_effort
///        if reasoning_effort is defined and reasoning_effort in ['low', 'high']
///        else 'max' -%}
/// <|system|>Reasoning Effort: {{ effective_reasoning_effort | capitalize }}
/// ```
///
/// So the model's own rungs are **low < high < max**, `max` is its DEFAULT, the line is always
/// rendered, and there is no off switch anywhere in the template (which is why an explicit
/// client off-request 400s upstream — `ModelCaps::qwen_think && !think_switch`).
///
/// The canonical serve levels map onto those three rungs:
///
/// | client `reasoning_effort` | rendered line |
/// |---|---|
/// | (absent)                  | `Reasoning Effort: Max` — the template's own default |
/// | `low`                     | `Reasoning Effort: Low` |
/// | `medium`                  | `Reasoning Effort: Low` — see below |
/// | `high`                    | `Reasoning Effort: High` |
/// | `xhigh` / `max` / `ultra` | `Reasoning Effort: Max` — the real tier above high |
///
/// `medium` CLAMPS DOWN to `low` rather than falling through to the template's `else 'max'`.
/// The else-arm is the template's *unset* default, not its medium rung: routing a client's
/// `medium` there would answer a request to reason LESS with the model's deepest setting, the
/// never-corrupt-clamp law read backwards. hy3 already clamps `medium` -> `low` for the same
/// reason (its ladder has no medium either); GLM differs from hy3 only in having a rung above
/// `high`, which is why `canonical_effort_for` must not fold `max` into `high` for this model.
fn glm5_effort_level(reasoning_effort: Option<&str>) -> Result<&'static str, String> {
    match reasoning_effort {
        None => Ok("Max"),
        Some("low") | Some("medium") => Ok("Low"),
        Some("high") => Ok("High"),
        Some("max") | Some("xhigh") | Some("ultra") => Ok("Max"),
        // `none`/`minimal` never arrive here: the serve path refuses an explicit off-request on
        // this template (no enable_thinking) and maps a deployment default of none/minimal to
        // `ThinkMode::NoThink` + level "low", which lands on the Low rung above. Anything else
        // is a level this model was never trained on, and the template's own `else` would have
        // silently rendered Max for it — the accepted-and-ignored shape this arm exists to end.
        Some(other) => Err(format!(
            "reasoning effort {other:?} is not a level this chat template defines \
             (low|medium|high|max; the template's own ladder is low|high|max with max the \
             default)"
        )),
    }
}

/// One tool DECLARATION, rendered as the template's `tool_to_json` macro renders it: the
/// unwrapped `function` object as `json.dumps(ensure_ascii=False)` in INSERTION key order, with
/// the two client-side-only keys `defer_loading` and `strict` dropped. (`strict` is the one that
/// actually shows up: stock OpenAI-shaped clients put it inside `function`.)
fn glm5_tool_json(func: &Val) -> String {
    let mut out = String::new();
    let Some(obj) = as_obj(func) else {
        py_json(func, &mut out);
        return out;
    };
    out.push('{');
    let mut first = true;
    for (k, v) in obj {
        if k == "defer_loading" || k == "strict" {
            continue;
        }
        if !first {
            out.push_str(", ");
        }
        first = false;
        out.push('"');
        py_json_escape(k, &mut out);
        out.push_str("\": ");
        py_json(v, &mut out);
    }
    out.push('}');
    out
}

/// The message id a tool-result / tool-call turn is keyed by — the template's `id_of` macro
/// (`obj.tool_call_id` first, then `obj.id`). An empty string is jinja-falsey, so it is `None`
/// here too.
fn glm5_id_of(id: Option<&str>) -> Option<&str> {
    id.filter(|s| !s.is_empty())
}

/// Can this run of consecutive `tool` turns be re-ordered onto the preceding assistant turn's
/// `tool_calls` order? The template's `can_sort` predicate, reproduced in its own order:
/// the run must be immediately preceded by an assistant turn WITH tool_calls; every result in
/// the run must carry an id that is unique within the run and present among those calls; and
/// every call must carry an id, unique among the calls. Any miss renders the run in message
/// order instead.
fn glm5_can_sort(results: &[&Turn], calls: &[ToolCall]) -> bool {
    if calls.is_empty() {
        return false;
    }
    for (i, r) in results.iter().enumerate() {
        let Some(id) = glm5_id_of(r.tool_call_id.as_deref()) else {
            return false;
        };
        if results
            .iter()
            .enumerate()
            .any(|(j, o)| j != i && glm5_id_of(o.tool_call_id.as_deref()) == Some(id))
        {
            return false;
        }
        if !calls
            .iter()
            .any(|c| glm5_id_of(c.id.as_deref()) == Some(id))
        {
            return false;
        }
    }
    for (i, c) in calls.iter().enumerate() {
        let Some(id) = glm5_id_of(c.id.as_deref()) else {
            return false;
        };
        if calls
            .iter()
            .enumerate()
            .any(|(j, o)| j != i && glm5_id_of(o.id.as_deref()) == Some(id))
        {
            return false;
        }
    }
    true
}

/// GLM-5.3-Flash (`glm5_next`) chat template — the vendor jinja, reproduced.
///
/// Shape, in the template's own order:
///
/// ```text
/// [gMASK]<sop>
/// <|system|>Reasoning Effort: {Low|High|Max}                      (always, no off switch)
/// <|system|>\n# Tools\n\n…<tools>\n\n{json}\n\n\n</tools>\n\n…     (only when tools present)
/// <|user|>{content}                                               (content NOT trimmed)
/// <|system|>{content}                                             (anywhere, NOT trimmed)
/// <|assistant|><think>{reasoning}</think>{content.strip()}
///     \n<tool_call>NAME<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>…\n
/// <|observation|><tool_response>{r1}</tool_response><tool_response>{r2}</tool_response>
/// <|assistant|><think>                                            (add_generation_prompt)
/// ```
///
/// Load-bearing details, each measured against the jinja rather than assumed:
///
/// - **BOS is the template's own literal.** `[gMASK]<sop>` is emitted here; the checkpoint's
///   tokenizer_config declares no `bos_token` and no `add_bos_token`, so `encode(add_special)`
///   prepends nothing and there is no double-BOS trap (the one the gemma4/step35 arms document).
/// - **The reasoning-effort system line is unconditional** — `effective_reasoning_effort` is
///   always a string, so the `is not none` guard is always true. There is no prompt shape of
///   this model without it.
/// - **`<think>` is ALWAYS replayed on assistant history**, empty when the turn carries no
///   reasoning (`<think></think>`), because `clear_thinking` defaults false and the guard is
///   `(not clear_thinking or …)`. Reasoning comes from the turn's own `reasoning` field, else
///   from an inline `<think>…</think>` span inside the content, which is then stripped out of
///   the content — the same split the template performs.
/// - **Assistant content is `.strip()`ed; user/system content is NOT.** (The qwen arm trims
///   every role; copying that here would have been a silent byte divergence.)
/// - **Tool calls carry no separators**: one `\n` before the first, none between, one after
///   the last. Argument values are strings raw, everything else `json.dumps` — which is exactly
///   the pre-rendering `ToolCall::params` already carries.
/// - **A run of consecutive `tool` turns renders as ONE `<|observation|>` block**, re-ordered
///   onto the preceding assistant turn's `tool_calls` order when every id resolves uniquely
///   (`glm5_can_sort`), in message order otherwise.
///
/// NOT reproduced, because `Turn` cannot express them and no OpenAI/Anthropic/Responses request
/// can produce them: the native `tool_reference` content type (`<tool_response><tools>…`), the
/// list-of-outputs tool message shape (`m.content[i].output`), and the image/video/audio
/// `visible_text` arms (this server serves this model text-only). ONE deliberate divergence,
/// matching the step35 arm's: the vendor's body loop has no `else`, so a role outside
/// {user, assistant, tool, system} renders as NOTHING — the turn silently vanishes. A dropped
/// turn is the worse failure, so an unknown role renders here as a `<|user|>` turn. It cannot
/// fire from the serve surface, whose roles are exactly system/user/assistant/tool (`developer`
/// is normalized to `system` upstream).
fn apply_glm5_template(
    turns: &[Turn],
    add_generation_prompt: bool,
    tools_struct: &[Val],
    reasoning_effort: Option<&str>,
) -> Result<String, String> {
    let mut out = String::new();
    out.push_str("[gMASK]<sop>");
    out.push_str("<|system|>Reasoning Effort: ");
    out.push_str(glm5_effort_level(reasoning_effort)?);
    if !tools_struct.is_empty() {
        out.push_str(
            "<|system|>\n# Tools\n\nYou may call one or more functions to assist with the \
             user query.\n\nYou are provided with function signatures within <tools></tools> \
             XML tags:\n<tools>\n",
        );
        for f in tools_struct {
            out.push('\n');
            out.push_str(&glm5_tool_json(f));
            out.push_str("\n\n");
        }
        out.push_str(
            "\n</tools>\n\nFor each function call, output the function name and arguments \
             within the following XML format:\n<tool_call>{function-name}<arg_key>{arg-key-1}\
             </arg_key><arg_value>{arg-value-1}</arg_value><arg_key>{arg-key-2}</arg_key>\
             <arg_value>{arg-value-2}</arg_value>...</tool_call>",
        );
    }
    for (i, turn) in turns.iter().enumerate() {
        match turn.role.as_str() {
            "assistant" => {
                out.push_str("<|assistant|>");
                // `m.reasoning_content is string` first, then the inline `</think>` split —
                // the template's own order.
                let (reasoning, content) = match turn.reasoning.as_deref() {
                    Some(r) => (Some(r), turn.content.as_str()),
                    None => match turn.content.split_once("</think>") {
                        Some((head, tail)) => (
                            Some(head.rsplit("<think>").next().unwrap_or(head)),
                            // jinja `split('</think>')[-1]`: the LAST segment, so a second
                            // `</think>` inside the reply keeps only what follows it.
                            turn.content.rsplit("</think>").next().unwrap_or(tail),
                        ),
                        None => (None, turn.content.as_str()),
                    },
                };
                out.push_str("<think>");
                out.push_str(reasoning.unwrap_or(""));
                out.push_str("</think>");
                out.push_str(content.trim());
                if !turn.tool_calls.is_empty() {
                    out.push('\n');
                    for call in &turn.tool_calls {
                        out.push_str("<tool_call>");
                        out.push_str(&call.name);
                        for (key, value) in &call.params {
                            out.push_str("<arg_key>");
                            out.push_str(key);
                            out.push_str("</arg_key><arg_value>");
                            out.push_str(value);
                            out.push_str("</arg_value>");
                        }
                        out.push_str("</tool_call>");
                    }
                    out.push('\n');
                }
            }
            "tool" => {
                // Only the FIRST turn of a run emits: it renders the whole `<|observation|>`
                // block. The vendor's `if loop.first or previous.role != 'tool'` has no else,
                // so every following turn of the run renders nothing at all.
                if i > 0 && turns[i - 1].role == "tool" {
                    continue;
                }
                let end = turns[i..].iter().take_while(|t| t.role == "tool").count();
                let run: Vec<&Turn> = turns[i..i + end].iter().collect();
                let calls: &[ToolCall] = if i > 0 && turns[i - 1].role == "assistant" {
                    &turns[i - 1].tool_calls
                } else {
                    &[]
                };
                out.push_str("<|observation|>");
                if glm5_can_sort(&run, calls) {
                    for call in calls {
                        for r in &run {
                            if glm5_id_of(r.tool_call_id.as_deref())
                                == glm5_id_of(call.id.as_deref())
                            {
                                out.push_str("<tool_response>");
                                out.push_str(&r.content);
                                out.push_str("</tool_response>");
                            }
                        }
                    }
                } else {
                    for r in &run {
                        out.push_str("<tool_response>");
                        out.push_str(&r.content);
                        out.push_str("</tool_response>");
                    }
                }
            }
            "system" => {
                out.push_str("<|system|>");
                out.push_str(&turn.content);
            }
            // `user`, and the documented unknown-role divergence.
            _ => {
                out.push_str("<|user|>");
                out.push_str(&turn.content);
            }
        }
    }
    if add_generation_prompt {
        out.push_str("<|assistant|><think>");
    }
    Ok(out)
}

/// A template carries a tools branch iff it has the qwen/step/HY3 `<tools>` block, the gemma4
/// tooluse dialect (both the `<|turn>` turn framing and the `<|tool>` declaration marker), the
/// dsv4 protocol, or the glm5 `<tool_call>` grammar. Shared by the renderer dispatch and the
/// worker caps probe.
pub fn template_has_tools_branch(t: &str) -> bool {
    template_is_dsv4(t)
        || template_is_glm5(t)
        || t.contains("<tools>")
        || (t.contains("<|turn>") && t.contains("<|tool>"))
}

/// deepseek-v4 (`encoding_dsv4`) template detector: the `<｜Assistant｜>` turn prefix AND the
/// `｜DSML｜` tool-call markup token. Both are unique to the DeepSeek-V4 chat dialect (`｜`
/// is U+FF5C, `<think>` alone would be ambiguous with the qwen class). Shared by the renderer
/// dispatch, the tools-branch probe, and the worker caps.
pub fn template_is_dsv4(t: &str) -> bool {
    t.contains("<\u{ff5c}Assistant\u{ff5c}>") && t.contains("\u{ff5c}DSML\u{ff5c}")
}

// ---- gemma4 tooluse dialect ---------------------------------------------------------------
// A faithful port of research/gemma4-tools-20260817/official-tooluse-template.jinja (extracted
// byte-identical from the official Q8_0-MTP GGUF — the served trunk). The jinja is the LAW;
// byte parity is pinned by research/gemma4-tools-20260817/fixtures (the `gemma4_tools_fixtures`
// test in memra-server renders the official jinja under jinja2 and asserts equality). Deviation
// from the jinja: an unresolved tool-response name falls back to "unknown" instead of crashing
// on `str + None` (the jinja's `.get('name') | default('unknown')` renders None, then the
// concat raises) — unreachable from OpenAI histories, where the id always resolves.

/// jinja `| dictsort`: case-insensitive by key, STABLE (ties keep insertion order).
fn dictsort(pairs: &[(String, Val)]) -> Vec<&(String, Val)> {
    let mut v: Vec<&(String, Val)> = pairs.iter().collect();
    v.sort_by_key(|a| a.0.to_lowercase());
    v
}

/// jinja `format_argument(argument, escape_keys)`: strings wrapped in `<|"|>`, bools `true`/
/// `false`, mappings `{k:v,...}` (keys bare unless `escape_keys`, dictsorted, recursive),
/// sequences `[v,...]`, null -> `None` (jinja `{{ none }}`), numbers bare.
fn format_argument(v: &Val, escape_keys: bool) -> String {
    match v {
        Val::Str(s) => format!("<|\"|>{s}<|\"|>"),
        Val::Bool(b) => if *b { "true" } else { "false" }.to_string(),
        Val::Obj(pairs) => {
            let mut out = String::from("{");
            for (i, (k, val)) in dictsort(pairs).iter().map(|p| (&p.0, &p.1)).enumerate() {
                if i > 0 {
                    out.push(',');
                }
                if escape_keys {
                    out.push_str(&format!("<|\"|>{k}<|\"|>"));
                } else {
                    out.push_str(k);
                }
                out.push(':');
                out.push_str(&format_argument(val, escape_keys));
            }
            out.push('}');
            out
        }
        Val::Arr(items) => {
            let mut out = String::from("[");
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&format_argument(item, escape_keys));
            }
            out.push(']');
            out
        }
        Val::Null => "None".to_string(),
        Val::Num(s) => s.clone(),
    }
}

/// jinja `strip_thinking(text)`: drop every `<|channel>...<channel|>` span, then `| trim`.
/// Split on `<channel|>`; for each part, keep everything before a `<|channel>` (dropping the
/// channel body), else keep the whole part.
fn strip_thinking(text: &str) -> String {
    let mut result = String::new();
    for part in text.split("<channel|>") {
        match part.find("<|channel>") {
            Some(o) => result.push_str(&part[..o]),
            None => result.push_str(part),
        }
    }
    result.trim().to_string()
}

fn val_get<'a>(obj: &'a [(String, Val)], key: &str) -> Option<&'a Val> {
    obj.iter().find(|(k, _)| k == key).map(|(_, v)| v)
}
fn as_obj(v: &Val) -> Option<&[(String, Val)]> {
    match v {
        Val::Obj(p) => Some(p),
        _ => None,
    }
}
fn as_str(v: &Val) -> Option<&str> {
    match v {
        Val::Str(s) => Some(s),
        _ => None,
    }
}
/// jinja truthiness for `if value[...]`: None/false/""/[]/{} are falsy.
fn truthy(v: &Val) -> bool {
    match v {
        Val::Null => false,
        Val::Bool(b) => *b,
        Val::Str(s) => !s.is_empty(),
        Val::Num(s) => s != "0" && s != "0.0",
        Val::Arr(a) => !a.is_empty(),
        Val::Obj(o) => !o.is_empty(),
    }
}

/// jinja comma helper: emit ',' iff a prior element was written in THIS property object, then
/// mark that at least one has been written.
fn comma(out: &mut String, add: &mut bool) {
    if *add {
        out.push(',');
    } else {
        *add = true;
    }
}

/// jinja `format_parameters(properties, _required_unused, filter_keys)`. The second jinja arg
/// (`required`) is never referenced in the macro body, so it is dropped here.
fn format_parameters(out: &mut String, props: &[(String, Val)], filter_keys: bool) {
    const STANDARD: [&str; 5] = ["description", "type", "properties", "required", "nullable"];
    let mut found_first = false;
    for (key, value) in dictsort(props).iter().map(|p| (&p.0, &p.1)) {
        if filter_keys && STANDARD.contains(&key.as_str()) {
            continue;
        }
        if found_first {
            out.push(',');
        }
        found_first = true;
        out.push_str(key);
        out.push_str(":{");
        let vobj = as_obj(value);
        let mut add = false;
        // description
        if let Some(d) = vobj
            .and_then(|o| val_get(o, "description"))
            .filter(|d| truthy(d))
        {
            out.push_str("description:<|\"|>");
            out.push_str(as_str(d).unwrap_or(""));
            out.push_str("<|\"|>");
            add = true;
        }
        let ty_up = vobj
            .and_then(|o| val_get(o, "type"))
            .and_then(as_str)
            .map(|s| s.to_uppercase());
        match ty_up.as_deref() {
            Some("STRING") => {
                if let Some(en) = vobj.and_then(|o| val_get(o, "enum")).filter(|e| truthy(e)) {
                    comma(out, &mut add);
                    out.push_str("enum:");
                    out.push_str(&format_argument(en, true));
                }
            }
            Some("ARRAY") => {
                if let Some(items) = vobj
                    .and_then(|o| val_get(o, "items"))
                    .filter(|it| matches!(it, Val::Obj(o) if !o.is_empty()))
                {
                    comma(out, &mut add);
                    out.push_str("items:{");
                    format_items(out, as_obj(items).unwrap());
                    out.push('}');
                }
            }
            _ => {}
        }
        // nullable
        if vobj
            .and_then(|o| val_get(o, "nullable"))
            .is_some_and(truthy)
        {
            comma(out, &mut add);
            out.push_str("nullable:true");
        }
        // OBJECT: nested properties + required
        if ty_up.as_deref() == Some("OBJECT") {
            if let Some(sub) = vobj.and_then(|o| val_get(o, "properties")).and_then(as_obj) {
                comma(out, &mut add);
                out.push_str("properties:{");
                format_parameters(out, sub, false);
                out.push('}');
            } else if let Some(o) = vobj {
                // no explicit `properties`: treat the value's own keys as sub-properties,
                // filtering the standard schema keys (jinja `filter_keys=true` branch).
                comma(out, &mut add);
                out.push_str("properties:{");
                format_parameters(out, o, true);
                out.push('}');
            }
            if let Some(req) = vobj
                .and_then(|o| val_get(o, "required"))
                .filter(|r| truthy(r))
            {
                comma(out, &mut add);
                out.push_str("required:[");
                push_str_list(out, req);
                out.push(']');
            }
        }
        // closing `type:<|"|>UPPER<|"|>}` (always) — carries a leading comma iff anything above.
        comma(out, &mut add);
        out.push_str("type:<|\"|>");
        out.push_str(ty_up.as_deref().unwrap_or(""));
        out.push_str("<|\"|>}");
    }
}

/// The ARRAY `items` mapping loop: dictsorts item keys, skips None values, and renders
/// properties/required/type specially, else generic `key:format_argument(value)`.
fn format_items(out: &mut String, items: &[(String, Val)]) {
    let mut found_first = false;
    for (k, v) in dictsort(items).iter().map(|p| (&p.0, &p.1)) {
        if matches!(v, Val::Null) {
            continue;
        }
        if found_first {
            out.push(',');
        }
        found_first = true;
        match k.as_str() {
            "properties" => {
                out.push_str("properties:{");
                if let Some(o) = as_obj(v) {
                    format_parameters(out, o, false);
                }
                out.push('}');
            }
            "required" => {
                out.push_str("required:[");
                push_str_list(out, v);
                out.push(']');
            }
            "type" => {
                out.push_str("type:");
                match v {
                    Val::Str(s) => {
                        out.push_str(&format_argument(&Val::Str(s.to_uppercase()), true))
                    }
                    Val::Arr(a) => {
                        let upper: Vec<Val> = a
                            .iter()
                            .map(|x| Val::Str(as_str(x).unwrap_or("").to_uppercase()))
                            .collect();
                        out.push_str(&format_argument(&Val::Arr(upper), true));
                    }
                    other => out.push_str(&format_argument(other, true)),
                }
            }
            _ => {
                out.push_str(k);
                out.push(':');
                out.push_str(&format_argument(v, true));
            }
        }
    }
}

/// `[<|"|>a<|"|>,<|"|>b<|"|>]` body (without the brackets) from a Val::Arr of strings.
fn push_str_list(out: &mut String, v: &Val) {
    if let Val::Arr(items) = v {
        for (i, item) in items.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str("<|\"|>");
            out.push_str(as_str(item).unwrap_or(""));
            out.push_str("<|\"|>");
        }
    }
}

/// jinja `format_function_declaration(tool_data)` — `func` is the tool's `function` object.
fn format_function_declaration(func: &[(String, Val)]) -> String {
    let mut out = String::new();
    out.push_str("declaration:");
    out.push_str(val_get(func, "name").and_then(as_str).unwrap_or(""));
    out.push_str("{description:<|\"|>");
    out.push_str(val_get(func, "description").and_then(as_str).unwrap_or(""));
    out.push_str("<|\"|>");
    if let Some(params) = val_get(func, "parameters").filter(|p| truthy(p)) {
        let pobj = as_obj(params);
        out.push_str(",parameters:{");
        if let Some(props) = pobj
            .and_then(|o| val_get(o, "properties"))
            .filter(|p| truthy(p))
            .and_then(as_obj)
        {
            out.push_str("properties:{");
            format_parameters(&mut out, props, false);
            out.push_str("},");
        }
        if let Some(req) = pobj
            .and_then(|o| val_get(o, "required"))
            .filter(|r| truthy(r))
        {
            out.push_str("required:[");
            push_str_list(&mut out, req);
            out.push_str("],");
        }
        if let Some(ty) = pobj.and_then(|o| val_get(o, "type")).filter(|t| truthy(t)) {
            out.push_str("type:<|\"|>");
            out.push_str(&as_str(ty).unwrap_or("").to_uppercase());
            out.push_str("<|\"|>}");
        }
    }
    if let Some(resp) = val_get(func, "response").and_then(as_obj) {
        out.push_str(",response:{");
        if let Some(d) = val_get(resp, "description").filter(|d| truthy(d)) {
            out.push_str("description:<|\"|>");
            out.push_str(as_str(d).unwrap_or(""));
            out.push_str("<|\"|>,");
        }
        if val_get(resp, "type")
            .and_then(as_str)
            .map(|s| s.to_uppercase())
            == Some("OBJECT".into())
        {
            out.push_str("type:<|\"|>OBJECT<|\"|>}");
        }
    }
    out.push('}');
    out
}

/// jinja `format_tool_response_block(tool_name, response)`.
fn format_tool_response_block(name: &str, response: &Val) -> String {
    let mut out = String::from("<|tool_response>");
    match response {
        Val::Obj(pairs) => {
            out.push_str("response:");
            out.push_str(name);
            out.push('{');
            for (i, (k, v)) in dictsort(pairs).iter().map(|p| (&p.0, &p.1)).enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(k);
                out.push(':');
                out.push_str(&format_argument(v, false));
            }
            out.push('}');
        }
        other => {
            out.push_str("response:");
            out.push_str(name);
            out.push_str("{value:");
            out.push_str(&format_argument(other, false));
            out.push('}');
        }
    }
    out.push_str("<tool_response|>");
    out
}

/// gemma4 tooluse renderer. `tools` are the tool `function` objects; `thinking` = jinja
/// `enable_thinking`; `closed_tail` = the QAT-trunk variant that emits a closed thought
/// channel on the thinking-off generation prompt (the official served trunk does not). BOS is
/// NOT emitted (encode(add_special) supplies it — the jinja's `{{ bos_token }}` is dropped).
fn apply_gemma4_tools_template(
    turns: &[Turn],
    add_generation_prompt: bool,
    tools: &[Val],
    thinking: bool,
    closed_tail: bool,
) -> String {
    let mut out = String::new();
    let mut prev: Option<&str> = None;
    let mut msgs = turns;
    let is_sys = |r: &str| r == "system" || r == "developer";

    let leading_system = msgs.first().filter(|t| is_sys(&t.role));
    if thinking || !tools.is_empty() || leading_system.is_some() {
        out.push_str("<|turn>system\n");
        if thinking {
            out.push_str("<|think|>\n");
            prev = Some("think");
        }
        if let Some(sys) = leading_system {
            out.push_str(sys.content.trim());
            msgs = &msgs[1..];
        }
        for tool in tools {
            out.push_str("<|tool>");
            if let Some(func) = as_obj(tool) {
                out.push_str(format_function_declaration(func).trim());
            }
            out.push_str("<tool|>");
        }
        if !tools.is_empty() {
            prev = Some("tool");
        }
        out.push_str("<turn|>\n");
    }

    let last_user_idx: isize = msgs
        .iter()
        .enumerate()
        .rev()
        .find(|(_, t)| t.role == "user")
        .map(|(i, _)| i as isize)
        .unwrap_or(-1);

    for (i, m) in msgs.iter().enumerate() {
        if m.role == "tool" {
            continue; // consumed by a preceding assistant's forward-scan
        }
        prev = None;
        let role = if m.role == "assistant" {
            "model"
        } else {
            m.role.as_str()
        };
        let prev_nt_role = (0..i)
            .rev()
            .map(|j| &msgs[j])
            .find(|t| t.role != "tool")
            .map(|t| t.role.as_str());
        let continue_same_model_turn = role == "model" && prev_nt_role == Some("assistant");
        if !continue_same_model_turn {
            out.push_str("<|turn>");
            out.push_str(role);
            out.push('\n');
        }

        // reasoning re-render (tool_calls-carrying assistant after the last user turn)
        if let Some(rt) = m.reasoning.as_deref()
            && !rt.is_empty()
            && (i as isize) > last_user_idx
            && !m.tool_calls.is_empty()
        {
            out.push_str("<|channel>thought\n");
            out.push_str(rt);
            out.push_str("\n<channel|>");
        }

        // tool_calls
        if !m.tool_calls.is_empty() {
            for tc in &m.tool_calls {
                out.push_str("<|tool_call>call:");
                out.push_str(&tc.name);
                out.push('{');
                for (j, (k, v)) in dictsort(&tc.args).iter().map(|p| (&p.0, &p.1)).enumerate() {
                    if j > 0 {
                        out.push(',');
                    }
                    out.push_str(k);
                    out.push(':');
                    out.push_str(&format_argument(v, false));
                }
                out.push_str("}<tool_call|>");
            }
            prev = Some("tool_call");
        }

        // tool responses: native (Google) on the assistant, else OpenAI role:"tool" forward-scan
        let mut tr_flag = false;
        if !m.tool_responses.is_empty() {
            for (name, resp) in &m.tool_responses {
                out.push_str(&format_tool_response_block(name, resp));
                tr_flag = true;
                prev = Some("tool_response");
            }
        } else if !m.tool_calls.is_empty() {
            #[allow(clippy::needless_range_loop)]
            // allow: the explicit index loop keeps the offset arithmetic visible and aligned with the device-side indexing
            for k in (i + 1)..msgs.len() {
                let follow = &msgs[k];
                if follow.role != "tool" {
                    break;
                }
                let mut name = follow
                    .tool_name
                    .clone()
                    .unwrap_or_else(|| "unknown".to_string());
                if let Some(fid) = follow.tool_call_id.as_deref() {
                    for tc in &m.tool_calls {
                        if tc.id.as_deref() == Some(fid) {
                            name = tc.name.clone();
                        }
                    }
                }
                out.push_str(&format_tool_response_block(
                    &name,
                    &Val::Str(follow.content.clone()),
                ));
                tr_flag = true;
                prev = Some("tool_response");
            }
        }

        // content (model content strips thought channels; other roles trim)
        let captured = if role == "model" {
            strip_thinking(&m.content)
        } else {
            m.content.trim().to_string()
        };
        out.push_str(&captured);
        let has_content = !captured.trim().is_empty();

        if prev == Some("tool_call") && !tr_flag {
            out.push_str("<|tool_response>"); // dangling open: calls with no responses yet
        } else if !(tr_flag && !has_content) {
            out.push_str("<turn|>\n");
        }
    }

    if add_generation_prompt && prev != Some("tool_response") && prev != Some("tool_call") {
        out.push_str("<|turn>model\n");
        if closed_tail && !thinking {
            out.push_str("<|channel>thought\n<channel|>");
        }
    }
    out
}

// ---- deepseek-v4 (encoding_dsv4) dialect --------------------------------------------------
// A faithful port of encoding_dsv4.py in BOTH shipped revisions: the preview oracle
// (research/dsv4-template-20260818/ref/encoding/encoding_dsv4.py, sha256 bdbd57c1…) and the
// 0731 oracle (…/ref-0731/encoding/encoding_dsv4.py, sha256 abc0d261…), which differ ONLY in
// the reasoning-effort ladder (full behavioral diff: ENCODING-DIFF.md; selection law:
// `Dsv4Encoding`). The python IS the law; byte parity is pinned by
// research/dsv4-template-20260818/fixtures (preview matrix) + fixtures-0731 (0731 matrix)
// plus the artifact's authoritative encoding/tests/test_output_{1..4} (byte-identical across
// both revisions). See TEMPLATE-SEMANTICS.md for the census + banked ambiguities. Deviation
// from the python: none in the renderer (the parser deviates on malformed spans per house
// policy — see toolcall.rs).

// U+FF5C is the fullwidth vertical line `｜` in every DeepSeek special token; U+2581 the ▁.
const DS_BOS: &str = "<\u{ff5c}begin\u{2581}of\u{2581}sentence\u{ff5c}>";
const DS_EOS: &str = "<\u{ff5c}end\u{2581}of\u{2581}sentence\u{ff5c}>";
const DS_USER: &str = "<\u{ff5c}User\u{ff5c}>";
const DS_ASSISTANT: &str = "<\u{ff5c}Assistant\u{ff5c}>";
const DS_REMINDER: &str = "<\u{ff5c}latest_reminder\u{ff5c}>";
const DS_THINK_START: &str = "<think>";
const DS_THINK_END: &str = "</think>";
const DS_DSML: &str = "\u{ff5c}DSML\u{ff5c}";
// preview encoding_dsv4 REASONING_EFFORT_MAX (E:64-68) == 0731 REASONING_EFFORT_PROMPTS["high"]
// (0731 E:64-77 — same bytes, one ladder rung lower). Ends with "\n\n".
const DS_EFFORT_ABSOLUTE_MAX: &str = "Reasoning Effort: Absolute maximum with no shortcuts permitted.\nYou MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\nExplicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n";
// 0731 REASONING_EFFORT_PROMPTS["max"] (0731 E:70-75) — the new, stronger top rung. The dash
// is U+2014 EM DASH in the source; ends with "\n\n". Not present in the preview encoding.
const DS_EFFORT_BEYOND_MAX: &str = "Reasoning Effort: Beyond maximum \u{2014} exhaustive, relentless, and uncompromising.\nYou MUST reason with the utmost depth and rigor, leaving absolutely nothing to chance: exhaustively decompose the problem into its most fundamental components, trace every causal chain to its root, and resolve the underlying cause rather than any surface symptom.\nDo not stop reasoning until you have independently verified the solution from multiple angles and are certain that no assumption remains unchecked and no error remains undiscovered.\n\n";

/// The reasoning-effort prompt prefix for one render (encoding_dsv4 preview E:260-263 /
/// 0731 E:270-277). `Ok("")` = no prefix. Errs ONLY on the ambiguous cell: an effort level
/// whose bytes differ between the two encodings (`"high"`/`"max"` in thinking mode) with no
/// encoding revision supplied — every other input renders identically under both revisions,
/// so it stays infallible there (the legacy no-effort dispatch relies on that).
///
/// Levels outside the encoding's accepted set (e.g. OpenAI "medium", which neither revision
/// defines) render as the default level, i.e. no prefix — the renderer never corrupts a
/// prompt over a knob the template does not consume (hy3 medium-clamp precedent).
fn dsv4_effort_prefix(
    thinking: bool,
    effort: Option<&str>,
    encoding: Option<Dsv4Encoding>,
) -> Result<&'static str, String> {
    if !thinking {
        // chat mode: no prefix under either encoding (preview E:262 / 0731 E:275 both gate
        // on thinking_mode == "thinking").
        return Ok("");
    }
    match effort {
        // None: preview renders nothing; 0731 defaults None -> "low" -> "" (E:271, E:66).
        // "low": 0731 default rung (no prefix); the preview oracle rejects the string, and
        // rendering no prefix is the only never-corrupt reading (banked, ENCODING-DIFF.md).
        None | Some("low") => Ok(""),
        Some("high") => match encoding {
            Some(Dsv4Encoding::Preview) => Ok(""), // preview law: "high" == None (E:261-263)
            Some(Dsv4Encoding::V0731) => Ok(DS_EFFORT_ABSOLUTE_MAX),
            None => Err(
                "dsv4 reasoning_effort \"high\" renders differently on the preview vs 0731 \
                 encoding and this artifact's encoding revision is unknown (config.json \
                 dspark_* census unavailable) — refusing rather than guessing"
                    .into(),
            ),
        },
        Some("max") => match encoding {
            Some(Dsv4Encoding::Preview) => Ok(DS_EFFORT_ABSOLUTE_MAX),
            Some(Dsv4Encoding::V0731) => Ok(DS_EFFORT_BEYOND_MAX),
            None => Err(
                "dsv4 reasoning_effort \"max\" renders differently on the preview vs 0731 \
                 encoding and this artifact's encoding revision is unknown (config.json \
                 dspark_* census unavailable) — refusing rather than guessing"
                    .into(),
            ),
        },
        Some(_) => Ok(""),
    }
}

/// encoding_dsv4 DS_TASK_SP_TOKENS (E:28-35). The task token for a quick-instruction head.
fn ds_task_token(task: &str) -> Option<&'static str> {
    match task {
        "action" => Some("<\u{ff5c}action\u{ff5c}>"),
        "query" => Some("<\u{ff5c}query\u{ff5c}>"),
        "authority" => Some("<\u{ff5c}authority\u{ff5c}>"),
        "domain" => Some("<\u{ff5c}domain\u{ff5c}>"),
        "title" => Some("<\u{ff5c}title\u{ff5c}>"),
        "read_url" => Some("<\u{ff5c}read_url\u{ff5c}>"),
        _ => None,
    }
}

/// python `json.dumps(v, ensure_ascii=False)` over a `Val` (encoding_dsv4 `to_json`, E:101-106):
/// default separators `", "` / `": "`, insertion key order, non-ASCII raw, `Num` exact text.
/// serde-free (this crate ships no serde) — the escaper below matches json.dumps exactly.
///
/// SHARED by the dsv4 arm (which named it) and the GLM-5.3-Flash arm: both templates render
/// their tool JSON through jinja's `tojson`, which `transformers` binds to exactly this call.
fn py_json(v: &Val, out: &mut String) {
    match v {
        Val::Null => out.push_str("null"),
        Val::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Val::Num(s) => out.push_str(s),
        Val::Str(s) => {
            out.push('"');
            py_json_escape(s, out);
            out.push('"');
        }
        Val::Arr(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                py_json(x, out);
            }
            out.push(']');
        }
        Val::Obj(o) => {
            out.push('{');
            for (i, (k, val)) in o.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                out.push('"');
                py_json_escape(k, out);
                out.push_str("\": ");
                py_json(val, out);
            }
            out.push('}');
        }
    }
}

/// JSON string escaping matching python `json.dumps(ensure_ascii=False)`: `"` `\` and the
/// C0 escapes; other control chars < 0x20 become `\u00xx`; everything else (incl. non-ASCII)
/// passes through raw. json.dumps does NOT escape `/` or DEL.
fn py_json_escape(s: &str, out: &mut String) {
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
}

/// encoding_dsv4 `render_tools` (E:189-206) + TOOLS_TEMPLATE (E:70-95): the tool-declaration
/// block appended to a system/developer turn. `funcs` are the tool `function` objects
/// (encoding_dsv4 `tools_from_openai_format`). Ends with a trailing `\n`.
fn dsv4_render_tools(funcs: &[Val]) -> String {
    let mut schemas = String::new();
    for (i, f) in funcs.iter().enumerate() {
        if i > 0 {
            schemas.push('\n');
        }
        py_json(f, &mut schemas);
    }
    format!(
        "## Tools\n\nYou have access to a set of tools to help answer the user's question. \
You can invoke tools by writing a \"<{d}tool_calls>\" block like the following:\n\n\
<{d}tool_calls>\n<{d}invoke name=\"$TOOL_NAME\">\n\
<{d}parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</{d}parameter>\n\
...\n</{d}invoke>\n<{d}invoke name=\"$TOOL_NAME2\">\n...\n</{d}invoke>\n</{d}tool_calls>\n\n\
String parameters should be specified as is and set `string=\"true\"`. For all other types \
(numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.\
\n\nIf thinking_mode is enabled (triggered by {ts}), you MUST output your complete reasoning \
inside {ts}...{te} BEFORE any tool calls or final response.\n\nOtherwise, output directly \
after {te} with tool calls or final response.\n\n### Available Tool Schemas\n\n{schemas}\n\n\
You MUST strictly follow the above defined tool name and parameter schemas to invoke tool \
calls.\n",
        d = DS_DSML,
        ts = DS_THINK_START,
        te = DS_THINK_END,
        schemas = schemas,
    )
}

/// One assistant tool_calls block (encoding_dsv4 E:52-58, E:139-166, E:323-336): the `\n\n`
/// prefix + `<｜DSML｜tool_calls>` wrapper + one `<｜DSML｜invoke>` per call, each argument a
/// `<｜DSML｜parameter>` line (string values raw with `string="true"`, everything else
/// json.dumps'd with `string="false"`). Argument order = insertion order (NO dictsort).
fn dsv4_render_tool_calls(calls: &[ToolCall]) -> String {
    let mut invokes = String::new();
    for (i, call) in calls.iter().enumerate() {
        if i > 0 {
            invokes.push('\n');
        }
        invokes.push_str(&format!(
            "<{d}invoke name=\"{n}\">\n",
            d = DS_DSML,
            n = call.name
        ));
        for (j, (k, v)) in call.args.iter().enumerate() {
            if j > 0 {
                invokes.push('\n');
            }
            let is_str = matches!(v, Val::Str(_));
            invokes.push_str(&format!(
                "<{d}parameter name=\"{k}\" string=\"{b}\">",
                d = DS_DSML,
                k = k,
                b = if is_str { "true" } else { "false" },
            ));
            match v {
                Val::Str(s) => invokes.push_str(s),
                other => py_json(other, &mut invokes),
            }
            invokes.push_str(&format!("</{d}parameter>", d = DS_DSML));
        }
        invokes.push_str(&format!("\n</{d}invoke>", d = DS_DSML));
    }
    format!(
        "\n\n<{d}tool_calls>\n{invokes}\n</{d}tool_calls>",
        d = DS_DSML,
        invokes = invokes
    )
}

/// One merged content block on a user turn (encoding_dsv4 content_blocks, E:289-309).
enum DsBlock {
    Text(String),
    ToolResult {
        content: String,
        tool_use_id: String,
    },
}

/// One preprocessed message (post merge_tool_messages / sort). `blocks` is Some for user
/// turns (a merged run of user text + tool results); other roles carry `content`.
struct DsMsg {
    role: String,
    content: String,
    blocks: Option<Vec<DsBlock>>,
    reasoning: String,
    tool_calls: Vec<ToolCall>,
    tools: Vec<Val>,
    task: Option<String>,
}

/// encoding_dsv4 `merge_tool_messages` (E:401-457): fold role:"tool" turns and consecutive
/// user turns into single `<｜User｜>` turns carrying `content_blocks`. `req_tools` are the
/// request-level tool `function` objects attached to the LEADING system turn (matching the
/// serve surface; a synthetic empty system turn is created when tools exist with no system
/// turn — the oracle's render of {"role":"system","content":"","tools":[...]}). A turn's own
/// `tools` (fixture harness, e.g. tools on a developer message) take precedence.
fn dsv4_merge(turns: &[Turn], req_tools: &[Val]) -> Vec<DsMsg> {
    let mut merged: Vec<DsMsg> = Vec::new();
    let any_turn_tools = turns.iter().any(|t| !t.tools.is_empty());
    // Serve surface: request-level tools ride the leading system turn (or a synthetic one).
    let mut leading_tools_pending = !req_tools.is_empty() && !any_turn_tools;
    if leading_tools_pending && !turns.first().map(|t| t.role == "system").unwrap_or(false) {
        merged.push(DsMsg {
            role: "system".into(),
            content: String::new(),
            blocks: None,
            reasoning: String::new(),
            tool_calls: Vec::new(),
            tools: req_tools.to_vec(),
            task: None,
        });
        leading_tools_pending = false;
    }
    for turn in turns {
        match turn.role.as_str() {
            "tool" => {
                let block = DsBlock::ToolResult {
                    content: turn.content.clone(),
                    tool_use_id: turn.tool_call_id.clone().unwrap_or_default(),
                };
                match merged.last_mut() {
                    Some(m) if m.role == "user" && m.blocks.is_some() => {
                        m.blocks.as_mut().unwrap().push(block);
                    }
                    _ => merged.push(DsMsg {
                        role: "user".into(),
                        content: String::new(),
                        blocks: Some(vec![block]),
                        reasoning: String::new(),
                        tool_calls: Vec::new(),
                        tools: Vec::new(),
                        task: None,
                    }),
                }
            }
            "user" => {
                let text = DsBlock::Text(turn.content.clone());
                match merged.last_mut() {
                    Some(m) if m.role == "user" && m.blocks.is_some() && m.task.is_none() => {
                        m.blocks.as_mut().unwrap().push(text);
                    }
                    _ => merged.push(DsMsg {
                        role: "user".into(),
                        content: turn.content.clone(),
                        blocks: Some(vec![text]),
                        reasoning: String::new(),
                        tool_calls: Vec::new(),
                        tools: turn.tools.clone(),
                        task: turn.task.clone(),
                    }),
                }
            }
            role => {
                let mut tools = turn.tools.clone();
                if role == "system" && leading_tools_pending && merged.is_empty() {
                    tools = req_tools.to_vec();
                    leading_tools_pending = false;
                }
                merged.push(DsMsg {
                    role: role.to_string(),
                    content: turn.content.clone(),
                    blocks: None,
                    reasoning: turn.reasoning.clone().unwrap_or_default(),
                    tool_calls: turn.tool_calls.clone(),
                    tools,
                    task: turn.task.clone(),
                });
            }
        }
    }
    merged
}

/// encoding_dsv4 `sort_tool_results_by_call_order` (E:460-499): within a user turn holding
/// more than one tool_result block, order those blocks by the preceding assistant's
/// tool_calls id order (stable; an unknown id sorts as 0). Non-tool block positions are kept.
#[allow(clippy::needless_range_loop)] // indexed: reads earlier turns' order, mutates msgs[i]
fn dsv4_sort_tool_results(msgs: &mut [DsMsg]) {
    let mut order: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    // walk without holding an immutable borrow across the mutable block edit.
    for i in 0..msgs.len() {
        if msgs[i].role == "assistant" && !msgs[i].tool_calls.is_empty() {
            order.clear();
            for (idx, tc) in msgs[i].tool_calls.iter().enumerate() {
                if let Some(id) = tc.id.as_deref()
                    && !id.is_empty()
                {
                    order.insert(id.to_string(), idx);
                }
            }
        } else if msgs[i].role == "user" {
            let n_tool = msgs[i]
                .blocks
                .as_ref()
                .map(|b| {
                    b.iter()
                        .filter(|x| matches!(x, DsBlock::ToolResult { .. }))
                        .count()
                })
                .unwrap_or(0);
            if n_tool > 1 && !order.is_empty() {
                let blocks = msgs[i].blocks.take().unwrap();
                // stable sort the tool_result blocks by call order; keep others in place.
                let mut tool_blocks: Vec<DsBlock> = Vec::new();
                let mut positions: Vec<bool> = Vec::new(); // true = tool_result slot
                let mut others: Vec<DsBlock> = Vec::new();
                for b in blocks {
                    match b {
                        DsBlock::ToolResult { .. } => {
                            positions.push(true);
                            tool_blocks.push(b);
                        }
                        other => {
                            positions.push(false);
                            others.push(other);
                        }
                    }
                }
                tool_blocks.sort_by_key(|b| match b {
                    DsBlock::ToolResult { tool_use_id, .. } => {
                        *order.get(tool_use_id).unwrap_or(&0)
                    }
                    _ => 0,
                });
                let mut ti = tool_blocks.into_iter();
                let mut oi = others.into_iter();
                let rebuilt: Vec<DsBlock> = positions
                    .into_iter()
                    .map(|is_tool| {
                        if is_tool {
                            ti.next().unwrap()
                        } else {
                            oi.next().unwrap()
                        }
                    })
                    .collect();
                msgs[i].blocks = Some(rebuilt);
            }
        }
    }
}

/// index of the last user/developer message (encoding_dsv4 `find_last_user_index`, E:209-216).
fn dsv4_last_user_idx(msgs: &[DsMsg]) -> isize {
    for i in (0..msgs.len()).rev() {
        if msgs[i].role == "user" || msgs[i].role == "developer" {
            return i as isize;
        }
    }
    -1
}

/// encoding_dsv4 `_drop_thinking_messages` (E:575-599): keep user/system/latest_reminder and
/// everything at/after the last user; strip reasoning from earlier assistants; drop earlier
/// developer (and other) turns entirely. Runs only in thinking mode with no tools declared.
fn dsv4_drop_thinking(msgs: Vec<DsMsg>) -> Vec<DsMsg> {
    let last = dsv4_last_user_idx(&msgs);
    let mut out = Vec::with_capacity(msgs.len());
    for (i, mut m) in msgs.into_iter().enumerate() {
        let keep_role = matches!(
            m.role.as_str(),
            "user" | "system" | "latest_reminder" | "direct_search_results"
        );
        if keep_role || (i as isize) >= last {
            out.push(m);
        } else if m.role == "assistant" {
            m.reasoning.clear();
            out.push(m);
        }
        // developer + others before the last user are dropped.
    }
    out
}

/// Full port of encoding_dsv4 `encode_messages` (E:506-572) + `render_message` (E:223-394),
/// covering BOTH shipped encoding revisions (they differ only in the effort ladder — see
/// `Dsv4Encoding`).
///
/// ThinkMode maps onto encoding_dsv4's (thinking_mode, reasoning_effort):
///
///   - `Default` → thinking (the model has no template-own default; thinking_mode is a
///     REQUIRED arg and the README example + the model's agentic positioning make thinking
///     the honest default — see TEMPLATE-SEMANTICS.md finding #1);
///   - `Think`   → thinking;
///   - `NoThink` → chat (the DeepSeek "Non-think" mode: `<｜Assistant｜></think>`).
///
/// The `reasoning_effort` string resolves through `dsv4_effort_prefix` per the artifact's
/// `encoding` revision (preview: "max" prefix only, "high" a documented no-op; 0731:
/// low/high/max ladder). `Err` ONLY when the requested (thinking, effort) cell renders
/// differently across revisions and `encoding` is `None` — the refuse-on-ambiguity law.
/// On the serve path the encoding rides the `Tokenizer` (config.json dspark_* census at
/// `from_hf_dir`); the HTTP layer forwards the OpenAI level for dsv4 models
/// (`ModelCaps::dsv4`), so "high" now reaches the 0731 ladder for real.
///
/// `req_tools` are the request-level tool `function` objects (attached to the leading system
/// turn); `add_generation_prompt` gates ONLY the final-message generation-prompt transition
/// (mid-conversation continuation transitions are always emitted, matching the python's
/// unconditional transition law).
fn apply_dsv4_template(
    turns: &[Turn],
    add_generation_prompt: bool,
    req_tools: &[Val],
    think: ThinkMode,
    reasoning_effort: Option<&str>,
    encoding: Option<Dsv4Encoding>,
) -> Result<String, String> {
    let thinking = think != ThinkMode::NoThink; // Default + Think -> thinking; NoThink -> chat
    let effort_prefix = dsv4_effort_prefix(thinking, reasoning_effort, encoding)?;

    let mut msgs = dsv4_merge(turns, req_tools);
    dsv4_sort_tool_results(&mut msgs);
    // effective drop_thinking: default True, auto-disabled when any message declares tools.
    let any_tools = msgs.iter().any(|m| !m.tools.is_empty());
    let effective_drop = !any_tools;
    if thinking && effective_drop {
        msgs = dsv4_drop_thinking(msgs);
    }
    let last_user = dsv4_last_user_idx(&msgs);
    let n = msgs.len();

    let mut out = String::from(DS_BOS);
    for idx in 0..n {
        let m = &msgs[idx];
        if idx == 0 {
            // effort prefix before the first rendered message (preview E:262-263 / 0731
            // E:275-277); "" when no prefix applies, so this is a no-op push then.
            out.push_str(effort_prefix);
        }
        match m.role.as_str() {
            "system" => {
                out.push_str(&m.content);
                if !m.tools.is_empty() {
                    out.push_str("\n\n");
                    out.push_str(&dsv4_render_tools(&m.tools));
                }
            }
            "developer" => {
                out.push_str(DS_USER);
                out.push_str(&m.content);
                if !m.tools.is_empty() {
                    out.push_str("\n\n");
                    out.push_str(&dsv4_render_tools(&m.tools));
                }
            }
            "user" => {
                out.push_str(DS_USER);
                if let Some(blocks) = &m.blocks {
                    for (i, b) in blocks.iter().enumerate() {
                        if i > 0 {
                            out.push_str("\n\n");
                        }
                        match b {
                            DsBlock::Text(t) => out.push_str(t),
                            DsBlock::ToolResult { content, .. } => {
                                out.push_str("<tool_result>");
                                out.push_str(content);
                                out.push_str("</tool_result>");
                            }
                        }
                    }
                } else {
                    out.push_str(&m.content);
                }
            }
            "latest_reminder" => {
                out.push_str(DS_REMINDER);
                out.push_str(&m.content);
            }
            "assistant" => {
                let prev_has_task = idx > 0 && msgs[idx - 1].task.is_some();
                let mut thinking_part = String::new();
                if thinking && !prev_has_task && (!effective_drop || (idx as isize) > last_user) {
                    thinking_part.push_str(&m.reasoning);
                    thinking_part.push_str(DS_THINK_END);
                }
                out.push_str(&thinking_part);
                out.push_str(&m.content);
                if !m.tool_calls.is_empty() {
                    out.push_str(&dsv4_render_tool_calls(&m.tool_calls));
                }
                out.push_str(DS_EOS);
            }
            _ => {} // direct_search_results and unknown roles never render (E:362-363).
        }

        // --- transition tokens (E:365-394) ---
        // Early-out: a non-final message whose next turn is NOT assistant/latest_reminder gets
        // no transition (the python's E:366 guard).
        if idx + 1 < n {
            let next = msgs[idx + 1].role.as_str();
            if next != "assistant" && next != "latest_reminder" {
                continue;
            }
        }
        let is_last = idx + 1 >= n;
        if let Some(task) = m.task.as_deref() {
            // generation-prompt-shaped: a task on the final message is gated on the gen prompt.
            if is_last && !add_generation_prompt {
                continue;
            }
            if let Some(tok) = ds_task_token(task) {
                if task != "action" {
                    out.push_str(tok);
                } else {
                    out.push_str(DS_ASSISTANT);
                    out.push_str(if thinking {
                        DS_THINK_START
                    } else {
                        DS_THINK_END
                    });
                    out.push_str(tok);
                }
            }
        } else if m.role == "user" || m.role == "developer" {
            if is_last && !add_generation_prompt {
                continue;
            }
            out.push_str(DS_ASSISTANT);
            // E:387-392: thinking opens `<think>` when drop_thinking is OFF (tools present)
            // OR (drop on) at/after the last user turn; else it closes `</think>`. chat mode
            // (thinking=false) always closes.
            if thinking && (!effective_drop || (idx as isize) >= last_user) {
                out.push_str(DS_THINK_START);
            } else {
                out.push_str(DS_THINK_END);
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ds4f rung-3 regression (the first real serve 400): the REAL dsv4 artifacts
    /// ship NO chat_template string — dispatch and the tools branch must key on the
    /// detected encoding revision, or a fully-defined dialect 400s at the door.
    #[test]
    fn templateless_dsv4_artifact_dispatches_on_encoding() {
        let s =
            apply_chat_template_enc(None, &[("user", "Hello")], true, Some(Dsv4Encoding::V0731))
                .unwrap();
        assert!(
            s.contains("<\u{ff5c}User\u{ff5c}>") && s.contains("<\u{ff5c}Assistant\u{ff5c}>"),
            "encoding dispatch did not reach the dsv4 renderer: {s:?}"
        );
        assert!(!s.contains("<|im_start|>"), "fell back to ChatML: {s:?}");
        let legacy = apply_chat_template_enc(None, &[("user", "Hello")], true, None).unwrap();
        assert_eq!(
            legacy,
            apply_chat_template_str(None, &[("user", "Hello")], true)
        );

        let turns = vec![Turn {
            role: "user".into(),
            content: "What is the weather in Paris? Use the tool.".into(),
            ..Default::default()
        }];
        let tj = vec![
            r#"{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}"#.to_string(),
        ];
        // the dsv4 renderer consumes the typed tree (tools_struct), like the gemma dialect
        let tv = vec![Val::Obj(vec![
            ("name".into(), Val::Str("get_weather".into())),
            (
                "description".into(),
                Val::Str("Get weather for a city".into()),
            ),
            (
                "parameters".into(),
                Val::Obj(vec![
                    ("type".into(), Val::Str("object".into())),
                    (
                        "properties".into(),
                        Val::Obj(vec![(
                            "city".into(),
                            Val::Obj(vec![("type".into(), Val::Str("string".into()))]),
                        )]),
                    ),
                ]),
            ),
        ])];
        let out = apply_chat_template_tools_ex(
            None,
            &turns,
            true,
            &tj,
            &tv,
            ThinkMode::Default,
            None,
            Some(Dsv4Encoding::V0731),
        )
        .expect("templateless dsv4 artifact must render tools (DSML is its protocol)");
        assert!(
            out.contains("\u{ff5c}DSML\u{ff5c}") || out.contains("get_weather"),
            "tools block missing from the DSML render: {out:?}"
        );
    }

    #[test]
    fn plain_chatml() {
        let s = apply_chat_template_str(None, &[("user", "Hello")], true);
        assert_eq!(
            s,
            "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    /// A template stand-in carrying every marker the real qwen3.5/3.6 dumps carry
    /// (tools branch + think tail + enable_thinking switch).
    const QWEN_TOOLS_TMPL: &str =
        "... <tools> ... add_generation_prompt ... enable_thinking ... '<think>\\n' ...";

    /// Isolation contract: the tools renderer on a PLAIN request (no tools, no tool turns,
    /// Default think) is byte-identical to the legacy renderer, across the message shapes
    /// the serve path sees.
    #[test]
    fn tools_renderer_matches_legacy_when_plain() {
        let batteries: &[&[(&str, &str)]] = &[
            &[("user", "Hello")],
            &[("system", "You are helpful."), ("user", "Hi")],
            &[
                ("system", "rules"),
                ("user", "task"),
                ("assistant", "work"),
                ("user", "more"),
            ],
            &[("user", "  padded  "), ("assistant", "reply\nwith lines")],
        ];
        for tmpl in [None, Some(QWEN_TOOLS_TMPL)] {
            for msgs in batteries {
                let legacy = apply_chat_template_str(tmpl, msgs, true);
                let turns: Vec<Turn> = msgs
                    .iter()
                    .map(|(r, c)| Turn {
                        role: r.to_string(),
                        content: c.to_string(),
                        tool_calls: Vec::new(),
                        ..Default::default()
                    })
                    .collect();
                let ext =
                    apply_chat_template_tools(tmpl, &turns, true, &[], ThinkMode::Default, None)
                        .unwrap();
                assert_eq!(legacy, ext, "template={tmpl:?} msgs={msgs:?}");
            }
        }
    }

    #[test]
    fn tools_header_and_tool_response_render_per_template_law() {
        let tools =
            vec![r#"{"type": "function", "function": {"name": "get_weather"}}"#.to_string()];
        let turns = vec![
            Turn {
                role: "system".into(),
                content: "Be terse.".into(),
                tool_calls: Vec::new(),
                ..Default::default()
            },
            Turn {
                role: "user".into(),
                content: "Weather in Paris?".into(),
                tool_calls: Vec::new(),
                ..Default::default()
            },
            Turn {
                role: "assistant".into(),
                content: "".into(),
                tool_calls: vec![ToolCall {
                    name: "get_weather".into(),
                    params: vec![("city".into(), "Paris".into())],
                    ..Default::default()
                }],
                ..Default::default()
            },
            Turn {
                role: "tool".into(),
                content: "{\"temp_c\": 21}".into(),
                tool_calls: Vec::new(),
                ..Default::default()
            },
        ];
        let s = apply_chat_template_tools(
            Some(QWEN_TOOLS_TMPL),
            &turns,
            true,
            &tools,
            ThinkMode::Default,
            None,
        )
        .unwrap();
        let expected = concat!(
            "<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n",
            "<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"get_weather\"}}\n</tools>",
            "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:",
            "\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\n",
            "value_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the ",
            "second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>",
            "\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner ",
            "<function=...></function> block must be nested within <tool_call></tool_call> XML tags\n",
            "- Required parameters MUST be specified\n- You may provide optional reasoning for your ",
            "function call in natural language BEFORE the function call, but NOT after\n- If there is ",
            "no function call available, answer the question like normal with your current knowledge ",
            "and do not tell the user about function calls\n</IMPORTANT>",
            "\n\nBe terse.<|im_end|>\n",
            "<|im_start|>user\nWeather in Paris?<|im_end|>\n",
            "<|im_start|>assistant\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n",
            "</parameter>\n</function>\n</tool_call><|im_end|>\n",
            "<|im_start|>user\n<tool_response>\n{\"temp_c\": 21}\n</tool_response><|im_end|>\n",
            "<|im_start|>assistant\n<think>\n",
        );
        assert_eq!(s, expected);
    }

    #[test]
    fn assistant_content_plus_calls_and_consecutive_tool_turns_group() {
        let turns = vec![
            Turn {
                role: "user".into(),
                content: "both".into(),
                tool_calls: Vec::new(),
                ..Default::default()
            },
            Turn {
                role: "assistant".into(),
                content: "checking".into(),
                tool_calls: vec![
                    ToolCall {
                        name: "a".into(),
                        params: vec![("x".into(), "1".into())],
                        ..Default::default()
                    },
                    ToolCall {
                        name: "b".into(),
                        params: Vec::new(),
                        ..Default::default()
                    },
                ],
                ..Default::default()
            },
            Turn {
                role: "tool".into(),
                content: "r1".into(),
                tool_calls: Vec::new(),
                ..Default::default()
            },
            Turn {
                role: "tool".into(),
                content: "r2".into(),
                tool_calls: Vec::new(),
                ..Default::default()
            },
        ];
        let s = apply_chat_template_tools(
            Some(QWEN_TOOLS_TMPL),
            &turns,
            false,
            &[],
            ThinkMode::Default,
            None,
        )
        .unwrap();
        assert_eq!(
            s,
            concat!(
                "<|im_start|>user\nboth<|im_end|>\n",
                "<|im_start|>assistant\nchecking\n\n",
                "<tool_call>\n<function=a>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>\n",
                "<tool_call>\n<function=b>\n</function>\n</tool_call><|im_end|>\n",
                "<|im_start|>user\n<tool_response>\nr1\n</tool_response>",
                "\n<tool_response>\nr2\n</tool_response><|im_end|>\n",
            )
        );
    }

    #[test]
    fn nothink_maps_to_enable_thinking_false_tail_and_degrades_gracefully() {
        let turns = vec![Turn {
            role: "user".into(),
            content: "hi".into(),
            tool_calls: Vec::new(),
            ..Default::default()
        }];
        // switch present: NoThink renders the closed think block.
        let s = apply_chat_template_tools(
            Some(QWEN_TOOLS_TMPL),
            &turns,
            true,
            &[],
            ThinkMode::NoThink,
            None,
        )
        .unwrap();
        assert!(
            s.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
            "{s:?}"
        );
        // no enable_thinking switch: NoThink is ignored (template default stands).
        let tmpl_no_switch = "... add_generation_prompt ... '<think>\\n' ...";
        let s = apply_chat_template_tools(
            Some(tmpl_no_switch),
            &turns,
            true,
            &[],
            ThinkMode::NoThink,
            None,
        )
        .unwrap();
        assert!(s.ends_with("<|im_start|>assistant\n<think>\n"), "{s:?}");
        // no template at all: plain ChatML, no tail either way.
        let s =
            apply_chat_template_tools(None, &turns, true, &[], ThinkMode::NoThink, None).unwrap();
        assert!(s.ends_with("<|im_start|>assistant\n"), "{s:?}");
    }

    #[test]
    fn tools_on_templates_without_tools_branch_error() {
        let turns = vec![Turn {
            role: "user".into(),
            content: "hi".into(),
            tool_calls: Vec::new(),
            ..Default::default()
        }];
        let tools = vec!["{}".to_string()];
        for tmpl in [None, Some("... <|turn> ...")] {
            let err =
                apply_chat_template_tools(tmpl, &turns, true, &tools, ThinkMode::Default, None);
            assert!(err.is_err(), "template={tmpl:?}");
        }
        // tool-role turns need the branch too.
        let tool_turns = vec![Turn {
            role: "tool".into(),
            content: "r".into(),
            tool_calls: Vec::new(),
            ..Default::default()
        }];
        assert!(
            apply_chat_template_tools(None, &tool_turns, true, &[], ThinkMode::Default, None)
                .is_err()
        );
    }

    // ---- per-arch thinking control (owner directive 2026-08-07) -------------------------
    // Every `expected` below is the EXACT string the arch's REAL shipped template renders,
    // from research/step-sku-20260807/raw/thinking-goldens.txt (render-thinking-goldens.py:
    // jinja2 trim_blocks/lstrip_blocks over the pinned template dumps — gemma4 sha 36e3a42e
    // from the local QAT GGUF header, hy3 sha 7fc351fe from the pinned tencent/Hy3 snapshot).

    fn one_user() -> Vec<Turn> {
        vec![turn("user", "Hi")]
    }

    #[test]
    fn gemma4_thinking_maps_to_the_think_token_and_open_turn() {
        let g = |think: ThinkMode| {
            apply_chat_template_tools(Some("... <|turn> ..."), &one_user(), true, &[], think, None)
                .unwrap()
        };
        // Default AND NoThink = the template's own default(false): closed thought channel.
        // Byte-identical to the legacy renderer (no silent behavior change).
        let closed = "<|turn>user\nHi<turn|>\n<|turn>model\n<|channel>thought\n<channel|>";
        assert_eq!(g(ThinkMode::Default), closed);
        assert_eq!(g(ThinkMode::NoThink), closed);
        assert_eq!(
            apply_chat_template_str(Some("... <|turn> ..."), &[("user", "Hi")], true),
            closed,
            "legacy renderer = the default arm"
        );
        // Think = enable_thinking=true: <|think|> injected into a CREATED system turn and
        // the generation turn left open (golden: gemma4 enable_thinking=true, no system).
        assert_eq!(
            g(ThinkMode::Think),
            "<|turn>system\n<|think|>\n<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\n"
        );
        // with a client system turn the token lands at the very top of it (golden).
        let turns = vec![turn("system", "Be terse."), turn("user", "Hi")];
        let s = apply_chat_template_tools(
            Some("... <|turn> ..."),
            &turns,
            true,
            &[],
            ThinkMode::Think,
            None,
        )
        .unwrap();
        assert_eq!(
            s,
            "<|turn>system\n<|think|>\nBe terse.<turn|>\n\
                       <|turn>user\nHi<turn|>\n<|turn>model\n"
        );
    }

    /// A QAT-tooluse stand-in: carries `<|turn>` + `<|tool>` (engages the gemma4 tools arm)
    /// AND the closed-tail literal (the QAT trunk's thinking-off generation tail). The
    /// official served trunk omits that literal, so its tools arm emits the bare `<|turn>model`
    /// on thinking-off — the fixtures cover that side.
    const GEMMA_TOOLUSE_QAT_TMPL: &str =
        "... <|turn> ... <|tool> ... <|channel>thought\\n<channel|> ...";

    #[test]
    fn gemma4_tools_arm_is_byte_identical_to_legacy_on_toolless_requests() {
        // REGRESSION (deliverable 6): a NO-tools request through the gemma4 tools arm renders
        // byte-identically to the standalone gemma4 renderer, across think modes and message
        // shapes — the tool path never perturbs plain gemma traffic on the tooluse trunk.
        let batteries: &[&[(&str, &str)]] = &[
            &[("user", "Hi")],
            &[("system", "Be terse."), ("user", "Weather?")],
            &[
                ("system", "rules"),
                ("user", "task"),
                ("assistant", "work"),
                ("user", "more"),
            ],
            &[("user", "  padded  "), ("assistant", "reply\nwith lines")],
        ];
        for msgs in batteries {
            let turns: Vec<Turn> = msgs
                .iter()
                .map(|(r, c)| Turn {
                    role: r.to_string(),
                    content: c.to_string(),
                    ..Default::default()
                })
                .collect();
            for (mode, thinking) in [
                (ThinkMode::Default, false),
                (ThinkMode::NoThink, false),
                (ThinkMode::Think, true),
            ] {
                let legacy = apply_gemma4_template(msgs, true, thinking);
                let arm = apply_chat_template_tools(
                    Some(GEMMA_TOOLUSE_QAT_TMPL),
                    &turns,
                    true,
                    &[],
                    mode,
                    None,
                )
                .unwrap();
                assert_eq!(legacy, arm, "mode={mode:?} msgs={msgs:?}");
            }
        }
    }

    #[test]
    fn gemma4_tools_arm_still_rejects_tools_without_the_tool_marker() {
        // a `<|turn>` template WITHOUT `<|tool>` keeps rejecting tool features with the clear
        // error (no committed tools reference for that trunk).
        let turns = vec![turn("user", "Weather?")];
        let tools = vec![r#"{"function":{"name":"f"}}"#.to_string()];
        let err = apply_chat_template_tools(
            Some("... <|turn> ..."),
            &turns,
            true,
            &tools,
            ThinkMode::Default,
            None,
        );
        assert!(err.is_err());
    }

    #[test]
    fn hy3_thinking_maps_to_its_reasoning_effort_levels() {
        const HY_TMPL: Option<&str> = Some("... hy_User ...");
        let h = |think: ThinkMode, effort: Option<&str>| {
            apply_chat_template_tools(HY_TMPL, &one_user(), true, &[], think, effort).unwrap()
        };
        // Default AND NoThink = the template's own default: no_think header + CLOSED think.
        // Byte-identical to the legacy renderer.
        let closed = "<\u{ff5c}hy_begin_of_sentence:opensource\u{ff5c}>\
                      <\u{ff5c}reasoning_mode:opensource\u{ff5c}>reasoning_effort:no_think\
                      <\u{ff5c}hy_User:opensource\u{ff5c}>Hi\
                      <\u{ff5c}hy_Assistant:opensource\u{ff5c}>\
                      <think:opensource></think:opensource>";
        assert_eq!(h(ThinkMode::Default, None), closed);
        assert_eq!(
            h(ThinkMode::NoThink, Some("low")),
            closed,
            "NoThink wins over a level: thinking off IS no_think"
        );
        assert_eq!(
            apply_chat_template_str(HY_TMPL, &[("user", "Hi")], true),
            closed,
            "legacy renderer = the default arm"
        );
        // Think at low/high = the template's own open-think levels (goldens: header carries
        // the level, generation prompt ends with an OPEN <think:opensource>).
        let low = h(ThinkMode::Think, Some("low"));
        assert!(low.contains("reasoning_effort:low"), "{low:?}");
        assert!(low.ends_with("<think:opensource>"), "{low:?}");
        let high = h(ThinkMode::Think, Some("high"));
        assert!(high.contains("reasoning_effort:high"), "{high:?}");
        assert!(high.ends_with("<think:opensource>"), "{high:?}");
        // medium clamps to low (hy3's accepted set is exactly no_think|low|high — the jinja
        // raise_exceptions on anything else); Think with no level also lands at low.
        assert_eq!(h(ThinkMode::Think, Some("medium")), low);
        assert_eq!(h(ThinkMode::Think, None), low);
        // History assistant turns stay CLOSED-think at every effort (the template opens only
        // turns past last_user_index; golden: "hy3 assistant history stays closed-think").
        let turns = vec![
            turn("user", "q"),
            turn("assistant", "a"),
            turn("user", "more"),
        ];
        let s =
            apply_chat_template_tools(HY_TMPL, &turns, true, &[], ThinkMode::Think, Some("low"))
                .unwrap();
        assert_eq!(
            s,
            "<\u{ff5c}hy_begin_of_sentence:opensource\u{ff5c}>\
                       <\u{ff5c}reasoning_mode:opensource\u{ff5c}>reasoning_effort:low\
                       <\u{ff5c}hy_User:opensource\u{ff5c}>q\
                       <\u{ff5c}hy_Assistant:opensource\u{ff5c}>\
                       <think:opensource></think:opensource>a\
                       <\u{ff5c}hy_eos:opensource\u{ff5c}>\
                       <\u{ff5c}hy_User:opensource\u{ff5c}>more\
                       <\u{ff5c}hy_Assistant:opensource\u{ff5c}><think:opensource>"
        );
    }

    fn hy3_tools_header(tool: &str, effort: &str) -> String {
        [
            "<\u{ff5c}hy_begin_of_sentence:opensource\u{ff5c}>You are concise.\n\n# Tools\n\n",
            "You may call one or more functions to assist with the user query.\n\n",
            "You are provided with function signatures within <tools></tools> XML tags:\n",
            "<tools>\n",
            tool,
            "\n</tools>\n\nFor function call returns, you should first print ",
            "<tool_calls:opensource>\nFor each function call, you should return object like:\n",
            "<tool_call:opensource>{function-name}<tool_sep:opensource>\n",
            "<arg_key:opensource>{arg-key-1}</arg_key:opensource>\n",
            "<arg_value:opensource>{arg-value-1}</arg_value:opensource>\n",
            "<arg_key:opensource>{arg-key-2}</arg_key:opensource>\n",
            "<arg_value:opensource>{arg-value-2}</arg_value:opensource>\n...\n",
            "</tool_call:opensource>\nAt the end of function call returns, you should print ",
            "</tool_calls:opensource><\u{ff5c}reasoning_mode:opensource\u{ff5c}>reasoning_effort:",
            effort,
        ]
        .concat()
    }

    #[test]
    fn hy3_tools_definitions_match_the_pinned_jinja() {
        const TEMPLATE: &str = "... hy_User ... <tools> ... <tool_calls{}> ...";
        let tool = r#"{"type": "function", "function": {"name": "get_weather", "description": "Get weather.", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}, "required": ["city"]}}}"#;
        let turns = vec![turn("system", "You are concise."), turn("user", "Weather?")];
        let got = apply_chat_template_tools(
            Some(TEMPLATE),
            &turns,
            true,
            &[tool.to_string()],
            ThinkMode::Default,
            None,
        )
        .unwrap();
        let expected = [
            &hy3_tools_header(tool, "no_think"),
            "<\u{ff5c}hy_User:opensource\u{ff5c}>Weather?",
            "<\u{ff5c}hy_Assistant:opensource\u{ff5c}><think:opensource></think:opensource>",
        ]
        .concat();
        assert_eq!(got, expected);
        assert!(template_has_tools_branch(TEMPLATE));
    }

    #[test]
    fn hy3_tool_call_and_response_history_match_the_pinned_jinja() {
        const TEMPLATE: &str = "... hy_User ... <tools> ... <tool_calls{}> ...";
        let tool = r#"{"type": "function", "function": {"name": "get_weather", "description": "Get weather.", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}, "required": ["city"]}}}"#;
        let turns = vec![
            turn("system", "You are concise."),
            turn("user", "Weather?"),
            Turn {
                role: "assistant".into(),
                reasoning: Some("Need weather.".into()),
                tool_calls: vec![ToolCall {
                    name: "get_weather".into(),
                    params: vec![("city".into(), "Paris".into()), ("days".into(), "2".into())],
                    ..Default::default()
                }],
                ..Default::default()
            },
            turn("tool", "sunny"),
            turn("user", "Summarize."),
        ];
        let got = apply_chat_template_tools(
            Some(TEMPLATE),
            &turns,
            true,
            &[tool.to_string()],
            ThinkMode::Think,
            Some("high"),
        )
        .unwrap();
        let expected = [
            &hy3_tools_header(tool, "high"),
            "<\u{ff5c}hy_User:opensource\u{ff5c}>Weather?",
            "<\u{ff5c}hy_Assistant:opensource\u{ff5c}><think:opensource>Need weather.</think:opensource>",
            "<tool_calls:opensource>\n<tool_call:opensource>get_weather<tool_sep:opensource>\n",
            "<arg_key:opensource>city</arg_key:opensource>\n<arg_value:opensource>Paris</arg_value:opensource>\n",
            "<arg_key:opensource>days</arg_key:opensource>\n<arg_value:opensource>2</arg_value:opensource>\n",
            "</tool_call:opensource>\n</tool_calls:opensource><\u{ff5c}hy_eos:opensource\u{ff5c}>",
            "<tool_responses:opensource>\n<tool_response:opensource>\nsunny\n",
            "</tool_response:opensource>\n</tool_responses:opensource>",
            "<\u{ff5c}hy_User:opensource\u{ff5c}>Summarize.",
            "<\u{ff5c}hy_Assistant:opensource\u{ff5c}><think:opensource>",
        ]
        .concat();
        assert_eq!(got, expected);
    }

    #[test]
    fn qwen_think_mode_covers_all_three_directions() {
        let q = |think: ThinkMode| {
            apply_chat_template_tools(Some(QWEN_TOOLS_TMPL), &one_user(), true, &[], think, None)
                .unwrap()
        };
        // qwen's template default IS thinking-on, so Default and Think render identically.
        assert!(q(ThinkMode::Default).ends_with("<|im_start|>assistant\n<think>\n"));
        assert_eq!(q(ThinkMode::Think), q(ThinkMode::Default));
        assert!(q(ThinkMode::NoThink).ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    }

    // ---- StepFun Step-3.7-Flash (arch step35) -------------------------------------------
    // Every `expected` below is the EXACT string the shipped jinja renders, taken from
    // research/step37-p2-20260806/raw/step35-template-goldens.txt (generated by
    // render_step35_template.py under jinja2 with trim_blocks/lstrip_blocks — the settings HF
    // transformers and llama.cpp's minja use). `{{bos_token}}` renders as "" there because
    // encode(add_special) supplies BOS.

    /// A step35 template stand-in: the real one is 5723 chars, and the detector keys on
    /// `render_message_content` (the macro no other committed template defines). The other
    /// markers are present to prove the step35 arm WINS the dispatch — a qwen-marker template
    /// carrying `<tools>`/`<think>`/`add_generation_prompt` would otherwise take the qwen arm.
    const STEP35_TMPL: &str = "{% macro render_message_content(message) %}... <tools> ... add_generation_prompt ... '<think>\\n' ...";

    fn s35(msgs: &[(&str, &str)], genp: bool) -> String {
        apply_chat_template_str(Some(STEP35_TMPL), msgs, genp)
    }

    fn s35_turns(turns: Vec<Turn>, genp: bool, tools: &[String]) -> String {
        apply_chat_template_tools(
            Some(STEP35_TMPL),
            &turns,
            genp,
            tools,
            ThinkMode::Default,
            None,
        )
        .unwrap()
    }

    fn turn(role: &str, content: &str) -> Turn {
        Turn {
            role: role.into(),
            content: content.into(),
            tool_calls: Vec::new(),
            ..Default::default()
        }
    }

    #[test]
    fn step35_plain_paths_match_the_shipped_jinja() {
        assert_eq!(
            s35(&[("user", "Hello")], true),
            "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n"
        );
        assert_eq!(
            s35(&[("user", "Hello")], false),
            "<|im_start|>user\nHello<|im_end|>\n"
        );
        assert_eq!(
            s35(&[("system", "You are helpful."), ("user", "Hi")], true),
            "<|im_start|>system\nYou are helpful.<|im_end|>\n\
                    <|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n"
        );
        // multi-turn: the prior assistant is BEFORE the last user query, so it carries NO
        // think block — the reasoning boundary the qwen arms have no concept of.
        assert_eq!(
            s35(
                &[
                    ("system", "rules"),
                    ("user", "task"),
                    ("assistant", "work"),
                    ("user", "more")
                ],
                true
            ),
            "<|im_start|>system\nrules<|im_end|>\n<|im_start|>user\ntask<|im_end|>\n\
             <|im_start|>assistant\nwork<|im_end|>\n<|im_start|>user\nmore<|im_end|>\n\
             <|im_start|>assistant\n<think>\n"
        );
        // content is NOT trimmed (this template applies no `|trim`) — the qwen arms trim.
        assert_eq!(
            s35(&[("user", "  padded  ")], true),
            "<|im_start|>user\n  padded  <|im_end|>\n<|im_start|>assistant\n<think>\n"
        );
    }

    #[test]
    fn step35_dispatch_beats_the_qwen_marker_arm() {
        // The step35 template carries every qwen marker. If the dispatch order regressed, the
        // think tail would still be right and the BODY would be wrong (trimmed content, wrong
        // tools header) — so assert a body-shaped difference, not the tail.
        let qwen = apply_chat_template_str(Some(QWEN_TOOLS_TMPL), &[("user", " pad ")], true);
        let step = s35(&[("user", " pad ")], true);
        assert_eq!(
            qwen,
            "<|im_start|>user\npad<|im_end|>\n<|im_start|>assistant\n<think>\n"
        );
        assert_eq!(
            step,
            "<|im_start|>user\n pad <|im_end|>\n<|im_start|>assistant\n<think>\n"
        );
        assert_ne!(qwen, step);
    }

    #[test]
    fn step35_reasoning_effort_renders_in_the_system_turn() {
        assert_eq!(
            apply_step35_template(&[turn("user", "Hi")], true, &[], Some("high")),
            "<|im_start|>system\nReasoning: high\n\n<|im_end|>\n\
             <|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n"
        );
        assert_eq!(
            apply_step35_template(
                &[turn("system", "Be terse."), turn("user", "Hi")],
                true,
                &[],
                Some("low")
            ),
            "<|im_start|>system\nReasoning: low\n\nBe terse.<|im_end|>\n\
             <|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n"
        );
        // with tools the order flips: Reasoning, then the system content, then `# Tools`.
        let tools = vec![r#"{"type": "function", "function": {"name": "f"}}"#.to_string()];
        let s = apply_step35_template(
            &[turn("system", "Be terse."), turn("user", "q")],
            true,
            &tools,
            Some("medium"),
        );
        assert!(
            s.starts_with("<|im_start|>system\nReasoning: medium\n\nBe terse.\n\n# Tools\n"),
            "{s:?}"
        );
    }

    #[test]
    fn reasoning_effort_reaches_step35_through_the_public_entry_and_only_step35() {
        // The serve path enters via apply_chat_template_tools: the level must land in the
        // rendered system turn on the step35 dialect...
        let turns = vec![turn("user", "Hi")];
        let s = apply_chat_template_tools(
            Some(STEP35_TMPL),
            &turns,
            true,
            &[],
            ThinkMode::Default,
            Some("high"),
        )
        .unwrap();
        assert!(
            s.starts_with("<|im_start|>system\nReasoning: high\n\n<|im_end|>\n"),
            "{s:?}"
        );
        // ...None keeps the template's own default (no Reasoning: line at all)...
        let s = apply_chat_template_tools(
            Some(STEP35_TMPL),
            &turns,
            true,
            &[],
            ThinkMode::Default,
            None,
        )
        .unwrap();
        assert!(!s.contains("Reasoning:"), "{s:?}");
        // ...and every non-step35 dialect ignores the parameter (their templates have no
        // reasoning_effort input) — byte-identical with and without it.
        for tmpl in [
            None,
            Some(QWEN_TOOLS_TMPL),
            Some("... hy_User ..."),
            Some("... <|turn> ..."),
        ] {
            let with = apply_chat_template_tools(
                tmpl,
                &turns,
                true,
                &[],
                ThinkMode::Default,
                Some("high"),
            )
            .unwrap();
            let without =
                apply_chat_template_tools(tmpl, &turns, true, &[], ThinkMode::Default, None)
                    .unwrap();
            assert_eq!(with, without, "template={tmpl:?}");
        }
    }

    #[test]
    fn step35_tools_header_is_not_the_qwen_header() {
        let tools = vec![
            r#"{"type": "function", "function": {"name": "get_weather"}}"#.to_string(),
            r#"{"type": "function", "function": {"name": "search"}}"#.to_string(),
        ];
        let s = s35_turns(
            vec![
                turn("system", "Be terse."),
                turn("user", "Weather in Paris?"),
            ],
            true,
            &tools,
        );
        assert_eq!(
            s,
            concat!(
                // leading system folds in BEFORE `# Tools` (the qwen arm appends it AFTER the
                // instruction block), and the header says "in JSONSchema format".
                "<|im_start|>system\nBe terse.\n\n# Tools\n\n",
                "You have access to the following functions in JSONSchema format:\n\n<tools>\n",
                "{\"type\": \"function\", \"function\": {\"name\": \"get_weather\"}}\n",
                "{\"type\": \"function\", \"function\": {\"name\": \"search\"}}\n</tools>",
                "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:",
                "\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\n",
                "value_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the ",
                "second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>",
                // the nesting reminder carries literal \n...\n INSIDE the example tags, and the
                // Reminder list stops after 2 bullets (the qwen block has 4).
                "\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner ",
                "<function=...>\n...\n</function> block must be nested within <tool_call>\n...\n",
                "</tool_call> XML tags\n- Required parameters MUST be specified\n</IMPORTANT>",
                "<|im_end|>\n",
                "<|im_start|>user\nWeather in Paris?<|im_end|>\n",
                "<|im_start|>assistant\n<think>\n",
            )
        );
        // and it is NOT the qwen instruction block.
        assert!(!s.contains(QWEN_TOOLS_INSTRUCTION));
    }

    #[test]
    fn step35_tool_results_take_their_own_role_and_group() {
        let tools =
            vec![r#"{"type": "function", "function": {"name": "get_weather"}}"#.to_string()];
        let turns = vec![
            turn("user", "both"),
            Turn {
                role: "assistant".into(),
                content: "checking".into(),
                tool_calls: vec![
                    ToolCall {
                        name: "a".into(),
                        params: vec![("x".into(), "1".into())],
                        ..Default::default()
                    },
                    ToolCall {
                        name: "b".into(),
                        params: Vec::new(),
                        ..Default::default()
                    },
                ],
                ..Default::default()
            },
            turn("tool", "r1"),
            turn("tool", "r2"),
        ];
        let s = s35_turns(turns, true, &tools);
        let body = s
            .split("<|im_end|>\n")
            .skip(1)
            .collect::<Vec<_>>()
            .join("<|im_end|>\n");
        assert_eq!(
            body,
            concat!(
                "<|im_start|>user\nboth<|im_end|>\n",
                // the assistant is AFTER the last user query, so it carries a think block — empty,
                // because its content has no `</think>` marker.
                "<|im_start|>assistant\n<think>\n\n</think>\nchecking",
                // NO separator before the first call and NONE between calls.
                "<tool_call>\n<function=a>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>",
                "<tool_call>\n<function=b>\n</function>\n</tool_call><|im_end|>\n",
                // own `tool_response` ROLE (not a user turn), and NO newlines inside the wrappers.
                "<|im_start|>tool_response\n<tool_response>r1</tool_response>",
                "<tool_response>r2</tool_response><|im_end|>\n",
                "<|im_start|>assistant\n<think>\n",
            )
        );
    }

    #[test]
    fn step35_assistant_think_split_and_the_reasoning_boundary() {
        // inline <think>…</think> in content splits into the reasoning block + body.
        assert_eq!(
            s35(
                &[
                    ("user", "q"),
                    ("assistant", "<think>\nreasoned\n</think>\nanswer")
                ],
                false
            ),
            "<|im_start|>user\nq<|im_end|>\n\
             <|im_start|>assistant\n<think>\nreasoned\n</think>\nanswer<|im_end|>\n"
        );
        // no markers, but still after the last query -> an EMPTY reasoning block is emitted.
        assert_eq!(
            s35(&[("user", "q"), ("assistant", "plain")], false),
            "<|im_start|>user\nq<|im_end|>\n\
             <|im_start|>assistant\n<think>\n\n</think>\nplain<|im_end|>\n"
        );
        // a user turn that IS a <tool_response> wrapper does NOT move the boundary: the
        // assistant before it still counts as after-the-last-real-query.
        assert_eq!(
            s35(
                &[
                    ("user", "real question"),
                    ("assistant", "thinking about it"),
                    ("user", "<tool_response>r</tool_response>")
                ],
                true
            ),
            "<|im_start|>user\nreal question<|im_end|>\n\
             <|im_start|>assistant\n<think>\n\n</think>\nthinking about it<|im_end|>\n\
             <|im_start|>user\n<tool_response>r</tool_response><|im_end|>\n\
             <|im_start|>assistant\n<think>\n"
        );
    }

    #[test]
    fn step35_think_tail_is_unconditional_and_nothink_is_a_noop() {
        // No `enable_thinking` in this template, so ThinkMode::NoThink cannot close the tail —
        // the same graceful-no-op contract the other switchless templates get. A NoThink that
        // silently emitted `<think>\n\n</think>\n\n` would be a prompt the model never saw.
        let turns = vec![turn("user", "hi")];
        for mode in [ThinkMode::Default, ThinkMode::NoThink] {
            let s = apply_chat_template_tools(Some(STEP35_TMPL), &turns, true, &[], mode, None)
                .unwrap();
            assert!(
                s.ends_with("<|im_start|>assistant\n<think>\n"),
                "mode={mode:?} {s:?}"
            );
        }
    }

    #[test]
    fn step35_plain_path_is_identical_through_both_renderers() {
        // same isolation contract the qwen arms hold: a plain request renders byte-identically
        // whether it enters via apply_chat_template_str or apply_chat_template_tools.
        let batteries: &[&[(&str, &str)]] = &[
            &[("user", "Hello")],
            &[("system", "You are helpful."), ("user", "Hi")],
            &[
                ("system", "rules"),
                ("user", "task"),
                ("assistant", "work"),
                ("user", "more"),
            ],
            &[("user", "  padded  "), ("assistant", "reply\nwith lines")],
        ];
        for msgs in batteries {
            let legacy = s35(msgs, true);
            let ext = s35_turns(msgs.iter().map(|(r, c)| turn(r, c)).collect(), true, &[]);
            assert_eq!(legacy, ext, "msgs={msgs:?}");
        }
    }

    #[test]
    fn qwen_think_tail() {
        // a template string containing both markers triggers the <think> tail.
        let tmpl = "... add_generation_prompt ... '<think>\\n' ...";
        let s = apply_chat_template_str(
            Some(tmpl),
            &[("system", "You are helpful."), ("user", "Hi")],
            true,
        );
        assert_eq!(
            s,
            "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n"
        );
    }

    /// The dsv4 effort-prefix law across BOTH encoding revisions (0731 re-gate,
    /// ENCODING-DIFF.md): the exact (thinking, effort, encoding) -> prefix table, including
    /// the refuse-on-ambiguity cells (unknown revision where the two encodings' bytes
    /// differ) and the never-corrupt clamps ("low"/"medium"/unknown levels -> no prefix).
    #[test]
    fn dsv4_effort_prefix_law() {
        use Dsv4Encoding::{Preview, V0731};
        let p = dsv4_effort_prefix;
        // chat mode: never a prefix, under any encoding or level (incl. unknown revision).
        for enc in [None, Some(Preview), Some(V0731)] {
            for eff in [None, Some("low"), Some("high"), Some("max")] {
                assert_eq!(p(false, eff, enc), Ok(""), "chat eff={eff:?} enc={enc:?}");
            }
        }
        // encoding-independent thinking cells: None/"low"/foreign levels -> no prefix.
        for enc in [None, Some(Preview), Some(V0731)] {
            assert_eq!(p(true, None, enc), Ok(""));
            assert_eq!(p(true, Some("low"), enc), Ok(""));
            assert_eq!(p(true, Some("medium"), enc), Ok(""));
        }
        // preview law: "high" == None (documented no-op), "max" -> the absolute text.
        assert_eq!(p(true, Some("high"), Some(Preview)), Ok(""));
        assert_eq!(
            p(true, Some("max"), Some(Preview)),
            Ok(DS_EFFORT_ABSOLUTE_MAX)
        );
        // 0731 law: "high" -> the absolute text (the OLD max), "max" -> the new beyond text.
        assert_eq!(
            p(true, Some("high"), Some(V0731)),
            Ok(DS_EFFORT_ABSOLUTE_MAX)
        );
        assert_eq!(p(true, Some("max"), Some(V0731)), Ok(DS_EFFORT_BEYOND_MAX));
        // ambiguity refusal: exactly the two cells whose bytes differ across revisions.
        assert!(p(true, Some("high"), None).is_err());
        assert!(p(true, Some("max"), None).is_err());
        // prefix text invariants pinned against the oracle constants: both end "\n\n",
        // both open with the ladder header, and they are distinct rungs.
        assert!(DS_EFFORT_ABSOLUTE_MAX.starts_with("Reasoning Effort: Absolute maximum"));
        assert!(DS_EFFORT_BEYOND_MAX.starts_with("Reasoning Effort: Beyond maximum \u{2014}"));
        assert!(DS_EFFORT_ABSOLUTE_MAX.ends_with("\n\n"));
        assert!(DS_EFFORT_BEYOND_MAX.ends_with("\n\n"));
        assert_ne!(DS_EFFORT_ABSOLUTE_MAX, DS_EFFORT_BEYOND_MAX);
    }

    /// End-to-end through the dispatch: the same request renders per-revision prefixes, and
    /// an unknown revision refuses ONLY when the requested cell is ambiguous.
    #[test]
    fn dsv4_effort_renders_per_encoding_through_dispatch() {
        const DSV4_TMPL: &str = "<\u{ff5c}Assistant\u{ff5c}> \u{ff5c}DSML\u{ff5c}";
        let turns = vec![Turn {
            role: "user".into(),
            content: "Hi".into(),
            ..Default::default()
        }];
        let render = |effort: Option<&str>, enc: Option<Dsv4Encoding>| {
            apply_chat_template_tools_ex(
                Some(DSV4_TMPL),
                &turns,
                true,
                &[],
                &[],
                ThinkMode::Think,
                effort,
                enc,
            )
        };
        let base = render(None, None).unwrap();
        // preview: high is a no-op; max prefixes the absolute text right after BOS.
        assert_eq!(
            render(Some("high"), Some(Dsv4Encoding::Preview)).unwrap(),
            base
        );
        let pv_max = render(Some("max"), Some(Dsv4Encoding::Preview)).unwrap();
        assert_eq!(
            pv_max,
            format!("{DS_BOS}{DS_EFFORT_ABSOLUTE_MAX}{}", &base[DS_BOS.len()..])
        );
        // 0731: low == default; high == the preview's max bytes; max is the new text.
        let v_low = render(Some("low"), Some(Dsv4Encoding::V0731)).unwrap();
        assert_eq!(v_low, base);
        let v_high = render(Some("high"), Some(Dsv4Encoding::V0731)).unwrap();
        assert_eq!(v_high, pv_max);
        let v_max = render(Some("max"), Some(Dsv4Encoding::V0731)).unwrap();
        assert_eq!(
            v_max,
            format!("{DS_BOS}{DS_EFFORT_BEYOND_MAX}{}", &base[DS_BOS.len()..])
        );
        // unknown revision: unambiguous cells render, ambiguous cells refuse.
        assert_eq!(render(Some("low"), None).unwrap(), base);
        assert!(render(Some("high"), None).is_err());
        assert!(render(Some("max"), None).is_err());
    }

    // ================= QWEN3.8 REASONING-EFFORT LADDER (lane/reasoning-schema-20260823) ======
    //
    // THE DEFECT: `reasoning_effort: low|medium|high` was accepted-and-ignored on every qwen3.8
    // request. The `effort_levels` cap probed for the substring `reasoning_effort is defined`,
    // and this template spells its input `reasoning_effort|default('xhigh')` — so the level was
    // parsed, validated, then dropped before the render, and the template's own `xhigh` default
    // never rendered either.
    //
    // THE GATE: memra's Rust renderer must reproduce the VENDOR's jinja byte-for-byte. The
    // template and the goldens are both committed; the goldens come from
    // `research/reasoning-schema-20260823/render_qwen38_goldens.py`, which renders the real
    // template under jinja2 with `trim_blocks`/`lstrip_blocks` — the settings HF transformers
    // and llama.cpp's minja both use, so the goldens are what the DEPLOYED template does.
    //
    // Lab authority (owner ruling 2026-08-23, "use the lab of the model, not a guess"): the
    // three rungs and both instruction sentences are Qwen's own — Qwen/Qwen3.8-27B's card
    // documents `reasoning_effort` as xhigh (default) | medium | low, and the sentences here are
    // that template's verbatim strings. `medium` injecting NOTHING is the vendor's choice, not a
    // gap. The served mint adds one thing the open-weights jinja lacks — a `high` -> `xhigh`
    // alias — which reproduces Qwen's own documented hosted-API mapping (high/max -> xhigh,
    // minimal -> low, none -> enable_thinking=False), so it is vendor semantics rather than ours.
    const Q38_TMPL: &str =
        include_str!("../../../research/reasoning-schema-20260823/qwen38-27b.chat_template.jinja");

    fn q38(turns: &[Turn], think: ThinkMode, effort: Option<&str>, tools: &[String]) -> String {
        apply_chat_template_tools(Some(Q38_TMPL), turns, true, tools, think, effort)
            .expect("q38 render")
    }

    #[test]
    fn qwen38_effort_ladder_reproduces_the_vendor_jinja_byte_for_byte() {
        let plain = [turn("user", "hi")];
        let with_system = [turn("system", "You are terse."), turn("user", "hi")];
        let empty_system = [turn("system", ""), turn("user", "hi")];
        // TWO leading system turns: the vendor MERGES the run into one turn joined by `\n`. This
        // server produces the shape itself (it normalizes `developer` to `system`), and the
        // historical per-turn emission diverged from the template here.
        let two_system = [
            turn("system", "rules"),
            turn("system", "dev rules"),
            turn("user", "hi"),
        ];
        let multiturn = [
            turn("user", "hi"),
            turn("assistant", "hello there"),
            turn("user", "again"),
        ];
        let multiturn_reasoned = [
            turn("user", "hi"),
            Turn {
                reasoning: Some("the user greets; greet back".into()),
                ..turn("assistant", "hello there")
            },
            turn("user", "again"),
        ];
        // (golden name, turns, think, effort) -> the jinja's own output.
        let cases: &[(&str, &[Turn], ThinkMode, Option<&str>)] = &[
            // THE LADDER, thinking on. `None` is the template's `default('xhigh')`.
            ("plain_default", &plain, ThinkMode::Default, None),
            ("plain_xhigh", &plain, ThinkMode::Think, Some("high")),
            ("plain_medium", &plain, ThinkMode::Think, Some("medium")),
            ("plain_low", &plain, ThinkMode::Think, Some("low")),
            // a leading system turn: the sentence PREPENDS it across a blank line.
            ("system_default", &with_system, ThinkMode::Default, None),
            ("system_xhigh", &with_system, ThinkMode::Think, Some("high")),
            (
                "system_medium",
                &with_system,
                ThinkMode::Think,
                Some("medium"),
            ),
            ("system_low", &with_system, ThinkMode::Think, Some("low")),
            // A system turn with NO content: the sentence renders ALONE. An unconditional
            // separator would leave a stray blank line before `<|im_end|>`.
            (
                "empty_system_low",
                &empty_system,
                ThinkMode::Think,
                Some("low"),
            ),
            (
                "two_system_xhigh",
                &two_system,
                ThinkMode::Think,
                Some("high"),
            ),
            ("two_system_off", &two_system, ThinkMode::NoThink, None),
            // THE BINARY AXIS: thinking off carries NO effort sentence, even with a level
            // named, because the template wraps the whole block in `enable_thinking is true`.
            ("plain_off", &plain, ThinkMode::NoThink, None),
            (
                "plain_off_with_level",
                &plain,
                ThinkMode::NoThink,
                Some("low"),
            ),
            ("system_off", &with_system, ThinkMode::NoThink, None),
            // MULTI-TURN: the template's preserve_thinking DEFAULT replays every prior
            // assistant turn's <think> block — empty when the client sent no reasoning,
            // the client's reasoning_content|trim when it did. These are the bytes the
            // reuse pools' text tier matches a parked stream against
            // (lane/dflash2-session-reuse).
            ("multiturn_off", &multiturn, ThinkMode::NoThink, None),
            ("multiturn_default", &multiturn, ThinkMode::Default, None),
            (
                "multiturn_reasoned_off",
                &multiturn_reasoned,
                ThinkMode::NoThink,
                None,
            ),
        ];
        for (name, turns, think, effort) in cases {
            let golden = golden(name);
            let got = q38(turns, *think, *effort, &[]);
            assert_eq!(
                got, golden,
                "{name}: memra's render diverges from the vendor's own jinja.\n\
                 got:    {got:?}\nwanted: {golden:?}"
            );
        }
    }

    #[test]
    fn qwen38_effort_ladder_holds_on_the_tools_branch_too() {
        // The sentence goes BEFORE the `# Tools` header inside the one system turn. A separate
        // arm because the tools branch builds that turn on a different code path, and an effort
        // control honoured only on plain requests is the same defect wearing a different hat.
        let plain = [turn("user", "hi")];
        let tools = vec![
            concat!(
                r#"{"type": "function", "function": {"name": "get_weather", "#,
                r#""description": "Get the weather", "parameters": {"type": "object", "#,
                r#""properties": {"city": {"type": "string"}}, "required": ["city"]}}}"#
            )
            .to_string(),
        ];
        let two_system = [
            turn("system", "rules"),
            turn("system", "dev rules"),
            turn("user", "hi"),
        ];
        for (name, turns, think, effort) in [
            ("tools_default", &plain[..], ThinkMode::Default, None),
            ("tools_xhigh", &plain[..], ThinkMode::Think, Some("high")),
            ("tools_medium", &plain[..], ThinkMode::Think, Some("medium")),
            ("tools_low", &plain[..], ThinkMode::Think, Some("low")),
            // the leading system RUN folds into the tools header, merged — not leaked out as a
            // second body system turn.
            (
                "two_system_tools_low",
                &two_system[..],
                ThinkMode::Think,
                Some("low"),
            ),
        ] {
            let golden = golden(name);
            let got = q38(turns, think, effort, &tools);
            assert_eq!(
                got, golden,
                "{name}: tools-branch render diverges from the vendor's own jinja.\n\
                 got:    {got:?}\nwanted: {golden:?}"
            );
        }
    }

    #[test]
    fn qwen38_ladder_rungs_are_distinct_prompts_and_medium_is_the_neutral_one() {
        // The owner's standard: a level that returns 200 must have an EFFECT, and any gradation
        // must be real. Effect here is measured the only way that cannot lie — prompt bytes.
        let plain = [turn("user", "hi")];
        let r = |effort: Option<&str>| q38(&plain, ThinkMode::Think, effort, &[]);
        let xhigh = r(Some("high"));
        let medium = r(Some("medium"));
        let low = r(Some("low"));
        assert_ne!(
            xhigh, medium,
            "xhigh and medium must not render the same prompt"
        );
        assert_ne!(xhigh, low, "xhigh and low must not render the same prompt");
        assert_ne!(
            medium, low,
            "medium and low must not render the same prompt"
        );
        // `medium` is the vendor's zero-steering rung: no sentence at all, so it renders exactly
        // what a bare ChatML request renders. THIS is what memra produced for EVERY q38 request
        // before this lane, at every effort level and at the default — which is why landing the
        // fix changes the default prompt (an operator `default_reasoning_effort: "medium"` keeps
        // it byte-identical to that history, and that is the documented no-op migration).
        assert!(
            !medium.contains("Reasoning effort is set to"),
            "medium must inject no sentence: {medium:?}"
        );
        assert_eq!(
            medium, "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n",
            "medium is the pre-lane byte history for q38"
        );
        // The default is NOT medium — it is the vendor's xhigh. The serving-behaviour change.
        assert_eq!(
            r(None),
            xhigh,
            "an unset level is the template's own xhigh default"
        );
        assert!(
            xhigh.len() > medium.len() + 200,
            "xhigh adds a real instruction"
        );
    }

    #[test]
    fn a_qwen_template_without_the_ladder_is_byte_identical_at_every_level() {
        // ORNITH, and the construction fact behind the server's TRANSLATION rule. Ornith AI
        // documents no graded effort anywhere (zero `reasoning_effort` occurrences across every
        // card in the org, both generations, all sizes; the entire control surface is one
        // `enable_thinking` guard). So the level has nothing to land on, and low/medium/high
        // render the SAME BYTES as an unset request. The server therefore folds a graded level
        // onto the binary axis as reasoning ON (coordinator ruling 2026-08-23 — stock codex and
        // Claude Code send `xhigh` on every request, and a caller who asked for reasoning and
        // gets reasoning has their promise kept); this test pins the byte-identity that makes
        // that translation honest rather than decorative.
        const ORNITH_TMPL: &str = include_str!(
            "../../../research/reasoning-schema-20260823/ornith15.chat_template.jinja"
        );
        let plain = [turn("user", "hi")];
        let r = |think: ThinkMode, effort: Option<&str>| {
            apply_chat_template_tools(Some(ORNITH_TMPL), &plain, true, &[], think, effort)
                .expect("ornith render")
        };
        let base = r(ThinkMode::Default, None);
        for level in ["low", "medium", "high"] {
            assert_eq!(
                r(ThinkMode::Think, Some(level)),
                base,
                "{level} must be byte-identical on a ladder-less template — the fact that makes \
                 the graded->ON translation exact rather than approximate"
            );
        }
        // The BINARY axis is real here, and it is the one control ornith's lab defines.
        assert!(base.ends_with("<think>\n"), "{base:?}");
        assert!(
            r(ThinkMode::NoThink, None).ends_with("<think>\n\n</think>\n\n"),
            "ornith honours reasoning-off through its enable_thinking guard"
        );
        assert!(
            !base.contains("Reasoning effort is set to"),
            "the qwen3.8 sentence must NEVER leak onto a template that does not define it"
        );
    }

    fn golden(name: &str) -> String {
        // Goldens live next to the generator that made them, so a reviewer can regenerate and
        // diff. `include_str!` rather than a runtime read: the path is checked at compile time,
        // so moving the fixture breaks the build instead of silently skipping the gate.
        macro_rules! g {
            ($($n:literal),* $(,)?) => {
                match name {
                    $($n => include_str!(concat!(
                        "../../../research/reasoning-schema-20260823/goldens/", $n, ".txt"
                    )).to_string(),)*
                    other => panic!("no golden named {other}"),
                }
            };
        }
        g!(
            "plain_default",
            "plain_xhigh",
            "plain_medium",
            "plain_low",
            "plain_off",
            "plain_off_with_level",
            "system_default",
            "system_xhigh",
            "system_medium",
            "system_low",
            "system_off",
            "empty_system_low",
            "two_system_xhigh",
            "two_system_off",
            "two_system_tools_low",
            "multiturn_off",
            "multiturn_default",
            "multiturn_reasoned_off",
            "tools_default",
            "tools_xhigh",
            "tools_medium",
            "tools_low",
        )
    }
}
