//! Compare Memra's pinned MiMo text chat bytes and IDs with Transformers goldens.

use std::{collections::HashMap, path::Path};

use memra_tokenizer::{
    Tokenizer,
    chat::{ThinkMode, ToolCall, Turn, Val},
};

const TOOL_JSON: &str = r#"{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}}}}"#;

struct Case {
    name: &'static str,
    turns: Vec<Turn>,
    generation_prompt: bool,
    think: ThinkMode,
    tools_json: Vec<String>,
    plain: bool,
}

fn turn(role: &str, content: &str) -> Turn {
    Turn {
        role: role.into(),
        content: content.into(),
        ..Default::default()
    }
}

fn cases() -> Vec<Case> {
    let system_user = vec![
        turn("system", "Keep punctuation exact."),
        turn("user", "  Café and 你好?  "),
    ];
    let answer = vec![turn(
        "user",
        "What is 17 plus 25? Answer with the number only.",
    )];
    let tool = vec![
        turn("user", "Weather for 東京, please."),
        Turn {
            role: "assistant".into(),
            content: "Checking now.".into(),
            reasoning: Some("Call lookup.".into()),
            tool_calls: vec![ToolCall {
                name: "lookup".into(),
                params: vec![("city".into(), "東京".into()), ("days".into(), "2".into())],
                args: vec![
                    ("city".into(), Val::Str("東京".into())),
                    ("days".into(), Val::Num("2".into())),
                ],
                id: Some("call_weather".into()),
            }],
            ..Default::default()
        },
        turn("tool", r#"{"temp_c": 21}"#),
    ];
    vec![
        Case {
            name: "system_user_gen",
            turns: system_user.clone(),
            generation_prompt: true,
            think: ThinkMode::Default,
            tools_json: vec![],
            plain: true,
        },
        Case {
            name: "system_user_closed",
            turns: system_user.clone(),
            generation_prompt: true,
            think: ThinkMode::NoThink,
            tools_json: vec![],
            plain: false,
        },
        Case {
            name: "system_user_final",
            turns: system_user,
            generation_prompt: false,
            think: ThinkMode::Default,
            tools_json: vec![],
            plain: true,
        },
        Case {
            name: "tool_history_gen",
            turns: tool.clone(),
            generation_prompt: true,
            think: ThinkMode::Default,
            tools_json: vec![TOOL_JSON.into()],
            plain: false,
        },
        Case {
            name: "tool_history_closed",
            turns: tool.clone(),
            generation_prompt: true,
            think: ThinkMode::NoThink,
            tools_json: vec![TOOL_JSON.into()],
            plain: false,
        },
        Case {
            name: "tool_history_final",
            turns: tool,
            generation_prompt: false,
            think: ThinkMode::Default,
            tools_json: vec![TOOL_JSON.into()],
            plain: false,
        },
        Case {
            name: "answer_17_plus_25",
            turns: answer,
            generation_prompt: true,
            think: ThinkMode::Default,
            tools_json: vec![],
            plain: true,
        },
    ]
}

fn unhex(hex: &str) -> Result<Vec<u8>, String> {
    if !hex.len().is_multiple_of(2) {
        return Err("golden has odd hex length".into());
    }
    (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).map_err(|e| e.to_string()))
        .collect()
}

fn first_diff<T: PartialEq>(a: &[T], b: &[T]) -> usize {
    a.iter()
        .zip(b)
        .position(|(x, y)| x != y)
        .unwrap_or(a.len().min(b.len()))
}

fn run(source: &Path, golden_path: &Path) -> Result<(), String> {
    let tokenizer = Tokenizer::from_hf_dir(source)?;
    if tokenizer.chat_template()
        != Some(include_str!(
            "../../../../research/mimo-chat-template-cpu-20260927/source-template.jinja"
        ))
    {
        return Err("pinned MiMo source template differs".into());
    }
    if tokenizer.pre() != "qwen2" {
        return Err(format!(
            "unexpected source pre-tokenizer: {}",
            tokenizer.pre()
        ));
    }

    let mut goldens = HashMap::new();
    let lines = std::fs::read_to_string(golden_path).map_err(|e| e.to_string())?;
    for line in lines.lines() {
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() != 3 {
            return Err("golden row must have three fields".into());
        }
        let ids = fields[2]
            .split(',')
            .map(|id| id.parse::<u32>().map_err(|e| e.to_string()))
            .collect::<Result<Vec<_>, _>>()?;
        if goldens
            .insert(fields[0], (unhex(fields[1])?, ids))
            .is_some()
        {
            return Err(format!("duplicate golden case {}", fields[0]));
        }
    }

    let mut failed = 0;
    let cases = cases();
    for case in &cases {
        let (expected_bytes, expected_ids) = goldens
            .remove(case.name)
            .ok_or_else(|| format!("missing golden case {}", case.name))?;
        let rendered = if case.plain {
            let messages = case
                .turns
                .iter()
                .map(|turn| (turn.role.as_str(), turn.content.as_str()))
                .collect::<Vec<_>>();
            tokenizer.apply_chat_template(&messages, case.generation_prompt)
        } else {
            tokenizer.apply_chat_template_tools(
                &case.turns,
                case.generation_prompt,
                &case.tools_json,
                case.think,
                None,
            )?
        };
        let got_bytes = rendered.as_bytes();
        let got_ids = tokenizer.encode(&rendered, false);
        if got_bytes != expected_bytes || got_ids != expected_ids {
            failed += 1;
            println!(
                "MISMATCH {}: bytes first_diff={} memra={} source={}; ids first_diff={} memra={} source={}",
                case.name,
                first_diff(got_bytes, &expected_bytes),
                got_bytes.len(),
                expected_bytes.len(),
                first_diff(&got_ids, &expected_ids),
                got_ids.len(),
                expected_ids.len()
            );
        } else {
            println!(
                "OK {}: {} bytes, {} ids",
                case.name,
                got_bytes.len(),
                got_ids.len()
            );
        }
    }
    if !goldens.is_empty() {
        return Err(format!("unexpected golden cases: {:?}", goldens.keys()));
    }
    if failed != 0 {
        return Err(format!("{failed}/{} MiMo chat cases differ", cases.len()));
    }
    println!(
        "MiMo chat parity: {}/{} exact byte and token-ID cases",
        cases.len(),
        cases.len()
    );
    Ok(())
}

fn main() -> Result<(), String> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("usage: tok-mimo-chat-parity <pinned_hf_dir> <goldens.tsv>".into());
    }
    run(Path::new(&args[1]), Path::new(&args[2]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use memra_tokenizer::chat;

    const SOURCE: &str =
        include_str!("../../../../research/mimo-chat-template-cpu-20260927/source-template.jinja");
    const GOLDENS: &str =
        include_str!("../../../../research/mimo-chat-template-cpu-20260927/goldens.tsv");

    #[test]
    fn pinned_source_render_bytes() {
        let expected = GOLDENS
            .lines()
            .map(|line| {
                let fields: Vec<&str> = line.split('\t').collect();
                (fields[0], unhex(fields[1]).unwrap())
            })
            .collect::<HashMap<_, _>>();
        let cases = cases();
        assert_eq!(expected.len(), cases.len());
        for case in cases {
            let rendered = if case.plain {
                let messages = case
                    .turns
                    .iter()
                    .map(|turn| (turn.role.as_str(), turn.content.as_str()))
                    .collect::<Vec<_>>();
                chat::apply_chat_template_str(Some(SOURCE), &messages, case.generation_prompt)
            } else {
                chat::apply_chat_template_tools(
                    Some(SOURCE),
                    &case.turns,
                    case.generation_prompt,
                    &case.tools_json,
                    case.think,
                    None,
                )
                .unwrap()
            };
            assert_eq!(
                rendered.as_bytes(),
                expected[case.name],
                "case {}",
                case.name
            );
        }
    }

    #[test]
    fn unimplemented_mimo_template_inputs_refuse() {
        let err = chat::apply_chat_template_tools(
            Some(SOURCE),
            &[turn("user", "Hi")],
            true,
            &[],
            ThinkMode::Default,
            Some("high"),
        )
        .unwrap_err();
        assert!(err.contains("no reasoning_effort"));

        let mut with_message_tools = turn("user", "Hi");
        with_message_tools.tools.push(Val::Str("lookup".into()));
        let err = chat::apply_chat_template_tools(
            Some(SOURCE),
            &[with_message_tools],
            true,
            &[],
            ThinkMode::Default,
            None,
        )
        .unwrap_err();
        assert!(err.contains("per-message tool declarations"));
    }
}
