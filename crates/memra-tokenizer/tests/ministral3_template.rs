use memra_tokenizer::chat::{self, ThinkMode, ToolCall, Turn};
const TEMPLATE: &str = include_str!("fixtures/ministral3/chat_template.jinja");
fn turn(role: &str, content: &str) -> Turn {
    Turn {
        role: role.into(),
        content: content.into(),
        ..Default::default()
    }
}
fn render(turns: &[Turn], tools: &[String]) -> String {
    chat::apply_chat_template_tools(Some(TEMPLATE), turns, true, tools, ThinkMode::Default, None)
        .unwrap()
}
fn expected(name: &str) -> String {
    std::fs::read_to_string(format!(
        "{}/tests/fixtures/ministral3/{name}.txt",
        env!("CARGO_MANIFEST_DIR")
    ))
    .expect("generate pinned Jinja oracle fixtures remotely before tests")
}
#[test]
fn canonical_template_bytes() {
    assert_eq!(render(&[turn("user", "שלום")], &[]), expected("plain"));
    assert_eq!(
        render(&[turn("system", "ענה בעברית"), turn("user", "Hello!")], &[]),
        expected("system")
    );
    assert_eq!(
        render(
            &[turn("user", "א"), turn("user", ""), turn("user", "ב")],
            &[]
        ),
        expected("aggregate")
    );
    let mut assistant = turn("assistant", "");
    assistant.tool_calls = vec![
        ToolCall {
            name: "clock".into(),
            raw_arguments: Some("{ \"zone\" : \"Asia/Jerusalem\" }".into()),
            ..Default::default()
        },
        ToolCall {
            name: "count".into(),
            raw_arguments: Some("".into()),
            ..Default::default()
        },
    ];
    assert_eq!(
        render(
            &[
                turn("system", "עברית"),
                turn("user", "בדוק"),
                assistant,
                turn("tool", "12:00"),
                turn("tool", "7"),
                turn("assistant", "השעה שתים עשרה.")
            ],
            &[]
        ),
        expected("multiple_calls")
    );
    let tools=vec![r#"{"type": "function", "function": {"name": "clock", "parameters": {"type": "object", "properties": {}}}}"#.to_string()];
    assert_eq!(
        render(&[turn("system", "עברית"), turn("user", "בדוק")], &tools),
        expected("tools")
    );
}
#[test]
fn invalid_role_order_is_refused() {
    for turns in [
        vec![],
        vec![turn("assistant", "x")],
        vec![turn("user", "x"), turn("tool", "x")],
        vec![turn("system", "x"), turn("tool", "x")],
        vec![turn("user", "x"), turn("assistant", "")],
    ] {
        assert!(
            chat::apply_chat_template_tools(
                Some(TEMPLATE),
                &turns,
                true,
                &[],
                ThinkMode::Default,
                None
            )
            .is_err()
        );
    }
}
