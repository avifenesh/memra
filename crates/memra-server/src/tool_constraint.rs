//! Request-scoped tool languages. Ordinary auto/none requests never enter this module.
//! Delimiters stay template-specific; arguments use one schema-checked JSON object.

use serde_json::{Value, json};

use crate::{constrained, worker::ModelCaps};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolDialect {
    Qwen,
    Gemma,
}

impl ToolDialect {
    pub fn delimiters(self) -> (&'static str, &'static str) {
        match self {
            Self::Qwen => ("<tool_call>", "</tool_call>"),
            Self::Gemma => ("<|tool_call>", "<tool_call|>"),
        }
    }

    pub fn from_caps(caps: Option<&ModelCaps>) -> Result<Self, String> {
        let caps = caps
            .filter(|c| c.tools_branch)
            .ok_or("tool_choice/parallel_tool_calls: this template has no tools branch")?;
        if caps.dsv4 || caps.hy3 || caps.glm5 {
            return Err("tool_choice/parallel_tool_calls: constrained tool calls are not supported by this template dialect".into());
        }
        Ok(if caps.gemma_think {
            Self::Gemma
        } else {
            Self::Qwen
        })
    }
}

#[derive(Debug, Clone)]
pub struct ToolLanguage {
    pub dialect: ToolDialect,
    pub required: bool,
    pub parallel: bool,
    pub functions: Vec<(String, Value)>,
}

impl ToolLanguage {
    pub fn new(
        tools: &[Value],
        named: Option<&str>,
        required: bool,
        dialect: ToolDialect,
    ) -> Result<Self, String> {
        if tools.is_empty() || tools.len() > 128 {
            return Err("tool_choice requires between 1 and 128 declared functions".into());
        }
        if serde_json::to_vec(tools).map_err(|e| e.to_string())?.len()
            > constrained::MAX_SCHEMA_BYTES
        {
            return Err("tool_choice tools exceed the constrained schema byte limit".into());
        }
        let mut seen = std::collections::HashSet::new();
        let mut functions = Vec::new();
        for tool in tools {
            if tool.get("type").and_then(Value::as_str) != Some("function") {
                return Err("tool_choice requires function tools".into());
            }
            let function = tool
                .get("function")
                .ok_or("tool_choice needs a function object")?;
            let name = function
                .get("name")
                .and_then(Value::as_str)
                .ok_or("tool_choice needs function.name")?;
            if name.is_empty()
                || name.len() > 64
                || !name
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
                || !seen.insert(name.to_owned())
            {
                return Err(format!(
                    "tool_choice has invalid or duplicate function name {name:?}"
                ));
            }
            let mut schema = function
                .get("parameters")
                .cloned()
                .unwrap_or_else(|| json!({"type":"object"}));
            if !schema.is_object() {
                return Err(format!(
                    "tool_choice function {name:?} parameters must be a schema object"
                ));
            }
            constrained::validate_json_schema(&schema).map_err(|e| {
                format!(
                    "tool_choice function {name:?} parameters: {}",
                    e.replace("response_format.json_schema.schema", "schema")
                )
            })?;
            // Arguments must be an object. Keep the original document root so local
            // $ref/$defs pointers do not move under a synthetic allOf wrapper.
            let allows_object = match schema.get("type") {
                None => true,
                Some(Value::String(kind)) => kind == "object",
                Some(Value::Array(kinds)) => {
                    kinds.iter().any(|kind| kind.as_str() == Some("object"))
                }
                _ => false,
            };
            if !allows_object {
                return Err(format!(
                    "tool_choice function {name:?} parameters must allow an object"
                ));
            }
            schema["type"] = json!("object");
            constrained::validate_json_schema(&schema).map_err(|e| {
                format!(
                    "tool_choice function {name:?} parameters: {}",
                    e.replace("response_format.json_schema.schema", "schema")
                )
            })?;
            if named.is_none_or(|selected| selected == name) {
                functions.push((name.to_owned(), schema));
            }
        }
        if functions.is_empty() {
            return Err("tool_choice function is not declared in tools".into());
        }
        Ok(Self {
            dialect,
            required,
            parallel: false,
            functions,
        })
    }

    /// Protocol controls are explicit llguidance terminals, not literal JSON bytes.
    /// The factory resolves these against the actual loaded tokenizer.
    pub fn lark(&self, terminal: impl Fn(&str) -> String) -> String {
        let (open, close) = self.dialect.delimiters();
        let choices = (0..self.functions.len())
            .map(|i| format!("call{i}"))
            .collect::<Vec<_>>()
            .join(" | ");
        let mut grammar = if self.required {
            let repeat = if self.parallel { "+" } else { "" };
            format!("start: call{repeat}\ncall: {choices}\n")
        } else {
            // A whole-prefix lexeme greedily consumes partial openers before the
            // parser can hand off to a call. These DFA states recognize arbitrary
            // ordinary bytes while reserving only the complete call opener.
            let mut prefix = String::from("prefix: p0\np0: | /[^<]+/ p0 | \"<\" p1\n");
            for (i, ch) in open.chars().enumerate().skip(1) {
                prefix.push_str(&format!("p{i}: | \"<\" p1 | /[^<{ch}]/ p0"));
                if i + 1 < open.len() {
                    prefix.push_str(&format!(
                        " | {} p{}",
                        serde_json::to_string(&ch.to_string()).expect("character"),
                        i + 1
                    ));
                }
                prefix.push('\n');
            }
            format!("start: prefix call?\n{prefix}call: {choices}\n")
        };
        for (i, (name, schema)) in self.functions.iter().enumerate() {
            let (head, tail) = match self.dialect {
                ToolDialect::Qwen => (format!("<function={name}>"), "</function>"),
                ToolDialect::Gemma => (format!("call:{name}"), ""),
            };
            grammar.push_str(&format!(
                "call{i}: {} {} args{i} {} {}\nargs{i}: %json {}\n",
                terminal(open),
                serde_json::to_string(&head).expect("string"),
                serde_json::to_string(tail).expect("string"),
                terminal(close),
                schema,
            ));
        }
        grammar
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tools() -> Vec<Value> {
        vec![
            json!({"type":"function","function":{"name":"weather",
            "parameters":{"type":"object","properties":{"city":{"type":"string"}},
            "required":["city"],"additionalProperties":false}}}),
            json!({"type":"function","function":{"name":"clock"}}),
        ]
    }

    #[test]
    fn named_selects_exact_schema_required_union_and_optional_single_call() {
        let named = ToolLanguage::new(&tools(), Some("weather"), true, ToolDialect::Qwen).unwrap();
        assert_eq!(named.functions.len(), 1);
        assert_eq!(named.functions[0].1["required"], json!(["city"]));
        let literal = |s: &str| serde_json::to_string(s).unwrap();
        assert!(!named.lark(literal).contains("PREFIX"));
        let required = ToolLanguage::new(&tools(), None, true, ToolDialect::Gemma).unwrap();
        assert_eq!(required.functions.len(), 2);
        assert!(required.lark(literal).contains("call0 | call1"));
        let optional = ToolLanguage::new(&tools(), None, false, ToolDialect::Qwen).unwrap();
        assert!(optional.lark(literal).contains("start: prefix call?"));
    }

    #[test]
    fn names_schema_and_unknown_routes_refuse_before_compile() {
        for name in ["", "bad>name", "bad/name", "two words"] {
            let mut value = tools();
            value[0]["function"]["name"] = json!(name);
            assert!(ToolLanguage::new(&value, None, true, ToolDialect::Qwen).is_err());
        }
        let mut duplicate = tools();
        duplicate.push(duplicate[0].clone());
        assert!(ToolLanguage::new(&duplicate, None, true, ToolDialect::Qwen).is_err());
        assert!(ToolLanguage::new(&[], None, true, ToolDialect::Qwen).is_err());
        assert!(ToolLanguage::new(&tools(), Some("missing"), true, ToolDialect::Qwen).is_err());
        let mut invalid = tools();
        invalid[0]["function"]["parameters"] = json!(17);
        assert!(ToolLanguage::new(&invalid, None, true, ToolDialect::Qwen).is_err());
        assert!(ToolDialect::from_caps(None).is_err());
        for caps in [
            ModelCaps {
                tools_branch: true,
                dsv4: true,
                ..Default::default()
            },
            ModelCaps {
                tools_branch: true,
                hy3: true,
                ..Default::default()
            },
            ModelCaps {
                tools_branch: true,
                glm5: true,
                ..Default::default()
            },
        ] {
            assert!(ToolDialect::from_caps(Some(&caps)).is_err());
        }
    }
}
