//! Parse JSON declarations into one tree, rejecting duplicate decoded keys recursively.
//!
//! Preserve number tokens until the consumer chooses its numeric type. Going through f64
//! can change f32 rounding or turn a fractional dimension into an apparent integer.

use serde::de::{self, Deserialize, DeserializeSeed, Deserializer, MapAccess, SeqAccess, Visitor};
use serde_json::{Map, Value, value::RawValue};
use std::fmt;

const MAX_DEPTH: usize = 128;

struct UniqueValueSeed {
    depth: usize,
}

impl<'de> DeserializeSeed<'de> for UniqueValueSeed {
    type Value = Value;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<Value, D::Error> {
        if self.depth > MAX_DEPTH {
            return Err(de::Error::custom(
                "JSON declaration nesting exceeds 128 levels",
            ));
        }
        let raw = <&RawValue>::deserialize(deserializer)?;
        let text = raw.get();
        // serde's integer visitor represents the valid JSON token -0 as i64(0).
        // Keep its sign for direct f32 normalization, just as parsing the token did.
        if text == "-0" {
            return Ok(Value::Number(serde_json::Number::from_f64(-0.0).unwrap()));
        }
        let mut decoder = serde_json::Deserializer::from_str(text);
        let visitor = UniqueVisitor { depth: self.depth };
        // Classify the original token, not a decoded object's keys. In arbitrary_precision
        // mode serde uses a private map sentinel for numbers; an actual JSON object with
        // that key must remain an object and still receive duplicate-key checks.
        let value = match text.as_bytes()[0] {
            b'{' => decoder.deserialize_map(visitor),
            b'[' => decoder.deserialize_seq(visitor),
            _ => Value::deserialize(&mut decoder),
        }
        .map_err(de::Error::custom)?;
        decoder.end().map_err(de::Error::custom)?;
        Ok(value)
    }
}

struct UniqueVisitor {
    depth: usize,
}

impl<'de> Visitor<'de> for UniqueVisitor {
    type Value = Value;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("JSON with unique decoded object keys")
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Value, A::Error> {
        let mut values = Map::new();
        while let Some(key) = map.next_key::<String>()? {
            if values.contains_key(&key) {
                return Err(de::Error::custom(format!("duplicate JSON key {key:?}")));
            }
            let value = map.next_value_seed(UniqueValueSeed {
                depth: self.depth + 1,
            })?;
            values.insert(key, value);
        }
        Ok(Value::Object(values))
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut sequence: A) -> Result<Value, A::Error> {
        let mut values = Vec::new();
        while let Some(value) = sequence.next_element_seed(UniqueValueSeed {
            depth: self.depth + 1,
        })? {
            values.push(value);
        }
        Ok(Value::Array(values))
    }
}

pub(crate) fn parse_object(text: &str) -> Result<Map<String, Value>, String> {
    let mut decoder = serde_json::Deserializer::from_str(text);
    let value = decoder
        .deserialize_map(UniqueVisitor { depth: 0 })
        .map_err(|e| e.to_string())?;
    decoder.end().map_err(|e| e.to_string())?;
    match value {
        Value::Object(object) => Ok(object),
        _ => Err("JSON declaration must be an object".into()),
    }
}

pub(crate) fn optional_string<'a>(
    object: &'a Map<String, Value>,
    key: &str,
) -> Result<Option<&'a str>, String> {
    object
        .get(key)
        .map(|value| {
            value
                .as_str()
                .ok_or_else(|| format!("JSON field {key} must be a string"))
        })
        .transpose()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn number_tokens_remain_exact_and_cannot_be_forged_by_object_keys() {
        let value = parse_object(r#"{"x":1.0000000596046448,"n":2.0000000000000001,"zero":-0,"object":{"$serde_json::private::Number":"17"}}"#).unwrap();
        assert_eq!(
            value["x"].as_number().unwrap().as_str(),
            "1.0000000596046448"
        );
        assert_eq!(
            value["n"].as_number().unwrap().as_str(),
            "2.0000000000000001"
        );
        assert!(value["object"].is_object());
        assert!(value["zero"].as_number().unwrap().as_str().starts_with('-'));
        assert_eq!(value["object"]["$serde_json::private::Number"], "17");
        assert!(parse_object(r#"{"object":{"$serde_json::private::Number":"17","$serde_json::private::\u004eumber":"18"}}"#).is_err());
    }

    #[test]
    fn raw_value_recursion_remains_bounded() {
        let nested = |count| format!("{{\"x\":{}0{}}}", "[".repeat(count), "]".repeat(count));
        assert!(parse_object(&nested(32)).is_ok());
        assert!(parse_object(&nested(129)).is_err());
    }

    #[test]
    fn integer_schema_consumers_can_still_reject_float_string_and_null_tokens() {
        let values = parse_object(r#"{"integer":17,"max":18446744073709551615,"overflow":18446744073709551616,"float":17.0,"exponent":17e0,"string":"17","null":null,"negative":-1,"negative_zero":-0}"#).unwrap();
        assert_eq!(values["integer"].as_u64(), Some(17));
        assert_eq!(values["max"].as_u64(), Some(u64::MAX));
        for key in [
            "overflow",
            "float",
            "exponent",
            "string",
            "null",
            "negative",
            "negative_zero",
        ] {
            assert_eq!(values[key].as_u64(), None, "{key}");
        }
    }

    #[test]
    fn declaration_keys_and_values_share_unicode_decoding() {
        let value = parse_object(r#"{"pruned\u005fexperts":{"\u0030":[0,2]},"format":"memra-expert-overlay\u002dv2","file":"caf\u00e9-\ud83d\ude00.bin"}"#).unwrap();
        assert_eq!(value["pruned_experts"]["0"], serde_json::json!([0, 2]));
        assert_eq!(
            optional_string(&value, "format").unwrap(),
            Some("memra-expert-overlay-v2")
        );
        assert_eq!(
            optional_string(&value, "file").unwrap(),
            Some("café-😀.bin")
        );
    }

    #[test]
    fn decoded_duplicates_and_malformed_strings_refuse_at_every_level() {
        for text in [
            r#"{"pruned_experts":{},"pruned\u005fexperts":{}}"#,
            r#"{"tensors":{"x":1,"\u0078":2}}"#,
            r#"{"tensors":{"x":{"qtype":"F32","q\u0074ype":"BF16"}}}"#,
            r#"{"x":"\ud800"}"#,
            r#"{"x":"\q"}"#,
            r#"{"x":1} trailing"#,
        ] {
            assert!(parse_object(text).is_err(), "{text}");
        }
    }
}
