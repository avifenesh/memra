//! Typed HF configuration access over the single strict decoded JSON tree.
use serde_json::{Map, Value};

type Result<T> = std::result::Result<T, String>;

pub(super) struct ConfigObject<'a> {
    object: &'a Map<String, Value>,
    path: String,
}

impl<'a> ConfigObject<'a> {
    pub(super) fn root(object: &'a Map<String, Value>) -> Self {
        Self {
            object,
            path: "$".into(),
        }
    }

    fn field(&self, key: &str) -> String {
        format!("{}.{}", self.path, key)
    }
    fn invalid(&self, key: &str, expected: &str) -> String {
        format!("config {} must be {expected}", self.field(key))
    }
    fn value(&self, key: &str) -> Result<Option<&'a Value>> {
        match self.object.get(key) {
            None => Ok(None),
            Some(Value::Null)
                if matches!(
                    key,
                    // These optional schema values explicitly represent an absent override.
                    "sliding_window"
                        | "num_key_value_heads"
                        | "num_attention_groups"
                        | "head_dim"
                        | "n_shared_experts"
                        | "routed_scaling_factor"
                        | "eos_token_id"
                        | "final_logit_softcapping"
                        | "name_or_path"
                        | "_name_or_path"
                        | "rope_scaling"
                        | "rope_parameters"
                        | "text_config"
                        | "vision_config"
                        | "quantization_config"
                        | "linear_attn_config"
                        | "mtp"
                        | "attention_other_setting"
                ) =>
            {
                Ok(None)
            }
            Some(Value::Null) => Err(self.invalid(key, "a non-null declared value")),
            Some(value) => Ok(Some(value)),
        }
    }

    pub(super) fn string(&self, key: &str) -> Result<Option<String>> {
        if self.value(key)?.is_none() {
            return Ok(None);
        }
        crate::strict_json::optional_string(self.object, key)
            .map(|value| value.map(str::to_owned))
            .map_err(|_| self.invalid(key, "a string"))
    }

    fn unsigned(&self, value: &Value, key: &str, max: u64) -> Result<u64> {
        let number = value.as_number().and_then(|n| exact_unsigned(n.as_str()));
        match number {
            Some(n) if n <= max => Ok(n),
            _ => Err(self.invalid(key, &format!("an unsigned integer in 0..={max}"))),
        }
    }
    fn float(&self, value: &Value, key: &str) -> Result<f32> {
        match value
            .as_number()
            .and_then(|n| n.as_str().parse::<f32>().ok())
        {
            Some(n) if n.is_finite() => Ok(n),
            _ => Err(self.invalid(key, "a finite f32 number")),
        }
    }
    fn transformer_scope(&self) -> bool {
        self.path == "$" || self.path == "$.text_config"
    }

    pub(super) fn u32(&self, key: &str) -> Result<Option<u32>> {
        match self.value(key)? {
            None => Ok(None),
            Some(Value::Array(_)) if key == "eos_token_id" && self.transformer_scope() => Ok(None),
            Some(value) => self
                .unsigned(value, key, u32::MAX.into())
                .map(|n| Some(n as u32)),
        }
    }
    pub(super) fn u64(&self, key: &str) -> Result<Option<u64>> {
        self.value(key)?
            .map(|v| self.unsigned(v, key, u64::MAX))
            .transpose()
    }
    pub(super) fn f32(&self, key: &str) -> Result<Option<f32>> {
        match self.value(key)? {
            None => Ok(None),
            Some(Value::Array(_)) if key == "rope_theta" && self.transformer_scope() => Ok(None),
            Some(value) => self.float(value, key).map(Some),
        }
    }
    pub(super) fn boolean(&self, key: &str) -> Result<Option<bool>> {
        self.value(key)?
            .map(|v| v.as_bool().ok_or_else(|| self.invalid(key, "a boolean")))
            .transpose()
    }
    fn array(&self, key: &str, scalar_alternative: bool) -> Result<Option<&'a Vec<Value>>> {
        match self.value(key)? {
            None => Ok(None),
            Some(Value::Array(values)) => Ok(Some(values)),
            Some(Value::Number(_)) if scalar_alternative => Ok(None),
            Some(_) => Err(self.invalid(key, "an array")),
        }
    }
    pub(super) fn u32_array(&self, key: &str) -> Result<Option<Vec<u32>>> {
        self.array(key, key == "eos_token_id" && self.transformer_scope())?
            .map(|values| {
                values
                    .iter()
                    .enumerate()
                    .map(|(i, v)| {
                        self.unsigned(v, &format!("{key}[{i}]"), u32::MAX.into())
                            .map(|n| n as u32)
                    })
                    .collect()
            })
            .transpose()
    }
    pub(super) fn moe_layer_freq(&self, glm_dsa: bool) -> Result<Option<Vec<u32>>> {
        if let Some(value @ Value::Number(_)) = self.value("moe_layer_freq")? {
            // GLM-DSA declares an interval, while MiniMax declares a layer mask.
            // The existing GLM plan covers every layer after first_k_dense_replace.
            // Only interval 1 denotes that program; never discard a sparse schedule.
            if glm_dsa && self.unsigned(value, "moe_layer_freq", u32::MAX.into())? == 1 {
                return Ok(None);
            }
            return Err(self.invalid(
                "moe_layer_freq",
                "a layer-mask array, or interval 1 for the GLM-DSA program",
            ));
        }
        self.u32_array("moe_layer_freq")
    }
    pub(super) fn f32_array(&self, key: &str) -> Result<Option<Vec<f32>>> {
        self.array(key, key == "rope_theta" && self.transformer_scope())?
            .map(|values| {
                values
                    .iter()
                    .enumerate()
                    .map(|(i, v)| self.float(v, &format!("{key}[{i}]")))
                    .collect()
            })
            .transpose()
    }
    pub(super) fn string_array(&self, key: &str) -> Result<Option<Vec<String>>> {
        self.array(key, false)?
            .map(|values| {
                values
                    .iter()
                    .enumerate()
                    .map(|(i, v)| {
                        v.as_str()
                            .map(str::to_owned)
                            .ok_or_else(|| self.invalid(&format!("{key}[{i}]"), "a string"))
                    })
                    .collect()
            })
            .transpose()
    }
    pub(super) fn first_string_in_array(&self, key: &str) -> Result<Option<String>> {
        Ok(self
            .string_array(key)?
            .and_then(|values| values.into_iter().next()))
    }
    pub(super) fn object(&self, key: &str) -> Result<Option<Self>> {
        self.value(key)?
            .map(|value| {
                value
                    .as_object()
                    .map(|object| Self {
                        object,
                        path: self.field(key),
                    })
                    .ok_or_else(|| self.invalid(key, "an object"))
            })
            .transpose()
    }
}

// JSON syntax is already validated. Accept integer-valued decimal/exponent spellings
// without any floating-point intermediate, truncation, saturation, or wraparound.
fn exact_unsigned(text: &str) -> Option<u64> {
    if text.starts_with('-') {
        return None;
    }
    let (mantissa, exponent) = text.split_once(['e', 'E']).unwrap_or((text, "0"));
    let fractional_digits = mantissa
        .split_once('.')
        .map_or(0, |(_, fraction)| fraction.len());
    let digits: Vec<u8> = mantissa.bytes().filter(|&b| b != b'.').collect();
    if digits.iter().all(|&b| b == b'0') {
        return Some(0);
    }
    let shift = exponent
        .parse::<i64>()
        .ok()?
        .checked_sub(i64::try_from(fractional_digits).ok()?)?;
    let (digits, zeros) = if shift < 0 {
        let remove = usize::try_from(shift.checked_neg()?).ok()?;
        let keep = digits.len().checked_sub(remove)?;
        if !digits[keep..].iter().all(|&b| b == b'0') {
            return None;
        }
        (&digits[..keep], 0)
    } else {
        (&digits[..], usize::try_from(shift).ok()?)
    };
    // Any nonzero integer with more than 20 significant decimal places exceeds u64.
    let first = digits.iter().position(|&b| b != b'0')?;
    let digits = &digits[first..];
    if digits.len().checked_add(zeros)? > 20 {
        return None;
    }
    digits
        .iter()
        .copied()
        .chain(std::iter::repeat_n(b'0', zeros))
        .try_fold(0u64, |n, digit| {
            n.checked_mul(10)?.checked_add(u64::from(digit - b'0'))
        })
}
