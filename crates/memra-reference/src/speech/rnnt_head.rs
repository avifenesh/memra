//! Native RNNT head: prompt conditioning, predictor, joint and greedy decoding.
//!
//! The encoder produces one frame per 80 ms chunk; this turns a run of those frames into token
//! ids. The prompt is not a token prepended to the sequence: it is a one-hot language slot
//! concatenated onto every encoder row and projected, which is how the checkpoint's own model
//! class wires it.
//!
//! Reference operation. No serving surface, no GPU, no streaming session lifecycle.

use super::encoder::{Linear, WhisperNumeric, apply_linear};
use memra_gguf::model_packs::nemotron_rnnt::{BoundRnnt, RnntGeometry};
use std::collections::BTreeMap;

type StorageMap = BTreeMap<String, (usize, usize)>;

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

struct LstmLayer {
    input_weight: Linear,
    hidden_weight: Linear,
    /// Both bias vectors are kept separate because the checkpoint stores them separately and
    /// summing them at load would be a different tensor from the one it shipped.
    input_bias: Vec<f32>,
    hidden_bias: Vec<f32>,
}

/// A greedy decode in progress. Its predictor state and last token are what make a streamed
/// session different from a run of independent frames.
pub struct GreedySession {
    predictor: PredictorState,
    last: Option<u32>,
    started: bool,
}

impl GreedySession {
    pub fn last_token(&self) -> Option<u32> {
        self.last
    }
}

/// One LSTM cell state, per layer.
#[derive(Clone)]
pub struct PredictorState {
    hidden: Vec<Vec<f32>>,
    cell: Vec<Vec<f32>>,
}

impl PredictorState {
    pub fn hidden(&self, layer: usize) -> &[f32] {
        &self.hidden[layer]
    }

    pub fn cell(&self, layer: usize) -> &[f32] {
        &self.cell[layer]
    }
}

pub struct RnntHead {
    geometry: RnntGeometry,
    prompt_in: Linear,
    prompt_out: Linear,
    embedding: Vec<f32>,
    layers: Vec<LstmLayer>,
    joint_encoder: Linear,
    joint_predictor: Linear,
    joint_out: Linear,
    blank: u32,
}

impl RnntHead {
    pub fn load(
        geometry: RnntGeometry,
        bound: &BoundRnnt,
        archive: &[u8],
        storages: &StorageMap,
    ) -> Result<Self, String> {
        let read = |name: &str| -> Result<Vec<f32>, String> {
            let entry = bound
                .tensors
                .get(name)
                .ok_or_else(|| format!("bound checkpoint is missing {name}"))?;
            let (at, len) = storages
                .get(&entry.storage_key)
                .ok_or_else(|| format!("{name} has no storage member"))?;
            Ok(archive[*at..*at + *len]
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect())
        };
        let linear = |name: &str, input: usize, output: usize| -> Result<Linear, String> {
            Ok(Linear {
                input,
                output,
                weight: read(&format!("{name}.weight"))?,
                bias: Some(read(&format!("{name}.bias"))?),
            })
        };
        let width = geometry.encoder_width as usize;
        let predictor = geometry.predictor_width as usize;
        let joint = geometry.joint_width as usize;
        let vocabulary = geometry.vocabulary as usize;
        let mut layers = Vec::new();
        for layer in 0..geometry.predictor_layers {
            let stem = "decoder.prediction.dec_rnn.lstm";
            layers.push(LstmLayer {
                input_weight: Linear {
                    input: predictor,
                    output: 4 * predictor,
                    weight: read(&format!("{stem}.weight_ih_l{layer}"))?,
                    bias: None,
                },
                hidden_weight: Linear {
                    input: predictor,
                    output: 4 * predictor,
                    weight: read(&format!("{stem}.weight_hh_l{layer}"))?,
                    bias: None,
                },
                input_bias: read(&format!("{stem}.bias_ih_l{layer}"))?,
                hidden_bias: read(&format!("{stem}.bias_hh_l{layer}"))?,
            });
        }
        Ok(Self {
            geometry,
            prompt_in: linear(
                "prompt_kernel.0",
                width + geometry.prompt_slots as usize,
                geometry.prompt_hidden as usize,
            )?,
            prompt_out: linear("prompt_kernel.2", geometry.prompt_hidden as usize, width)?,
            embedding: read("decoder.prediction.embed.weight")?,
            layers,
            joint_encoder: linear("joint.enc", width, joint)?,
            joint_predictor: linear("joint.pred", predictor, joint)?,
            joint_out: linear("joint.joint_net.2", joint, vocabulary)?,
            // The blank is the last class: 13087 in a 13088-wide output.
            blank: geometry.vocabulary - 1,
        })
    }

    pub fn blank(&self) -> u32 {
        self.blank
    }

    /// Condition encoder frames on a language slot. `frames` is `[count, width]`.
    pub fn prompt(&self, frames: &[f32], count: usize, slot: u32) -> Result<Vec<f32>, String> {
        let width = self.geometry.encoder_width as usize;
        let slots = self.geometry.prompt_slots as usize;
        if slot as usize >= slots {
            return Err(format!(
                "prompt slot {slot} is outside the {slots} the model has"
            ));
        }
        if frames.len() != count * width {
            return Err("prompt input is not [frames, width]".into());
        }
        let mut joined = vec![0.0f32; count * (width + slots)];
        for frame in 0..count {
            let row = frame * (width + slots);
            joined[row..row + width].copy_from_slice(&frames[frame * width..(frame + 1) * width]);
            joined[row + width + slot as usize] = 1.0;
        }
        let mut hidden = apply_linear(&joined, count, &self.prompt_in, WhisperNumeric::F32);
        for value in hidden.iter_mut() {
            *value = value.max(0.0);
        }
        Ok(apply_linear(
            &hidden,
            count,
            &self.prompt_out,
            WhisperNumeric::F32,
        ))
    }

    pub fn new_state(&self) -> PredictorState {
        let width = self.geometry.predictor_width as usize;
        PredictorState {
            hidden: vec![vec![0.0; width]; self.layers.len()],
            cell: vec![vec![0.0; width]; self.layers.len()],
        }
    }

    /// One predictor step. `token` is `None` at the start of a sequence, which the reference
    /// feeds as a zero row rather than as an embedded blank.
    pub fn predict(
        &self,
        token: Option<u32>,
        state: &PredictorState,
    ) -> Result<(Vec<f32>, PredictorState), String> {
        let width = self.geometry.predictor_width as usize;
        let mut x = match token {
            None => vec![0.0f32; width],
            Some(id) => {
                if id as usize >= self.embedding.len() / width {
                    return Err(format!("token {id} is outside the predictor embedding"));
                }
                self.embedding[id as usize * width..(id as usize + 1) * width].to_vec()
            }
        };
        let mut next = state.clone();
        for (index, layer) in self.layers.iter().enumerate() {
            let from_input = apply_linear(&x, 1, &layer.input_weight, WhisperNumeric::F32);
            let from_hidden = apply_linear(
                &state.hidden[index],
                1,
                &layer.hidden_weight,
                WhisperNumeric::F32,
            );
            let mut hidden = vec![0.0f32; width];
            let mut cell = vec![0.0f32; width];
            for unit in 0..width {
                let gate = |slot: usize| {
                    from_input[slot * width + unit]
                        + layer.input_bias[slot * width + unit]
                        + from_hidden[slot * width + unit]
                        + layer.hidden_bias[slot * width + unit]
                };
                // Torch packs the gates input, forget, cell, output in that order.
                let input_gate = sigmoid(gate(0));
                let forget_gate = sigmoid(gate(1));
                let candidate = gate(2).tanh();
                let output_gate = sigmoid(gate(3));
                let updated = forget_gate.mul_add(state.cell[index][unit], input_gate * candidate);
                cell[unit] = updated;
                hidden[unit] = output_gate * updated.tanh();
            }
            next.hidden[index] = hidden.clone();
            next.cell[index] = cell;
            x = hidden;
        }
        Ok((x, next))
    }

    /// Joint logits for one encoder frame against one predictor row.
    pub fn joint(&self, frame: &[f32], predictor: &[f32]) -> Result<Vec<f32>, String> {
        let width = self.geometry.encoder_width as usize;
        if frame.len() != width || predictor.len() != self.geometry.predictor_width as usize {
            return Err("joint inputs do not match the declared widths".into());
        }
        let encoded = apply_linear(frame, 1, &self.joint_encoder, WhisperNumeric::F32);
        let predicted = apply_linear(predictor, 1, &self.joint_predictor, WhisperNumeric::F32);
        let sum: Vec<f32> = encoded
            .iter()
            .zip(&predicted)
            .map(|(a, b)| (a + b).max(0.0))
            .collect();
        Ok(apply_linear(&sum, 1, &self.joint_out, WhisperNumeric::F32))
    }

    /// A greedy session: the predictor state and the last emitted token, carried across
    /// chunks. A streaming decode that reset this per chunk would be a different program.
    pub fn new_greedy(&self) -> GreedySession {
        GreedySession {
            predictor: self.new_state(),
            last: None,
            started: false,
        }
    }

    /// One frame through the greedy loop, returning what it emitted for that frame.
    pub fn greedy_frame(
        &self,
        frame: &[f32],
        session: &mut GreedySession,
        max_symbols: usize,
    ) -> Result<Vec<u32>, String> {
        let mut emitted = Vec::new();
        let mut symbols = 0usize;
        let mut blank = false;
        while !blank && symbols < max_symbols {
            let token = if session.started { session.last } else { None };
            let (row, next) = self.predict(token, &session.predictor)?;
            let logits = self.joint(frame, &row)?;
            let mut best = 0u32;
            let mut value = f32::NEG_INFINITY;
            for (id, &logit) in logits.iter().enumerate() {
                if logit > value {
                    value = logit;
                    best = id as u32;
                }
            }
            if best == self.blank {
                blank = true;
            } else {
                emitted.push(best);
                session.predictor = next;
                session.last = Some(best);
                session.started = true;
            }
            // The counter advances on a blank as well, so this bounds work per frame and not
            // only emissions.
            symbols += 1;
        }
        Ok(emitted)
    }

    /// Greedy RNNT decode over prompted encoder frames.
    ///
    /// The symbol counter advances on a blank as well as on an emission, so `max_symbols`
    /// bounds the work per frame and not only the emissions.
    pub fn greedy(
        &self,
        prompted: &[f32],
        frames: usize,
        max_symbols: usize,
    ) -> Result<Vec<u32>, String> {
        let width = self.geometry.encoder_width as usize;
        if prompted.len() != frames * width {
            return Err("greedy input is not [frames, width]".into());
        }
        let mut emitted: Vec<u32> = Vec::new();
        let mut session = self.new_greedy();
        for frame in 0..frames {
            let f = &prompted[frame * width..(frame + 1) * width];
            emitted.extend(self.greedy_frame(f, &mut session, max_symbols)?);
        }
        Ok(emitted)
    }
}

#[cfg(test)]
mod tests;
