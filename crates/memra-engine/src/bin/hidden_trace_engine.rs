//! One side of a hidden-state bisect: the reference executor or the CUDA engine, on a
//! deterministic f32 fixture.
//!
//! `--side reference` runs `memra_reference::execute_tokens`; `--side engine` loads the same f32
//! fixture into the CUDA engine (`HybridModel::load_from_source_without_mtp`) and runs
//! `forward_last`. Each side writes a `memra-hidden-trace-v1` trace to the path in
//! `MEMRA_HYPER_TRACE` (the existing seam): the taps the executors already carry (today the
//! hyper-connection walk), then on the reference side one `layer_hidden` line per trunk layer
//! (`ReferenceOutput::layer_hidden`), and a final `logits` line. Run the sides as separate
//! processes and compare with `tools/hidden_trace_diff.py`. The fixture is
//! `deterministic_fixture` of the plan, so both sides read the same numbers.
//!
//! The plan comes from a `<config.json>` (both sides) or from a pack's tiny plan (`--tiny
//! <family>`, reference side only: a tiny plan has no config for the engine loader).
//!
//! `--inject <layer>:<substring|largest>:<scale>` scales dense weight tensors of one trunk layer
//! before the run, on the side it is given to, so a gate can prove the bisect names the layer it
//! broke (`tools/test_hidden_trace_localizer.py`). `largest` picks the layer's largest dense
//! matrix, so one harness covers every pack without knowing its tensor names.
//!
//! The engine side sets `NVIDIA_TF32_OVERRIDE=0` before CUDA init unless the caller set it: an
//! f32 parity comparison must not ride TF32.
//!
//! Usage: MEMRA_HYPER_TRACE=<trace> hidden_trace_engine (<config.json> | --tiny <family>)
//!            --side reference|engine [--inject L:SUB:SCALE]

use memra_engine::Engine;
use memra_engine::hybrid::HybridModel;
use memra_gguf::GgmlType;
use memra_gguf::config::{HfConfig, ModelConfig};
use memra_gguf::model_plan::ModelPlan;
use memra_gguf::source::{TensorSource, TensorView};
use memra_gguf::tensor_contract::{
    CheckpointDialect, ContractOptions, OutputHead, TensorContract, TensorId, TensorMatch,
};
use memra_reference::{ReferenceWeights, deterministic_fixture, hidden_trace};
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::process::ExitCode;

struct OwnedTensor {
    bytes: Vec<u8>,
    ne: Vec<u64>,
}

struct FixtureSource {
    config: ModelConfig,
    tensors: BTreeMap<String, OwnedTensor>,
}

impl TensorSource for FixtureSource {
    fn config(&self) -> ModelConfig {
        self.config.clone()
    }

    fn find(&self, name: &str) -> Option<TensorView<'_>> {
        let tensor = self.tensors.get(name)?;
        Some(TensorView {
            bytes: Cow::Borrowed(&tensor.bytes),
            ggml_type: GgmlType::F32,
            ne: tensor.ne.clone(),
        })
    }

    fn tensor_census(&self) -> Result<memra_gguf::source::TensorCensus, String> {
        Ok(memra_gguf::source::census_from_views(
            self.tensors.iter().map(|(name, tensor)| {
                (
                    name.as_str(),
                    GgmlType::F32,
                    tensor.ne.as_slice(),
                    tensor.bytes.len() as u64,
                )
            }),
        ))
    }
}

fn fixture_source(
    config: &ModelConfig,
    plan: &ModelPlan,
    weights: &ReferenceWeights,
) -> Result<FixtureSource, String> {
    let contract = TensorContract::for_plan(
        plan,
        CheckpointDialect::Gguf,
        ContractOptions {
            output_head: OutputHead::Separate,
        },
    )
    .map_err(|error| format!("tensor contract: {error:?}"))?;
    let mut tensors = BTreeMap::new();
    for requirement in contract
        .requirements
        .iter()
        .filter(|requirement| requirement.required || weights.contains_key(&requirement.id))
    {
        // A family that declares a separate head reads the embedding rows as `output.weight`
        // in this fixture, as the reference does.
        let tensor = weights
            .get(&requirement.id)
            .or_else(|| {
                (requirement.id == TensorId::OutputProjection)
                    .then(|| weights.get(&TensorId::TokenEmbedding))
                    .flatten()
            })
            .ok_or_else(|| format!("fixture is missing {:?}", requirement.id))?;
        let elements: usize = requirement.shape.iter().map(|&dim| dim as usize).product();
        if elements != tensor.data.len() {
            return Err(format!(
                "fixture {:?} has {} elements, the contract requires {elements}",
                requirement.id,
                tensor.data.len()
            ));
        }
        let bytes: Vec<u8> = tensor
            .data
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let names = match requirement.match_mode {
            TensorMatch::OneOf => &requirement.names[..1],
            TensorMatch::All => requirement.names.as_slice(),
        };
        for name in names {
            if tensors
                .insert(
                    name.clone(),
                    OwnedTensor {
                        bytes: bytes.clone(),
                        ne: requirement.shape.clone(),
                    },
                )
                .is_some()
            {
                return Err(format!("duplicate fixture tensor {name}"));
            }
        }
    }
    Ok(FixtureSource {
        config: config.clone(),
        tensors,
    })
}

fn inject(weights: &mut ReferenceWeights, spec: &str) -> Result<Vec<String>, String> {
    let mut parts = spec.splitn(3, ':');
    let layer: u32 = parts
        .next()
        .and_then(|value| value.parse().ok())
        .ok_or_else(|| format!("--inject {spec:?}: layer is not an integer"))?;
    let substring = parts
        .next()
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("--inject {spec:?}: missing tensor substring"))?;
    let scale: f32 = parts
        .next()
        .and_then(|value| value.parse().ok())
        .filter(|value: &f32| value.is_finite())
        .ok_or_else(|| format!("--inject {spec:?}: scale is not a finite number"))?;
    let in_layer = |id: &TensorId| matches!(id, TensorId::Layer { index, .. } if *index == layer);
    let chosen: Vec<TensorId> = if substring == "largest" {
        weights
            .iter()
            .filter(|(id, tensor)| in_layer(id) && tensor.ints.is_none() && tensor.shape.len() >= 2)
            .max_by_key(|(id, tensor)| (tensor.data.len(), std::cmp::Reverse((*id).clone())))
            .map(|(id, _)| id.clone())
            .into_iter()
            .collect()
    } else {
        weights
            .iter()
            .filter(|(id, tensor)| {
                in_layer(id) && tensor.ints.is_none() && format!("{id:?}").contains(substring)
            })
            .map(|(id, _)| id.clone())
            .collect()
    };
    if chosen.is_empty() {
        return Err(format!("--inject {spec:?} matched no weight tensor"));
    }
    let mut names = Vec::new();
    for id in chosen {
        let tensor = weights.get_mut(&id).expect("chosen id is present");
        for value in &mut tensor.data {
            *value *= scale;
        }
        names.push(format!("{id:?}"));
    }
    Ok(names)
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let mut config_path = None;
    let mut tiny = None;
    let mut side = None;
    let mut injection = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--side" => side = args.next(),
            "--inject" => injection = Some(args.next().ok_or("--inject needs a value")?),
            "--tiny" => tiny = Some(args.next().ok_or("--tiny needs a pack family")?),
            other if !other.starts_with("--") && config_path.is_none() => {
                config_path = Some(other.to_string())
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    let side = side.ok_or("--side reference|engine is required")?;
    if !hidden_trace::enabled() {
        return Err("set MEMRA_HYPER_TRACE=<path> for the trace output".into());
    }
    // A config compiles a plan the engine can also load; a pack's tiny plan has no config, so
    // it runs on the reference side only.
    let (config, family, plan) = match (&config_path, &tiny) {
        (Some(path), None) => {
            let text = std::fs::read_to_string(path).map_err(|error| format!("{path}: {error}"))?;
            let hf = HfConfig::try_parse(&text).map_err(|error| format!("{path}: {error:?}"))?;
            let config = ModelConfig::from_hf(&hf);
            let pack = memra_gguf::model_packs::for_config(&config)
                .ok_or("no model pack matches the config")?;
            let plan = pack
                .compile_plan(&config)
                .map_err(|error| format!("plan: {error:?}"))?;
            (Some(config), pack.family, plan)
        }
        (None, Some(family)) => {
            let pack = memra_gguf::model_packs::by_alias(family)
                .ok_or_else(|| format!("unknown pack {family:?}"))?;
            let plan = pack
                .compile_tiny_plan()
                .map_err(|error| format!("{family}: tiny plan: {error:?}"))?;
            (None, pack.family, plan)
        }
        _ => return Err("give exactly one of <config.json> or --tiny <family>".into()),
    };
    let mut fixture =
        deterministic_fixture(&plan).map_err(|error| format!("fixture: {error:?}"))?;
    let injected = match &injection {
        Some(spec) => inject(&mut fixture.weights, spec)?,
        None => Vec::new(),
    };
    let tokens = &fixture.token_ids;
    hidden_trace::emit_tokens(tokens);
    let mut layers = 0;
    let logits = match side.as_str() {
        "reference" => {
            // A config fixture is compared with the engine's stateless `forward_last`, so it runs
            // as one prefill; a tiny fixture runs its own prefill/decode split, as `verify tiny`.
            let output = if tiny.is_some() {
                memra_reference::execute_tokens(
                    &plan,
                    &fixture.weights,
                    tokens,
                    fixture.decode_from,
                )
            } else {
                memra_reference::execute(&plan, &fixture.weights, tokens)
            }
            .map_err(|error| format!("reference execute: {error:?}"))?;
            let rows = output.tokens;
            layers = output.layer_hidden.len();
            for (layer, hidden) in output.layer_hidden.iter().enumerate() {
                if rows == 0 || hidden.len() % rows != 0 {
                    return Err(format!(
                        "layer {layer}: {} values do not split into {rows} token rows",
                        hidden.len()
                    ));
                }
                hidden_trace::emit_last_row(
                    "layer_hidden",
                    layer as i64,
                    rows,
                    hidden.len() / rows,
                    hidden,
                );
            }
            let start = (rows - 1) * output.vocab;
            output.logits[start..start + output.vocab].to_vec()
        }
        "engine" => {
            let config = config
                .as_ref()
                .ok_or("--side engine needs a <config.json>: a tiny plan has no config to load")?;
            if std::env::var_os("NVIDIA_TF32_OVERRIDE").is_none() {
                // SAFETY: single-threaded, before the first CUDA call in this process.
                unsafe { std::env::set_var("NVIDIA_TF32_OVERRIDE", "0") };
            }
            let source = fixture_source(config, &plan, &fixture.weights)?;
            let engine = Engine::new(0).map_err(|error| format!("CUDA engine: {error}"))?;
            let model = HybridModel::load_from_source_without_mtp(&engine, &source)
                .map_err(|error| format!("engine load: {error}"))?;
            model
                .forward_last(&engine, tokens)
                .map_err(|error| format!("engine forward_last: {error}"))?
        }
        other => return Err(format!("--side {other:?}: expected reference or engine")),
    };
    hidden_trace::emit_last_row("logits", -1, 1, logits.len(), &logits);
    println!(
        "side={side} family={family} layers={layers} tokens={} injected={}",
        tokens.len(),
        if injected.is_empty() {
            "none".to_string()
        } else {
            injected.join(";")
        }
    );
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("hidden_trace_engine: {error}");
            ExitCode::FAILURE
        }
    }
}
