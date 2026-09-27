//! Shared semantic ABI resolution for compiler-bound ordinary and composite sources.
use super::*;
use crate::tensor_contract::LayerTensor;

#[derive(Debug, Clone)]
pub(super) enum Access {
    Tensor(TensorId),
    Member(TensorId, usize),
    Derived(TensorId, BoundTensorView),
    Auxiliary(Box<Access>, QuantAuxTensor),
}

pub(super) trait RuntimeSemantics {
    fn plan(&self) -> &ModelPlan;
    fn contains(&self, id: &TensorId) -> bool;
    fn transform(&self, id: &TensorId) -> Option<TensorTransform>;
}

pub(super) struct RuntimeAbi {
    aliases: BTreeMap<String, TensorId>,
}
impl RuntimeAbi {
    pub(super) fn external_draft(
        plan: &ModelPlan,
        options: ContractOptions,
        contract: &TensorContract,
    ) -> Result<Self, String> {
        let mut abi = Self::new(plan, options)?;
        for r in &contract.requirements {
            if r.transform == TensorTransform::StackExperts {
                continue;
            }
            for name in &r.names {
                if let Some(previous) = abi.aliases.insert(name.clone(), r.id.clone())
                    && previous != r.id
                {
                    return Err(format!("ambiguous draft ABI alias {name}"));
                }
            }
        }
        Ok(abi)
    }
    pub(super) fn new(plan: &ModelPlan, options: ContractOptions) -> Result<Self, String> {
        let mut text_plan = plan.clone();
        text_plan.vision = None;
        let mut aliases = BTreeMap::new();
        for (name, id) in
            TensorContract::engine_abi_aliases(&text_plan, options).map_err(|e| e.to_string())?
        {
            if let Some(previous) = aliases.insert(name.clone(), id.clone())
                && previous != id
            {
                return Err(format!(
                    "runtime alias {name} names both {previous:?} and {id:?}"
                ));
            }
        }
        aliases.insert("output.weight".into(), TensorId::OutputProjection);
        // The text loader may use config-derived factors when the artifact has no factor
        // tensor. A present undeclared tensor was already rejected by the complete census.
        aliases
            .entry("rope_freqs.weight".into())
            .or_insert(TensorId::RopeFactors);
        Ok(Self { aliases })
    }
    pub(super) fn access(
        &self,
        view: &impl RuntimeSemantics,
        name: &str,
    ) -> Result<Access, String> {
        // Operation-owned scales (Gemma router input and routed output) are independent
        // semantic operands. Only QuantAux names use the codec-owner resolution below.
        if let Some(id) = self.aliases.get(name)
            && !matches!(id, TensorId::QuantAux { .. })
        {
            return Ok(self.tensor_access(view, id.clone()));
        }
        // Auxiliaries are views of their owner, including folded HF scale planes. Resolve these
        // before the schema's optional GGUF rows so both formats use the same ownership rule.
        for (suffix, kind) in [
            (".pre_quant_scale", QuantAuxTensor::PreQuantScale),
            (".input_scale", QuantAuxTensor::InputScale),
            (".scale", QuantAuxTensor::WeightScale),
        ] {
            if let Some(stem) = name.strip_suffix(suffix) {
                return Ok(Access::Auxiliary(
                    Box::new(self.access(view, &format!("{stem}.weight"))?),
                    kind,
                ));
            }
        }
        if let Some(id) = self.aliases.get(name) {
            return Ok(self.tensor_access(view, id.clone()));
        }
        if let Some((layer, suffix)) = name.strip_prefix("blk.").and_then(|s| s.split_once('.')) {
            let index: u32 = layer
                .parse()
                .map_err(|_| format!("invalid runtime layer {name}"))?;
            if layer != index.to_string() {
                return Err(format!("noncanonical runtime layer {name}"));
            }
            if !view
                .plan()
                .layers
                .iter()
                .chain(view.plan().mtp_blocks.iter().map(|b| &b.layer))
                .any(|l| l.index == index)
            {
                return Err(format!(
                    "runtime tensor {name} refers to a layer outside the plan"
                ));
            }
            for (prefix, bank) in [
                ("ffn_gate_exps.", LayerTensor::MoeExpertGateBank),
                ("ffn_up_exps.", LayerTensor::MoeExpertUpBank),
                ("ffn_down_exps.", LayerTensor::MoeExpertDownBank),
            ] {
                if let Some(member) = suffix
                    .strip_prefix(prefix)
                    .and_then(|s| s.strip_suffix(".weight"))
                    && let Ok(member) = member.parse::<usize>()
                {
                    return Ok(Access::Member(
                        TensorId::Layer {
                            index,
                            tensor: bank,
                        },
                        member,
                    ));
                }
            }
            // Optional executor probes have semantic meaning even when their operation is absent
            // from this plan. Unknown spellings still refuse rather than hiding a loader typo.
            let tensor = match suffix {
                "attn_q_norm.weight" => LayerTensor::QueryNorm,
                "attn_k_norm.weight" => LayerTensor::KeyNorm,
                "attn_k.weight" => LayerTensor::Key,
                "attn_v.weight" => LayerTensor::Value,
                "ffn_gate.weight" => LayerTensor::MlpGate,
                "ffn_up.weight" => LayerTensor::MlpUp,
                "ffn_down.weight" => LayerTensor::MlpDown,
                "ffn_gate_exps.weight" => LayerTensor::MoeExpertGateBank,
                "ffn_up_exps.weight" => LayerTensor::MoeExpertUpBank,
                "ffn_down_exps.weight" => LayerTensor::MoeExpertDownBank,
                "ffn_gate_up_exps.weight" => LayerTensor::MoeExpertGateUpBank,
                "ffn_gate_inp.weight" => LayerTensor::MoeRouter,
                "inp_gate.weight" if view.plan().arch == crate::config::Arch::Gemma4 => {
                    LayerTensor::PerLayerEmbeddingInputGate
                }
                "exp_probs_b.bias" => LayerTensor::MoeRouterBias,
                "ffn_gate_shexp.weight" => LayerTensor::SharedMlpGate,
                "ffn_up_shexp.weight" => LayerTensor::SharedMlpUp,
                "ffn_down_shexp.weight" => LayerTensor::SharedMlpDown,
                "ffn_gate_inp_shexp.weight" => LayerTensor::SharedMlpInputGate,
                _ => {
                    return Err(format!(
                        "runtime tensor {name:?} has no semantic ABI binding"
                    ));
                }
            };
            return Ok(self.tensor_access(view, TensorId::Layer { index, tensor }));
        }
        Err(format!(
            "runtime tensor {name:?} has no semantic ABI binding"
        ))
    }

    fn tensor_access(&self, view: &impl RuntimeSemantics, id: TensorId) -> Access {
        if !view.contains(&id)
            && let TensorId::Layer { index, tensor } = &id
        {
            let derived = match tensor {
                LayerTensor::MlaKeyUp => Some(BoundTensorView::MlaKey),
                LayerTensor::MlaValueUp => Some(BoundTensorView::MlaValue),
                _ => None,
            };
            let fused = TensorId::Layer {
                index: *index,
                tensor: LayerTensor::MlaKvSource,
            };
            if let Some(derived) = derived
                && view.contains(&fused)
            {
                return Access::Derived(fused, derived);
            }
        }
        if view.transform(&id) == Some(TensorTransform::SplitExpertGateUp) {
            Access::Derived(id, BoundTensorView::EncodedBank)
        } else {
            Access::Tensor(id)
        }
    }
}
