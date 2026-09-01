//! Measured expert-placement map consumption for the glm5 EP walk (`MEMRA_GLM5_EP_MAP`,
//! lane/glm5-ep-place, 2026-08-31) — the fail-closed `memra-ep-map-v1` reader the shard
//! builders trust.
//!
//! LAW:coactivation-expert-placement (darklanes agent-knowledge/gpu/kernel-craft.md,
//! owner directive 2026-08-31): expert placement is MEASURED, never even-split —
//! (1) measure per-layer expert co-activation on real traffic, (2) partition experts
//! into per-card bundles maximizing same-card top-k co-residency under VRAM balance,
//! (3) pin the always-active set to a KNOWN card the token visits first. In the glm5
//! TP-2 seam that known first-hop card is rank 0 (root): the router runs there, the
//! combine lands there, and the shared expert is already root-owned STRUCTURALLY
//! (`moe_shexp_add`) — which is why the loader requires the map's `entry_rank` to be 0.
//!
//! DIVISION OF LABOR (fleet coordination 2026-08-31): maps are MINTED by the shared
//! fleet tool — `tools/build_expert_placement_map.py` (stdlib-only; strategies
//! coactivation/frequency/even; self-receipting per-layer stats vs the even control;
//! spec + example receipts in `research/ep-placement-map-20260831/REPORT.md`) — from
//! `MEMRA_MOE_TRACE` id lines (+ optional `MEMRA_MOE_WEIGHT_TRACE` hotness). This
//! module is the ENGINE-SIDE reader: one parser, one validation law, consumed by
//! `glm5_tp::prepare_glm5_tp_load` / `arm_moe_ep`.
//!
//! THE FROZEN FORMAT (`memra-ep-map-v1`, JSON — quoted from the tool's REPORT):
//!
//! ```json
//! {"format":"memra-ep-map-v1","strategy":"coactivation|frequency|even","ranks":N,
//!  "entry_rank":0,"expert_count":E,"traces":[...],"params":{...},
//!  "layers":[{"layer":L,"assignment":[rank per expert 0..E-1],"stats":{...}}]}
//! ```
//!
//! The reader consumes the LOAD-BEARING fields only (`format`, `ranks`, `entry_rank`,
//! `expert_count`, `layers[].layer`, `layers[].assignment`); `traces`/`params`/`stats`
//! are the mint's self-receipt and ride along uninspected. Parsing uses the house
//! minimal JSON reader (`memra_gguf::config::JsonObj`) — no serde dependency. Every
//! refusal names the field and the law it broke; the LOADER additionally refuses maps
//! whose layer set does not exactly match the EP-armed layers of the model being
//! loaded (`validate_layer_cover`).
//!
//! CORRECTNESS CONTRACT the engine holds regardless of this file's content: the EP
//! walk is placement-independent by construction — ownership only selects WHICH rank
//! runs the identical per-expert dot program over identical (host-canonically
//! fanned-out) input bytes, and the combine is slot-ordered on root either way. The
//! map changes bytes MOVED, never bytes COMPUTED. `glm5-tp-gate` proves it with a
//! deliberately skewed map against the even split (arm M) and bites the corrupted-map
//! red (R4).

use memra_gguf::config::JsonObj;
use std::collections::BTreeMap;

/// One parsed placement map: per layer, `owners[expert] = rank`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EpMap {
    pub n_experts: usize,
    pub ranks: usize,
    /// The first-hop card of the law: the rank the always-active bundle is pinned to.
    /// The glm5 TP-2 loader requires 0 (root).
    pub entry_rank: usize,
    /// layer index -> owner rank per expert (`n_experts` entries, each `< ranks`).
    pub layers: BTreeMap<usize, Vec<u8>>,
}

fn raw_usize(obj: &JsonObj, key: &str) -> Result<usize, String> {
    let v = obj
        .raw(key)
        .ok_or(format!("ep-map: missing required field {key:?}"))?
        .trim();
    v.parse::<usize>()
        .map_err(|_| format!("ep-map: field {key:?} = {v:?} is not an unsigned integer"))
}

/// Split a raw JSON array substring (`[ {...}, {...} ]`) into its top-level object
/// substrings. String-aware (escapes handled) so paths/shas inside the mint's
/// self-receipt fields can never desynchronize the brace depth.
fn split_objects(raw: &str) -> Result<Vec<&str>, String> {
    let b = raw.as_bytes();
    let mut out = Vec::new();
    let mut depth = 0usize;
    let mut start = None;
    let mut in_str = false;
    let mut escaped = false;
    for (i, &c) in b.iter().enumerate() {
        if in_str {
            if escaped {
                escaped = false;
            } else if c == b'\\' {
                escaped = true;
            } else if c == b'"' {
                in_str = false;
            }
            continue;
        }
        match c {
            b'"' => in_str = true,
            b'{' => {
                if depth == 0 {
                    start = Some(i);
                }
                depth += 1;
            }
            b'}' => {
                depth = depth
                    .checked_sub(1)
                    .ok_or("ep-map: unbalanced braces in layers array")?;
                if depth == 0 {
                    let s = start.take().ok_or("ep-map: object end without start")?;
                    out.push(&raw[s..=i]);
                }
            }
            _ => {}
        }
    }
    if depth != 0 || in_str {
        return Err("ep-map: layers array ends inside an object or string".into());
    }
    Ok(out)
}

impl EpMap {
    /// The even split this map format generalizes: rank = expert / (n/ranks),
    /// contiguous halves — byte-for-byte the pre-map `arm_moe_ep` ownership, and
    /// exactly the tool's `even` strategy (the control arm).
    pub fn even_owners(n_experts: usize, ranks: usize) -> Vec<u8> {
        let per = n_experts / ranks;
        (0..n_experts).map(|ex| (ex / per) as u8).collect()
    }

    /// Fail-closed parse of a `memra-ep-map-v1` JSON document. Every refusal names
    /// the field and the law it broke.
    pub fn parse(text: &str) -> Result<EpMap, String> {
        let obj = JsonObj::parse(text);
        match obj.string("format") {
            Some(f) if f == "memra-ep-map-v1" => {}
            Some(f) => {
                return Err(format!(
                    "ep-map: format {f:?} is not \"memra-ep-map-v1\" (fail-closed: one \
                     frozen format, no silent best-effort read)"
                ));
            }
            None => {
                return Err("ep-map: missing \"format\" field (not a memra-ep-map-v1 \
                            document)"
                    .into());
            }
        }
        let ranks = raw_usize(&obj, "ranks")?;
        let n_experts = raw_usize(&obj, "expert_count")?;
        let entry_rank = raw_usize(&obj, "entry_rank")?;
        if n_experts == 0 || ranks < 2 {
            return Err(format!(
                "ep-map: expert_count={n_experts} ranks={ranks} is not a partitionable \
                 geometry"
            ));
        }
        if entry_rank >= ranks {
            return Err(format!(
                "ep-map: entry_rank {entry_rank} outside the {ranks}-rank map"
            ));
        }
        let layers_raw = obj
            .raw("layers")
            .ok_or("ep-map: missing \"layers\" array")?;
        let mut layers: BTreeMap<usize, Vec<u8>> = BTreeMap::new();
        for layer_obj in split_objects(layers_raw)? {
            let lo = JsonObj::parse(layer_obj);
            let layer = raw_usize(&lo, "layer")?;
            let assignment = lo
                .u32_array("assignment")
                .ok_or(format!("ep-map: layer {layer} is missing \"assignment\""))?;
            if assignment.len() != n_experts {
                return Err(format!(
                    "ep-map: layer {layer} assignment carries {} entries, the map \
                     declares expert_count={n_experts}",
                    assignment.len()
                ));
            }
            if let Some(bad) = assignment.iter().find(|&&r| (r as usize) >= ranks) {
                return Err(format!(
                    "ep-map: layer {layer} assigns rank {bad} >= ranks {ranks}"
                ));
            }
            for r in 0..ranks {
                if !assignment.iter().any(|&a| a as usize == r) {
                    return Err(format!(
                        "ep-map: layer {layer} leaves rank {r} with ZERO experts \
                         (refused: an empty rank slab is an unmeasured degenerate arm)"
                    ));
                }
            }
            let owners: Vec<u8> = assignment.iter().map(|&r| r as u8).collect();
            if layers.insert(layer, owners).is_some() {
                return Err(format!("ep-map: duplicate row for layer {layer}"));
            }
        }
        if layers.is_empty() {
            return Err("ep-map: empty \"layers\" array (fail-closed: an empty map \
                        places nothing)"
                .into());
        }
        Ok(EpMap {
            n_experts,
            ranks,
            entry_rank,
            layers,
        })
    }

    /// Deterministic minimal serialization (gate harnesses and tests emit through
    /// this; carries exactly the load-bearing fields, keys in the tool's sorted
    /// order). The MINT tool is the artifact producer in production — this exists so
    /// the gate's skew/red maps are real `memra-ep-map-v1` documents.
    pub fn render(&self) -> String {
        let mut out = String::from("{\n");
        out.push_str(&format!("  \"entry_rank\": {},\n", self.entry_rank));
        out.push_str(&format!("  \"expert_count\": {},\n", self.n_experts));
        out.push_str("  \"format\": \"memra-ep-map-v1\",\n");
        out.push_str("  \"layers\": [\n");
        let rows: Vec<String> = self
            .layers
            .iter()
            .map(|(layer, owners)| {
                let a: Vec<String> = owners.iter().map(|r| r.to_string()).collect();
                format!(
                    "    {{\"assignment\": [{}], \"layer\": {layer}}}",
                    a.join(", ")
                )
            })
            .collect();
        out.push_str(&rows.join(",\n"));
        out.push_str("\n  ],\n");
        out.push_str(&format!("  \"ranks\": {}\n", self.ranks));
        out.push_str("}\n");
        out
    }

    /// Loader-side cover law: the map's layer set must EXACTLY match the EP-armed
    /// MoE layers of the load. A missing layer would silently fall to the even split
    /// (the trap this refusal exists for); an extra layer is a map minted for a
    /// different arrangement.
    pub fn validate_layer_cover(&self, ep_layers: &[usize]) -> Result<(), String> {
        for il in ep_layers {
            if !self.layers.contains_key(il) {
                return Err(format!(
                    "ep-map: EP-armed MoE layer {il} has no map row (fail-closed: a missing \
                     row must never silently fall back to the even split)"
                ));
            }
        }
        for il in self.layers.keys() {
            if !ep_layers.contains(il) {
                return Err(format!(
                    "ep-map: map row for layer {il} does not match any EP-armed MoE layer of \
                     this load (a map minted for a different arrangement is refused by name)"
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc(body: &str) -> String {
        format!(
            "{{\"format\": \"memra-ep-map-v1\", \"ranks\": 2, \"entry_rank\": 0, \
             \"expert_count\": 4, {body}}}"
        )
    }

    #[test]
    fn parse_render_roundtrip_and_even_split() {
        assert_eq!(EpMap::even_owners(4, 2), vec![0, 0, 1, 1]);
        let m = EpMap {
            n_experts: 4,
            ranks: 2,
            entry_rank: 0,
            layers: [(1usize, EpMap::even_owners(4, 2)), (2, vec![0, 1, 1, 0])]
                .into_iter()
                .collect(),
        };
        let text = m.render();
        let back = EpMap::parse(&text).unwrap();
        assert_eq!(back, m);
        // Deterministic: render is a pure function of the map.
        assert_eq!(text, back.render());
    }

    #[test]
    fn parses_the_shared_tools_emission_shape() {
        // The committed example-map shape (json.dumps indent=2 sort_keys=True), with
        // the self-receipt fields the reader must TOLERATE (stats/traces/params carry
        // strings with braces — the string-aware splitter law).
        let text = r#"{
  "entry_rank": 0,
  "expert_count": 4,
  "format": "memra-ep-map-v1",
  "layers": [
    {
      "assignment": [0, 1, 1, 0],
      "layer": 3,
      "stats": {
        "even_baseline_expected_max_rank_touch": 2.75,
        "expected_max_rank_touch": 2.0,
        "intra_rank_coactivation_fraction": 0.58,
        "peer_touch_fraction": 0.625
      }
    }
  ],
  "params": {"balance_tolerance": 0.05, "decode_only": true, "hotness_signal": "pick-count"},
  "ranks": 2,
  "strategy": "coactivation",
  "traces": [{"lines": 59, "path": "odd{path}.txt", "sha256": "ab12"}]
}
"#;
        let m = EpMap::parse(text).unwrap();
        assert_eq!(m.n_experts, 4);
        assert_eq!(m.ranks, 2);
        assert_eq!(m.entry_rank, 0);
        assert_eq!(m.layers[&3], vec![0, 1, 1, 0]);
    }

    #[test]
    fn parse_refusals_name_the_law() {
        // wrong / missing format
        let e = EpMap::parse("{\"format\": \"other-v9\"}").unwrap_err();
        assert!(e.contains("memra-ep-map-v1"), "{e}");
        assert!(EpMap::parse("{}").unwrap_err().contains("format"));
        // wrong assignment length
        let e =
            EpMap::parse(&doc("\"layers\": [{\"layer\": 1, \"assignment\": [0, 1]}]")).unwrap_err();
        assert!(
            e.contains("2 entries") && e.contains("expert_count=4"),
            "{e}"
        );
        // rank out of range
        let e = EpMap::parse(&doc(
            "\"layers\": [{\"layer\": 1, \"assignment\": [0, 1, 2, 1]}]",
        ))
        .unwrap_err();
        assert!(e.contains(">= ranks"), "{e}");
        // empty rank
        let e = EpMap::parse(&doc(
            "\"layers\": [{\"layer\": 1, \"assignment\": [0, 0, 0, 0]}]",
        ))
        .unwrap_err();
        assert!(e.contains("ZERO experts"), "{e}");
        // duplicate layer
        let e = EpMap::parse(&doc(
            "\"layers\": [{\"layer\": 1, \"assignment\": [0, 0, 1, 1]}, \
             {\"layer\": 1, \"assignment\": [0, 0, 1, 1]}]",
        ))
        .unwrap_err();
        assert!(e.contains("duplicate"), "{e}");
        // no rows
        let e = EpMap::parse(&doc("\"layers\": []")).unwrap_err();
        assert!(e.contains("empty"), "{e}");
        // entry rank out of range
        let e = EpMap::parse(
            "{\"format\": \"memra-ep-map-v1\", \"ranks\": 2, \"entry_rank\": 2, \
             \"expert_count\": 4, \"layers\": [{\"layer\": 1, \"assignment\": [0, 0, 1, 1]}]}",
        )
        .unwrap_err();
        assert!(e.contains("entry_rank"), "{e}");
    }

    #[test]
    fn parses_the_committed_example_map_bytes() {
        // The shared tool's own committed emission (research/ep-placement-map-20260831,
        // cherry-picked into this tree): the reader is anchored on the REAL artifact
        // bytes, not a hand-typed imitation — the wiring-assertions-match-prose law.
        let text = include_str!(
            "../../../research/ep-placement-map-20260831/example-map-coactivation.json"
        );
        let m = EpMap::parse(text).unwrap();
        assert_eq!(m.n_experts, 16);
        assert_eq!(m.ranks, 2);
        assert_eq!(m.entry_rank, 0);
        assert_eq!(m.layers.len(), 2);
        for owners in m.layers.values() {
            assert_eq!(owners.len(), 16);
            assert_eq!(owners.iter().filter(|&&r| r == 0).count(), 8);
        }
    }

    #[test]
    fn layer_cover_is_exact_both_ways() {
        let m = EpMap::parse(&doc(
            "\"layers\": [{\"layer\": 1, \"assignment\": [0, 0, 1, 1]}]",
        ))
        .unwrap();
        assert!(m.validate_layer_cover(&[1]).is_ok());
        assert!(
            m.validate_layer_cover(&[1, 2])
                .unwrap_err()
                .contains("layer 2")
        );
        assert!(
            m.validate_layer_cover(&[2])
                .unwrap_err()
                .contains("layer 2")
        );
    }
}
