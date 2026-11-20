//! The tape-set checkpoint gate: `memra model verify checkpoint --oracle DIR`, where `DIR`
//! holds one `memra-checkpoint-tape-v2` directory per request the pack pins
//! ([`ModelPack::checkpoint_requests`]).
//!
//! A tape is `tape.json` (the request, the engine, the numeric class, per-call indexer records
//! and Engram row digests, the sha256 of each array) beside raw little-endian arrays:
//! `logits.f32` (`[positions, vocab]`), `indexer_sel.i32` (per indexer record, `[queries, k]`
//! recalled groups, sorted) and `indexer_logits.f32` (per record, `[queries, groups]` group
//! logits), `router_sel.i32` and `router_scores.f32` (per MoE layer and call, `[queries, k]`
//! selected experts, sorted, and `[queries, experts]` selection scores). Both tapes are first
//! checked against the compiled plan ([`Geometry`]): the vocabulary, the layer count, and for
//! every call the indexed layers that must run the indexer (the call holds more than `top_k`
//! keys) with their query, group and pick counts, one router record per MoE layer and one Engram
//! record per Engram layer. The gate then compares, in this order:
//!
//! 1. the request: token ids, prefill end, vocabulary, numeric class;
//! 2. every indexer call's recalled groups. The native selection must be its own logits' top-k
//!    under the plan's tie order (an exact tie decided the other way is a wrong program, never a
//!    precision difference), and every finite group logit must sit within the class's selection
//!    band of the oracle's. A query whose groups differ from the oracle's is a precision flip
//!    only when every group that entered or left sits within that band of the oracle's last kept
//!    group; any other difference fails. The router's expert picks and selection scores are held
//!    to the same rules (no own group);
//! 3. every Engram call's table rows (integer, exact);
//! 4. the logits of every position: argmax equal at each position, and every logit within the
//!    class band `max_abs + max_rel * |oracle|`.
//!
//! The pack names each native class's terms ([`TapeClassGate`]): the oracle class it is scored
//! against, its two selection bands, and whether the native runs with the oracle's selections
//! applied (`forced_selections`, the gate passes the runner `--force-selections ORACLE_TAPE`),
//! as the class's band backend ran. A forced native tape must name the oracle tape it was
//! forced by. The runner names its class (`RUNNER --numeric-class`); a runner that does not is
//! scored as the oracle's class.

use memra_gguf::model_packs::{
    CheckpointParityGate, CheckpointRequestPin, ModelPack, TapeClassGate,
};
use memra_gguf::model_plan::{
    AttentionPlan, HyV4AttentionKind, MlpPlan, ModelPlan, ResidualTopology,
};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

type Error = Box<dyn std::error::Error>;

pub const TAPE_FORMAT: &str = "memra-checkpoint-tape-v2";

/// What the plan says a tape of a request must hold.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Geometry {
    pub vocab: usize,
    pub layers: usize,
    /// decoder layers that run the pooled indexer
    pub indexed_layers: Vec<usize>,
    /// decoder layers that carry an Engram memory
    pub engram_layers: Vec<usize>,
    /// MoE layers, and their experts and picks
    pub moe_layers: Vec<usize>,
    pub experts: usize,
    pub experts_per_token: usize,
    /// the indexer's budget (`index_topk`) and group width (`index_kpool`)
    pub top_k: usize,
    pub pool: usize,
}

impl Geometry {
    pub fn from_plan(plan: &ModelPlan) -> Result<Self, Error> {
        let mut g = Geometry {
            vocab: plan.vocab_size as usize,
            layers: plan.layers.len(),
            indexed_layers: Vec::new(),
            engram_layers: Vec::new(),
            moe_layers: Vec::new(),
            experts: 0,
            experts_per_token: 0,
            top_k: 0,
            pool: 0,
        };
        for layer in &plan.layers {
            if let AttentionPlan::HyV4(a) = &layer.attention
                && let HyV4AttentionKind::Indexed(ip) = a.kind
            {
                let (k, p) = (ip.top_k as usize, ip.pool as usize);
                if (g.pool != 0 && (g.top_k, g.pool) != (k, p)) || p == 0 || k < p {
                    return Err("the tape gate reads one indexer geometry a plan".into());
                }
                (g.top_k, g.pool) = (k, p);
                g.indexed_layers.push(layer.index as usize);
            }
            if let ResidualTopology::HyV4(r) = &layer.residual
                && r.engram.is_some()
            {
                g.engram_layers.push(layer.index as usize);
            }
            if let MlpPlan::LatentMoe(m) = &layer.mlp {
                let (e, k) = (
                    m.moe.expert_count as usize,
                    m.moe.experts_per_token as usize,
                );
                if (g.experts != 0 && (g.experts, g.experts_per_token) != (e, k)) || k == 0 || k > e
                {
                    return Err("the tape gate reads one router geometry a plan".into());
                }
                (g.experts, g.experts_per_token) = (e, k);
                g.moe_layers.push(layer.index as usize);
            }
        }
        if g.vocab == 0 || g.layers == 0 {
            return Err("the plan has no vocabulary or no layer".into());
        }
        if g.pool == 0 {
            // no indexer: no call may carry a record
            (g.top_k, g.pool) = (usize::MAX, 1);
        }
        Ok(g)
    }

    /// The calls of a request: (first position, positions). A call holds `start + tokens` keys.
    fn calls(tape: &Tape) -> Vec<(usize, usize)> {
        let mut calls = vec![(0, tape.decode_from)];
        calls.extend((tape.decode_from..tape.positions).map(|t| (t, 1)));
        calls
    }

    /// The tape holds exactly what the plan runs on its request. `forced` admits a tape whose
    /// selections were applied from another tape (a forced class's native run).
    pub fn check(&self, tape: &Tape, side: &str, forced: bool) -> Result<(), Error> {
        let bad =
            |what: String| -> Error { format!("{side} tape {}: {what}", tape.request).into() };
        if tape.vocab != self.vocab {
            return Err(bad(format!(
                "vocabulary {}, the plan {}",
                tape.vocab, self.vocab
            )));
        }
        if tape.layers != self.layers {
            return Err(bad(format!(
                "{} layers, the plan {}",
                tape.layers, self.layers
            )));
        }
        if tape.decode_from == 0 || tape.decode_from > tape.positions {
            return Err(bad("the prefill is not a non-empty prefix".into()));
        }
        let calls = Self::calls(tape);
        let mut want_ix = BTreeMap::new();
        let mut want_en = Vec::new();
        for (ci, &(start, tokens)) in calls.iter().enumerate() {
            let kv = start + tokens;
            if kv > self.top_k {
                let groups = kv / self.pool;
                let k = (self.top_k / self.pool - 1).min(groups) + 1;
                for &il in &self.indexed_layers {
                    want_ix.insert((ci, il), (tokens, k, groups));
                }
            }
            for &il in &self.engram_layers {
                want_en.push((ci, il));
            }
        }
        let got_ix: BTreeMap<_, _> = tape
            .records
            .iter()
            .map(|r| ((r.call, r.layer), (r.queries, r.k, r.groups)))
            .collect();
        if got_ix != want_ix {
            let missing = want_ix.keys().find(|k| !got_ix.contains_key(k));
            let extra = got_ix.keys().find(|k| !want_ix.contains_key(k));
            let shape = want_ix
                .iter()
                .find(|(k, v)| got_ix.get(k).is_some_and(|g| g != *v));
            return Err(bad(format!(
                "the indexer records are not the plan's: {} records, the plan {}; first missing (call, layer) {missing:?}, first extra {extra:?}, first shape mismatch {shape:?}",
                got_ix.len(),
                want_ix.len()
            )));
        }
        let want_rt: BTreeMap<_, _> = calls
            .iter()
            .enumerate()
            .flat_map(|(ci, &(_, tokens))| {
                self.moe_layers
                    .iter()
                    .map(move |&il| ((ci, il), (tokens, self.experts_per_token, self.experts)))
            })
            .collect();
        let got_rt: BTreeMap<_, _> = tape
            .router
            .records
            .iter()
            .map(|r| ((r.call, r.layer), (r.queries, r.k, r.experts)))
            .collect();
        if got_rt != want_rt {
            return Err(bad(format!(
                "the router records are not the plan's: {} records, the plan {}",
                got_rt.len(),
                want_rt.len()
            )));
        }
        for r in &tape.router.records {
            for q in 0..r.queries {
                let sel = &tape.router.sel[r.sel_offset + q * r.k..r.sel_offset + (q + 1) * r.k];
                if !(sel.windows(2).all(|w| w[0] < w[1])
                    && sel.iter().all(|e| *e >= 0 && (*e as usize) < r.experts))
                {
                    return Err(bad(format!(
                        "call {} layer {} query {q}: the experts are not sorted, unique and in range",
                        r.call, r.layer
                    )));
                }
            }
        }
        let got_en: Vec<_> = tape.engram.keys().copied().collect();
        want_en.sort_unstable();
        if got_en != want_en {
            return Err(bad(format!(
                "{} Engram records, the plan {}",
                got_en.len(),
                want_en.len()
            )));
        }
        // the plan's recall: the top-k among all of the call's groups (those not closed before
        // the query score -inf and are taken in tie order when too few are closed) plus the
        // query's own group (which may be the incomplete one, == groups, in a decode call),
        // sorted. Less one occurrence of the own group, the picks are unique groups of the call.
        // The group logits are finite exactly for the groups closed before the query.
        for r in &tape.records {
            let start = calls[r.call].0;
            for q in 0..r.queries {
                let own = ((start + q) / self.pool) as i32;
                let sel = &tape.sel[r.sel_offset + q * r.k..r.sel_offset + (q + 1) * r.k];
                let lg = &tape.group_logits
                    [r.logit_offset + q * r.groups..r.logit_offset + (q + 1) * r.groups];
                let picks = picks_of(sel, own);
                let ok = sel.windows(2).all(|w| w[0] <= w[1])
                    && picks.as_ref().is_some_and(|p| {
                        p.windows(2).all(|w| w[0] < w[1])
                            && p.iter().all(|g| *g >= 0 && (*g as usize) < r.groups)
                    })
                    && lg.iter().enumerate().all(|(g, v)| {
                        if (g as i32) < own {
                            v.is_finite()
                        } else {
                            *v == f32::NEG_INFINITY
                        }
                    });
                if !ok {
                    return Err(bad(format!(
                        "call {} layer {} query {}: the recall is not the call's unique groups plus the own group, or its logits are not finite exactly for the closed groups",
                        r.call,
                        r.layer,
                        start + q
                    )));
                }
            }
        }
        for r in &tape.router.records {
            let sc = &tape.router.scores[r.score_offset..r.score_offset + r.queries * r.experts];
            if !sc.iter().all(|v| v.is_finite()) {
                return Err(bad(format!(
                    "call {} layer {}: a non-finite router score",
                    r.call, r.layer
                )));
            }
        }
        if tape.forced_selections && !forced {
            return Err(bad(
                "its selections were forced (a band measurement, not a run of the program)".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct IndexerRecord {
    pub call: usize,
    pub layer: usize,
    pub queries: usize,
    pub k: usize,
    pub groups: usize,
    pub sel_offset: usize,
    pub logit_offset: usize,
}

/// One MoE layer's router in one call: `[queries, k]` picks from `sel_offset`, `[queries,
/// experts]` selection scores from `score_offset`.
#[derive(Debug, Clone)]
pub struct RouteRecord {
    pub call: usize,
    pub layer: usize,
    pub queries: usize,
    pub k: usize,
    pub experts: usize,
    pub sel_offset: usize,
    pub score_offset: usize,
}

#[derive(Debug, Clone, Default)]
pub struct Routes {
    pub sel: Vec<i32>,
    pub scores: Vec<f32>,
    pub records: Vec<RouteRecord>,
    /// sha256 of the two arrays
    pub sha256: [String; 2],
}

#[derive(Debug, Clone)]
pub struct Tape {
    pub engine: String,
    pub numeric_class: String,
    pub request: String,
    pub token_ids: Vec<u32>,
    pub decode_from: usize,
    pub vocab: usize,
    pub positions: usize,
    pub layers: usize,
    pub logits: Vec<f32>,
    pub sel: Vec<i32>,
    pub group_logits: Vec<f32>,
    pub records: Vec<IndexerRecord>,
    pub engram: BTreeMap<(usize, usize), String>,
    pub router: Routes,
    /// its picks were taken from another tape: a band backend's tape (never an oracle), or a
    /// forced class's native run
    pub forced_selections: bool,
    /// sha256 of the forcing tape's `tape.json`
    pub forcing_tape_sha256: Option<String>,
    /// the wrong programs a runner ran on purpose (a gate's red arm); a tape with any never
    /// passes
    pub red_arms: Vec<String>,
    /// sha256 of `tape.json`
    pub tape_sha256: String,
    /// sha256 of `logits.f32`, `indexer_sel.i32` and `indexer_logits.f32`, each checked against
    /// the one `tape.json` declares
    pub arrays_sha256: [String; 3],
}

fn field<'a>(v: &'a serde_json::Value, key: &str) -> Result<&'a serde_json::Value, Error> {
    v.get(key)
        .ok_or_else(|| format!("tape has no {key}").into())
}

fn uint(v: &serde_json::Value, key: &str) -> Result<usize, Error> {
    let n = field(v, key)?
        .as_u64()
        .ok_or_else(|| format!("tape {key} is not an unsigned integer"))?;
    usize::try_from(n).map_err(|_| format!("tape {key} overflows usize").into())
}

fn text(v: &serde_json::Value, key: &str) -> Result<String, Error> {
    Ok(field(v, key)?
        .as_str()
        .ok_or_else(|| format!("tape {key} is not a string"))?
        .to_string())
}

/// An array file, checked against the sha256 its manifest declares.
fn read_array(
    dir: &Path,
    meta: &serde_json::Value,
    file: &str,
    sha: &str,
) -> Result<(Vec<u8>, String), Error> {
    let name = text(meta, file)?;
    if name.is_empty() || name.contains('/') || name.contains("..") {
        return Err(format!("{}: array file {name:?} is not a plain name", dir.display()).into());
    }
    let path = dir.join(&name);
    let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let got = crate::hex_sha256(&bytes);
    if got != text(meta, sha)? {
        return Err(format!(
            "{}: sha256 {got} is not the one its tape declares",
            path.display()
        )
        .into());
    }
    if !bytes.len().is_multiple_of(4) {
        return Err(format!("{} is not whole 4-byte values", path.display()).into());
    }
    Ok((bytes, got))
}

fn words(bytes: &[u8]) -> impl Iterator<Item = [u8; 4]> + '_ {
    bytes.chunks_exact(4).map(|c| [c[0], c[1], c[2], c[3]])
}

pub fn read_tape(dir: &Path) -> Result<Tape, Error> {
    let raw = std::fs::read(dir.join("tape.json"))
        .map_err(|e| format!("{}: {e}", dir.join("tape.json").display()))?;
    let v: serde_json::Value = serde_json::from_slice(&raw)?;
    if text(&v, "format")? != TAPE_FORMAT {
        return Err(format!("{} is not a {TAPE_FORMAT} tape", dir.display()).into());
    }
    let token_ids = field(&v, "token_ids")?
        .as_array()
        .ok_or("tape token_ids is not a list")?
        .iter()
        .map(|t| {
            t.as_u64()
                .and_then(|t| u32::try_from(t).ok())
                .ok_or("tape token ids are u32")
        })
        .collect::<Result<Vec<_>, _>>()?;
    let vocab = uint(&v, "vocab")?;
    let positions = uint(&v, "positions")?;
    let (logit_bytes, logits_sha) = read_array(dir, field(&v, "logits")?, "file", "sha256")?;
    let logits: Vec<f32> = words(&logit_bytes).map(f32::from_le_bytes).collect();
    drop(logit_bytes);
    if vocab == 0
        || positions != token_ids.len()
        || Some(logits.len()) != vocab.checked_mul(positions)
    {
        return Err(format!(
            "{}: {} logits for {positions} positions of {vocab} ({} token ids)",
            dir.display(),
            logits.len(),
            token_ids.len()
        )
        .into());
    }
    let ix = field(&v, "indexer")?;
    let (sel_bytes, sel_sha) = read_array(dir, ix, "sel_file", "sel_sha256")?;
    let sel: Vec<i32> = words(&sel_bytes).map(i32::from_le_bytes).collect();
    let (lg_bytes, lg_sha) = read_array(dir, ix, "logit_file", "logit_sha256")?;
    let group_logits: Vec<f32> = words(&lg_bytes).map(f32::from_le_bytes).collect();
    let mut records = Vec::new();
    let mut keys = BTreeSet::new();
    for r in field(ix, "records")?
        .as_array()
        .ok_or("tape indexer records is not a list")?
    {
        let rec = IndexerRecord {
            call: uint(r, "call")?,
            layer: uint(r, "layer")?,
            queries: uint(r, "queries")?,
            k: uint(r, "k")?,
            groups: uint(r, "groups")?,
            sel_offset: uint(r, "sel_offset")?,
            logit_offset: uint(r, "logit_offset")?,
        };
        let sel_end = rec
            .queries
            .checked_mul(rec.k)
            .and_then(|n| n.checked_add(rec.sel_offset));
        let lg_end = rec
            .queries
            .checked_mul(rec.groups)
            .and_then(|n| n.checked_add(rec.logit_offset));
        if sel_end.is_none_or(|e| e > sel.len()) || lg_end.is_none_or(|e| e > group_logits.len()) {
            return Err(format!("{}: an indexer record runs past its file", dir.display()).into());
        }
        if !keys.insert((rec.call, rec.layer)) {
            return Err(format!(
                "{}: two indexer records for call {} layer {}",
                dir.display(),
                rec.call,
                rec.layer
            )
            .into());
        }
        records.push(rec);
    }
    let rt = field(&v, "router")?;
    let (rs_bytes, rs_sha) = read_array(dir, rt, "sel_file", "sel_sha256")?;
    let (rc_bytes, rc_sha) = read_array(dir, rt, "score_file", "score_sha256")?;
    let mut router = Routes {
        sel: words(&rs_bytes).map(i32::from_le_bytes).collect(),
        scores: words(&rc_bytes).map(f32::from_le_bytes).collect(),
        records: Vec::new(),
        sha256: [rs_sha, rc_sha],
    };
    drop((rs_bytes, rc_bytes));
    let mut rkeys = BTreeSet::new();
    for r in field(rt, "records")?
        .as_array()
        .ok_or("tape router records is not a list")?
    {
        let rec = RouteRecord {
            call: uint(r, "call")?,
            layer: uint(r, "layer")?,
            queries: uint(r, "queries")?,
            k: uint(r, "k")?,
            experts: uint(r, "experts")?,
            sel_offset: uint(r, "sel_offset")?,
            score_offset: uint(r, "score_offset")?,
        };
        let sel_end = rec
            .queries
            .checked_mul(rec.k)
            .and_then(|n| n.checked_add(rec.sel_offset));
        let sc_end = rec
            .queries
            .checked_mul(rec.experts)
            .and_then(|n| n.checked_add(rec.score_offset));
        if sel_end.is_none_or(|e| e > router.sel.len())
            || sc_end.is_none_or(|e| e > router.scores.len())
        {
            return Err(format!("{}: a router record runs past its file", dir.display()).into());
        }
        if !rkeys.insert((rec.call, rec.layer)) {
            return Err(format!(
                "{}: two router records for call {} layer {}",
                dir.display(),
                rec.call,
                rec.layer
            )
            .into());
        }
        router.records.push(rec);
    }
    let mut engram = BTreeMap::new();
    for r in field(&v, "engram")?
        .as_array()
        .ok_or("tape engram is not a list")?
    {
        if engram
            .insert((uint(r, "call")?, uint(r, "layer")?), text(r, "rows")?)
            .is_some()
        {
            return Err(format!("{}: a duplicate Engram record", dir.display()).into());
        }
    }
    let red_arms = match v.get("red_arms") {
        None => Vec::new(),
        Some(r) => r
            .as_array()
            .ok_or("tape red_arms is not a list")?
            .iter()
            .map(|a| a.as_str().map(str::to_string).ok_or("a red arm is a name"))
            .collect::<Result<_, _>>()?,
    };
    Ok(Tape {
        engine: text(&v, "engine")?,
        numeric_class: text(&v, "numeric_class")?,
        request: text(&v, "request")?,
        token_ids,
        decode_from: uint(&v, "decode_from")?,
        vocab,
        positions,
        layers: uint(&v, "layers")?,
        logits,
        sel,
        group_logits,
        records,
        engram,
        router,
        forced_selections: match v.get("forced_selections") {
            None => false,
            Some(f) => f.as_bool().ok_or("tape forced_selections is not a bool")?,
        },
        forcing_tape_sha256: match v.get("forcing_tape_sha256") {
            None | Some(serde_json::Value::Null) => None,
            Some(f) => Some(
                f.as_str()
                    .ok_or("tape forcing_tape_sha256 is not a string")?
                    .to_string(),
            ),
        },
        red_arms,
        tape_sha256: crate::hex_sha256(&raw),
        arrays_sha256: [logits_sha, sel_sha, lg_sha],
    })
}

/// The request text a tape set's request runs: `{"name", "token_ids", "decode_from"}`.
pub fn request_json(tape: &Tape) -> String {
    serde_json::json!({
        "name": tape.request,
        "token_ids": tape.token_ids,
        "decode_from": tape.decode_from,
    })
    .to_string()
}

/// A digest of a pack's pinned request set (names, token digests, lengths and prefill ends in
/// order): the artifact lock and the receipt carry it, so another request set is another record.
pub fn requests_sha256(pins: &[CheckpointRequestPin]) -> String {
    let text: String = pins
        .iter()
        .map(|p| {
            format!(
                "{}\t{}\t{}\t{}\n",
                p.name, p.tokens_sha256, p.positions, p.decode_from
            )
        })
        .collect();
    crate::hex_sha256(text.as_bytes())
}

/// The oracle tape must be the pinned request.
pub fn check_pin(tape: &Tape, pin: &CheckpointRequestPin) -> Result<(), Error> {
    let tokens = crate::probe_sha256(&tape.token_ids);
    if tape.request != pin.name
        || tokens != pin.tokens_sha256
        || tape.positions != pin.positions
        || tape.decode_from != pin.decode_from
    {
        return Err(format!(
            "the oracle tape {} (tokens sha256 {tokens}, {} positions, prefill {}) is not the pinned request {} ({}, {} positions, prefill {}); no fallback is allowed",
            tape.request,
            tape.positions,
            tape.decode_from,
            pin.name,
            pin.tokens_sha256,
            pin.positions,
            pin.decode_from
        )
        .into());
    }
    Ok(())
}

/// The k-1 best groups of `row` under the CUDA tie order (equal values in ascending index), plus
/// `own`, sorted: what the plan's indexer recalls.
fn plan_pick(row: &[f32], k: usize, own: i32) -> Vec<i32> {
    let mut pick = top_k(row, k.saturating_sub(1).min(row.len()));
    pick.push(own);
    pick.sort_unstable();
    pick
}

/// The `k` best entries of `row` under the CUDA tie order (equal values in ascending index).
fn top_k(row: &[f32], k: usize) -> Vec<i32> {
    let mut order: Vec<usize> = (0..row.len()).collect();
    order.sort_by(|a, b| row[*b].total_cmp(&row[*a]));
    order.into_iter().take(k).map(|g| g as i32).collect()
}

/// An indexer recall less one occurrence of the query's own group: its top-k picks (`None` when
/// the own group is missing).
fn picks_of(sel: &[i32], own: i32) -> Option<Vec<i32>> {
    let at = sel.iter().position(|g| *g == own)?;
    let mut p = sel.to_vec();
    p.remove(at);
    Some(p)
}

/// Whether the oracle's picks `os` and the native picks `ns` (unique, sorted) differ only by a
/// near tie of the oracle's own scores `scores`: every entry that entered or left lies within
/// `band` of the oracle's last kept entry.
fn near_tie(os: &[i32], ns: &[i32], scores: &[f32], band: f64) -> bool {
    let boundary = os
        .iter()
        .map(|g| scores[*g as usize])
        .fold(f32::INFINITY, f32::min) as f64;
    if !boundary.is_finite() {
        return false;
    }
    os.iter()
        .filter(|g| !ns.contains(g))
        .chain(ns.iter().filter(|g| !os.contains(g)))
        .all(|g| {
            let v = scores[*g as usize] as f64;
            v.is_finite() && (v - boundary).abs() <= band
        })
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct Comparison {
    pub indexer_calls: usize,
    pub indexer_queries: usize,
    pub precision_flips: usize,
    pub router_calls: usize,
    pub router_queries: usize,
    pub router_flips: usize,
    /// largest native-vs-oracle distance of a router selection score and of a finite group
    /// logit
    pub router_score_max_abs: f64,
    pub indexer_logit_max_abs: f64,
    pub engram_calls: usize,
    pub positions: usize,
    pub argmax_equal: usize,
    pub max_abs: f32,
    pub max_rel: f32,
    pub worst_position: usize,
    pub worst_token: usize,
}

fn stable_argmax(row: &[f32]) -> usize {
    let mut best = 0;
    for (i, v) in row.iter().enumerate() {
        if v.total_cmp(&row[best]).is_gt() {
            best = i;
        }
    }
    best
}

/// Compare a native tape to its oracle under `gate`, both first checked against the plan's
/// `geometry`; `Err` names the first failure.
pub fn compare(
    oracle: &Tape,
    native: &Tape,
    native_class: &str,
    gate: CheckpointParityGate,
    terms: &TapeClassGate,
    geometry: &Geometry,
) -> Result<Comparison, Error> {
    if native.numeric_class != native_class {
        return Err(format!(
            "native numeric class mismatch: the runner named {native_class}, its tape {}",
            native.numeric_class
        )
        .into());
    }
    if oracle.numeric_class != terms.oracle_class {
        return Err(format!(
            "oracle numeric class mismatch: class {native_class} is scored against {}, the oracle tape is {}",
            terms.oracle_class, oracle.numeric_class
        )
        .into());
    }
    if oracle.forced_selections {
        return Err(format!(
            "oracle tape {}: its selections were forced (a band measurement, not an oracle)",
            oracle.request
        )
        .into());
    }
    if native.forced_selections != terms.forced_selections {
        return Err(format!(
            "request {}: class {native_class} runs with forced selections {}, the native tape {}",
            oracle.request, terms.forced_selections, native.forced_selections
        )
        .into());
    }
    if native.forced_selections
        && native.forcing_tape_sha256.as_deref() != Some(oracle.tape_sha256.as_str())
    {
        return Err(format!(
            "request {}: the native tape was forced by another tape than its oracle",
            oracle.request
        )
        .into());
    }
    if oracle.token_ids != native.token_ids
        || oracle.decode_from != native.decode_from
        || oracle.vocab != native.vocab
    {
        return Err(format!(
            "request {}: the native tape scored another request",
            oracle.request
        )
        .into());
    }
    geometry.check(oracle, "oracle", false)?;
    geometry.check(native, "native", terms.forced_selections)?;
    let calls = Geometry::calls(oracle);
    let mut out = Comparison::default();
    // 2. selections (the geometry check made both record sets the plan's, key for key)
    let native_records: BTreeMap<_, _> = native
        .records
        .iter()
        .map(|r| ((r.call, r.layer), r))
        .collect();
    for r in &oracle.records {
        let n = native_records[&(r.call, r.layer)];
        let start = calls[r.call].0;
        for q in 0..r.queries {
            out.indexer_queries += 1;
            let os = &oracle.sel[r.sel_offset + q * r.k..r.sel_offset + (q + 1) * r.k];
            let ns = &native.sel[n.sel_offset + q * n.k..n.sel_offset + (q + 1) * n.k];
            let ol = &oracle.group_logits
                [r.logit_offset + q * r.groups..r.logit_offset + (q + 1) * r.groups];
            let nl = &native.group_logits
                [n.logit_offset + q * n.groups..n.logit_offset + (q + 1) * n.groups];
            let own = ((start + q) / geometry.pool) as i32;
            if plan_pick(nl, n.k, own) != ns {
                return Err(format!(
                    "request {} call {} layer {} query {}: the native groups are not its own logits' top-k under the plan's tie order",
                    oracle.request,
                    r.call,
                    r.layer,
                    start + q
                )
                .into());
            }
            for (g, (a, b)) in ol.iter().zip(nl).enumerate() {
                if a.is_finite() {
                    let d = (*a as f64 - *b as f64).abs();
                    out.indexer_logit_max_abs = out.indexer_logit_max_abs.max(d);
                    if d.is_nan() || d > terms.indexer_band {
                        return Err(format!(
                            "request {} call {} layer {} query {} group {g}: the native group logit {b} is {d} from the oracle's {a}, past the selection band {}",
                            oracle.request,
                            r.call,
                            r.layer,
                            start + q,
                            terms.indexer_band
                        )
                        .into());
                    }
                }
            }
            if os == ns {
                continue;
            }
            let (Some(op), Some(np)) = (picks_of(os, own), picks_of(ns, own)) else {
                unreachable!("the geometry check found the own group in both")
            };
            if !near_tie(&op, &np, ol, terms.indexer_band) {
                return Err(format!(
                    "request {} call {} layer {} query {}: the native indexer recalled other groups than the oracle (not a near tie)",
                    oracle.request,
                    r.call,
                    r.layer,
                    start + q
                )
                .into());
            }
            out.precision_flips += 1;
        }
        out.indexer_calls += 1;
    }
    // 2b. router picks (the geometry check made both record sets the plan's)
    let native_routes: BTreeMap<_, _> = native
        .router
        .records
        .iter()
        .map(|r| ((r.call, r.layer), r))
        .collect();
    for r in &oracle.router.records {
        let n = native_routes[&(r.call, r.layer)];
        let start = calls[r.call].0;
        for q in 0..r.queries {
            out.router_queries += 1;
            let os = &oracle.router.sel[r.sel_offset + q * r.k..r.sel_offset + (q + 1) * r.k];
            let ns = &native.router.sel[n.sel_offset + q * n.k..n.sel_offset + (q + 1) * n.k];
            let oc = &oracle.router.scores
                [r.score_offset + q * r.experts..r.score_offset + (q + 1) * r.experts];
            let nc = &native.router.scores
                [n.score_offset + q * n.experts..n.score_offset + (q + 1) * n.experts];
            let mut pick = top_k(nc, n.k);
            pick.sort_unstable();
            if pick != ns {
                return Err(format!(
                    "request {} call {} layer {} query {}: the native experts are not its own scores' top-k under the plan's tie order",
                    oracle.request,
                    r.call,
                    r.layer,
                    start + q
                )
                .into());
            }
            for (e, (a, b)) in oc.iter().zip(nc).enumerate() {
                let d = (*a as f64 - *b as f64).abs();
                out.router_score_max_abs = out.router_score_max_abs.max(d);
                if d.is_nan() || d > terms.router_band {
                    return Err(format!(
                        "request {} call {} layer {} query {} expert {e}: the native selection score {b} is {d} from the oracle's {a}, past the selection band {}",
                        oracle.request,
                        r.call,
                        r.layer,
                        start + q,
                        terms.router_band
                    )
                    .into());
                }
            }
            if os == ns {
                continue;
            }
            if !near_tie(os, ns, oc, terms.router_band) {
                return Err(format!(
                    "request {} call {} layer {} query {}: the native router picked other experts than the oracle (not a near tie)",
                    oracle.request,
                    r.call,
                    r.layer,
                    start + q
                )
                .into());
            }
            out.router_flips += 1;
        }
        out.router_calls += 1;
    }
    // 3. Engram rows (the geometry check made both key sets the plan's)
    for (k, rows) in &oracle.engram {
        if native.engram.get(k) != Some(rows) {
            return Err(format!(
                "request {} call {} layer {}: the Engram table rows differ",
                oracle.request, k.0, k.1
            )
            .into());
        }
        out.engram_calls += 1;
    }
    // 4. logits
    let v = oracle.vocab;
    let mut first_violation = None;
    for p in 0..oracle.positions {
        let (o, n) = (
            &oracle.logits[p * v..(p + 1) * v],
            &native.logits[p * v..(p + 1) * v],
        );
        for (t, (&a, &b)) in o.iter().zip(n).enumerate() {
            if !a.is_finite() || !b.is_finite() {
                return Err(format!(
                    "request {} position {p}: a non-finite logit at token {t}",
                    oracle.request
                )
                .into());
            }
            let d = (a - b).abs();
            if d > out.max_abs {
                out.max_abs = d;
                out.worst_position = p;
                out.worst_token = t;
            }
            out.max_rel = out.max_rel.max(d / a.abs().max(1e-6));
            if first_violation.is_none() && d > gate.max_abs + gate.max_rel * a.abs() {
                first_violation = Some((p, t, d, gate.max_abs + gate.max_rel * a.abs()));
            }
        }
        let (ao, an) = (stable_argmax(o), stable_argmax(n));
        if ao == an {
            out.argmax_equal += 1;
        } else if gate.require_argmax {
            return Err(format!(
                "request {} position {p}: argmax mismatch: oracle={ao} native={an}",
                oracle.request
            )
            .into());
        }
        out.positions += 1;
    }
    if let Some((p, t, d, allowed)) = first_violation {
        return Err(format!(
            "request {} position {p} token {t}: abs={d} exceeds max_abs+max_rel*|oracle|={allowed}; observed max_abs={} at position {} token {}",
            oracle.request, out.max_abs, out.worst_position, out.worst_token
        )
        .into());
    }
    Ok(out)
}

/// The tape-set gate over every pinned request: the native runner scores each into a fresh
/// directory, and each is compared to its oracle. Returns the receipt rows.
pub fn verify_tape_set(
    pack: &ModelPack,
    geometry: &Geometry,
    source: &str,
    oracle_dir: &Path,
    runner: &Path,
    out_dir: &Path,
) -> Result<String, Error> {
    let mut rows = String::new();
    let mut red = None;
    let runner_class = runner_numeric_class(runner);
    for pin in pack.checkpoint_requests {
        let oracle_tape = oracle_dir.join(pin.name);
        let oracle = read_tape(&oracle_tape)?;
        check_pin(&oracle, pin)?;
        if !oracle.red_arms.is_empty() {
            return Err(
                format!("the oracle tape {} was written by a red-arm run", pin.name).into(),
            );
        }
        // the oracle is checked against the plan before any of its picks reach a runner
        geometry.check(&oracle, "oracle", false)?;
        let native_class = runner_class
            .clone()
            .unwrap_or_else(|| oracle.numeric_class.clone());
        let (gate, terms) = pack.checkpoint_tape_class(&native_class).ok_or_else(|| {
            format!(
                "model pack {} has no tape-set checkpoint terms for numeric class {native_class}; no fallback is allowed",
                pack.family
            )
        })?;
        std::fs::create_dir_all(out_dir.join("requests"))?;
        let req_path = out_dir.join("requests").join(format!("{}.json", pin.name));
        crate::write_atomic(&req_path, request_json(&oracle).as_bytes())?;
        // a fresh tape directory: a runner that writes nothing cannot pass on an earlier tape
        let native_dir = out_dir.join("native").join(pin.name);
        if native_dir.exists() {
            std::fs::remove_dir_all(&native_dir)?;
        }
        let mut command = std::process::Command::new(runner);
        command
            .arg(source)
            .arg("--request")
            .arg(&req_path)
            .arg("--tape")
            .arg(&native_dir)
            .env("MEMRA_FULL_PREC", "1");
        if terms.forced_selections {
            command.arg("--force-selections").arg(&oracle_tape);
        }
        let result = command.output()?;
        if !result.status.success() {
            return Err(format!(
                "native checkpoint runner failed on {} ({}): stderr={}",
                pin.name,
                result.status,
                String::from_utf8_lossy(&result.stderr)
            )
            .into());
        }
        let native = read_tape(&native_dir)?;
        let c = compare(&oracle, &native, &native_class, gate, &terms, geometry)
            .map_err(|e| format!("{e}"))?;
        if !native.red_arms.is_empty() {
            // a red-arm tape never passes; the next requests still run, so an arm that only a
            // later request can expose is refused by its comparison, not by the flag alone
            red.get_or_insert((pin.name, native.red_arms.clone()));
            continue;
        }
        rows.push_str(&format!(
            "request\t{}\tclass={}\tforced_selections={}\toracle={}\tnative={}\toracle_tape_sha256={}\tnative_tape_sha256={}\toracle_logits_sha256={}\tnative_logits_sha256={}\toracle_selections_sha256={}\tnative_selections_sha256={}\tpositions={}\targmax_equal={}\tindexer_calls={}\tindexer_queries={}\tprecision_flips={}\trouter_calls={}\trouter_queries={}\trouter_flips={}\trouter_score_max_abs={}\tindexer_logit_max_abs={}\toracle_routes_sha256={}\tnative_routes_sha256={}\tengram_calls={}\tmax_abs={}\tmax_rel={}\tworst={}:{}\n",
            pin.name,
            native_class,
            terms.forced_selections,
            oracle.engine,
            native.engine,
            oracle.tape_sha256,
            native.tape_sha256,
            oracle.arrays_sha256[0],
            native.arrays_sha256[0],
            oracle.arrays_sha256[1],
            native.arrays_sha256[1],
            c.positions,
            c.argmax_equal,
            c.indexer_calls,
            c.indexer_queries,
            c.precision_flips,
            c.router_calls,
            c.router_queries,
            c.router_flips,
            c.router_score_max_abs,
            c.indexer_logit_max_abs,
            oracle.router.sha256[0],
            native.router.sha256[0],
            c.engram_calls,
            c.max_abs,
            c.max_rel,
            c.worst_position,
            c.worst_token,
        ));
    }
    if let Some((name, arms)) = red {
        return Err(format!(
            "request {name}: the native tape ran the red arms {arms:?}; a red-arm tape never passes"
        )
        .into());
    }
    Ok(rows)
}

/// The class a runner names (`RUNNER --numeric-class` exits 0 and prints it); `None` for a runner
/// that names none, which is then scored as its oracle's class.
pub fn runner_numeric_class(runner: &Path) -> Option<String> {
    let out = std::process::Command::new(runner)
        .arg("--numeric-class")
        .env_remove("MEMRA_FULL_PREC")
        .output()
        .ok()?;
    let text = String::from_utf8(out.stdout).ok()?;
    let class = text.trim();
    (out.status.success()
        && !class.is_empty()
        && class
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-'))
    .then(|| class.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    const GATE: CheckpointParityGate = CheckpointParityGate {
        max_abs: 1e-3,
        max_rel: 0.0,
        require_argmax: true,
    };
    const CLASS: &str = "hy-v4-flash-source-program";
    const TERMS: TapeClassGate = TapeClassGate {
        oracle_class: CLASS,
        router_band: 0.009,
        indexer_band: 0.024,
        forced_selections: false,
    };

    fn cmp(o: &Tape, n: &Tape, g: &Geometry) -> Result<Comparison, Error> {
        compare(o, n, CLASS, GATE, &TERMS, g)
    }

    /// Two layers: layer 0 indexed (budget 11, groups of 2), layer 1 with an Engram memory.
    fn geometry() -> Geometry {
        Geometry {
            vocab: 4,
            layers: 2,
            indexed_layers: vec![0],
            engram_layers: vec![1],
            moe_layers: vec![1],
            experts: 4,
            experts_per_token: 2,
            top_k: 11,
            pool: 2,
        }
    }

    /// Layer 1's router in both calls: experts 1 and 3 of 4 at every position.
    fn routes() -> Routes {
        let row = [0.1f32, 0.7, 0.2, 0.6];
        let mut r = Routes::default();
        for (call, queries) in [(0usize, 11usize), (1, 1)] {
            r.records.push(RouteRecord {
                call,
                layer: 1,
                queries,
                k: 2,
                experts: 4,
                sel_offset: r.sel.len(),
                score_offset: r.scores.len(),
            });
            for _ in 0..queries {
                r.sel.extend([1, 3]);
                r.scores.extend(row);
            }
        }
        r
    }

    /// A 12-token request scored as a prefill of 11 (11 keys: dense) and one decode call (12
    /// keys: the indexer runs). Query 11 sits in group 5 and keeps the 4 best of the 5 closed
    /// groups plus its own.
    fn tape() -> Tape {
        let vocab = 4;
        let positions = 12;
        let logits = (0..positions * vocab)
            .map(|i| {
                if i % vocab == 1 {
                    2.0
                } else {
                    (i % 7) as f32 * 0.1
                }
            })
            .collect();
        Tape {
            engine: "test".into(),
            numeric_class: "hy-v4-flash-source-program".into(),
            request: "r".into(),
            token_ids: (0..12).collect(),
            decode_from: 11,
            vocab,
            positions,
            layers: 2,
            logits,
            sel: vec![0, 1, 2, 4, 5],
            group_logits: vec![0.3, 0.2, 0.9, 0.1, 0.5, f32::NEG_INFINITY],
            records: vec![IndexerRecord {
                call: 1,
                layer: 0,
                queries: 1,
                k: 5,
                groups: 6,
                sel_offset: 0,
                logit_offset: 0,
            }],
            engram: (0..2).map(|c| ((c, 1), format!("{c:016x}"))).collect(),
            router: routes(),
            forced_selections: false,
            forcing_tape_sha256: None,
            red_arms: Vec::new(),
            tape_sha256: String::new(),
            arrays_sha256: Default::default(),
        }
    }

    #[test]
    fn equal_tapes_pass_and_count_every_check() {
        let c = cmp(&tape(), &tape(), &geometry()).unwrap();
        assert_eq!((c.positions, c.argmax_equal), (12, 12));
        assert_eq!(
            (c.indexer_calls, c.indexer_queries, c.precision_flips),
            (1, 1, 0)
        );
        assert_eq!(c.engram_calls, 2);
        assert_eq!(
            (c.router_calls, c.router_queries, c.router_flips),
            (2, 12, 0)
        );
        assert_eq!(c.max_abs, 0.0);
    }

    #[test]
    fn router_picks_follow_the_tie_order_and_flip_only_at_near_ties() {
        // the last call's query: experts 3 and 2 tie natively; the plan keeps 1 and the lower, 2
        let last = 11 * 4;
        let mut n = tape();
        n.router.scores[last + 3] = 0.2;
        let e = cmp(&tape(), &n, &geometry()).unwrap_err().to_string();
        assert!(e.contains("tie order"), "{e}");
        // a near tie of the oracle's scores at the boundary: attributed
        let mut o = tape();
        o.router.scores[last + 2] = 0.599_999;
        let mut n = o.clone();
        n.router.scores[last + 2] = 0.600_001;
        n.router.scores[last + 3] = 0.599_999;
        let lo = n.router.sel.len() - 2;
        n.router.sel[lo..].copy_from_slice(&[1, 2]);
        let c = cmp(&o, &n, &geometry()).unwrap();
        assert_eq!(c.router_flips, 1);
        // every score within the band, but expert 2 entered from 0.0135 under the boundary: a
        // wrong program
        let mut o = tape();
        o.router.scores[last + 2] = 0.5865;
        let mut n = o.clone();
        n.router.scores[last + 2] = 0.5950;
        n.router.scores[last + 3] = 0.5915;
        let lo = n.router.sel.len() - 2;
        n.router.sel[lo..].copy_from_slice(&[1, 2]);
        let e = cmp(&o, &n, &geometry()).unwrap_err().to_string();
        assert!(e.contains("not a near tie"), "{e}");
        // a score past the band fails even where the picks agree
        let mut n = tape();
        n.router.scores[last] = 0.12;
        let e = cmp(&tape(), &n, &geometry()).unwrap_err().to_string();
        assert!(e.contains("selection band"), "{e}");
        // a call's router record missing on both sides
        let mut o = tape();
        o.router.records.pop();
        let e = cmp(&o, &o.clone(), &geometry()).unwrap_err().to_string();
        assert!(e.contains("router records"), "{e}");
    }

    #[test]
    fn a_logit_past_the_band_or_a_moved_argmax_fails() {
        let mut n = tape();
        n.logits[5] += 2e-3;
        let e = cmp(&tape(), &n, &geometry()).unwrap_err();
        assert!(e.to_string().contains("exceeds"), "{e}");
        let mut n = tape();
        n.logits[4 * 3] = 9.0;
        let e = cmp(&tape(), &n, &geometry()).unwrap_err();
        assert!(e.to_string().contains("argmax"), "{e}");
    }

    #[test]
    fn engram_rows_are_exact() {
        let mut n = tape();
        n.engram.insert((1, 1), "ffff".into());
        let e = cmp(&tape(), &n, &geometry()).unwrap_err();
        assert!(e.to_string().contains("Engram"), "{e}");
    }

    #[test]
    fn an_exact_tie_decided_the_other_way_fails() {
        // groups 1 and 3 tie at 0.0: the plan keeps the lower index
        let mut o = tape();
        o.group_logits = vec![0.3, 0.0, 0.9, 0.0, 0.5, f32::NEG_INFINITY];
        assert!(cmp(&o, &o.clone(), &geometry()).is_ok());
        let mut n = o.clone();
        n.sel = vec![0, 2, 3, 4, 5];
        let e = cmp(&o, &n, &geometry()).unwrap_err().to_string();
        assert!(e.contains("tie order"), "{e}");
    }

    #[test]
    fn a_near_tie_flip_is_attributed_and_a_hard_flip_fails() {
        // the oracle keeps group 1 (0.2) over group 3 (0.199999); natively 3 scores 0.200001
        let mut o = tape();
        o.group_logits[3] = 0.199_999;
        let mut n = o.clone();
        n.group_logits[1] = 0.199_999;
        n.group_logits[3] = 0.200_001;
        n.sel = vec![0, 2, 3, 4, 5];
        let c = cmp(&o, &n, &geometry()).unwrap();
        assert_eq!(c.precision_flips, 1);
        // every logit within the band, but group 3 entered from 0.03 under the boundary: a wrong
        // program
        let mut o = tape();
        o.group_logits[3] = 0.17;
        let mut n = o.clone();
        n.group_logits[1] = 0.18;
        n.group_logits[3] = 0.19;
        n.sel = vec![0, 2, 3, 4, 5];
        let e = cmp(&o, &n, &geometry()).unwrap_err().to_string();
        assert!(e.contains("not a near tie"), "{e}");
        // a group logit past the band fails even where the recall agrees
        let mut n = tape();
        n.group_logits[0] = 0.35;
        let e = cmp(&tape(), &n, &geometry()).unwrap_err().to_string();
        assert!(e.contains("selection band"), "{e}");
    }

    #[test]
    fn both_tapes_must_hold_what_the_plan_runs() {
        // both omit the indexer call: the oracle alone cannot vouch for it
        let mut o = tape();
        o.records.clear();
        let e = cmp(&o, &o.clone(), &geometry()).unwrap_err().to_string();
        assert!(e.contains("not the plan's"), "{e}");
        // a diagnostic run of fewer layers
        let mut o = tape();
        o.layers = 1;
        assert!(cmp(&o, &o.clone(), &geometry()).is_err());
        // an Engram record missing on both sides
        let mut o = tape();
        o.engram.remove(&(0, 1));
        assert!(cmp(&o, &o.clone(), &geometry()).is_err());
        // a recalled group past the call's groups, or a duplicate that is not the own group
        let mut o = tape();
        o.sel = vec![0, 1, 2, 5, 6];
        assert!(cmp(&o, &o.clone(), &geometry()).is_err());
        let mut o = tape();
        o.sel = vec![0, 1, 1, 2, 5];
        assert!(cmp(&o, &o.clone(), &geometry()).is_err());
        // the own group repeated past the one the recall appends
        let mut o = tape();
        o.sel = vec![0, 1, 5, 5, 5];
        assert!(cmp(&o, &o.clone(), &geometry()).is_err());
        // a finite logit for a group not closed before the query, a non-finite router score
        let mut o = tape();
        o.group_logits[5] = 0.0;
        assert!(cmp(&o, &o.clone(), &geometry()).is_err());
        let mut o = tape();
        o.router.scores[0] = f32::NAN;
        assert!(cmp(&o, &o.clone(), &geometry()).is_err());
        // a band backend's forced tape
        let mut n = tape();
        n.forced_selections = true;
        let e = cmp(&tape(), &n, &geometry()).unwrap_err().to_string();
        assert!(e.contains("forced"), "{e}");
        // another vocabulary
        let mut g = geometry();
        g.vocab = 8;
        assert!(cmp(&tape(), &tape(), &g).is_err());
    }

    #[test]
    fn another_request_or_class_fails() {
        let mut n = tape();
        n.decode_from = 10;
        assert!(cmp(&tape(), &n, &geometry()).is_err());
        let mut n = tape();
        n.numeric_class = "other".into();
        assert!(cmp(&tape(), &n, &geometry()).is_err());
    }

    #[test]
    fn a_forced_class_runs_on_its_own_oracle_and_others_stay_unforced() {
        let forced = TapeClassGate {
            forced_selections: true,
            ..TERMS
        };
        let mut o = tape();
        o.tape_sha256 = "aa".into();
        let mut n = tape();
        n.forced_selections = true;
        n.forcing_tape_sha256 = Some("aa".into());
        assert!(compare(&o, &n, CLASS, GATE, &forced, &geometry()).is_ok());
        // forced by another tape
        n.forcing_tape_sha256 = Some("bb".into());
        let e = compare(&o, &n, CLASS, GATE, &forced, &geometry()).unwrap_err();
        assert!(e.to_string().contains("another tape"), "{e}");
        // a forced class run free, a free class run forced
        let e = compare(&o, &tape(), CLASS, GATE, &forced, &geometry()).unwrap_err();
        assert!(e.to_string().contains("forced selections"), "{e}");
        n.forcing_tape_sha256 = Some("aa".into());
        let e = compare(&o, &n, CLASS, GATE, &TERMS, &geometry()).unwrap_err();
        assert!(e.to_string().contains("forced"), "{e}");
        // an oracle that was itself forced
        let mut fo = o.clone();
        fo.forced_selections = true;
        let e = compare(&fo, &n, CLASS, GATE, &forced, &geometry()).unwrap_err();
        assert!(e.to_string().contains("not an oracle"), "{e}");
    }

    #[test]
    fn a_native_class_is_scored_against_its_oracle_class() {
        let terms = TapeClassGate {
            oracle_class: "hy-v4-flash-source-program",
            ..TERMS
        };
        let mut n = tape();
        n.numeric_class = "hy-v4-flash-bf16-activations".into();
        assert!(
            compare(
                &tape(),
                &n,
                "hy-v4-flash-bf16-activations",
                GATE,
                &terms,
                &geometry()
            )
            .is_ok()
        );
        // the runner named another class than its tape
        let e = compare(&tape(), &n, CLASS, GATE, &terms, &geometry()).unwrap_err();
        assert!(e.to_string().contains("native numeric class"), "{e}");
        // an oracle of another class
        let mut o = tape();
        o.numeric_class = "other".into();
        let e = compare(
            &o,
            &n,
            "hy-v4-flash-bf16-activations",
            GATE,
            &terms,
            &geometry(),
        )
        .unwrap_err();
        assert!(e.to_string().contains("oracle numeric class"), "{e}");
    }

    #[test]
    fn the_request_digest_binds_every_pin() {
        let pins = [CheckpointRequestPin {
            name: "a",
            tokens_sha256: "00",
            positions: 3,
            decode_from: 2,
        }];
        let mut other = pins;
        other[0].decode_from = 1;
        assert_ne!(requests_sha256(&pins), requests_sha256(&other));
    }
}
