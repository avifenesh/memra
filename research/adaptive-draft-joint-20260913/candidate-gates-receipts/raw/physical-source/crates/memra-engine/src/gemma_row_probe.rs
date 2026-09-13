//! Offline one-swap headroom diagnostic. Never a serving or learned-policy path.
use crate::{Engine, model::GpuTensor};
use std::io::Write;

pub(crate) struct RowProbe {
    pub full_head: GpuTensor,
    file: std::io::BufWriter<std::fs::File>,
    pub overhead_ns: u128,
    pub states: usize,
    pub replay: bool,
    source_rows: Vec<u8>,
    row_bytes: usize,
    width: usize,
    shadow: Option<GpuTensor>,
}

impl RowProbe {
    pub fn new(
        e: &Engine,
        path: &str,
        rows: &[u8],
        row_bytes: usize,
        width: usize,
        vocab: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let started = std::time::Instant::now();
        let head = GpuTensor::Quant {
            bytes: e.htod_bytes(rows)?,
            qtype: crate::QT_Q8_0,
            row_bytes,
            ne: vec![width as u64, vocab as u64],
            scale: 1.0,
            rp: false,
            #[cfg(memra_cutlass)]
            cutlass: None,
            fp8: None,
            blk: None,
            rp4: None,
            f16: None,
            a4: None,
        };
        e.stream().synchronize()?;
        Ok(Self {
            full_head: head,
            file: std::io::BufWriter::new(
                std::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(path)?,
            ),
            overhead_ns: started.elapsed().as_nanos(),
            states: 0,
            replay: std::env::var("MEMRA_GEMMA_ROW_REPLAY").as_deref() == Ok("1"),
            source_rows: if std::env::var("MEMRA_GEMMA_ROW_REPLAY").as_deref() == Ok("1") {
                rows.to_vec()
            } else {
                Vec::new()
            },
            row_bytes,
            width,
            shadow: None,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &mut self,
        e: &Engine,
        hidden: Option<&[f32]>,
        round: usize,
        position: usize,
        context: usize,
        core: usize,
        map: &[u32],
        active: &[f32],
        full: &[f32],
        proposal: u32,
        target: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let started = std::time::Instant::now();
        if active.len() != map.len() || active.iter().chain(full).any(|x| !x.is_finite()) {
            return Err("row probe: shape mismatch or non-finite score".into());
        }
        let mut ranked: Vec<usize> = (0..map.len()).collect();
        ranked.sort_unstable_by(|&a, &b| active[b].total_cmp(&active[a]).then(a.cmp(&b)));
        let mut top = Vec::new();
        for i in ranked {
            if top.iter().all(|&(id, _, _)| id != map[i]) {
                top.push((map[i], active[i], i));
                if top.len() == 3 {
                    break;
                }
            }
        }
        if top.len() < 2 {
            return Err("row probe needs two distinct active tokens".into());
        }
        let present: std::collections::HashSet<u32> = map.iter().copied().collect();
        let mut outside: Vec<usize> = (0..full.len())
            .filter(|&i| !present.contains(&(i as u32)))
            .collect();
        let order = |a: &usize, b: &usize| full[*b].total_cmp(&full[*a]).then(a.cmp(b));
        if outside.len() > 16 {
            outside.select_nth_unstable_by(16, order);
            outside.truncate(16);
        }
        outside.sort_unstable_by(order);
        let candidates: Vec<_> = outside.iter().map(|&i| (i as u32, full[i])).collect();
        let max_error = map
            .iter()
            .enumerate()
            .map(|(i, &id)| (active[i] - full[id as usize]).abs())
            .fold(0.0_f32, f32::max);
        let mapped_best = (0..map.len())
            .max_by(|&a, &b| {
                full[map[a] as usize]
                    .total_cmp(&full[map[b] as usize])
                    .then(b.cmp(&a))
            })
            .unwrap();
        let numerical_match = top[0].0 == proposal && map[mapped_best] == proposal;
        let tolerance = 2.0 * max_error + 1e-5;
        let target_present = present.contains(&target);
        let target_score = full[target as usize];
        let mutable_winner =
            top[0].2 >= core && map.iter().filter(|&&id| id == proposal).count() == 1;
        let wrong = proposal != target;
        let add_rescue = wrong && !target_present && target_score > top[0].1 + tolerance;
        // Evicting the unique mutable winner permits a low-score duplicate of a
        // core row as filler. Capacity stays fixed; this is a greedy-only oracle.
        let remove_rescue = wrong
            && mutable_winner
            && top[1].0 == target
            && top[1].1 > top[2.min(top.len() - 1)].1 + tolerance;
        let swap_rescue =
            wrong && !target_present && mutable_winner && target_score > top[1].1 + tolerance;
        let harmful_candidates = candidates
            .iter()
            .filter(|&&(id, score)| {
                proposal == target && id != target && score > top[0].1 + tolerance
            })
            .count();
        if self.replay {
            let hidden = hidden.ok_or("physical replay requires the original hidden vector")?;
            self.replay_state(
                e,
                hidden,
                round,
                position,
                core,
                map,
                active,
                full,
                &candidates,
                proposal,
                target,
                tolerance,
            )?;
        }
        // Only finite numeric fields and booleans; no text needs JSON escaping.
        let top_json = top
            .iter()
            .map(|(id, score, slot)| format!("[{id},{score},{slot}]"))
            .collect::<Vec<_>>()
            .join(",");
        let candidates_json = candidates
            .iter()
            .map(|(id, score)| format!("[{id},{score}]"))
            .collect::<Vec<_>>()
            .join(",");
        let correct = !wrong;
        let head_rows = map.len();
        writeln!(
            self.file,
            "{{\"schema\":1,\"round\":{round},\"position\":{position},\"context\":{context},\"core_rows\":{core},\"head_rows\":{head_rows},\"proposal\":{proposal},\"target\":{target},\"correct\":{correct},\"target_in_head\":{target_present},\"target_full_score\":{target_score},\"active_top\":[{top_json}],\"outside_top16\":[{candidates_json}],\"overlap_max_abs_error\":{max_error},\"overlap_argmax_match\":{numerical_match},\"tolerance\":{tolerance},\"mutable_winner\":{mutable_winner},\"addition_rescue\":{add_rescue},\"removal_rescue\":{remove_rescue},\"swap_rescue\":{swap_rescue},\"harmful_outside_top16\":{harmful_candidates}}}"
        )?;
        self.states += 1;
        self.overhead_ns += started.elapsed().as_nanos();
        Ok(())
    }

    pub fn finish(&mut self) -> std::io::Result<()> {
        self.file.flush()?;
        eprintln!(
            "[row-probe] states={} measured_overhead_ns={}",
            self.states, self.overhead_ns
        );
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn replay_state(
        &mut self,
        e: &Engine,
        hidden: &[f32],
        round: usize,
        position: usize,
        core: usize,
        map: &[u32],
        active: &[f32],
        full: &[f32],
        candidates: &[(u32, f32)],
        proposal: u32,
        target: u32,
        tolerance: f32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if core >= map.len() || hidden.len() != self.width {
            return Err("physical replay requires a mutable slot and exact hidden width".into());
        }
        if self.shadow.is_none() {
            let mut bytes = Vec::with_capacity(map.len() * self.row_bytes);
            for &id in map {
                bytes.extend_from_slice(
                    &self.source_rows
                        [id as usize * self.row_bytes..(id as usize + 1) * self.row_bytes],
                );
            }
            self.shadow = Some(GpuTensor::Quant {
                bytes: e.htod_bytes(&bytes)?,
                qtype: crate::QT_Q8_0,
                row_bytes: self.row_bytes,
                ne: vec![self.width as u64, map.len() as u64],
                scale: 1.0,
                rp: false,
                #[cfg(memra_cutlass)]
                cutlass: None,
                fp8: None,
                blk: None,
                rp4: None,
                f16: None,
                a4: None,
            });
        }
        let h = e.htod(hidden)?;
        let mut choices: Vec<u32> = candidates.iter().map(|x| x.0).collect();
        // Oracle-target insertion is qualification only, never a policy input.
        if !map.contains(&target) && !choices.contains(&target) {
            choices.push(target);
        }
        if !choices.contains(&proposal) {
            choices.push(proposal);
        } // Duplicate-winner tie fixture.
        choices.push(map[core]); // Explicit exact restoration/no-op case.
        for candidate in choices {
            let shadow = self.shadow.as_mut().unwrap();
            if let GpuTensor::Quant { bytes, .. } = shadow {
                e.htod_u8_into(
                    bytes,
                    core * self.row_bytes,
                    &self.source_rows[candidate as usize * self.row_bytes
                        ..(candidate as usize + 1) * self.row_bytes],
                )?;
            }
            let logits = e.matmul(shadow, &h, 1)?;
            let actual_slot = e.dtoh_u32(&e.argmax_token_device(&logits, map.len())?)?[0] as usize;
            let actual = if actual_slot == core {
                candidate
            } else {
                map[actual_slot]
            };
            let mut expected = active.to_vec();
            expected[core] = full[candidate as usize];
            let mut slots: Vec<usize> = (0..map.len()).collect();
            slots.sort_unstable_by(|&a, &b| expected[b].total_cmp(&expected[a]).then(a.cmp(&b)));
            let predicted = if slots[0] == core {
                candidate
            } else {
                map[slots[0]]
            };
            let margin = expected[slots[0]] - expected[slots[1]];
            let measured = e.dtoh(&logits)?;
            if measured.iter().any(|x| !x.is_finite()) {
                return Err("non-finite physical replay score".into());
            }
            let max_error = measured
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            let ambiguous = margin <= tolerance;
            let matched = predicted == actual;
            let delta = i32::from(actual == target) - i32::from(proposal == target);
            writeln!(
                self.file,
                "{{\"schema\":2,\"kind\":\"physical_swap\",\"round\":{round},\"position\":{position},\"victim_slot\":{core},\"candidate\":{candidate},\"predicted\":{predicted},\"actual\":{actual},\"margin\":{margin},\"ambiguous\":{ambiguous},\"matched\":{matched},\"max_abs_error\":{max_error},\"delta\":{delta}}}"
            )?;
            if !ambiguous && !matched {
                return Err("physical swap disagrees with score prediction".into());
            }
            if candidate == map[core] && actual != proposal {
                return Err("physical shadow restoration changed winner".into());
            }
        }
        Ok(())
    }
}
