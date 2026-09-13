//! Bounded, causal candidate discovery diagnostic; never changes live head rows.
use crate::{Engine, model::GpuTensor};
use std::io::Write;

pub(crate) struct CandidateProbe {
    file: std::io::BufWriter<std::fs::File>,
    present: Vec<bool>,
    recent: std::collections::VecDeque<u32>,
    pub states: usize,
}

impl CandidateProbe {
    pub fn new(path: &str, map: &[u32], vocab: usize, prompt: &[u32]) -> Result<Self, Box<dyn std::error::Error>> {
        let mut present = vec![false; vocab];
        for &id in map { present[id as usize] = true; }
        if present.iter().filter(|&&x| !x).count() < 32 { return Err("bounded probe needs 32 outside rows".into()); }
        let mut probe = Self { file: std::io::BufWriter::new(std::fs::OpenOptions::new().write(true).create_new(true).open(path)?),
            present, recent: std::collections::VecDeque::new(), states: 0 };
        probe.observe(prompt);
        Ok(probe)
    }

    pub fn observe(&mut self, committed: &[u32]) {
        for &id in committed {
            if self.present[id as usize] { continue; }
            if let Some(i) = self.recent.iter().position(|&x| x == id) { self.recent.remove(i); }
            self.recent.push_back(id);
            if self.recent.len() > 256 { self.recent.pop_front(); }
        }
    }

    pub fn prepare(&self, e: &Engine, source: &[u8], stride: usize, width: usize, round: usize)
        -> Result<(GpuTensor, Vec<u32>, usize), Box<dyn std::error::Error>> {
        let mut ids: Vec<u32> = self.recent.iter().rev().take(24).copied().collect();
        let history_rows = ids.len();
        // Deterministic exploration, not an unbiased population estimator.
        let mut rng = 0x9e3779b97f4a7c15_u64 ^ round as u64;
        while ids.len() < 32 {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            let id = (rng % self.present.len() as u64) as u32;
            if !self.present[id as usize] && !ids.contains(&id) { ids.push(id); }
        }
        let mut bytes = Vec::with_capacity(32 * stride);
        for &id in &ids { bytes.extend_from_slice(&source[id as usize * stride..(id as usize + 1) * stride]); }
        let head = GpuTensor::Quant { bytes: e.htod_bytes(&bytes)?, qtype: crate::QT_Q8_0, row_bytes: stride,
            ne: vec![width as u64, 32], scale: 1.0, rp: false,
            #[cfg(memra_cutlass)]
            cutlass: None,
            fp8: None, blk: None, rp4: None, f16: None, a4: None };
        Ok((head, ids, history_rows))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn record(&mut self, round: usize, position: usize, context: usize, ids: &[u32],
        history_rows: usize, scores: &[f32], proposal: u32, target: u32) -> Result<(), Box<dyn std::error::Error>> {
        if ids.len() != 32 || scores.len() != 32 || scores.iter().any(|x| !x.is_finite()) {
            return Err("invalid bounded candidate probe scores".into());
        }
        let rows = ids.iter().zip(scores).map(|(id, score)| format!("[{id},{score}]")).collect::<Vec<_>>().join(",");
        writeln!(self.file, "{{\"schema\":1,\"kind\":\"bounded_candidates\",\"round\":{round},\"position\":{position},\"context\":{context},\"history_rows\":{history_rows},\"proposal\":{proposal},\"target\":{target},\"candidates\":[{rows}]}}")?;
        self.states += 1;
        Ok(())
    }

    pub fn finish(&mut self) -> std::io::Result<()> { self.file.flush() }
}
