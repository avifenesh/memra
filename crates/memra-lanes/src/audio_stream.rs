//! Streaming-audio session scheduling — the drive shape IS the capacity.
//!
//! This module exists because of one measurement (darklanes `research/speech-k1-20260910`,
//! 2026-09-10, one A100 80GB PCIe, the pinned two-tier streaming RNNT student):
//!
//! - the identical 80 ms streaming step costs **31.522 ms at batch 1** and **35.890 ms at
//!   batch 32** (+13.9%, reproduced at +14.7% in a second process);
//! - only **12.06 ms of a 26.18 ms step is kernel execution**, across **2,904 device kernel
//!   launches per 80 ms of audio**;
//! - on the same card, in the same session, with the same audio and the same 64-frame
//!   bounded queue, a one-worker batch-1 driver **sheds 2 of 4 streams at c4** while one
//!   fused batch-N step consuming a frame from every lane **holds 64 of 64** at max queue
//!   depth 3.
//!
//! So the c4 collapse was never SM saturation. It was the drive shape, and a bounded queue
//! that *lost* streams instead of refusing them. Both of those are decisions, so both of
//! them live here, in one place, named, with the losing arm kept as an explicit red arm
//! rather than deleted — a capacity claim whose failing shape cannot be re-run is not a
//! claim (`agent-knowledge/gpu/gate-craft.md`).
//!
//! Two invariants this module is built to hold:
//!
//! 1. **Shed is typed, at admission.** A capacity limit that is reached is product
//!    behaviour; a capacity limit that silently eats a stream is a defect. Every refusal
//!    here carries a code, the limit it hit and the value observed. Nothing drops.
//! 2. **One step serves every ready lane.** The fused arm consumes one frame from every
//!    lane that has one, because the step's cost is dominated by a per-step launch tax that
//!    batch barely moves (+13.9% from batch 1 to batch 32): paying it once per lane is
//!    paying it 32 times for nothing.
//!
//! This module owns scheduling and admission only. It holds no model, computes no audio and
//! makes no speed claim: `step_ms` is supplied by the caller (the engine's measured step in
//! production, a measured-anchored model in the replay tests).

use std::collections::{HashMap, VecDeque};

use crate::Lane;

/// One streaming step of audio for the pinned RNNT student, in milliseconds. The whole
/// real-time budget is derived from this: a stream that produces a frame every `FRAME_MS`
/// is served in real time only if the scheduler consumes one from it every `FRAME_MS`.
pub const FRAME_MS: u64 = 80;

/// Why a stream was refused. Every arm is a *refusal*, never a drop: the caller gets a
/// status and a reason it can act on, which is exactly what the c4 shape did not provide.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ShedCode {
    /// The resident-session cap is full. Retryable elsewhere, not here.
    SessionsExceeded,
    /// This session's bounded frame queue is full: the scheduler is not keeping up with
    /// real time for this lane. THE c4 CODE — the incident's shape, now typed.
    QueueOverflow,
    /// The session id is not open (closed, expired, or never opened).
    UnknownSession,
    /// The endpoint has no speech engine bound, so nothing can be transcribed. Fail closed:
    /// a serving surface that answers fluently with no model behind it is worse than a 503.
    EngineUnbound,
}

impl ShedCode {
    pub fn as_str(&self) -> &'static str {
        match self {
            ShedCode::SessionsExceeded => "sessions_exceeded",
            ShedCode::QueueOverflow => "queue_overflow",
            ShedCode::UnknownSession => "unknown_session",
            ShedCode::EngineUnbound => "engine_unbound",
        }
    }
    /// HTTP status a surface answers with. 503 for capacity (the box is full, come back),
    /// 429 for a lane that is over its own real-time budget (slow down or reconnect),
    /// 404 for an id that does not exist.
    pub fn http_status(&self) -> u16 {
        match self {
            ShedCode::SessionsExceeded => 503,
            ShedCode::QueueOverflow => 429,
            ShedCode::UnknownSession => 404,
            ShedCode::EngineUnbound => 503,
        }
    }
    pub fn retryable(&self) -> bool {
        matches!(
            self,
            ShedCode::SessionsExceeded | ShedCode::QueueOverflow | ShedCode::EngineUnbound
        )
    }
}

/// A typed refusal. `limit` and `observed` are carried so an operator can tell "the box is
/// full" from "this lane fell behind" without reading the server's logs.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct AudioShed {
    pub code: ShedCode,
    pub limit: usize,
    pub observed: usize,
    pub session: Option<String>,
}

impl AudioShed {
    fn new(code: ShedCode, limit: usize, observed: usize, session: Option<&str>) -> Self {
        Self {
            code,
            limit,
            observed,
            session: session.map(|s| s.to_string()),
        }
    }
    pub fn message(&self) -> String {
        match self.code {
            ShedCode::SessionsExceeded => format!(
                "audio stream capacity reached: {} of {} resident sessions",
                self.observed, self.limit
            ),
            ShedCode::QueueOverflow => format!(
                "audio session fell behind real time: queue depth {} of {} frames",
                self.observed, self.limit
            ),
            ShedCode::UnknownSession => "audio session is not open".to_string(),
            ShedCode::EngineUnbound => "no speech engine is bound to this endpoint".to_string(),
        }
    }
}

/// How concurrent streams reach the device.
///
/// `Fused` is the default and the only shape with a receipt behind it. `PerLaneWorker` is
/// the pinned c4 shape (`twotier/replay_stream.py:22`, `ThreadPoolExecutor(max_workers=1)`
/// with no cross-stream batching) and exists for exactly one reason: G8's red arm has to be
/// able to re-run the failure it protects against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DriveShape {
    /// One batched step per tick consumes one frame from EVERY ready lane.
    Fused,
    /// One worker, batch 1, lanes served in round-robin. The measured collapse.
    PerLaneWorker,
}

impl DriveShape {
    pub fn as_str(&self) -> &'static str {
        match self {
            DriveShape::Fused => "fused",
            DriveShape::PerLaneWorker => "per_lane_worker",
        }
    }
    pub fn parse(v: &str) -> Option<DriveShape> {
        match v {
            "fused" => Some(DriveShape::Fused),
            "per_lane_worker" => Some(DriveShape::PerLaneWorker),
            _ => None,
        }
    }
    /// `MEMRA_AUDIO_DRIVE`. Default `fused`. An unparseable value is the default, not a
    /// panic, and not the red arm: a typo must never select the shape that sheds.
    pub fn from_env() -> DriveShape {
        std::env::var("MEMRA_AUDIO_DRIVE")
            .ok()
            .as_deref()
            .and_then(DriveShape::parse)
            .unwrap_or(DriveShape::Fused)
    }
}

/// Admission and drive policy for the audio surface.
#[derive(Debug, Clone, Copy)]
pub struct AudioPolicy {
    /// Resident concurrent sessions. 64 is the count the fused arm was measured holding,
    /// not an aspiration: the cap is a measured number until a battery moves it.
    pub max_sessions: usize,
    /// Bounded per-session frame queue. 64 frames is the pinned c4 harness cap, kept so the
    /// red arm reproduces the incident's geometry and not a friendlier one.
    pub queue_frames: usize,
    pub drive: DriveShape,
    /// The SLO inside `streams@SLO`: p95 end-of-speech to final, in ms.
    pub final_slo_ms: u64,
}

impl Default for AudioPolicy {
    fn default() -> Self {
        Self {
            max_sessions: 64,
            queue_frames: 64,
            drive: DriveShape::Fused,
            final_slo_ms: 800,
        }
    }
}

impl AudioPolicy {
    pub fn from_env() -> Self {
        let u = |k: &str, d: usize| {
            std::env::var(k)
                .ok()
                .and_then(|v| v.parse().ok())
                .filter(|&n: &usize| n > 0)
                .unwrap_or(d)
        };
        let d = Self::default();
        Self {
            max_sessions: u("MEMRA_AUDIO_MAX_STREAMS", d.max_sessions),
            queue_frames: u("MEMRA_AUDIO_QUEUE_FRAMES", d.queue_frames),
            drive: DriveShape::from_env(),
            final_slo_ms: u("MEMRA_AUDIO_FINAL_SLO_MS", d.final_slo_ms as usize) as u64,
        }
    }
}

/// One live streaming session. The queue holds frame arrival stamps only: this module
/// schedules frames, it does not own audio.
#[derive(Debug)]
pub struct AudioSession {
    pub id: String,
    pub lane: Lane,
    queue: VecDeque<u64>,
    pub opened_ms: u64,
    pub frames_in: u64,
    pub frames_stepped: u64,
    pub max_depth_seen: usize,
    /// Arrival stamp of the last frame this session has NOT yet been stepped past, i.e. the
    /// head of the backlog. The finalization clock starts here, never at submission of a
    /// decode: "timing starts from capture of the last owned speech sample".
    pub last_final_ms: Option<u64>,
}

impl AudioSession {
    pub fn depth(&self) -> usize {
        self.queue.len()
    }
}

/// Counters an operator reads. Shed is counted PER CODE: "we refused 900 things" and "we
/// refused 900 things because one lane fell behind" are different operational facts.
#[derive(Debug, Default, Clone, serde::Serialize)]
pub struct AudioMetrics {
    pub opened: u64,
    pub closed: u64,
    pub frames_in: u64,
    pub frames_stepped: u64,
    pub steps: u64,
    pub batch_last: usize,
    pub batch_max: usize,
    pub max_queue_depth: usize,
    pub shed_sessions_exceeded: u64,
    pub shed_queue_overflow: u64,
    pub shed_unknown_session: u64,
    pub shed_engine_unbound: u64,
    /// Device time spent in steps, ms. Against wall time this is the duty cycle, which is
    /// the only honest way to say how much room a card has left.
    pub step_ms_total: f64,
    /// Audio ingested across every lane, ms. Against wall time this is the concurrency the
    /// box actually served in real time, which is the numerator of `streams@SLO`.
    pub audio_ms_in: u64,
    /// Wall time observed, ms (the newest `now_ms` any call passed in).
    pub wall_ms: u64,
}

impl AudioMetrics {
    fn count(&mut self, code: ShedCode) {
        match code {
            ShedCode::SessionsExceeded => self.shed_sessions_exceeded += 1,
            ShedCode::QueueOverflow => self.shed_queue_overflow += 1,
            ShedCode::UnknownSession => self.shed_unknown_session += 1,
            ShedCode::EngineUnbound => self.shed_engine_unbound += 1,
        }
    }
    /// Fraction of WALL time the device spent stepping. Approaching 1.0 means the box
    /// cannot take another stream in real time whatever the latency percentiles say.
    pub fn step_duty(&self) -> f64 {
        if self.wall_ms == 0 {
            return 0.0;
        }
        self.step_ms_total / self.wall_ms as f64
    }
    /// Audio served per unit of wall time: the real-time concurrency actually carried.
    /// 64.0 means sixty-four streams were kept level with the clock.
    pub fn realtime_streams(&self) -> f64 {
        if self.wall_ms == 0 {
            return 0.0;
        }
        self.audio_ms_in as f64 / self.wall_ms as f64
    }
}

/// The scheduler the served endpoint drives, and the same object the replay tests drive.
/// The clock is the caller's (`now_ms`): the server passes wall time, the replay passes a
/// virtual clock, and neither gets a different code path.
pub struct AudioScheduler {
    pub policy: AudioPolicy,
    sessions: HashMap<String, AudioSession>,
    /// Round-robin cursor for `PerLaneWorker`, so the red arm's unfairness is the real
    /// one (a single worker walking lanes) and not "the first lane always wins".
    order: VecDeque<String>,
    metrics: AudioMetrics,
    engine_bound: bool,
}

impl AudioScheduler {
    pub fn new(policy: AudioPolicy, engine_bound: bool) -> Self {
        Self {
            policy,
            sessions: HashMap::new(),
            order: VecDeque::new(),
            metrics: AudioMetrics::default(),
            engine_bound,
        }
    }

    pub fn metrics(&self) -> &AudioMetrics {
        &self.metrics
    }
    pub fn live(&self) -> usize {
        self.sessions.len()
    }
    pub fn session(&self, id: &str) -> Option<&AudioSession> {
        self.sessions.get(id)
    }

    /// Open a session, or refuse with a code. The two refusals are deliberately ordered:
    /// an unbound engine is refused before capacity, because "full" would be a lie.
    pub fn open(&mut self, id: &str, lane: Lane, now_ms: u64) -> Result<(), AudioShed> {
        if !self.engine_bound {
            let shed = AudioShed::new(ShedCode::EngineUnbound, 0, 0, Some(id));
            self.metrics.count(shed.code);
            return Err(shed);
        }
        if self.sessions.len() >= self.policy.max_sessions {
            let shed = AudioShed::new(
                ShedCode::SessionsExceeded,
                self.policy.max_sessions,
                self.sessions.len(),
                Some(id),
            );
            self.metrics.count(shed.code);
            return Err(shed);
        }
        self.sessions.insert(
            id.to_string(),
            AudioSession {
                id: id.to_string(),
                lane,
                queue: VecDeque::with_capacity(self.policy.queue_frames),
                opened_ms: now_ms,
                frames_in: 0,
                frames_stepped: 0,
                max_depth_seen: 0,
                last_final_ms: None,
            },
        );
        self.order.push_back(id.to_string());
        self.metrics.opened += 1;
        Ok(())
    }

    /// Offer one frame of audio. Refuses with `QueueOverflow` when this lane's bounded
    /// queue is full — the c4 shape, now a typed refusal instead of a lost stream.
    pub fn offer(&mut self, id: &str, now_ms: u64) -> Result<usize, AudioShed> {
        let cap = self.policy.queue_frames;
        let Some(s) = self.sessions.get_mut(id) else {
            let shed = AudioShed::new(ShedCode::UnknownSession, 0, 0, Some(id));
            self.metrics.count(shed.code);
            return Err(shed);
        };
        if s.queue.len() >= cap {
            let depth = s.queue.len();
            let shed = AudioShed::new(ShedCode::QueueOverflow, cap, depth, Some(id));
            self.metrics.count(shed.code);
            return Err(shed);
        }
        s.queue.push_back(now_ms);
        s.frames_in += 1;
        let depth = s.queue.len();
        s.max_depth_seen = s.max_depth_seen.max(depth);
        self.metrics.frames_in += 1;
        self.metrics.audio_ms_in += FRAME_MS;
        self.metrics.max_queue_depth = self.metrics.max_queue_depth.max(depth);
        self.metrics.wall_ms = self.metrics.wall_ms.max(now_ms);
        Ok(depth)
    }

    /// The batch the next device step should run, chosen by the drive shape.
    ///
    /// `Fused`: every lane with a frame, capped by the resident-session cap.
    /// `PerLaneWorker`: at most one lane, round-robin.
    pub fn next_batch(&mut self) -> Vec<String> {
        match self.policy.drive {
            DriveShape::Fused => {
                let mut batch: Vec<String> = Vec::new();
                for id in self.order.iter() {
                    if self.sessions.get(id).is_some_and(|s| !s.queue.is_empty()) {
                        batch.push(id.clone());
                    }
                }
                batch
            }
            DriveShape::PerLaneWorker => {
                for _ in 0..self.order.len() {
                    let Some(id) = self.order.pop_front() else {
                        break;
                    };
                    self.order.push_back(id.clone());
                    if self.sessions.get(&id).is_some_and(|s| !s.queue.is_empty()) {
                        return vec![id];
                    }
                }
                Vec::new()
            }
        }
    }

    /// Account one completed device step: consume one frame from every session in the
    /// batch and record what the step cost. `step_ms` is the engine's measurement in
    /// production; nothing in this module invents it.
    pub fn complete_step(&mut self, batch: &[String], step_ms: f64, now_ms: u64) {
        for id in batch {
            if let Some(s) = self.sessions.get_mut(id) {
                if s.queue.pop_front().is_some() {
                    s.frames_stepped += 1;
                    s.last_final_ms = Some(now_ms);
                    self.metrics.frames_stepped += 1;
                }
            }
        }
        self.metrics.wall_ms = self.metrics.wall_ms.max(now_ms);
        if !batch.is_empty() {
            self.metrics.steps += 1;
            self.metrics.batch_last = batch.len();
            self.metrics.batch_max = self.metrics.batch_max.max(batch.len());
            self.metrics.step_ms_total += step_ms;
        }
    }

    pub fn close(&mut self, id: &str) -> Result<(), AudioShed> {
        if self.sessions.remove(id).is_none() {
            let shed = AudioShed::new(ShedCode::UnknownSession, 0, 0, Some(id));
            self.metrics.count(shed.code);
            return Err(shed);
        }
        self.order.retain(|x| x != id);
        self.metrics.closed += 1;
        Ok(())
    }
}

/// Cost of one fused streaming step at a given batch size, in ms.
pub trait StepCost {
    fn step_ms(&self, batch: usize) -> f64;
}

/// The measured step of the pinned two-tier RNNT student on one A100 80GB PCIe
/// (darklanes `research/speech-k1-20260910/RESULTS.md` §4).
///
/// Two anchors, both measured, nothing else: **31.522 ms at batch 1** and **35.890 ms at
/// batch 32**. Between them it interpolates linearly; beyond 32 it EXTRAPOLATES linearly,
/// which is deliberately pessimistic — the same cell measured the fused arm holding 64
/// streams at 44% step duty (35.2 ms per 80 ms frame) while this model predicts 40.400 ms
/// at batch 64, so the model charges the fused arm ~15% more than the card actually cost.
/// A model that flattered the arm we want to ship would be worthless as a red arm.
///
/// This is a scheduling model, not a performance claim. It says what the drive shapes do
/// with a step of the measured size. `streams@SLO` is a GPU measurement (G8) and no number
/// this model produces may be published as one.
#[derive(Debug, Clone, Copy, Default)]
pub struct MeasuredA100RnntStep;

impl MeasuredA100RnntStep {
    pub const BATCH1_MS: f64 = 31.522;
    pub const BATCH32_MS: f64 = 35.890;
    /// Measured fused-arm duty at 64 streams: 44% of an 80 ms frame.
    pub const MEASURED_B64_DUTY: f64 = 0.44;
}

impl StepCost for MeasuredA100RnntStep {
    fn step_ms(&self, batch: usize) -> f64 {
        let b = batch.max(1) as f64;
        let slope = (Self::BATCH32_MS - Self::BATCH1_MS) / 31.0;
        Self::BATCH1_MS + slope * (b - 1.0)
    }
}

/// What a real-time replay did.
#[derive(Debug, Clone)]
pub struct ReplayOutcome {
    /// Streams that were never refused anything and finished level with real time.
    pub held: usize,
    pub shed_streams: usize,
    pub shed: Vec<AudioShed>,
    pub max_queue_depth: usize,
    pub step_duty: f64,
    /// Real-time streams actually carried (audio served / wall elapsed).
    pub realtime_streams: f64,
    pub steps: u64,
    pub batch_max: usize,
    pub drive: DriveShape,
}

/// Drive `streams` concurrent real-time streams through the scheduler for `audio_ms` of
/// audio, on a virtual clock, and report what happened.
///
/// Real time is the whole point: every stream offers one frame every `FRAME_MS`, whether or
/// not the device is ready, exactly as a microphone does. The device is modelled as one
/// serial resource: it takes `cost.step_ms(batch)` of the virtual clock per step and cannot
/// start another until it finishes. That is the same seriality a single CUDA stream has.
///
/// This exercises the production `AudioScheduler` — `open`/`offer`/`next_batch`/
/// `complete_step` are the same calls the served endpoint makes. The only thing replaced is
/// the clock and the step's cost.
pub fn replay_realtime(
    policy: AudioPolicy,
    streams: usize,
    audio_ms: u64,
    cost: &dyn StepCost,
) -> ReplayOutcome {
    let mut sched = AudioScheduler::new(policy, true);
    let mut refused: Vec<AudioShed> = Vec::new();
    let mut bad: std::collections::HashSet<String> = std::collections::HashSet::new();

    for i in 0..streams {
        let id = format!("s{i}");
        if let Err(e) = sched.open(&id, Lane::Interactive, 0) {
            bad.insert(id);
            refused.push(e);
        }
    }

    // Virtual clock in 1 ms ticks. Two independent timelines: frame arrivals (every
    // FRAME_MS per stream) and the device (busy until `device_free_ms`).
    let mut device_free_ms: u64 = 0;
    let mut in_flight: Vec<String> = Vec::new();
    let mut in_flight_ms: f64 = 0.0;
    let horizon = audio_ms + FRAME_MS * 4;

    for now in 0..=horizon {
        // arrivals
        if now % FRAME_MS == 0 && now < audio_ms {
            for i in 0..streams {
                let id = format!("s{i}");
                if bad.contains(&id) {
                    continue;
                }
                if let Err(e) = sched.offer(&id, now) {
                    bad.insert(id);
                    refused.push(e);
                }
            }
        }
        // device completion
        if !in_flight.is_empty() && now >= device_free_ms {
            let batch = std::mem::take(&mut in_flight);
            sched.complete_step(&batch, in_flight_ms, now);
            in_flight_ms = 0.0;
        }
        // device dispatch
        if in_flight.is_empty() && now >= device_free_ms {
            let batch = sched.next_batch();
            if !batch.is_empty() {
                let ms = cost.step_ms(batch.len());
                in_flight_ms = ms;
                device_free_ms = now + ms.ceil() as u64;
                in_flight = batch;
            }
        }
    }

    let m = sched.metrics();
    ReplayOutcome {
        held: streams.saturating_sub(bad.len()),
        shed_streams: bad.len(),
        shed: refused,
        max_queue_depth: m.max_queue_depth,
        step_duty: m.step_duty(),
        realtime_streams: m.realtime_streams(),
        steps: m.steps,
        batch_max: m.batch_max,
        drive: policy.drive,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy(drive: DriveShape, max_sessions: usize) -> AudioPolicy {
        AudioPolicy {
            max_sessions,
            queue_frames: 64,
            drive,
            final_slo_ms: 800,
        }
    }

    /// The cost model must reproduce its own anchors, or every number below is decoration.
    #[test]
    fn cost_model_reproduces_its_measured_anchors() {
        let c = MeasuredA100RnntStep;
        assert!((c.step_ms(1) - 31.522).abs() < 1e-9, "batch 1 anchor");
        assert!((c.step_ms(32) - 35.890).abs() < 1e-9, "batch 32 anchor");
        // +13.9% from batch 1 to batch 32 — the measured ratio, not a typed one.
        let ratio = c.step_ms(32) / c.step_ms(1);
        assert!(
            (ratio - 1.139).abs() < 0.001,
            "batch 1 -> 32 must cost the measured +13.9%, got {ratio}"
        );
    }

    /// And it must not flatter the arm we intend to ship: at batch 64 the model has to
    /// charge at least as much as the card was measured costing (44% of an 80 ms frame).
    #[test]
    fn cost_model_is_pessimistic_against_the_measured_c64_duty() {
        let modelled = MeasuredA100RnntStep.step_ms(64) / FRAME_MS as f64;
        assert!(
            modelled >= MeasuredA100RnntStep::MEASURED_B64_DUTY,
            "model duty {modelled} must not undercut the measured 0.44"
        );
        assert!((MeasuredA100RnntStep.step_ms(64) - 40.3989).abs() < 1e-3);
    }

    /// THE RED ARM. The pinned c4 shape: one worker, batch 1, four real-time streams
    /// (`twotier/replay_stream.py:22`). One worker serves floor(80 / 31.522) = 2 frames per
    /// 80 ms of audio, so four lanes offering 4 frames per 80 ms bleed backlog at 2 frames
    /// per frame-period until every lane is pinned at the 64-frame cap and refused.
    ///
    /// The 2026-09-10 cell measured "sheds 2 of 4 streams at c4 with all lanes at the queue
    /// cap". This model sheds 4 of 4 over a 60 s replay, i.e. it is HARSHER than the card
    /// was, because the cell's clips ended while the two surviving lanes were still draining
    /// their backlog — "all lanes at the queue cap" is the cell saying exactly that. The
    /// invariant under test is not the count; it is that the shape refuses under sustained
    /// real-time load and that every refusal is typed.
    #[test]
    fn red_arm_per_lane_worker_sheds_at_c4() {
        let out = replay_realtime(
            policy(DriveShape::PerLaneWorker, 64),
            4,
            60_000,
            &MeasuredA100RnntStep,
        );
        assert_eq!(out.drive, DriveShape::PerLaneWorker);
        assert_eq!(out.held, 0, "every lane must eventually be refused");
        assert_eq!(out.shed_streams, 4);
        assert!(
            out.shed.iter().all(|s| s.code == ShedCode::QueueOverflow),
            "the c4 failure is a queue overflow, and it must be TYPED: {:?}",
            out.shed
        );
        // The incident's signature: every refused lane was pinned at the queue cap.
        for s in &out.shed {
            assert_eq!(s.limit, 64);
            assert_eq!(s.observed, 64);
            assert_eq!(s.code.http_status(), 429);
            assert!(s.code.retryable());
        }
        // And the card was never the problem: one worker at batch 1 carries about two
        // streams of real time, whatever four clients asked for.
        assert!(
            out.realtime_streams < 3.0,
            "one batch-1 worker cannot carry 4 streams: {}",
            out.realtime_streams
        );
    }

    /// THE RED ARM'S OWN TRAP, and the reason G8's concurrency ladder carries a minimum
    /// duration. The collapse is a slow bleed, not an instant failure: at c4 the per-lane
    /// worker refuses NOTHING in the first ten seconds while its queues climb to 48 of 64.
    /// A short capacity probe therefore returns a clean pass on the broken shape. The probe
    /// has to run longer than `queue_frames * FRAME_MS / (offered - served)` or it is
    /// measuring the queue, not the scheduler.
    #[test]
    fn red_arm_is_invisible_to_a_short_probe() {
        let short = replay_realtime(
            policy(DriveShape::PerLaneWorker, 64),
            4,
            10_000,
            &MeasuredA100RnntStep,
        );
        assert_eq!(short.shed_streams, 0, "a 10 s probe sees no refusal at all");
        assert!(
            short.max_queue_depth > 32,
            "and yet the backlog is already {} of 64 — the pass is an artifact",
            short.max_queue_depth
        );
        let long = replay_realtime(
            policy(DriveShape::PerLaneWorker, 64),
            4,
            60_000,
            &MeasuredA100RnntStep,
        );
        assert_eq!(
            long.shed_streams, 4,
            "the same shape, run long enough, fails"
        );
    }

    /// THE GREEN ARM, same card cost, same queue cap, same audio: one fused step per tick
    /// consuming a frame from every lane holds 64 of 64. The 2026-09-10 cell measured
    /// 64 of 64 at max queue depth 3 and 44% step duty.
    #[test]
    fn fused_drive_holds_64_streams() {
        let out = replay_realtime(
            policy(DriveShape::Fused, 64),
            64,
            60_000,
            &MeasuredA100RnntStep,
        );
        assert_eq!(out.held, 64, "fused must hold 64 of 64");
        assert!(
            out.shed.is_empty(),
            "nothing may be refused: {:?}",
            out.shed
        );
        assert_eq!(out.batch_max, 64, "the step must actually be fusing lanes");
        assert!(
            out.max_queue_depth <= 3,
            "measured max queue depth at c64 was 3, model says {}",
            out.max_queue_depth
        );
        assert!(
            out.realtime_streams > 63.0,
            "64 streams must be carried at real time, got {}",
            out.realtime_streams
        );
        // Duty against wall time, the measured quantity. The cell measured 0.44; this
        // model's pessimistic step cost puts it near 0.5 and it must stay under 1.0.
        assert!(
            out.step_duty > 0.40 && out.step_duty < 0.60,
            "duty {} should land near the measured 0.44",
            out.step_duty
        );
    }

    /// The two arms differ ONLY in the drive shape — same cost model, same cap, same
    /// audio, same stream count. A paired claim needs both assertions: the arms differed
    /// AND the instrument moved.
    #[test]
    fn drive_shape_alone_decides_c4() {
        let fused = replay_realtime(
            policy(DriveShape::Fused, 64),
            4,
            60_000,
            &MeasuredA100RnntStep,
        );
        let per_lane = replay_realtime(
            policy(DriveShape::PerLaneWorker, 64),
            4,
            60_000,
            &MeasuredA100RnntStep,
        );
        assert_eq!(fused.held, 4, "fused holds all four");
        assert_eq!(per_lane.held, 0, "per-lane refuses all four");
        assert!(
            fused.realtime_streams > 3.9 && per_lane.realtime_streams < 3.0,
            "the instrument has to move too: fused {} vs per-lane {}",
            fused.realtime_streams,
            per_lane.realtime_streams
        );
    }

    /// A cap that is reached must REFUSE, with the cap and the observed count on the
    /// refusal. This is the invariant the incident violated by dropping.
    #[test]
    fn session_cap_refuses_typed_and_never_drops() {
        let mut s = AudioScheduler::new(policy(DriveShape::Fused, 2), true);
        assert!(s.open("a", Lane::Interactive, 0).is_ok());
        assert!(s.open("b", Lane::Interactive, 0).is_ok());
        let e = s.open("c", Lane::Interactive, 0).expect_err("must refuse");
        assert_eq!(e.code, ShedCode::SessionsExceeded);
        assert_eq!((e.limit, e.observed), (2, 2));
        assert_eq!(e.code.http_status(), 503);
        assert_eq!(s.live(), 2, "the refusal must not have evicted a live lane");
        assert_eq!(s.metrics().shed_sessions_exceeded, 1);
        // and the counter is per-code, so an operator can tell the two limits apart
        assert_eq!(s.metrics().shed_queue_overflow, 0);
    }

    /// An unbound engine fails closed, and says so, BEFORE capacity is consulted — "full"
    /// would be a lie and a 200 would be worse.
    #[test]
    fn unbound_engine_fails_closed_before_capacity() {
        let mut s = AudioScheduler::new(policy(DriveShape::Fused, 64), false);
        let e = s.open("a", Lane::Interactive, 0).expect_err("must refuse");
        assert_eq!(e.code, ShedCode::EngineUnbound);
        assert_eq!(e.code.http_status(), 503);
        assert_eq!(s.live(), 0);
    }

    /// A frame for a session nobody opened is a 404 with a code, not a silent no-op: the
    /// caller has to be able to learn its session is gone.
    #[test]
    fn unknown_session_is_typed_not_ignored() {
        let mut s = AudioScheduler::new(policy(DriveShape::Fused, 64), true);
        let e = s.offer("ghost", 0).expect_err("must refuse");
        assert_eq!(e.code, ShedCode::UnknownSession);
        assert_eq!(e.code.http_status(), 404);
        assert!(!e.code.retryable());
        assert_eq!(s.metrics().shed_unknown_session, 1);
        assert_eq!(
            s.close("ghost").expect_err("close too").code,
            ShedCode::UnknownSession
        );
    }

    /// A typo in the drive env var must not select the shedding shape. The red arm is
    /// reachable only by naming it exactly.
    #[test]
    fn drive_env_default_and_typo_both_resolve_fused() {
        assert_eq!(DriveShape::parse("fused"), Some(DriveShape::Fused));
        assert_eq!(
            DriveShape::parse("per_lane_worker"),
            Some(DriveShape::PerLaneWorker)
        );
        assert_eq!(DriveShape::parse("per-lane-worker"), None);
        assert_eq!(DriveShape::parse("1"), None);
        assert_eq!(DriveShape::parse(""), None);
    }

    /// Every shed code round-trips its wire name, so a client can branch on it and a
    /// registry row can be grepped for it.
    #[test]
    fn shed_codes_carry_stable_wire_names() {
        for (code, name, status) in [
            (ShedCode::SessionsExceeded, "sessions_exceeded", 503),
            (ShedCode::QueueOverflow, "queue_overflow", 429),
            (ShedCode::UnknownSession, "unknown_session", 404),
            (ShedCode::EngineUnbound, "engine_unbound", 503),
        ] {
            assert_eq!(code.as_str(), name);
            assert_eq!(code.http_status(), status);
            assert!(!AudioShed::new(code, 1, 1, Some("x")).message().is_empty());
        }
    }

    /// Closing a session frees its slot for the next caller: the cap is a live gauge, not
    /// a high-water mark.
    #[test]
    fn close_frees_the_slot() {
        let mut s = AudioScheduler::new(policy(DriveShape::Fused, 1), true);
        s.open("a", Lane::Interactive, 0).unwrap();
        assert!(s.open("b", Lane::Interactive, 0).is_err());
        s.close("a").unwrap();
        assert!(s.open("b", Lane::Interactive, 0).is_ok());
        assert_eq!(s.metrics().opened, 2);
        assert_eq!(s.metrics().closed, 1);
    }

    /// The fused batch must contain EVERY ready lane and nothing else — a fused step that
    /// quietly served one lane would pass a "held" assertion while being the red arm.
    #[test]
    fn fused_batch_is_every_ready_lane() {
        let mut s = AudioScheduler::new(policy(DriveShape::Fused, 8), true);
        for id in ["a", "b", "c"] {
            s.open(id, Lane::Interactive, 0).unwrap();
        }
        s.offer("a", 0).unwrap();
        s.offer("c", 0).unwrap();
        let batch = s.next_batch();
        assert_eq!(batch, vec!["a".to_string(), "c".to_string()]);
        s.complete_step(&batch, 31.9, 32);
        assert!(s.next_batch().is_empty(), "frames must be consumed once");
        assert_eq!(s.metrics().frames_stepped, 2);
    }

    /// The per-lane worker walks lanes round-robin. Without this the red arm would be
    /// "lane 0 starves everyone", which is a different (and easier) defect.
    #[test]
    fn per_lane_worker_walks_round_robin() {
        let mut s = AudioScheduler::new(policy(DriveShape::PerLaneWorker, 8), true);
        for id in ["a", "b"] {
            s.open(id, Lane::Interactive, 0).unwrap();
            s.offer(id, 0).unwrap();
            s.offer(id, 0).unwrap();
        }
        let first = s.next_batch();
        s.complete_step(&first, 31.522, 32);
        let second = s.next_batch();
        s.complete_step(&second, 31.522, 64);
        assert_eq!(first.len(), 1);
        assert_eq!(second.len(), 1);
        assert_ne!(first[0], second[0], "one worker must not pin one lane");
    }

    /// Duty is wall-relative, and concurrency is audio-relative: those are two different
    /// questions ("how full is the card" vs "how many streams did it carry") and a single
    /// ratio cannot answer both. An empty scheduler reports 0, not NaN.
    #[test]
    fn duty_is_wall_relative_and_concurrency_is_audio_relative() {
        let mut s = AudioScheduler::new(policy(DriveShape::Fused, 8), true);
        assert_eq!(s.metrics().step_duty(), 0.0);
        assert_eq!(s.metrics().realtime_streams(), 0.0);
        // two lanes, one frame each, fused into one 40 ms step, over 80 ms of wall time:
        // the card was busy half the time and carried two streams of real time.
        for id in ["a", "b"] {
            s.open(id, Lane::Interactive, 0).unwrap();
            s.offer(id, 0).unwrap();
        }
        let b = s.next_batch();
        s.complete_step(&b, 40.0, 80);
        assert!((s.metrics().step_duty() - 0.5).abs() < 1e-9);
        assert!((s.metrics().realtime_streams() - 2.0).abs() < 1e-9);
    }
}
