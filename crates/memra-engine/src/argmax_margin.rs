//! The argmax margin gate's verdict (memra#1199). CPU only: no engine, no device.
//!
//! The gate compares two programs' logits at a window of prompt positions: the prefill
//! reference and a decode program. Per row it keeps each program's top-1 id and top-2 margin.
//!
//! **Gap shift.** A flip (prefill's top-1 `i`, decode's top-1 `j`, `i != j`) needs
//! `p_i - p_j >= margin_p` and `d_j - d_i >= margin_d`, so the two programs disagree about `i`
//! against `j` by at least `margin_p + margin_d`. At an agreeing row they disagree about the
//! top-1 against the runner-up by `|margin_p - margin_d|` (exact when both programs have the
//! same runner-up). That number is the row's gap shift.
//!
//! **Floor.** The largest gap shift at an agreeing row of a calibration receipt: this probe
//! over every row of the gate's own prompt, for one (model, prompt, decode program, device).
//!
//! **Verdict.** A flip whose gap shift is inside the floor is a near tie, counted against the
//! flip budget. A flip above the floor is a wide flip and fails at any count. An agreeing row
//! above the floor is a wide row and fails. With no floor a flip cannot be classified and
//! fails; agreeing rows are not judged and the verdict says so.
//!
//! **Known rows.** A calibration can hold flips above its own floor: rows where main's two
//! programs already disagree by more than they do at any agreeing row. The floor row lists
//! them with their recorded shift. A known row inside a judged window is printed and named,
//! and does not fail the cell by itself. A candidate whose gap shift at a known row exceeds
//! main's recorded shift at that row fails.
//!
//! The rule this replaced called a flip "margin-explained" when the spread at the contending
//! ids exceeded the smaller margin. The spread is at least the mean of the two margins at any
//! flip, so that held for every flip and could not fail (the `tests` module pins it).

use std::fmt::Write as _;

/// One window position: what the probe table holds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Row {
    pub pos: usize,
    pub prefill_top1: usize,
    pub margin_p: f32,
    pub decode_top1: usize,
    pub margin_d: f32,
    /// Largest |prefill - decode| over both programs' top-2 ids. Reported, not judged.
    pub spread: f32,
}

/// `(top-1 id, top-1 logit, top-2 id, top-2 logit)` of one logits row.
pub fn top2(l: &[f32]) -> (usize, f32, usize, f32) {
    let (mut i1, mut v1, mut i2, mut v2) = (0usize, f32::NEG_INFINITY, 0usize, f32::NEG_INFINITY);
    for (i, &v) in l.iter().enumerate() {
        if v > v1 {
            i2 = i1;
            v2 = v1;
            i1 = i;
            v1 = v;
        } else if v > v2 {
            i2 = i;
            v2 = v;
        }
    }
    (i1, v1, i2, v2)
}

impl Row {
    /// The table row of two programs' logits at one position.
    pub fn from_logits(pos: usize, prefill: &[f32], decode: &[f32]) -> Row {
        let (p1, pv1, p2, pv2) = top2(prefill);
        let (d1, dv1, d2, dv2) = top2(decode);
        let spread = [p1, p2, d1, d2]
            .iter()
            .map(|&i| (prefill[i] - decode[i]).abs())
            .fold(0.0f32, f32::max);
        Row {
            pos,
            prefill_top1: p1,
            margin_p: pv1 - pv2,
            decode_top1: d1,
            margin_d: dv1 - dv2,
            spread,
        }
    }

    pub fn is_flip(&self) -> bool {
        self.prefill_top1 != self.decode_top1
    }

    /// See the module doc: the margin sum at a flip, the margin difference at an agreeing row.
    pub fn gap_shift(&self) -> f32 {
        if self.is_flip() {
            self.margin_p + self.margin_d
        } else {
            (self.margin_p - self.margin_d).abs()
        }
    }

    /// The table line. Ten significant digits preserve every finite f32, so the judge reads
    /// back the values the probe measured.
    pub fn table_line(&self) -> String {
        format!(
            "{:<8} {:<13} {:<16.9e} {:<12} {:<16.9e} {:<16.9e} {}",
            self.pos,
            self.prefill_top1,
            self.margin_p,
            self.decode_top1,
            self.margin_d,
            self.spread,
            if self.is_flip() { "NO <-- FLIP" } else { "yes" }
        )
    }
}

/// The examination line of one row: both programs' top-2 ids with their text, logits and
/// margins, each program's logit at the other's top-1, the spread and the gap shift. The gate
/// repeats this line for every flip and every wide row, so a reader can examine the row from
/// the log alone.
pub fn detail_line(
    pos: usize,
    prefill: &[f32],
    decode: &[f32],
    text: &dyn Fn(usize) -> String,
) -> String {
    let (p1, pv1, p2, pv2) = top2(prefill);
    let (d1, dv1, d2, dv2) = top2(decode);
    let row = Row::from_logits(pos, prefill, decode);
    format!(
        "detail pos={pos} prefill: top1={p1} {:?} {pv1:.4} top2={p2} {:?} {pv2:.4} margin={:.4} \
         at-decode-top1={:.4} | decode: top1={d1} {:?} {dv1:.4} top2={d2} {:?} {dv2:.4} \
         margin={:.4} at-prefill-top1={:.4} | spread={:.4} gap_shift={:.4}",
        text(p1),
        text(p2),
        row.margin_p,
        prefill[d1],
        text(d1),
        text(d2),
        row.margin_d,
        decode[p1],
        row.spread,
        row.gap_shift(),
    )
}

/// Parse the probe table out of a log: every line `pos id margin id margin spread yes|NO`.
/// A duplicate position or a negative or non-finite number is an error; so is an empty table.
pub fn parse_table(log: &str) -> Result<Vec<Row>, String> {
    let mut rows: Vec<Row> = Vec::new();
    for line in log.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() < 7 || f[0].parse::<usize>().is_err() || f[1].parse::<usize>().is_err() {
            continue;
        }
        let num = |s: &str| -> Result<f32, String> {
            let v: f32 = s
                .parse()
                .map_err(|_| format!("table row {}: {s:?} is not a number", f[0]))?;
            if !v.is_finite() || v < 0.0 {
                return Err(format!(
                    "table row {}: {s:?} is negative or not finite",
                    f[0]
                ));
            }
            Ok(v)
        };
        let row = Row {
            pos: f[0].parse().expect("checked above"),
            prefill_top1: f[1].parse().expect("checked above"),
            margin_p: num(f[2])?,
            decode_top1: f[3]
                .parse()
                .map_err(|_| format!("table row {}: decode id {:?}", f[0], f[3]))?,
            margin_d: num(f[4])?,
            spread: num(f[5])?,
        };
        let agree = match f[6] {
            "yes" => true,
            "NO" => false,
            other => return Err(format!("table row {}: agree column {other:?}", f[0])),
        };
        if agree == row.is_flip() {
            return Err(format!(
                "table row {}: the agree column contradicts the two top-1 ids",
                f[0]
            ));
        }
        if rows.iter().any(|r| r.pos == row.pos) {
            return Err(format!("table row {}: duplicate position", f[0]));
        }
        rows.push(row);
    }
    if rows.is_empty() {
        return Err("no probe table rows".into());
    }
    Ok(rows)
}

/// A flip of the calibrated binary whose gap shift is above its own floor: a row where main's
/// two programs disagree by more than they do at any agreeing row of the prompt. It is kept
/// in the floor row with its shift, so the gate names it when a judged window holds it and a
/// candidate may not widen it (memra#1203 asks which program is right at these rows).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KnownRow {
    pub pos: usize,
    pub prefill_top1: usize,
    pub margin_p: f32,
    pub decode_top1: usize,
    pub margin_d: f32,
}

impl KnownRow {
    /// The recorded gap shift: the margin sum of the calibration flip.
    pub fn gap_shift(&self) -> f32 {
        self.margin_p + self.margin_d
    }

    /// `pos:prefill_top1:margin_p:decode_top1:margin_d`, the floors-table form.
    fn encode(&self) -> String {
        format!(
            "{}:{}:{:.9e}:{}:{:.9e}",
            self.pos, self.prefill_top1, self.margin_p, self.decode_top1, self.margin_d
        )
    }

    fn decode(text: &str) -> Option<KnownRow> {
        let f: Vec<&str> = text.split(':').collect();
        let [pos, p1, mp, d1, md] = f.as_slice() else {
            return None;
        };
        let margin = |s: &str| s.parse::<f32>().ok().filter(|v| v.is_finite() && *v >= 0.0);
        Some(KnownRow {
            pos: pos.parse().ok()?,
            prefill_top1: p1.parse().ok()?,
            margin_p: margin(mp)?,
            decode_top1: d1.parse().ok()?,
            margin_d: margin(md)?,
        })
    }
}

/// What a wide window measured: the floor it sets, and how loose that floor is.
#[derive(Debug, Clone, PartialEq)]
pub struct Calibration {
    /// Largest gap shift at an agreeing row.
    pub floor: f32,
    /// The row that set it.
    pub set_by_row: usize,
    /// Agreeing rows measured.
    pub agreeing: usize,
    /// Flips in the window (not part of the floor).
    pub flips: usize,
    /// The median, 99th and 99.9th percentile of the agreeing rows' gap shift (nearest rank).
    /// A floor far above them was set by one or two rows.
    pub p50: f32,
    pub p99: f32,
    pub p999: f32,
    /// The calibration table's own flips whose gap shift is above the floor.
    pub known: Vec<KnownRow>,
}

/// The floor a table sets, or None when no row agrees.
pub fn calibrate(rows: &[Row]) -> Option<Calibration> {
    let mut agree: Vec<(f32, usize)> = rows
        .iter()
        .filter(|r| !r.is_flip())
        .map(|r| (r.gap_shift(), r.pos))
        .collect();
    // The first row at the maximum sets the floor.
    let &(floor, set_by_row) = agree
        .iter()
        .reduce(|best, r| if r.0 > best.0 { r } else { best })?;
    agree.sort_by(|a, b| a.0.total_cmp(&b.0));
    let rank = |p: f64| agree[((agree.len() - 1) as f64 * p).round() as usize].0;
    Some(Calibration {
        floor,
        set_by_row,
        agreeing: agree.len(),
        flips: rows.len() - agree.len(),
        p50: rank(0.5),
        p99: rank(0.99),
        p999: rank(0.999),
        known: rows
            .iter()
            .filter(|r| r.is_flip() && r.gap_shift() > floor)
            .map(|r| KnownRow {
                pos: r.pos,
                prefill_top1: r.prefill_top1,
                margin_p: r.margin_p,
                decode_top1: r.decode_top1,
                margin_d: r.margin_d,
            })
            .collect(),
    })
}

/// The largest count of flips in any `window` consecutive rows of a table: how many near ties
/// one gate window of this prompt can hold.
pub fn max_flips_in_window(rows: &[Row], window: usize) -> usize {
    let flips: Vec<usize> = rows.iter().map(|r| r.is_flip() as usize).collect();
    flips
        .windows(window.clamp(1, flips.len().max(1)))
        .map(|w| w.iter().sum())
        .max()
        .unwrap_or(0)
}

/// Which clause a row falls under.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Clause {
    /// A flip whose gap shift is inside the floor. Counted against the budget.
    NearTie,
    /// A flip whose gap shift is above the floor. Fails.
    WideFlip,
    /// An agreeing row whose gap shift is above the floor. Fails.
    WideRow,
    /// A flip with no floor to judge it by. Fails.
    UncalibratedFlip,
    /// A known row of the calibration, no wider here than it was recorded. Printed and named;
    /// it does not fail the cell by itself and is not counted against the budget.
    KnownRow,
    /// A known row whose gap shift here exceeds the recorded one. Fails.
    KnownRowWider,
}

impl Clause {
    pub fn name(self) -> &'static str {
        match self {
            Clause::NearTie => "near-tie",
            Clause::WideFlip => "WIDE-FLIP",
            Clause::WideRow => "WIDE-ROW",
            Clause::UncalibratedFlip => "UNCALIBRATED-FLIP",
            Clause::KnownRow => "KNOWN-ROW",
            Clause::KnownRowWider => "KNOWN-ROW-WIDER",
        }
    }

    pub fn fails(self) -> bool {
        !matches!(self, Clause::NearTie | Clause::KnownRow)
    }
}

/// The gate's verdict on one table.
#[derive(Debug, Clone, PartialEq)]
pub struct Verdict {
    pub floor: Option<f32>,
    pub budget: usize,
    pub flips: usize,
    pub near_ties: usize,
    /// Every flip, every wide row and every known row in the table, in table order.
    pub findings: Vec<(Row, Clause)>,
    /// The known rows the table was judged with.
    pub known: Vec<KnownRow>,
    /// Near ties above the budget.
    pub over_budget: bool,
}

impl Verdict {
    /// Failing findings, plus one for a budget overrun: the `bad=` count of the summary line.
    pub fn bad(&self) -> usize {
        self.findings.iter().filter(|(_, c)| c.fails()).count() + self.over_budget as usize
    }

    pub fn passes(&self) -> bool {
        self.bad() == 0
    }

    /// One line per finding and the summary line. `detail` returns the probe's examination
    /// line for a position when the log holds one.
    pub fn report(&self, detail: &dyn Fn(usize) -> Option<String>) -> String {
        let mut out = String::new();
        let floor = match self.floor {
            Some(f) => format!("{f:.4}"),
            None => "none".into(),
        };
        for (r, clause) in &self.findings {
            let against = match clause {
                Clause::NearTie => format!("<= floor={floor}"),
                Clause::WideFlip | Clause::WideRow => format!("> floor={floor}"),
                Clause::UncalibratedFlip => format!("vs floor={floor}"),
                Clause::KnownRow | Clause::KnownRowWider => {
                    let k = self
                        .known
                        .iter()
                        .find(|k| k.pos == r.pos)
                        .expect("a known-row finding has its known row");
                    format!(
                        "{} recorded={:.4} (calibration: prefill_top1={} margin_p={:.4} \
                         decode_top1={} margin_d={:.4}; memra#1203) floor={floor}",
                        if *clause == Clause::KnownRow {
                            "<="
                        } else {
                            ">"
                        },
                        k.gap_shift(),
                        k.prefill_top1,
                        k.margin_p,
                        k.decode_top1,
                        k.margin_d,
                    )
                }
            };
            let _ = writeln!(
                out,
                "{} pos={} prefill_top1={} margin_p={:.4} decode_top1={} margin_d={:.4} \
                 spread={:.4} gap_shift={:.4} {against}",
                clause.name(),
                r.pos,
                r.prefill_top1,
                r.margin_p,
                r.decode_top1,
                r.margin_d,
                r.spread,
                r.gap_shift(),
            );
            if let Some(line) = detail(r.pos) {
                let _ = writeln!(out, "  {line}");
            }
        }
        if self.over_budget {
            let _ = writeln!(
                out,
                "TOO-MANY-NEAR-TIES {} > budget {}",
                self.near_ties, self.budget
            );
        }
        let count = |c: Clause| self.findings.iter().filter(|(_, x)| *x == c).count();
        let _ = writeln!(
            out,
            "SUMMARY flips={} bad={} near_ties={} wide_flips={} wide_rows={} uncalibrated_flips={} \
             known_rows={} known_rows_wider={} floor={floor} budget={}",
            self.flips,
            self.bad(),
            self.near_ties,
            count(Clause::WideFlip),
            count(Clause::WideRow),
            count(Clause::UncalibratedFlip),
            count(Clause::KnownRow),
            count(Clause::KnownRowWider),
            self.budget,
        );
        out
    }
}

/// Judge a table against a floor, the floor's known rows and a near-tie budget. This is the
/// gate's only verdict function: the probe's `--judge` mode calls it on the table of a log.
///
/// A row at a known position is judged by its recorded shift, not by the floor: at or below
/// it the row is named and passes; above it the row fails. Every other row is judged by the
/// floor.
pub fn judge(rows: &[Row], floor: Option<f32>, known: &[KnownRow], budget: usize) -> Verdict {
    let mut findings = Vec::new();
    let (mut flips, mut near_ties) = (0usize, 0usize);
    for r in rows {
        let recorded = floor.and(known.iter().find(|k| k.pos == r.pos));
        let clause = match (recorded, r.is_flip(), floor) {
            (Some(k), _, _) if r.gap_shift() > k.gap_shift() => Some(Clause::KnownRowWider),
            (Some(_), _, _) => Some(Clause::KnownRow),
            (None, true, None) => Some(Clause::UncalibratedFlip),
            (None, true, Some(f)) if r.gap_shift() > f => Some(Clause::WideFlip),
            (None, true, Some(_)) => Some(Clause::NearTie),
            (None, false, Some(f)) if r.gap_shift() > f => Some(Clause::WideRow),
            (None, false, _) => None,
        };
        if r.is_flip() {
            flips += 1;
        }
        if clause == Some(Clause::NearTie) {
            near_ties += 1;
        }
        if let Some(c) = clause {
            findings.push((*r, c));
        }
    }
    Verdict {
        floor,
        budget,
        flips,
        near_ties,
        findings,
        known: known.to_vec(),
        over_budget: near_ties > budget,
    }
}

/// One row of `tools/argmax-margin-floors.tsv`.
#[derive(Debug, Clone, PartialEq)]
pub struct FloorRow {
    /// Model file name without `.gguf`, or the checkpoint directory name.
    pub model: String,
    /// Prompt file name without `.txt`.
    pub prompt: String,
    /// sha256 of the prompt file: a floor belongs to the prompt's bytes, not its name.
    pub prompt_sha256: String,
    /// `eager` or `batch`.
    pub decode: String,
    /// The device name `nvidia-smi` reports.
    pub device: String,
    pub floor: f32,
    /// Agreeing rows the floor is the maximum of.
    pub rows: usize,
    pub set_by_row: usize,
    /// The median, 99th and 99.9th percentile of the agreeing rows' gap shift: how far above
    /// the bulk the floor sits.
    pub p50: f32,
    pub p99: f32,
    pub p999: f32,
    /// The calibration table's own flips above the floor (`-` in the table for none).
    pub known: Vec<KnownRow>,
    /// The calibration receipt, a path in this repo.
    pub receipt: String,
}

impl FloorRow {
    /// The `FLOOR:` line the judge prints: the floor, what it is the maximum of and how loose
    /// it is.
    pub fn describe(&self) -> String {
        let known = match self.known.as_slice() {
            [] => "no calibration flip above it".to_string(),
            rows => format!(
                "known rows (calibration flips above it, memra#1203): {}",
                rows.iter()
                    .map(|k| format!("{} (shift {:.4})", k.pos, k.gap_shift()))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        };
        format!(
            "{:.4} (largest gap shift over {} agreeing rows, set by row {}; p50 {:.4}, p99 \
             {:.4}, p99.9 {:.4}; {known}; {})",
            self.floor, self.rows, self.set_by_row, self.p50, self.p99, self.p999, self.receipt
        )
    }
}

/// What a floor row is looked up by.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FloorKey<'a> {
    pub model: &'a str,
    pub prompt: &'a str,
    pub prompt_sha256: &'a str,
    pub decode: &'a str,
    pub device: &'a str,
}

/// Parse the floors table: tab-separated, `#` comments. A malformed row, a number that is
/// negative or not finite, percentiles out of order, a known row inside the floor, or two rows
/// with one key is an error.
pub fn parse_floors(text: &str) -> Result<Vec<FloorRow>, String> {
    let mut rows: Vec<FloorRow> = Vec::new();
    for (n, line) in text.lines().enumerate() {
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() != 13 || f.iter().any(|s| s.trim().is_empty()) {
            return Err(format!(
                "floors line {}: expected 13 tab-separated fields",
                n + 1
            ));
        }
        let number = |name: &str, text: &str| -> Result<f32, String> {
            let v: f32 = text
                .parse()
                .map_err(|_| format!("floors line {}: {name} {text:?}", n + 1))?;
            if !v.is_finite() || v < 0.0 {
                return Err(format!(
                    "floors line {}: {name} {text:?} is negative or not finite",
                    n + 1
                ));
            }
            Ok(v)
        };
        let floor = number("floor", f[5])?;
        let (p50, p99, p999) = (
            number("p50", f[8])?,
            number("p99", f[9])?,
            number("p99.9", f[10])?,
        );
        if p50 > p99 || p99 > p999 || p999 > floor {
            return Err(format!(
                "floors line {}: p50 <= p99 <= p99.9 <= floor does not hold",
                n + 1
            ));
        }
        let mut known = Vec::new();
        if f[11] != "-" {
            for entry in f[11].split(';') {
                let k = KnownRow::decode(entry)
                    .ok_or_else(|| format!("floors line {}: known row {entry:?}", n + 1))?;
                if k.gap_shift() <= floor || known.iter().any(|o: &KnownRow| o.pos == k.pos) {
                    return Err(format!(
                        "floors line {}: known row {} is inside the floor or listed twice",
                        n + 1,
                        k.pos
                    ));
                }
                known.push(k);
            }
        }
        if !matches!(f[3], "eager" | "batch") {
            return Err(format!("floors line {}: decode {:?}", n + 1, f[3]));
        }
        let row = FloorRow {
            model: f[0].into(),
            prompt: f[1].into(),
            prompt_sha256: f[2].into(),
            decode: f[3].into(),
            device: f[4].into(),
            floor,
            rows: f[6]
                .parse()
                .map_err(|_| format!("floors line {}: rows {:?}", n + 1, f[6]))?,
            set_by_row: f[7]
                .parse()
                .map_err(|_| format!("floors line {}: set_by_row {:?}", n + 1, f[7]))?,
            p50,
            p99,
            p999,
            known,
            receipt: f[12].into(),
        };
        if rows.iter().any(|r| {
            (&r.model, &r.prompt, &r.prompt_sha256, &r.decode, &r.device)
                == (
                    &row.model,
                    &row.prompt,
                    &row.prompt_sha256,
                    &row.decode,
                    &row.device,
                )
        }) {
            return Err(format!("floors line {}: a second row for one key", n + 1));
        }
        rows.push(row);
    }
    Ok(rows)
}

/// The floor row of a key. Every field must match exactly: there is no fallback row.
pub fn find_floor<'a>(rows: &'a [FloorRow], key: &FloorKey<'_>) -> Option<&'a FloorRow> {
    rows.iter().find(|r| {
        r.model == key.model
            && r.prompt == key.prompt
            && r.prompt_sha256 == key.prompt_sha256
            && r.decode == key.decode
            && r.device == key.device
    })
}

/// The floors-table line a calibration sets, tab-separated in column order.
pub fn floor_line(key: &FloorKey<'_>, cal: &Calibration, receipt: &str) -> String {
    let known = match cal.known.as_slice() {
        [] => "-".to_string(),
        rows => rows
            .iter()
            .map(KnownRow::encode)
            .collect::<Vec<_>>()
            .join(";"),
    };
    format!(
        "{}\t{}\t{}\t{}\t{}\t{:.9e}\t{}\t{}\t{:.9e}\t{:.9e}\t{:.9e}\t{known}\t{}",
        key.model,
        key.prompt,
        key.prompt_sha256,
        key.decode,
        key.device,
        cal.floor,
        cal.agreeing,
        cal.set_by_row,
        cal.p50,
        cal.p99,
        cal.p999,
        receipt
    )
}

// ---- stored rows (memra#1204) ---------------------------------------------------------
//
// The floor judges prefill against decode inside one build, and it is loose where one row
// sets it. The stored rows judge a candidate against main: at calibration, main's probe
// table for the gate's window is stored as printed, and at gate time the candidate's table
// for the same window must equal it line for line. There is no tolerance.

/// What a stored table belongs to. Every field is part of the key: a changed prompt or model
/// artifact reads as "no stored table", never as a difference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoredKey {
    /// Model file name without `.gguf`, or the checkpoint directory name.
    pub model: String,
    /// sha256 of the model artifact's bytes.
    pub model_sha256: String,
    /// Prompt file name without `.txt`.
    pub prompt: String,
    pub prompt_sha256: String,
    /// `eager` or `batch`.
    pub decode: String,
    /// The device name `nvidia-smi` reports.
    pub device: String,
    /// Rows in the gate's window.
    pub window: usize,
    /// Where the MoE experts ran (`experts_of`): `resident`, `slru` or `none`. The SLRU cache is
    /// another dispatch program than the resident experts, so its table is another table.
    pub experts: String,
}

/// The MoE expert placement a probe log ran with, from the engine's `[moe] resident-experts
/// decision` lines: `resident` when every one says RESIDENT, `slru` when any says otherwise
/// (the SLRU cache), `none` when the log has none (a dense model, or experts forced off the card
/// with MEMRA_MOE_RESIDENT=0, which prints no decision). Main's rows for an MoE model measured
/// resident are then no stored table for a run that was not.
pub fn experts_of(log: &str) -> &'static str {
    let decisions: Vec<&str> = log
        .lines()
        .filter(|l| l.starts_with("[moe] resident-experts decision"))
        .collect();
    if decisions.is_empty() {
        "none"
    } else if decisions
        .iter()
        .all(|l| l.trim_end().ends_with("-> RESIDENT"))
    {
        "resident"
    } else {
        "slru"
    }
}

/// Main's probe table for one cell's window, as printed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoredTable {
    pub key: StoredKey,
    /// The commit whose binary measured the rows.
    pub commit: String,
    /// The calibration receipt, a path in this repo.
    pub receipt: String,
    /// The probe's table lines, verbatim apart from trailing whitespace.
    pub rows: Vec<String>,
}

/// The table lines of a log, as printed (trailing whitespace dropped): the lines `parse_table`
/// reads. Errors are `parse_table`'s.
pub fn table_lines(log: &str) -> Result<Vec<String>, String> {
    parse_table(log)?;
    Ok(log
        .lines()
        .filter(|line| {
            let f: Vec<&str> = line.split_whitespace().collect();
            f.len() >= 7 && f[0].parse::<usize>().is_ok() && f[1].parse::<usize>().is_ok()
        })
        .map(|line| line.trim_end().to_string())
        .collect())
}

impl StoredTable {
    /// The file form: `name<TAB>value` header lines, a `rows` line, then the table lines.
    pub fn render(&self) -> String {
        let k = &self.key;
        let mut out = format!(
            "# Main's argmax margin probe rows for one gate cell (memra#1204). The gate compares a\n\
             # candidate's table for this window with these lines, byte for byte.\n\
             model\t{}\nmodel_sha256\t{}\nprompt\t{}\nprompt_sha256\t{}\ndecode\t{}\ndevice\t{}\n\
             window\t{}\nexperts\t{}\ncommit\t{}\nreceipt\t{}\nrows\n",
            k.model,
            k.model_sha256,
            k.prompt,
            k.prompt_sha256,
            k.decode,
            k.device,
            k.window,
            k.experts,
            self.commit,
            self.receipt
        );
        for row in &self.rows {
            out.push_str(row);
            out.push('\n');
        }
        out
    }

    /// Parse the file form. A missing or repeated header field, an unknown one, a row count
    /// that is not the window, or rows that are not a probe table is an error.
    pub fn parse(text: &str) -> Result<StoredTable, String> {
        const FIELDS: [&str; 10] = [
            "model",
            "model_sha256",
            "prompt",
            "prompt_sha256",
            "decode",
            "device",
            "window",
            "experts",
            "commit",
            "receipt",
        ];
        let mut values: Vec<Option<String>> = vec![None; FIELDS.len()];
        let mut lines = text.lines();
        let mut saw_rows = false;
        for line in lines.by_ref() {
            if line.starts_with('#') || line.trim().is_empty() {
                continue;
            }
            if line == "rows" {
                saw_rows = true;
                break;
            }
            let (name, value) = line
                .split_once('\t')
                .ok_or_else(|| format!("stored table: header line {line:?}"))?;
            let slot = FIELDS
                .iter()
                .position(|f| *f == name)
                .ok_or_else(|| format!("stored table: unknown field {name:?}"))?;
            if value.trim().is_empty() || values[slot].replace(value.to_string()).is_some() {
                return Err(format!("stored table: field {name:?} is empty or repeated"));
            }
        }
        if !saw_rows {
            return Err("stored table: no `rows` line".into());
        }
        let mut field = |name: &str| -> Result<String, String> {
            let slot = FIELDS.iter().position(|f| *f == name).expect("known field");
            values[slot]
                .take()
                .ok_or_else(|| format!("stored table: field {name:?} is missing"))
        };
        let window: usize = field("window")?
            .parse()
            .map_err(|_| "stored table: window is not a number".to_string())?;
        let body: String = lines.map(|l| format!("{l}\n")).collect();
        let rows = table_lines(&body).map_err(|e| format!("stored table: {e}"))?;
        if rows.len() != window || window == 0 {
            return Err(format!(
                "stored table: {} rows for a window of {window}",
                rows.len()
            ));
        }
        let decode = field("decode")?;
        if !matches!(decode.as_str(), "eager" | "batch") {
            return Err(format!("stored table: decode {decode:?}"));
        }
        let experts = field("experts")?;
        if !matches!(experts.as_str(), "resident" | "slru" | "none") {
            return Err(format!("stored table: experts {experts:?}"));
        }
        Ok(StoredTable {
            key: StoredKey {
                model: field("model")?,
                model_sha256: field("model_sha256")?,
                prompt: field("prompt")?,
                prompt_sha256: field("prompt_sha256")?,
                decode,
                device: field("device")?,
                window,
                experts,
            },
            commit: field("commit")?,
            receipt: field("receipt")?,
            rows,
        })
    }
}

/// The stored table of a key among parsed tables. Every key field must match exactly; two
/// tables with one key is an error.
pub fn find_stored<'a>(
    tables: &'a [StoredTable],
    key: &StoredKey,
) -> Result<Option<&'a StoredTable>, String> {
    let mut found = tables.iter().filter(|t| t.key == *key);
    let first = found.next();
    if found.next().is_some() {
        return Err(format!(
            "two stored tables for model={} prompt={} decode={} device={:?} window={}",
            key.model, key.prompt, key.decode, key.device, key.window
        ));
    }
    Ok(first)
}

/// One position where the candidate's table line is not the stored one. `None` is a
/// position one side does not have.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RowDiff {
    pub pos: usize,
    pub stored: Option<String>,
    pub candidate: Option<String>,
}

fn line_pos(line: &str) -> usize {
    line.split_whitespace()
        .next()
        .and_then(|p| p.parse().ok())
        .expect("a table line starts with its position")
}

/// The positions where a candidate's table lines differ from the stored ones, in position
/// order. Identity is byte identity of the printed line: no tolerance.
pub fn compare_rows(stored: &[String], candidate: &[String]) -> Vec<RowDiff> {
    let mut positions: Vec<usize> = stored
        .iter()
        .chain(candidate)
        .map(|l| line_pos(l))
        .collect();
    positions.sort_unstable();
    positions.dedup();
    positions
        .into_iter()
        .filter_map(|pos| {
            let at = |lines: &[String]| lines.iter().find(|l| line_pos(l) == pos).cloned();
            let (s, c) = (at(stored), at(candidate));
            (s != c).then_some(RowDiff {
                pos,
                stored: s,
                candidate: c,
            })
        })
        .collect()
}

/// One side of a difference: both programs' top-1 ids and margins and the spread, or the
/// reason the side is absent.
fn side(line: &Option<String>) -> String {
    match line.as_deref().map(parse_table) {
        Some(Ok(rows)) => {
            let r = rows[0];
            format!(
                "prefill_top1={} margin_p={:.9e} decode_top1={} margin_d={:.9e} spread={:.9e}{}",
                r.prefill_top1,
                r.margin_p,
                r.decode_top1,
                r.margin_d,
                r.spread,
                if r.is_flip() { " FLIP" } else { "" }
            )
        }
        Some(Err(err)) => format!("unreadable ({err})"),
        None => "no row at this position".into(),
    }
}

/// The report of a comparison: one `ROW-DIFF` block per differing position, both sides.
/// `detail` returns the candidate probe's examination line for a position when the log holds
/// one.
pub fn rows_report(diffs: &[RowDiff], detail: &dyn Fn(usize) -> Option<String>) -> String {
    let mut out = String::new();
    for d in diffs {
        let _ = writeln!(out, "ROW-DIFF pos={}", d.pos);
        let _ = writeln!(out, "  stored (main): {}", side(&d.stored));
        let _ = writeln!(out, "  candidate:     {}", side(&d.candidate));
        if let Some(line) = detail(d.pos) {
            let _ = writeln!(out, "  {line}");
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A logits row over a small vocabulary: everything at 0 except the named ids.
    fn logits(named: &[(usize, f32)]) -> Vec<f32> {
        let mut l = vec![0.0f32; 16];
        for &(i, v) in named {
            l[i] = v;
        }
        l
    }

    fn row(pos: usize, prefill: &[(usize, f32)], decode: &[(usize, f32)]) -> Row {
        Row::from_logits(pos, &logits(prefill), &logits(decode))
    }

    /// The predicate the gate used until memra#1199: a flip is "margin-explained" when the
    /// spread at the contending ids exceeds the smaller of the two top-2 margins.
    fn old_rule_explains(r: &Row) -> bool {
        r.spread > r.margin_p.min(r.margin_d)
    }

    /// A deterministic stream of logits rows (xorshift), wide enough to hit every case.
    fn random_rows(n: usize) -> Vec<(Vec<f32>, Vec<f32>)> {
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 24) as f32
        };
        (0..n)
            .map(|_| {
                let p: Vec<f32> = (0..12).map(|_| next() * 20.0).collect();
                // The other program: the same row moved by up to 6 logits per id.
                let d: Vec<f32> = p.iter().map(|v| v + (next() - 0.5) * 12.0).collect();
                (p, d)
            })
            .collect()
    }

    #[test]
    fn the_old_predicate_is_true_for_every_flip() {
        // Why the rule was replaced: p_i - p_j >= margin_p and d_j - d_i >= margin_d give
        // (p_i - d_i) + (d_j - p_j) >= margin_p + margin_d, so the spread over the contending
        // ids is at least the mean margin, which is at least the smaller one. No flip, at any
        // margin, was ever "unexplained".
        let mut flips = 0;
        let mut widest = 0.0f32;
        for (i, (p, d)) in random_rows(20_000).iter().enumerate() {
            let r = Row::from_logits(i, p, d);
            if !r.is_flip() {
                continue;
            }
            flips += 1;
            widest = widest.max(r.margin_p.max(r.margin_d));
            assert!(r.spread >= (r.margin_p + r.margin_d) / 2.0 - 1e-4, "{r:?}");
            assert!(old_rule_explains(&r), "{r:?}");
        }
        assert!(flips > 5_000, "{flips} flips");
        // The stream holds flips nobody would call a near tie, and the old rule passed them.
        assert!(widest > 5.0, "widest margin at a flip {widest}");
    }

    #[test]
    fn a_wide_margin_flip_fails() {
        // Prefill holds id 1 at margin 14.4; decode picks id 2 at margin 2.5 (memra#1175's row
        // on the split-rung candidate). The old predicate passes it; the gate must not.
        let r = row(7, &[(1, 25.0), (2, 10.6)], &[(1, 9.5), (2, 12.0)]);
        assert!(r.is_flip());
        assert!(old_rule_explains(&r));
        let v = judge(&[r], Some(1.0), &[], 1);
        assert_eq!(v.findings, vec![(r, Clause::WideFlip)]);
        assert!(!v.passes());
        assert_eq!((v.flips, v.near_ties, v.bad()), (1, 0, 1));
    }

    #[test]
    fn a_near_tie_inside_the_floor_passes_and_is_counted() {
        // Both programs see ids 1 and 2 within 0.05; they pick differently.
        let r = row(3, &[(1, 10.02), (2, 10.0)], &[(1, 10.0), (2, 10.03)]);
        assert!(r.is_flip());
        assert!((r.gap_shift() - 0.05).abs() < 1e-4);
        let v = judge(&[r], Some(0.4), &[], 1);
        assert_eq!(v.findings, vec![(r, Clause::NearTie)]);
        assert!(v.passes());
        assert_eq!((v.flips, v.near_ties, v.bad()), (1, 1, 0));
    }

    #[test]
    fn a_flip_at_the_floor_passes_and_just_above_it_fails() {
        let r = Row {
            pos: 1,
            prefill_top1: 1,
            margin_p: 0.25,
            decode_top1: 2,
            margin_d: 0.25,
            spread: 0.3,
        };
        assert!(judge(&[r], Some(0.5), &[], 1).passes());
        assert!(!judge(&[r], Some(0.499_999), &[], 1).passes());
    }

    #[test]
    fn near_ties_above_the_budget_fail() {
        let tie = |pos| row(pos, &[(1, 10.02), (2, 10.0)], &[(1, 10.0), (2, 10.03)]);
        let rows = [tie(1), tie(2), tie(3)];
        let v = judge(&rows, Some(0.4), &[], 2);
        assert!(v.over_budget);
        assert_eq!((v.flips, v.near_ties, v.bad()), (3, 3, 1));
        assert!(judge(&rows, Some(0.4), &[], 3).passes());
        assert!(
            v.report(&|_| None)
                .contains("TOO-MANY-NEAR-TIES 3 > budget 2")
        );
    }

    #[test]
    fn a_wide_row_fails_without_a_flip() {
        // memra#1175 on main: both programs pick id 1, at margins 14.4 and 7.2.
        let r = row(5, &[(1, 25.0), (2, 10.6)], &[(1, 20.0), (2, 12.8)]);
        assert!(!r.is_flip());
        let v = judge(&[r], Some(1.5), &[], 1);
        assert_eq!(v.findings, vec![(r, Clause::WideRow)]);
        assert!(!v.passes());
        assert_eq!(v.flips, 0);
        // The same row inside a floor that covers it is not a finding.
        assert!(judge(&[r], Some(8.0), &[], 1).findings.is_empty());
    }

    #[test]
    fn without_a_floor_a_flip_fails_and_agreeing_rows_are_not_judged() {
        let tie = row(3, &[(1, 10.02), (2, 10.0)], &[(1, 10.0), (2, 10.03)]);
        let wide_row = row(5, &[(1, 25.0), (2, 10.6)], &[(1, 20.0), (2, 12.8)]);
        let v = judge(&[tie, wide_row], None, &[], 1);
        assert_eq!(v.findings, vec![(tie, Clause::UncalibratedFlip)]);
        assert!(!v.passes());
        let v = judge(&[wide_row], None, &[], 1);
        assert!(v.passes() && v.findings.is_empty());
        assert!(v.report(&|_| None).contains("floor=none"));
    }

    #[test]
    fn the_report_names_the_row_the_clause_and_the_floor() {
        let r = row(7, &[(1, 25.0), (2, 10.6)], &[(1, 9.5), (2, 12.0)]);
        let v = judge(&[r], Some(1.0), &[], 1);
        let text = v.report(&|pos| (pos == 7).then(|| "detail pos=7 ...".to_string()));
        let first = text.lines().next().unwrap();
        assert!(
            first.starts_with("WIDE-FLIP pos=7 prefill_top1=1 margin_p=14.4000 decode_top1=2 "),
            "{first}"
        );
        assert!(first.ends_with("> floor=1.0000"), "{first}");
        assert_eq!(text.lines().nth(1), Some("  detail pos=7 ..."));
        assert!(
            text.lines()
                .last()
                .unwrap()
                .starts_with("SUMMARY flips=1 bad=1 near_ties=0 wide_flips=1 wide_rows=0 "),
            "{text}"
        );
    }

    #[test]
    fn the_detail_line_holds_both_programs_top2_and_the_cross_logits() {
        let p = logits(&[(1, 25.0), (2, 10.5)]);
        let d = logits(&[(1, 9.5), (2, 12.0)]);
        let line = detail_line(7, &p, &d, &|id| format!("t{id}"));
        assert!(
            line.starts_with(
                "detail pos=7 prefill: top1=1 \"t1\" 25.0000 top2=2 \"t2\" 10.5000 margin=14.5000 \
                 at-decode-top1=10.5000 | decode: top1=2 \"t2\" 12.0000 top2=1 \"t1\" 9.5000 \
                 margin=2.5000 at-prefill-top1=9.5000 | spread=15.5000 gap_shift=17.0000"
            ),
            "{line}"
        );
    }

    #[test]
    fn the_table_round_trips_through_the_parser() {
        let rows = [
            row(10, &[(1, 10.02), (2, 10.0)], &[(1, 10.0), (2, 10.03)]),
            row(11, &[(1, 3.0e-5), (2, 1.0e-5)], &[(1, 5.0e-5), (2, 1.0e-5)]),
            row(
                12,
                &[(1, 1.234_567_8), (2, 0.5)],
                &[(1, 2.345_678_8), (2, 0.5)],
            ),
        ];
        let text: String = rows.iter().map(|r| r.table_line() + "\n").collect();
        let parsed = parse_table(&format!("header line\n{text}\ntrailer 1 2 3")).unwrap();
        assert_eq!(parsed.len(), 3);
        for (a, b) in rows.iter().zip(&parsed) {
            assert_eq!(a.margin_p.to_bits(), b.margin_p.to_bits());
            assert_eq!(a.margin_d.to_bits(), b.margin_d.to_bits());
            assert_eq!(a.spread.to_bits(), b.spread.to_bits());
            assert_eq!(a.gap_shift().to_bits(), b.gap_shift().to_bits());
        }
    }

    #[test]
    fn a_malformed_table_is_an_error() {
        assert!(parse_table("").is_err());
        assert!(parse_table("no rows here").is_err());
        let good = "10 1 1.0 1 1.0 0.1 yes";
        assert!(parse_table(good).is_ok());
        for bad in [
            "10 1 1.0 1 1.0 0.1 yes\n10 1 1.0 1 1.0 0.1 yes", // duplicate position
            "10 1 nan 1 1.0 0.1 yes",
            "10 1 inf 1 1.0 0.1 yes",
            "10 1 -1.0 1 1.0 0.1 yes",
            "10 1 1.0 1 1.0 0.1 maybe",
            "10 1 1.0 2 1.0 0.1 yes", // ids differ, column says agree
            "10 1 1.0 1 1.0 0.1 NO",  // ids agree, column says flip
            "10 1 1.0 x 1.0 0.1 yes",
        ] {
            assert!(parse_table(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn calibration_is_the_largest_agreeing_gap_shift_and_passes_its_own_table() {
        let rows = [
            row(1, &[(1, 12.0), (2, 10.0)], &[(1, 12.5), (2, 10.0)]), // 0.5
            row(2, &[(1, 12.0), (2, 10.0)], &[(1, 13.25), (2, 10.0)]), // 1.25
            row(3, &[(1, 10.02), (2, 10.0)], &[(1, 10.0), (2, 10.03)]), // a flip, not counted
            row(4, &[(1, 12.0), (2, 10.0)], &[(1, 11.25), (2, 10.0)]), // 0.75
        ];
        let cal = calibrate(&rows).unwrap();
        assert_eq!((cal.set_by_row, cal.agreeing, cal.flips), (2, 3, 1));
        assert_eq!(cal.floor.to_bits(), rows[1].gap_shift().to_bits());
        // Three agreeing rows, nearest rank: the median is the middle one, both high
        // percentiles are the largest. The flip (0.05) is inside the floor: no known row.
        assert_eq!((cal.p50, cal.p99, cal.p999), (0.75, cal.floor, cal.floor));
        assert!(cal.known.is_empty());
        // The floor passes the table it came from: the row that set it sits exactly on it.
        assert!(judge(&rows, Some(cal.floor), &cal.known, 1).passes());
        assert!(calibrate(&rows[2..3]).is_none());
        assert_eq!(max_flips_in_window(&rows, 2), 1);
        assert_eq!(
            max_flips_in_window(&[rows[2], rows[2], rows[0], rows[2]], 3),
            2
        );
    }

    /// A calibration table with one flip above its own floor (margins 14.4 and 2.5 at row 9),
    /// and the calibration it gives.
    fn table_with_a_known_row() -> (Vec<Row>, Calibration) {
        let rows = vec![
            row(1, &[(1, 12.0), (2, 10.0)], &[(1, 12.5), (2, 10.0)]),
            row(2, &[(1, 12.0), (2, 10.0)], &[(1, 13.25), (2, 10.0)]),
            row(9, &[(1, 25.0), (2, 10.6)], &[(1, 9.5), (2, 12.0)]),
        ];
        let cal = calibrate(&rows).unwrap();
        (rows, cal)
    }

    #[test]
    fn a_calibration_flip_above_its_floor_is_recorded_as_a_known_row() {
        let (rows, cal) = table_with_a_known_row();
        assert_eq!(cal.floor, 1.25);
        assert_eq!(cal.known.len(), 1);
        let k = cal.known[0];
        assert_eq!((k.pos, k.prefill_top1, k.decode_top1), (9, 1, 2));
        assert_eq!(k.gap_shift().to_bits(), rows[2].gap_shift().to_bits());
        // Without the known rows the same table fails on that flip: it is not silently inside
        // the floor.
        let v = judge(&rows, Some(cal.floor), &[], 1);
        assert_eq!(v.findings, vec![(rows[2], Clause::WideFlip)]);
    }

    #[test]
    fn a_known_row_no_wider_than_recorded_passes_and_is_named() {
        let (rows, cal) = table_with_a_known_row();
        // Main's own table: the known row sits exactly on its recorded shift.
        let v = judge(&rows, Some(cal.floor), &cal.known, 1);
        assert_eq!(v.findings, vec![(rows[2], Clause::KnownRow)]);
        assert!(v.passes());
        // Named, not counted as a near tie, and it does not use the budget.
        assert_eq!((v.flips, v.near_ties, v.bad()), (1, 0, 0));
        assert!(judge(&rows, Some(cal.floor), &cal.known, 0).passes());
        let text = v.report(&|_| None);
        let first = text.lines().next().unwrap();
        assert!(
            first.starts_with(
                "KNOWN-ROW pos=9 prefill_top1=1 margin_p=14.4000 decode_top1=2 margin_d=2.5000 "
            ),
            "{first}"
        );
        assert!(
            first.ends_with(
                "gap_shift=16.9000 <= recorded=16.9000 (calibration: prefill_top1=1 \
                 margin_p=14.4000 decode_top1=2 margin_d=2.5000; memra#1203) floor=1.2500"
            ),
            "{first}"
        );
        assert!(text.contains(" known_rows=1 known_rows_wider=0 "), "{text}");
        // A candidate that narrows the row, or stops flipping there, is named and passes too.
        let narrower = row(9, &[(1, 25.0), (2, 10.6)], &[(1, 11.0), (2, 12.0)]);
        let agreeing = row(9, &[(1, 25.0), (2, 10.6)], &[(1, 20.0), (2, 12.8)]);
        for r in [narrower, agreeing] {
            let v = judge(&[rows[0], rows[1], r], Some(cal.floor), &cal.known, 1);
            assert_eq!(v.findings, vec![(r, Clause::KnownRow)], "{r:?}");
            assert!(v.passes());
        }
    }

    #[test]
    fn a_known_row_wider_than_recorded_fails() {
        let (rows, cal) = table_with_a_known_row();
        // The candidate holds decode's token by 2.6 where main held it by 2.5.
        let wider = row(9, &[(1, 25.0), (2, 10.6)], &[(1, 9.5), (2, 12.1)]);
        assert!(wider.gap_shift() > cal.known[0].gap_shift());
        let v = judge(&[rows[0], rows[1], wider], Some(cal.floor), &cal.known, 1);
        assert_eq!(v.findings, vec![(wider, Clause::KnownRowWider)]);
        assert!(!v.passes());
        let text = v.report(&|_| None);
        assert!(text.starts_with("KNOWN-ROW-WIDER pos=9 "), "{text}");
        assert!(text.contains(" > recorded=16.9000 "), "{text}");
        assert!(
            text.contains(" bad=1 ") && text.contains(" known_rows_wider=1 "),
            "{text}"
        );
        // A known row covers its own position only: the same flip one row later is a wide flip.
        let elsewhere = row(10, &[(1, 25.0), (2, 10.6)], &[(1, 9.5), (2, 12.0)]);
        let v = judge(&[elsewhere], Some(cal.floor), &cal.known, 1);
        assert_eq!(v.findings, vec![(elsewhere, Clause::WideFlip)]);
    }

    #[test]
    fn the_floors_table_has_one_exact_row_per_key_and_no_fallback() {
        let key = FloorKey {
            model: "m",
            prompt: "board-2048",
            prompt_sha256: "abc",
            decode: "eager",
            device: "device A",
        };
        let cal = Calibration {
            floor: 0.408_123_45,
            set_by_row: 2047,
            agreeing: 2040,
            flips: 5,
            p50: 0.05,
            p99: 0.21,
            p999: 0.33,
            known: vec![KnownRow {
                pos: 41,
                prefill_top1: 18,
                margin_p: 1.472,
                decode_top1: 15,
                margin_d: 2.899,
            }],
        };
        let line = floor_line(&key, &cal, "research/x/gate.log");
        let table = format!("# comment\n\n{line}\n");
        let rows = parse_floors(&table).unwrap();
        assert_eq!(rows.len(), 1);
        let found = find_floor(&rows, &key).unwrap();
        assert_eq!(found.floor.to_bits(), cal.floor.to_bits());
        assert_eq!((found.rows, found.set_by_row), (2040, 2047));
        assert_eq!((found.p50, found.p99, found.p999), (0.05, 0.21, 0.33));
        // The known row survives the table bit for bit, so its recorded shift does too.
        assert_eq!(found.known, cal.known);
        assert_eq!(
            found.describe(),
            "0.4081 (largest gap shift over 2040 agreeing rows, set by row 2047; p50 0.0500, \
             p99 0.2100, p99.9 0.3300; known rows (calibration flips above it, memra#1203): 41 \
             (shift 4.3710); research/x/gate.log)"
        );
        for other in [
            FloorKey {
                model: "m2",
                ..key.clone()
            },
            FloorKey {
                prompt: "p4-16k",
                ..key.clone()
            },
            FloorKey {
                prompt_sha256: "abd",
                ..key.clone()
            },
            FloorKey {
                decode: "batch",
                ..key.clone()
            },
            FloorKey {
                device: "device B",
                ..key.clone()
            },
        ] {
            assert!(find_floor(&rows, &other).is_none(), "{other:?}");
        }
        assert!(parse_floors(&format!("{line}\n{line}\n")).is_err());
        let good = "m\tp\ts\teager\td\t1.0\t5\t3\t0.25\t0.5\t0.75\t-\tr";
        assert_eq!(
            parse_floors(good).unwrap()[0].describe(),
            "1.0000 (largest gap shift over 5 agreeing rows, set by row 3; p50 0.2500, p99 \
             0.5000, p99.9 0.7500; no calibration flip above it; r)"
        );
        for bad in [
            "m\tp\ts\teager\td\t-1.0\t5\t3\t0.25\t0.5\t0.75\t-\tr", // negative floor
            "m\tp\ts\teager\td\tnan\t5\t3\t0.25\t0.5\t0.75\t-\tr",
            "m\tp\ts\tgraph\td\t1.0\t5\t3\t0.25\t0.5\t0.75\t-\tr", // unknown decode program
            "m\tp\ts\teager\td\t1.0\t5\t3\t0.25\t0.5\t0.75\t-",    // 12 fields
            "m\tp\ts\teager\td\t1.0\t5\t3\tr",                     // the first 9-field form
            "m\tp\ts\teager\td\t1.0\t5\t3\t0.6\t0.5\t0.75\t-\tr",  // p50 above p99
            "m\tp\ts\teager\td\t1.0\t5\t3\t0.25\t0.5\t1.5\t-\tr",  // p99.9 above the floor
            "m\tp\ts\teager\td\t1.0\t5\t3\t0.25\t0.5\t0.75\t41\tr", // a bare position
            "m\tp\ts\teager\td\t1.0\t5\t3\t0.25\t0.5\t0.75\t41:1:0.2:2:0.3\tr", // inside the floor
            "m\tp\ts\teager\td\t1.0\t5\t3\t0.25\t0.5\t0.75\t41:1:2.0:2:3.0;41:1:2.0:2:3.0\tr",
            "m\tp\ts\teager\td\t1.0\t5\t3\t0.25\t0.5\t0.75\t41:1:x:2:3.0\tr",
        ] {
            assert!(parse_floors(bad).is_err(), "{bad}");
        }
    }

    /// The tree's floors table, read with the judge's own parser, and every row against its
    /// receipt: the row must be exactly the `--calibrate` line of the receipt's table, so a hand
    /// edit, a stale row or a receipt that is not the one measured fails on CPU (memra#1199).
    #[test]
    fn the_tree_floors_table_parses_and_every_row_is_its_receipts_calibration() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let text = std::fs::read_to_string(root.join("tools/argmax-margin-floors.tsv"))
            .expect("tools/argmax-margin-floors.tsv");
        let rows = parse_floors(&text).expect("the tree's floors table parses");
        assert!(!rows.is_empty());
        for line in text
            .lines()
            .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
        {
            let f: Vec<&str> = line.split('\t').collect();
            let receipt = f[12];
            // (tools/check-argmax-floors.py requires the receipt directory; this only keeps the
            // read inside the tree.)
            assert!(
                !receipt.starts_with('/') && !receipt.split('/').any(|c| c == ".."),
                "receipt {receipt:?} is not a path in the tree"
            );
            let log = std::fs::read_to_string(root.join(receipt))
                .unwrap_or_else(|e| panic!("receipt {receipt}: {e}"));
            let table = parse_table(&log).unwrap_or_else(|e| panic!("receipt {receipt}: {e}"));
            let cal = calibrate(&table).unwrap_or_else(|| panic!("receipt {receipt}: no floor"));
            let key = FloorKey {
                model: f[0],
                prompt: f[1],
                prompt_sha256: f[2],
                decode: f[3],
                device: f[4],
            };
            assert_eq!(floor_line(&key, &cal, receipt), line, "{receipt}");
        }
    }

    // ---- stored rows (memra#1204) ----

    fn stored_key() -> StoredKey {
        StoredKey {
            model: "m".into(),
            model_sha256: "aa".into(),
            prompt: "board-2048".into(),
            prompt_sha256: "bb".into(),
            decode: "eager".into(),
            device: "device A".into(),
            window: 3,
            experts: "resident".into(),
        }
    }

    /// Main's window: two agreeing rows and a near tie.
    fn main_rows() -> Vec<Row> {
        vec![
            row(10, &[(1, 12.0), (2, 10.0)], &[(1, 12.5), (2, 10.0)]),
            row(11, &[(1, 14.0), (2, 10.0)], &[(1, 14.25), (2, 10.0)]),
            row(12, &[(1, 10.02), (2, 10.0)], &[(1, 10.0), (2, 10.03)]),
        ]
    }

    fn lines(rows: &[Row]) -> Vec<String> {
        rows.iter().map(|r| r.table_line()).collect()
    }

    fn stored(rows: &[Row]) -> StoredTable {
        StoredTable {
            key: stored_key(),
            commit: "031f146690".into(),
            receipt: "research/x/probe.log".into(),
            rows: lines(rows),
        }
    }

    #[test]
    fn a_stored_table_round_trips_through_its_file_form() {
        let table = stored(&main_rows());
        let text = table.render();
        assert_eq!(StoredTable::parse(&text).unwrap(), table);
        // The rows are the probe's own lines: a log's table read back equals them.
        let log = format!("header\n{}\ntrailer", table.rows.join("\n"));
        assert_eq!(table_lines(&log).unwrap(), table.rows);
        for (bad, why) in [
            (
                text.replace("window\t3", "window\t4"),
                "row count is not the window",
            ),
            (text.replace("model_sha256\taa\n", ""), "a missing field"),
            (
                text.replace("decode\teager", "decode\tgraph"),
                "an unknown decode program",
            ),
            (
                text.replace("commit\t", "commit\t\ncommit\t"),
                "an empty field",
            ),
            (
                text.replace("prompt\tboard-2048", "prompt\tboard-2048\nprompt\tp4"),
                "a repeated field",
            ),
            (
                text.replace("device\t", "unit\tx\ndevice\t"),
                "an unknown field",
            ),
            (text.replace("rows\n", ""), "no rows line"),
            (
                text.replace(" yes", " maybe"),
                "rows that are not a probe table",
            ),
        ] {
            assert!(StoredTable::parse(&bad).is_err(), "{why}");
        }
    }

    #[test]
    fn an_identical_table_has_no_difference() {
        let table = stored(&main_rows());
        assert!(compare_rows(&table.rows, &lines(&main_rows())).is_empty());
        assert_eq!(rows_report(&[], &|_| None), "");
    }

    #[test]
    fn a_one_logit_move_passes_the_floor_and_fails_the_stored_rows() {
        // The red arm of memra#1204. The candidate moves decode's margin at row 11 by one
        // logit (14.25 to 13.25 over the runner-up). The floor rule sees an agreeing row with
        // a gap shift of 0.75 under a floor of 3.7 and passes. The stored rows do not.
        let table = stored(&main_rows());
        let mut candidate = main_rows();
        candidate[1] = row(11, &[(1, 14.0), (2, 10.0)], &[(1, 13.25), (2, 10.0)]);
        assert!(judge(&main_rows(), Some(3.7), &[], 1).passes());
        assert!(judge(&candidate, Some(3.7), &[], 1).passes());
        let diffs = compare_rows(&table.rows, &lines(&candidate));
        assert_eq!(diffs.len(), 1);
        assert_eq!(diffs[0].pos, 11);
        let text = rows_report(&diffs, &|pos| {
            (pos == 11).then(|| "detail pos=11 ...".to_string())
        });
        assert_eq!(
            text,
            "ROW-DIFF pos=11\n\
             \x20 stored (main): prefill_top1=1 margin_p=4.000000000e0 decode_top1=1 margin_d=4.250000000e0 spread=2.500000000e-1\n\
             \x20 candidate:     prefill_top1=1 margin_p=4.000000000e0 decode_top1=1 margin_d=3.250000000e0 spread=7.500000000e-1\n\
             \x20 detail pos=11 ...\n"
        );
    }

    #[test]
    fn a_token_change_at_a_margin_under_the_floor_passes_the_floor_and_fails_the_stored_rows() {
        // Row 10: main's two programs agree on token 1; the candidate's decode picks token 2
        // by 0.5. A flip with a margin sum of 2.5 is a near tie under a floor of 3.7.
        let table = stored(&main_rows());
        let mut candidate = main_rows();
        candidate[0] = row(10, &[(1, 12.0), (2, 10.0)], &[(1, 10.0), (2, 10.5)]);
        assert!(candidate[0].is_flip());
        // Two near ties now (rows 10 and 12): inside a budget of 2, and each inside the floor.
        assert!(judge(&candidate, Some(3.7), &[], 2).passes());
        let diffs = compare_rows(&table.rows, &lines(&candidate));
        assert_eq!(diffs.iter().map(|d| d.pos).collect::<Vec<_>>(), vec![10]);
        let text = rows_report(&diffs, &|_| None);
        assert!(
            text.contains("stored (main): prefill_top1=1 margin_p=2.000000000e0 decode_top1=1 "),
            "{text}"
        );
        assert!(text.contains("candidate:     prefill_top1=1 margin_p=2.000000000e0 decode_top1=2 margin_d=5.000000000e-1"), "{text}");
        assert!(text.contains(" FLIP\n"), "{text}");
    }

    #[test]
    fn the_smallest_float_step_in_one_margin_is_a_difference() {
        // No tolerance: one unit in the last place of one margin differs.
        let table = stored(&main_rows());
        let mut candidate = main_rows();
        candidate[1].margin_d = f32::from_bits(candidate[1].margin_d.to_bits() + 1);
        assert_eq!(compare_rows(&table.rows, &lines(&candidate)).len(), 1);
    }

    #[test]
    fn a_missing_or_extra_position_is_a_difference() {
        let table = stored(&main_rows());
        let mut shifted = lines(&main_rows());
        shifted.remove(0);
        shifted.push(row(13, &[(1, 12.0), (2, 10.0)], &[(1, 12.5), (2, 10.0)]).table_line());
        let diffs = compare_rows(&table.rows, &shifted);
        assert_eq!(
            diffs.iter().map(|d| d.pos).collect::<Vec<_>>(),
            vec![10, 13]
        );
        assert!(diffs[0].candidate.is_none() && diffs[1].stored.is_none());
        let text = rows_report(&diffs, &|_| None);
        assert!(
            text.contains("candidate:     no row at this position"),
            "{text}"
        );
        assert!(
            text.contains("stored (main): no row at this position"),
            "{text}"
        );
    }

    #[test]
    fn the_expert_placement_is_read_from_the_logs_decision_lines() {
        let resident = "[moe] resident-experts decision (dev0): experts 18.57GB + trunk 1.60GB vs free 24.52GB (expert budget 20.91GB) -> RESIDENT";
        let slru = "[moe] resident-experts decision (dev0): experts 18.57GB + trunk 1.60GB vs free 14.00GB (expert budget 10.40GB) -> SLRU cache";
        assert_eq!(experts_of("table\n"), "none");
        assert_eq!(experts_of(&format!("{resident}\ntable\n")), "resident");
        assert_eq!(experts_of(&format!("{slru}\n")), "slru");
        // Two devices, one of them off the card: not resident.
        assert_eq!(experts_of(&format!("{resident}\n{slru}\n")), "slru");
    }

    #[test]
    fn a_stored_table_is_found_only_by_its_whole_key() {
        let tables = vec![stored(&main_rows())];
        let key = stored_key();
        assert!(find_stored(&tables, &key).unwrap().is_some());
        // Other prompt bytes, another artifact, device, decode program or window: no stored
        // table, not a difference.
        for other in [
            StoredKey {
                prompt_sha256: "cc".into(),
                ..key.clone()
            },
            StoredKey {
                model_sha256: "dd".into(),
                ..key.clone()
            },
            StoredKey {
                model: "m2".into(),
                ..key.clone()
            },
            StoredKey {
                prompt: "p4-16k".into(),
                ..key.clone()
            },
            StoredKey {
                device: "device B".into(),
                ..key.clone()
            },
            StoredKey {
                decode: "batch".into(),
                ..key.clone()
            },
            StoredKey {
                experts: "slru".into(),
                ..key.clone()
            },
            StoredKey {
                window: 12,
                ..key.clone()
            },
        ] {
            assert!(find_stored(&tables, &other).unwrap().is_none(), "{other:?}");
        }
        let twice = vec![stored(&main_rows()), stored(&main_rows())];
        assert!(find_stored(&twice, &key).is_err());
    }
}
