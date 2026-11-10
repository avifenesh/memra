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

/// What a wide window measured: the floor it sets.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Calibration {
    /// Largest gap shift at an agreeing row.
    pub floor: f32,
    /// The row that set it.
    pub set_by_row: usize,
    /// Agreeing rows measured.
    pub agreeing: usize,
    /// Flips in the window (not part of the floor).
    pub flips: usize,
}

/// The floor a table sets, or None when no row agrees.
pub fn calibrate(rows: &[Row]) -> Option<Calibration> {
    let mut best: Option<(f32, usize)> = None;
    let mut agreeing = 0usize;
    for r in rows.iter().filter(|r| !r.is_flip()) {
        agreeing += 1;
        if best.is_none_or(|(s, _)| r.gap_shift() > s) {
            best = Some((r.gap_shift(), r.pos));
        }
    }
    best.map(|(floor, set_by_row)| Calibration {
        floor,
        set_by_row,
        agreeing,
        flips: rows.len() - agreeing,
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
}

impl Clause {
    pub fn name(self) -> &'static str {
        match self {
            Clause::NearTie => "near-tie",
            Clause::WideFlip => "WIDE-FLIP",
            Clause::WideRow => "WIDE-ROW",
            Clause::UncalibratedFlip => "UNCALIBRATED-FLIP",
        }
    }

    pub fn fails(self) -> bool {
        !matches!(self, Clause::NearTie)
    }
}

/// The gate's verdict on one table.
#[derive(Debug, Clone, PartialEq)]
pub struct Verdict {
    pub floor: Option<f32>,
    pub budget: usize,
    pub flips: usize,
    pub near_ties: usize,
    /// Every flip and every wide row, in table order.
    pub findings: Vec<(Row, Clause)>,
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
            let relation = match clause {
                Clause::NearTie => "<=",
                Clause::WideFlip | Clause::WideRow => ">",
                Clause::UncalibratedFlip => "vs",
            };
            let _ = writeln!(
                out,
                "{} pos={} prefill_top1={} margin_p={:.4} decode_top1={} margin_d={:.4} \
                 spread={:.4} gap_shift={:.4} {relation} floor={floor}",
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
             floor={floor} budget={}",
            self.flips,
            self.bad(),
            self.near_ties,
            count(Clause::WideFlip),
            count(Clause::WideRow),
            count(Clause::UncalibratedFlip),
            self.budget,
        );
        out
    }
}

/// Judge a table against a floor and a near-tie budget. This is the gate's only verdict
/// function: the probe's `--judge` mode calls it on the table of a log.
pub fn judge(rows: &[Row], floor: Option<f32>, budget: usize) -> Verdict {
    let mut findings = Vec::new();
    let (mut flips, mut near_ties) = (0usize, 0usize);
    for r in rows {
        let clause = match (r.is_flip(), floor) {
            (true, None) => Some(Clause::UncalibratedFlip),
            (true, Some(f)) if r.gap_shift() > f => Some(Clause::WideFlip),
            (true, Some(_)) => Some(Clause::NearTie),
            (false, Some(f)) if r.gap_shift() > f => Some(Clause::WideRow),
            (false, _) => None,
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
    /// The calibration receipt, a path in this repo.
    pub receipt: String,
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

/// Parse the floors table: tab-separated, `#` comments. A malformed row, a floor that is
/// negative or not finite, or two rows with one key is an error.
pub fn parse_floors(text: &str) -> Result<Vec<FloorRow>, String> {
    let mut rows: Vec<FloorRow> = Vec::new();
    for (n, line) in text.lines().enumerate() {
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() != 9 || f.iter().any(|s| s.trim().is_empty()) {
            return Err(format!(
                "floors line {}: expected 9 tab-separated fields",
                n + 1
            ));
        }
        let floor: f32 = f[5]
            .parse()
            .map_err(|_| format!("floors line {}: floor {:?}", n + 1, f[5]))?;
        if !floor.is_finite() || floor < 0.0 {
            return Err(format!(
                "floors line {}: floor {:?} is negative or not finite",
                n + 1,
                f[5]
            ));
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
            receipt: f[8].into(),
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
    format!(
        "{}\t{}\t{}\t{}\t{}\t{:.9e}\t{}\t{}\t{}",
        key.model,
        key.prompt,
        key.prompt_sha256,
        key.decode,
        key.device,
        cal.floor,
        cal.agreeing,
        cal.set_by_row,
        receipt
    )
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
        let v = judge(&[r], Some(1.0), 1);
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
        let v = judge(&[r], Some(0.4), 1);
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
        assert!(judge(&[r], Some(0.5), 1).passes());
        assert!(!judge(&[r], Some(0.499_999), 1).passes());
    }

    #[test]
    fn near_ties_above_the_budget_fail() {
        let tie = |pos| row(pos, &[(1, 10.02), (2, 10.0)], &[(1, 10.0), (2, 10.03)]);
        let rows = [tie(1), tie(2), tie(3)];
        let v = judge(&rows, Some(0.4), 2);
        assert!(v.over_budget);
        assert_eq!((v.flips, v.near_ties, v.bad()), (3, 3, 1));
        assert!(judge(&rows, Some(0.4), 3).passes());
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
        let v = judge(&[r], Some(1.5), 1);
        assert_eq!(v.findings, vec![(r, Clause::WideRow)]);
        assert!(!v.passes());
        assert_eq!(v.flips, 0);
        // The same row inside a floor that covers it is not a finding.
        assert!(judge(&[r], Some(8.0), 1).findings.is_empty());
    }

    #[test]
    fn without_a_floor_a_flip_fails_and_agreeing_rows_are_not_judged() {
        let tie = row(3, &[(1, 10.02), (2, 10.0)], &[(1, 10.0), (2, 10.03)]);
        let wide_row = row(5, &[(1, 25.0), (2, 10.6)], &[(1, 20.0), (2, 12.8)]);
        let v = judge(&[tie, wide_row], None, 1);
        assert_eq!(v.findings, vec![(tie, Clause::UncalibratedFlip)]);
        assert!(!v.passes());
        let v = judge(&[wide_row], None, 1);
        assert!(v.passes() && v.findings.is_empty());
        assert!(v.report(&|_| None).contains("floor=none"));
    }

    #[test]
    fn the_report_names_the_row_the_clause_and_the_floor() {
        let r = row(7, &[(1, 25.0), (2, 10.6)], &[(1, 9.5), (2, 12.0)]);
        let v = judge(&[r], Some(1.0), 1);
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
        // The floor passes the table it came from: the row that set it sits exactly on it.
        assert!(judge(&rows, Some(cal.floor), 1).passes());
        assert!(calibrate(&rows[2..3]).is_none());
        assert_eq!(max_flips_in_window(&rows, 2), 1);
        assert_eq!(
            max_flips_in_window(&[rows[2], rows[2], rows[0], rows[2]], 3),
            2
        );
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
        };
        let line = floor_line(&key, &cal, "research/x/gate.log");
        let table = format!("# comment\n\n{line}\n");
        let rows = parse_floors(&table).unwrap();
        assert_eq!(rows.len(), 1);
        let found = find_floor(&rows, &key).unwrap();
        assert_eq!(found.floor.to_bits(), cal.floor.to_bits());
        assert_eq!((found.rows, found.set_by_row), (2040, 2047));
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
        assert!(parse_floors("m\tp\ts\teager\td\t-1.0\t5\t3\tr").is_err());
        assert!(parse_floors("m\tp\ts\teager\td\tnan\t5\t3\tr").is_err());
        assert!(parse_floors("m\tp\ts\tgraph\td\t1.0\t5\t3\tr").is_err());
        assert!(parse_floors("m\tp\ts\teager\td\t1.0\t5\t3").is_err());
    }
}
