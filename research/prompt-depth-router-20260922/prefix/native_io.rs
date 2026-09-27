use super::prefix_policy;
use super::router::Kind;
use memra_tokenizer::Tokenizer;
use memra_tokenizer::chat::{ThinkMode, Turn};
use std::io::{self, BufRead, Write};
use std::path::Path;
use std::time::Instant;

const MAX_HEADER_TOKENS: usize = 256;

pub fn render(tok: &Tokenizer, user: &str) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let text = tok
        .apply_chat_template_tools(
            &[Turn {
                role: "user".into(),
                content: user.into(),
                ..Default::default()
            }],
            true,
            &[],
            ThinkMode::Default,
            None,
        )
        .map_err(|error| format!("chat template: {error}"))?;
    Ok(tok.encode(&text, true))
}

pub struct TemplateShape {
    header: Vec<u8>,
    exact_header: Option<Vec<u32>>,
    suffix: Vec<u32>,
}

#[derive(Clone, Copy)]
pub struct UserSpan {
    pub start: usize,
    pub end: usize,
    pub leading_skip: usize,
    pub header_tokens: usize,
}

impl TemplateShape {
    pub fn new(tok: &Tokenizer) -> Result<Self, Box<dyn std::error::Error>> {
        const SENTINEL: &str = "__PREFIX_USER_BOUNDARY_9b71__";
        let ids = render(tok, SENTINEL)?;
        let decoded = tok.decode_bytes_special(&ids, true);
        let positions: Vec<_> = decoded
            .windows(SENTINEL.len())
            .enumerate()
            .filter(|(_, window)| *window == SENTINEL.as_bytes())
            .map(|(position, _)| position)
            .collect();
        if positions.len() != 1 {
            return Err("template does not expose one unique user span".into());
        }
        let start = positions[0];
        let end = start + SENTINEL.len();
        let mut offset = 0;
        let mut header_boundary = None;
        let mut suffix_start = None;
        for (index, id) in ids.iter().enumerate() {
            if offset == start {
                header_boundary = Some(index);
            }
            if offset == end {
                suffix_start = Some(index);
                break;
            }
            offset += tok.decode_bytes_special(&[*id], true).len();
        }
        let boundary = suffix_start.ok_or("user/template suffix is not a token boundary")?;
        if !tok.token_is_special(ids[boundary]) {
            return Err("user span must end at a separate template special token".into());
        }
        Ok(Self {
            header: decoded[..start].to_vec(),
            exact_header: header_boundary.map(|boundary| ids[..boundary].to_vec()),
            suffix: ids[boundary..].to_vec(),
        })
    }

    /// Decode only the short template header. The suffix is checked by its fixed
    /// token IDs; no user-tail text is decoded or sent to the forecaster.
    pub fn locate(&self, tok: &Tokenizer, input: &[u32]) -> Result<UserSpan, &'static str> {
        if !input.ends_with(&self.suffix) {
            return Err("rendered template suffix changed");
        }
        let end = input.len() - self.suffix.len();
        if let Some(header) = &self.exact_header
            && input.starts_with(header)
            && header.len() <= end
        {
            return Ok(UserSpan {
                start: header.len(),
                end,
                leading_skip: 0,
                header_tokens: header.len(),
            });
        }
        let mut matched = 0;
        for (index, id) in input[..end].iter().take(MAX_HEADER_TOKENS).enumerate() {
            if matched == self.header.len() {
                return Ok(UserSpan {
                    start: index,
                    end,
                    leading_skip: 0,
                    header_tokens: index,
                });
            }
            let piece = tok.decode_bytes_special(&[*id], true);
            let take = piece.len().min(self.header.len() - matched);
            if piece[..take] != self.header[matched..matched + take] {
                return Err("rendered template header changed");
            }
            matched += take;
            if take < piece.len() {
                return Ok(UserSpan {
                    start: index,
                    end,
                    leading_skip: take,
                    header_tokens: index + 1,
                });
            }
            if matched == self.header.len() {
                return Ok(UserSpan {
                    start: index + 1,
                    end,
                    leading_skip: 0,
                    header_tokens: index + 1,
                });
            }
        }
        Err("user start was not found within the bounded template header")
    }
}

#[derive(Clone, Copy)]
pub enum Routing {
    Prefix(usize),
    Schedule([u8; 8]),
}

pub struct Record {
    pub source: &'static str,
    pub kind: Kind,
    pub k: u8,
    pub budget: usize,
    pub tokens_read: usize,
    pub decoded_bytes: usize,
    pub inspected_bytes: usize,
    pub core_ns: u64,
    pub elapsed_ns: u64,
    pub prefix: Vec<u8>,
    pub span: Option<UserSpan>,
}

impl Routing {
    pub fn parse(arm: &str) -> Result<Option<Self>, &'static str> {
        if let Some(value) = arm.strip_prefix("prefix:") {
            let budget = value
                .parse::<usize>()
                .map_err(|_| "invalid prefix budget")?;
            if budget == 0 || budget > prefix_policy::MAX_PREFIX_TOKENS {
                return Err("prefix budget exceeds the bound");
            }
            return Ok(Some(Self::Prefix(budget)));
        }
        if let Some(value) = arm.strip_prefix("schedule:") {
            let values: Vec<_> = value
                .split(',')
                .map(str::parse::<u8>)
                .collect::<Result<_, _>>()
                .map_err(|_| "invalid K schedule")?;
            let schedule: [u8; 8] = values
                .try_into()
                .map_err(|_| "K schedule needs eight entries")?;
            if schedule.iter().any(|&k| !(2..=4).contains(&k)) {
                return Err("scheduled K must be 2, 3 or 4");
            }
            return Ok(Some(Self::Schedule(schedule)));
        }
        if !matches!(arm, "fixed:2" | "fixed:3" | "fixed:4") {
            return Err("prefix study accepts fixed:2/3/4, prefix:X or schedule:K,...");
        }
        Ok(None)
    }

    pub fn initial_k(self) -> u8 {
        3
    }

    pub fn select(
        self,
        turn: usize,
        tok: &Tokenizer,
        prompt: &[u32],
        shape: &TemplateShape,
    ) -> Result<Record, &'static str> {
        let started = Instant::now();
        match self {
            Self::Schedule(values) => Ok(Record {
                source: "schedule",
                kind: Kind::Unknown,
                k: *values
                    .get(turn.checked_sub(1).ok_or("turn index is zero")?)
                    .ok_or("turn exceeds schedule")?,
                budget: 0,
                tokens_read: 0,
                decoded_bytes: 0,
                inspected_bytes: 0,
                core_ns: 0,
                elapsed_ns: started.elapsed().as_nanos() as u64,
                prefix: Vec::new(),
                span: None,
            }),
            Self::Prefix(budget) => {
                let span = shape.locate(tok, prompt)?;
                let selected = prefix_policy::select(
                    &prompt[span.start..span.end],
                    span.leading_skip,
                    budget,
                    |ids| tok.decode_bytes_special(ids, false),
                )?;
                Ok(Record {
                    source: "prefix",
                    kind: selected.decision.kind,
                    k: selected.decision.k,
                    budget,
                    tokens_read: selected.token_count,
                    decoded_bytes: selected.decoded_bytes,
                    inspected_bytes: selected.inspected_bytes,
                    core_ns: selected.elapsed_ns,
                    elapsed_ns: started.elapsed().as_nanos() as u64,
                    prefix: selected.prefix,
                    span: Some(span),
                })
            }
        }
    }
}

pub struct Recorder<'a> {
    pub log: &'a mut std::fs::File,
    pub out: &'a Path,
}

impl Recorder<'_> {
    pub fn write(
        &mut self,
        turn: usize,
        selected: Option<&Record>,
        tok: &Tokenizer,
        prompt: &[u32],
        shape: &TemplateShape,
        fixed_k: usize,
    ) -> io::Result<()> {
        let log = &mut self.log;
        let out = self.out;
        let span = match selected.and_then(|row| row.span) {
            Some(span) => span,
            None => shape.locate(tok, prompt).map_err(io::Error::other)?,
        };
        if let Some(row) = selected {
            writeln!(
                log,
                "{turn}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                row.source,
                row.budget,
                row.kind.name(),
                row.k,
                row.tokens_read,
                span.end - span.start,
                prompt.len(),
                row.decoded_bytes,
                row.inspected_bytes,
                span.header_tokens,
                row.core_ns,
                row.elapsed_ns
            )?;
            if row.source == "prefix" {
                let mut ids =
                    std::fs::File::create_new(out.join(format!("turn-{turn}.prefix.ids")))?;
                let mut pieces =
                    std::fs::File::create_new(out.join(format!("turn-{turn}.prefix-bytes.tsv")))?;
                writeln!(pieces, "id\thex")?;
                for &id in &prompt[span.start..span.start + row.tokens_read] {
                    writeln!(ids, "{id}")?;
                    write!(pieces, "{id}\t")?;
                    for byte in tok.decode_bytes_special(&[id], false) {
                        write!(pieces, "{byte:02x}")?;
                    }
                    writeln!(pieces)?;
                }
                std::fs::write(out.join(format!("turn-{turn}.prefix.bin")), &row.prefix)?;
                std::fs::write(
                    out.join(format!("turn-{turn}.prefix-span.tsv")),
                    format!(
                        "start\tend\tleading_skip\n{}\t{}\t{}\n",
                        span.start, span.end, span.leading_skip
                    ),
                )?;
            }
        } else {
            writeln!(
                log,
                "{turn}\tfixed\t0\tnot_run\t{fixed_k}\t0\t{}\t{}\t0\t0\t{}\t0\t0",
                span.end - span.start,
                prompt.len(),
                span.header_tokens
            )?;
        }
        log.flush()
    }
}

pub fn count_server(tok: &Tokenizer) -> Result<(), Box<dyn std::error::Error>> {
    let shape = TemplateShape::new(tok)?;
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout().lock());
    for line in stdin.lock().lines() {
        let encoded = line?;
        if encoded.len() > 8 * 1024 * 1024 || !encoded.len().is_multiple_of(2) {
            return Err("invalid preparation request size".into());
        }
        let bytes = (0..encoded.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&encoded[i..i + 2], 16))
            .collect::<Result<Vec<_>, _>>()?;
        let user = std::str::from_utf8(&bytes)?.trim();
        let prompt = render(tok, user)?;
        let span = shape.locate(tok, &prompt)?;
        write!(stdout, "{}\t{}", prompt.len(), span.end - span.start)?;
        for budget in [64, 128, 256] {
            let selected = Routing::Prefix(budget).select(1, tok, &prompt, &shape)?;
            write!(
                stdout,
                "\t{}\t{}\t{}\t{}\t{}",
                selected.kind.name(),
                selected.k,
                selected.tokens_read,
                selected.prefix.len(),
                selected.elapsed_ns
            )?;
        }
        writeln!(stdout)?;
        stdout.flush()?;
    }
    Ok(())
}
