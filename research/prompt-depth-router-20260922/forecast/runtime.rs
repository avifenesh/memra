use std::hint::black_box;
use std::io::{self, Write};
use std::time::Instant;

const INTRO: &[&[u8]] = &[
    b"code",
    b"python",
    b"rust",
    b"json",
    b"sql",
    b"implementation",
    b"function",
    b"script",
];
const KEYWORDS: &[&[u8]] = &[
    b"def",
    b"class",
    b"return",
    b"import",
    b"fn",
    b"let",
    b"const",
    b"function",
    b"select",
];

#[derive(Clone)]
struct State {
    hint: i32,
    bytes: [u8; 128],
    next: usize,
    len: usize,
    seen: i32,
    fenced: bool,
    ticks: u8,
}

impl State {
    fn new(hint: i32) -> Self {
        Self {
            hint,
            bytes: [0; 128],
            next: 0,
            len: 0,
            seen: 0,
            fenced: false,
            ticks: 0,
        }
    }

    fn observe(&mut self, input: &[u8]) {
        for &byte in input {
            self.seen = (self.seen + 1).min(512);
            self.ticks = if byte == b'`' {
                (self.ticks + 1).min(4)
            } else {
                0
            };
            if self.ticks == 3 {
                self.fenced = !self.fenced;
            }
            self.bytes[self.next] = byte;
            self.next = (self.next + 1) % 128;
            self.len = (self.len + 1).min(128);
        }
    }

    fn features(&self) -> [i32; 16] {
        let mut window = [0_u8; 128];
        for (i, byte) in window.iter_mut().take(self.len).enumerate() {
            *byte = self.bytes[(self.next + 128 - self.len + i) % 128].to_ascii_lowercase();
        }
        let window = &window[..self.len];
        let mut result = [0; 16];
        result[..4].copy_from_slice(&[
            self.hint,
            self.seen,
            self.fenced as i32,
            self.ticks.min(3) as i32,
        ]);
        for &byte in window {
            result[4] += byte.is_ascii_digit() as i32;
            result[5] += byte.is_ascii_alphabetic() as i32;
            result[6] += b"+-*/=.,:%".contains(&byte) as i32;
            result[7] += b"{}[];".contains(&byte) as i32;
            result[8] += (byte == b' ') as i32;
            result[9] += (byte == b'\n') as i32;
            result[10] += (byte == b'_') as i32;
        }
        let line_start = window
            .iter()
            .rposition(|&b| b == b'\n')
            .map_or(0, |i| i + 1);
        result[11] = window[line_start..]
            .iter()
            .take_while(|&&b| b == b' ')
            .count()
            .min(8) as i32;
        let last = window
            .iter()
            .rev()
            .find(|&&b| !b" \t\n\r\x0b\x0c".contains(&b))
            .copied();
        result[12] = (last == Some(b':')) as i32;
        result[15] = (last == Some(b'`')) as i32;
        let mut start = 0;
        while start < window.len() {
            if !(window[start].is_ascii_alphabetic() || window[start] == b'_') {
                start += 1;
                continue;
            }
            let mut end = start + 1;
            while end < window.len() && (window[end].is_ascii_alphabetic() || window[end] == b'_') {
                end += 1;
            }
            result[13] += INTRO.contains(&&window[start..end]) as i32;
            result[14] += KEYWORDS.contains(&&window[start..end]) as i32;
            start = end;
        }
        result
    }
}

struct Node {
    feature: i32,
    threshold: i32,
    left: usize,
    right: usize,
    counts: [usize; 3],
}

include!(env!("FORECAST_TREE_RS"));

fn predict(features: &[i32; 16]) -> usize {
    let mut node = &TREE[0];
    while node.feature >= 0 {
        node = &TREE[if features[node.feature as usize] <= node.threshold {
            node.left
        } else {
            node.right
        }];
    }
    let mut best = 0;
    for kind in 1..3 {
        if node.counts[kind] > node.counts[best] {
            best = kind;
        }
    }
    if node.counts[best] * 5 >= node.counts.iter().sum::<usize>() * 4 {
        best
    } else {
        3
    }
}

#[derive(Clone)]
struct Policy {
    k: i32,
    previous: i32,
    votes: usize,
    first: bool,
}

impl Policy {
    fn new() -> Self {
        Self {
            k: 3,
            previous: 3,
            votes: 0,
            first: true,
        }
    }
    fn select(&mut self, kind: usize) -> i32 {
        let proposed = [2, 4, 4, 3][kind];
        self.votes = if proposed == self.previous {
            self.votes.saturating_add(1)
        } else {
            1
        };
        self.previous = proposed;
        if self.first {
            self.first = false;
        } else if self.votes >= 2 {
            self.k += (proposed - self.k).signum();
        }
        self.k
    }
}

fn unhex(value: &str) -> Vec<u8> {
    assert!(value.len().is_multiple_of(2));
    (0..value.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&value[i..i + 2], 16).unwrap())
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let text = std::fs::read_to_string(&args[2])?;
    let cases: Vec<_> = text
        .lines()
        .map(|line| {
            let columns: Vec<_> = line.split('\t').collect();
            let mut state = State::new(columns[0].parse().unwrap());
            state.observe(&unhex(columns[1]));
            (state, unhex(columns[2]))
        })
        .collect();
    assert!(!cases.is_empty());
    let mut output = io::BufWriter::new(io::stdout().lock());
    if args[1] == "check" {
        for (state, appended) in &cases {
            let mut state = state.clone();
            state.observe(appended);
            let features = state.features();
            writeln!(
                output,
                "{}\t{}",
                features.map(|n| n.to_string()).join(","),
                predict(&features)
            )?;
        }
    } else if args[1] == "bench" {
        let iterations: usize = args[3].parse()?;
        for i in 0..1000 {
            let (state, appended) = &cases[i % cases.len()];
            let mut state = black_box(state.clone());
            state.observe(black_box(appended));
            black_box(predict(&state.features()));
        }
        let mut samples = Vec::with_capacity(iterations);
        let mut policy = Policy::new();
        for i in 0..iterations {
            let (state, appended) = &cases[i % cases.len()];
            let start = Instant::now();
            let mut state = black_box(state.clone());
            state.observe(black_box(appended));
            let prediction = black_box(predict(&state.features()));
            let k = black_box(policy.select(prediction));
            samples.push((start.elapsed().as_nanos(), prediction, k));
        }
        writeln!(output, "call\telapsed_ns\tprediction\tk")?;
        for (i, (ns, kind, k)) in samples.iter().enumerate() {
            writeln!(output, "{i}\t{ns}\t{kind}\t{k}")?;
        }
    } else {
        return Err("expected check or bench".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_starts_at_three_and_requires_repeated_evidence() {
        let mut policy = Policy::new();
        assert_eq!(policy.select(0), 3);
        assert_eq!(policy.select(0), 2);
        assert_eq!(policy.select(1), 2);
        assert_eq!(policy.select(1), 3);
        assert_eq!(policy.select(2), 4);
        assert_eq!(policy.select(3), 4);
        assert_eq!(policy.select(3), 3);
    }

    #[test]
    fn observation_chunking_does_not_change_features() {
        let text = "Here is Python code:\n```python\nreturn 1234\n```\nExplanation 🙂".as_bytes();
        let mut whole = State::new(1);
        whole.observe(text);
        for size in 1..8 {
            let mut chunks = State::new(1);
            for chunk in text.chunks(size) {
                chunks.observe(chunk);
            }
            assert_eq!(whole.features(), chunks.features());
        }
    }
}
