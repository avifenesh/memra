mod router;

use router::{Kind, Profile, classify, select_depth};
use std::error::Error;
use std::hint::black_box;
use std::io::{self, Read};
use std::time::Instant;

struct Case {
    id: String,
    expected: String,
    prompt: String,
}

fn unescape(text: &str) -> Result<String, Box<dyn Error>> {
    let mut output = String::new();
    let mut chars = text.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            output.push(c);
            continue;
        }
        output.push(match chars.next() {
            Some('n') => '\n',
            Some('t') => '\t',
            Some('r') => '\r',
            Some('\\') => '\\',
            _ => return Err("unsupported fixture escape".into()),
        });
    }
    Ok(output)
}

fn cases(text: &str) -> Result<Vec<Case>, Box<dyn Error>> {
    let mut seen = std::collections::HashSet::new();
    let mut result = Vec::new();
    if text.lines().next() != Some("id\texpected\tprompt") {
        return Err("wrong corpus header".into());
    }
    for line in text.lines().skip(1) {
        let fields: Vec<_> = line.splitn(3, '\t').collect();
        if fields.len() != 3
            || fields[0].is_empty()
            || !fields[0]
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-')
            || !seen.insert(fields[0])
            || !["prose", "code", "numeric", "mixed", "unknown"].contains(&fields[1])
        {
            return Err("invalid or duplicate corpus row".into());
        }
        result.push(Case {
            id: fields[0].to_owned(),
            expected: fields[1].to_owned(),
            prompt: unescape(fields[2])?,
        });
    }
    if result.is_empty() {
        return Err("empty corpus".into());
    }
    Ok(result)
}

fn profile(text: &str) -> Result<Profile, Box<dyn Error>> {
    let values: Vec<u8> = text.split(',').map(str::parse).collect::<Result<_, _>>()?;
    if values.len() != 4 || values.iter().any(|&k| k > 64) {
        return Err("profile must have four depths in 0..=64: prose,code,numeric,fallback".into());
    }
    Ok(Profile {
        prose: values[0],
        code: values[1],
        numeric: values[2],
        fallback: values[3],
    })
}

fn check(rows: &[Case]) -> Result<(), Box<dyn Error>> {
    let mut failures = 0;
    for row in rows {
        let actual = classify(&row.prompt).name();
        println!("{}\t{}\t{}", row.id, row.expected, actual);
        failures += usize::from(actual != row.expected);
    }
    println!("{{\"cases\":{},\"failures\":{failures}}}", rows.len());
    if failures != 0 {
        return Err("classification cases failed".into());
    }
    Ok(())
}

fn percentile(sorted: &[u64], percent: usize) -> u64 {
    sorted[((sorted.len() - 1) * percent) / 100]
}

fn measure(name: &str, prompts: &[&str], profile: Profile, ceiling: u8, iterations: usize) {
    let mut times = Vec::with_capacity(iterations);
    let mut checksum = 0usize;
    for i in 0..512 {
        black_box(select_depth(
            black_box(prompts[i % prompts.len()]),
            black_box(profile),
            ceiling,
        ));
    }
    // Interleave prompts so one repeated instruction does not dominate branch history.
    let mut index = 0x1234_5678u32;
    for _ in 0..iterations {
        index = index.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let prompt = black_box(prompts[index as usize % prompts.len()]);
        let start = Instant::now();
        let decision = black_box(select_depth(prompt, black_box(profile), black_box(ceiling)));
        let ns = start.elapsed().as_nanos() as u64;
        times.push(ns);
        checksum += decision.k as usize;
    }
    let mut sorted = times.clone();
    sorted.sort_unstable();
    println!(
        "{{\"group\":\"{name}\",\"calls\":{iterations},\"prompts\":{},\"p50_ns\":{},\"p95_ns\":{},\"p99_ns\":{},\"max_ns\":{},\"checksum\":{checksum},\"samples_ns\":{times:?}}}",
        prompts.len(),
        percentile(&sorted, 50),
        percentile(&sorted, 95),
        percentile(&sorted, 99),
        sorted[sorted.len() - 1]
    );
}

fn run() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("classify") if args.len() == 4 => {
            let p = profile(&args[2])?;
            let ceiling: u8 = args[3].parse()?;
            let mut input = Vec::new();
            io::stdin().lock().take(router::MAX_BYTES as u64 + 1).read_to_end(&mut input)?;
            let decision = if input.len() > router::MAX_BYTES {
                router::Decision { kind: Kind::Unknown, k: p.fallback.min(ceiling) }
            } else {
                select_depth(std::str::from_utf8(&input)?, p, ceiling)
            };
            println!("{{\"kind\":\"{}\",\"k\":{}}}", decision.kind.name(), decision.k);
        }
        Some("check") if args.len() == 3 => {
            check(&cases(&std::fs::read_to_string(&args[2])?)?)?;
        }
        Some("bench") if args.len() == 6 => {
            let rows = cases(&std::fs::read_to_string(&args[2])?)?;
            let p = profile(&args[3])?;
            let ceiling = args[4].parse()?;
            let iterations: usize = args[5].parse()?;
            if !(1_000..=1_000_000).contains(&iterations) {
                return Err("benchmark iterations must be in 1000..=1000000".into());
            }
            let classified: Vec<&str> = rows.iter()
                .filter(|r| matches!(classify(&r.prompt), Kind::Prose | Kind::Code | Kind::Numeric))
                .map(|r| r.prompt.as_str()).collect();
            let fallback: Vec<&str> = rows.iter()
                .filter(|r| matches!(classify(&r.prompt), Kind::Mixed | Kind::Unknown))
                .map(|r| r.prompt.as_str()).collect();
            if classified.is_empty() || fallback.is_empty() {
                return Err("benchmark needs classified and fallback inputs".into());
            }
            measure("classified_instructions", &classified, p, ceiling, iterations);
            measure("semantic_fallbacks", &fallback, p, ceiling, iterations);
            let long = format!("Explain this:\n```\n{}\n```", "x".repeat(router::MAX_BYTES - 22));
            assert_eq!(classify(&long), Kind::Prose);
            measure("near_byte_limit_quoted_input", &[&long], p, ceiling, iterations);
            let excessive = "x".repeat(router::MAX_BYTES + 1);
            measure("byte_limit_fallback", &[&excessive], p, ceiling, iterations);
            let many = format!("Explain {}", "a ".repeat(router::MAX_WORDS));
            measure("word_limit_fallback", &[&many], p, ceiling, iterations);
        }
        _ => return Err("usage: prompt-depth-router classify PROSE,CODE,NUMERIC,FALLBACK CEILING < instruction\n       prompt-depth-router check cases.tsv\n       prompt-depth-router bench cases.tsv PROSE,CODE,NUMERIC,FALLBACK CEILING ITERATIONS".into()),
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PROFILE: Profile = Profile {
        prose: 2,
        code: 4,
        numeric: 4,
        fallback: 3,
    };

    #[test]
    fn requested_output_behavioral_cases() {
        let rows = cases(include_str!("cases.tsv")).unwrap();
        check(&rows).unwrap();
    }

    #[test]
    fn separate_requests_do_not_carry_a_previous_depth() {
        let prompts = [
            "Explain this code.",
            "Write a Python function.",
            "Calculate 7 * 8.",
            "Hello!",
            "Write a poem.",
        ];
        let actual: Vec<u8> = prompts
            .iter()
            .map(|p| select_depth(p, PROFILE, 8).k)
            .collect();
        assert_eq!(actual, [2, 4, 4, 3, 2]);
    }

    #[test]
    fn admission_can_disable_or_cap_speculation() {
        for prompt in [
            "Write code.",
            "Explain it.",
            "Calculate 7 * 8.",
            "Continue.",
        ] {
            assert_eq!(select_depth(prompt, PROFILE, 0).k, 0);
            assert_eq!(select_depth(prompt, PROFILE, 1).k, 1);
        }
    }

    #[test]
    fn input_limits_do_not_route_from_a_partial_instruction() {
        let exact = format!("Explain {}", "x".repeat(router::MAX_BYTES - 8));
        assert_eq!(exact.len(), router::MAX_BYTES);
        assert_eq!(classify(&exact), Kind::Prose);
        assert_eq!(
            select_depth(&(exact + "x"), PROFILE, 8),
            router::Decision {
                kind: Kind::Unknown,
                k: 3
            }
        );
        assert_eq!(
            classify(&format!("Explain {}", "word ".repeat(router::MAX_WORDS))),
            Kind::Unknown
        );
    }

    #[test]
    fn unicode_and_unclosed_quotes_are_safe() {
        for source in ["λ😊漢字", "`", "\"", "“", "‘", "````", "\\"] {
            assert_eq!(classify(source), Kind::Unknown);
        }
        assert_eq!(classify("EXPLAIN this 😊 function."), Kind::Prose);
        assert_eq!(classify("Don’t write code."), Kind::Unknown);
    }

    #[test]
    fn quoted_operators_and_requests_do_not_become_output_intent() {
        for source in [
            "\"What is 7 * 8?\"",
            "`Calculate 7 * 8.`",
            "‘Generate code.’",
            "> Write code.\n> Calculate 7 * 8.",
        ] {
            assert_eq!(classify(source), Kind::Unknown);
        }
    }

    #[test]
    fn corpus_parser_rejects_silent_row_loss() {
        assert!(
            cases("id\texpected\tprompt\nx\tprose\tExplain it.\nx\tcode\tWrite code.").is_err()
        );
        assert!(cases("id\texpected\tprompt\nx\tunknown\tbad\\q").is_err());
        assert!(cases("id\texpected\tprompt\nx\tother\tHello").is_err());
        assert!(cases("id\texpected\tprompt\n").is_err());
    }
}
