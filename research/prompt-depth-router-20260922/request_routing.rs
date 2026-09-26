//! Thin binding for the hash-pinned native research drivers.
use super::router::{Decision, Kind, Profile, select_depth};
use std::io::{self, Write};
use std::time::Instant;

#[derive(Clone, Copy)]
pub enum RequestRouting {
    Prompt(Profile),
    Schedule([u8; 8]),
}

#[derive(Clone, Copy)]
pub struct Selection {
    pub decision: Decision,
    pub elapsed_ns: u128,
    pub source: &'static str,
}

impl RequestRouting {
    pub fn parse(arm: &str, cap: u8) -> Result<Option<Self>, String> {
        let (text, count) = if let Some(text) = arm.strip_prefix("prompt:") {
            (text, 4)
        } else if let Some(text) = arm.strip_prefix("schedule:") {
            (text, 8)
        } else {
            return Ok(None);
        };
        let values: Vec<u8> = text
            .split(',')
            .map(|v| v.parse().map_err(|_| "invalid request depth".to_owned()))
            .collect::<Result<_, _>>()?;
        if values.len() != count || values.iter().any(|&k| k == 0 || k > cap) {
            return Err(
                "request depths must match the profile/schedule length and native cap".into(),
            );
        }
        Ok(Some(if count == 4 {
            Self::Prompt(Profile {
                prose: values[0],
                code: values[1],
                numeric: values[2],
                fallback: values[3],
            })
        } else {
            Self::Schedule(
                values
                    .as_slice()
                    .try_into()
                    .map_err(|_| "schedule length")?,
            )
        }))
    }

    pub fn initial_k(self) -> u8 {
        match self {
            Self::Prompt(profile) => profile.fallback,
            Self::Schedule(depths) => depths[0],
        }
    }

    pub fn select(
        self,
        turn: usize,
        instruction: &str,
        ceiling: u8,
    ) -> Result<Selection, &'static str> {
        if ceiling == 0 {
            return Err("native research route needs an enabled speculation ceiling");
        }
        let started = Instant::now();
        let (decision, source) = match self {
            Self::Prompt(profile) => (select_depth(instruction, profile, ceiling), "prompt"),
            Self::Schedule(depths) => {
                let index = turn.checked_sub(1).ok_or("schedule turns start at one")?;
                let k = *depths.get(index).ok_or("schedule has eight turns")?;
                (
                    Decision {
                        kind: Kind::Unknown,
                        k: k.min(ceiling),
                    },
                    "schedule",
                )
            }
        };
        Ok(Selection {
            decision,
            elapsed_ns: started.elapsed().as_nanos(),
            source,
        })
    }
}

pub fn write_record(
    output: &mut impl Write,
    turn: usize,
    selection: Option<Selection>,
) -> io::Result<()> {
    if let Some(value) = selection {
        writeln!(
            output,
            "{turn}\t{}\t{}\t{}\t{}",
            value.source,
            value.decision.kind.name(),
            value.decision.k,
            value.elapsed_ns,
        )
    } else {
        writeln!(output, "{turn}\tcontrol\tnot-invoked\t-\t0")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_profile_bounds_and_control_arms_are_explicit() {
        assert!(RequestRouting::parse("native", 5).unwrap().is_none());
        assert!(RequestRouting::parse("fixed:4", 5).unwrap().is_none());
        assert!(RequestRouting::parse("context", 5).unwrap().is_none());
        for arm in [
            "prompt:0,4,4,4",
            "prompt:2,9,4,4",
            "prompt:2,4",
            "prompt:x,4,4,4",
        ] {
            assert!(RequestRouting::parse(arm, 5).is_err());
        }
        let route = RequestRouting::parse("prompt:2,4,4,4", 5).unwrap().unwrap();
        assert_eq!(route.initial_k(), 4);
        assert_eq!(route.select(1, "Write code.", 3).unwrap().decision.k, 3);
        assert!(route.select(1, "Write code.", 0).is_err());
    }

    #[test]
    fn fixed_schedule_is_a_separate_replay_control() {
        let route = RequestRouting::parse("schedule:2,4,4,2,4,4,2,4", 5)
            .unwrap()
            .unwrap();
        assert_eq!(route.initial_k(), 2);
        assert_eq!(
            route
                .select(4, "This text must not select the depth.", 5)
                .unwrap()
                .decision
                .k,
            2
        );
        assert!(route.select(0, "x", 5).is_err());
        assert!(route.select(9, "x", 5).is_err());
        assert!(RequestRouting::parse("schedule:2,4", 5).is_err());
    }

    #[test]
    fn routing_receipt_keeps_controls_distinct_from_classification() {
        let route = RequestRouting::parse("prompt:2,4,4,4", 5).unwrap().unwrap();
        let mut out = Vec::new();
        write_record(&mut out, 1, None).unwrap();
        write_record(
            &mut out,
            2,
            Some(route.select(2, "Explain the code.", 5).unwrap()),
        )
        .unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(text.starts_with("1\tcontrol\tnot-invoked\t-\t0\n2\tprompt\tprose\t2\t"));
        assert_eq!(text.lines().count(), 2);
    }
}
