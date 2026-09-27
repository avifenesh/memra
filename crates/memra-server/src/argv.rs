//! Command-line argument admission for `memra-server` (memra#617).
//!
//! The server takes almost everything from the environment; argv carries exactly two kinds of
//! thing: the build-identity print (`--version` / `-V`) and the key-lifecycle commands
//! (`--gen-key`, `--revoke-key` with their modifiers `--lane`, `--rate-limit`, `--keys`). Before
//! this module the boot consulted argv only for those tokens and ignored every other one, so a
//! misspelled or retired flag on a launcher changed nothing and said nothing (lane C day 18,
//! `research/spill-c-20260919/DAY18.md` item 6: `flag_silently_accepted=True` on both cards).
//! [`validate`] runs first in the STOCK binary's `serve_main`, before the version print, before
//! any environment read and before any device work: an unknown token is a refusal that names
//! the token and lists what is accepted. The same posture memra#483 asks for unknown `MEMRA_*`
//! names. `serve_with` stays argv-agnostic: a deployment-owned binary parses its own command
//! line before delegating and may call [`validate`] itself for the stock set (revuto on #619).

/// The accepted argument shapes, printed in every refusal. Kept in one place so the message
/// and the tables below cannot drift apart.
pub const ACCEPTED: &str = "--version | -V | --gen-key <tenant> [--lane interactive|batch] \
                            [--rate-limit N] [--keys <path>] | --revoke-key <prefix> [--keys <path>]";

/// Flags that stand alone and take no value.
const BARE: &[&str] = &["--version", "-V"];
/// Key-lifecycle commands; each takes one value (the next token). A missing value is
/// `auth::run_cli`'s usage error, not a refusal here.
const KEY_COMMANDS: &[&str] = &["--gen-key", "--revoke-key"];
/// Modifiers of a key-lifecycle command; each takes one value and is accepted only beside a
/// command. Alone they used to boot the server and be ignored (a `--keys` path that names no
/// keyring is the silent class this module exists to refuse).
const KEY_MODIFIERS: &[&str] = &["--lane", "--rate-limit", "--keys"];

/// Admit `args` (argv without the program name) or return the refusal text naming the first
/// offending token. Every documented argument shape passes; nothing else does.
pub fn validate(args: &[String]) -> Result<(), String> {
    let has_key_command = args.iter().any(|a| KEY_COMMANDS.contains(&a.as_str()));
    let mut i = 0;
    while i < args.len() {
        let tok = args[i].as_str();
        if BARE.contains(&tok) {
            i += 1;
        } else if KEY_COMMANDS.contains(&tok) {
            i += 2;
        } else if KEY_MODIFIERS.contains(&tok) {
            if !has_key_command {
                return Err(format!(
                    "argument {tok:?} is accepted only beside --gen-key or --revoke-key; \
                     accepted arguments: {ACCEPTED}"
                ));
            }
            i += 2;
        } else {
            return Err(format!(
                "unknown argument {tok:?} refused at boot; accepted arguments: {ACCEPTED}"
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(args: &[&str]) -> Result<(), String> {
        validate(&args.iter().map(|s| s.to_string()).collect::<Vec<_>>())
    }

    #[test]
    fn empty_argv_boots() {
        assert_eq!(v(&[]), Ok(()));
    }

    #[test]
    fn every_documented_shape_is_accepted() {
        // docs/SERVING.md "Reading it without a GPU" and "API keys"; tools/apikeys-gate.sh.
        for shape in [
            &["--version"][..],
            &["-V"],
            &["--gen-key", "acme", "--keys", "/tmp/keys.toml"],
            &["--gen-key", "acme"],
            &[
                "--gen-key",
                "bulk",
                "--lane",
                "batch",
                "--rate-limit",
                "2",
                "--keys",
                "/k",
            ],
            &["--gen-key", "acme", "--lane", "batch"],
            &["--gen-key", "acme", "--rate-limit", "4"],
            &["--revoke-key", "mk-dead-", "--keys", "/k"],
            &["--revoke-key", "mk-dead-"],
            &["--keys", "/k", "--revoke-key", "mk-dead-"],
        ] {
            assert_eq!(v(shape), Ok(()), "{shape:?}");
        }
    }

    #[test]
    fn unknown_flag_is_refused_and_named() {
        let err = v(&["--experts-via-tier"]).unwrap_err();
        assert!(
            err.contains("unknown argument \"--experts-via-tier\""),
            "{err}"
        );
        assert!(err.contains("refused at boot"), "{err}");
        assert!(err.contains(ACCEPTED), "{err}");
    }

    #[test]
    fn unknown_token_after_a_valid_one_is_refused() {
        let err = v(&["--version", "--bogus"]).unwrap_err();
        assert!(err.contains("\"--bogus\""), "{err}");
        let err = v(&["--gen-key", "acme", "--keys", "/k", "extra"]).unwrap_err();
        assert!(err.contains("\"extra\""), "{err}");
    }

    #[test]
    fn positional_token_is_refused() {
        let err = v(&["serve"]).unwrap_err();
        assert!(err.contains("unknown argument \"serve\""), "{err}");
    }

    #[test]
    fn key_modifier_without_a_command_is_refused() {
        for tok in KEY_MODIFIERS {
            let err = v(&[tok, "x"]).unwrap_err();
            assert!(
                err.contains(&format!("argument {tok:?} is accepted only beside")),
                "{err}"
            );
        }
    }

    #[test]
    fn a_missing_value_is_left_to_run_cli() {
        // `--gen-key` with no tenant: admitted here, `auth::run_cli` prints its usage and exits 2.
        assert_eq!(v(&["--gen-key"]), Ok(()));
        assert_eq!(v(&["--revoke-key"]), Ok(()));
    }
}
