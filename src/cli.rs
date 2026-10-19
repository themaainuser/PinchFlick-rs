//! CLI surface rules (final-review fixes): argument reading and
//! error-reporting extracted from `main` so the *rules* are pure and
//! unit-tested. The wiring in `main.rs` stays a declared spike like every
//! other CLI flag there.

use crate::config::ConfigError;
use std::error::Error;

/// Flags whose next token is their value — never a positional file.
const VALUE_FLAGS: &[&str] = &["--config", "--camera", "--video"];

/// Dev/probe modes that read no config (they outrank config-implies-live).
const DEV_FLAGS: &[&str] = &["--detect", "--landmarks", "--video"];

/// The positional frame path: the first token that is neither a flag nor
/// the value of a value-taking flag (fixes `--camera 1`, `--config f.json`
/// and `--video 3` being mistaken for the file — the M1-era parser read the
/// first non-`--` token blindly). Default `frame.png`.
pub fn frame_path(args: &[String]) -> String {
    let mut skip_value = false;
    args.iter()
        .find(|a| {
            if skip_value {
                skip_value = false;
                return false;
            }
            if VALUE_FLAGS.contains(&a.as_str()) {
                skip_value = true;
                return false;
            }
            !a.starts_with("--")
        })
        .cloned()
        .unwrap_or_else(|| "frame.png".into())
}

/// Live mode: explicit `--live`, or v1 parity — v1's argparse had no dev
/// harness, so `--camera`/`--config` launched the live app (`controller.py`
/// argparse → `run(cfg)`). Dev probe flags outrank the implication: they
/// never read a config, so `--video 3 --config c.json` stays a video run.
pub fn implies_live(args: &[String]) -> bool {
    let has = |f: &str| args.iter().any(|a| a == f);
    if has("--live") {
        return true;
    }
    if DEV_FLAGS.iter().any(|f| has(f)) {
        return false;
    }
    has("--camera") || has("--config")
}

/// v1 `controller.py:333-348`: config failures print `Config error: {msg}`
/// to stderr and exit **2**; everything else prints `Fatal error: ...` and
/// exits **1**. Rust has no exception classes, so the fatal line carries
/// the message only (documented difference) — the config line and both
/// exit codes are exact.
pub fn error_output(e: &(dyn Error + 'static)) -> (String, i32) {
    if let Some(ce) = e.downcast_ref::<ConfigError>() {
        (format!("Config error: {ce}"), 2)
    } else {
        (format!("Fatal error: {e}"), 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;

    fn v(xs: &[&str]) -> Vec<String> {
        xs.iter().map(|s| s.to_string()).collect()
    }

    // frame_path

    #[test]
    fn cli_frame_path_no_args_defaults() {
        assert_eq!(frame_path(&[]), "frame.png");
    }

    #[test]
    fn cli_frame_path_skips_flag_values() {
        assert_eq!(frame_path(&v(&["--camera", "1"])), "frame.png");
        assert_eq!(frame_path(&v(&["--config", "my.json"])), "frame.png");
        assert_eq!(frame_path(&v(&["--video", "3"])), "frame.png");
        assert_eq!(frame_path(&v(&["--live"])), "frame.png");
    }

    #[test]
    fn cli_frame_path_takes_first_positional() {
        assert_eq!(frame_path(&v(&["frame.png", "--video", "3"])), "frame.png");
        assert_eq!(
            frame_path(&v(&["--config", "c.json", "shot.png"])),
            "shot.png"
        );
    }

    // implies_live

    #[test]
    fn cli_live_explicit_flag() {
        assert!(implies_live(&v(&["--live"])));
    }

    #[test]
    fn cli_live_implied_by_camera_and_config() {
        assert!(implies_live(&v(&["--camera", "1"])));
        assert!(implies_live(&v(&["--config", "my.json"])));
    }

    #[test]
    fn cli_live_not_implied_by_dev_flags() {
        assert!(!implies_live(&v(&["--video", "3"])));
        assert!(!implies_live(&v(&["--detect", "--camera", "1"])));
        assert!(!implies_live(&v(&["--landmarks", "--config", "c.json"])));
    }

    #[test]
    fn cli_live_default_is_harness() {
        assert!(!implies_live(&[]));
        assert!(!implies_live(&v(&["frame.png"])));
    }

    // error_output

    #[test]
    fn cli_config_error_line_and_exit_2() {
        let e = ConfigError("seek_seconds: 12 must be a multiple of 5".into());
        let (line, code) = error_output(&e);
        assert_eq!(line, "Config error: seek_seconds: 12 must be a multiple of 5");
        assert_eq!(code, 2);
    }

    #[test]
    fn cli_fatal_error_line_and_exit_1() {
        let e = io::Error::new(io::ErrorKind::NotFound, "frame.png not found");
        let (line, code) = error_output(&e);
        assert_eq!(line, "Fatal error: frame.png not found");
        assert_eq!(code, 1);
    }
}
