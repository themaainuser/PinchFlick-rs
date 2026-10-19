//! Port of v1's config section: identical 32-key schema, identical merge
//! semantics, identical validation messages (`{source}: '{key}' {why}
//! (got {got!r})` — Python repr reproduced in [`py_repr`]).

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::fmt;
use std::path::Path;

/// v1's ConfigError: raised for a missing, malformed, or invalid config file.
#[derive(Debug, Clone)]
pub struct ConfigError(pub String);

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ConfigError {}

/// YouTube shortcut bindings; a user config may override individual fields.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SeekKeys {
    pub forward_5: String,
    pub backward_5: String,
    pub forward_10: String,
    pub backward_10: String,
}

impl Default for SeekKeys {
    fn default() -> Self {
        Self {
            forward_5: "right".into(),
            backward_5: "left".into(),
            forward_10: "l".into(),
            backward_10: "j".into(),
        }
    }
}

/// The full 32-key configuration. Field names == JSON keys == v1's
/// DEFAULT_CONFIG dict keys — no config-format divergence from v1.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Config {
    // --- camera / tracking ---
    pub camera_index: i64,
    pub frame_width: i64,
    pub frame_height: i64,
    pub mirror_display: bool,
    /// Validated 1–2 like v1, but the runtime tracks **1** hand — v1's
    /// shipped default, and all v1's controller consumes (it takes
    /// `hand_landmarks[0]`, controller.py:192). See README differences.
    pub num_hands: i64,
    pub min_detection_confidence: f64,
    pub min_presence_confidence: f64,
    pub min_tracking_confidence: f64,

    // --- scroll gesture (works everywhere) ---
    pub scroll_enabled: bool,
    pub scroll_up_fingers: i64,
    pub scroll_down_fingers: i64,
    pub scroll_speed: i64,
    pub scroll_interval_ms: i64,
    pub scroll_confirm_frames: i64,
    pub scroll_invert: bool,
    pub scroll_open_hand_ratio: f64,

    // --- seek gesture (YouTube only) ---
    pub seek_enabled: bool,
    pub seek_requires_youtube: bool,
    pub seek_seconds: i64,
    pub double_tap_window_ms: i64,
    pub seek_cooldown_ms: i64,
    pub tap_side_dead_zone: f64,
    pub scroll_suppress_after_seek_ms: i64,
    pub pinch_on_ratio: f64,
    pub pinch_off_ratio: f64,
    pub seek_keys: SeekKeys,

    // --- YouTube detection (foreground window title) ---
    pub youtube_title_markers: Vec<String>,
    pub youtube_check_interval_ms: i64,

    // --- misc ---
    pub show_debug_window: bool,
    pub always_on_top: bool,
    pub pyautogui_failsafe: bool,
    pub model_path: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            camera_index: 0,
            frame_width: 640,
            frame_height: 480,
            mirror_display: true,
            num_hands: 1,
            min_detection_confidence: 0.6,
            min_presence_confidence: 0.6,
            min_tracking_confidence: 0.6,
            scroll_enabled: true,
            scroll_up_fingers: 1,
            scroll_down_fingers: 2,
            scroll_speed: 5,
            scroll_interval_ms: 0,
            scroll_confirm_frames: 2,
            scroll_invert: false,
            scroll_open_hand_ratio: 0.55,
            seek_enabled: true,
            seek_requires_youtube: true,
            seek_seconds: 10,
            double_tap_window_ms: 550,
            seek_cooldown_ms: 0,
            tap_side_dead_zone: 0.04,
            scroll_suppress_after_seek_ms: 500,
            pinch_on_ratio: 0.35,
            pinch_off_ratio: 0.55,
            seek_keys: SeekKeys::default(),
            youtube_title_markers: vec!["- YouTube".into(), "\u{2013} YouTube".into()],
            youtube_check_interval_ms: 200,
            show_debug_window: false,
            always_on_top: true,
            pyautogui_failsafe: true,
            model_path: None,
        }
    }
}

impl Config {
    /// v1's `validate_config(cfg, source)` — runs the JSON-value validator on
    /// this config's serialized form, so there is exactly ONE implementation
    /// of the rules and messages, shared with the load path.
    pub fn validate(&self, source: &str) -> Result<(), ConfigError> {
        let v = serde_json::to_value(self).expect("Config always serializes");
        validate_value(&v, source)
    }
}

/// v1's `load_config(path=None)`: merge a JSON config over the defaults and
/// validate. `None` uses `config.json` next to this exe if present (else
/// defaults); an explicitly given path that does not exist is an error.
pub fn load_config(path: Option<&str>) -> Result<Config, ConfigError> {
    let explicit = path.is_some();
    let path = path.map(str::to_owned).unwrap_or_else(default_config_path);

    let mut cfg = default_config_value();

    if Path::new(&path).exists() {
        let text = std::fs::read_to_string(&path)
            .map_err(|e| ConfigError(format!("{path} could not be read: {e}")))?;
        let user: Value = serde_json::from_str(&text)
            .map_err(|e| ConfigError(format!("{path} is not valid JSON: {e}")))?;
        let Some(user_map) = user.as_object() else {
            return Err(ConfigError(format!(
                "{path} must contain a JSON object"
            )));
        };

        // unknown keys (v1 checks before merging; sorted in the message)
        let defaults = default_config_value();
        let mut known: Vec<String> = defaults.as_object().unwrap().keys().cloned().collect();
        known.sort();
        let mut unknown: Vec<String> = user_map
            .keys()
            .filter(|k| !known.contains(k))
            .cloned()
            .collect();
        if !unknown.is_empty() {
            unknown.sort();
            return Err(ConfigError(format!(
                "unknown key(s) in {path}: {} (valid keys: {})",
                unknown.join(", "),
                known.join(", ")
            )));
        }

        // partial override of the seek_keys table is allowed
        let mut user_map = user_map.clone();
        if user_map.get("seek_keys").is_some_and(Value::is_object) {
            let mut keys = defaults["seek_keys"].clone();
            if let (Some(base), Some(overlay)) = (keys.as_object_mut(), user_map["seek_keys"].as_object()) {
                for (k, v) in overlay {
                    base.insert(k.clone(), v.clone());
                }
            }
            user_map.insert("seek_keys".into(), keys);
        }

        // cfg.update(user)
        if let Some(cfg_map) = cfg.as_object_mut() {
            for (k, v) in user_map {
                cfg_map.insert(k.clone(), v.clone());
            }
        }
    } else if explicit {
        return Err(ConfigError(format!("config file not found: {path}")));
    }

    validate_value(&cfg, &path)?;
    serde_json::from_value(cfg).map_err(|e| ConfigError(format!("{path}: invalid config: {e}")))
}

/// v1's `default_config_path()`: config.json next to this binary.
pub fn default_config_path() -> String {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.join("config.json")))
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|| "config.json".into())
}

/// The defaults as the merged-space JSON value (post-serialization).
fn default_config_value() -> Value {
    serde_json::to_value(Config::default()).expect("defaults serialize")
}

// ---------------------------------------------------------------------------
// Validation (line-by-line port of v1's validate_config)
// ---------------------------------------------------------------------------

fn err(source: &str, key: &str, why: &str, got: &Value) -> ConfigError {
    err_repr(source, key, why, py_repr(got))
}

fn err_repr(source: &str, key: &str, why: &str, got: String) -> ConfigError {
    ConfigError(format!("{source}: '{key}' {why} (got {got})"))
}

fn get(cfg: &Map<String, Value>, key: &str) -> Value {
    cfg.get(key).cloned().unwrap_or(Value::Null)
}

fn want_bool(cfg: &Map<String, Value>, key: &str, source: &str) -> Result<(), ConfigError> {
    let v = get(cfg, key);
    if !v.is_boolean() {
        return Err(err(source, key, "must be true or false", &v));
    }
    Ok(())
}

fn want_int(
    cfg: &Map<String, Value>,
    key: &str,
    source: &str,
    lo: Option<i64>,
    hi: Option<i64>,
) -> Result<(), ConfigError> {
    let v = get(cfg, key);
    let Some(n) = v.as_i64() else {
        // v1 rejects bools here too (isinstance(v, bool) first)
        return Err(err(source, key, "must be a whole number", &v));
    };
    if let Some(lo) = lo
        && n < lo
    {
        return Err(err(source, key, &format!("must be >= {lo}"), &v));
    }
    if let Some(hi) = hi
        && n > hi
    {
        return Err(err(source, key, &format!("must be <= {hi}"), &v));
    }
    Ok(())
}

fn want_number(
    cfg: &Map<String, Value>,
    key: &str,
    source: &str,
    lo: Option<f64>,
    hi: Option<f64>,
) -> Result<(), ConfigError> {
    let v = get(cfg, key);
    // serde: JSON bools are Value::Bool, never as_f64 — matches v1's
    // isinstance(v, bool) rejection
    let Some(n) = v.as_f64() else {
        return Err(err(source, key, "must be a number", &v));
    };
    if let Some(lo) = lo
        && n < lo
    {
        return Err(err(source, key, &format!("must be >= {lo}"), &v));
    }
    if let Some(hi) = hi
        && n > hi
    {
        return Err(err(source, key, &format!("must be <= {hi}"), &v));
    }
    Ok(())
}

/// Raise ConfigError naming the exact key if anything is off — v1's
/// `validate_config`, in v1's check order.
fn validate_value(v: &Value, source: &str) -> Result<(), ConfigError> {
    let cfg = v.as_object().cloned().unwrap_or_default();

    for key in [
        "mirror_display",
        "scroll_enabled",
        "scroll_invert",
        "seek_enabled",
        "seek_requires_youtube",
        "show_debug_window",
        "always_on_top",
        "pyautogui_failsafe",
    ] {
        want_bool(&cfg, key, source)?;
    }

    for key in [
        "camera_index",
        "frame_width",
        "frame_height",
        "num_hands",
        "scroll_up_fingers",
        "scroll_down_fingers",
        "scroll_speed",
        "scroll_interval_ms",
        "scroll_confirm_frames",
        "seek_seconds",
        "double_tap_window_ms",
        "seek_cooldown_ms",
        "scroll_suppress_after_seek_ms",
        "youtube_check_interval_ms",
    ] {
        want_int(&cfg, key, source, Some(0), None)?;
    }

    if get(&cfg, "camera_index").as_i64().unwrap_or(0) < 0 {
        return Err(err(
            source,
            "camera_index",
            "must be >= 0",
            &get(&cfg, "camera_index"),
        ));
    }
    let (fw, fh) = (
        get(&cfg, "frame_width").as_i64().unwrap_or(0),
        get(&cfg, "frame_height").as_i64().unwrap_or(0),
    );
    if fw < 64 || fh < 64 {
        return Err(err_repr(
            source,
            "frame_width/frame_height",
            "must be at least 64",
            format!("({fw}, {fh})"),
        ));
    }
    let hands = get(&cfg, "num_hands").as_i64().unwrap_or(0);
    if hands != 1 && hands != 2 {
        return Err(err(
            source,
            "num_hands",
            "must be 1 or 2",
            &get(&cfg, "num_hands"),
        ));
    }
    let (up, down) = (
        get(&cfg, "scroll_up_fingers").as_i64().unwrap_or(0),
        get(&cfg, "scroll_down_fingers").as_i64().unwrap_or(0),
    );
    if !(0..5).contains(&up) || !(0..5).contains(&down) {
        return Err(err_repr(
            source,
            "scroll_up_fingers/scroll_down_fingers",
            "must be 0-4",
            format!("({up}, {down})"),
        ));
    }
    if up == down {
        return Err(err_repr(
            source,
            "scroll_up_fingers/scroll_down_fingers",
            "must differ",
            format!("({up}, {down})"),
        ));
    }
    if get(&cfg, "scroll_speed").as_i64().unwrap_or(0) < 1 {
        return Err(err(
            source,
            "scroll_speed",
            "must be >= 1",
            &get(&cfg, "scroll_speed"),
        ));
    }
    if get(&cfg, "scroll_confirm_frames").as_i64().unwrap_or(0) < 1 {
        return Err(err(
            source,
            "scroll_confirm_frames",
            "must be >= 1",
            &get(&cfg, "scroll_confirm_frames"),
        ));
    }
    if get(&cfg, "double_tap_window_ms").as_i64().unwrap_or(0) < 100 {
        return Err(err(
            source,
            "double_tap_window_ms",
            "must be >= 100",
            &get(&cfg, "double_tap_window_ms"),
        ));
    }

    let seek_seconds = get(&cfg, "seek_seconds").as_i64().unwrap_or(0);
    if seek_seconds < 5 || seek_seconds > 60 || seek_seconds % 5 != 0 {
        return Err(err(
            source,
            "seek_seconds",
            "must be a multiple of 5 between 5 and 60 (YouTube's shortcuts only move in 5s steps)",
            &get(&cfg, "seek_seconds"),
        ));
    }

    for key in [
        "pinch_on_ratio",
        "pinch_off_ratio",
        "scroll_open_hand_ratio",
        "tap_side_dead_zone",
        "min_detection_confidence",
        "min_presence_confidence",
        "min_tracking_confidence",
    ] {
        want_number(&cfg, key, source, Some(0.0), Some(1.0))?;
    }
    if get(&cfg, "tap_side_dead_zone").as_f64().unwrap_or(0.0) > 0.49 {
        return Err(err(
            source,
            "tap_side_dead_zone",
            "must be <= 0.49",
            &get(&cfg, "tap_side_dead_zone"),
        ));
    }
    let (on, off) = (
        get(&cfg, "pinch_on_ratio").as_f64().unwrap_or(0.0),
        get(&cfg, "pinch_off_ratio").as_f64().unwrap_or(0.0),
    );
    if on >= off {
        return Err(err(
            source,
            "pinch_on_ratio",
            "must be smaller than pinch_off_ratio",
            &get(&cfg, "pinch_on_ratio"),
        ));
    }
    let open_ratio = get(&cfg, "scroll_open_hand_ratio").as_f64().unwrap_or(0.0);
    if on >= open_ratio {
        return Err(err(
            source,
            "scroll_open_hand_ratio",
            "must be larger than pinch_on_ratio",
            &get(&cfg, "scroll_open_hand_ratio"),
        ));
    }

    let keys = get(&cfg, "seek_keys");
    let Some(keys_map) = keys.as_object() else {
        return Err(err(source, "seek_keys", "must be an object", &keys));
    };
    let missing: Vec<&str> = SeekKeys::default()
        .iter_keys()
        .into_iter()
        .filter(|k| {
            keys_map
                .get(*k)
                .is_none_or(|v| !v.as_str().is_some_and(|s| !s.is_empty()))
        })
        .collect();
    if !missing.is_empty() {
        return Err(err(
            source,
            "seek_keys",
            &format!("missing/empty key binding(s): {}", missing.join(", ")),
            &keys,
        ));
    }

    let markers = get(&cfg, "youtube_title_markers");
    let markers_ok = markers.as_array().is_some_and(|a| {
        !a.is_empty() && a.iter().all(|m| m.as_str().is_some_and(|s| !s.is_empty()))
    });
    if !markers_ok {
        return Err(err(
            source,
            "youtube_title_markers",
            "must be a non-empty list of strings",
            &markers,
        ));
    }
    if get(&cfg, "seek_requires_youtube").as_bool().unwrap_or(false)
        && markers.as_array().is_some_and(|a| a.is_empty())
    {
        return Err(err(
            source,
            "youtube_title_markers",
            "cannot be empty while seek_requires_youtube is true",
            &markers,
        ));
    }

    let model_path = get(&cfg, "model_path");
    if !model_path.is_null() && !model_path.is_string() {
        return Err(err(
            source,
            "model_path",
            "must be a string path or null",
            &model_path,
        ));
    }

    Ok(())
}

impl SeekKeys {
    /// Required binding names, in v1's DEFAULT_CONFIG insertion order.
    fn iter_keys(&self) -> [&'static str; 4] {
        ["forward_5", "backward_5", "forward_10", "backward_10"]
    }
}

// ---------------------------------------------------------------------------
// Python repr() for validation messages
// ---------------------------------------------------------------------------

/// Reproduces Python's `repr()` for JSON-derived values so messages match
/// v1's `f"... (got {got!r})"` exactly: `None`, `True`/`False`, `'strings'`,
/// plain numbers (float `1.0` keeps its `.0`), `[list]` / `{'dict'}` forms.
fn py_repr(v: &Value) -> String {
    match v {
        Value::Null => "None".into(),
        Value::Bool(b) => if *b { "True" } else { "False" }.into(),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.to_string()
            } else if let Some(u) = n.as_u64() {
                u.to_string()
            } else {
                let f = n.as_f64().unwrap_or(f64::NAN);
                if f.is_finite() && f.fract() == 0.0 && f.abs() < 1e16 {
                    format!("{f:.1}") // Python: repr(1.0) == "1.0"
                } else {
                    format!("{f}")
                }
            }
        }
        Value::String(s) => format!("'{}'", s.replace('\\', "\\\\").replace('\'', "\\'")),
        Value::Array(items) => {
            let inner: Vec<String> = items.iter().map(py_repr).collect();
            format!("[{}]", inner.join(", "))
        }
        Value::Object(map) => {
            let inner: Vec<String> = map
                .iter()
                .map(|(k, v)| format!("'{}': {}", k.replace('\\', "\\\\").replace('\'', "\\'"), py_repr(v)))
                .collect();
            format!("{{{}}}", inner.join(", "))
        }
    }
}
