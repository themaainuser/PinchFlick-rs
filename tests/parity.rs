//! tests/parity.rs — the 25 v1 selftests (selftest.py), same scenarios,
//! same numbers. This file IS the M2 gate: all 25 green in `cargo test`.
//!
//! Translation notes (v1 behavior wins, spec section 4):
//! - v1's `Point` -> `Landmark`, `make_hand` copied numerically verbatim
//! - Python `None` landmarks -> `Option<&[Landmark]>`; `str` sides/directions
//!   -> `Side` / `SeekDirection` enums compared with `== Some(..)`
//! - v1's `cfg(**overrides)` -> `cfg_with(|c| ..)` on `Config::default()`,
//!   which still runs `validate` exactly like v1's helper
//! - `seek_key_sequence(True, ...)` in v1 is rejected by its bool check;
//!   Rust passes `1` there (rejected by `< 5` instead — same Err either way)

use pinchflick_rs::config::{load_config, Config, ConfigError};
use pinchflick_rs::gestures::engine::{
    count_extended_fingers, hand_size, pinch_ratio, tap_side, GestureEngine, Landmark,
    INDEX_TIP,
};
use pinchflick_rs::gestures::pinch::PinchState;
use pinchflick_rs::gestures::scroll::ScrollDriver;
use pinchflick_rs::gestures::seek::{seek_key_sequence, DoubleTapSeeker, SeekDirection, Side};
use pinchflick_rs::sys::title::{get_foreground_window_title, is_youtube_title};

/// Build 21 landmarks. `extended = (index, middle, ring, pinky)` extended
/// upward; `pinched` = thumb tip touching the index fingertip; `dx` shifts
/// the whole hand sideways (right = positive).
fn make_hand(pinched: bool, extended: (bool, bool, bool, bool), dx: f64) -> Vec<Landmark> {
    let (idx, mid, ring, pk) = extended;
    let mut lm = vec![Landmark::new(0.0, 0.0); 21];

    lm[0] = Landmark::new(0.50 + dx, 0.75); // wrist
    lm[9] = Landmark::new(0.50 + dx, 0.55); // middle MCP -> hand size 0.20
    lm[5] = Landmark::new(0.45 + dx, 0.55);
    lm[13] = Landmark::new(0.55 + dx, 0.55);
    lm[17] = Landmark::new(0.60 + dx, 0.56);
    lm[1] = Landmark::new(0.42 + dx, 0.70);
    lm[2] = Landmark::new(0.40 + dx, 0.65);
    lm[3] = Landmark::new(0.41 + dx, 0.60);

    // index finger (8 = tip, 6 = pip)
    lm[6] = Landmark::new(0.45 + dx, 0.45);
    lm[7] = Landmark::new(0.445 + dx, if idx { 0.385 } else { 0.47 });
    lm[8] = Landmark::new(0.44 + dx, if idx { 0.32 } else { 0.49 });

    // middle finger
    lm[10] = Landmark::new(0.50 + dx, 0.43);
    lm[11] = Landmark::new(0.50 + dx, if mid { 0.36 } else { 0.45 });
    lm[12] = Landmark::new(0.50 + dx, if mid { 0.29 } else { 0.47 });

    // ring finger
    lm[14] = Landmark::new(0.55 + dx, 0.45);
    lm[15] = Landmark::new(0.555 + dx, if ring { 0.39 } else { 0.47 });
    lm[16] = Landmark::new(0.56 + dx, if ring { 0.33 } else { 0.49 });

    // pinky
    lm[18] = Landmark::new(0.60 + dx, 0.47);
    lm[19] = Landmark::new(0.605 + dx, if pk { 0.42 } else { 0.49 });
    lm[20] = Landmark::new(0.61 + dx, if pk { 0.37 } else { 0.51 });

    // thumb tip: touching index tip when pinched, relaxed otherwise
    if pinched {
        lm[4] = Landmark::new(lm[INDEX_TIP].x + 0.012, lm[INDEX_TIP].y + 0.012);
    } else {
        lm[4] = Landmark::new(0.38 + dx, 0.66);
    }
    lm
}

/// v1's `cfg(**overrides)`: defaults + overrides + validate.
fn cfg_with(f: impl FnOnce(&mut Config)) -> Config {
    let mut c = Config::default();
    f(&mut c);
    c.validate("<test>").expect("test config must validate");
    c
}

fn cfg() -> Config {
    cfg_with(|_| {})
}

fn expect_error(result: Result<Config, ConfigError>, fragment: &str, label: &str) {
    match result {
        Ok(_) => panic!("{label}: expected ConfigError, none raised"),
        Err(e) => assert!(
            e.to_string().contains(fragment),
            "{label}: expected '{fragment}' in error, got: {e}"
        ),
    }
}

/// Write `text` to a temp config file, run `body(path)`, clean up after.
fn with_config(text: &str, body: impl FnOnce(&str)) {
    let dir = tempfile::tempdir().expect("tempdir");
    let p = dir.path().join("cfg.json");
    std::fs::write(&p, text).expect("write config");
    body(p.to_str().expect("utf-8 path"));
}

// ---------------------------------------------------------------------------
// Landmark math
// ---------------------------------------------------------------------------
#[test]
fn test_pinch_ratio_bounds() {
    assert!(hand_size(&make_hand(false, (false, false, false, false), 0.0)) > 0.0);
    assert!(pinch_ratio(&make_hand(true, (false, false, false, false), 0.0)) < 0.35); // clearly a pinch
    let open = make_hand(false, (false, false, false, false), 0.0);
    assert!(pinch_ratio(&open) > 0.55); // clearly open
    // ratio is scale-invariant: shrink the whole hand toward its wrist
    let mut tiny = make_hand(false, (false, false, false, false), 0.0);
    let (wx, wy) = (tiny[0].x, tiny[0].y);
    for p in &mut tiny {
        p.x = wx + (p.x - wx) * 0.4;
        p.y = wy + (p.y - wy) * 0.4;
    }
    assert!((pinch_ratio(&tiny) - pinch_ratio(&open)).abs() < 1e-9);
}

#[test]
fn test_count_extended_fingers() {
    assert_eq!(
        count_extended_fingers(&make_hand(false, (true, false, false, false), 0.0)),
        1
    );
    assert_eq!(
        count_extended_fingers(&make_hand(false, (true, true, false, false), 0.0)),
        2
    );
    assert_eq!(
        count_extended_fingers(&make_hand(false, (true, true, true, true), 0.0)),
        4
    );
    assert_eq!(
        count_extended_fingers(&make_hand(false, (false, false, false, false), 0.0)),
        0
    );
}

#[test]
fn test_tap_side_mirrored_and_raw() {
    let left_hand = make_hand(true, (false, false, false, false), -0.15); // mid x ~0.30
    let right_hand = make_hand(true, (false, false, false, false), 0.15); // mid x ~0.60
    assert_eq!(tap_side(&left_hand, true), Side::Left);
    assert_eq!(tap_side(&right_hand, true), Side::Right);
    // raw (unflipped) camera space: left/right swap
    assert_eq!(tap_side(&left_hand, false), Side::Right);
    assert_eq!(tap_side(&right_hand, false), Side::Left);
}

// ---------------------------------------------------------------------------
// Pinch hysteresis
// ---------------------------------------------------------------------------
#[test]
fn test_pinch_hysteresis_single_edge() {
    let mut latch = PinchState::new(0.35, 0.55);
    assert!(!latch.update(Some(0.60)) && !latch.active); // open
    assert!(latch.update(Some(0.34)) && latch.active); // pinch -> edge
    assert!(!latch.update(Some(0.45)) && latch.active); // in the gap: stays
    assert!(!latch.update(Some(0.50)) && latch.active); // still stays
    assert!(!latch.update(Some(0.56)) && !latch.active); // released
    assert!(latch.update(Some(0.34)) && latch.active); // second pinch -> new edge
    assert!(!latch.update(None) && !latch.active); // hand lost
}

// ---------------------------------------------------------------------------
// Double-tap seeker
// ---------------------------------------------------------------------------
#[test]
fn test_double_tap_fires_once() {
    let mut seeker = DoubleTapSeeker::new(550, 600);
    assert_eq!(seeker.register(Side::Left, 0), None); // first tap
    assert_eq!(
        seeker.register(Side::Left, 400),
        Some(SeekDirection::Backward)
    );
    // state reset after firing
    assert_eq!(seeker.register(Side::Left, 500), None); // new pair starts
}

#[test]
fn test_double_tap_side_must_match_and_window_must_hold() {
    let mut seeker = DoubleTapSeeker::new(550, 600);
    assert_eq!(seeker.register(Side::Left, 0), None);
    assert_eq!(
        seeker.register(Side::Right, 300),
        None // side switch: pair restarts on right
    );
    assert_eq!(
        seeker.register(Side::Right, 400),
        Some(SeekDirection::Forward) // 100ms later: right pair fires
    );

    let mut slow = DoubleTapSeeker::new(550, 600);
    assert_eq!(slow.register(Side::Left, 0), None);
    assert_eq!(
        slow.register(Side::Left, 700),
        None // 700ms > 550ms window: too slow,
    );
    assert_eq!(
        slow.register(Side::Left, 1000),
        Some(SeekDirection::Backward) // so this pairs with the 700 tap
    );
}

#[test]
fn test_seek_cooldown_blocks_second_seek() {
    let mut seeker = DoubleTapSeeker::new(550, 600);
    assert_eq!(seeker.register(Side::Left, 0), None);
    assert_eq!(
        seeker.register(Side::Left, 400),
        Some(SeekDirection::Backward) // seek #1 at t=400
    );
    assert_eq!(seeker.register(Side::Right, 700), None);
    assert_eq!(
        seeker.register(Side::Right, 950),
        None // 950-400=550 < 600 -> blocked
    );
    assert_eq!(
        seeker.register(Side::Right, 1400),
        None // starts a fresh pair
    );
    assert_eq!(
        seeker.register(Side::Right, 1750),
        Some(SeekDirection::Forward) // 1750-400 > 600 -> allowed
    );
}

// ---------------------------------------------------------------------------
// Scroll driver
// ---------------------------------------------------------------------------
#[test]
fn test_scroll_driver_direction_and_confirm() {
    let mut d = ScrollDriver::new(5, 2, 0, false, 1, 2);
    assert_eq!(d.update(1, 0, true), 0); // frame 1: confirming
    assert_eq!(d.update(1, 33, true), 5); // frame 2: stable 1 finger -> up
    assert_eq!(d.update(1, 66, true), 5); // held: keeps scrolling
    assert_eq!(d.update(2, 99, true), 0); // changed: reconfirming
    assert_eq!(d.update(2, 132, true), -5); // stable 2 fingers -> down
    assert_eq!(d.update(0, 165, true), 0); // fist: no scroll
    assert_eq!(d.update(3, 198, true), 0); // 3 fingers: no scroll
}

#[test]
fn test_scroll_driver_gate_clears_pending_frames() {
    let mut d = ScrollDriver::new(5, 2, 0, false, 1, 2);
    assert_eq!(d.update(1, 0, true), 0); // half-confirmed
    assert_eq!(d.update(1, 33, false), 0); // gate closed (pinch!) -> dropped
    assert_eq!(d.update(1, 66, false), 0); // still closed
    assert_eq!(d.update(1, 99, true), 0); // gate open: starts fresh
    assert_eq!(d.update(1, 132, true), 5); // confirmed now
}

#[test]
fn test_scroll_driver_interval_and_invert() {
    let mut d = ScrollDriver::new(3, 1, 100, true, 1, 2);
    assert_eq!(d.update(1, 0, true), -3); // inverted: "up fingers" scrolls down
    assert_eq!(d.update(1, 50, true), 0); // throttled by interval
    assert_eq!(d.update(1, 101, true), -3); // interval elapsed
}

// ---------------------------------------------------------------------------
// Seek keys
// ---------------------------------------------------------------------------
#[test]
fn test_seek_key_sequences() {
    let keys = Config::default().seek_keys;
    assert_eq!(seek_key_sequence(10, "forward", &keys).unwrap(), ["l"]);
    assert_eq!(seek_key_sequence(10, "backward", &keys).unwrap(), ["j"]);
    assert_eq!(seek_key_sequence(5, "backward", &keys).unwrap(), ["left"]);
    assert_eq!(
        seek_key_sequence(15, "forward", &keys).unwrap(),
        ["l", "right"]
    );
    assert_eq!(
        seek_key_sequence(20, "backward", &keys).unwrap(),
        ["j", "j"]
    );
    assert_eq!(
        seek_key_sequence(25, "forward", &keys).unwrap(),
        ["l", "l", "right"]
    );
    // v1 also rejects True; 1 stands in for it (see file header note)
    for bad in [7_i64, 0, -10, 12, 1] {
        assert!(
            seek_key_sequence(bad, "forward", &keys).is_err(),
            "seek_seconds={bad} should be rejected"
        );
    }
    assert!(
        seek_key_sequence(10, "sideways", &keys).is_err(),
        "bad direction should be rejected"
    );
}

// ---------------------------------------------------------------------------
// YouTube detection
// ---------------------------------------------------------------------------
#[test]
fn test_youtube_title_markers() {
    let m = Config::default().youtube_title_markers;
    assert!(is_youtube_title(Some("Funny Cat Video - YouTube"), &m));
    assert!(is_youtube_title(
        Some("Funny Cat Video \u{2013} YouTube"),
        &m
    ));
    assert!(is_youtube_title(Some("FUNNY CAT - YOUTUBE"), &m)); // case-insensitive
    // regression: real-world titles from this user's machine (Brave appends
    // its own name AFTER the YouTube marker)
    assert!(is_youtube_title(
        Some("React Performance Optimizations: How to Fix a Slow App - YouTube - Brave"),
        &m
    ));
    assert!(is_youtube_title(Some("Some Video - YouTube - Google Chrome"), &m));
    assert!(is_youtube_title(
        Some("Some Video - YouTube \u{2014} Mozilla Firefox"),
        &m
    ));
    assert!(!is_youtube_title(Some(""), &m));
    assert!(!is_youtube_title(Some("My blog post about YouTube"), &m)); // no marker
    assert!(!is_youtube_title(Some("My rant - YouTube is great"), &m)); // marker not at tail
    assert!(!is_youtube_title(Some("GitHub - YouTube.js"), &m));
    assert!(!is_youtube_title(None, &m));
}

#[test]
fn test_foreground_title_returns_string() {
    // v1 asserts isinstance(title, str) — in Rust the type is String by
    // construction; the value of the test is exercising the Win32 FFI call
    // (must not crash) and bounding the result as a real title would be.
    let title: String = get_foreground_window_title();
    assert!(title.len() < 4096, "implausible title length {}", title.len());
}

// ---------------------------------------------------------------------------
// Config loading + validation
// ---------------------------------------------------------------------------
#[test]
fn test_config_defaults_load() {
    let c = load_config(None).expect("defaults must load");
    let v = serde_json::to_value(&c).expect("config serializes");
    let obj = v.as_object().expect("config is an object");
    let def = serde_json::to_value(Config::default()).expect("defaults serialize");
    let def_obj = def.as_object().unwrap();
    let mut keys: Vec<&String> = obj.keys().collect();
    keys.sort();
    let mut def_keys: Vec<&String> = def_obj.keys().collect();
    def_keys.sort();
    assert_eq!(keys, def_keys, "loaded keys must equal default keys");
    assert_eq!(keys.len(), 32, "the schema has exactly 32 keys");
    assert!(v["seek_keys"].is_object());
}

#[test]
fn test_config_partial_merge() {
    with_config(
        r#"{"seek_seconds": 15, "scroll_speed": 8, "seek_keys": {"forward_10": "k"}}"#,
        |path| {
            let c = load_config(Some(path)).expect("partial config loads");
            assert_eq!(c.seek_seconds, 15);
            assert_eq!(c.scroll_speed, 8);
            assert_eq!(c.seek_keys.forward_10, "k"); // overridden
            assert_eq!(c.seek_keys.backward_10, "j"); // rest kept
            assert_eq!(
                c.pinch_on_ratio,
                Config::default().pinch_on_ratio // untouched key survives
            );
        },
    );
}

#[test]
fn test_config_errors_are_specific() {
    let dir = tempfile::tempdir().expect("tempdir");
    let missing = dir.path().join("__nope__.json");
    expect_error(
        load_config(Some(missing.to_str().expect("utf-8"))),
        "not found",
        "missing explicit file",
    );

    let expect_load_error = |text: &str, fragment: &str, label: &str| {
        with_config(text, |path| {
            expect_error(load_config(Some(path)), fragment, label);
        });
    };

    expect_load_error("{not json", "not valid JSON", "bad JSON");
    expect_load_error(r#"{"scroll_spd": 5}"#, "unknown key", "unknown key");
    expect_load_error(
        r#"{"pinch_on_ratio": 0.6}"#,
        "smaller than pinch_off_ratio",
        "on > off",
    );
    expect_load_error(
        r#"{"seek_seconds": 12}"#,
        "multiple of 5",
        "seek 12",
    );
    expect_load_error(
        r#"{"scroll_confirm_frames": 0}"#,
        ">= 1",
        "confirm 0",
    );
    expect_load_error(
        r#"{"scroll_speed": "5"}"#,
        "whole number",
        "speed str",
    );
    expect_load_error(
        r#"{"scroll_up_fingers": 2}"#,
        "must differ",
        "up == down",
    );
    expect_load_error(
        r#"{"mirror_display": "yes"}"#,
        "true or false",
        "bool str",
    );
    expect_load_error(r#"{"seek_keys": null}"#, "object", "null keys");
    expect_load_error(
        r#"{"tap_side_dead_zone": 0.6}"#,
        "must be <= 0.49",
        "dead zone too wide",
    );
}

// ---------------------------------------------------------------------------
// Full engine pipeline (the part that used to be unreliable)
// ---------------------------------------------------------------------------
#[test]
fn test_engine_scroll_works_without_youtube() {
    // Scrolling must fire on any page -- no YouTube involved.
    let mut e = GestureEngine::new(&cfg(), true);
    let open_one = make_hand(false, (true, false, false, false), 0.0);
    let r1 = e.frame(Some(&open_one), 0, false);
    let r2 = e.frame(Some(&open_one), 33, false);
    assert_eq!(r1.scroll_clicks, 0);
    assert_eq!(r2.scroll_clicks, 5);

    let open_two = make_hand(false, (true, true, false, false), 0.0);
    assert_eq!(e.frame(Some(&open_two), 66, false).scroll_clicks, 0); // changing
    assert_eq!(e.frame(Some(&open_two), 99, false).scroll_clicks, -5);
}

#[test]
fn test_engine_seek_fires_only_on_youtube() {
    let open_one = make_hand(false, (true, false, false, false), 0.0);
    let pinch = make_hand(true, (false, false, false, false), 0.0);

    // -- not YouTube: double pinch must NOT seek
    let mut e = GestureEngine::new(&cfg(), true);
    e.frame(Some(&open_one), 0, false);
    e.frame(Some(&pinch), 100, false); // tap 1
    e.frame(Some(&open_one), 150, false);
    let r = e.frame(Some(&pinch), 450, false); // tap 2 within window
    assert_eq!(r.seek, None, "seek must not fire off YouTube: {r:?}");
    assert_eq!(r.seek_blocked.as_deref(), Some("not on YouTube"));
    assert_eq!(r.scroll_clicks, 0); // and no phantom scroll from the pinch

    // -- IS YouTube: same sequence must seek exactly once
    let mut e2 = GestureEngine::new(&cfg(), true);
    e2.frame(Some(&open_one), 0, true);
    e2.frame(Some(&pinch), 100, true);
    e2.frame(Some(&open_one), 150, true);
    let r = e2.frame(Some(&pinch), 450, true);
    assert_eq!(
        r.seek,
        Some(SeekDirection::Backward),
        "expected backward seek, got {r:?}"
    );
}

#[test]
fn test_engine_direction_by_frame_side() {
    // v1 keeps this variable named-but-unused; mirrored here with `_` to
    // stay warning-free (parity: it never participates in the assertions)
    let _keys_left = make_hand(true, (false, false, false, false), -0.15);
    let keys_right = make_hand(true, (false, false, false, false), 0.15);
    let open_h = make_hand(false, (true, false, false, false), 0.0);

    let mut e = GestureEngine::new(&cfg(), true);
    e.frame(Some(&open_h), 0, true);
    e.frame(Some(&keys_right), 100, true);
    e.frame(Some(&open_h), 150, true);
    let r = e.frame(Some(&keys_right), 450, true);
    assert_eq!(r.seek, Some(SeekDirection::Forward), "{r:?}");
}

#[test]
fn test_no_seek_cooldown_by_default_back_to_back_seeks() {
    // seek_cooldown_ms defaults to 0: two double-taps in a row both fire.
    assert_eq!(Config::default().seek_cooldown_ms, 0);
    let mut e = GestureEngine::new(&cfg(), true);
    let open_h = make_hand(false, (true, false, false, false), 0.0);
    let right = make_hand(true, (false, false, false, false), 0.15);

    let pair = |engine: &mut GestureEngine, t0: i64| {
        engine.frame(Some(&open_h), t0, true);
        engine.frame(Some(&right), t0 + 100, true); // tap 1
        engine.frame(Some(&open_h), t0 + 150, true);
        engine.frame(Some(&right), t0 + 450, true) // tap 2
    };

    let r1 = pair(&mut e, 0);
    assert_eq!(r1.seek, Some(SeekDirection::Forward), "{r1:?}");
    // second pair starts the moment the first seek fired -> must seek again
    let r2 = pair(&mut e, 460);
    assert_eq!(
        r2.seek,
        Some(SeekDirection::Forward),
        "cooldown still blocking: {r2:?}"
    );
}

#[test]
fn test_center_dead_zone_refuses_ambiguous_pinches() {
    // A pinch on the middle line must not be assigned a random side.
    // It is refused with a visible message instead, and a decisive left
    // pinch right after still registers and seeks normally.
    let mut e = GestureEngine::new(&cfg(), true);
    let open_h = make_hand(false, (true, false, false, false), 0.0);
    let center = make_hand(true, (false, false, false, false), 0.054); // midpoint x=0.500

    e.frame(Some(&open_h), 0, true);
    let r = e.frame(Some(&center), 100, true);
    assert_eq!(r.tap, None);
    assert_eq!(r.seek, None, "{r:?}");
    assert!(r.tap_status.contains("middle"), "{}", r.tap_status);
    e.frame(Some(&open_h), 150, true);
    let r = e.frame(Some(&center), 450, true);
    assert_eq!(r.tap, None, "center pinch paired up: {r:?}");
    assert_eq!(r.seek, None);

    let left = make_hand(true, (false, false, false, false), -0.15);
    e.frame(Some(&open_h), 500, true);
    let r = e.frame(Some(&left), 560, true);
    assert_eq!(r.tap, Some(Side::Left), "{r:?}");
    e.frame(Some(&open_h), 610, true);
    let r = e.frame(Some(&left), 900, true);
    assert_eq!(r.seek, Some(SeekDirection::Backward), "{r:?}");
    assert_eq!(r.tap, Some(Side::Left), "{r:?}");
}

#[test]
fn test_engine_no_phantom_scroll_during_or_right_after_pinch() {
    // The old failure: curling the index to pinch leaked scroll events.
    let mut e = GestureEngine::new(&cfg(), true);
    let mut t = 0;

    // steady 1-finger scroll beforehand
    let open_one = make_hand(false, (true, false, false, false), 0.0);
    e.frame(Some(&open_one), t, true);
    t += 33;
    let r = e.frame(Some(&open_one), t, true);
    t += 33;
    assert_eq!(r.scroll_clicks, 5);

    // pinch #1 -- middle finger happens to be extended (count would be 1!)
    let pinch_mid = make_hand(true, (false, true, false, false), 0.0);
    let r = e.frame(Some(&pinch_mid), t, true);
    t += 33;
    assert_eq!(r.scroll_clicks, 0, "pinch leaked scroll: {r:?}");
    assert!(r.pinched);
    let r = e.frame(Some(&pinch_mid), t, true);
    t += 33;
    assert_eq!(r.scroll_clicks, 0);

    // release, re-pinch within the double-tap window -> seek fires
    e.frame(Some(&open_one), t, true);
    t += 33;
    let r = e.frame(Some(&pinch_mid), t, true);
    assert_eq!(r.seek, Some(SeekDirection::Backward), "{r:?}");
    assert_eq!(r.scroll_clicks, 0);

    // immediately after the seek, an open 1-finger hand must stay quiet
    let r = e.frame(Some(&open_one), t, true);
    t += 33;
    assert_eq!(r.scroll_clicks, 0, "scroll leaked into post-seek window: {r:?}");
    let r = e.frame(Some(&open_one), t, true);
    t += 33;
    assert_eq!(r.scroll_clicks, 0);

    // past the suppression window it scrolls again
    t += 600;
    e.frame(Some(&open_one), t, true);
    t += 33;
    let r = e.frame(Some(&open_one), t, true);
    assert_eq!(r.scroll_clicks, 5, "scroll never recovered: {r:?}");
}

#[test]
fn test_engine_no_hand_resets_pinch() {
    let mut e = GestureEngine::new(&cfg(), true);
    let pinch = make_hand(true, (false, false, false, false), 0.0);
    e.frame(Some(&pinch), 0, true);
    assert!(e.pinch.active);
    let r = e.frame(None, 33, true);
    assert!(!e.pinch.active);
    assert_eq!(r.scroll_clicks, 0);
    assert_eq!(r.seek, None);
    assert_eq!(r.ratio, None);
    // a one-frame hand loss doesn't break tap pairing (tap@0 + pinch@100 = seek)
    let r = e.frame(Some(&pinch), 100, true);
    assert_eq!(r.seek, Some(SeekDirection::Backward), "{r:?}");
}

#[test]
fn test_engine_scroll_disabled_and_seek_disabled() {
    let c = cfg_with(|c| {
        c.scroll_enabled = false;
        c.seek_enabled = true;
    });
    let mut e = GestureEngine::new(&c, true);
    let open_one = make_hand(false, (true, false, false, false), 0.0);
    e.frame(Some(&open_one), 0, true);
    let r = e.frame(Some(&open_one), 33, true);
    assert_eq!(r.scroll_clicks, 0);

    let c2 = cfg_with(|c| {
        c.seek_enabled = false;
        c.seek_requires_youtube = true;
    });
    let mut e2 = GestureEngine::new(&c2, true);
    let pinch = make_hand(true, (false, false, false, false), 0.0);
    e2.frame(Some(&open_one), 0, true);
    e2.frame(Some(&pinch), 100, true);
    e2.frame(Some(&open_one), 150, true);
    let r = e2.frame(Some(&pinch), 450, true);
    assert_eq!(r.seek, None);
    assert_eq!(
        r.seek_blocked.as_deref(),
        Some("seek is disabled in config")
    );
}

#[test]
fn test_engine_raw_camera_space_flips_side() {
    // With an unflipped frame, left/right must be swapped to match the view.
    let mut e = GestureEngine::new(&cfg(), false);
    let right_screen = make_hand(true, (false, false, false, false), -0.15); // raw-left = display-right
    let open_h = make_hand(false, (true, false, false, false), 0.0);
    e.frame(Some(&open_h), 0, true);
    e.frame(Some(&right_screen), 100, true);
    e.frame(Some(&open_h), 150, true);
    let r = e.frame(Some(&right_screen), 450, true);
    assert_eq!(r.seek, Some(SeekDirection::Forward), "{r:?}");
}
