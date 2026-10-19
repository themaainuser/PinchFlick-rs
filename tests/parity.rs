//! tests/parity.rs — the 25 v1 selftests (selftest.py), same scenarios,
//! same numbers. This file IS the M2 gate: all 25 green in `cargo test`.
//!
//! The check bodies now live in `pinchflick_rs::selftest` (single source of
//! truth — the same code `--selftest` runs at runtime, no camera needed).
//! Each `#[test]` here is a thin wrapper that surfaces a failed check as a
//! test failure; names are kept verbatim so suite output stays stable.

use pinchflick_rs::selftest;

macro_rules! parity {
    ($fn_name:ident, $check:ident) => {
        #[test]
        fn $fn_name() {
            if let Err(e) = selftest::$check() {
                panic!("{e}");
            }
        }
    };
}

// Landmark math
parity!(test_pinch_ratio_bounds, check_pinch_ratio_bounds);
parity!(test_count_extended_fingers, check_count_extended_fingers);
parity!(test_tap_side_mirrored_and_raw, check_tap_side_mirrored_and_raw);

// Pinch hysteresis
parity!(test_pinch_hysteresis_single_edge, check_pinch_hysteresis_single_edge);

// Double-tap seeker
parity!(test_double_tap_fires_once, check_double_tap_fires_once);
parity!(
    test_double_tap_side_must_match_and_window_must_hold,
    check_double_tap_side_must_match_and_window_must_hold
);
parity!(test_seek_cooldown_blocks_second_seek, check_seek_cooldown_blocks_second_seek);

// Scroll driver
parity!(test_scroll_driver_direction_and_confirm, check_scroll_driver_direction_and_confirm);
parity!(test_scroll_driver_gate_clears_pending_frames, check_scroll_driver_gate_clears_pending_frames);
parity!(test_scroll_driver_interval_and_invert, check_scroll_driver_interval_and_invert);

// Seek keys
parity!(test_seek_key_sequences, check_seek_key_sequences);

// YouTube detection
parity!(test_youtube_title_markers, check_youtube_title_markers);
parity!(test_foreground_title_returns_string, check_foreground_title_returns_string);

// Config loading + validation
parity!(test_config_defaults_load, check_config_defaults_load);
parity!(test_config_partial_merge, check_config_partial_merge);
parity!(test_config_errors_are_specific, check_config_errors_are_specific);

// Full engine pipeline
parity!(test_engine_scroll_works_without_youtube, check_engine_scroll_works_without_youtube);
parity!(test_engine_seek_fires_only_on_youtube, check_engine_seek_fires_only_on_youtube);
parity!(test_engine_direction_by_frame_side, check_engine_direction_by_frame_side);
parity!(
    test_no_seek_cooldown_by_default_back_to_back_seeks,
    check_no_seek_cooldown_by_default_back_to_back_seeks
);
parity!(test_center_dead_zone_refuses_ambiguous_pinches, check_center_dead_zone_refuses_ambiguous_pinches);
parity!(
    test_engine_no_phantom_scroll_during_or_right_after_pinch,
    check_engine_no_phantom_scroll_during_or_right_after_pinch
);
parity!(test_engine_no_hand_resets_pinch, check_engine_no_hand_resets_pinch);
parity!(test_engine_scroll_disabled_and_seek_disabled, check_engine_scroll_disabled_and_seek_disabled);
parity!(test_engine_raw_camera_space_flips_side, check_engine_raw_camera_space_flips_side);
