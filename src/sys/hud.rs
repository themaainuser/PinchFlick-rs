//! HUD scene model — v1's on-frame overlay state + text, pure logic (no GUI
//! framework here; the egui wiring lives in `main.rs` as the M4 spike).
//!
//! v1 parity target: `PinchFlick/controller.py` `_put`/`draw_hand` + the
//! draw section of `run()` (lines 247-286). v1 draws the HUD ON the camera
//! frame and lets OpenCV scale the annotated frame into the 480x360 window,
//! so every coordinate below is in FRAME space (640x480) — the renderer
//! applies the frame->window scale, which is uniform because both are 4:3.
//!
//! All v1 colors are BGR (cv2 convention); the constants here are the same
//! values converted to RGB for the renderer, conversion pinned by test.
//!
//! Timings (controller.py): flash lives 700 ms (`ts + 700`, visible while
//! `ts < flash_until`), seek-blocked message 2500 ms, fps is an EMA
//! (`0.9*fps + 0.1*(1/dt)`).

use crate::gestures::engine::{EngineResult, Landmark};
use crate::gestures::seek::{SeekDirection, Side};

/// (r, g, b) triple — renderer-agnostic color.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rgb(pub u8, pub u8, pub u8);

// v1 BGR -> RGB (controller.py draw section).
/// BGR (60, 60, 255) — hand skeleton while pinched.
pub const HAND_PINCHED_RGB: Rgb = Rgb(255, 60, 60);
/// BGR (0, 200, 0) — hand skeleton while open.
pub const HAND_OPEN_RGB: Rgb = Rgb(0, 200, 0);
/// BGR (130, 130, 130) — center divider.
pub const DIVIDER_RGB: Rgb = Rgb(130, 130, 130);
/// BGR (240, 240, 240) — default status text.
pub const STATUS_TEXT_RGB: Rgb = Rgb(240, 240, 240);
/// BGR (255, 220, 120) — tap-status line.
pub const TAP_STATUS_TEXT_RGB: Rgb = Rgb(120, 220, 255);
/// BGR (80, 80, 255) — seek-blocked message.
pub const BLOCK_TEXT_RGB: Rgb = Rgb(255, 80, 80);
/// BGR (170, 220, 170) — debug overlay line.
pub const DEBUG_TEXT_RGB: Rgb = Rgb(170, 220, 170);
/// BGR (255, 200, 80) — tap flash.
pub const FLASH_TAP_RGB: Rgb = Rgb(80, 200, 255);
/// BGR (90, 230, 90) — seek flash.
pub const FLASH_SEEK_RGB: Rgb = Rgb(90, 230, 90);

/// Flash lifetime in ms (controller.py: `flash_until = ts + 700`).
pub const FLASH_MS: i64 = 700;
/// Seek-blocked message lifetime in ms (controller.py: `block_until = ts + 2500`).
pub const BLOCK_MS: i64 = 2500;

/// One active flash: which side, what text, what color, until when.
#[derive(Debug, Clone, PartialEq)]
pub struct Flash {
    pub side: Side,
    pub text: String,
    pub color: Rgb,
    pub until_ms: i64,
}

/// The HUD's mutable state — v1's `flash_side/flash_until/flash_text/
/// flash_color`, `block_text/block_until`, `fps`, `show_debug` locals.
#[derive(Debug, Clone)]
pub struct HudState {
    pub show_debug: bool,
    pub fps: f64,
    last_frame_time: Option<f64>,
    block_text: Option<String>,
    block_until_ms: i64,
    flash: Option<Flash>,
}

impl HudState {
    pub fn new(show_debug: bool) -> Self {
        Self {
            show_debug,
            fps: 0.0,
            last_frame_time: None,
            block_text: None,
            block_until_ms: 0,
            flash: None,
        }
    }

    /// v1's `d` key: toggle the debug overlay line.
    pub fn toggle_debug(&mut self) {
        self.show_debug = !self.show_debug;
    }

    /// v1's fps EMA, once per frame: `fps = 0.9*fps + 0.1*(1/max(dt, 1e-6))`
    /// with `dt` = wall-clock seconds since the previous frame. The first
    /// call has no previous frame and only arms `last_frame_time` (v1 seeds
    /// it right before the loop, so its first EMA step covers startup; ours
    /// covers first-capture time — same formula, cosmetic difference).
    pub fn update_fps(&mut self, now_secs: f64) {
        if let Some(last) = self.last_frame_time {
            let dt = (now_secs - last).max(1e-6);
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt);
        }
        self.last_frame_time = Some(now_secs);
    }

    /// v1's visual-echo + seek-blocked bookkeeping (controller.py:225-245),
    /// run once per engine result. `seek_seconds` feeds the flash text.
    pub fn on_engine_result(&mut self, res: &EngineResult, ts_ms: i64, seek_seconds: i64) {
        if let Some(blocked) = &res.seek_blocked {
            self.block_text = Some(format!("Seek ignored -- {blocked}"));
            self.block_until_ms = ts_ms + BLOCK_MS;
        }
        // v1: tap flash only when this frame did NOT also seek...
        if let Some(side) = res.tap && res.seek.is_none() {
            let text = match side {
                Side::Left => "\u{25c0} tap again!".to_string(),
                Side::Right => "tap again! \u{25b6}".to_string(),
            };
            self.flash = Some(Flash {
                side,
                text,
                color: FLASH_TAP_RGB,
                until_ms: ts_ms + FLASH_MS,
            });
        }
        // ...and a seek flash overwrites everything (v1 runs this block
        // second, unconditionally).
        if let Some(dir) = res.seek {
            let (side, text) = match dir {
                SeekDirection::Backward => (
                    Side::Left,
                    format!("\u{25c0} {seek_seconds}s"),
                ),
                SeekDirection::Forward => (
                    Side::Right,
                    format!("{seek_seconds}s \u{25b6}"),
                ),
            };
            self.flash = Some(Flash {
                side,
                text,
                color: FLASH_SEEK_RGB,
                until_ms: ts_ms + FLASH_MS,
            });
        }
    }

    /// The flash to draw at `ts_ms`, if any (v1: `if flash_text and ts < flash_until`).
    pub fn flash_at(&self, ts_ms: i64) -> Option<&Flash> {
        self.flash
            .as_ref()
            .filter(|f| ts_ms < f.until_ms)
    }

    /// The seek-blocked line to draw at `ts_ms`, if any (v1: `if block_text
    /// and ts < block_until`).
    pub fn block_text_at(&self, ts_ms: i64) -> Option<&str> {
        self.block_text
            .as_deref()
            .filter(|_| ts_ms < self.block_until_ms)
    }
}

/// v1: `"no hand" if out.ratio is None else f"{out.ratio:.2f}"`.
pub fn ratio_text(ratio: Option<f64>) -> String {
    match ratio {
        None => "no hand".to_string(),
        Some(r) => format!("{r:.2}"),
    }
}

/// v1's first status line (controller.py:258-262), exact spacing.
pub fn status_line(on_youtube: bool, res: &EngineResult) -> String {
    let yt = if on_youtube { "SEEK ON" } else { "off" };
    let pinched = if res.pinched { "  <PINCHED>" } else { "" };
    format!(
        "YouTube: {yt}    pinch: {}{pinched}    fingers up: {}",
        ratio_text(res.ratio),
        res.fingers
    )
}

/// v1's debug overlay line (controller.py:266-269): title truncated to 44
/// chars, empty title becomes "(none)".
pub fn debug_line(fps: f64, scroll_clicks: i32, seeks: u64, title: &str) -> String {
    let head: String = title.chars().take(44).collect();
    let window = if head.is_empty() { "(none)" } else { &head };
    format!("fps: {fps:.1}  scroll: {scroll_clicks:+}  seeks: {seeks}  window: {window}")
}

/// v1's per-frame window title (controller.py:281-282).
pub fn short_title(on_youtube: bool, ratio: Option<f64>, scroll_clicks: i32, seeks: u64) -> String {
    let yt = if on_youtube { "YT" } else { "--" };
    format!(
        "[{yt}] pinch {} | scroll {scroll_clicks:+} | seek x{seeks}",
        ratio_text(ratio)
    )
}

/// v1 window placement (controller.py:146): `x = max(8, sw - 500)`, `y = 16`
/// — the 480-wide window sits 20px in from the right edge of a 1920 screen.
pub fn window_position(screen_w: f32) -> (f32, f32) {
    ((screen_w - 500.0).max(8.0), 16.0)
}

/// v1 flash x-anchor (controller.py:275): left flashes start 14px in,
/// right flashes end 14px from the right edge (`w - tw - 14`). `text_w` is
/// the measured flash text width in frame space.
pub fn flash_anchor_x(side: Side, frame_w: f32, text_w: f32) -> f32 {
    match side {
        Side::Left => 14.0,
        Side::Right => frame_w - text_w - 14.0,
    }
}

/// v1's hand skeleton pixel projection (controller.py:80):
/// `(int(p.x * w), int(p.y * h))` — truncation, not rounding.
pub fn skeleton_points(landmarks: &[Landmark], w: u32, h: u32) -> Vec<(f32, f32)> {
    landmarks
        .iter()
        .map(|p| ((p.x * w as f64) as u32 as f32, (p.y * h as f64) as u32 as f32))
        .collect()
}

/// Frame-space -> window-space projection for the renderer. v1 draws the
/// HUD ON the 640x480 frame and lets cv2.imshow(WINDOW_NORMAL) stretch the
/// result to the window exactly (aspect ignored), so the two axes scale
/// independently (`sx = win_w / frame_w`, `sy = win_h / frame_h`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Scaler {
    pub sx: f32,
    pub sy: f32,
}

impl Scaler {
    pub fn new(frame_w: f32, frame_h: f32, win_w: f32, win_h: f32) -> Self {
        Self {
            sx: win_w / frame_w,
            sy: win_h / frame_h,
        }
    }

    pub fn pt(&self, p: (f32, f32)) -> (f32, f32) {
        (p.0 * self.sx, p.1 * self.sy)
    }
}

/// v1's skeleton line list (gesture_core.py:32-39): 20 finger/palm-chain
/// pairs + 3 palm arches, verbatim order.
pub const HAND_CONNECTIONS: [(usize, usize); 23] = [
    (0, 1), (1, 2), (2, 3), (3, 4),          // thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          // index
    (0, 9), (9, 10), (10, 11), (11, 12),     // middle
    (0, 13), (13, 14), (14, 15), (15, 16),   // ring
    (0, 17), (17, 18), (18, 19), (19, 20),   // pinky
    (5, 9), (9, 13), (13, 17),               // palm
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gestures::engine::EngineResult;

    /// EngineResult with every field defaulted, for targeted mutations.
    fn base() -> EngineResult {
        EngineResult {
            ratio: None,
            fingers: 0,
            pinched: false,
            scroll_clicks: 0,
            seek: None,
            seek_blocked: None,
            tap_status: String::new(),
            side: None,
            tap: None,
        }
    }

    // --- colors: v1 BGR -> RGB conversion, pinned -------------------------

    #[test]
    fn colors_are_v1_bgr_values_converted_to_rgb() {
        // v1 (cv2 BGR) -> ours (RGB): red channel and blue channel swap.
        assert_eq!(HAND_PINCHED_RGB, Rgb(255, 60, 60)); // BGR (60,60,255)
        assert_eq!(HAND_OPEN_RGB, Rgb(0, 200, 0)); // BGR (0,200,0)
        assert_eq!(DIVIDER_RGB, Rgb(130, 130, 130));
        assert_eq!(STATUS_TEXT_RGB, Rgb(240, 240, 240));
        assert_eq!(TAP_STATUS_TEXT_RGB, Rgb(120, 220, 255)); // BGR (255,220,120)
        assert_eq!(BLOCK_TEXT_RGB, Rgb(255, 80, 80)); // BGR (80,80,255)
        assert_eq!(DEBUG_TEXT_RGB, Rgb(170, 220, 170));
        assert_eq!(FLASH_TAP_RGB, Rgb(80, 200, 255)); // BGR (255,200,80)
        assert_eq!(FLASH_SEEK_RGB, Rgb(90, 230, 90));
    }

    // --- flash state machine ---------------------------------------------

    #[test]
    fn tap_left_fires_left_flash_with_tap_text() {
        let mut hud = HudState::new(false);
        let mut res = base();
        res.tap = Some(Side::Left);
        hud.on_engine_result(&res, 1000, 10);
        let f = hud.flash_at(1000).expect("flash");
        assert_eq!(f.side, Side::Left);
        assert_eq!(f.text, "\u{25c0} tap again!");
        assert_eq!(f.color, FLASH_TAP_RGB);
        assert_eq!(f.until_ms, 1700);
    }

    #[test]
    fn tap_right_fires_right_flash() {
        let mut hud = HudState::new(false);
        let mut res = base();
        res.tap = Some(Side::Right);
        hud.on_engine_result(&res, 500, 10);
        let f = hud.flash_at(500).expect("flash");
        assert_eq!(f.side, Side::Right);
        assert_eq!(f.text, "tap again! \u{25b6}");
    }

    #[test]
    fn seek_backward_fires_left_seek_flash() {
        let mut hud = HudState::new(false);
        let mut res = base();
        res.seek = Some(SeekDirection::Backward);
        hud.on_engine_result(&res, 2000, 10);
        let f = hud.flash_at(2000).expect("flash");
        assert_eq!(f.side, Side::Left);
        assert_eq!(f.text, "\u{25c0} 10s");
        assert_eq!(f.color, FLASH_SEEK_RGB);
        assert_eq!(f.until_ms, 2700);
    }

    #[test]
    fn seek_forward_fires_right_seek_flash_with_config_seconds() {
        let mut hud = HudState::new(false);
        let mut res = base();
        res.seek = Some(SeekDirection::Forward);
        hud.on_engine_result(&res, 2000, 15);
        let f = hud.flash_at(2000).expect("flash");
        assert_eq!(f.side, Side::Right);
        assert_eq!(f.text, "15s \u{25b6}");
        assert_eq!(f.color, FLASH_SEEK_RGB);
    }

    #[test]
    fn seek_overwrites_tap_flash_same_frame() {
        // v1 runs the tap block guarded by `not out.seek`, then the seek
        // block unconditionally — a frame with both shows the seek flash.
        let mut hud = HudState::new(false);
        let mut res = base();
        res.tap = Some(Side::Left);
        res.seek = Some(SeekDirection::Forward);
        hud.on_engine_result(&res, 3000, 10);
        let f = hud.flash_at(3000).expect("flash");
        assert_eq!(f.side, Side::Right);
        assert_eq!(f.color, FLASH_SEEK_RGB);
    }

    #[test]
    fn flash_lives_exactly_700ms_visible_until_ts_plus_699() {
        let mut hud = HudState::new(false);
        let mut res = base();
        res.tap = Some(Side::Left);
        hud.on_engine_result(&res, 1000, 10);
        assert!(hud.flash_at(1699).is_some());
        assert!(hud.flash_at(1700).is_none());
    }

    #[test]
    fn no_flash_initially() {
        let hud = HudState::new(false);
        assert!(hud.flash_at(0).is_none());
    }

    #[test]
    fn expired_flash_stays_expired_on_later_checks() {
        let mut hud = HudState::new(false);
        let mut res = base();
        res.tap = Some(Side::Right);
        hud.on_engine_result(&res, 1000, 10);
        assert!(hud.flash_at(9999).is_none());
    }

    // --- seek-blocked message --------------------------------------------

    #[test]
    fn seek_blocked_sets_message_for_2500ms() {
        let mut hud = HudState::new(false);
        let mut res = base();
        res.seek_blocked = Some("YouTube tab not focused".to_string());
        hud.on_engine_result(&res, 4000, 10);
        assert_eq!(
            hud.block_text_at(4000),
            Some("Seek ignored -- YouTube tab not focused")
        );
        assert_eq!(hud.block_text_at(6499), Some("Seek ignored -- YouTube tab not focused"));
        assert_eq!(hud.block_text_at(6500), None);
    }

    #[test]
    fn no_block_message_initially() {
        let hud = HudState::new(false);
        assert_eq!(hud.block_text_at(0), None);
    }

    // --- fps EMA ----------------------------------------------------------

    #[test]
    fn fps_first_call_only_arms_the_clock() {
        let mut hud = HudState::new(false);
        hud.update_fps(10.0);
        assert_eq!(hud.fps, 0.0);
    }

    #[test]
    fn fps_ema_matches_v1_formula() {
        let mut hud = HudState::new(false);
        hud.update_fps(10.0);
        hud.update_fps(10.05); // dt = 0.05 -> 1/dt = 20
        assert!((hud.fps - 2.0).abs() < 1e-9); // 0.9*0 + 0.1*20
        hud.update_fps(10.10); // dt = 0.05 again -> 0.9*2 + 0.1*20
        assert!((hud.fps - 3.8).abs() < 1e-9);
    }

    #[test]
    fn fps_dt_clamped_at_one_microsecond() {
        let mut hud = HudState::new(false);
        hud.update_fps(1.0);
        hud.update_fps(1.0); // dt = 0 -> clamped to 1e-6 -> 1e6 * 0.1
        assert!((hud.fps - 100_000.0).abs() < 1e-3);
    }

    // --- debug toggle -----------------------------------------------------

    #[test]
    fn toggle_debug_flips() {
        let mut hud = HudState::new(false);
        assert!(!hud.show_debug);
        hud.toggle_debug();
        assert!(hud.show_debug);
        hud.toggle_debug();
        assert!(!hud.show_debug);
    }

    // --- text formats -----------------------------------------------------

    #[test]
    fn ratio_text_no_hand_and_fixed_two_decimals() {
        assert_eq!(ratio_text(None), "no hand");
        assert_eq!(ratio_text(Some(0.42)), "0.42");
        // 0.425's f64 repr is 0.4249999..., so both Rust {:.2} and Python
        // .2f round DOWN to "0.42" (verified against the v1 venv).
        assert_eq!(ratio_text(Some(0.425)), "0.42");
        assert_eq!(ratio_text(Some(1.0)), "1.00");
    }

    #[test]
    fn status_line_exact_v1_spacing() {
        let mut res = base();
        res.ratio = Some(0.42);
        res.pinched = true;
        res.fingers = 2;
        assert_eq!(
            status_line(true, &res),
            "YouTube: SEEK ON    pinch: 0.42  <PINCHED>    fingers up: 2"
        );
        res.pinched = false;
        assert_eq!(
            status_line(false, &res),
            "YouTube: off    pinch: 0.42    fingers up: 2"
        );
    }

    #[test]
    fn status_line_no_hand() {
        let res = base();
        assert_eq!(
            status_line(false, &res),
            "YouTube: off    pinch: no hand    fingers up: 0"
        );
    }

    #[test]
    fn debug_line_format_and_truncation() {
        assert_eq!(
            debug_line(29.8, 5, 3, "Some Video Title - YouTube"),
            "fps: 29.8  scroll: +5  seeks: 3  window: Some Video Title - YouTube"
        );
        assert_eq!(
            debug_line(0.0, -5, 0, ""),
            "fps: 0.0  scroll: -5  seeks: 0  window: (none)"
        );
        // 45 chars -> truncated to 44 (chars, like Python's title[:44]).
        let long = "x".repeat(45);
        let line = debug_line(1.0, 0, 0, &long);
        assert!(line.ends_with(&"x".repeat(44)));
        assert!(!line.ends_with(&"x".repeat(45)));
    }

    #[test]
    fn short_title_format() {
        assert_eq!(
            short_title(true, Some(0.42), 5, 3),
            "[YT] pinch 0.42 | scroll +5 | seek x3"
        );
        assert_eq!(
            short_title(false, None, -5, 0),
            "[--] pinch no hand | scroll -5 | seek x0"
        );
    }

    // --- geometry ---------------------------------------------------------

    #[test]
    fn window_position_matches_v1() {
        assert_eq!(window_position(1920.0), (1420.0, 16.0));
        assert_eq!(window_position(2560.0), (2060.0, 16.0));
        // Narrow screen: clamps to x=8 like v1's max(8, sw-500).
        assert_eq!(window_position(480.0), (8.0, 16.0));
    }

    #[test]
    fn flash_anchor_left_fixed_right_measured() {
        assert_eq!(flash_anchor_x(Side::Left, 640.0, 120.0), 14.0);
        assert_eq!(flash_anchor_x(Side::Right, 640.0, 120.0), 506.0);
    }

    #[test]
    fn skeleton_points_truncate_like_python_int() {
        let lms = vec![
            Landmark::new(0.5, 0.25),   // (320.0, 120.0)
            Landmark::new(0.999, 0.1),  // (639.36 -> 639, 48.0)
            Landmark::new(0.333, 0.666), // (213.12 -> 213, 319.68 -> 319)
        ];
        let pts = skeleton_points(&lms, 640, 480);
        assert_eq!(pts, vec![(320.0, 120.0), (639.0, 48.0), (213.0, 319.0)]);
    }

    // --- M4.2: frame -> window scaling -----------------------------------

    #[test]
    fn scaler_at_defaults_is_uniform_0_75() {
        // 640x480 frame into v1's 480x360 window: both axes 0.75.
        let s = Scaler::new(640.0, 480.0, 480.0, 360.0);
        assert_eq!(s.sx, 0.75);
        assert_eq!(s.sy, 0.75);
        assert_eq!(s.pt((320.0, 240.0)), (240.0, 180.0)); // divider midpoint
        assert_eq!(s.pt((10.0, 26.0)), (7.5, 19.5)); // status line org
    }

    #[test]
    fn scaler_axes_scale_independently_on_odd_window() {
        // v1 cv2.resize stretches to the window; aspect is NOT preserved
        // for a non-4:3 window (parity: be equally stretchy).
        let s = Scaler::new(640.0, 480.0, 400.0, 400.0);
        assert_eq!(s.sx, 400.0 / 640.0);
        assert!((s.sy - 400.0 / 480.0).abs() < 1e-6);
        assert_eq!(s.pt((640.0, 0.0)), (400.0, 0.0));
    }

    #[test]
    fn flash_right_anchor_through_scaler_matches_window_geometry() {
        // Renderer measures text in WINDOW px, converts to frame px, runs it
        // through v1's flash_anchor_x, scales back: right flash must land
        // text_w from... right edge minus 14 frame-px margin.
        let s = Scaler::new(640.0, 480.0, 480.0, 360.0);
        let text_w_win = 100.0f32;
        let text_w_frame = text_w_win / s.sx;
        let anchor = s.pt((flash_anchor_x(Side::Right, 640.0, text_w_frame), 240.0));
        let expected_x = 480.0 - text_w_win - 14.0 * 0.75;
        assert!((anchor.0 - expected_x).abs() < 1e-3, "{} vs {}", anchor.0, expected_x);
        assert_eq!(s.pt((flash_anchor_x(Side::Left, 640.0, text_w_frame), 240.0)).0, 10.5);
    }

    #[test]
    fn hand_connections_match_v1_gesture_core_exactly() {
        // PinchFlick/gesture_core.py:32-39, verbatim (order matters: it is
        // the draw order).
        assert_eq!(
            HAND_CONNECTIONS,
            [
                (0, 1), (1, 2), (2, 3), (3, 4),          // thumb
                (0, 5), (5, 6), (6, 7), (7, 8),          // index
                (0, 9), (9, 10), (10, 11), (11, 12),     // middle
                (0, 13), (13, 14), (14, 15), (15, 16),   // ring
                (0, 17), (17, 18), (18, 19), (19, 20),   // pinky
                (5, 9), (9, 13), (13, 17),               // palm
            ]
        );
    }
}
