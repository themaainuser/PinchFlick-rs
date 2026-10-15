//! v1's landmark math + GestureEngine — one frame in, decisions out.
//!
//! Suppression rules carried over from v1:
//! - while pinched (or hand not clearly open) -> no scroll
//! - for `scroll_suppress_after_seek_ms` after a seek -> no scroll
//! - seek fires only if enabled and (optionally) a YouTube tab is focused

use crate::config::Config;
use crate::gestures::pinch::PinchState;
use crate::gestures::scroll::ScrollDriver;
use crate::gestures::seek::{DoubleTapSeeker, SeekDirection, Side};

// --- MediaPipe hand landmark indices (standard 21-point skeleton) ---
pub const WRIST: usize = 0;
pub const THUMB_TIP: usize = 4;
pub const INDEX_TIP: usize = 8;
pub const MIDDLE_MCP: usize = 9;
/// (tip, pip) pairs for index, middle, ring, pinky — tip above pip = extended.
pub const FINGER_LANDMARKS: [(usize, usize); 4] = [(8, 6), (12, 10), (16, 14), (20, 18)];

pub const HAND_CONNECTIONS: [(usize, usize); 23] = [
    (0, 1), (1, 2), (2, 3), (3, 4),          // thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          // index
    (0, 9), (9, 10), (10, 11), (11, 12),     // middle
    (0, 13), (13, 14), (14, 15), (15, 16),   // ring
    (0, 17), (17, 18), (18, 19), (19, 20),   // pinky
    (5, 9), (9, 13), (13, 17),               // palm
];

/// One normalized landmark (v1's Point / MediaPipe's NormalizedLandmark).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Landmark {
    pub x: f64,
    pub y: f64,
}

impl Landmark {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

pub fn hand_size(landmarks: &[Landmark]) -> f64 {
    let w = landmarks[WRIST];
    let m = landmarks[MIDDLE_MCP];
    ((w.x - m.x).hypot(w.y - m.y)).max(1e-6)
}

/// Thumb-tip to index-tip distance, normalized by hand size.
pub fn pinch_ratio(landmarks: &[Landmark]) -> f64 {
    let t = landmarks[THUMB_TIP];
    let i = landmarks[INDEX_TIP];
    (t.x - i.x).hypot(t.y - i.y) / hand_size(landmarks)
}

/// How many of index/middle/ring/pinky point up (tip above pip).
pub fn count_extended_fingers(landmarks: &[Landmark]) -> i32 {
    FINGER_LANDMARKS
        .iter()
        .filter(|(tip, pip)| landmarks[*tip].y < landmarks[*pip].y)
        .count() as i32
}

/// Horizontal position of the pinch in VIEW space (0=left, 1=right).
/// `input_mirrored=true`: landmarks already in mirror/display space, use x as
/// is. `false`: raw camera space — flip x so left/right match what the user sees.
pub fn tap_x(landmarks: &[Landmark], input_mirrored: bool) -> f64 {
    let raw_x = (landmarks[THUMB_TIP].x + landmarks[INDEX_TIP].x) / 2.0;
    if input_mirrored {
        raw_x
    } else {
        1.0 - raw_x
    }
}

/// Which half of the view the user SEES (split at x=0.5).
pub fn tap_side(landmarks: &[Landmark], input_mirrored: bool) -> Side {
    if tap_x(landmarks, input_mirrored) < 0.5 {
        Side::Left
    } else {
        Side::Right
    }
}

/// v1's EngineResult — everything the engine decided this frame.
#[derive(Debug, Clone)]
pub struct EngineResult {
    pub ratio: Option<f64>,
    pub fingers: i32,
    pub pinched: bool,
    pub scroll_clicks: i32,
    pub seek: Option<SeekDirection>,
    pub seek_blocked: Option<String>,
    pub tap_status: String,
    pub side: Option<Side>,
    /// Side of a freshly registered tap this frame, else None.
    pub tap: Option<Side>,
}

pub struct GestureEngine {
    pub cfg: Config,
    pub input_mirrored: bool,
    pub pinch: PinchState,
    pub seeker: DoubleTapSeeker,
    pub scroller: ScrollDriver,
    pub last_seek_ms: Option<i64>,
    pub last_side: Option<Side>,
}

impl GestureEngine {
    pub fn new(cfg: &Config, input_mirrored: bool) -> Self {
        Self {
            cfg: cfg.clone(),
            input_mirrored,
            pinch: PinchState::new(cfg.pinch_on_ratio, cfg.pinch_off_ratio),
            seeker: DoubleTapSeeker::new(cfg.double_tap_window_ms, cfg.seek_cooldown_ms),
            scroller: ScrollDriver::new(
                cfg.scroll_speed as i32,
                cfg.scroll_confirm_frames as usize,
                cfg.scroll_interval_ms,
                cfg.scroll_invert,
                cfg.scroll_up_fingers as i32,
                cfg.scroll_down_fingers as i32,
            ),
            last_seek_ms: None,
            last_side: None,
        }
    }

    /// One frame in: `landmarks = None` (or empty) means no hand in view.
    pub fn frame(
        &mut self,
        landmarks: Option<&[Landmark]>,
        now_ms: i64,
        on_youtube: bool,
    ) -> EngineResult {
        // v1 reads these off the cfg dict each frame; copy the scalars so the
        // rest of the method can mutate self freely
        let dead = self.cfg.tap_side_dead_zone;
        let seek_enabled = self.cfg.seek_enabled;
        let requires_youtube = self.cfg.seek_requires_youtube;
        let scroll_enabled = self.cfg.scroll_enabled;
        let open_hand_ratio = self.cfg.scroll_open_hand_ratio;
        let suppress_after_seek_ms = self.cfg.scroll_suppress_after_seek_ms;

        let landmarks = landmarks.filter(|l| !l.is_empty());

        let Some(lm) = landmarks else {
            // no hand in view
            self.pinch.release();
            self.scroller.reset();
            return EngineResult {
                ratio: None,
                fingers: 0,
                pinched: false,
                scroll_clicks: 0,
                seek: None,
                seek_blocked: None,
                tap_status: self.seeker.status.clone(),
                side: self.last_side,
                tap: None,
            };
        };

        let ratio = pinch_ratio(lm);
        let fresh_pinch = self.pinch.update(Some(ratio));

        let mut seek = None;
        let mut seek_blocked = None;
        let mut tap = None;
        if fresh_pinch {
            let pos_x = tap_x(lm, self.input_mirrored);
            if (pos_x - 0.5).abs() < dead {
                // ambiguous middle: better to refuse than to assign a random
                // side and silently break the double-tap pairing
                self.seeker.status =
                    "Too close to the middle -- pinch clearly left or right of the line".into();
            } else {
                let side = if pos_x < 0.5 { Side::Left } else { Side::Right };
                self.last_side = Some(side);
                tap = Some(side);
                if let Some(direction) = self.seeker.register(side, now_ms) {
                    if !seek_enabled {
                        seek_blocked = Some("seek is disabled in config".into());
                    } else if requires_youtube && !on_youtube {
                        seek_blocked = Some("not on YouTube".into());
                    } else {
                        seek = Some(direction);
                        self.last_seek_ms = Some(now_ms);
                    }
                }
            }
        }

        let fingers = count_extended_fingers(lm);

        let mut scroll_ok =
            scroll_enabled && !self.pinch.active && ratio >= open_hand_ratio;
        if scroll_ok
            && let Some(last) = self.last_seek_ms
            && now_ms - last < suppress_after_seek_ms
        {
            scroll_ok = false;
        }
        let clicks = self.scroller.update(fingers, now_ms, scroll_ok);

        EngineResult {
            ratio: Some(ratio),
            fingers,
            pinched: self.pinch.active,
            scroll_clicks: clicks,
            seek,
            seek_blocked,
            tap_status: self.seeker.status.clone(),
            side: self.last_side,
            tap,
        }
    }
}
