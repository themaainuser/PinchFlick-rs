//! v1's seek pieces — DoubleTapSeeker, side classification, key composition.

use crate::config::SeekKeys;

/// Which half of the view the user SEES a tap landed in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Left,
    Right,
}

impl Side {
    pub fn as_str(&self) -> &'static str {
        match self {
            Side::Left => "left",
            Side::Right => "right",
        }
    }
}

/// Seek direction (backward = left side, forward = right side).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeekDirection {
    Backward,
    Forward,
}

impl SeekDirection {
    pub fn as_str(&self) -> &'static str {
        match self {
            SeekDirection::Backward => "backward",
            SeekDirection::Forward => "forward",
        }
    }
}

/// Two pinches on the same side of the frame inside the window = a seek.
pub struct DoubleTapSeeker {
    pub window_ms: i64,
    pub cooldown_ms: i64,
    pub last_tap_time: Option<i64>,
    pub last_tap_side: Option<Side>,
    pub last_seek_time: Option<i64>,
    pub status: String,
}

impl DoubleTapSeeker {
    pub fn new(window_ms: i64, cooldown_ms: i64) -> Self {
        Self {
            window_ms,
            cooldown_ms,
            last_tap_time: None,
            last_tap_side: None,
            last_seek_time: None,
            status: "Ready -- pinch twice quickly on one side to seek".into(),
        }
    }

    /// Feed a pinch event; returns the seek direction on a completed pair.
    pub fn register(&mut self, side: Side, now_ms: i64) -> Option<SeekDirection> {
        let paired = self.last_tap_side == Some(side)
            && self.last_tap_time.is_some_and(|t| now_ms - t <= self.window_ms);
        if paired {
            self.last_tap_time = None;
            self.last_tap_side = None;
            if self
                .last_seek_time
                .is_none_or(|t| now_ms - t > self.cooldown_ms)
            {
                self.last_seek_time = Some(now_ms);
                let direction = match side {
                    Side::Left => SeekDirection::Backward,
                    Side::Right => SeekDirection::Forward,
                };
                let sign = if side == Side::Left { '-' } else { '+' };
                self.status = format!("Double-tap {} -> {sign}seek", side.as_str());
                return Some(direction);
            }
            self.status = "Too soon after the last seek -- wait a moment".into();
            return None;
        }

        self.last_tap_time = Some(now_ms);
        self.last_tap_side = Some(side);
        self.status = format!("Tap ({}) -- tap again quickly to confirm", side.as_str());
        None
    }
}

/// List of key presses for e.g. 15s forward -> `["l", "right"]`.
/// Err = v1's ValueError (bad direction / seek_seconds not a multiple of 5).
pub fn seek_key_sequence(
    seek_seconds: i64,
    direction: &str,
    keys: &SeekKeys,
) -> Result<Vec<String>, String> {
    if direction != "forward" && direction != "backward" {
        return Err(format!(
            "direction must be 'forward' or 'backward', got {direction:?}"
        ));
    }
    if seek_seconds < 5 || seek_seconds % 5 != 0 {
        return Err(format!(
            "seek_seconds must be a multiple of 5 (>=5), got {seek_seconds}"
        ));
    }
    let tens = seek_seconds / 10;
    let remainder = seek_seconds % 10;
    let key_10 = if direction == "forward" {
        &keys.forward_10
    } else {
        &keys.backward_10
    };
    let key_5 = if direction == "forward" {
        &keys.forward_5
    } else {
        &keys.backward_5
    };
    let mut seq = vec![key_10.clone(); tens as usize];
    if remainder != 0 {
        seq.push(key_5.clone());
    }
    Ok(seq)
}
