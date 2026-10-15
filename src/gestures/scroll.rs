//! v1's ScrollDriver — per-frame finger counts into wheel clicks.

/// A gesture must hold steady for `confirm_frames` consecutive frames before
/// it fires (kills single-frame flicker), and the gate (`allowed = false`,
/// set by pinch/seek suppression in GestureEngine) drops any pending frames
/// so a pinch can never leak a phantom scroll.
pub struct ScrollDriver {
    speed: i32,
    confirm: usize,
    interval_ms: i64,
    invert: bool,
    up_fingers: i32,
    down_fingers: i32,
    recent: Vec<i32>,
    last_fire: Option<i64>,
}

impl ScrollDriver {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        speed: i32,
        confirm: usize,
        interval_ms: i64,
        invert: bool,
        up_fingers: i32,
        down_fingers: i32,
    ) -> Self {
        Self {
            speed,
            confirm,
            interval_ms,
            invert,
            up_fingers,
            down_fingers,
            recent: Vec::new(),
            last_fire: None,
        }
    }

    /// Drop any half-confirmed frames (gate closed).
    pub fn reset(&mut self) {
        self.recent.clear();
    }

    /// Returns wheel clicks to send this frame (0 = send nothing).
    pub fn update(&mut self, fingers: i32, now_ms: i64, allowed: bool) -> i32 {
        if !allowed {
            self.recent.clear();
            return 0;
        }
        self.recent.push(fingers);
        if self.recent.len() > self.confirm {
            self.recent.remove(0);
        }
        // needs `confirm` consecutive identical frames
        if self.recent.len() < self.confirm || self.recent.iter().any(|&f| f != self.recent[0]) {
            return 0;
        }
        let target = self.recent[self.recent.len() - 1];
        if target != self.up_fingers && target != self.down_fingers {
            return 0;
        }
        if self.interval_ms > 0
            && let Some(last) = self.last_fire
            && now_ms - last < self.interval_ms
        {
            return 0;
        }
        self.last_fire = Some(now_ms);
        let clicks = if target == self.up_fingers {
            self.speed
        } else {
            -self.speed
        };
        if self.invert {
            -clicks
        } else {
            clicks
        }
    }
}
