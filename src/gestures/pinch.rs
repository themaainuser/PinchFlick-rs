//! v1's PinchState — hysteresis latch.

/// Enters below `on_ratio`, only leaves past `off_ratio`. Frame-to-frame
/// jitter inside the gap can't re-fire or drop the state, so one physical
/// pinch produces exactly one rising edge.
pub struct PinchState {
    pub on_ratio: f64,
    pub off_ratio: f64,
    pub active: bool,
}

impl PinchState {
    pub fn new(on_ratio: f64, off_ratio: f64) -> Self {
        Self {
            on_ratio,
            off_ratio,
            active: false,
        }
    }

    /// Feed this frame's ratio; returns `true` on a fresh pinch (rising edge).
    /// `None` = hand lost.
    pub fn update(&mut self, ratio: Option<f64>) -> bool {
        let Some(r) = ratio else {
            // hand lost
            self.active = false;
            return false;
        };
        if !self.active && r < self.on_ratio {
            self.active = true;
            return true;
        }
        if self.active && r > self.off_ratio {
            self.active = false;
        }
        false
    }

    /// Hand lost: drop the latch.
    pub fn release(&mut self) {
        self.active = false;
    }
}
