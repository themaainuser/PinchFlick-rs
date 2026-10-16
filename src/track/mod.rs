//! M3.5 — hardware/OS wiring: webcam capture (nokhwa/MSMF) and the
//! composed ort inference pipeline. Per the M3 ruling these two modules are
//! declared spike exceptions (capture/inference wiring); every mathematical
//! step they invoke is TDD'd in `vision/` (decode 22, crop/projection 46,
//! tracking 18 tests) and both gate tools (`main.rs --landmarks`,
//! `--video`) exercise the same calls.

pub mod capture;
pub mod inference;
