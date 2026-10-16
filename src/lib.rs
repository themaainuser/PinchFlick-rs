//! pinchflick-rs — Rust rewrite of PinchFlick (v1 gesture semantics win).
//!
//! M2: the gesture engine port. Pure decision logic (landmarks + timestamps
//! in, decisions out), no camera / no inference / no OS input — that is what
//! lets the 25 v1 selftests run as `cargo test`.

pub mod config;
pub mod gestures;
pub mod sys;
pub mod vision;
