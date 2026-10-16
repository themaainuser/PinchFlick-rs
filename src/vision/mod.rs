//! M3: the MediaPipe hand-pipeline port — anchors, detection decode, input prep.
//!
//! Pure math over model tensors (no inference, no OS). Values and order of
//! operations follow mediapipe v1.1.1 sources in `reference/mediapipe/v1.1.1/`
//! (the version v1's pip wheel runs) so v1 behavior is preserved bit-for-bit
//! where float math allows.

pub mod anchors;
pub mod decode;
pub mod image;
pub mod landmark;
pub mod rect;
pub mod warp;
