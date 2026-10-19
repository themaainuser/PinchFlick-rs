//! M3.5 — the product inference pipeline: two ort sessions plus the M3.2
//! detect path and the M3.3/M3.4 track path composed behind one
//! `process()` call (same code the `--video` gate exercises, factored out).
//!
//! Threshold wiring is pinned to what v1's C++ graph actually reads:
//! - presence gate = config `min_detection_confidence` compared as the
//!   graph does — `(double)float_value > double_threshold` on the RAW flag
//!   tensor (`ThresholdingCalculator` fed by `TensorsToFloatsCalculator`
//!   with activation NONE). The config key `min_presence_confidence` is
//!   dead in v1: no graph node in mediapipe 1.1.1 ever reads
//!   `min_hand_presence_confidence` — parity therefore means ignoring it
//!   here too.
//! - palm-detector score = the same config key, narrowed to f32 because
//!   upstream `set_min_score_thresh` takes a proto float.
//! - `min_tracking_confidence` = HandAssociation IoU threshold (f32).

use crate::vision::anchors::{generate_anchors, Anchor};
use crate::vision::{decode, image as vimage, landmark, rect, tracking, warp};
use ort::environment::Environment;
use ort::session::Session;
use ort::value::Tensor;
use std::time::{Duration, Instant};

/// One processed frame.
pub struct FrameOutcome {
    /// Projected 21 landmarks when the presence gate passed this frame.
    pub landmarks: Option<Vec<[f32; 3]>>,
    /// Raw flag tensor value (what the graph thresholds).
    pub presence_raw: f32,
    /// `"detect"` (palm detector ran) or `"track"` (prev next-frame rect).
    pub path: &'static str,
    /// Palm-detector stage time (ZERO when tracking — the M1 gate needs
    /// "amortized detector overhead" separately from landmark time).
    pub detect_ms: Duration,
    /// Crop + landmark model + projection + next-frame rect.
    pub landmark_ms: Duration,
}

pub struct Pipeline {
    det_session: Session,
    lm_session: Session,
    anchors: Vec<Anchor>,
    /// Loopback state (prev frame's gated next-frame rects).
    pub tracker: tracking::TrackerState,
    presence_threshold: f64,
    min_score_thresh: f32,
    min_tracking_confidence: f32,
}

/// Model-output lookup that errors instead of panicking: `[]` indexing on a
/// renamed/wrong model file used to abort the live loop with an index panic.
/// (Wiring-side hardening — ort output maps need a real session to build, so
/// this isn't unit-tested; happy paths are gate-run territory, like the rest
/// of the inference wiring.)
fn missing_output(name: &str) -> Box<dyn std::error::Error> {
    format!(
        "model output \"{name}\" not found - wrong model file in models\\? (scripts\\download-models.ps1)"
    )
    .into()
}

impl Pipeline {
    pub fn new(
        environment: &Environment,
        min_detection_confidence: f64,
        min_tracking_confidence: f64,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            det_session: Session::builder(environment)?
                .commit_from_file("models/palm_detector.onnx")?,
            lm_session: Session::builder(environment)?
                .commit_from_file("models/hand_landmark.onnx")?,
            anchors: generate_anchors(),
            tracker: tracking::TrackerState::default(),
            presence_threshold: min_detection_confidence,
            min_score_thresh: min_detection_confidence as f32,
            min_tracking_confidence: min_tracking_confidence as f32,
        })
    }

    /// Stream-mode (v1 runs VIDEO) frame in: detect or track per
    /// `tracking::plan_frame`, crop, landmark, presence-gate, update the
    /// loopback rect. `rgb` is a raw RGB8 buffer of `fw` x `fh`.
    pub fn process(
        &mut self,
        rgb: &[u8],
        fw: u32,
        fh: u32,
    ) -> Result<FrameOutcome, Box<dyn std::error::Error>> {
        let image_size = (fw as f32, fh as f32);
        let plan = tracking::plan_frame(
            &self.tracker.prev_rects,
            &[],
            1, // v1 config num_hands: 1
            self.min_tracking_confidence,
        );
        let path = if plan.run_detector { "detect" } else { "track" };

        let mut detect_ms = Duration::ZERO;
        let hand_rect = if plan.run_detector {
            let t0 = Instant::now();
            let rect = self.detect_rect(rgb, fw, fh, image_size)?;
            detect_ms = t0.elapsed();
            rect
        } else {
            self.tracker.prev_rects.first().copied()
        };

        let Some(r) = hand_rect else {
            // No detection this frame: gate emits nothing, loopback empties.
            self.tracker.update(vec![]);
            return Ok(FrameOutcome {
                landmarks: None,
                presence_raw: 0.0,
                path,
                detect_ms,
                landmark_ms: Duration::ZERO,
            });
        };

        let t1 = Instant::now();
        let mut roi = warp::get_roi(fw, fh, &r);
        let padding = warp::pad_roi(224, 224, false, &mut roi);
        let crop = warp::warp_crop(rgb, fw, fh, &roi, 224, (0.0, 1.0));
        let crop_tensor = Tensor::from_array(([1i64, 224, 224, 3], crop))?;
        let lm_outputs = self.lm_session.run(ort::inputs![crop_tensor])?;
        // Output names pinned at M3.3, same name-pinning as the detector
        // below: Identity = [1,63] 224-px landmarks, Identity_1 = [1,1]
        // presence flag (raw, activation NONE), Identity_2 = handedness
        // score, Identity_3 = [1,63] world landmarks.
        let landmark_raw = {
            let (_, data) = lm_outputs
                .get("Identity")
                .ok_or_else(|| missing_output("Identity"))?
                .try_extract_tensor::<f32>()?;
            data.to_vec()
        };
        let presence_raw = {
            let (_, data) = lm_outputs
                .get("Identity_1")
                .ok_or_else(|| missing_output("Identity_1"))?
                .try_extract_tensor::<f32>()?;
            data[0]
        };
        drop(lm_outputs);

        let passed = tracking::presence_gate(presence_raw, self.presence_threshold);
        let landmarks = if passed {
            let mut lms = landmark::decode_landmarks(&landmark_raw);
            landmark::letterbox_remove(&mut lms, padding);
            let projected = landmark::project_landmarks(&lms, &r);
            let next = tracking::next_frame_rect(&projected, image_size);
            self.tracker.update(vec![next]);
            Some(projected)
        } else {
            self.tracker.update(vec![]);
            None
        };
        let landmark_ms = t1.elapsed();

        Ok(FrameOutcome {
            landmarks,
            presence_raw,
            path,
            detect_ms,
            landmark_ms,
        })
    }

    /// Palm detector on the full frame -> best rect (M3.2 decode + M3.3
    /// detection-rect transform), through association + clip. Returns
    /// `None` when the detector found no hand (v1 graph: empty rect list).
    fn detect_rect(
        &mut self,
        rgb: &[u8],
        fw: u32,
        fh: u32,
        image_size: (f32, f32),
    ) -> Result<Option<rect::NormRect>, Box<dyn std::error::Error>> {
        let input = vimage::letterbox_normalize(rgb, fw, fh, 192);
        let tensor = Tensor::from_array(([1i64, 192, 192, 3], input))?;
        let outputs = self.det_session.run(ort::inputs![tensor])?;
        let (boxes, scores) = {
            let b = outputs
                .get("Identity")
                .ok_or_else(|| missing_output("Identity"))?
                .try_extract_tensor::<f32>()?;
            let s = outputs
                .get("Identity_1")
                .ok_or_else(|| missing_output("Identity_1"))?
                .try_extract_tensor::<f32>()?;
            (b.1.to_vec(), s.1.to_vec())
        };
        drop(outputs);
        let mut dets = decode::decode_detections(&boxes, &scores, &self.anchors, self.min_score_thresh);
        dets = decode::weighted_nms(&dets, 0.3);
        decode::project_detections_to_image(&mut dets, fw, fh, 192);
        let det_rects: Vec<rect::NormRect> = dets
            .iter()
            .map(|d| {
                let rotation = rect::detection_rotation(d.keypoints[0], d.keypoints[2], image_size);
                let (x_center, y_center, width, height) = rect::norm_rect_from_keypoints(&d.keypoints);
                rect::transform_rect(
                    &rect::NormRect {
                        x_center,
                        y_center,
                        width,
                        height,
                        rotation,
                    },
                    image_size,
                    rect::DETECT_SCALE,
                    rect::DETECT_SCALE,
                    rect::DETECT_SHIFT_X,
                    rect::DETECT_SHIFT_Y,
                )
            })
            .collect();
        let rects = tracking::associate_rects(
            &self.tracker.prev_rects,
            &det_rects,
            self.min_tracking_confidence,
        );
        Ok(tracking::clip_rects(rects, 1).into_iter().next())
    }
}
