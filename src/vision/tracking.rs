//! M3.4: presence gating + VIDEO-mode tracking — ports of MediaPipe
//! v1.1.1's `ThresholdingCalculator` (presence gate),
//! `HandLandmarksToRectCalculator` (next-frame rect from landmarks),
//! `HandAssociationCalculator` + `rectangle_util`, the
//! `HandLandmarkerGraph` stream-mode state machine (PreviousLoopback +
//! min-size + DisallowIf + clip), and
//! `HandLandmarksDeduplicationCalculator`.
//!
//! v1 runs RunningMode.VIDEO (controller.py line 123), so the stream branch
//! is the parity path: palm detection runs only when last frame's
//! gated next-frame rects number fewer than `num_hands`.

use crate::vision::rect::{normalize_radians, NormRect};

/// `HandLandmarksToRectCalculator::GetPartialLandmarks` — when given all 21
/// landmarks it first extracts this 12-landmark subset (tips and most DIPs
/// are NOT part of the next-frame rect).
pub const PARTIAL_LANDMARK_INDICES: [usize; 12] = [0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18];

/// Reference-point constants (indices into the PARTIAL list, upstream names
/// are legacy): wrist 0; reference = 0.25*full[5] + 0.25*full[13] +
/// 0.5*full[9] (index/ring/middle MCP) — built as (((a + b) / 2) + c) / 2.
const PARTIAL_WRIST: usize = 0;
const PARTIAL_INDEX_MCP: usize = 4; // -> full 5
const PARTIAL_MIDDLE_MCP: usize = 6; // -> full 9
const PARTIAL_RING_MCP: usize = 8; // -> full 13

/// `HandLandmarkerGraph`'s `min_tracking_confidence` flows only into
/// `HandAssociationCalculator.min_similarity_threshold` (IoU, strict >) —
/// and for num_hands=1 the detector is skipped whenever a prev rect exists,
/// so this threshold is never decisive on the v1 config. Still ported.
pub const DEFAULT_MIN_TRACKING_CONFIDENCE: f32 = 0.6;

/// `ThresholdingCalculator::Process` verbatim: FLAG =
/// `(double)float_value > (double)threshold` — a widening f32->f64 compare
/// against the proto's f64 threshold (v1 sets 0.6 as double). Raw flag
/// tensor value, because `TensorsToFloatsCalculator` runs with its default
/// activation NONE. The threshold is `min_detection_confidence` (0.6 in v1)
/// — the graphs never read `min_hand_presence_confidence`.
pub fn presence_gate(presence_raw: f32, min_detection_confidence: f64) -> bool {
    f64::from(presence_raw) > min_detection_confidence
}

/// `HandLandmarksToRectCalculator::ComputeRotation` verbatim on the partial
/// subset: reference point averaged in NORMALIZED space first
/// (((index+ring)/2 + middle)/2), scaled to pixels after; target is
/// kTargetAngle = pi/2 — the tracking path does NOT carry the detector's
/// 90-radian quirk.
fn landmarks_rotation(partial: &[[f32; 3]], image_size: (f32, f32)) -> f32 {
    let (iw, ih) = image_size;
    let (x0, y0) = (
        partial[PARTIAL_WRIST][0] * iw,
        partial[PARTIAL_WRIST][1] * ih,
    );
    let mut x1 = (partial[PARTIAL_INDEX_MCP][0] + partial[PARTIAL_RING_MCP][0]) / 2.0;
    let mut y1 = (partial[PARTIAL_INDEX_MCP][1] + partial[PARTIAL_RING_MCP][1]) / 2.0;
    x1 = (x1 + partial[PARTIAL_MIDDLE_MCP][0]) / 2.0 * iw;
    y1 = (y1 + partial[PARTIAL_MIDDLE_MCP][1]) / 2.0 * ih;
    normalize_radians(std::f32::consts::FRAC_PI_2 - (-(y1 - y0)).atan2(x1 - x0))
}

/// `HandLandmarksToRectCalculator::NormalizedLandmarkListToRect` verbatim
/// (bbox over the 12 partial landmarks, rotated-extent sizing). Replicates
/// upstream's `numeric_limits<float>::min()` initialization for maxima
/// (f32::MIN — smallest *positive* float, not lowest()).
pub fn landmarks_to_rect(landmarks: &[[f32; 3]], image_size: (f32, f32)) -> NormRect {
    let partial: Vec<[f32; 3]> = if landmarks.len() == 21 {
        PARTIAL_LANDMARK_INDICES
            .iter()
            .map(|&i| landmarks[i])
            .collect()
    } else {
        landmarks.to_vec()
    };
    let (iw, ih) = image_size;

    let rotation = landmarks_rotation(&partial, image_size);
    let reverse_angle = normalize_radians(-rotation);

    // Axis-aligned center over the partial subset (normalized coords).
    // Verbatim init: maxima = f32::MIN, minima = f32::MAX.
    let mut max_x = f32::MIN;
    let mut max_y = f32::MIN;
    let mut min_x = f32::MAX;
    let mut min_y = f32::MAX;
    for lm in &partial {
        max_x = max_x.max(lm[0]);
        max_y = max_y.max(lm[1]);
        min_x = min_x.min(lm[0]);
        min_y = min_y.min(lm[1]);
    }
    let axis_aligned_center_x = (max_x + min_x) / 2.0;
    let axis_aligned_center_y = (max_y + min_y) / 2.0;

    // Rotated extents (pixel units).
    let (rs, rc) = (reverse_angle.sin(), reverse_angle.cos());
    max_x = f32::MIN;
    max_y = f32::MIN;
    min_x = f32::MAX;
    min_y = f32::MAX;
    for lm in &partial {
        let original_x = (lm[0] - axis_aligned_center_x) * iw;
        let original_y = (lm[1] - axis_aligned_center_y) * ih;
        let projected_x = original_x * rc - original_y * rs;
        let projected_y = original_x * rs + original_y * rc;
        max_x = max_x.max(projected_x);
        max_y = max_y.max(projected_y);
        min_x = min_x.min(projected_x);
        min_y = min_y.min(projected_y);
    }
    let projected_center_x = (max_x + min_x) / 2.0;
    let projected_center_y = (max_y + min_y) / 2.0;

    let (s, c) = (rotation.sin(), rotation.cos());
    let center_x = projected_center_x * c - projected_center_y * s + iw * axis_aligned_center_x;
    let center_y = projected_center_x * s + projected_center_y * c + ih * axis_aligned_center_y;

    NormRect {
        x_center: center_x / iw,
        y_center: center_y / ih,
        width: (max_x - min_x) / iw,
        height: (max_y - min_y) / ih,
        rotation,
    }
}

/// `HandLandmarksToRectCalculator` + `ConfigureHandRectTransformationCalculator`
/// (scale 2.0, shift (0, -0.1), square_long) — the rect handed to the next
/// frame's landmark crop.
pub fn next_frame_rect(landmarks: &[[f32; 3]], image_size: (f32, f32)) -> NormRect {
    let rect = landmarks_to_rect(landmarks, image_size);
    crate::vision::rect::transform_rect(
        &rect,
        image_size,
        crate::vision::rect::NEXT_FRAME_SCALE,
        crate::vision::rect::NEXT_FRAME_SCALE,
        crate::vision::rect::NEXT_FRAME_SHIFT_X,
        crate::vision::rect::NEXT_FRAME_SHIFT_Y,
    )
}

/// Axis-aligned IoU (`rectangle_util::CalculateIou` — ToRectangle ignores
/// rotation, upstream TODO) — f32 math, standard intersection/union.
fn axis_iou(a: &NormRect, b: &NormRect) -> f32 {
    let (a_l, a_t) = (a.x_center - a.width / 2.0, a.y_center - a.height / 2.0);
    let (b_l, b_t) = (b.x_center - b.width / 2.0, b.y_center - b.height / 2.0);
    let inter_w = (a_l + a.width).min(b_l + b.width) - a_l.max(b_l);
    let inter_h = (a_t + a.height).min(b_t + b.height) - a_t.max(b_t);
    if inter_w <= 0.0 || inter_h <= 0.0 {
        return 0.0;
    }
    let inter = inter_w * inter_h;
    let union = a.width * a.height + b.width * b.height - inter;
    if union > 0.0 {
        inter / union
    } else {
        0.0
    }
}

/// `rectangle_util::DoesRectOverlap` verbatim: true if the new rect's IoU
/// with ANY existing rect is STRICTLY greater than the threshold
/// (rotation ignored).
pub fn does_rect_overlap(new_rect: &NormRect, existing: &[NormRect], min_similarity: f32) -> bool {
    existing
        .iter()
        .any(|e| axis_iou(new_rect, e) > min_similarity)
}

/// `HandAssociationCalculator::GetNonOverlappingElements` verbatim:
/// BASE_RECTS (tracked rects) come first unconditionally, then each
/// detector rect is appended only if it does not overlap the result built
/// so far at IoU > min_similarity_threshold. (Upstream also assigns rect
/// ids here; no downstream consumer reads them on this graph, so they are
/// omitted.)
pub fn associate_rects(base: &[NormRect], detected: &[NormRect], min_similarity: f32) -> Vec<NormRect> {
    let mut result: Vec<NormRect> = base.to_vec();
    for det in detected {
        if !does_rect_overlap(det, &result, min_similarity) {
            result.push(*det);
        }
    }
    result
}

/// `ClipNormalizedRectVectorSizeCalculator` — truncate to `num_hands`.
pub fn clip_rects(mut rects: Vec<NormRect>, num_hands: usize) -> Vec<NormRect> {
    rects.truncate(num_hands);
    rects
}

/// One frame's plan under `HandLandmarkerGraph`'s stream mode
/// (`use_stream_mode` = VIDEO): `has_enough_hands` = prev gated rects
/// >= num_hands; when true the palm detector input is DisallowIf-blocked
/// (detections ignored), otherwise the detector runs. Either way: associate
/// (base first) then clip.
pub struct FramePlan {
    pub run_detector: bool,
    pub hand_rects: Vec<NormRect>,
}

pub fn plan_frame(
    prev_rects: &[NormRect],
    detections: &[NormRect],
    num_hands: usize,
    min_similarity: f32,
) -> FramePlan {
    let run_detector = prev_rects.len() < num_hands;
    let dets: &[NormRect] = if run_detector { detections } else { &[] };
    FramePlan {
        run_detector,
        hand_rects: clip_rects(associate_rects(prev_rects, dets, min_similarity), num_hands),
    }
}

/// Loopback state: the previous frame's *gated* next-frame rects.
/// Presence failure => the detector subgraph's AllowIf drops the
/// hand_rect_next_frame => empty vector => next frame re-detects.
#[derive(Default, Debug, Clone)]
pub struct TrackerState {
    pub prev_rects: Vec<NormRect>,
}

impl TrackerState {
    /// Write back this frame's gated next-frame rects.
    pub fn update(&mut self, gated_next_rects: Vec<NormRect>) {
        self.prev_rects = gated_next_rects;
    }
}

/// `HandDuplicatesFinder::FindDuplicates` (start_from_the_end=false)
/// verbatim: suppress hand j (kept in iteration order) when >= 10 of 21
/// corresponding landmarks are within `0.2 * max(baseline_i, baseline_j)`
/// px AND the axis-aligned IoU of the landmark bboxes is > 0.2.
/// For v1 (num_hands=1, clip before the landmark stage) the input can
/// never exceed one hand, so this is a structural no-op — ported anyway.
pub fn find_duplicate_hand_indices(
    multi_landmarks: &[Vec<[f32; 3]>],
    image_size: (f32, f32),
) -> Vec<usize> {
    let (iw, ih) = image_size;
    let dist = |a: &[f32; 3], b: &[f32; 3]| -> f32 {
        let (dx, dy) = ((a[0] - b[0]) * iw, (a[1] - b[1]) * ih);
        (dx * dx + dy * dy).sqrt()
    };
    let baseline = |lms: &[[f32; 3]]| -> f32 {
        // max of wrist<->index_mcp, index_mcp<->pinky_mcp, pinky_mcp<->wrist.
        let d1 = dist(&lms[0], &lms[5]);
        let d2 = dist(&lms[5], &lms[17]);
        let d3 = dist(&lms[17], &lms[0]);
        d1.max(d2).max(d3)
    };

    let baselines: Vec<f32> = multi_landmarks.iter().map(|l| baseline(l)).collect();
    let bounds: Vec<(f32, f32, f32, f32)> = multi_landmarks
        .iter()
        .map(|lms| {
            let (mut l, mut t, mut r, mut b) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
            for lm in lms {
                l = l.min(lm[0]);
                t = t.min(lm[1]);
                r = r.max(lm[0]);
                b = b.max(lm[1]);
            }
            (l, t, r, b)
        })
        .collect();

    let mut retained: Vec<usize> = Vec::new();
    let mut suppressed: Vec<usize> = Vec::new();
    for (i, lms_i) in multi_landmarks.iter().enumerate() {
        let mut is_suppressed = false;
        for &j in &retained {
            let lms_j = &multi_landmarks[j];
            let threshold = baselines[i].max(baselines[j]) * 0.2;
            let matched = (0..21)
                .filter(|&k| dist(&lms_i[k], &lms_j[k]) < threshold)
                .count();
            // Axis IoU of bboxes (normalized).
            let (li, ti, ri, bi) = bounds[i];
            let (lj, tj, rj, bj) = bounds[j];
            let (iw2, ih2) = (
                (ri.min(rj) - li.max(lj)).max(0.0),
                (bi.min(bj) - ti.max(tj)).max(0.0),
            );
            let inter = iw2 * ih2;
            let union = (ri - li) * (bi - ti) + (rj - lj) * (bj - tj) - inter;
            let iou = if union > 0.0 { inter / union } else { 0.0 };
            if matched >= 10 && iou > 0.2 {
                is_suppressed = true;
                break;
            }
        }
        if is_suppressed {
            suppressed.push(i);
        } else {
            retained.push(i);
        }
    }
    suppressed
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    fn rect(x: f32, y: f32, w: f32, h: f32) -> NormRect {
        NormRect {
            x_center: x,
            y_center: y,
            width: w,
            height: h,
            rotation: 0.0,
        }
    }

    // ---- presence gate (ThresholdingCalculator verbatim) ----

    #[test]
    fn presence_gate_is_strictly_greater_with_f64_widening() {
        // f32 0.6 widened to f64 is 0.6000000238... > f64 0.6 -> PASS.
        assert!(presence_gate(0.6f32, 0.6));
        // Just below threshold (f32 value below f64 0.6) -> fail.
        assert!(!presence_gate(0.5999f32, 0.6));
        // Raw logit semantics: no sigmoid is applied by the graph
        // (TensorsToFloats activation default NONE) -> 0.4 fails even
        // though sigmoid(0.4) ~ 0.598 would also fail; a raw 0.5 fails.
        assert!(!presence_gate(0.5f32, 0.6));
        assert!(presence_gate(0.9391f32, 0.6));
    }

    #[test]
    fn presence_gate_threshold_comes_from_min_detection_confidence() {
        // Verbatim graph wiring: threshold = subgraph_options
        // .min_detection_confidence(), NOT min_hand_presence_confidence
        // (never consumed by any graph node in v1.1.1).
        assert!(presence_gate(0.6000001f32, 0.6));
        assert!(!presence_gate(0.35f32, 0.6));
    }

    // ---- HandLandmarksToRect verbatim ----

    /// A synthetic upright hand: wrist bottom-center, MCPs above it.
    fn upright_hand() -> Vec<[f32; 3]> {
        let mut lms = [[0.0f32; 3]; 21];
        // wrist (0)
        lms[0] = [0.5, 0.8, 0.0];
        // thumb 1..4
        lms[1] = [0.42, 0.72, 0.0];
        lms[2] = [0.39, 0.66, 0.0];
        lms[3] = [0.37, 0.61, 0.0];
        lms[4] = [0.35, 0.57, 0.0];
        // index 5..8
        lms[5] = [0.44, 0.62, 0.0];
        lms[6] = [0.43, 0.54, 0.0];
        lms[7] = [0.43, 0.49, 0.0];
        lms[8] = [0.43, 0.45, 0.0];
        // middle 9..12
        lms[9] = [0.49, 0.60, 0.0];
        lms[10] = [0.49, 0.51, 0.0];
        lms[11] = [0.49, 0.45, 0.0];
        lms[12] = [0.49, 0.41, 0.0];
        // ring 13..16
        lms[13] = [0.54, 0.62, 0.0];
        lms[14] = [0.55, 0.54, 0.0];
        lms[15] = [0.55, 0.49, 0.0];
        lms[16] = [0.55, 0.45, 0.0];
        // pinky 17..20
        lms[17] = [0.58, 0.65, 0.0];
        lms[18] = [0.59, 0.59, 0.0];
        lms[19] = [0.60, 0.55, 0.0];
        lms[20] = [0.60, 0.51, 0.0];
        lms.to_vec()
    }

    #[test]
    fn landmarks_rotation_upright_hand_is_zero_not_quirked() {
        // Tracking path target is pi/2 (kTargetAngle) -> upright hand
        // rotates 0.0, in contrast to the detection path's +0.4646 quirk.
        let lms = upright_hand();
        let partial: Vec<[f32; 3]> = PARTIAL_LANDMARK_INDICES
            .iter()
            .map(|&i| lms[i])
            .collect();
        let r = landmarks_rotation(&partial, (640.0, 480.0));
        // Reference point: ((index+ring)/2 + middle)/2 in normalized x,
        // all ~0.495; y avg over MCPs ~0.6133 vs wrist 0.8 -> atan2 of
        // upward vector = pi/2 -> rotation = 0 (small x-offset may remain).
        assert!(r.abs() < 0.2, "rotation {r} should be near 0, not 0.4646");
        assert!((r - 0.464_609_3).abs() > 0.3, "must NOT carry the detector quirk");
    }

    #[test]
    fn landmarks_to_rect_uses_partial_subset_ignoring_tips() {
        let mut lms = upright_hand();
        let base = landmarks_to_rect(&lms, (640.0, 480.0));
        // Wild tips (full 4, 8, 12, 16, 20) and wild DIPs (7, 11, 15, 19)
        // are outside the 12-subset -> rect must not change.
        for &i in &[4usize, 7, 8, 11, 12, 15, 16, 19, 20] {
            lms[i] = [0.99, 0.99, 0.0];
        }
        let with_wild_tips = landmarks_to_rect(&lms, (640.0, 480.0));
        assert!(
            (with_wild_tips.width - base.width).abs() < EPS
                && (with_wild_tips.height - base.height).abs() < EPS
                && (with_wild_tips.x_center - base.x_center).abs() < EPS,
            "tips/DIPs must not affect the rect: {base:?} vs {with_wild_tips:?}"
        );
        // A subset landmark (index PIP = full 6) DOES change it.
        let mut lms2 = upright_hand();
        lms2[6] = [0.99, 0.99, 0.0];
        let with_pip = landmarks_to_rect(&lms2, (640.0, 480.0));
        assert!(
            (with_pip.width - base.width).abs() > 1e-3
                || (with_pip.height - base.height).abs() > 1e-3
                || (with_pip.x_center - base.x_center).abs() > 1e-3,
            "subset landmark (full 6) must affect the rect"
        );
    }

    #[test]
    fn landmarks_to_rect_axis_aligned_hand_sizes_match_bbox() {
        // With rotation ~0 the rect should be the axis-aligned extent of
        // the 12 partial landmarks (normalized), then square_long is NOT
        // applied here (only the transform stage squares it).
        let lms = upright_hand();
        let r = landmarks_to_rect(&lms, (640.0, 480.0));
        // Not the axis-aligned x-span (0.59-0.37 = 0.22 normalized here,
        // incl. thumb IP at 0.37): the rect is sized in the ROTATED frame.
        // Tilt: reference point x1 = ((0.44+0.54)/2 + 0.49)/2 = 0.49 vs
        // wrist 0.5 -> rotation = -0.0708 rad (-4.05 deg), which mixes the
        // y-span into the width. Independent hand-calc of the verbatim
        // rotated extent (points 3 -> 18): 141.1 px / 640 = 0.2205.
        assert!(
            (r.width - 0.2205).abs() < 0.003,
            "width {} (expect 0.2205)",
            r.width
        );
        // Center: axis-aligned box is x 0.37..0.59 (center 0.48), y
        // 0.51..0.8 (center 0.655); the rotated-frame center maps back
        // with only a small offset.
        assert!((r.x_center - 0.485).abs() < 0.015, "x_center {}", r.x_center);
        assert!((r.y_center - 0.655).abs() < 0.02, "y_center {}", r.y_center);
        assert!((r.rotation + 0.0708).abs() < 0.01, "rotation {}", r.rotation);
    }

    #[test]
    fn next_frame_rect_applies_tracking_transform() {
        // Next-frame: square_long + scale 2.0 + shift_y -0.1 (versus the
        // detection path's 2.6/-0.5).
        let lms = upright_hand();
        let raw = landmarks_to_rect(&lms, (640.0, 480.0));
        let next = next_frame_rect(&lms, (640.0, 480.0));
        // Square in pixels.
        let w_px = next.width * 640.0;
        let h_px = next.height * 480.0;
        assert!((w_px - h_px).abs() < 1e-3, "square_long: {w_px} vs {h_px}");
        // Scale: raw (pre-square) long side * 2.0.
        let long_px = (raw.width * 640.0).max(raw.height * 480.0);
        assert!((w_px - long_px * 2.0).abs() < 1e-2, "w_px {w_px} vs {long_px}*2");
        // Shift: rotation ~= 0 branch (approx) -> y moves by h*shift_y.
        assert!(
            (next.y_center - (raw.y_center + raw.height * -0.1)).abs() < 0.02,
            "shift_y: {} vs {}",
            next.y_center,
            raw.y_center
        );
        assert!((next.rotation - raw.rotation).abs() < EPS, "rotation preserved");
    }

    // ---- association (IoU, strict >) ----

    #[test]
    fn overlap_uses_strict_greater_than() {
        let a = rect(0.25, 0.25, 0.5, 0.5);
        let b = rect(0.75, 0.75, 0.4, 0.4);
        let iou = {
            // Disjoint -> 0.
            assert_eq!(axis_iou(&a, &b), 0.0);
            // Identical -> 1.
            axis_iou(&a, &a)
        };
        assert_eq!(iou, 1.0);
        // Threshold exactly equal to the IoU -> NOT overlapping (strict).
        assert!(!does_rect_overlap(&a, &[b], 0.0)); // IoU 0 > 0 is false
        assert!(!does_rect_overlap(&a, &[a], 1.0)); // 1.0 > 1.0 false
        assert!(does_rect_overlap(&a, &[a], 0.999));
        // Boundary via computed iou: use a partially overlapping pair.
        let c = rect(0.5, 0.25, 0.5, 0.5);
        let i = axis_iou(&a, &c);
        assert!(i > 0.0 && i < 1.0, "partial IoU {i}");
        assert!(!does_rect_overlap(&c, &[a], i), "strict > at exact iou");
        assert!(does_rect_overlap(&c, &[a], i * 0.999));
    }

    #[test]
    fn associate_puts_base_first_and_drops_overlapping_detections() {
        let base = rect(0.5, 0.5, 0.3, 0.3);
        let det_overlap = rect(0.51, 0.5, 0.3, 0.3); // high IoU
        let det_far = rect(0.1, 0.1, 0.2, 0.2); // IoU 0
        let out = associate_rects(&[base], &[det_overlap, det_far], 0.6);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].x_center, 0.5, "base first");
        assert!((out[1].x_center - 0.1).abs() < EPS, "far detection appended");
    }

    #[test]
    fn associate_without_base_takes_all_detections_in_order() {
        let d1 = rect(0.2, 0.2, 0.3, 0.3);
        let d2 = rect(0.8, 0.8, 0.3, 0.3);
        let out = associate_rects(&[], &[d1, d2], 0.6);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].x_center, 0.2);
        assert_eq!(out[1].x_center, 0.8);
    }

    #[test]
    fn clip_limits_to_num_hands() {
        let out = clip_rects(vec![rect(0.2, 0.2, 0.1, 0.1), rect(0.8, 0.8, 0.1, 0.1)], 1);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].x_center, 0.2, "first kept");
    }

    // ---- stream-mode state machine ----

    #[test]
    fn first_frame_runs_detector_with_empty_base() {
        let plan = plan_frame(&[], &[], 1, 0.6);
        assert!(plan.run_detector, "no prev rect -> detect");
        assert!(plan.hand_rects.is_empty());
        // With detections present they flow through (empty base -> all).
        let det = rect(0.4, 0.4, 0.2, 0.2);
        let plan = plan_frame(&[], &[det], 1, 0.6);
        assert!(plan.run_detector);
        assert_eq!(plan.hand_rects.len(), 1);
    }

    #[test]
    fn tracked_frame_skips_detector_and_reuses_prev_rect() {
        let prev = rect(0.45, 0.45, 0.3, 0.3);
        let det = rect(0.46, 0.46, 0.3, 0.3);
        let plan = plan_frame(&[prev], &[det], 1, 0.6);
        assert!(!plan.run_detector, "has_enough_hands -> detector blocked");
        assert_eq!(plan.hand_rects.len(), 1);
        assert_eq!(plan.hand_rects[0].x_center, 0.45, "prev rect used, det ignored");
    }

    #[test]
    fn presence_failure_clears_state_so_next_frame_re_detects() {
        // Loopback carries only GATED next-frame rects: presence false ->
        // AllowIf drops the rect -> empty vector -> re-detect.
        let mut state = TrackerState::default();
        let prev = rect(0.45, 0.45, 0.3, 0.3);
        state.update(vec![prev]);
        // Frame 2: tracked.
        let plan = plan_frame(&state.prev_rects, &[], 1, 0.6);
        assert!(!plan.run_detector);
        // Presence fails this frame -> gate drops next rect -> update([]).
        assert!(!presence_gate(0.4, 0.6));
        state.update(vec![]);
        // Frame 3: detector runs again.
        let plan = plan_frame(&state.prev_rects, &[], 1, 0.6);
        assert!(plan.run_detector, "empty gated rects -> re-detect");
    }

    #[test]
    fn presence_success_feeds_next_rect_back_into_loop() {
        let mut state = TrackerState::default();
        let det = rect(0.45, 0.45, 0.3, 0.3);
        let plan = plan_frame(&state.prev_rects, &[det], 1, 0.6);
        assert!(plan.run_detector);
        // Landmark stage succeeds (presence raw > 0.6) -> next rect written.
        assert!(presence_gate(0.9391, 0.6));
        let lms = upright_hand();
        let next = next_frame_rect(&lms, (640.0, 480.0));
        state.update(vec![next]);
        let plan2 = plan_frame(&state.prev_rects, &[det], 1, 0.6);
        assert!(!plan2.run_detector, "gated rect present -> track");
        assert_eq!(plan2.hand_rects.len(), 1);
    }

    #[test]
    fn num_hands_two_detects_until_enough_rects_tracked() {
        // num_hands=2: one prev rect is NOT enough -> detector runs and
        // the association yields base + non-overlapping detection (clipped
        // to 2).
        let prev = rect(0.2, 0.2, 0.3, 0.3);
        let det = rect(0.7, 0.7, 0.3, 0.3);
        let plan = plan_frame(&[prev], &[det], 2, 0.6);
        assert!(plan.run_detector, "1 < 2 -> detect");
        assert_eq!(plan.hand_rects.len(), 2);
        assert_eq!(plan.hand_rects[0].x_center, 0.2, "base first");
        assert_eq!(plan.hand_rects[1].x_center, 0.7);
        // Two prev rects -> detector blocked.
        let plan = plan_frame(&[prev, det], &[rect(0.5, 0.5, 0.1, 0.1)], 2, 0.6);
        assert!(!plan.run_detector);
        assert_eq!(plan.hand_rects.len(), 2);
    }

    // ---- deduplication ----

    #[test]
    fn dedup_noop_for_zero_or_one_hand() {
        let lms = upright_hand();
        assert!(find_duplicate_hand_indices(&[], (640.0, 480.0)).is_empty());
        assert!(find_duplicate_hand_indices(&[lms], (640.0, 480.0)).is_empty());
    }

    #[test]
    fn dedup_suppresses_near_identical_second_hand() {
        let a = upright_hand();
        // b = a shifted by 5 px (well within 0.2 * baseline threshold).
        let b: Vec<[f32; 3]> = a
            .iter()
            .map(|lm| [lm[0] + 5.0 / 640.0, lm[1] + 5.0 / 480.0, lm[2]])
            .collect();
        let out = find_duplicate_hand_indices(&[a, b], (640.0, 480.0));
        assert_eq!(out, vec![1], "later duplicate suppressed");
    }

    #[test]
    fn dedup_keeps_spatially_distinct_hands() {
        let a = upright_hand();
        let b: Vec<[f32; 3]> = a
            .iter()
            .map(|lm| [lm[0] + 0.3, lm[1], lm[2]])
            .collect();
        let out = find_duplicate_hand_indices(&[a, b], (640.0, 480.0));
        assert!(out.is_empty(), "distinct hands kept: {out:?}");
    }
}
