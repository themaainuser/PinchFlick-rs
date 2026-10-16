//! Detector output decoding — a direct port of MediaPipe v1.1.1's
//! `TensorsToDetectionsCalculator` (CPU path), `NonMaxSuppressionCalculator`
//! (WEIGHTED), and `DetectionProjectionCalculator` with the options
//! `hand_detector_graph.cc` sets. Input: the palm detector's two raw output
//! tensors + the 2016 anchors. Output: palm detections with 7 keypoints,
//! normalized to the input image.

use crate::vision::anchors::Anchor;

/// One decoded palm detection. Box and keypoints are normalized to the
/// 192x192 input tensor until projected, then to the input image.
#[derive(Debug, Clone, PartialEq)]
pub struct PalmDetection {
    /// Sigmoid-applied score, already threshold-filtered by the caller.
    pub score: f32,
    /// (ymin, xmin, ymax, xmax), normalized.
    pub bbox: [f32; 4],
    /// 7 keypoints, (x, y) normalized.
    pub keypoints: Vec<[f32; 2]>,
}

/// Decodes the palm detector's raw tensors into detections, exactly as
/// `TensorsToDetectionsCalculator` with `hand_detector_graph.cc`'s options:
/// XYWH format (reverse_output_order), linear scale by 192 (no exponential),
/// score = sigmoid(clip(logit, ±100)), boxes with negative width/height or
/// NaN dropped, `min_score_thresh` applied. Keypoint offset 4, 7 keypoints
/// of (x, y) each, same scale/anchor transform as the box centers.
///
/// `raw_boxes` is row-major [2016][18], `raw_scores` is [2016] raw logits.
pub fn decode_detections(
    raw_boxes: &[f32],
    raw_scores: &[f32],
    anchors: &[Anchor],
    min_score_thresh: f32,
) -> Vec<PalmDetection> {
    const NUM_COORDS: usize = 18;
    const NUM_KEYPOINTS: usize = 7;
    const KEYPOINT_OFFSET: usize = 4;
    const SCALE: f32 = 192.0;
    const CLIP: f32 = 100.0;

    let num_boxes = raw_scores.len();
    debug_assert_eq!(raw_boxes.len(), num_boxes * NUM_COORDS);
    debug_assert_eq!(anchors.len(), num_boxes);

    let mut out = Vec::new();
    for i in 0..num_boxes {
        let logit = raw_scores[i].clamp(-CLIP, CLIP);
        let score = 1.0 / (1.0 + (-logit).exp());
        if score < min_score_thresh {
            continue;
        }
        let a = &anchors[i];
        let row = &raw_boxes[i * NUM_COORDS..(i + 1) * NUM_COORDS];
        // XYWH (reverse_output_order=true): x, y, w, h raw.
        let x_center = row[0] / SCALE * a.w + a.x_center;
        let y_center = row[1] / SCALE * a.h + a.y_center;
        let w = row[2] / SCALE * a.w;
        let h = row[3] / SCALE * a.h;
        if w < 0.0 || h < 0.0 || w.is_nan() || h.is_nan() {
            continue;
        }
        let keypoints: Vec<[f32; 2]> = (0..NUM_KEYPOINTS)
            .map(|k| {
                let off = KEYPOINT_OFFSET + k * 2;
                [
                    row[off] / SCALE * a.w + a.x_center,
                    row[off + 1] / SCALE * a.h + a.y_center,
                ]
            })
            .collect();
        out.push(PalmDetection {
            score,
            bbox: [
                y_center - h / 2.0,
                x_center - w / 2.0,
                y_center + h / 2.0,
                x_center + w / 2.0,
            ],
            keypoints,
        });
    }
    out
}

/// IoU of two boxes in (ymin, xmin, ymax, xmax) form, matching MediaPipe's
/// `OverlapSimilarity` with INTERSECTION_OVER_UNION.
fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let inter_ymin = a[0].max(b[0]);
    let inter_xmin = a[1].max(b[1]);
    let inter_ymax = a[2].min(b[2]);
    let inter_xmax = a[3].min(b[3]);
    let inter_w = inter_xmax - inter_xmin;
    let inter_h = inter_ymax - inter_ymin;
    if inter_w <= 0.0 || inter_h <= 0.0 {
        return 0.0;
    }
    let inter = inter_w * inter_h;
    let area_a = (a[2] - a[0]) * (a[3] - a[1]);
    let area_b = (b[2] - b[0]) * (b[3] - b[1]);
    inter / (area_a + area_b - inter)
}

/// WEIGHTED non-maximum suppression — port of
/// `NonMaxSuppressionCalculator::WeightedNonMaxSuppression` with the palm
/// graph's options (IoU overlap, threshold 0.3, no min-score break since
/// detections are already thresholded upstream). Repeatedly takes the
/// highest-scoring remaining detection, clusters everything overlapping it
/// above the threshold, and emits one detection whose box and keypoints are
/// the score-weighted average of the cluster; the emitted score is the
/// cluster leader's. Output is in descending score order.
pub fn weighted_nms(detections: &[PalmDetection], iou_thresh: f32) -> Vec<PalmDetection> {
    let mut remained: Vec<usize> = (0..detections.len()).collect();
    // Descending by score (stable: preserves original order for ties).
    remained.sort_by(|&a, &b| {
        detections[b]
            .score
            .partial_cmp(&detections[a].score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut out = Vec::new();
    while let Some(&leader) = remained.first() {
        let leader_det = &detections[leader];
        let mut candidates: Vec<usize> = Vec::new();
        let mut rest: Vec<usize> = Vec::new();
        for &idx in &remained {
            if iou(&detections[idx].bbox, &leader_det.bbox) > iou_thresh {
                candidates.push(idx);
            } else {
                rest.push(idx);
            }
        }
        if candidates.len() == 1 {
            out.push(leader_det.clone());
        } else {
            let total: f32 = candidates.iter().map(|&i| detections[i].score).sum();
            let mut bbox = [0.0f32; 4];
            for &i in &candidates {
                let s = detections[i].score;
                // Weighted average of corners (matches xmin/xmax weighting
                // in the reference: it averages xmin, ymin, xmax, ymax).
                for c in 0..4 {
                    bbox[c] += detections[i].bbox[c] * s;
                }
            }
            for c in 0..4 {
                bbox[c] /= total;
            }
            let num_kp = leader_det.keypoints.len();
            let mut keypoints = vec![[0.0f32; 2]; num_kp];
            for &i in &candidates {
                let s = detections[i].score;
                for k in 0..num_kp {
                    keypoints[k][0] += detections[i].keypoints[k][0] * s;
                    keypoints[k][1] += detections[i].keypoints[k][1] * s;
                }
            }
            for k in 0..num_kp {
                keypoints[k][0] /= total;
                keypoints[k][1] /= total;
            }
            out.push(PalmDetection {
                score: leader_det.score,
                bbox,
                keypoints,
            });
        }
        remained = rest;
    }
    out
}

/// Projects tensor-space detections back to image-normalized coordinates via
/// the letterbox projection matrix (`GetRotatedSubRectToRectTransformMatrix`
/// for the padded square ROI). For a whole-image ROI with a square tensor:
/// x maps identity, y maps through the vertical padding — e.g. for 640x480
/// into 192x192, y_img = (y_t - 0.125) / 0.75.
pub fn project_detections_to_image(
    detections: &mut [PalmDetection],
    image_w: u32,
    image_h: u32,
    tensor_size: u32,
) {
    let (iw, ih, t) = (image_w as f32, image_h as f32, tensor_size as f32);
    // PadRoi: letterbox a whole-image ROI (center = image center) into the
    // square tensor — the padded ROI in image pixels.
    let tensor_ar = 1.0; // square tensor
    let roi_ar = ih / iw;
    let (new_w, new_h) = if tensor_ar > roi_ar {
        (iw, iw * tensor_ar)
    } else {
        (ih / tensor_ar, ih)
    };
    let (cx, cy) = (iw / 2.0, ih / 2.0);

    // GetRotatedSubRectToRectTransformMatrix with rotation=0, flip=+1:
    // out = ((in - t/2) * (new / t) + center) / image, per axis.
    let px_to_norm = |v: f32, new: f32, c: f32, img: f32| ((v - t / 2.0) * (new / t) + c) / img;

    for d in detections.iter_mut() {
        for kp in d.keypoints.iter_mut() {
            kp[0] = px_to_norm(kp[0] * t, new_w, cx, iw);
            kp[1] = px_to_norm(kp[1] * t, new_h, cy, ih);
        }
        let [ymin, xmin, ymax, xmax] = d.bbox;
        d.bbox = [
            px_to_norm(ymin * t, new_h, cy, ih),
            px_to_norm(xmin * t, new_w, cx, iw),
            px_to_norm(ymax * t, new_h, cy, ih),
            px_to_norm(xmax * t, new_w, cx, iw),
        ];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn anchor(xc: f32, yc: f32) -> Anchor {
        Anchor {
            x_center: xc,
            y_center: yc,
            w: 1.0,
            h: 1.0,
        }
    }

    /// 18-value row: [x, y, w, h, kp0x, kp0y, ... kp6x, kp6y] (XYWH).
    fn row(x: f32, y: f32, w: f32, h: f32) -> Vec<f32> {
        let mut r = vec![0.0; 18];
        r[0] = x;
        r[1] = y;
        r[2] = w;
        r[3] = h;
        r
    }

    #[test]
    fn decodes_xywh_linear_with_anchor_offset() {
        // Raw x=96/192=0.5 relative to a full-frame anchor at (0.5, 0.5):
        // center lands at 1.0 (right edge), w=0.25 -> [0.875, 1.125].
        let raw = row(96.0, 96.0, 48.0, 48.0);
        let scores = vec![2.0]; // sigmoid ~0.88
        let dets = decode_detections(&raw, &scores, &[anchor(0.5, 0.5)], 0.5);
        assert_eq!(dets.len(), 1);
        let d = &dets[0];
        assert!((d.bbox[1] - 0.875).abs() < 1e-6, "xmin {:?}", d.bbox);
        assert!((d.bbox[3] - 1.125).abs() < 1e-6, "xmax {:?}", d.bbox);
        // Same for y: y_center = 0.5 + 0.5 = 1.0, h = 0.25.
        assert!((d.bbox[0] - 0.875).abs() < 1e-6, "ymin {:?}", d.bbox);
        assert!((d.bbox[2] - 1.125).abs() < 1e-6, "ymax {:?}", d.bbox);
    }

    #[test]
    fn keypoints_decode_with_same_transform() {
        // kp0 raw (192, 192) -> (1.0+0.5, 1.0+0.5) relative to anchor.
        let mut raw = row(0.0, 0.0, 0.0, 0.0);
        raw[4] = 192.0;
        raw[5] = 192.0;
        let scores = vec![2.0];
        let dets = decode_detections(&raw, &scores, &[anchor(0.5, 0.5)], 0.5);
        assert_eq!(dets[0].keypoints.len(), 7);
        assert!((dets[0].keypoints[0][0] - 1.5).abs() < 1e-6);
        assert!((dets[0].keypoints[0][1] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn score_clipped_then_sigmoided() {
        // Logit 1e6 must clip to 100 -> sigmoid == 1.0; logit -1e6 -> 0.0.
        let raw = row(0.0, 0.0, 0.0, 0.0);
        let dets = decode_detections(&raw, &[1e6], &[anchor(0.5, 0.5)], 0.5);
        assert_eq!(dets.len(), 1);
        assert_eq!(dets[0].score, 1.0);
        let dets = decode_detections(&raw, &[-1e6], &[anchor(0.5, 0.5)], 0.5);
        assert!(dets.is_empty(), "sigmoid(clip(-1e6)) = 0 < 0.5");
    }

    #[test]
    fn min_score_thresh_filters() {
        // Logit 0 -> sigmoid 0.5; thresh 0.6 drops it.
        let raw = row(0.0, 0.0, 0.0, 0.0);
        let dets = decode_detections(&raw, &[0.0], &[anchor(0.5, 0.5)], 0.6);
        assert!(dets.is_empty());
    }

    #[test]
    fn negative_or_nan_box_dropped() {
        let raw = row(96.0, 96.0, -48.0, 48.0); // w < 0
        let scores = vec![2.0];
        assert!(decode_detections(&raw, &scores, &[anchor(0.5, 0.5)], 0.5).is_empty());

        let raw = row(96.0, 96.0, f32::NAN, 48.0);
        assert!(decode_detections(&raw, &scores, &[anchor(0.5, 0.5)], 0.5).is_empty());
    }

    fn det(score: f32, ymin: f32, xmin: f32, ymax: f32, xmax: f32) -> PalmDetection {
        PalmDetection {
            score,
            bbox: [ymin, xmin, ymax, xmax],
            keypoints: vec![[xmin, ymin]; 7],
        }
    }

    #[test]
    fn nms_suppresses_overlapping_and_averages_cluster() {
        // Three detections: two heavily overlapping (merged, score-weighted),
        // one disjoint (kept as-is). Output sorted by score descending.
        let a = det(0.9, 0.10, 0.10, 0.30, 0.30);
        let b = det(0.7, 0.12, 0.12, 0.32, 0.32); // IoU with a > 0.3
        let c = det(0.5, 0.60, 0.60, 0.80, 0.80); // disjoint
        let out = weighted_nms(&[c.clone(), b.clone(), a.clone()], 0.3);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].score, 0.9, "leader's score is kept");
        // Weighted average of a,b: weight sum 1.6.
        let w = 0.9f32;
        let s = 0.7f32;
        let expect_ymin = (0.10 * w + 0.12 * s) / (w + s);
        assert!((out[0].bbox[0] - expect_ymin).abs() < 1e-6, "got {}", out[0].bbox[0]);
        // Keypoints averaged with the same weights.
        let expect_kp = (0.10 * w + 0.12 * s) / (w + s);
        assert!((out[0].keypoints[0][0] - expect_kp).abs() < 1e-6);
        assert_eq!(out[1], c);
    }

    #[test]
    fn nms_empty_input_stays_empty() {
        assert!(weighted_nms(&[], 0.3).is_empty());
    }

    #[test]
    fn nms_respects_iou_threshold_boundary() {
        // IoU exactly at/below threshold must NOT merge; use two identical
        // boxes shifted so IoU is just under 0.3 -> both retained.
        // Box 1: [0,0,0.5,0.5] area 0.25. Box 2: [0.2,0.2,0.7,0.7] area 0.25.
        // Intersection [0.2,0.2,0.5,0.5] = 0.09; union = 0.41;
        // IoU = 0.09/0.41 = 0.2195 < 0.3 -> kept.
        let a = det(0.9, 0.0, 0.0, 0.5, 0.5);
        let b = det(0.8, 0.2, 0.2, 0.7, 0.7);
        let out = weighted_nms(&[a, b], 0.3);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn projection_maps_640x480_letterbox() {
        // 640x480 -> 192x192 keeps aspect: horizontal fills, vertical pads
        // 12.5% top and bottom. y_img = (y_t - 0.125) / 0.75; x identity.
        let mut dets = [det(1.0, 0.125, 0.5, 0.875, 0.5)];
        project_detections_to_image(&mut dets, 640, 480, 192);
        assert!((dets[0].bbox[0]).abs() < 1e-6, "top pad edge -> 0: {:?}", dets[0].bbox);
        assert!((dets[0].bbox[2] - 1.0).abs() < 1e-6, "bottom pad edge -> 1");
        assert!((dets[0].bbox[1] - 0.5).abs() < 1e-6, "x unchanged");
        assert!((dets[0].bbox[3] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn projection_maps_square_image_identity() {
        let mut dets = [det(1.0, 0.25, 0.25, 0.75, 0.75)];
        project_detections_to_image(&mut dets, 480, 480, 192);
        assert_eq!(dets[0].bbox, [0.25, 0.25, 0.75, 0.75]);
    }

    #[test]
    fn projection_also_moves_keypoints() {
        let mut dets = [det(1.0, 0.125, 0.0, 0.875, 1.0)];
        project_detections_to_image(&mut dets, 640, 480, 192);
        // keypoint was (xmin, ymin) = (0.0, 0.125) -> (0.0, 0.0).
        assert!((dets[0].keypoints[0][1]).abs() < 1e-6);
        assert!((dets[0].keypoints[0][0]).abs() < 1e-6);
    }
}
