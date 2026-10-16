//! Detection-to-rect stage (M3.3) — ports of MediaPipe v1.1.1
//! `DetectionsToRectsCalculator` (keypoints -> raw rect + rotation) and
//! `RectTransformationCalculator` (expand/shift/square), wired exactly like
//! the task pipeline's `HandDetectorGraph::Configure*` functions.

/// A normalized rect: centers/size in image-fraction units, rotation in
/// radians (MediaPipe `NormalizedRect`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormRect {
    pub x_center: f32,
    pub y_center: f32,
    pub width: f32,
    pub height: f32,
    pub rotation: f32,
}

// HandDetectorGraph::ConfigureDetectionsToRectsCalculator:
// start=kp0 (wrist), end=kp2 (middle MCP), target angle set on the
// *radians* field with value 90 (upstream quirk — an upright hand gets a
// rect rotated by normalize(90 - pi/2) = 0.4646 rad = 26.6 deg). v1 runs
// this same C++, so parity = replicate verbatim; the crop/landmark
// projection both use the same rect rotation, so the skew cancels out.
pub const TARGET_ANGLE_RADIANS: f32 = 90.0;

// HandDetectorGraph::ConfigureRectTransformationCalculator (detection rect).
pub const DETECT_SCALE: f32 = 2.6;
pub const DETECT_SHIFT_X: f32 = 0.0;
pub const DETECT_SHIFT_Y: f32 = -0.5;

// SingleHandLandmarksDetectorGraph::ConfigureHandRectTransformationCalculator
// (next-frame tracking rect, used from M3.4 on).
pub const NEXT_FRAME_SCALE: f32 = 2.0;
pub const NEXT_FRAME_SHIFT_X: f32 = 0.0;
pub const NEXT_FRAME_SHIFT_Y: f32 = -0.1;

/// `NormalizeRadians` — wraps into [-pi, pi).
pub fn normalize_radians(angle: f32) -> f32 {
    let two_pi = 2.0 * std::f32::consts::PI;
    angle - two_pi * ((angle + std::f32::consts::PI) / two_pi).floor()
}

/// `DetectionsToRectsCalculator::ComputeRotation` verbatim:
/// `normalize(target - atan2(-(y1 - y0), x1 - x0))` with keypoint
/// coordinates scaled to pixels (x by width, y by height — the anisotropic
/// scale affects the angle and is part of the spec).
pub fn detection_rotation(
    kp_start: [f32; 2],
    kp_end: [f32; 2],
    image_size: (f32, f32),
) -> f32 {
    let (x0, y0) = (kp_start[0] * image_size.0, kp_start[1] * image_size.1);
    let (x1, y1) = (kp_end[0] * image_size.0, kp_end[1] * image_size.1);
    normalize_radians(TARGET_ANGLE_RADIANS - (-(y1 - y0)).atan2(x1 - x0))
}

/// `NormRectFromKeyPoints` verbatim: axis-aligned bbox over all keypoints,
/// normalized coords.
pub fn norm_rect_from_keypoints(keypoints: &[[f32; 2]]) -> (f32, f32, f32, f32) {
    let mut min_x = f32::MAX;
    let mut min_y = f32::MAX;
    let mut max_x = f32::MIN;
    let mut max_y = f32::MIN;
    for kp in keypoints {
        min_x = min_x.min(kp[0]);
        max_x = max_x.max(kp[0]);
        min_y = min_y.min(kp[1]);
        max_y = max_y.max(kp[1]);
    }
    (
        (min_x + max_x) / 2.0,
        (min_y + max_y) / 2.0,
        max_x - min_x,
        max_y - min_y,
    )
}

/// `RectTransformationCalculator::TransformNormalizedRect` verbatim for the
/// HandDetectorGraph options (shift_x=0, shift_y=-0.5, scale 2.6,
/// square_long) and the next-frame options (shift_y=-0.1, scale 2.0).
/// Rotation is preserved.
pub fn transform_rect(
    rect: &NormRect,
    image_size: (f32, f32),
    scale_x: f32,
    scale_y: f32,
    shift_x: f32,
    shift_y: f32,
) -> NormRect {
    let (image_width, image_height) = image_size;
    let mut x_center = rect.x_center;
    let mut y_center = rect.y_center;
    let mut width = rect.width;
    let mut height = rect.height;
    let rotation = rect.rotation;

    if rotation == 0.0 {
        x_center += width * shift_x;
        y_center += height * shift_y;
    } else {
        let x_shift = (image_width * width * shift_x * rotation.cos()
            - image_height * height * shift_y * rotation.sin())
            / image_width;
        let y_shift = (image_width * width * shift_x * rotation.sin()
            + image_height * height * shift_y * rotation.cos())
            / image_height;
        x_center += x_shift;
        y_center += y_shift;
    }

    // square_long (always on in both HandDetectorGraph configs).
    let long_side = (width * image_width).max(height * image_height);
    width = long_side / image_width;
    height = long_side / image_height;

    NormRect {
        x_center,
        y_center,
        width: width * scale_x,
        height: height * scale_y,
        rotation,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    #[test]
    fn normalize_radians_wraps_to_pi_range() {
        assert!((normalize_radians(0.0)).abs() < EPS);
        assert!((normalize_radians(0.5 + 2.0 * std::f32::consts::PI) - 0.5).abs() < EPS);
        assert!((normalize_radians(-0.5 - 2.0 * std::f32::consts::PI) + 0.5).abs() < EPS);
        // Exactly -pi maps to +pi boundary of the half-open range: -pi stays.
        assert!((normalize_radians(-std::f32::consts::PI) + std::f32::consts::PI).abs() < EPS);
    }

    #[test]
    fn upright_hand_gets_quirked_rotation() {
        // Wrist directly below middle MCP -> atan2 term = pi/2, so with the
        // correct target (pi/2) rotation would be 0; v1's radians-field 90
        // gives normalize(90 - pi/2) = 0.46461 rad = 26.62 deg.
        let r = detection_rotation([0.5, 0.8], [0.5, 0.4], (640.0, 480.0));
        assert!(
            (r - 0.464_609_3).abs() < 1e-4,
            "expected quirked 0.46461, got {r}"
        );
    }

    #[test]
    fn horizontal_hand_rotation_is_quirk_shifted_too() {
        // Wrist left of MCP, same y: atan2 term = 0 -> normalize(90) =
        // 2.03541 rad (correct target would give pi/2 = 1.5708).
        let r = detection_rotation([0.4, 0.5], [0.6, 0.5], (640.0, 480.0));
        assert!((r - 2.035_405_7).abs() < 1e-4, "expected 2.03541, got {r}");
    }

    #[test]
    fn rotation_uses_pixel_space_with_anisotropic_scale() {
        // Same normalized delta, but image 1280x480 vs 640x480 -> different
        // pixel dx/dy ratio -> different angle. Upright vector in normalized
        // space stays upright only when x-delta is 0 (this test: delta x = 0
        // and delta y != 0 gives atan2(+-|dy|, 0) = +-pi/2 regardless of
        // scale, so instead use a slanted vector).
        let a = detection_rotation([0.4, 0.8], [0.6, 0.4], (640.0, 480.0));
        let b = detection_rotation([0.4, 0.8], [0.6, 0.4], (1280.0, 480.0));
        assert!((a - b).abs() > 0.1, "scale must affect the angle: {a} vs {b}");
        // Explicit value for 640x480: dx = 128, dy = -192.
        // atan2(192, 128) = 0.98279; normalize(90 - 0.98279) = 1.05262.
        assert!((a - 1.052_616_6).abs() < 1e-3, "expected ~1.05262, got {a}");
    }

    #[test]
    fn keypoints_bbox_forms_raw_rect() {
        // 7 palm keypoints, two extremes (x 0.4..0.6, y 0.3..0.7).
        let kps = [
            [0.4, 0.7],
            [0.6, 0.3],
            [0.5, 0.5],
            [0.45, 0.6],
            [0.55, 0.4],
            [0.42, 0.65],
            [0.58, 0.35],
        ];
        let (cx, cy, w, h) = norm_rect_from_keypoints(&kps);
        assert!((cx - 0.5).abs() < EPS);
        assert!((cy - 0.5).abs() < EPS);
        assert!((w - 0.2).abs() < EPS);
        assert!((h - 0.4).abs() < EPS);
    }

    #[test]
    fn transform_zero_rotation_shifts_scales_and_squares() {
        // rotation = 0 branch: shift is (w*shift_x, h*shift_y) unscaled by
        // image size; square_long uses PIXEL dims; then scale_x/scale_y.
        let rect = NormRect {
            x_center: 0.5,
            y_center: 0.5,
            width: 0.072,
            height: 0.096, // 46.08 px at 640x480 -> already equal in px
            rotation: 0.0,
        };
        let out = transform_rect(
            &rect,
            (640.0, 480.0),
            DETECT_SCALE,
            DETECT_SCALE,
            DETECT_SHIFT_X,
            DETECT_SHIFT_Y,
        );
        assert!((out.x_center - 0.5).abs() < EPS, "shift_x is 0 -> {}", out.x_center);
        assert!(
            (out.y_center - (0.5 - 0.096 * 0.5)).abs() < EPS,
            "y += h * -0.5 -> {}",
            out.y_center
        );
        // long side = max(0.072*640, 0.096*480) = 46.08 px both ways;
        // normalized per axis then * 2.6.
        assert!((out.width - 0.072 * 2.6).abs() < EPS, "w -> {}", out.width);
        assert!((out.height - 0.096 * 2.6).abs() < EPS, "h -> {}", out.height);
        assert!((out.rotation - 0.0).abs() < EPS);
    }

    #[test]
    fn transform_rotated_branch_uses_pixel_mixed_shift() {
        // rotation != 0 branch (RectTransformationCalculator lines 220-230):
        // x_shift = (iw*w*sx*cos - ih*h*sy*sin) / iw
        // y_shift = (iw*w*sx*sin + ih*h*sy*cos) / ih
        let rot = 0.464_609_3f32;
        let rect = NormRect {
            x_center: 0.5,
            y_center: 0.5,
            width: 0.072,
            height: 0.096,
            rotation: rot,
        };
        let (iw, ih) = (640.0f32, 480.0f32);
        let out = transform_rect(&rect, (iw, ih), 2.6, 2.6, 0.0, -0.5);
        let (s, c) = (rot.sin(), rot.cos());
        let x_shift = (iw * 0.072 * 0.0 * c - ih * 0.096 * -0.5 * s) / iw;
        let y_shift = (iw * 0.072 * 0.0 * s + ih * 0.096 * -0.5 * c) / ih;
        assert!((out.x_center - (0.5 + x_shift)).abs() < EPS, "{}", out.x_center);
        assert!((out.y_center - (0.5 + y_shift)).abs() < EPS, "{}", out.y_center);
        // square_long in pixels: long = max(0.072*640, 0.096*480) = 46.08.
        assert!((out.width - (46.08 / 640.0) * 2.6).abs() < EPS);
        assert!((out.height - (46.08 / 480.0) * 2.6).abs() < EPS);
        // Square in PIXELS after transform.
        assert!((out.width * iw - out.height * ih).abs() < 1e-3);
        assert!((out.rotation - rot).abs() < EPS, "rotation preserved");
    }

    #[test]
    fn transform_handles_non_square_raw_bbox() {
        // Raw bbox 0.1 x 0.4 at 640x480 -> px 64 x 192 -> long = 192 px.
        let rect = NormRect {
            x_center: 0.5,
            y_center: 0.5,
            width: 0.1,
            height: 0.4,
            rotation: 0.0,
        };
        let out = transform_rect(&rect, (640.0, 480.0), 2.6, 2.6, 0.0, -0.5);
        assert!((out.width * 640.0 - 192.0 * 2.6).abs() < 1e-3, "px w {}", out.width * 640.0);
        assert!((out.height * 480.0 - 192.0 * 2.6).abs() < 1e-3, "px h {}", out.height * 480.0);
    }
}
