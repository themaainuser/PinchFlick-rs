//! Rotated-crop stage (M3.3) — ports of MediaPipe v1.1.1's
//! `image_to_tensor` path: `GetRoi`, `PadRoi`,
//! `GetRotatedSubRectToRectTransformMatrix`, and the OpenCV
//! `warpPerspective`-equivalent bilinear crop (replicate border — the
//! proto's documented default for the landmark stage).

use crate::vision::rect::NormRect;

/// A rect in pixel space (MediaPipe `RotatedRect`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RotatedRect {
    pub center_x: f32,
    pub center_y: f32,
    pub width: f32,
    pub height: f32,
    pub rotation: f32,
}

/// `GetRoi`: normalized rect -> pixel-space rotated rect.
/// Centers scale by image size, sizes scale likewise; rotation carries over.
pub fn get_roi(image_width: u32, image_height: u32, rect: &NormRect) -> RotatedRect {
    RotatedRect {
        center_x: rect.x_center * image_width as f32,
        center_y: rect.y_center * image_height as f32,
        width: rect.width * image_width as f32,
        height: rect.height * image_height as f32,
        rotation: rect.rotation,
    }
}

/// `PadRoi`: when `keep_aspect_ratio`, grows the roi to a square in
/// parent-pixel units (the letterbox bands fall outside the image and get
/// filled by the border mode) and RETURNS the tensor-space padding as
/// [left, top, right, bottom] fractions. Without keep_aspect_ratio, roi and
/// padding ([0,0,0,0]) are unchanged.
/// v1's landmark stage runs with keep_aspect_ratio = false (proto default,
/// unset in ConfigureImageToTensorCalculator); the square roi makes it moot.
pub fn pad_roi(
    tensor_width: u32,
    tensor_height: u32,
    keep_aspect_ratio: bool,
    roi: &mut RotatedRect,
) -> [f32; 4] {
    if !keep_aspect_ratio {
        return [0.0, 0.0, 0.0, 0.0];
    }
    let tensor_aspect_ratio = tensor_height as f32 / tensor_width as f32;
    let roi_aspect_ratio = roi.height / roi.width;
    let (new_width, new_height, vertical, horizontal) = if tensor_aspect_ratio > roi_aspect_ratio {
        (
            roi.width,
            roi.width * tensor_aspect_ratio,
            (1.0 - roi_aspect_ratio / tensor_aspect_ratio) / 2.0,
            0.0,
        )
    } else {
        (
            roi.height / tensor_aspect_ratio,
            roi.height,
            0.0,
            (1.0 - tensor_aspect_ratio / roi_aspect_ratio) / 2.0,
        )
    };
    roi.width = new_width;
    roi.height = new_height;
    [horizontal, vertical, horizontal, vertical]
}

/// `GetRotatedSubRectToRectTransformMatrix`: 4x4 row-major affine mapping
/// IMAGE coords (normalized 0..1) -> TENSOR coords (normalized 0..1).
/// Only the affine 2x3 part (m0,m1,m3,m4,m5,m7) is meaningful for 2D.
pub fn subrect_to_image_matrix(roi: &RotatedRect, image_width: u32, image_height: u32) -> [f32; 16] {
    let (a, b) = (roi.width, roi.height);
    let (c, d) = (roi.rotation.cos(), roi.rotation.sin());
    let (e, f) = (roi.center_x, roi.center_y);
    let (g, h) = (1.0 / image_width as f32, 1.0 / image_height as f32);
    // flip = +1 (no horizontal flip in the landmark path).
    let mut m = [0.0f32; 16];
    m[0] = a * c * g;
    m[1] = -b * d * g;
    m[3] = (-0.5 * a * c + 0.5 * b * d + e) * g;
    m[4] = a * d * h;
    m[5] = b * c * h;
    m[7] = (-0.5 * b * c - 0.5 * a * d + f) * h;
    m[10] = a * g;
    m[15] = 1.0;
    m
}

/// The OpenCV `getPerspectiveTransform` + `warpPerspective` crop, ported
/// directly: for each output pixel (integer coords = corner convention of
/// dst_corners {0..tensor_size}), sample
///   src = center + R(rotation) * ((u/W - 0.5) * roi_w, (v/H - 0.5) * roi_h)
/// with bilinear interpolation, replicate border (kReplicate — the proto's
/// documented default), then map the 0..255 value linearly into
/// `value_range`.
pub fn warp_crop(
    rgb: &[u8],
    image_width: u32,
    image_height: u32,
    roi: &RotatedRect,
    tensor_size: u32,
    value_range: (f32, f32),
) -> Vec<f32> {
    let (c, s) = (roi.rotation.cos(), roi.rotation.sin());
    let (lo, hi) = value_range;
    let scale = (hi - lo) / 255.0;
    let mut out = vec![0.0f32; (tensor_size * tensor_size * 3) as usize];
    for v in 0..tensor_size {
        // Local offset of this output row in roi pixels. Corner convention:
        // dst_corners {0..tensor_size} pair with boxPoints, integer dst pixel
        // coords map through the inverse homography (OpenCV warpPerspective,
        // pixel centers at integers, no half-pixel shift).
        let local_y = ((v as f32 / tensor_size as f32) - 0.5) * roi.height;
        for u in 0..tensor_size {
            let local_x = ((u as f32 / tensor_size as f32) - 0.5) * roi.width;
            // src = center + R * local.
            let sx = roi.center_x + c * local_x - s * local_y;
            let sy = roi.center_y + s * local_x + c * local_y;
            // Bilinear with replicate border (pixel centers at integers).
            let x0 = sx.floor();
            let y0 = sy.floor();
            let (fx, fy) = (sx - x0, sy - y0);
            let base = ((v * tensor_size + u) * 3) as usize;
            for ch in 0..3usize {
                let p = |xx: f32, yy: f32| {
                    let xx = (xx as i32).clamp(0, image_width as i32 - 1) as u32;
                    let yy = (yy as i32).clamp(0, image_height as i32 - 1) as u32;
                    rgb[((yy * image_width + xx) as usize) * 3 + ch] as f32
                };
                let v00 = p(x0, y0);
                let v10 = p(x0 + 1.0, y0);
                let v01 = p(x0, y0 + 1.0);
                let v11 = p(x0 + 1.0, y0 + 1.0);
                let val = v00 * (1.0 - fx) * (1.0 - fy)
                    + v10 * fx * (1.0 - fy)
                    + v01 * (1.0 - fx) * fy
                    + v11 * fx * fy;
                out[base + ch] = lo + val * scale;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    #[test]
    fn get_roi_maps_normalized_to_pixels() {
        let rect = NormRect {
            x_center: 0.5,
            y_center: 0.5,
            width: 1.0,
            height: 1.0,
            rotation: 0.3,
        };
        let roi = get_roi(640, 480, &rect);
        assert!((roi.center_x - 320.0).abs() < EPS);
        assert!((roi.center_y - 240.0).abs() < EPS);
        assert!((roi.width - 640.0).abs() < EPS);
        assert!((roi.height - 480.0).abs() < EPS);
        assert!((roi.rotation - 0.3).abs() < EPS);
    }

    #[test]
    fn pad_roi_disabled_returns_zero_padding() {
        let mut roi = RotatedRect {
            center_x: 320.0,
            center_y: 240.0,
            width: 640.0,
            height: 480.0,
            rotation: 0.0,
        };
        let pad = pad_roi(224, 224, false, &mut roi);
        assert_eq!(pad, [0.0, 0.0, 0.0, 0.0]);
        assert!((roi.width - 640.0).abs() < EPS && (roi.height - 480.0).abs() < EPS);
    }

    #[test]
    fn pad_roi_letterboxes_wide_roi_like_the_detector() {
        // Cross-validates the M3.2 detector letterbox: a 640x480 roi into a
        // square tensor pads top/bottom by (1 - 480/640)/2 = 0.125.
        let mut roi = RotatedRect {
            center_x: 320.0,
            center_y: 240.0,
            width: 640.0,
            height: 480.0,
            rotation: 0.0,
        };
        let pad = pad_roi(192, 192, true, &mut roi);
        assert!((pad[0]).abs() < EPS, "left {}", pad[0]);
        assert!((pad[1] - 0.125).abs() < EPS, "top {}", pad[1]);
        assert!((pad[2]).abs() < EPS, "right {}", pad[2]);
        assert!((pad[3] - 0.125).abs() < EPS, "bottom {}", pad[3]);
        // Roi grows to a SQUARE in parent-pixel units (PadRoi keeps the roi
        // in image pixels; the tensor-sized letterbox falls out of the
        // sampling, with the padded bands filled by the border mode).
        assert!((roi.width - 640.0).abs() < 1e-3, "w {}", roi.width);
        assert!((roi.height - 640.0).abs() < 1e-3, "h {}", roi.height);
    }

    #[test]
    fn matrix_is_identity_for_full_frame_rect() {
        let roi = RotatedRect {
            center_x: 320.0,
            center_y: 240.0,
            width: 640.0,
            height: 480.0,
            rotation: 0.0,
        };
        let m = subrect_to_image_matrix(&roi, 640, 480);
        let ident = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        for (i, &v) in ident.iter().enumerate() {
            assert!((m[i] - v).abs() < EPS, "m[{i}] = {v}, got {m:?}");
        }
    }

    #[test]
    fn matrix_rotates_with_pi_over_two() {
        // roi rotated 90 deg, square 480x480 at image center:
        // x_t = -0.75*y + 0.875, y_t = x (image-norm -> tensor-norm).
        let roi = RotatedRect {
            center_x: 320.0,
            center_y: 240.0,
            width: 480.0,
            height: 480.0,
            rotation: std::f32::consts::FRAC_PI_2,
        };
        let m = subrect_to_image_matrix(&roi, 640, 480);
        assert!((m[0]).abs() < EPS, "m0 {}", m[0]);
        assert!((m[1] + 0.75).abs() < EPS, "m1 {}", m[1]);
        assert!((m[3] - 0.875).abs() < EPS, "m3 {}", m[3]);
        assert!((m[4] - 1.0).abs() < EPS, "m4 {}", m[4]);
        assert!((m[5]).abs() < EPS, "m5 {}", m[5]);
        assert!((m[7]).abs() < EPS, "m7 {}", m[7]);
        // Center point maps to center.
        let (x, y) = (m[0] * 0.5 + m[1] * 0.5 + m[3], m[4] * 0.5 + m[5] * 0.5 + m[7]);
        assert!((x - 0.5).abs() < EPS && (y - 0.5).abs() < EPS);
    }

    #[test]
    fn warp_full_frame_rect_equals_direct_sampling() {
        // Identity roi over a solid-color frame -> every output pixel is
        // that color, in range (0,1) scaling.
        let img = vec![255u8; 8 * 8 * 3];
        let roi = RotatedRect {
            center_x: 4.0,
            center_y: 4.0,
            width: 8.0,
            height: 8.0,
            rotation: 0.0,
        };
        let out = warp_crop(&img, 8, 8, &roi, 4, (0.0, 1.0));
        assert_eq!(out.len(), 4 * 4 * 3);
        assert!(out.iter().all(|&v| (v - 1.0).abs() < 1e-6));
    }

    #[test]
    fn warp_replicates_border_outside_the_image() {
        // Corner output pixels sample outside a shifted roi -> replicate
        // the nearest edge pixel (255), not black.
        let img = vec![255u8; 8 * 8 * 3];
        let roi = RotatedRect {
            center_x: 12.0, // mostly outside the 8x8 image, to the right
            center_y: 4.0,
            width: 8.0,
            height: 8.0,
            rotation: 0.0,
        };
        let out = warp_crop(&img, 8, 8, &roi, 4, (0.0, 1.0));
        assert!(out.iter().all(|&v| (v - 1.0).abs() < 1e-6), "replicate, not zero-fill");
    }

    #[test]
    fn warp_places_a_bright_dot_at_its_rotated_position() {
        // 32x32 frame, bright dot at (20, 12). Rect: 16x16 square at center
        // (16,16), rotation pi/2. Sampling (from the dst_corners/boxPoints
        // correspondence): src = c + R(rot) * (u - 8, v - 8), so dst (u,v)
        // samples (24 - v, 8 + u). The dot is hit where 24 - v = 20 and
        // 8 + u = 12 -> dst (4, 4).
        let (w, h) = (32u32, 32u32);
        let mut img = vec![0u8; (w * h * 3) as usize];
        let (dx, dy) = (20, 12);
        img[((dy * w + dx) * 3) as usize] = 255;
        img[((dy * w + dx) * 3 + 1) as usize] = 255;
        img[((dy * w + dx) * 3 + 2) as usize] = 255;
        let roi = RotatedRect {
            center_x: 16.0,
            center_y: 16.0,
            width: 16.0,
            height: 16.0,
            rotation: std::f32::consts::FRAC_PI_2,
        };
        let t = 16u32;
        let out = warp_crop(&img, w, h, &roi, t, (0.0, 1.0));
        let luma = |x: u32, y: u32| out[((y * t + x) * 3) as usize];
        // Argmax over the tensor.
        let (mut bx, mut by, mut bv) = (0, 0, 0.0f32);
        for y in 0..t {
            for x in 0..t {
                if luma(x, y) > bv {
                    (bx, by, bv) = (x, y, luma(x, y));
                }
            }
        }
        assert_eq!((bx, by), (4, 4), "dot at tensor (4,4), got ({bx},{by})");
        assert!(bv > 0.9, "bilinear peak on an integer sample is exact: {bv}");
    }

    #[test]
    fn letterbox_padding_round_trips_with_projection() {
        // crop corners of a padded roi project back to the roi corners in
        // image space — validated here against the identity rotation case
        // via the projection formula (used in landmark.rs tests too).
        use crate::vision::landmark::project_landmarks;
        let rect = NormRect {
            x_center: 0.5,
            y_center: 0.5,
            width: 0.25,
            height: 0.25,
            rotation: 0.0,
        };
        let lms = [[0.0f32, 0.0, 0.0], [1.0, 1.0, 0.0]];
        let out = project_landmarks(&lms, &rect);
        assert!((out[0][0] - 0.375).abs() < EPS && (out[0][1] - 0.375).abs() < EPS);
        assert!((out[1][0] - 0.625).abs() < EPS && (out[1][1] - 0.625).abs() < EPS);
    }
}
