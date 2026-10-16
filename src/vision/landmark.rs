//! Landmark decode + inverse mapping (M3.3) — ports of MediaPipe v1.1.1's
//! `TensorsToLandmarksCalculator` (normalize branch), 
//! `LandmarkLetterboxRemovalCalculator`, and
//! `LandmarkProjectionCalculator`'s no-IMAGE_DIMENSIONS branch (the one v1
//! actually runs — confirmed by its runtime warning on the reference frame).

use crate::vision::rect::NormRect;

/// `TensorsToLandmarksCalculator` with normalize=true (the hand landmarks
/// branch): raw model output [x, y, z] *triples* in 224-space pixel coords
/// become crop-normalized landmarks: x/224, y/224, z/224/0.4.
pub const LANDMARK_TENSOR_SIZE: f32 = 224.0;
pub const NORMALIZE_Z: f32 = 0.4;

/// Decode the raw [1, 63] landmark tensor into 21 crop-normalized [x,y,z].
pub fn decode_landmarks(raw: &[f32]) -> [[f32; 3]; 21] {
    let mut out = [[0.0f32; 3]; 21];
    for (i, lm) in out.iter_mut().enumerate() {
        lm[0] = raw[i * 3] / LANDMARK_TENSOR_SIZE;
        lm[1] = raw[i * 3 + 1] / LANDMARK_TENSOR_SIZE;
        lm[2] = raw[i * 3 + 2] / LANDMARK_TENSOR_SIZE / NORMALIZE_Z;
    }
    out
}

/// `LandmarkLetterboxRemovalCalculator` verbatim: un-letterbox normalized
/// landmarks by the padding [left, top, right, bottom] fractions.
/// v1's landmark stage pads [0,0,0,0] (square roi), so this is a no-op on
/// the parity path — ported for spec completeness and detector-style use.
pub fn letterbox_remove(landmarks: &mut [[f32; 3]], padding: [f32; 4]) {
    let [left, top, right, bottom] = padding;
    let z_scale = 1.0 / (1.0 - left - right);
    for lm in landmarks.iter_mut() {
        lm[0] = (lm[0] - left) * z_scale;
        lm[1] = (lm[1] - top) / (1.0 - top - bottom);
        lm[2] *= z_scale;
    }
}

/// `LandmarkProjectionCalculator`'s no-IMAGE_DIMENSIONS branch verbatim:
/// derotate about (0.5, 0.5), scale by the rect's normalized width/height,
/// shift to the rect center; z scales by the normalized width.
pub fn project_landmarks(landmarks: &[[f32; 3]], rect: &NormRect) -> Vec<[f32; 3]> {
    let (c, s) = (rect.rotation.cos(), rect.rotation.sin());
    landmarks
        .iter()
        .map(|lm| {
            let x = lm[0] - 0.5;
            let y = lm[1] - 0.5;
            [
                (c * x - s * y) * rect.width + rect.x_center,
                (s * x + c * y) * rect.height + rect.y_center,
                lm[2] * rect.width,
            ]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    #[test]
    fn decode_scales_pixel_coords_to_crop_normalized() {
        let mut raw = vec![0.0f32; 63];
        raw[0] = 112.0; // x -> 0.5
        raw[1] = 56.0; // y -> 0.25
        raw[2] = 89.6; // z -> 89.6/224/0.4 = 1.0
        raw[63 - 3] = 224.0; // last landmark x -> 1.0
        let lms = decode_landmarks(&raw);
        assert!((lms[0][0] - 0.5).abs() < EPS, "{}", lms[0][0]);
        assert!((lms[0][1] - 0.25).abs() < EPS, "{}", lms[0][1]);
        assert!((lms[0][2] - 1.0).abs() < EPS, "{}", lms[0][2]);
        assert!((lms[20][0] - 1.0).abs() < EPS, "{}", lms[20][0]);
    }

    #[test]
    fn letterbox_remove_is_identity_on_zero_padding() {
        let mut lms = [[0.3f32, 0.7, 0.1]; 21];
        let before = lms;
        letterbox_remove(&mut lms, [0.0, 0.0, 0.0, 0.0]);
        for (a, b) in lms.iter().zip(before.iter()) {
            assert!((a[0] - b[0]).abs() < EPS && (a[1] - b[1]).abs() < EPS);
        }
    }

    #[test]
    fn letterbox_remove_shifts_by_padding() {
        // 10% bars top/bottom: y in [0.1, 0.9] -> [0, 1]; x unchanged;
        // z scales by the x factor (1.0 here).
        let mut lms = [[0.5f32, 0.5, 2.0], [0.25, 0.1, -1.0], [0.75, 0.9, 0.5]];
        letterbox_remove(&mut lms, [0.0, 0.1, 0.0, 0.1]);
        assert!((lms[0][1] - 0.5).abs() < EPS, "center stays: {}", lms[0][1]);
        assert!((lms[1][1]).abs() < EPS, "top edge -> 0: {}", lms[1][1]);
        assert!((lms[2][1] - 1.0).abs() < EPS, "bottom edge -> 1: {}", lms[2][1]);
        assert!((lms[0][0] - 0.5).abs() < EPS);
        assert!((lms[0][2] - 2.0).abs() < EPS, "z x-factor is 1: {}", lms[0][2]);
    }

    #[test]
    fn projection_identity_rect_passes_through() {
        let rect = NormRect {
            x_center: 0.5,
            y_center: 0.5,
            width: 1.0,
            height: 1.0,
            rotation: 0.0,
        };
        let lms = [[0.3f32, 0.7, 0.5]];
        let out = project_landmarks(&lms, &rect);
        assert!((out[0][0] - 0.3).abs() < EPS && (out[0][1] - 0.7).abs() < EPS);
        assert!((out[0][2] - 0.5).abs() < EPS, "z * width(1) = z: {}", out[0][2]);
    }

    #[test]
    fn projection_scales_and_centers_rect() {
        let rect = NormRect {
            x_center: 0.5,
            y_center: 0.4,
            width: 0.25,
            height: 0.5,
            rotation: 0.0,
        };
        let lms = [
            [0.5f32, 0.5, 0.0],  // crop center -> rect center
            [0.75, 0.5, 2.0],    // right edge -> x = 0.5 + 0.25*0.25
            [0.5, 1.0, 0.0],     // bottom -> y = 0.4 + 0.5*0.5
        ];
        let out = project_landmarks(&lms, &rect);
        assert!((out[0][0] - 0.5).abs() < EPS && (out[0][1] - 0.4).abs() < EPS);
        assert!((out[1][0] - 0.5625).abs() < EPS, "{}", out[1][0]);
        assert!((out[2][1] - 0.65).abs() < EPS, "{}", out[2][1]);
        assert!((out[1][2] - 2.0 * 0.25).abs() < EPS, "z * width: {}", out[1][2]);
    }

    #[test]
    fn projection_rotates_about_rect_center() {
        // 90 deg CCW-in-math (R = [[0,-1],[1,0]]): crop-right maps to
        // image-down.
        let rect = NormRect {
            x_center: 0.5,
            y_center: 0.5,
            width: 0.25,
            height: 0.25,
            rotation: std::f32::consts::FRAC_PI_2,
        };
        let lms = [[0.75f32, 0.5, 0.0]]; // local (0.25, 0)
        let out = project_landmarks(&lms, &rect);
        assert!((out[0][0] - 0.5).abs() < EPS, "x {}", out[0][0]);
        assert!((out[0][1] - 0.5625).abs() < EPS, "y {}", out[0][1]);
    }

    #[test]
    fn projection_inverts_the_warp_sampling_convention() {
        // The crop samples src = c + R*(local), projection does local =
        // R^T*(p - c) -> a crop-corner landmark projects to the roi's corner
        // in IMAGE-NORMALIZED space. Exact only for square-in-pixels ROIs
        // (w*iw == h*ih) — the square_long invariant upstream requires
        // ("only supported for the square ROI"). Check against get_roi px.
        use crate::vision::warp::get_roi;
        let rect = NormRect {
            x_center: 0.5,
            y_center: 0.5,
            width: 0.125,      // 80 px at 640x480
            height: 0.125 * 640.0 / 480.0, // 80 px -> square-in-pixels
            rotation: 0.464_609_3,
        };
        let roi = get_roi(640, 480, &rect);
        // Crop corner (0,0) = local (-w/2, -h/2) -> src px:
        let (s, c) = (rect.rotation.sin(), rect.rotation.cos());
        let (lx, ly) = (-roi.width / 2.0, -roi.height / 2.0);
        let px = roi.center_x + c * lx - s * ly;
        let py = roi.center_y + s * lx + c * ly;
        let out = project_landmarks(&[[0.0f32, 0.0, 0.0]], &rect);
        assert!((out[0][0] * 640.0 - px).abs() < 1e-3, "x {}", out[0][0] * 640.0);
        assert!((out[0][1] * 480.0 - py).abs() < 1e-3, "y {}", out[0][1] * 480.0);
    }
}
