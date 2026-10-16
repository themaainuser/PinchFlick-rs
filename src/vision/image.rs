//! Frame preparation for the palm detector — a port of MediaPipe's
//! `ImageToTensorCalculator` with the palm graph's options: keep-aspect
//! letterbox with zero border, then `/255` scaling to [0, 1]. The vtuberkit
//! ONNX conversion folds MediaPipe's `v/127.5 - 1` into an in-graph `2u-1`,
//! so the external input is `v/255` — mathematically the same normalization,
//! verified against v1's ground-truth detection (M3.2 visual gate).

/// Letterboxes an RGB frame into a square tensor of `size` x `size`
/// (aspect preserved, zero-padded) and scales to [0, 1] (`v/255`).
/// Bilinear sampling matches MediaPipe's CPU affine warp. Returns
/// NHWC-ready data for the palm detector ONNX.
pub fn letterbox_normalize(rgb: &[u8], width: u32, height: u32, size: u32) -> Vec<f32> {
    let (fw, fh, t) = (width as f32, height as f32, size as f32);
    // Fit the frame inside the square, centered (keep-aspect, zero border).
    let scale = (t / fw).min(t / fh);
    let (dw, dh) = (fw * scale, fh * scale);
    let (off_x, off_y) = ((t - dw) / 2.0, (t - dh) / 2.0);

    let mut out = vec![0.0f32; (t * t * 3.0) as usize]; // zero border stays 0.0 below
    for oy in 0..size {
        let sy = (oy as f32 + 0.5 - off_y) / scale - 0.5;
        if sy < 0.0 || sy > fh - 1.0 {
            continue;
        }
        let y0 = sy.floor().max(0.0) as u32;
        let y1 = (y0 + 1).min(height - 1);
        let fy = sy - y0 as f32;
        for ox in 0..size {
            let sx = (ox as f32 + 0.5 - off_x) / scale - 0.5;
            if sx < 0.0 || sx > fw - 1.0 {
                continue;
            }
            let x0 = sx.floor().max(0.0) as u32;
            let x1 = (x0 + 1).min(width - 1);
            let fx = sx - x0 as f32;
            for c in 0..3 {
                let p = |yy: u32, xx: u32| rgb[((yy * width + xx) * 3 + c) as usize] as f32;
                let top = p(y0, x0) * (1.0 - fx) + p(y0, x1) * fx;
                let bot = p(y1, x0) * (1.0 - fx) + p(y1, x1) * fx;
                let v = top * (1.0 - fy) + bot * fy;
                out[((oy * size + ox) * 3 + c) as usize] = v / 255.0;
            }
        }
    }
    out
}

/// v1 parity: `cv2.flip(frame, 1)` when `mirror_display` is true
/// (controller.py:180) — horizontal mirror of the RGB frame, applied BEFORE
/// inference.
pub fn mirror_rgb(buf: &mut [u8], width: u32, height: u32) {
    let w = width as usize;
    if w == 0 {
        return;
    }
    for y in 0..height as usize {
        let row = &mut buf[y * w * 3..(y + 1) * w * 3];
        for x in 0..w / 2 {
            for c in 0..3 {
                row.swap(x * 3 + c, (w - 1 - x) * 3 + c);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Contract (M3.2 ruling, verified against v1 ground truth): the vtuberkit
    // ONNX folds MediaPipe's (v/127.5 - 1) into an in-graph 2u-1, so the
    // external input must be v/255 in [0, 1] with a black (0.0) letterbox
    // border. Feeding v/127.5-1 makes the detector's logits explode
    // ([-12500, 2555], 93 false hits); v/255 yields sane logits ([-16, 1])
    // and the single hand v1 reports on the reference frame.

    #[test]
    fn output_has_expected_length() {
        let rgb = vec![0u8; 640 * 480 * 3];
        let out = letterbox_normalize(&rgb, 640, 480, 192);
        assert_eq!(out.len(), 192 * 192 * 3);
    }

    #[test]
    fn black_frame_normalizes_to_zero() {
        let rgb = vec![0u8; 640 * 480 * 3];
        let out = letterbox_normalize(&rgb, 640, 480, 192);
        // Content and the zero-border padding are both black -> 0.0 everywhere.
        assert!(out.iter().all(|&v| v.abs() < 1e-6));
    }

    #[test]
    fn white_content_normalizes_to_one() {
        // Solid white 640x480 -> content band [24..168) rows of the 192 grid
        // (480 * 0.3 = 144 rows centered); padding stays black.
        let rgb = vec![255u8; 640 * 480 * 3];
        let out = letterbox_normalize(&rgb, 640, 480, 192);
        let at = |y: u32, x: u32| out[((y * 192 + x) * 3) as usize];
        assert!((at(96, 96) - 1.0).abs() < 1e-6, "content center is white");
        assert!(at(0, 0).abs() < 1e-6, "padding is black");
        assert!(at(191, 191).abs() < 1e-6, "padding is black");
    }

    #[test]
    fn mid_gray_scales_by_255() {
        let rgb = vec![128u8; 640 * 480 * 3];
        let out = letterbox_normalize(&rgb, 640, 480, 192);
        let v = out[((96 * 192 + 96) * 3) as usize];
        assert!((v - 128.0 / 255.0).abs() < 1e-6, "128/255 = {}, got {v}", 128.0 / 255.0);
    }

    #[test]
    fn mirror_rgb_swaps_columns_like_cv2_flip_1() {
        // 2x2, distinct pixels per cell: rows [A B] / [C D] -> [B A] / [D C].
        let mut buf = vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
        mirror_rgb(&mut buf, 2, 2);
        assert_eq!(
            buf,
            vec![40, 50, 60, 10, 20, 30, 100, 110, 120, 70, 80, 90]
        );
        // Odd width: the middle column stays put.
        let mut odd = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        mirror_rgb(&mut odd, 3, 1);
        assert_eq!(odd, vec![7, 8, 9, 4, 5, 6, 1, 2, 3]);
        // No-op on empty frames must not panic.
        let mut none: Vec<u8> = Vec::new();
        mirror_rgb(&mut none, 0, 0);
    }
}
