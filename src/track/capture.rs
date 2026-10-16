//! nokhwa capture wrapper (MSMF on Windows, design doc section 5) — v1
//! parity: open `camera_index`, deliver RGB frames at the configured size,
//! and mirror the frame BEFORE inference when `mirror_display` is true
//! (controller.py:180 flips before cv2tColor, so the landmarker sees the
//! mirrored frame).

use crate::vision::image::mirror_rgb;
use nokhwa::{
    pixel_format::RgbFormat,
    utils::{CameraFormat, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType},
    Camera as NokhwaCamera,
};

pub struct Capture {
    camera: NokhwaCamera,
    mirror: bool,
    /// Time blocked in `camera.frame()` last call — the driver's delivery
    /// pace (measured ~30fps at VGA once the vendor-patched bindings request
    /// an RGB24 output type; see third_party VENDOR-NOTICE.md).
    pub last_wait_ms: f64,
    /// Decode + optional mirror last call (pure compute).
    pub last_decode_ms: f64,
}

/// Best supported mode for the configured resolution: exact resolution
/// beats any mismatch, then highest frame rate, then uncompressed over
/// MJPEG (the vendor-patched bindings request an RGB24 output type — see
/// third_party/nokhwa-bindings-windows/VENDOR-NOTICE.md — so the native
/// fourcc only feeds the color converter; NV12 converts with measurably
/// cheaper decode than MJPEG: 1.4ms vs 3.2ms at VGA, M3.5 ledger).
fn best_format(fmts: &[CameraFormat], w: u32, h: u32) -> Option<CameraFormat> {
    fmts.iter().copied().min_by_key(|f| {
        let res = f.resolution();
        let tier: u8 = if res.x() == w && res.y() == h { 0 } else { 1 };
        let rank: u8 = match f.format() {
            FrameFormat::NV12 | FrameFormat::YUYV => 3,
            FrameFormat::MJPEG => 2,
            _ => 0,
        };
        (tier, std::cmp::Reverse(f.frame_rate()), std::cmp::Reverse(rank))
    })
}

impl Capture {
    /// Opens the camera at the configured resolution, then enumerates the
    /// driver's real format list and switches to the best mode: exact
    /// resolution first, highest frame rate, uncompressed before MJPEG.
    ///
    /// Why the enumeration dance: `RequestedFormatType::Closest` alone
    /// resolved this driver's `640x480@1FPS` ("frame rate unspecified")
    /// entry, which delivers ~15fps — while the enumeration finds a proper
    /// 30fps mode that the closest-query can never find (M3.5 ledger).
    pub fn open(index: i64, width: i64, height: i64, mirror: bool) -> Result<Self, String> {
        let idx: u32 = index.try_into().map_err(|_| {
            format!("camera_index must be >= 0, got {index}")
        })?;
        let (w, h) = (width as u32, height as u32);
        // Open permissively first (any decodable format), then pick + switch.
        let req =
            RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);
        let mut camera = NokhwaCamera::new(CameraIndex::Index(idx), req)
            // v1's exact wording (controller.py:116-117).
            .map_err(|_| {
                format!(
                    "Could not open camera index {index}.\n\
                     Close other apps using the webcam (or set camera_index in config.json)."
                )
            })?;
        let picked = camera
            .compatible_camera_formats()
            .ok()
            .and_then(|fmts| best_format(&fmts, w, h));
        if let Some(fmt) = picked {
            let req = RequestedFormat::new::<RgbFormat>(RequestedFormatType::Exact(fmt));
            if let Err(e) = camera.set_camera_requset(req) {
                println!("[camera] could not switch to {fmt}: {e} (staying on current mode)");
            }
        }
        // The vendor-patched bindings fix the upstream "@1FPS" read-back
        // bug and end on an RGB24 output type over the picked native mode,
        // so this prints the real negotiated rate and RAWBGR (see
        // third_party/nokhwa-bindings-windows/VENDOR-NOTICE.md; the actual
        // delivery pace is the wait stat in the loop).
        println!("[camera] {}", camera.camera_format());
        Ok(Self {
            camera,
            mirror,
            last_wait_ms: 0.0,
            last_decode_ms: 0.0,
        })
    }

    /// One frame as `(rgb bytes, width, height)`, mirrored if configured.
    pub fn capture(&mut self) -> Result<(Vec<u8>, u32, u32), String> {
        let t0 = std::time::Instant::now();
        let frame = self
            .camera
            .frame()
            .map_err(|e| format!("camera frame failed: {e}"))?;
        let wait = t0.elapsed().as_secs_f64() * 1e3;
        let t1 = std::time::Instant::now();
        let rgb = frame
            .decode_image::<RgbFormat>()
            .map_err(|e| format!("frame decode failed: {e}"))?;
        let (w, h) = (rgb.width(), rgb.height());
        let mut buf = rgb.into_raw();
        if self.mirror {
            mirror_rgb(&mut buf, w, h);
        }
        self.last_wait_ms = wait;
        self.last_decode_ms = t1.elapsed().as_secs_f64() * 1e3;
        Ok((buf, w, h))
    }
}
