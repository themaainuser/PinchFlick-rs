# Vendored: nokhwa-bindings-windows 0.4.6

Source: crates.io `nokhwa-bindings-windows` 0.4.6 (part of the `nokhwa`
0.10.11 camera crate, https://github.com/l1npengtul/nokhwa).
License: Apache-2.0 (see `LICENSE-APACHE` shipped with the crate on
crates.io). Vendored because the upstream crate's Windows Media Foundation
reader under-delivers frames on the tested UVC webcam; the fix needs to
change reader construction and media-type negotiation, which upstream does
not expose as configuration. Wired in via `[patch.crates-io]` in the
workspace `Cargo.toml`.

## Patches (all in `src/lib.rs`, marked with `--- PinchFlick patch ---`)

1. **Reader attributes in `new()`** — upstream sets
   `MF_READWRITE_DISABLE_CONVERTERS = TRUE` and nothing else. We mirror
   OpenCV's `cap_msmf` `getDefaultSourceConfig()` instead:
   converters enabled, `MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS`,
   `MF_SOURCE_READER_ENABLE_ADVANCED_VIDEO_PROCESSING`.

2. **RGB24 output type in `set_format()`** — after the matched native
   media type is set as current (upstream behavior kept), we replay
   OpenCV `cap_msmf`'s `configureVideoOutput()`: set a constructed
   "clean copy" of the native type, then an RGB24 output type on the
   same stream. If the camera rejects the RGB24 type we log to stderr
   and keep reading the native type (upstream path).

3. **Explicit frame-rate set** in `set_format()` — request the matched
   rate explicitly on the native type before `SetCurrentMediaType`.

4. **`format_refreshed()` frame-rate read-back fix** — `MF_MT_FRAME_RATE`
   is packed `(numerator << 32) | denominator`; upstream truncated with
   `as u32`, which keeps only the denominator and displays every camera
   as `@1FPS`. We parse the real fraction.

## Why (measured, M3.5 of pinchflick-rs, 2026-10-16)

Tested webcam: USB2.0 HD UVC WebCam (640x480, NV12/MJPEG natives, all
advertising 30/1).

- Upstream config (native NV12 type set as current, converters disabled):
  sensor delivered a ~64ms sample-timestamp grid = ~15fps (capture wait
  55-59ms), dropping to a 100ms grid (~10fps) in low light. The current
  media type read back correctly as 30/1 NV12 — the pacing difference is
  driver-side, keyed off the reader's output configuration.
- OpenCV `cap_msmf` on the same camera, same native negotiation: 30.0fps
  sustained (warmup-discarded measurement), regardless of FOURCC/FPS
  request.
- With patches 1+2 (the proven configuration): 25.8fps measured, capture
  wait ~29ms, RGB24 arrives as `FrameFormat::RAWBGR` (passthrough decode
  ~0.7ms). Patch order of `set_format` retains upstream behavior as
  fallback, so other cameras keep working.
