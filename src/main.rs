// M1 — the ort/ONNX spike (design doc section 12 re-scope).
// Models: ONNX conversions of the official MediaPipe hand models
// (takoyakisoft/vtuberkit-vision-landmarkers-onnx, Apache-2.0, SHA256-verified).
// Question: do they run on onnxruntime via `ort`, and how fast per stage?
//
// M3.2: `--detect` mode runs the full decode path (anchors -> tensors ->
// detections -> NMS -> projection) on a frame and writes an annotated copy,
// for the human/AI visual gate.
mod spike;

use std::time::Instant;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use ort::environment::Environment;
use ort::session::Session;
use ort::value::Tensor;
use pinchflick_rs::vision::{anchors::generate_anchors, decode};
use pinchflick_rs::config;
use pinchflick_rs::gestures::engine::{GestureEngine, Landmark};
use pinchflick_rs::gestures::seek::{seek_key_sequence, Side};
use pinchflick_rs::sys::{hud, input, title::YouTubeDetector};
use pinchflick_rs::track::{capture::Capture, inference::Pipeline};
use eframe::egui;

fn main() {
    // v1 controller.py:333-348: config failures -> `Config error: ...` on
    // stderr + exit 2; anything else -> `Fatal error: ...` + exit 1 (was
    // Rust's default `Error: {:?}` + exit 1 for everything).
    if let Err(e) = run_cli() {
        let (line, code) = pinchflick_rs::cli::error_output(&*e);
        eprintln!("{line}");
        std::process::exit(code);
    }
}

/// The whole CLI flow as a `Result` so `main` can apply v1's error lines
/// and exit codes. (CLI wiring = declared spike like every flag here; the
/// rules live in `cli::error_output` / `cli::frame_path` / `cli::implies_live`,
/// unit-tested in `src/cli.rs`.)
fn run_cli() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let frame_path = pinchflick_rs::cli::frame_path(&args);
    let detect_mode = args.iter().any(|a| a == "--detect");
    let landmarks_mode = args.iter().any(|a| a == "--landmarks");
    let video_iters: Option<usize> = args.iter().position(|a| a == "--video").map(|i| {
        args.get(i + 1)
            .and_then(|s| s.parse().ok())
            .unwrap_or(3)
    });
    // --camera/--config imply live mode: v1's argparse had no dev harness,
    // so those flags always launched the live app (controller.py).
    let live_mode = pinchflick_rs::cli::implies_live(&args);
    let dry_run = args.iter().any(|a| a == "--dry-run");
    let selftest_mode = args.iter().any(|a| a == "--selftest");
    let config_path = args
        .iter()
        .position(|a| a == "--config")
        .and_then(|i| args.get(i + 1))
        .cloned();
    let camera_override = args
        .iter()
        .position(|a| a == "--camera")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<i64>().ok());

    // `--selftest` = v1's `python selftest.py`: 25 logic checks against
    // synthetic landmarks, no camera and no ONNX runtime — handled before
    // `ort::init()` so it runs on a machine with no models present. Exit 0 =
    // ALL GREEN, 1 = failures printed. (CLI flag wiring is spike-gated like
    // every other flag here; the checks themselves are the parity-tested
    // code in `selftest.rs`, the exact functions `cargo test` wraps.)
    if selftest_mode {
        let failures = pinchflick_rs::selftest::run_and_print();
        if failures > 0 {
            std::process::exit(1);
        }
        return Ok(());
    }

    let environment = ort::init().with_name("pinchflick-rs").build()?;

    if live_mode {
        return run_live(&environment, config_path, camera_override, dry_run);
    }

    let img = image::open(&frame_path)?.to_rgb8();
    let (fw, fh) = img.dimensions();
    let fbytes = img.as_raw();
    println!("frame: {fw}x{fh} ({frame_path})");

    if video_iters.is_some() {
        return run_video(&environment, &img, fw, fh, video_iters.unwrap_or(3));
    }
    if landmarks_mode {
        return run_landmarks(&environment, &img, fw, fh);
    }
    if detect_mode {
        return run_detect(&environment, &img, fw, fh);
    }

    println!("\n=== palm_detector.onnx ===");
    // onnxruntime reports the model's true input: 192x192 (corrected from the 128 guess)
    run_ort(&environment, "models/palm_detector.onnx", "detector", fbytes, fw, fh, (192, 192), false)?;

    println!("\n=== hand_landmark.onnx ===");
    // ort reports 224x224 for this conversion; no detection yet -> center crop
    run_ort(&environment, "models/hand_landmark.onnx", "landmark", fbytes, fw, fh, (224, 224), true)?;

    println!("\n=== hand_recrop.onnx (metadata probe) ===");
    let recrop = Session::builder(&environment)?.commit_from_file("models/hand_recrop.onnx")?;
    for i in recrop.inputs() {
        println!("  in  {i:?}");
    }
    for o in recrop.outputs() {
        println!("  out {o:?}");
    }

    Ok(())
}

/// M4.3 — egui preview HUD wiring (declared spike: the eframe/camera/
/// injection plumbing is compiler- and human-gated, not unit-tested; the
/// pure pieces it composes — `HudState`, `Scaler`, string builders,
/// skeleton projection — live in `sys/hud.rs` with 27 passing tests).
///
/// The loop body is M3.6's `'live` iteration moved into [`Live::step`]
/// (console prints kept verbatim: dry-run evidence, `[stats]`, `[seek]`).
/// eframe drives it: `logic()` = keys + one step + repaint, `ui()` = paint.
struct Live {
    cfg: config::Config,
    engine: GestureEngine,
    ytd: YouTubeDetector,
    capture: Capture,
    pipeline: Pipeline,
    hud: hud::HudState,
    t0: Instant,
    last_ts: i64,
    // v1's stats dict.
    frames: u64,
    hand_frames: u64,
    seeks: u64,
    scroll_events: u64,
    wheel: i64,
    // Rolling 2 s window for the stage report.
    win_start: Instant,
    win_frames: u64,
    win_wait_ms: f64,
    win_decode_ms: f64,
    win_hand: u64,
    win_detects: u64,
    win_detect_ms: f64,
    win_landmark_ms: f64,
    win_landmark_n: u64,
    // Current frame's draw data, copied out under the lock for `ui()`.
    status: String,
    tap_status: String,
    pinched: bool,
    ratio: Option<f64>,
    on_youtube: bool,
    scroll_clicks: i32,
    landmarks: Vec<Landmark>,
    /// Latest captured frame, already mirrored inside `capture()` (same
    /// bytes v1 shows via cv2.imshow) — uploaded to the HUD texture each
    /// frame so the preview shows the camera feed under the overlay.
    frame_rgb: Vec<u8>,
    frame_w: f32,
    frame_h: f32,
}

/// What one `step()` says about the loop's future.
enum StepOutcome {
    Continue,
    Stop(String),
}

impl Live {
    /// M3.6 setup verbatim (config, engine, detector, capture, pipeline).
    fn new(
        environment: &Environment,
        config_path: Option<String>,
        camera_override: Option<i64>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut cfg = config::load_config(config_path.as_deref())?;
        if let Some(idx) = camera_override {
            cfg.camera_index = idx;
        }

        let engine = GestureEngine::new(&cfg, cfg.mirror_display);
        let ytd = YouTubeDetector::new(
            cfg.youtube_title_markers.clone(),
            cfg.youtube_check_interval_ms,
        );
        let capture = Capture::open(
            cfg.camera_index,
            cfg.frame_width,
            cfg.frame_height,
            cfg.mirror_display,
        )
        .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
        let pipeline = Pipeline::new(
            environment,
            cfg.min_detection_confidence,
            cfg.min_tracking_confidence,
        )?;

        Ok(Self {
            hud: hud::HudState::new(cfg.show_debug_window),
            frame_w: cfg.frame_width as f32,
            frame_h: cfg.frame_height as f32,
            t0: Instant::now(),
            last_ts: -1,
            frames: 0,
            hand_frames: 0,
            seeks: 0,
            scroll_events: 0,
            wheel: 0,
            win_start: Instant::now(),
            win_frames: 0,
            win_wait_ms: 0.0,
            win_decode_ms: 0.0,
            win_hand: 0,
            win_detects: 0,
            win_detect_ms: 0.0,
            win_landmark_ms: 0.0,
            win_landmark_n: 0,
            status: String::new(),
            tap_status: String::new(),
            pinched: false,
            ratio: None,
            on_youtube: false,
            scroll_clicks: 0,
            landmarks: Vec::new(),
            frame_rgb: Vec::new(),
            cfg,
            engine,
            ytd,
            capture,
            pipeline,
        })
    }

    /// One camera -> pipeline -> engine -> act iteration (M3.6 loop body).
    /// Console output is identical to M3.6; the HUD visuals come from the
    /// state written here, painted by [`HudApp::ui`].
    fn step(&mut self, dry_run: bool) -> StepOutcome {
        self.hud.update_fps(self.t0.elapsed().as_secs_f64());

        let (rgb, w, h) = match self.capture.capture() {
            Ok(v) => v,
            Err(e) => {
                println!("[capture error] {e}");
                return StepOutcome::Stop("camera stopped delivering frames".to_string());
            }
        };

        let mut ts = self.t0.elapsed().as_millis() as i64;
        if ts <= self.last_ts {
            ts = self.last_ts + 1; // VIDEO mode demands strictly increasing stamps
        }
        self.last_ts = ts;

        let out = match self.pipeline.process(&rgb, w, h) {
            Ok(o) => o,
            Err(e) => {
                println!("[pipeline error] {e}");
                return StepOutcome::Stop("inference error".to_string());
            }
        };
        self.frame_rgb = rgb;
        self.frames += 1;
        self.win_frames += 1;
        self.win_wait_ms += self.capture.last_wait_ms;
        self.win_decode_ms += self.capture.last_decode_ms;
        if out.landmarks.is_some() {
            self.hand_frames += 1;
            self.win_hand += 1;
        }
        if !out.detect_ms.is_zero() {
            self.win_detects += 1;
            self.win_detect_ms += out.detect_ms.as_secs_f64() * 1e3;
        }
        if !out.landmark_ms.is_zero() {
            self.win_landmark_n += 1;
            self.win_landmark_ms += out.landmark_ms.as_secs_f64() * 1e3;
        }

        let landmarks = out.landmarks.as_ref().map(|lms| {
            lms.iter()
                .map(|l| Landmark::new(l[0] as f64, l[1] as f64))
                .collect::<Vec<_>>()
        });
        let (on_youtube, _title) = self.ytd.check(ts, false);
        let res = self.engine.frame(landmarks.as_deref(), ts, on_youtube);

        // ---- HUD state + draw data for this frame -------------------------
        self.hud
            .on_engine_result(&res, ts, self.cfg.seek_seconds);
        self.status = hud::status_line(on_youtube, &res);
        self.tap_status = res.tap_status.clone();
        self.on_youtube = on_youtube;
        self.ratio = res.ratio;
        self.pinched = res.pinched;
        self.scroll_clicks = res.scroll_clicks;
        self.frame_w = w as f32;
        self.frame_h = h as f32;
        self.landmarks = landmarks.unwrap_or_default();

        // ---- act: scroll (anywhere) ---------------------------------------
        if res.scroll_clicks != 0 {
            if dry_run {
                self.scroll_events += 1;
                self.wheel += res.scroll_clicks as i64;
                println!("[scroll] {:+} clicks (dry-run)", res.scroll_clicks);
            } else {
                // v1: failSafeCheck runs before the wheel event; on failure it
                // stops without counting the event.
                if input::failsafe_triggered(self.cfg.pyautogui_failsafe) {
                    return StepOutcome::Stop(
                        "pyautogui failsafe (mouse hit a screen corner)".to_string(),
                    );
                }
                input::scroll(res.scroll_clicks);
                self.scroll_events += 1;
                self.wheel += res.scroll_clicks as i64;
                // real mode is HUD-silent — the HUD now carries the visuals.
            }
        }

        // ---- act: seek (engine already YouTube-gates it) -------------------
        if let Some(dir) = res.seek {
            match seek_key_sequence(self.cfg.seek_seconds, dir.as_str(), &self.cfg.seek_keys) {
                Ok(seq) => {
                    let arrow = if dir.as_str() == "backward" { "-" } else { "+" };
                    if dry_run {
                        self.seeks += 1;
                        println!(
                            "[seek] {arrow}{}s ({}) (dry-run)",
                            self.cfg.seek_seconds,
                            seq.join(" + ")
                        );
                    } else {
                        // v1: per key, 50ms gap, failSafeCheck before each
                        // press; mid-sequence failure stops without stats or
                        // the [seek] print (keys already sent stay sent).
                        let mut aborted = false;
                        for (i, key) in seq.iter().enumerate() {
                            if i > 0 {
                                std::thread::sleep(std::time::Duration::from_millis(50));
                            }
                            if input::failsafe_triggered(self.cfg.pyautogui_failsafe) {
                                aborted = true;
                                break;
                            }
                            input::press(key);
                        }
                        if aborted {
                            return StepOutcome::Stop(
                                "pyautogui failsafe (mouse hit a screen corner)".to_string(),
                            );
                        }
                        self.seeks += 1;
                        println!(
                            "[seek] {arrow}{}s ({})",
                            self.cfg.seek_seconds,
                            seq.join(" + ")
                        );
                    }
                }
                Err(e) => println!("[seek error] {e}"),
            }
        }
        if let Some(blocked) = &res.seek_blocked {
            println!("[seek blocked] {blocked}");
        }
        if let Some(side) = res.tap && res.seek.is_none() {
            println!(
                "[tap] {} side -- double-pinch again within {}ms",
                side.as_str(),
                self.cfg.double_tap_window_ms
            );
        }

        if self.win_start.elapsed().as_secs_f64() >= 2.0 {
            let secs = self.win_start.elapsed().as_secs_f64();
            let fps = self.win_frames as f64 / secs;
            let det_avg = if self.win_detects > 0 {
                self.win_detect_ms / self.win_detects as f64
            } else {
                0.0
            };
            let lmk_avg = if self.win_landmark_n > 0 {
                self.win_landmark_ms / self.win_landmark_n as f64
            } else {
                0.0
            };
            let wait_avg = if self.win_frames > 0 {
                self.win_wait_ms / self.win_frames as f64
            } else {
                0.0
            };
            let decode_avg = if self.win_frames > 0 {
                self.win_decode_ms / self.win_frames as f64
            } else {
                0.0
            };
            println!(
                "[stats] {fps:.1} fps | hand {}% | wait {wait_avg:.1}ms | decode {decode_avg:.1}ms | detect {det_avg:.1}ms ({} runs) | landmark {lmk_avg:.1}ms | total {}",
                if self.win_frames > 0 {
                    self.win_hand * 100 / self.win_frames
                } else {
                    0
                },
                self.win_detects,
                self.frames
            );
            self.win_start = Instant::now();
            self.win_frames = 0;
            self.win_wait_ms = 0.0;
            self.win_decode_ms = 0.0;
            self.win_hand = 0;
            self.win_detects = 0;
            self.win_detect_ms = 0.0;
            self.win_landmark_ms = 0.0;
            self.win_landmark_n = 0;
        }

        StepOutcome::Continue
    }

    /// v1's per-frame window title (controller.py:281-282).
    fn short_title(&self) -> String {
        hud::short_title(
            self.on_youtube,
            self.ratio,
            self.scroll_clicks,
            self.seeks,
        )
    }
}

/// Everything `ui()` paints for the current frame, snapshotted by `logic()`
/// while the [`Live`] lock is held (v1 draws straight from the same locals).
#[derive(Default)]
struct PaintFrame {
    status: String,
    tap_status: String,
    block: Option<String>,
    debug: Option<String>,
    flash: Option<hud::Flash>,
    points: Vec<(f32, f32)>,
    pinched: bool,
    frame_w: f32,
    frame_h: f32,
}

/// The eframe app: `logic()` = v1's waitKey branch + one loop iteration,
/// `ui()` = v1's draw section (controller.py:255-279) in egui.
struct HudApp {
    live: Arc<Mutex<Live>>,
    quit: Arc<AtomicBool>,
    out: Arc<Mutex<Option<String>>>,
    dry_run: bool,
    paint: PaintFrame,
    /// The preview texture (v1's cv2.imshow content): created on the first
    /// frame, updated per frame with the latest captured RGB buffer.
    tex: Option<egui::TextureHandle>,
    stop: Option<String>,
    finished: bool,
    last_title: String,
}

impl eframe::App for HudApp {
    fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // v1's waitKey branch (controller.py:289-293): q/ESC quit, d toggles.
        if ctx.input(|i| i.key_pressed(egui::Key::Q) || i.key_pressed(egui::Key::Escape)) {
            self.quit.store(true, Ordering::SeqCst);
        }
        if ctx.input(|i| i.key_pressed(egui::Key::D)) {
            self.live.lock().unwrap().hud.toggle_debug();
        }
        let window_closed = ctx.input(|i| i.viewport().close_requested());
        if window_closed {
            self.quit.store(true, Ordering::SeqCst);
        }

        if self.stop.is_none() && !self.quit.load(Ordering::SeqCst) {
            let mut live = self.live.lock().unwrap();
            if let StepOutcome::Stop(reason) = live.step(self.dry_run) {
                self.stop = Some(reason);
            }
            // Snapshot what ui() paints, all under this one lock.
            let ts = live.t0.elapsed().as_millis() as i64;
            let p = &mut self.paint;
            p.status.clone_from(&live.status);
            p.tap_status.clone_from(&live.tap_status);
            p.block = live.hud.block_text_at(ts).map(str::to_string);
            p.debug = live
                .hud
                .show_debug
                .then(|| hud::debug_line(live.hud.fps, live.scroll_clicks, live.seeks, &live.ytd.title));
            p.flash = live.hud.flash_at(ts).cloned();
            p.pinched = live.pinched;
            p.frame_w = live.frame_w;
            p.frame_h = live.frame_h;
            p.points =
                hud::skeleton_points(&live.landmarks, live.frame_w as u32, live.frame_h as u32);
            let title = live.short_title();
            if title != self.last_title {
                ctx.send_viewport_cmd(egui::ViewportCommand::Title(title.clone()));
                self.last_title = title;
            }
            // Preview texture: latest mirrored frame (v1's cv2.imshow feed),
            // stretched by ui() onto the window rect like WINDOW_NORMAL.
            let tw = live.frame_w as usize;
            let th = live.frame_h as usize;
            if tw > 0 && live.frame_rgb.len() == tw * th * 3 {
                let img = egui::ColorImage::from_rgb([tw, th], &live.frame_rgb);
                match &mut self.tex {
                    Some(t) => t.set(img, egui::TextureOptions::LINEAR),
                    None => {
                        self.tex =
                            Some(ctx.load_texture("pinchflick-frame", img, egui::TextureOptions::LINEAR));
                    }
                }
            }
        }

        if self.stop.is_none() && self.quit.load(Ordering::SeqCst) {
            // v1 wording for q/ESC (controller.py:168: stop_reason =
            // "quit key"); the ✕ button has no v1 analog, so it says so.
            self.stop = Some(if window_closed {
                "quit (window closed)"
            } else {
                "quit key"
            }
            .to_string());
        }

        if let Some(reason) = &self.stop {
            if !self.finished {
                {
                    let live = self.live.lock().unwrap();
                    park_summary(&live, reason, &self.out);
                }
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                self.finished = true;
            }
        }

        // Camera cadence drives the loop (each step blocks on the capture).
        ctx.request_repaint();
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let rect = ui.max_rect();
        let p = &self.paint;
        // v1 draws on the 640x480 frame and lets cv2.imshow stretch it to the
        // window exactly (aspect ignored) — axes scale independently.
        let sc = hud::Scaler::new(p.frame_w, p.frame_h, rect.width(), rect.height());
        let painter = ui.painter();
        let at = |pt: (f32, f32)| -> egui::Pos2 {
            let (x, y) = sc.pt(pt);
            egui::pos2(rect.left() + x, rect.top() + y)
        };

        // Preview: the camera feed (v1's cv2.imshow content), stretched to
        // the window exactly like WINDOW_NORMAL; black until the first
        // frame's texture exists.
        match &self.tex {
            Some(tex) => {
                painter.image(
                    tex.id(),
                    rect,
                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                    egui::Color32::WHITE,
                );
            }
            None => {
                painter.rect_filled(rect, 0.0, egui::Color32::BLACK);
            }
        }

        // Center divider (v1 controller.py:256).
        let mid = at((p.frame_w / 2.0, 0.0));
        painter.line_segment(
            [
                egui::pos2(mid.x, rect.top()),
                egui::pos2(mid.x, rect.bottom()),
            ],
            egui::Stroke::new((sc.sx + sc.sy) / 2.0, rgb(hud::DIVIDER_RGB)),
        );

        // Hand skeleton (v1 draw_hand, controller.py:79-86): lines thickness
        // 2, dots radius 3, red while pinched.
        let hand = rgb(if p.pinched {
            hud::HAND_PINCHED_RGB
        } else {
            hud::HAND_OPEN_RGB
        });
        for &(a, b) in hud::HAND_CONNECTIONS.iter() {
            if let (Some(pa), Some(pb)) = (p.points.get(a), p.points.get(b)) {
                painter.line_segment([at(*pa), at(*pb)], egui::Stroke::new(sc.sx + sc.sy, hand));
            }
        }
        for pt in &p.points {
            painter.circle_filled(at(*pt), 3.0 * (sc.sx + sc.sy) / 2.0, hand);
        }

        // Status lines (v1 controller.py:259-269): x=10, baselines y=26,
        // 52, 78, 104; frame-space font 14.
        let fs = 14.0 * sc.sx;
        let line = |y: f32| egui::pos2(rect.left() + 10.0 * sc.sx, rect.top() + y * sc.sy);
        halo_text(
            &painter,
            &p.status,
            line(26.0),
            egui::Align2::LEFT_BOTTOM,
            fs,
            rgb(hud::STATUS_TEXT_RGB),
            sc.sx,
        );
        halo_text(
            &painter,
            &p.tap_status,
            line(52.0),
            egui::Align2::LEFT_BOTTOM,
            fs,
            rgb(hud::TAP_STATUS_TEXT_RGB),
            sc.sx,
        );
        if let Some(block) = &p.block {
            halo_text(
                &painter,
                block,
                line(78.0),
                egui::Align2::LEFT_BOTTOM,
                fs,
                rgb(hud::BLOCK_TEXT_RGB),
                sc.sx,
            );
        }
        if let Some(debug) = &p.debug {
            halo_text(
                &painter,
                debug,
                line(104.0),
                egui::Align2::LEFT_BOTTOM,
                fs,
                rgb(hud::DEBUG_TEXT_RGB),
                sc.sx,
            );
        }

        // Side flash (v1 controller.py:270-279): frame-space font 26 at mid
        // height; left flush at x=14, right flush at the right edge - 14.
        if let Some(f) = &p.flash {
            let y = rect.top() + (p.frame_h / 2.0) * sc.sy;
            let (pos, anchor) = match f.side {
                Side::Left => (
                    egui::pos2(rect.left() + 14.0 * sc.sx, y),
                    egui::Align2::LEFT_BOTTOM,
                ),
                Side::Right => (
                    egui::pos2(rect.right() - 14.0 * sc.sx, y),
                    egui::Align2::RIGHT_BOTTOM,
                ),
            };
            halo_text(
                &painter,
                &f.text,
                pos,
                anchor,
                26.0 * sc.sx,
                rgb(f.color),
                1.5 * sc.sx,
            );
        }
    }
}

/// v1's `_put` (controller.py:72-76): a black outline pass behind the
/// colored pass. egui 0.37 has no text-stroke API (ruled at M4.3), so the
/// outline is eight offset black copies — same readability, same two-pass
/// visual. Offsets are in window points (frame px * scale).
fn halo_text(
    painter: &egui::Painter,
    text: &str,
    pos: egui::Pos2,
    anchor: egui::Align2,
    size: f32,
    color: egui::Color32,
    outline: f32,
) {
    if text.is_empty() {
        return;
    }
    let font = egui::FontId::proportional(size);
    for &(dx, dy) in &[
        (0.0, -outline),
        (0.0, outline),
        (-outline, 0.0),
        (outline, 0.0),
        (-outline, -outline),
        (outline, -outline),
        (-outline, outline),
        (outline, outline),
    ] {
        painter.text(
            pos + egui::vec2(dx, dy),
            anchor,
            text,
            font.clone(),
            egui::Color32::BLACK,
        );
    }
    painter.text(pos, anchor, text, font, color);
}

/// hud::Rgb -> egui color.
fn rgb(c: hud::Rgb) -> egui::Color32 {
    egui::Color32::from_rgb(c.0, c.1, c.2)
}

/// The old console loop's exit block (M3.6 / v1 controller.py:303-310),
/// parked as text: `run_native` consumes the app, so the summary is read
/// back out after the window closes.
fn park_summary(live: &Live, stop_reason: &str, out: &Arc<Mutex<Option<String>>>) {
    let elapsed = live.t0.elapsed().as_secs_f64().max(1e-6);
    let text = format!(
        "\nStopped: {stop_reason}\nSummary: {} frames in {elapsed:.1}s ({:.1} fps), hand visible {}/{}, seeks fired {}, scroll events {} ({:+} wheel clicks)\nLast focused window: {} -> YouTube: {}\n",
        live.frames,
        live.frames as f64 / elapsed,
        live.hand_frames,
        live.frames,
        live.seeks,
        live.scroll_events,
        live.wheel,
        if live.ytd.title.is_empty() {
            "(none)"
        } else {
            live.ytd.title.as_str()
        },
        if live.ytd.is_youtube { "yes" } else { "no" },
    );
    *out.lock().unwrap() = Some(text);
}

/// M3.6 live entry point — same signature, now with the always-on-top
/// preview window (v1's cv2.imshow replacement). The console banner,
/// stats and summary are unchanged; `d`/`q`/ESC live in the window
/// (v1 controller.py:156), and the stdin `q` thread still works.
fn run_live(
    environment: &Environment,
    config_path: Option<String>,
    camera_override: Option<i64>,
    dry_run: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let live = Live::new(environment, config_path, camera_override)?;
    let always_on_top = live.cfg.always_on_top;
    let (frame_w, frame_h) = (live.frame_w, live.frame_h);
    let seek_seconds = live.cfg.seek_seconds;
    let seek_requires_youtube = live.cfg.seek_requires_youtube;
    let pyautogui_failsafe = live.cfg.pyautogui_failsafe;

    println!();
    if dry_run {
        println!("PinchFlick running (dry-run: actions printed, not injected).");
    } else {
        println!("PinchFlick running.");
        if pyautogui_failsafe {
            println!("  failsafe: mouse parked in a screen corner stops PinchFlick");
        }
    }
    println!("  scroll : 1 finger = up, 2 fingers = down (works everywhere)");
    println!(
        "  seek   : double-pinch left/right = -/+{}s ({})",
        seek_seconds,
        if seek_requires_youtube {
            "YouTube tab must be focused"
        } else {
            "anywhere"
        }
    );
    println!("  keys   : q / ESC = quit, d = toggle debug overlay");
    println!("  note   : click your video window first -- seeks only fire while it's focused");
    println!();

    let quit = Arc::new(AtomicBool::new(false));
    {
        let quit = Arc::clone(&quit);
        std::thread::spawn(move || {
            use std::io::BufRead as _;
            let stdin = std::io::stdin();
            for line in stdin.lock().lines() {
                match line {
                    Ok(l) if l.trim() == "q" => {
                        quit.store(true, Ordering::SeqCst);
                        return;
                    }
                    Ok(_) => {}
                    Err(_) => return,
                }
            }
        });
    }

    // v1 window placement: top-right, 480x360 (config-driven always-on-top).
    // egui windows are positioned in POINTS: convert the (physical) metrics by
    // the system DPI so v1's `sw - 500` lands 20px inside the edge instead of
    // clipping off-screen right (observed at M4.4b before this conversion).
    let (screen_px, screen_py) = input::screen_size();
    let dpi_scale = input::system_dpi_scale();
    let screen_w = screen_px as f32 / dpi_scale;
    let screen_h = screen_py as f32 / dpi_scale;
    let (wx, wy) = hud::window_position(screen_w);
    println!(
        "[display] screen {screen_px}x{screen_py}px @ {dpi_scale:.2}x \
         -> {screen_w:.0}x{screen_h:.0}pt, window at ({wx}, {wy})"
    );
    let mut viewport = egui::ViewportBuilder::default()
        .with_title("PinchFlick (q=quit, d=debug)")
        .with_inner_size([480.0_f32, 360.0_f32])
        .with_position([wx, wy]);
    if always_on_top {
        viewport = viewport.with_always_on_top();
    }
    let options = eframe::NativeOptions {
        viewport,
        ..Default::default()
    };

    let out: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let app = HudApp {
        live: Arc::new(Mutex::new(live)),
        quit: Arc::clone(&quit),
        out: Arc::clone(&out),
        dry_run,
        paint: PaintFrame {
            frame_w,
            frame_h,
            ..Default::default()
        },
        tex: None,
        stop: None,
        finished: false,
        last_title: String::new(),
    };

    eframe::run_native(
        "PinchFlick (q=quit, d=debug)",
        options,
        Box::new(move |_cc| Ok(Box::new(app))),
    )
    .map_err(|e| -> Box<dyn std::error::Error> { format!("eframe: {e}").into() })?;

    if let Some(text) = out.lock().unwrap().take() {
        print!("{text}");
    }
    Ok(())
}

/// M3.2 visual gate: full detector decode on a static frame, annotated output.
fn run_detect(
    environment: &Environment,
    img: &image::RgbImage,
    fw: u32,
    fh: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    let t_load = Instant::now();
    let mut session = Session::builder(environment)?.commit_from_file("models/palm_detector.onnx")?;
    println!("detector: loaded in {:.1?}", t_load.elapsed());
    let anchors = generate_anchors();

    // v1 ground truth (v1_ground_truth.py): hand at image x 0.433..0.515,
    // y 0.838..0.981 -> palm detection box a bit larger. We test each
    // preprocessing combo for a detection whose bbox contains the hand center.
    let hand_center_img = (0.47f32, 0.90f32); // (x, y) in image-normalized coords

    // Preprocessing variants: how the raw 192x192 image gets fed to the model.
    //   layout: letterbox (MediaPipe keep_aspect_ratio+zero border) vs stretch
    //   norm:   m11 = v/127.5-1 | unit = v/255 | raw = v (norm baked in graph)
    let lb_raw = letterbox_u8(img.as_raw(), fw, fh, 192);
    let st_raw = spike::resize_nearest(img.as_raw(), fw, fh, 192, 192);
    let layouts: [(&str, Vec<u8>, bool); 2] =
        [("letterbox", lb_raw, true), ("stretch", st_raw, false)];
    let norms: [(&str, fn(f32) -> f32); 3] = [
        ("m11", |v: f32| v / 127.5 - 1.0),
        ("unit", |v: f32| v / 255.0),
        ("raw", |v: f32| v),
    ];

    let mut best: Option<(String, Vec<decode::PalmDetection>)> = None;
    for (lname, raw, is_lb) in layouts {
        for (nname, f) in norms {
            let input: Vec<f32> = raw.iter().map(|&v| f(v as f32)).collect();
            let tensor = Tensor::from_array(([1i64, 192i64, 192i64, 3i64], input))?;
            let outputs = session.run(ort::inputs![tensor])?;
            let (boxes, scores) = {
                let b = outputs["Identity"].try_extract_tensor::<f32>()?;
                let s = outputs["Identity_1"].try_extract_tensor::<f32>()?;
                (b.1.to_vec(), s.1.to_vec())
            };
            drop(outputs);

            // Score health: a trained detector on any input should keep most
            // logits deeply negative (background), few near/above zero.
            let n_over = scores
                .iter()
                .filter(|&&l| 1.0 / (1.0 + (-l.clamp(-100.0, 100.0)).exp()) > 0.6)
                .count();
            let lmax = scores.iter().copied().fold(f32::MIN, f32::max);
            let lmin = scores.iter().copied().fold(f32::MAX, f32::min);

            let mut dets = decode::decode_detections(&boxes, &scores, &anchors, 0.6);
            let n_pre = dets.len();
            dets = decode::weighted_nms(&dets, 0.3);

            // Hand center in TENSOR space for this layout (decode works in
            // tensor coords; projection comes later).
            let (cx_t, cy_t) = if is_lb {
                (hand_center_img.0, 0.125 + hand_center_img.1 * 0.75)
            } else {
                hand_center_img
            };
            let hits: Vec<usize> = dets
                .iter()
                .enumerate()
                .filter(|(_, d)| {
                    let b = &d.bbox;
                    cx_t >= b[1] && cx_t <= b[3] && cy_t >= b[0] && cy_t <= b[2]
                })
                .map(|(i, _)| i)
                .collect();
            // Plausible palm box: small (well under a quarter of the tensor);
            // garbage regressions hit the center only via giant boxes.
            let plausible: Vec<usize> = hits
                .iter()
                .copied()
                .filter(|&i| {
                    let b = &dets[i].bbox;
                    (b[2] - b[0]) * (b[3] - b[1]) < 0.25
                })
                .collect();

            println!(
                "{lname:>9} + {nname:<4}: logits [{lmin:8.1}, {lmax:7.1}] over.6={n_over:3} | det pre/post NMS {n_pre:3}/{} | hand-center hits: {}{}",
                dets.len(),
                hits.len(),
                if hits.is_empty() { "" } else { "  <<< HAND FOUND" }
            );
            for &i in hits.iter().take(3) {
                let d = &dets[i];
                println!(
                    "    hit det[{i}]: score {:.3} tensor bbox [{:.3}, {:.3}, {:.3}, {:.3}]",
                    d.score, d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3]
                );
            }
            if !plausible.is_empty() && best.is_none() {
                best = Some((format!("{lname}+{nname}"), dets));
            }
        }
    }

    let Some((combo, mut dets)) = best else {
        println!("\nNO combo found the hand -> preprocessing/model mismatch unresolved");
        return Ok(());
    };
    println!("\nusing combo: {combo}");

    decode::project_detections_to_image(&mut dets, fw, fh, 192);
    for (i, d) in dets.iter().enumerate().take(10) {
        println!(
            "  det[{i}]: score {:.3} bbox [{:.3}, {:.3}, {:.3}, {:.3}] (ymin xmin ymax xmax, image-normalized)",
            d.score, d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3]
        );
    }

    // Annotate: red box for the top detection, green dots for its keypoints.
    let mut out = img.clone();
    if let Some(top) = dets.first() {
        let (ymin, xmin, ymax, xmax) = (
            (top.bbox[0] * fh as f32) as i32,
            (top.bbox[1] * fw as f32) as i32,
            (top.bbox[2] * fh as f32) as i32,
            (top.bbox[3] * fw as f32) as i32,
        );
        let red = image::Rgb([255u8, 0, 0]);
        for x in xmin..=xmax {
            for w in 0..2 {
                set(&mut out, x, ymin + w, red);
                set(&mut out, x, ymax - w, red);
            }
        }
        for y in ymin..=ymax {
            for w in 0..2 {
                set(&mut out, xmin + w, y, red);
                set(&mut out, xmax - w, y, red);
            }
        }
        let green = image::Rgb([0u8, 255, 0]);
        for kp in &top.keypoints {
            let (kx, ky) = ((kp[0] * fw as f32) as i32, (kp[1] * fh as f32) as i32);
            for dy in -2..=2 {
                for dx in -2..=2 {
                    set(&mut out, kx + dx, ky + dy, green);
                }
            }
        }
    }
    out.save("frame_detected.png")?;
    println!("wrote frame_detected.png");
    Ok(())
}

/// M3.3 visual gate: detect -> rect (rotation quirk + transform) -> rotated
/// crop -> landmark model -> decode -> project -> annotated frame + JSON.
/// Value range (0,1) for the landmark crop is the same empirical question
/// the detector sweep answered; printed presence flag validates it.
fn run_landmarks(
    environment: &Environment,
    img: &image::RgbImage,
    fw: u32,
    fh: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    use pinchflick_rs::vision::{landmark, rect, warp};

    let img_bytes = img.as_raw().clone();
    // --- Stage 1: palm detection (M3.2 path, letterbox + v/255).
    let mut det_session = Session::builder(environment)?.commit_from_file("models/palm_detector.onnx")?;
    let input = pinchflick_rs::vision::image::letterbox_normalize(&img_bytes, fw, fh, 192);
    let tensor = Tensor::from_array(([1i64, 192i64, 192i64, 3i64], input))?;
    let outputs = det_session.run(ort::inputs![tensor])?;
    let (boxes, scores) = {
        let b = outputs["Identity"].try_extract_tensor::<f32>()?;
        let s = outputs["Identity_1"].try_extract_tensor::<f32>()?;
        (b.1.to_vec(), s.1.to_vec())
    };
    drop(outputs);
    let anchors = generate_anchors();
    let mut dets = decode::decode_detections(&boxes, &scores, &anchors, 0.6);
    dets = decode::weighted_nms(&dets, 0.3);
    decode::project_detections_to_image(&mut dets, fw, fh, 192);
    let Some(det) = dets.first() else {
        println!("no detection");
        return Ok(());
    };
    println!(
        "det: score {:.3} bbox [{:.3}, {:.3}, {:.3}, {:.3}]",
        det.score, det.bbox[0], det.bbox[1], det.bbox[2], det.bbox[3]
    );
    for (i, kp) in det.keypoints.iter().enumerate() {
        println!("  kp[{i}] ({:.3}, {:.3})", kp[0], kp[1]);
    }

    // --- Stage 2: rect (v1.1.1 HandDetectorGraph, verbatim incl. quirk).
    let image_size = (fw as f32, fh as f32);
    let rotation = rect::detection_rotation(det.keypoints[0], det.keypoints[2], image_size);
    println!(
        "rect rotation: {rotation:.5} rad ({:.2} deg)",
        rotation.to_degrees()
    );
    let (x_center, y_center, width, height) = rect::norm_rect_from_keypoints(&det.keypoints);
    let raw_rect = rect::NormRect {
        x_center,
        y_center,
        width,
        height,
        rotation,
    };
    let hand_rect = rect::transform_rect(
        &raw_rect,
        image_size,
        rect::DETECT_SCALE,
        rect::DETECT_SCALE,
        rect::DETECT_SHIFT_X,
        rect::DETECT_SHIFT_Y,
    );
    println!(
        "hand_rect: center ({:.3}, {:.3}) size ({:.3}, {:.3}) rot {:.3}",
        hand_rect.x_center, hand_rect.y_center, hand_rect.width, hand_rect.height, hand_rect.rotation
    );
    // Square-in-pixels check (square_long invariant; projection is only
    // specified for square ROIs).
    println!(
        "  px: {:.1} x {:.1} (square_long invariant)",
        hand_rect.width * fw as f32,
        hand_rect.height * fh as f32
    );

    // --- Stage 3: rotated crop (keep_aspect_ratio false = proto default).
    let mut roi = warp::get_roi(fw, fh, &hand_rect);
    let padding = warp::pad_roi(224, 224, false, &mut roi);
    println!("crop roi: {roi:?} padding {padding:?}");
    let crop = warp::warp_crop(&img_bytes, fw, fh, &roi, 224, (0.0, 1.0));
    // Save the crop for visual inspection.
    let mut crop_img = image::RgbImage::new(224, 224);
    for i in 0..224 * 224 {
        let v = |c: usize| (crop[i as usize * 3 + c] * 255.0).round().clamp(0.0, 255.0) as u8;
        crop_img.put_pixel((i % 224) as u32, (i / 224) as u32, image::Rgb([v(0), v(1), v(2)]));
    }
    crop_img.save("crop.png")?;
    println!("wrote crop.png");

    // --- Stage 4: landmark model.
    let mut lm_session = Session::builder(environment)?.commit_from_file("models/hand_landmark.onnx")?;
    let crop_tensor = Tensor::from_array(([1i64, 224i64, 224i64, 3i64], crop))?;
    let lm_outputs = lm_session.run(ort::inputs![crop_tensor])?;
    let mut collected = Vec::new();
    for (oname, val) in lm_outputs.iter() {
        let (shape, data) = val.try_extract_tensor::<f32>()?;
        collected.push((oname.to_string(), shape.to_vec(), data.to_vec()));
    }
    drop(lm_outputs);
    // Identify by shape + value signature: [1,63] pixel-scale = landmarks;
    // [1,63] small-scale = world; [1,1] = presence.
    let mut landmark_raw: Option<Vec<f32>> = None;
    let mut presence = 0.0f32;
    for (oname, shape, data) in &collected {
        let mean_abs = data.iter().map(|v| v.abs()).sum::<f32>() / data.len() as f32;
        let head: Vec<String> = data.iter().take(3).map(|v| format!("{v:.4}")).collect();
        println!(
            "lm out '{oname}' shape {shape:?} mean|v| {mean_abs:.4} head [{}]",
            head.join(", ")
        );
        if data.len() == 63 && mean_abs > 5.0 {
            landmark_raw = Some(data.clone());
        }
        if data.len() == 1 {
            presence = 1.0 / (1.0 + (-data[0].clamp(-100.0, 100.0)).exp());
        }
    }
    println!("presence (sigmoid): {presence:.4}  (v1 min_detection_confidence gate = 0.5)");
    let Some(raw) = landmark_raw else {
        println!("no [1,63] landmark tensor found");
        return Ok(());
    };

    // --- Stage 5: decode -> letterbox removal (zeros) -> projection.
    let mut lms = landmark::decode_landmarks(&raw);
    landmark::letterbox_remove(&mut lms, padding);
    let projected = landmark::project_landmarks(&lms, &hand_rect);

    // Landmark bbox + wrist/middle-mcp for ground-truth comparison.
    let (mut min_x, mut min_y, mut max_x, mut max_y) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for lm in &projected {
        min_x = min_x.min(lm[0]);
        max_x = max_x.max(lm[0]);
        min_y = min_y.min(lm[1]);
        max_y = max_y.max(lm[1]);
    }
    println!(
        "landmarks: bbox x {min_x:.3}..{max_x:.3} y {min_y:.3}..{max_y:.3}  (v1: x 0.433..0.515 y 0.838..0.981)"
    );
    println!(
        "  wrist ({:.3}, {:.3}) [v1 0.433, 0.838]  middle_mcp ({:.3}, {:.3}) [v1 0.461, 0.911]",
        projected[0][0], projected[0][1], projected[9][0], projected[9][1]
    );

    // Dump for numeric comparison against v1_ground_truth.py's JSON.
    let json: Vec<String> = projected
        .iter()
        .map(|lm| format!("[{:.6}, {:.6}, {:.6}]", lm[0], lm[1], lm[2]))
        .collect();
    std::fs::write("rust_landmarks.json", format!("[{}]", json.join(",")))?;
    println!("wrote rust_landmarks.json");

    // Annotate: 21 landmark dots (wrist = red, rest = green) + bbox.
    let mut out = img.clone();
    for (i, lm) in projected.iter().enumerate() {
        let (kx, ky) = ((lm[0] * fw as f32) as i32, (lm[1] * fh as f32) as i32);
        let color = if i == 0 {
            image::Rgb([255u8, 0, 0])
        } else {
            image::Rgb([0u8, 255, 0])
        };
        for dy in -3..=3 {
            for dx in -3..=3 {
                if dx * dx + dy * dy <= 9 {
                    set(&mut out, kx + dx, ky + dy, color);
                }
            }
        }
    }
    out.save("frame_landmarks.png")?;
    println!("wrote frame_landmarks.png");
    Ok(())
}

/// M3.4 behavioral gate: stream-mode state machine over N iterations of the
/// SAME frame — frame 1 detect path, later frames track path (prev
/// next-frame rect, palm detector skipped), presence gate (raw > 0.6 =
/// min_detection_confidence) deciding emission + loopback. Dumps
/// rust_video_landmarks.json = [frame][21][x,y,z] for parity against v1's
/// VIDEO-mode detect_for_video sequence.
fn run_video(
    environment: &Environment,
    img: &image::RgbImage,
    fw: u32,
    fh: u32,
    iters: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    use pinchflick_rs::vision::{decode, image as vimage, landmark, rect, tracking, warp};

    let img_bytes = img.as_raw().clone();
    let image_size = (fw as f32, fh as f32);
    let mut det_session = Session::builder(environment)?.commit_from_file("models/palm_detector.onnx")?;
    let mut lm_session = Session::builder(environment)?.commit_from_file("models/hand_landmark.onnx")?;
    let anchors = generate_anchors();
    let mut tracker = tracking::TrackerState::default();
    let mut frames: Vec<Option<Vec<[f32; 3]>>> = Vec::new();

    for iter in 0..iters {
        let plan = tracking::plan_frame(
            &tracker.prev_rects,
            &[],
            1,
            tracking::DEFAULT_MIN_TRACKING_CONFIDENCE,
        );
        let hand_rect = if plan.run_detector {
            // Palm detector on the full frame (M3.2 path), rect via the
            // M3.3 detection pipeline, associate + clip (base is empty
            // here because run_detector implies prev < num_hands).
            let input = vimage::letterbox_normalize(&img_bytes, fw, fh, 192);
            let tensor = Tensor::from_array(([1i64, 192i64, 192i64, 3i64], input))?;
            let outputs = det_session.run(ort::inputs![tensor])?;
            let (boxes, scores) = {
                let b = outputs["Identity"].try_extract_tensor::<f32>()?;
                let s = outputs["Identity_1"].try_extract_tensor::<f32>()?;
                (b.1.to_vec(), s.1.to_vec())
            };
            drop(outputs);
            let mut dets = decode::decode_detections(&boxes, &scores, &anchors, 0.6);
            dets = decode::weighted_nms(&dets, 0.3);
            decode::project_detections_to_image(&mut dets, fw, fh, 192);
            let det_rects: Vec<rect::NormRect> = dets
                .iter()
                .map(|d| {
                    let rotation =
                        rect::detection_rotation(d.keypoints[0], d.keypoints[2], image_size);
                    let (x_center, y_center, width, height) =
                        rect::norm_rect_from_keypoints(&d.keypoints);
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
                &tracker.prev_rects,
                &det_rects,
                tracking::DEFAULT_MIN_TRACKING_CONFIDENCE,
            );
            tracking::clip_rects(rects, 1).into_iter().next()
        } else {
            tracker.prev_rects.first().copied()
        };

        let Some(r) = hand_rect else {
            println!("frame {iter}: no hand rect (no detection)");
            tracker.update(vec![]);
            frames.push(None);
            continue;
        };
        let path = if plan.run_detector { "DETECT" } else { "TRACK " };

        // Crop + landmark model.
        let mut roi = warp::get_roi(fw, fh, &r);
        let padding = warp::pad_roi(224, 224, false, &mut roi);
        let crop = warp::warp_crop(&img_bytes, fw, fh, &roi, 224, (0.0, 1.0));
        let crop_tensor = Tensor::from_array(([1i64, 224i64, 224i64, 3i64], crop))?;
        let lm_outputs = lm_session.run(ort::inputs![crop_tensor])?;
        let mut landmark_raw: Option<Vec<f32>> = None;
        let mut presence_raw = 0.0f32;
        for (oname, val) in lm_outputs.iter() {
            let (_, data) = val.try_extract_tensor::<f32>()?;
            let mean_abs = data.iter().map(|v| v.abs()).sum::<f32>() / data.len() as f32;
            if data.len() == 63 && mean_abs > 5.0 {
                landmark_raw = Some(data.to_vec());
            }
            if data.len() == 1 && oname.ends_with("Identity_1") {
                presence_raw = data[0]; // RAW (TensorsToFloats activation NONE)
            }
        }
        drop(lm_outputs);

        // Presence gate verbatim: raw > min_detection_confidence (0.6).
        let passed = tracking::presence_gate(presence_raw, 0.6);
        let emitted = if passed {
            let raw = landmark_raw.expect("landmark tensor");
            let mut lms = landmark::decode_landmarks(&raw);
            landmark::letterbox_remove(&mut lms, padding);
            let projected = landmark::project_landmarks(&lms, &r);
            let next = tracking::next_frame_rect(&projected, image_size);
            println!(
                "frame {iter}: {path} rect ({:.3}, {:.3}) {:.3}x{:.3} rot {:.3} | presence {presence_raw:.4} PASS | wrist ({:.3}, {:.3}) | next rect ({:.3}, {:.3}) {:.3}x{:.3} rot {:.3}",
                r.x_center, r.y_center, r.width, r.height, r.rotation,
                projected[0][0], projected[0][1],
                next.x_center, next.y_center, next.width, next.height, next.rotation
            );
            tracker.update(vec![next]);
            Some(projected)
        } else {
            println!(
                "frame {iter}: {path} rect ({:.3}, {:.3}) | presence {presence_raw:.4} FAIL -> gated empty -> re-detect next frame",
                r.x_center, r.y_center
            );
            tracker.update(vec![]);
            None
        };
        frames.push(emitted);
    }

    let json: Vec<String> = frames
        .iter()
        .map(|f| match f {
            Some(lms) => format!(
                "[{}]",
                lms.iter()
                    .map(|lm| format!("[{:.6}, {:.6}, {:.6}]", lm[0], lm[1], lm[2]))
                    .collect::<Vec<_>>()
                    .join(",")
            ),
            None => "null".to_string(),
        })
        .collect();
    std::fs::write("rust_video_landmarks.json", format!("[{}]", json.join(",")))?;
    println!("wrote rust_video_landmarks.json ({} frames)", frames.len());
    Ok(())
}

/// Letterbox to a 192x192 RGB8 buffer (zero/BLACK border), raw 0..255.
/// Layout-only twin of `frame_prep::letterbox_normalize`, for the norm sweep.
fn letterbox_u8(rgb: &[u8], width: u32, height: u32, size: u32) -> Vec<u8> {
    let (fw, fh) = (width as f32, height as f32);
    let scale = (size as f32 / fw).min(size as f32 / fh);
    let (dw, dh) = (fw * scale, fh * scale);
    let (off_x, off_y) = ((size as f32 - dw) / 2.0, (size as f32 - dh) / 2.0);
    let mut out = vec![0u8; (size * size * 3) as usize];
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
            let base = ((oy * size + ox) * 3) as usize;
            for c in 0..3usize {
                let p = |yy: u32, xx: u32| rgb[((yy * width + xx) as usize) * 3 + c] as f32;
                let v = p(y0, x0) * (1.0 - fx) * (1.0 - fy)
                    + p(y0, x1) * fx * (1.0 - fy)
                    + p(y1, x0) * (1.0 - fx) * fy
                    + p(y1, x1) * fx * fy;
                out[base + c] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

fn set(img: &mut image::RgbImage, x: i32, y: i32, p: image::Rgb<u8>) {
    if x >= 0 && y >= 0 && (x as u32) < img.width() && (y as u32) < img.height() {
        img.put_pixel(x as u32, y as u32, p);
    }
}

fn run_ort(
    environment: &Environment,
    path: &str,
    name: &str,
    rgb: &[u8],
    fw: u32,
    fh: u32,
    (iw, ih): (u32, u32),
    center_crop: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let t_load = Instant::now();
    let mut session = Session::builder(environment)?.commit_from_file(path)?;
    println!("{name}: session loaded in {:.1?} ({path})", t_load.elapsed());
    for i in session.inputs() {
        println!("{name}:   in  {i:?}");
    }
    for o in session.outputs() {
        println!("{name}:   out {o:?}");
    }

    let prepared: Vec<u8> = if center_crop {
        let side = fw.min(fh);
        let x0 = (fw - side) / 2;
        let y0 = (fh - side) / 2;
        let mut sq = Vec::with_capacity((side * side * 3) as usize);
        for y in y0..y0 + side {
            for x in x0..x0 + side {
                let i = ((y * fw + x) * 3) as usize;
                sq.extend_from_slice(&rgb[i..i + 3]);
            }
        }
        spike::resize_nearest(&sq, side, side, iw, ih)
    } else {
        spike::resize_nearest(rgb, fw, fh, iw, ih)
    };
    let norm = spike::normalize_minus1_1(&prepared);
    let tensor = Tensor::from_array(([1i64, ih as i64, iw as i64, 3i64], norm))?;

    let t1 = Instant::now();
    let first = session.run(ort::inputs![tensor.clone()])?;
    println!("{name}: first run {:.2?}", t1.elapsed());

    // extract outputs while we hold them, before the timed loop re-borrows the session
    let mut collected = Vec::new();
    for (oname, val) in first.iter() {
        let (shape, data) = val.try_extract_tensor::<f32>()?;
        collected.push((oname.to_string(), shape.to_vec(), data.to_vec()));
    }
    drop(first);
    for (idx, (oname, shape, data)) in collected.iter().enumerate() {
        let head: Vec<String> = data.iter().take(8).map(|v| format!("{v:.4}")).collect();
        println!(
            "{name}: out[{idx}] '{oname}' shape {shape:?} head [{}] ({} values)",
            head.join(", "),
            data.len()
        );
    }
    if name == "detector" && collected.len() >= 2 {
        dump_detector(&collected[0].2, &collected[1].2);
    }

    let mut times = Vec::new();
    for _ in 0..10 {
        let t = Instant::now();
        session.run(ort::inputs![tensor.clone()])?;
        times.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let avg = times.iter().sum::<f64>() / times.len() as f64;
    println!(
        "{name}: avg {avg:.2} ms, min {:.2} ms (10 timed runs)",
        times[0]
    );
    Ok(())
}

/// M1 inspection dump: what the detector's 2016 raw anchors actually look like.
fn dump_detector(anchors: &[f32], scores: &[f32]) {
    let (na, w) = (anchors.len() / 18, 18);
    println!("\n--- DUMP: anchors tensor — {na} anchors x {w} values, first 3 rows in full ---");
    for r in 0..3.min(na) {
        let row: Vec<String> = anchors[r * w..(r + 1) * w]
            .iter()
            .map(|v| format!("{v:8.3}"))
            .collect();
        println!("  a{r:04}: {}", row.join(" "));
    }
    println!("--- DUMP: per-column min / mean / max across all {na} anchors ---");
    for c in 0..w {
        let mut mn = f32::MAX;
        let mut mx = f32::MIN;
        let mut sum = 0f64;
        for r in 0..na {
            let v = anchors[r * w + c];
            mn = mn.min(v);
            mx = mx.max(v);
            sum += v as f64;
        }
        println!("  col {c:02}: {mn:9.3} / {:8.3} / {mx:9.3}", sum / na as f64);
    }
    let sig = |x: f32| 1.0 / (1.0 + (-x).exp());
    let mut ranked: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let raw_min = scores.iter().copied().fold(f32::INFINITY, f32::min);
    let raw_max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    println!(
        "--- DUMP: score logits: raw min {raw_min:.3} max {raw_max:.3} (sigmoid {:.6}..{:.6}); top-5 anchors ---",
        sig(raw_min),
        sig(raw_max)
    );
    for (i, v) in ranked.iter().take(5) {
        println!("  anchor #{i:04}: logit {v:8.3}  ->  p = {:.6}", sig(*v));
    }
}

