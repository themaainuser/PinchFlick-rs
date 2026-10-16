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
use ort::environment::Environment;
use ort::session::Session;
use ort::value::Tensor;
use pinchflick_rs::vision::{anchors::generate_anchors, decode};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let frame_path = args
        .iter()
        .find(|a| !a.starts_with("--"))
        .cloned()
        .unwrap_or_else(|| "frame.png".into());
    let detect_mode = args.iter().any(|a| a == "--detect");
    let landmarks_mode = args.iter().any(|a| a == "--landmarks");
    let video_iters: Option<usize> = args.iter().position(|a| a == "--video").map(|i| {
        args.get(i + 1)
            .and_then(|s| s.parse().ok())
            .unwrap_or(3)
    });
    let live_mode = args.iter().any(|a| a == "--live");
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

    let environment = ort::init().with_name("pinchflick-rs").build()?;

    if live_mode {
        return run_live(&environment, config_path, camera_override);
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

/// M3.5 live CLI loop — camera -> Pipeline -> GestureEngine -> console
/// echo + rolling FPS/stage timings. DRY RUN: actions are printed, not
/// injected (enigo injection is M3.6). Mirrors v1's controller loop
/// (controller.py `run()`): mirror before inference, strictly increasing
/// VIDEO timestamps, YouTubeDetector polled at its interval.
fn run_live(
    environment: &Environment,
    config_path: Option<String>,
    camera_override: Option<i64>,
) -> Result<(), Box<dyn std::error::Error>> {
    use pinchflick_rs::config;
    use pinchflick_rs::gestures::engine::{GestureEngine, Landmark};
    use pinchflick_rs::gestures::seek::seek_key_sequence;
    use pinchflick_rs::sys::title::YouTubeDetector;
    use pinchflick_rs::track::{capture::Capture, inference::Pipeline};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    let mut cfg = config::load_config(config_path.as_deref())?;
    if let Some(idx) = camera_override {
        cfg.camera_index = idx;
    }

    let mut engine = GestureEngine::new(&cfg, cfg.mirror_display);
    let mut ytd = YouTubeDetector::new(
        cfg.youtube_title_markers.clone(),
        cfg.youtube_check_interval_ms,
    );
    let mut capture = Capture::open(
        cfg.camera_index,
        cfg.frame_width,
        cfg.frame_height,
        cfg.mirror_display,
    )
    .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
    let mut pipeline = Pipeline::new(
        environment,
        cfg.min_detection_confidence,
        cfg.min_tracking_confidence,
    )?;

    println!();
    println!("PinchFlick running (CLI dry-run; injection arrives in M3.6).");
    println!("  scroll : 1 finger = up, 2 fingers = down (works everywhere)");
    println!(
        "  seek   : double-pinch left/right = -/+{}s ({})",
        cfg.seek_seconds,
        if cfg.seek_requires_youtube {
            "YouTube tab must be focused"
        } else {
            "anywhere"
        }
    );
    println!("  keys   : q = quit (console)");
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

    let t0 = Instant::now();
    let mut last_ts = -1i64;
    let mut stop_reason = "quit (q)";
    // v1's stats dict.
    let mut frames = 0u64;
    let mut hand_frames = 0u64;
    let mut seeks = 0u64;
    let mut scroll_events = 0u64;
    let mut wheel = 0i64;
    // Rolling 2 s window for the stage report.
    let mut win_start = Instant::now();
    let mut win_frames = 0u64;
    let mut win_wait_ms = 0f64;
    let mut win_decode_ms = 0f64;
    let mut win_hand = 0u64;
    let mut win_detects = 0u64;
    let mut win_detect_ms = 0f64;
    let mut win_landmark_ms = 0f64;
    let mut win_landmark_n = 0u64;

    'live: loop {
        if quit.load(Ordering::SeqCst) {
            break 'live;
        }
        let (rgb, w, h) = match capture.capture() {
            Ok(v) => v,
            Err(e) => {
                stop_reason = "camera stopped delivering frames";
                println!("[capture error] {e}");
                break 'live;
            }
        };

        let mut ts = t0.elapsed().as_millis() as i64;
        if ts <= last_ts {
            ts = last_ts + 1; // VIDEO mode demands strictly increasing stamps
        }
        last_ts = ts;

        let out = match pipeline.process(&rgb, w, h) {
            Ok(o) => o,
            Err(e) => {
                stop_reason = "inference error";
                println!("[pipeline error] {e}");
                break 'live;
            }
        };
        frames += 1;
        win_frames += 1;
        win_wait_ms += capture.last_wait_ms;
        win_decode_ms += capture.last_decode_ms;
        if out.landmarks.is_some() {
            hand_frames += 1;
            win_hand += 1;
        }
        if !out.detect_ms.is_zero() {
            win_detects += 1;
            win_detect_ms += out.detect_ms.as_secs_f64() * 1e3;
        }
        if !out.landmark_ms.is_zero() {
            win_landmark_n += 1;
            win_landmark_ms += out.landmark_ms.as_secs_f64() * 1e3;
        }

        let landmarks = out.landmarks.as_ref().map(|lms| {
            lms.iter()
                .map(|l| Landmark::new(l[0] as f64, l[1] as f64))
                .collect::<Vec<_>>()
        });
        let (on_youtube, _title) = ytd.check(ts, false);
        let res = engine.frame(landmarks.as_deref(), ts, on_youtube);

        // ---- act: scroll (anywhere) — printed only until M3.6 ----
        if res.scroll_clicks != 0 {
            scroll_events += 1;
            wheel += res.scroll_clicks as i64;
            println!("[scroll] {:+} clicks (dry-run)", res.scroll_clicks);
        }

        // ---- act: seek (engine already YouTube-gates it) ----
        if let Some(dir) = res.seek {
            match seek_key_sequence(cfg.seek_seconds, dir.as_str(), &cfg.seek_keys) {
                Ok(seq) => {
                    seeks += 1;
                    let arrow = if dir.as_str() == "backward" { "-" } else { "+" };
                    println!(
                        "[seek] {arrow}{}s ({}) (dry-run)",
                        cfg.seek_seconds,
                        seq.join(" + ")
                    );
                }
                Err(e) => println!("[seek error] {e}"),
            }
        }
        if let Some(blocked) = &res.seek_blocked {
            println!("[seek blocked] {blocked}");
        }
        if res.tap.is_some() && res.seek.is_none() {
            println!("[tap] {} - double-pinch again to seek", res.tap_status);
        }

        if win_start.elapsed().as_secs_f64() >= 2.0 {
            let secs = win_start.elapsed().as_secs_f64();
            let fps = win_frames as f64 / secs;
            let det_avg = if win_detects > 0 {
                win_detect_ms / win_detects as f64
            } else {
                0.0
            };
            let lmk_avg = if win_landmark_n > 0 {
                win_landmark_ms / win_landmark_n as f64
            } else {
                0.0
            };
            let wait_avg = if win_frames > 0 {
                win_wait_ms / win_frames as f64
            } else {
                0.0
            };
            let decode_avg = if win_frames > 0 {
                win_decode_ms / win_frames as f64
            } else {
                0.0
            };
            println!(
                "[stats] {fps:.1} fps | hand {}% | wait {wait_avg:.1}ms | decode {decode_avg:.1}ms | detect {det_avg:.1}ms ({} runs) | landmark {lmk_avg:.1}ms | total {frames}",
                if win_frames > 0 {
                    win_hand * 100 / win_frames
                } else {
                    0
                },
                win_detects
            );
            win_start = Instant::now();
            win_frames = 0;
            win_wait_ms = 0.0;
            win_decode_ms = 0.0;
            win_hand = 0;
            win_detects = 0;
            win_detect_ms = 0.0;
            win_landmark_ms = 0.0;
            win_landmark_n = 0;
        }
    }

    let total = t0.elapsed().as_secs_f64();
    println!();
    println!("Stopped: {stop_reason}");
    println!(
        "  frames {frames} ({:.1} fps avg) | hand {hand_frames} | seeks {seeks} | scroll events {scroll_events} | wheel {wheel:+}",
        if total > 0.0 { frames as f64 / total } else { 0.0 }
    );
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
