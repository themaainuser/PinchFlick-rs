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
    let mut args = std::env::args().skip(1);
    let frame_path = args.next().unwrap_or_else(|| "frame.png".into());
    let detect_mode = args.any(|a| a == "--detect");
    let img = image::open(&frame_path)?.to_rgb8();
    let (fw, fh) = img.dimensions();
    let fbytes = img.as_raw();
    println!("frame: {fw}x{fh} ({frame_path})");

    let environment = ort::init().with_name("pinchflick-rs").build()?;

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
