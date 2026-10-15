// M1 — the ort/ONNX spike (design doc section 12 re-scope).
// Models: ONNX conversions of the official MediaPipe hand models
// (takoyakisoft/vtuberkit-vision-landmarkers-onnx, Apache-2.0, SHA256-verified).
// Question: do they run on onnxruntime via `ort`, and how fast per stage?
mod spike;

use std::time::Instant;
use ort::environment::Environment;
use ort::session::Session;
use ort::value::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let frame_path = std::env::args().nth(1).unwrap_or_else(|| "frame.png".into());
    let img = image::open(&frame_path)?.to_rgb8();
    let (fw, fh) = img.dimensions();
    let fbytes = img.as_raw();
    println!("frame: {fw}x{fh} ({frame_path})");

    let environment = ort::init().with_name("pinchflick-rs").build()?;

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
