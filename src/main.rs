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
    for (idx, (oname, val)) in first.iter().enumerate() {
        let (shape, data) = val.try_extract_tensor::<f32>()?;
        let head: Vec<String> = data.iter().take(8).map(|v| format!("{v:.4}")).collect();
        println!(
            "{name}: out[{idx}] '{oname}' shape {shape:?} head [{}] ({} values)",
            head.join(", "),
            data.len()
        );
    }
    drop(first);

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
