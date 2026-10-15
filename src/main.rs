// M0 — the make-or-break inference spike (design doc section 6, milestone M0).
// Question: can pure-Rust tract execute BOTH MediaPipe hand models from the
// .task archive, and how fast? Verdict decides tract vs ort+ONNX (section 5).
mod spike;

use std::io::Cursor;
use std::time::Instant;
use tract_tflite::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let task_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "../PinchFlick/hand_landmarker.task".into());
    let frame_path = std::env::args().nth(2).unwrap_or_else(|| "frame.png".into());

    let task_bytes = std::fs::read(&task_path)?;
    let models = spike::extract_models_from_task(&task_bytes)?;
    println!(
        "extracted from archive: hand_detector {} KB, hand_landmarks_detector {} KB",
        models.detector.len() / 1024,
        models.landmarks.len() / 1024
    );

    let img = image::open(&frame_path)?.to_rgb8();
    let (fw, fh) = img.dimensions();
    let fbytes = img.as_raw();
    println!("frame: {fw}x{fh} ({frame_path})");

    let only = std::env::args().nth(3).unwrap_or_else(|| "both".into());
    let mut results = Vec::new();
    if only != "landmarks" {
        println!("\n=== hand_detector ===");
        results.push(("detector", run_model("detector", &models.detector, fbytes, fw, fh)?));
    }
    if only != "detector" {
        println!("\n=== hand_landmarks_detector ===");
        results.push(("landmarks", run_model("landmarks", &models.landmarks, fbytes, fw, fh)?));
    }

    println!("\nM0 VERDICT:");
    for (name, verdict) in &results {
        println!("  {name}: {verdict}");
    }
    Ok(())
}

fn run_model(
    name: &str,
    bytes: &[u8],
    frame: &[u8],
    fw: u32,
    fh: u32,
) -> Result<String, Box<dyn std::error::Error>> {
    let t_load = Instant::now();
    let model = tract_tflite::tflite().model_for_read(&mut Cursor::new(bytes))?;
    let shape: Vec<usize> = model
        .input_fact(0)?
        .shape
        .as_concrete()
        .ok_or("dynamic input shape unsupported in spike")?
        .to_vec();
    println!(
        "{name}: model loaded in {:.1?}; input shape {:?}",
        t_load.elapsed(),
        shape
    );

    let (ih, iw, ic) = (shape[1], shape[2], shape[3]);
    let resized = spike::resize_nearest(frame, fw, fh, iw as u32, ih as u32);
    let norm = spike::normalize_minus1_1(&resized);
    assert_eq!(norm.len(), ih * iw * ic, "input buffer matches model shape");
    let tensor: Tensor = tract_ndarray::Array4::from_shape_vec((1, ih, iw, ic), norm)?.into();

    let t_opt = Instant::now();
    let runner: Box<dyn Fn(Tensor) -> TractResult<TVec<TValue>>> = match model.into_optimized() {
        Ok(m) => {
            let r = m.into_runnable()?;
            println!("{name}: optimized into runnable in {:.1?}", t_opt.elapsed());
            Box::new(move |t| r.run(tvec!(t.into())))
        }
        Err(e) => {
            println!("{name}: OPTIMIZE FAILED ({e}); falling back to typed-only runnable");
            let m2 = tract_tflite::tflite().model_for_read(&mut Cursor::new(bytes))?;
            let r2 = m2.into_runnable()?;
            Box::new(move |t| r2.run(tvec!(t.into())))
        }
    };

    // warmup, then timed runs
    let mut last = None;
    for _ in 0..2 {
        last = Some(runner(tensor.clone())?);
    }
    let mut times = Vec::new();
    for _ in 0..10 {
        let t = Instant::now();
        last = Some(runner(tensor.clone())?);
        times.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let avg = times.iter().sum::<f64>() / times.len() as f64;
    println!(
        "{name}: run time avg {avg:.2} ms, min {:.2} ms (10 timed runs after 2 warmups)",
        times[0]
    );

    let outs = last.unwrap();
    println!("{name}: {} output tensor(s):", outs.len());
    for (i, v) in outs.iter().enumerate() {
        let view = v.to_plain_array_view::<f32>()?;
        let shp: Vec<usize> = view.shape().to_vec();
        let head: Vec<String> = view.iter().take(6).map(|x| format!("{x:.4}")).collect();
        println!("  out[{i}] shape {shp:?} head [{}]", head.join(", "));
    }

    Ok(format!("ok, {avg:.2} ms avg per frame"))
}
