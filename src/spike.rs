// M0 spike pure logic: .task archive extraction + image prep (TDD'd).
// The tract inference experiment itself lives in main.rs (design doc M0).

// M0's tested .task-extraction skeleton (ledger: "reusable skeleton kept");
// unused since the ONNX switch, kept for the record.
#![allow(dead_code)]

pub struct TaskModels {
    pub detector: Vec<u8>,
    pub landmarks: Vec<u8>,
}

/// Pull the two MediaPipe tflite models out of a `.task` archive (in memory).
pub fn extract_models_from_task(task_bytes: &[u8]) -> Result<TaskModels, String> {
    let reader = std::io::Cursor::new(task_bytes);
    let mut archive = zip::ZipArchive::new(reader).map_err(|e| format!("open zip: {e}"))?;
    let mut grab = |name: &str| -> Result<Vec<u8>, String> {
        let mut f = archive.by_name(name).map_err(|e| format!("{name}: {e}"))?;
        let mut buf = Vec::new();
        std::io::Read::read_to_end(&mut f, &mut buf).map_err(|e| format!("{name}: {e}"))?;
        Ok(buf)
    };
    let detector = grab("hand_detector.tflite")?;
    let landmarks = grab("hand_landmarks_detector.tflite")?;
    Ok(TaskModels { detector, landmarks })
}

/// Nearest-neighbour resize of an interleaved RGB8 buffer.
pub fn resize_nearest(rgb: &[u8], w: u32, h: u32, nw: u32, nh: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((nw * nh * 3) as usize);
    for y in 0..nh {
        let sy = y * h / nh;
        for x in 0..nw {
            let sx = x * w / nw;
            let i = ((sy * w + sx) * 3) as usize;
            out.extend_from_slice(&rgb[i..i + 3]);
        }
    }
    out
}

/// MediaPipe-style normalization to [-1, 1].
pub fn normalize_minus1_1(rgb: &[u8]) -> Vec<f32> {
    rgb.iter().map(|&v| v as f32 / 127.5 - 1.0).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const TASK: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../PinchFlick/hand_landmarker.task"
    );

    #[test]
    fn task_archive_yields_both_tflite_models() {
        let bytes = std::fs::read(TASK).expect("hand_landmarker.task readable");
        let m = extract_models_from_task(&bytes).expect("extraction succeeds");
        assert!(!m.detector.is_empty(), "detector model non-empty");
        assert!(!m.landmarks.is_empty(), "landmark model non-empty");
        // TFLite flatbuffers carry the "TFL3" file identifier at bytes 4..8.
        assert_eq!(&m.detector[4..8], b"TFL3", "detector is a tflite flatbuffer");
        assert_eq!(&m.landmarks[4..8], b"TFL3", "landmarks is a tflite flatbuffer");
    }

    #[test]
    fn resize_nearest_doubles_dims_and_keeps_corner_pixels() {
        // 2x2 RGB: red, green / blue, white
        let src = vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255];
        let out = resize_nearest(&src, 2, 2, 4, 4);
        assert_eq!(out.len(), 4 * 4 * 3);
        assert_eq!(&out[0..3], &[255, 0, 0], "top-left stays red");
        let br = (4 * 3 + 3) * 3;
        assert_eq!(&out[br..br + 3], &[255, 255, 255], "bottom-right stays white");
    }

    #[test]
    fn normalize_maps_full_byte_range_to_minus1_plus1() {
        let out = normalize_minus1_1(&[0, 127, 128, 255]);
        assert_eq!(out.len(), 4);
        assert!((out[0] - (-1.0)).abs() < 1e-6);
        assert!((out[3] - 1.0).abs() < 1e-6);
        assert!(out[1] < 0.0 && out[2] > 0.0, "midpoints straddle zero");
    }
}
