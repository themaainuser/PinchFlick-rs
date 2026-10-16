//! SSD anchor generation for the palm detector — a direct port of MediaPipe's
//! `SsdAnchorsCalculator` with the exact options `hand_detector_graph.cc`
//! (v1.1.1) sets. Produces the 2016 normalized anchors the model's raw output
//! is decoded against.

/// One anchor box, normalized to the 192x192 input tensor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Anchor {
    pub x_center: f32,
    pub y_center: f32,
    pub w: f32,
    pub h: f32,
}

/// `CalculateScale` from ssd_anchors_calculator.cc — linear interpolation
/// between min and max scale across the layer indices.
fn calculate_scale(min_scale: f32, max_scale: f32, stride_index: usize, num_strides: usize) -> f32 {
    if num_strides == 1 {
        (min_scale + max_scale) * 0.5
    } else {
        min_scale + (max_scale - min_scale) * stride_index as f32 / (num_strides - 1) as f32
    }
}

/// Generates the palm detector's anchors exactly as configured in
/// `hand_detector_graph.cc` v1.1.1:
///   4 layers, min_scale 0.1484375, max_scale 0.75, input 192x192,
///   offsets (0.5, 0.5), strides [8, 16, 16, 16], aspect_ratios [1.0],
///   fixed_anchor_size true (every anchor is 1x1 — model regresses offsets
///   relative to the full frame), reduce_boxes_in_lowest_layer false and
///   interpolated_scale_aspect_ratio 1.0 (both proto defaults; the graph
///   relies on them).
///
/// Result: stride-8 group 24x24x2 = 1152 anchors + stride-16 group
/// 12x12x6 = 864 anchors = 2016 — the model's output row count.
pub fn generate_anchors() -> Vec<Anchor> {
    const MIN_SCALE: f32 = 0.1484375;
    const MAX_SCALE: f32 = 0.75;
    const INPUT: f32 = 192.0;
    const OFFSET: f32 = 0.5;
    const STRIDES: [u32; 4] = [8, 16, 16, 16];
    const ASPECT_RATIOS: [f32; 1] = [1.0];
    const INTERPOLATED_ASPECT: f32 = 1.0; // proto default

    let mut anchors = Vec::with_capacity(2016);
    let mut layer_id = 0;
    while layer_id < STRIDES.len() {
        // Merge consecutive layers sharing a stride (all three 16s merge).
        let stride = STRIDES[layer_id];
        let mut aspect_ratios: Vec<f32> = Vec::new();
        let mut scales: Vec<f32> = Vec::new();
        let mut last_same_stride_layer = layer_id;
        while last_same_stride_layer < STRIDES.len()
            && STRIDES[last_same_stride_layer] == stride
        {
            let scale = calculate_scale(
                MIN_SCALE,
                MAX_SCALE,
                last_same_stride_layer,
                STRIDES.len(),
            );
            for &ar in &ASPECT_RATIOS {
                aspect_ratios.push(ar);
                scales.push(scale);
            }
            if INTERPOLATED_ASPECT > 0.0 {
                let scale_next = if last_same_stride_layer == STRIDES.len() - 1 {
                    1.0
                } else {
                    calculate_scale(
                        MIN_SCALE,
                        MAX_SCALE,
                        last_same_stride_layer + 1,
                        STRIDES.len(),
                    )
                };
                scales.push((scale * scale_next).sqrt());
                aspect_ratios.push(INTERPOLATED_ASPECT);
            }
            last_same_stride_layer += 1;
        }

        let feature_map = (INPUT / stride as f32).ceil() as i32;
        for y in 0..feature_map {
            for x in 0..feature_map {
                for &ar in &aspect_ratios {
                    // Aspect/scale are computed above to mirror the reference
                    // loop (and determine the per-location shape count), but
                    // fixed_anchor_size discards them: every anchor is 1x1.
                    let _ = (ar, &scales);
                    anchors.push(Anchor {
                        x_center: (x as f32 + OFFSET) / feature_map as f32,
                        y_center: (y as f32 + OFFSET) / feature_map as f32,
                        w: 1.0,
                        h: 1.0,
                    });
                }
            }
        }
        layer_id = last_same_stride_layer;
    }
    anchors
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn produces_exactly_2016_anchors() {
        // Settles the M1 open question: the model emits 2016 boxes; the
        // graph config must generate a matching anchor count.
        assert_eq!(generate_anchors().len(), 2016);
    }

    #[test]
    fn stride8_group_starts_at_grid_center() {
        // First stride-8 location (x=0, y=0): center = (0.5/24, 0.5/24).
        let anchors = generate_anchors();
        let a = &anchors[0];
        let expect = 0.5 / 24.0;
        assert!((a.x_center - expect).abs() < 1e-6, "got {}", a.x_center);
        assert!((a.y_center - expect).abs() < 1e-6, "got {}", a.y_center);
    }

    #[test]
    fn fixed_anchor_size_makes_every_anchor_full_frame() {
        for a in generate_anchors() {
            assert_eq!(a.w, 1.0);
            assert_eq!(a.h, 1.0);
        }
    }

    #[test]
    fn stride8_locations_repeat_for_two_shapes() {
        // 2 shapes per stride-8 location -> indices 0,1 share a center;
        // index 2 moves to the next x.
        let anchors = generate_anchors();
        assert_eq!(anchors[0], anchors[1]);
        let expect_x = 1.5 / 24.0;
        assert!((anchors[2].x_center - expect_x).abs() < 1e-6);
        assert!((anchors[2].y_center - 0.5 / 24.0).abs() < 1e-6);
        assert_eq!(anchors[2], anchors[3]);
    }

    #[test]
    fn stride16_group_follows_after_1152() {
        // 576 locations x 2 shapes = 1152; next anchor starts the stride-16
        // grid at (0.5/12, 0.5/12), 6 shapes per location.
        let anchors = generate_anchors();
        let a = &anchors[1152];
        let expect = 0.5 / 12.0;
        assert!((a.x_center - expect).abs() < 1e-6, "got {}", a.x_center);
        assert!((a.y_center - expect).abs() < 1e-6, "got {}", a.y_center);
        assert_eq!(anchors[1152], anchors[1157], "6 shapes share a location");
    }

    #[test]
    fn last_anchor_is_bottom_right_of_stride16_grid() {
        let anchors = generate_anchors();
        let a = &anchors[2015];
        let expect = 11.5 / 12.0;
        assert!((a.x_center - expect).abs() < 1e-6, "got {}", a.x_center);
        assert!((a.y_center - expect).abs() < 1e-6, "got {}", a.y_center);
    }

    #[test]
    fn interpolated_scale_matches_linear_formula() {
        // calculate_scale is exposed for parity checks: scale at index 1 of 4
        // strides between 0.1484375 and 0.75.
        let s = calculate_scale(0.1484375, 0.75, 1, 4);
        assert!((s - (0.1484375 + (0.75 - 0.1484375) / 3.0)).abs() < 1e-7);
    }
}
