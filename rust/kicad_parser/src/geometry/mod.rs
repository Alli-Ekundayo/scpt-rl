//! Geometric operations for KiCad board data.
//!
//! Provides coordinate transforms, bounding box computation, board outline extraction,
//! and polygon operations (point-in-polygon, area) backed by the `geo` crate.

use crate::models::{BoundingBox, Point2D};
use crate::parser::pcb::{EdgeCutSegment, RawFootprint};
use geo::algorithm::area::Area;
use geo::algorithm::bounding_rect::BoundingRect;
use geo::algorithm::contains::Contains;
use geo::{LineString, Polygon as GeoPolygon};

/// Number of interpolation points used when converting arcs/circles to polylines.
const ARC_INTERPOLATION_STEPS: usize = 24;

/// Tolerance for matching segment endpoints during outline chaining (mm).
const CHAIN_SNAP_TOLERANCE: f64 = 0.01;

/// Rotates a point around origin (0,0) by `rotation_deg` degrees.
pub fn rotate_point(pt: &Point2D, rotation_deg: f64) -> Point2D {
    let rad = rotation_deg.to_radians();
    let cos_r = rad.cos();
    let sin_r = rad.sin();
    Point2D {
        x: pt.x * cos_r - pt.y * sin_r,
        y: pt.x * sin_r + pt.y * cos_r,
    }
}

/// Transforms local footprint pad position to world PCB coordinates.
pub fn transform_local_to_world(local: &Point2D, fp_pos: &Point2D, fp_rot_deg: f64) -> Point2D {
    let rotated = rotate_point(local, fp_rot_deg);
    Point2D {
        x: fp_pos.x + rotated.x,
        y: fp_pos.y + rotated.y,
    }
}

/// Computes the axis-aligned bounding box of a list of 2D points.
///
/// Returns a default (zero) bounding box if the input is empty.
pub fn compute_bounding_box(pts: &[Point2D]) -> BoundingBox {
    if pts.is_empty() {
        return BoundingBox::default();
    }
    // Use geo crate for bounding rect computation
    let points: Vec<geo::Point<f64>> = pts.iter().map(|p| geo::Point::new(p.x, p.y)).collect();
    let multi_point = geo::MultiPoint::new(points);
    if let Some(rect) = multi_point.bounding_rect() {
        BoundingBox {
            min_x: rect.min().x,
            min_y: rect.min().y,
            max_x: rect.max().x,
            max_y: rect.max().y,
        }
    } else {
        BoundingBox::default()
    }
}

/// Computes bounding box for a footprint based on pad positions and courtyard points.
///
/// Correctly accounts for pad rotation: computes the four corners of each pad rectangle
/// in local space, rotates them by the pad's local rotation, then transforms to world
/// coordinates via the footprint's position and rotation.
pub fn compute_footprint_bbox(fp: &RawFootprint) -> BoundingBox {
    let mut pts = Vec::new();

    // 1. Add pad world bounding box corners (with proper pad rotation)
    for pad in &fp.pads {
        let half_w = pad.size_mm.x / 2.0;
        let half_h = pad.size_mm.y / 2.0;
        // Four corners of the pad in pad-local space (centered at pad origin)
        let local_corners = [
            Point2D { x: -half_w, y: -half_h },
            Point2D { x:  half_w, y: -half_h },
            Point2D { x:  half_w, y:  half_h },
            Point2D { x: -half_w, y:  half_h },
        ];
        for corner in &local_corners {
            // Rotate by pad's local rotation
            let rotated_by_pad = rotate_point(corner, pad.local_rotation_deg);
            // Translate to footprint-local coordinates
            let fp_local = Point2D {
                x: rotated_by_pad.x + pad.local_pos.x,
                y: rotated_by_pad.y + pad.local_pos.y,
            };
            // Transform from footprint-local to world
            pts.push(transform_local_to_world(&fp_local, &fp.position, fp.rotation_deg));
        }
    }

    // 2. Add courtyard points if present
    for pt in &fp.courtyard_points {
        pts.push(transform_local_to_world(pt, &fp.position, fp.rotation_deg));
    }

    // Fallback if no pads or courtyard: use 2x2mm default centered at footprint position
    if pts.is_empty() {
        pts.push(Point2D {
            x: fp.position.x - 1.0,
            y: fp.position.y - 1.0,
        });
        pts.push(Point2D {
            x: fp.position.x + 1.0,
            y: fp.position.y + 1.0,
        });
    }

    compute_bounding_box(&pts)
}

/// Extracts an ordered board outline polygon and bounding box from Edge.Cuts graphics.
///
/// The algorithm:
/// 1. Converts each segment to a polyline (interpolating arcs and circles).
/// 2. Chains segments end-to-end by matching endpoints within a snap tolerance.
/// 3. Returns the longest closed contour as the outline polygon.
///
/// For simple cases (single closed loop of lines), this produces a correctly ordered
/// polygon. For complex boards with multiple contours, the largest contour is returned.
pub fn extract_board_outline(segments: &[EdgeCutSegment]) -> (Vec<Point2D>, BoundingBox) {
    if segments.is_empty() {
        return (Vec::new(), BoundingBox::default());
    }

    // Step 1: Convert each segment to a polyline (Vec<Point2D>)
    let polylines: Vec<Vec<Point2D>> = segments.iter().map(segment_to_polyline).collect();

    // Step 2: Chain polylines end-to-end
    let contours = chain_polylines(&polylines);

    // Step 3: Pick the longest contour (most points = likely the board outline)
    let best = contours
        .into_iter()
        .max_by_key(|c| c.len())
        .unwrap_or_default();

    let bbox = compute_bounding_box(&best);
    (best, bbox)
}

/// Converts an EdgeCutSegment to an ordered polyline.
fn segment_to_polyline(seg: &EdgeCutSegment) -> Vec<Point2D> {
    match seg {
        EdgeCutSegment::Line { start, end } => {
            vec![*start, *end]
        }
        EdgeCutSegment::Arc { start, mid, end } => interpolate_arc(start, mid, end),
        EdgeCutSegment::Circle { center, radius } => {
            let mut pts = Vec::with_capacity(ARC_INTERPOLATION_STEPS + 1);
            for i in 0..=ARC_INTERPOLATION_STEPS {
                let angle =
                    2.0 * std::f64::consts::PI * (i as f64) / (ARC_INTERPOLATION_STEPS as f64);
                pts.push(Point2D {
                    x: center.x + radius * angle.cos(),
                    y: center.y + radius * angle.sin(),
                });
            }
            pts
        }
        EdgeCutSegment::Polygon { pts } => pts.clone(),
    }
}

/// Interpolates a 3-point arc (start, mid, end) into a polyline.
///
/// Computes the circle center from three points, then interpolates from start angle
/// to end angle, ensuring the arc passes through the midpoint.
fn interpolate_arc(start: &Point2D, mid: &Point2D, end: &Point2D) -> Vec<Point2D> {
    // Find center of circle through three points using perpendicular bisectors
    let ax = start.x;
    let ay = start.y;
    let bx = mid.x;
    let by = mid.y;
    let cx = end.x;
    let cy = end.y;

    let d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
    if d.abs() < 1e-12 {
        // Degenerate arc (collinear points) — fall back to straight line
        return vec![*start, *mid, *end];
    }

    let ux = ((ax * ax + ay * ay) * (by - cy)
        + (bx * bx + by * by) * (cy - ay)
        + (cx * cx + cy * cy) * (ay - by))
        / d;
    let uy = ((ax * ax + ay * ay) * (cx - bx)
        + (bx * bx + by * by) * (ax - cx)
        + (cx * cx + cy * cy) * (bx - ax))
        / d;

    let center = Point2D { x: ux, y: uy };
    let radius = start.distance_to(&center);

    let start_angle = (ay - uy).atan2(ax - ux);
    let mid_angle = (by - uy).atan2(bx - ux);
    let end_angle = (cy - uy).atan2(cx - ux);

    // Determine sweep direction: we need to go from start_angle through mid_angle to end_angle
    let mut sweep = end_angle - start_angle;
    // Normalize sweep to (-2π, 2π)
    while sweep > 2.0 * std::f64::consts::PI {
        sweep -= 2.0 * std::f64::consts::PI;
    }
    while sweep < -2.0 * std::f64::consts::PI {
        sweep += 2.0 * std::f64::consts::PI;
    }

    // Check if mid_angle is within the sweep from start to end
    let mid_in_sweep = angle_in_range(mid_angle, start_angle, start_angle + sweep);
    if !mid_in_sweep {
        // Mid not in the current sweep direction — reverse
        sweep = if sweep > 0.0 {
            sweep - 2.0 * std::f64::consts::PI
        } else {
            sweep + 2.0 * std::f64::consts::PI
        };
    }

    let n_steps = ARC_INTERPOLATION_STEPS;
    let mut pts = Vec::with_capacity(n_steps + 1);
    for i in 0..=n_steps {
        let t = i as f64 / n_steps as f64;
        let angle = start_angle + sweep * t;
        pts.push(Point2D {
            x: center.x + radius * angle.cos(),
            y: center.y + radius * angle.sin(),
        });
    }
    pts
}

/// Returns true if `angle` lies within the range [start, start + sweep],
/// accounting for angle wrapping.
fn angle_in_range(angle: f64, start: f64, sweep_end: f64) -> bool {
    let range = sweep_end - start;
    let mut diff = angle - start;
    // Normalize diff to [0, 2π)
    while diff < 0.0 {
        diff += 2.0 * std::f64::consts::PI;
    }
    while diff >= 2.0 * std::f64::consts::PI {
        diff -= 2.0 * std::f64::consts::PI;
    }
    let mut range_norm = range;
    while range_norm < 0.0 {
        range_norm += 2.0 * std::f64::consts::PI;
    }
    diff <= range_norm
}

/// Chains polyline segments end-to-end into closed contours.
///
/// Uses a greedy nearest-endpoint approach: for each chain, find the unvisited
/// segment whose start is closest to the current chain's end (within snap tolerance).
fn chain_polylines(polylines: &[Vec<Point2D>]) -> Vec<Vec<Point2D>> {
    if polylines.is_empty() {
        return Vec::new();
    }

    let n = polylines.len();
    let mut used = vec![false; n];
    let mut contours = Vec::new();

    for start_idx in 0..n {
        if used[start_idx] {
            continue;
        }

        let mut chain = polylines[start_idx].clone();
        used[start_idx] = true;

        // Greedily extend the chain
        let mut changed = true;
        while changed {
            changed = false;
            let chain_end = *chain.last().unwrap();

            // Find best next segment whose start matches chain_end
            let mut best_idx = None;
            let mut best_dist = CHAIN_SNAP_TOLERANCE;

            for (i, pl) in polylines.iter().enumerate() {
                if used[i] || pl.is_empty() {
                    continue;
                }
                let dist = chain_end.distance_to(&pl[0]);
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = Some(i);
                }
            }

            if let Some(idx) = best_idx {
                used[idx] = true;
                // Append the new polyline, skipping the first point (it duplicates chain_end)
                chain.extend(polylines[idx].iter().skip(1));
                changed = true;
            }
        }

        if !chain.is_empty() {
            contours.push(chain);
        }
    }

    contours
}

/// Tests whether a point lies inside a polygon using the `geo` crate.
///
/// The polygon is treated as closed (the last point is implicitly connected to the first).
pub fn point_in_polygon(point: &Point2D, polygon: &[Point2D]) -> bool {
    if polygon.len() < 3 {
        return false;
    }
    let coords: Vec<geo::Coord<f64>> = polygon
        .iter()
        .map(|p| geo::Coord { x: p.x, y: p.y })
        .collect();
    let geo_poly = GeoPolygon::new(LineString::from(coords), vec![]);
    let geo_point = geo::Point::new(point.x, point.y);
    geo_poly.contains(&geo_point)
}

/// Computes the signed area of a polygon using the `geo` crate.
///
/// Positive for counter-clockwise winding, negative for clockwise.
/// Returns 0.0 if the polygon has fewer than 3 vertices.
pub fn polygon_area(polygon: &[Point2D]) -> f64 {
    if polygon.len() < 3 {
        return 0.0;
    }
    let coords: Vec<geo::Coord<f64>> = polygon
        .iter()
        .map(|p| geo::Coord { x: p.x, y: p.y })
        .collect();
    let geo_poly = GeoPolygon::new(LineString::from(coords), vec![]);
    geo_poly.signed_area()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rotate_point_zero() {
        let pt = Point2D { x: 1.0, y: 0.0 };
        let rotated = rotate_point(&pt, 0.0);
        assert!((rotated.x - 1.0).abs() < 1e-9);
        assert!((rotated.y - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_rotate_point_90() {
        let pt = Point2D { x: 1.0, y: 0.0 };
        let rotated = rotate_point(&pt, 90.0);
        assert!((rotated.x - 0.0).abs() < 1e-9);
        assert!((rotated.y - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_compute_bounding_box() {
        let pts = vec![
            Point2D { x: 0.0, y: 0.0 },
            Point2D { x: 10.0, y: 5.0 },
            Point2D { x: -3.0, y: 7.0 },
        ];
        let bbox = compute_bounding_box(&pts);
        assert!((bbox.min_x - (-3.0)).abs() < 1e-9);
        assert!((bbox.min_y - 0.0).abs() < 1e-9);
        assert!((bbox.max_x - 10.0).abs() < 1e-9);
        assert!((bbox.max_y - 7.0).abs() < 1e-9);
    }

    #[test]
    fn test_compute_bounding_box_empty() {
        let bbox = compute_bounding_box(&[]);
        assert_eq!(bbox, BoundingBox::default());
    }

    #[test]
    fn test_extract_board_outline_rectangle() {
        // A simple rectangular outline from 4 line segments
        let segments = vec![
            EdgeCutSegment::Line {
                start: Point2D { x: 0.0, y: 0.0 },
                end: Point2D { x: 100.0, y: 0.0 },
            },
            EdgeCutSegment::Line {
                start: Point2D { x: 100.0, y: 0.0 },
                end: Point2D { x: 100.0, y: 80.0 },
            },
            EdgeCutSegment::Line {
                start: Point2D { x: 100.0, y: 80.0 },
                end: Point2D { x: 0.0, y: 80.0 },
            },
            EdgeCutSegment::Line {
                start: Point2D { x: 0.0, y: 80.0 },
                end: Point2D { x: 0.0, y: 0.0 },
            },
        ];
        let (outline, bbox) = extract_board_outline(&segments);
        // Should chain into a single contour
        assert!(outline.len() >= 5, "Outline should have at least 5 points (4 corners + closing), got {}", outline.len());
        assert!((bbox.min_x - 0.0).abs() < 1e-9);
        assert!((bbox.max_x - 100.0).abs() < 1e-9);
        assert!((bbox.min_y - 0.0).abs() < 1e-9);
        assert!((bbox.max_y - 80.0).abs() < 1e-9);
    }

    #[test]
    fn test_extract_board_outline_empty() {
        let (outline, bbox) = extract_board_outline(&[]);
        assert!(outline.is_empty());
        assert_eq!(bbox, BoundingBox::default());
    }

    #[test]
    fn test_point_in_polygon() {
        let polygon = vec![
            Point2D { x: 0.0, y: 0.0 },
            Point2D { x: 10.0, y: 0.0 },
            Point2D { x: 10.0, y: 10.0 },
            Point2D { x: 0.0, y: 10.0 },
        ];
        assert!(point_in_polygon(&Point2D { x: 5.0, y: 5.0 }, &polygon));
        assert!(!point_in_polygon(&Point2D { x: 15.0, y: 5.0 }, &polygon));
        assert!(!point_in_polygon(&Point2D { x: -1.0, y: -1.0 }, &polygon));
    }

    #[test]
    fn test_polygon_area() {
        let polygon = vec![
            Point2D { x: 0.0, y: 0.0 },
            Point2D { x: 10.0, y: 0.0 },
            Point2D { x: 10.0, y: 10.0 },
            Point2D { x: 0.0, y: 10.0 },
        ];
        let area = polygon_area(&polygon).abs();
        assert!((area - 100.0).abs() < 1e-9, "Area of 10x10 square should be 100, got {}", area);
    }

    #[test]
    fn test_compute_footprint_bbox_with_rotated_pad() {
        use crate::parser::pcb::{RawPad};
        let fp = RawFootprint {
            ref_des: "R1".to_string(),
            value: "10k".to_string(),
            footprint_name: "R_0805".to_string(),
            position: Point2D { x: 10.0, y: 20.0 },
            rotation_deg: 90.0,
            layer: "F.Cu".to_string(),
            attributes: vec![],
            pads: vec![
                RawPad {
                    number: "1".to_string(),
                    pad_type: "smd".to_string(),
                    shape: "rect".to_string(),
                    local_pos: Point2D { x: -1.0, y: 0.0 },
                    local_rotation_deg: 0.0,
                    size_mm: Point2D { x: 1.0, y: 1.2 },
                    drill_mm: None,
                    net_id: 0,
                    net_name: String::new(),
                },
                RawPad {
                    number: "2".to_string(),
                    pad_type: "smd".to_string(),
                    shape: "rect".to_string(),
                    local_pos: Point2D { x: 1.0, y: 0.0 },
                    local_rotation_deg: 0.0,
                    size_mm: Point2D { x: 1.0, y: 1.2 },
                    drill_mm: None,
                    net_id: 0,
                    net_name: String::new(),
                },
            ],
            courtyard_points: vec![],
        };
        let bbox = compute_footprint_bbox(&fp);
        // After 90-degree rotation, pad positions swap x/y relative to center
        // Pads were at local (-1,0) and (1,0), after 90° rotation: (0,-1) and (0,1)
        // World positions: (10, 19) and (10, 21)
        // With pad size 1.0 x 1.2, after rotation the bbox should be ~1.2 wide, ~3.0 tall
        assert!(bbox.width() > 0.5);
        assert!(bbox.height() > 1.5);
    }
}
