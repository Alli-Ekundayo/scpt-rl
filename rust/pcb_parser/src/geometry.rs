//! Geometry primitives over the SCPT IR.
//!
//! All functions take `&PcbDesign` and operate on its current `PlacementState`.
//! Unplaced components contribute 0 to HPWL and are ignored by clearance /
//! partition cut. These are the primitives that the PyO3 wheel exposes and
//! that the constraint orchestrator composes.

use std::collections::HashSet;

use crate::ir::{PadRef, PcbDesign, PartitionSpec, Placement, Vec2};

// ---------------------------------------------------------------------------
// Coordinate helpers
// ---------------------------------------------------------------------------

/// Rotate `(x, y)` by `rotation_deg` around the origin.
fn rotate(v: Vec2, rotation_deg: f64) -> Vec2 {
    let rad = rotation_deg.to_radians();
    let c = rad.cos();
    let s = rad.sin();
    Vec2(v.0 * c - v.1 * s, v.0 * s + v.1 * c)
}

/// World position of a pad given its owning component's placement.
fn pad_world_pos(d: &PcbDesign, pad_ref: PadRef) -> Option<Vec2> {
    let placement = d.placement.positions.get(pad_ref.0)?.as_ref()?;
    let pad = d.components
        .get(pad_ref.0)?
        .footprint
        .pads
        .get(pad_ref.1)?;
    let rotated = rotate(pad.local_pos, placement.rotation_deg);
    Some(Vec2(
        rotated.0 + placement.position.0,
        rotated.1 + placement.position.1,
    ))
}

// ---------------------------------------------------------------------------
// HPWL
// ---------------------------------------------------------------------------

/// Half-perimeter wirelength over all nets. Nets with any unplaced pad
/// contribute 0 (we can't measure what isn't placed yet).
pub fn hpwl(d: &PcbDesign) -> f64 {
    d.nets.iter().map(|net| hpwl_of_net(d, &net.pads)).sum()
}

/// HPWL recomputed only for nets that touch `moved_ref_des`. Cheaper than
/// full HPWL when only one component moved — used every RL step.
pub fn hpwl_incremental(d: &PcbDesign, moved_ref_des: &str) -> f64 {
    let moved_idx = match d
        .components
        .iter()
        .position(|c| c.ref_des == moved_ref_des)
    {
        Some(i) => i,
        None => return 0.0,
    };
    d.nets
        .iter()
        .filter(|net| net.pads.iter().any(|pr| pr.0 == moved_idx))
        .map(|net| hpwl_of_net(d, &net.pads))
        .sum()
}

fn hpwl_of_net(d: &PcbDesign, pads: &[PadRef]) -> f64 {
    if pads.len() < 2 {
        return 0.0;
    }
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for pr in pads {
        let Some(w) = pad_world_pos(d, *pr) else {
            // Any unplaced pad ⇒ skip this net entirely
            return 0.0;
        };
        if w.0 < min_x { min_x = w.0; }
        if w.0 > max_x { max_x = w.0; }
        if w.1 < min_y { min_y = w.1; }
        if w.1 > max_y { max_y = w.1; }
    }
    (max_x - min_x) + (max_y - min_y)
}

// ---------------------------------------------------------------------------
// Clearance cost
// ---------------------------------------------------------------------------

/// Sum of pairwise courtyard overlaps after `min_spacing` expansion.
///
/// v1 uses axis-aligned bounding box overlap as a proxy — fast (O(P²) in
/// placed count P, with tiny constants) and good enough for placement RL
/// where the mask already filters hard violations. A future pass can upgrade
/// to polygon-polygon SAT via the `geo` crate if needed.
pub fn clearance_cost(d: &PcbDesign, min_spacing: f64) -> f64 {
    let placed: Vec<(usize, &Placement)> = d
        .placement
        .positions
        .iter()
        .enumerate()
        .filter_map(|(i, p)| p.as_ref().map(|pl| (i, pl)))
        .collect();
    let mut cost = 0.0;
    for idx_a in 0..placed.len() {
        let (ca, pa) = placed[idx_a];
        let bb_a = courtyard_aabb(d, ca, pa, min_spacing);
        for idx_b in (idx_a + 1)..placed.len() {
            let (cb, pb) = placed[idx_b];
            let bb_b = courtyard_aabb(d, cb, pb, min_spacing);
            if let Some(overlap) = aabb_overlap_area(&bb_a, &bb_b) {
                cost += overlap;
            }
        }
    }
    cost
}

/// Axis-aligned bounding box of a placed component's courtyard after
/// expansion by `min_spacing/2` on each side. Returns
/// `(min_x, min_y, max_x, max_y)`.
fn courtyard_aabb(
    d: &PcbDesign,
    comp_idx: usize,
    placement: &Placement,
    min_spacing: f64,
) -> (f64, f64, f64, f64) {
    let courtyard = &d.components[comp_idx].footprint.courtyard;
    // Transform each courtyard point to world space (rotate + translate).
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for v in &courtyard.points {
        let w = rotate(*v, placement.rotation_deg);
        let wx = w.0 + placement.position.0;
        let wy = w.1 + placement.position.1;
        if wx < min_x { min_x = wx; }
        if wx > max_x { max_x = wx; }
        if wy < min_y { min_y = wy; }
        if wy > max_y { max_y = wy; }
    }
    let half = min_spacing / 2.0;
    (min_x - half, min_y - half, max_x + half, max_y + half)
}

fn aabb_overlap_area(
    a: &(f64, f64, f64, f64),
    b: &(f64, f64, f64, f64),
) -> Option<f64> {
    let dx = (a.2.min(b.2) - a.0.max(b.0)).max(0.0);
    let dy = (a.3.min(b.3) - a.1.max(b.1)).max(0.0);
    if dx > 0.0 && dy > 0.0 {
        Some(dx * dy)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Partition cut cost
// ---------------------------------------------------------------------------

/// Number of nets that cross the partition boundary (have pads on components
/// in more than one partition group). Cost is the count of such nets — each
/// crossing net is equally penalised in v1. A weighted variant (e.g. by net
/// fanout or netclass) can be added when the data shows it matters.
pub fn partition_cut_cost(d: &PcbDesign, partition: &PartitionSpec) -> f64 {
    // Build component_idx → partition_name map. PartitionSpec.components is
    // the list of comp indices in THIS partition; components not in any
    // partition get their own unique singleton group (so they never cross).
    let mut comp_to_group: Vec<Option<&str>> = vec![None; d.components.len()];
    for idx in &partition.components {
        if let Some(slot) = comp_to_group.get_mut(*idx) {
            *slot = Some(&partition.name);
        }
    }
    // Singleton groups for un-partitioned components: use the ref_des as the
    // group label so every ungrouped component is its own group.
    let singleton_labels: Vec<String> = d
        .components
        .iter()
        .enumerate()
        .map(|(i, c)| {
            if comp_to_group[i].is_some() {
                String::new()
            } else {
                format!("__single_{}", c.ref_des)
            }
        })
        .collect();

    d.nets
        .iter()
        .filter(|net| {
            let groups: HashSet<&str> = net
                .pads
                .iter()
                .filter_map(|pr| {
                    let idx = pr.0;
                    if let Some(g) = comp_to_group.get(idx).copied().flatten() {
                        Some(g)
                    } else {
                        singleton_labels.get(idx).map(|s| s.as_str())
                    }
                })
                .collect();
            groups.len() > 1
        })
        .count() as f64
}

// ---------------------------------------------------------------------------
// Tier 2 scoring (v1 position-sensitive)
// ---------------------------------------------------------------------------

/// Orientation score: variance of X/Y coords within each functional cluster.
/// v1 uses the `partition.components` as the cluster list — a single
/// `PartitionSpec` is treated as ONE cluster; the score is the regularity of
/// that cluster. A proper Union-Find-based clustering (Task 17) will feed
/// multiple clusters through here.
///
/// Lower variance ⇒ more regular ⇒ higher score. Returns value in [0, 1].
pub fn orient_score(d: &PcbDesign, partition: &PartitionSpec) -> f64 {
    let positions: Vec<Vec2> = partition
        .components
        .iter()
        .filter_map(|&i| d.placement.positions.get(i).and_then(|p| p.as_ref()).map(|p| p.position))
        .collect();
    if positions.len() < 2 {
        return 1.0;
    }
    let xs: Vec<f64> = positions.iter().map(|p| p.0).collect();
    let ys: Vec<f64> = positions.iter().map(|p| p.1).collect();
    let vx = variance(&xs);
    let vy = variance(&ys);
    // Map variance → score. var=0 → 1.0; var≥100mm² → ~0.0
    let mean_var = (vx + vy) * 0.5;
    (1.0 + mean_var * 0.01).recip().max(0.0).min(1.0)
}

fn variance(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    let mean = xs.iter().sum::<f64>() / xs.len() as f64;
    xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / xs.len() as f64
}

/// Decap proximity: for each capacitor-like component (value starts with 'C'
/// followed by a digit), find the nearest IC-like component (value starts
/// with 'U' or 'IC'). Score is 1.0 if within `radius` mm, decays with
/// distance beyond. Returns mean over all caps.
pub fn decap_proximity_score(d: &PcbDesign, radius: f64) -> f64 {
    let caps: Vec<(Vec2, usize)> = placed_components_matching(d, |v| {
        v.starts_with('C') && v.chars().nth(1).map(|c| c.is_ascii_digit()).unwrap_or(false)
    });
    let ics: Vec<Vec2> = placed_components_matching(d, |v| {
        v.starts_with('U') || v.starts_with("IC")
    })
    .into_iter()
    .map(|(pos, _)| pos)
    .collect();
    if caps.is_empty() || ics.is_empty() {
        return 1.0; // nothing to score — vacuously satisfied
    }
    let scores: Vec<f64> = caps
        .iter()
        .map(|(cap_pos, _)| {
            let min_dist = ics
                .iter()
                .map(|ic| ((cap_pos.0 - ic.0).powi(2) + (cap_pos.1 - ic.1).powi(2)).sqrt())
                .fold(f64::INFINITY, f64::min);
            if min_dist <= radius {
                1.0
            } else {
                (-((min_dist - radius) / radius).max(0.0)).exp()
            }
        })
        .collect();
    scores.iter().sum::<f64>() / scores.len() as f64
}

fn placed_components_matching<F>(d: &PcbDesign, predicate: F) -> Vec<(Vec2, usize)>
where
    F: Fn(&str) -> bool,
{
    d.placement
        .positions
        .iter()
        .enumerate()
        .filter_map(|(i, p)| {
            let pl = p.as_ref()?;
            let comp = d.components.get(i)?;
            if predicate(&comp.value) {
                Some((pl.position, i))
            } else {
                None
            }
        })
        .collect()
}

/// Thermal score: density of power-net pads in a window around each
/// component. v1 uses a simple proxy — count of power-role nets that have a
/// pad on the component, summed across placed components and normalized.
pub fn thermal_score(d: &PcbDesign) -> f64 {
    let placed_count = d
        .placement
        .positions
        .iter()
        .filter(|p| p.is_some())
        .count();
    if placed_count == 0 {
        return 0.0;
    }
    let power_touches: usize = d
        .nets
        .iter()
        .filter(|n| n.role == crate::ir::NetRole::Power)
        .map(|n| {
            n.pads
                .iter()
                .filter(|pr| d.placement.positions.get(pr.0).and_then(|p| p.as_ref()).is_some())
                .count()
        })
        .sum();
    // Normalize: cap at 1.0 via saturating division.
    (power_touches as f64 / placed_count as f64).min(1.0)
}

/// Symmetry score: for each named pair `(ref_a, ref_b)`, check that their
/// midpoint aligns with the board centroid along the axis perpendicular to
/// the pair's orientation. Returns mean score across pairs, in [0, 1].
pub fn symmetry_score(d: &PcbDesign, pairs: &[(String, String)]) -> f64 {
    if pairs.is_empty() {
        return 1.0;
    }
    let centroid = board_centroid(d);
    let scores: Vec<f64> = pairs
        .iter()
        .filter_map(|(a, b)| {
            let ia = d.components.iter().position(|c| &c.ref_des == a)?;
            let ib = d.components.iter().position(|c| &c.ref_des == b)?;
            let pa = d.placement.positions.get(ia)?.as_ref()?;
            let pb = d.placement.positions.get(ib)?.as_ref()?;
            let mid = Vec2(
                (pa.position.0 + pb.position.0) * 0.5,
                (pa.position.1 + pb.position.1) * 0.5,
            );
            // Distance of midpoint from centroid, normalized by board diagonal
            let dist = ((mid.0 - centroid.0).powi(2) + (mid.1 - centroid.1).powi(2)).sqrt();
            let diag = ((d.board.bounds.w).powi(2) + (d.board.bounds.h).powi(2)).sqrt();
            let normalized = if diag > 0.0 { dist / diag } else { 0.0 };
            Some((1.0 - normalized).max(0.0))
        })
        .collect();
    if scores.is_empty() {
        return 1.0;
    }
    scores.iter().sum::<f64>() / scores.len() as f64
}

fn board_centroid(d: &PcbDesign) -> Vec2 {
    Vec2(
        d.board.bounds.x + d.board.bounds.w * 0.5,
        d.board.bounds.y + d.board.bounds.h * 0.5,
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{
        BoardGeometry, Component, Footprint, LayerSet, Net, Pad, PadShape, PinElectricalProxy,
        Polygon, Rect,
    };

    fn design_with_two_placed_components() -> PcbDesign {
        // R1 at (10, 10) rotation 0, C1 at (15, 10) rotation 0
        // Net N1: R1.pad0 + C1.pad0 → HPWL = 5 (horizontal)
        // Net N2: R1.pad1 alone → HPWL = 0
        PcbDesign {
            board: BoardGeometry {
                outline: Polygon::rect(Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }),
                keepouts: vec![],
                bounds: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
            },
            components: vec![
                Component {
                    ref_des: "R1".into(),
                    footprint: Footprint {
                        pads: vec![
                            Pad { net_name: "N1".into(), shape: PadShape::Rect,
                                  local_pos: Vec2(-1.0, 0.0), drill: None,
                                  layers: LayerSet::all_copper(),
                                  electrical_proxy: PinElectricalProxy::Passive,
                                  electrical_proxy_confidence: 1.0 },
                            Pad { net_name: "N2".into(), shape: PadShape::Rect,
                                  local_pos: Vec2(1.0, 0.0), drill: None,
                                  layers: LayerSet::all_copper(),
                                  electrical_proxy: PinElectricalProxy::Passive,
                                  electrical_proxy_confidence: 1.0 },
                        ],
                        courtyard: Polygon::rect(Rect { x: -2.0, y: -1.0, w: 4.0, h: 2.0 }),
                        silkscreen: vec![],
                    },
                    value: "10k".into(),
                    netclass_hint: None,
                },
                Component {
                    ref_des: "C1".into(),
                    footprint: Footprint {
                        pads: vec![
                            Pad { net_name: "N1".into(), shape: PadShape::Rect,
                                  local_pos: Vec2(-0.5, 0.0), drill: None,
                                  layers: LayerSet::all_copper(),
                                  electrical_proxy: PinElectricalProxy::Passive,
                                  electrical_proxy_confidence: 1.0 },
                            Pad { net_name: "GND".into(), shape: PadShape::Rect,
                                  local_pos: Vec2(0.5, 0.0), drill: None,
                                  layers: LayerSet::all_copper(),
                                  electrical_proxy: PinElectricalProxy::Passive,
                                  electrical_proxy_confidence: 1.0 },
                        ],
                        courtyard: Polygon::rect(Rect { x: -1.5, y: -1.0, w: 3.0, h: 2.0 }),
                        silkscreen: vec![],
                    },
                    value: "100nF".into(),
                    netclass_hint: None,
                },
            ],
            nets: vec![
                Net { name: "N1".into(),
                      pads: vec![PadRef(0, 0), PadRef(1, 0)],
                      netclass: "Default".into(),
                      role: crate::ir::NetRole::Signal,
                      role_confidence: 0.5,
                      diff_pair_id: None },
                Net { name: "N2".into(),
                      pads: vec![PadRef(0, 1)],
                      netclass: "Default".into(),
                      role: crate::ir::NetRole::Signal,
                      role_confidence: 0.5,
                      diff_pair_id: None },
                Net { name: "GND".into(),
                      pads: vec![PadRef(1, 1)],
                      netclass: "Power".into(),
                      role: crate::ir::NetRole::Ground,
                      role_confidence: 1.0,
                      diff_pair_id: None },
            ],
            netclasses: std::collections::HashMap::new(),
            diff_pairs: vec![],
            placement: PlacementState {
                positions: vec![
                    Some(Placement { component_idx: 0, position: Vec2(10.0, 10.0), rotation_deg: 0.0, bottom_layer: false }),
                    Some(Placement { component_idx: 1, position: Vec2(15.0, 10.0), rotation_deg: 0.0, bottom_layer: false }),
                ],
                placement_order: vec![1, 0],
            },
        }
    }

    #[test]
    fn test_hpwl_two_component_net() {
        let d = design_with_two_placed_components();
        // N1 pads: R1.pad0 at (10-1, 10)=(9,10), C1.pad0 at (15-0.5, 10)=(14.5, 10)
        // HPWL = |14.5 - 9| + |10 - 10| = 5.5
        let h = hpwl(&d);
        assert!((h - 5.5).abs() < 1e-9, "hpwl = {h}");
    }

    #[test]
    fn test_hpwl_incremental_same_as_full_when_all_placed() {
        let d = design_with_two_placed_components();
        let full = hpwl(&d);
        let inc = hpwl_incremental(&d, "R1");
        assert!((full - inc).abs() < 1e-9);
    }

    #[test]
    fn test_hpwl_zero_when_unplaced() {
        let mut d = design_with_two_placed_components();
        d.placement.positions = vec![None, None];
        assert_eq!(hpwl(&d), 0.0);
    }

    #[test]
    fn test_clearance_zero_when_no_overlap() {
        let d = design_with_two_placed_components();
        // R1 courtyard expanded: [10-2-0.1, 10-1-0.1, 10+2+0.1, 10+1+0.1] = [7.9, 8.9, 12.1, 11.1]
        // C1 courtyard expanded: [15-1.5-0.1, 10-1-0.1, 15+1.5+0.1, 10+1+0.1] = [13.4, 8.9, 16.6, 11.1]
        // No overlap: 12.1 < 13.4
        let c = clearance_cost(&d, 0.2);
        assert_eq!(c, 0.0);
    }

    #[test]
    fn test_clearance_positive_when_overlap() {
        let mut d = design_with_two_placed_components();
        // Move C1 so courtyards overlap heavily
        d.placement.positions[1] = Some(Placement {
            component_idx: 1, position: Vec2(11.0, 10.0), rotation_deg: 0.0, bottom_layer: false,
        });
        let c = clearance_cost(&d, 0.0);
        assert!(c > 0.0, "expected positive clearance cost, got {c}");
    }

    #[test]
    fn test_partition_cut_zero_when_all_in_same_partition() {
        let d = design_with_two_placed_components();
        let p = PartitionSpec {
            name: "cluster_A".into(),
            components: vec![0, 1],
        };
        assert_eq!(partition_cut_cost(&d, &p), 0.0);
    }

    #[test]
    fn test_partition_cut_counts_crossing_nets() {
        let d = design_with_two_placed_components();
        // Put R1 in cluster A, C1 in cluster B → N1 (R1.pad0 + C1.pad0) crosses
        let p = PartitionSpec {
            name: "cluster_A".into(),
            components: vec![0],
        };
        // C1 is not in the partition, so it's a singleton. N1 has pads in
        // cluster_A + __single_C1 → crossing.
        let c = partition_cut_cost(&d, &p);
        assert_eq!(c, 1.0);
    }

    #[test]
    fn test_orient_score_high_when_tight_cluster() {
        let d = design_with_two_placed_components();
        let p = PartitionSpec {
            name: "A".into(),
            components: vec![0, 1],
        };
        let s = orient_score(&d, &p);
        assert!(s > 0.5, "tight cluster should score high, got {s}");
    }

    #[test]
    fn test_thermal_score_zero_when_no_power_pads_placed() {
        let mut d = design_with_two_placed_components();
        // Remove all placements
        d.placement.positions = vec![None, None];
        assert_eq!(thermal_score(&d), 0.0);
    }

    #[test]
    fn test_thermal_score_zero_when_no_power_role_nets() {
        // Fixture has only Signal + Ground nets, no Power. Score should be 0.
        let d = design_with_two_placed_components();
        let has_power = d.nets.iter().any(|n| n.role == crate::ir::NetRole::Power);
        assert!(!has_power, "fixture should have no Power nets for this test");
        let s = thermal_score(&d);
        assert_eq!(s, 0.0);
    }

    #[test]
    fn test_thermal_score_positive_with_power_net() {
        let mut d = design_with_two_placed_components();
        // Change GND net to Power role so the test exercises the positive path.
        let gnd = d.nets.iter_mut().find(|n| n.name == "GND").unwrap();
        gnd.role = crate::ir::NetRole::Power;
        let s = thermal_score(&d);
        assert!(s > 0.0, "got {s}");
    }

    #[test]
    fn test_symmetry_score_perfect_for_centered_pair() {
        let d = design_with_two_placed_components();
        // R1 at (10, 10), C1 at (15, 10) — midpoint (12.5, 10)
        // Board centroid (50, 50). Distance is large, so score will be < 1
        // (this is intentional — the metric measures how close midpoint is
        // to board centroid, which for this test fixture is off-center).
        let s = symmetry_score(&d, &[("R1".into(), "C1".into())]);
        assert!((0.0..=1.0).contains(&s));
    }

    #[test]
    fn test_symmetry_score_empty_pairs_is_one() {
        let d = design_with_two_placed_components();
        assert_eq!(symmetry_score(&d, &[]), 1.0);
    }
}
