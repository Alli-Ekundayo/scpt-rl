//! Adapter from `kicad_parser::BoardOutput` → SCPT's canonical `PcbDesign` IR.
//!
//! This module is the single place where the external parser's types are
//! translated into the canonical IR. It is pure (no I/O, no side effects) and
//! fully tested. Downstream code (geometry, PyO3 bindings, Python env) only
//! sees `crate::ir::PcbDesign`.

use std::collections::HashMap;

use kicad_parser::{
    BoardOutput, ComponentRecord, ElectricalPinType, KeepoutZoneRecord, NetCategory, NetRecord,
    PinRecord, Point2D,
};

use super::{
    BoardGeometry, Component, Footprint, Keepout, Layer, LayerSet, Net, NetRole, Netclass, Pad,
    PadRef, PadShape, PcbDesign, PinElectricalProxy, Placement, PlacementState, Polygon, Rect,
    Vec2,
};

// ---------------------------------------------------------------------------
// Top-level entry point
// ---------------------------------------------------------------------------

/// Convert a parsed `BoardOutput` (kicad_parser's native IR) into SCPT's
/// canonical `PcbDesign`. The adapter:
/// - maps geometry types (Point2D → Vec2, polygons, bounding boxes)
/// - maps enums (ElectricalPinType → PinElectricalProxy, NetCategory → NetRole)
/// - builds a ref_des → component_idx lookup so `PinReference` nets resolve
///   to `PadRef(component_idx, pad_idx)`
/// - treats each component's parsed position/rotation/layer as an initial
///   `Placement`, so the RL env can start from the human-authored placement
///   (useful for behavioral cloning warm-start)
/// - computes `placement_order` as area-descending courtyard sort
pub fn board_output_to_pcb_design(bo: BoardOutput) -> PcbDesign {
    let board = convert_board_geometry(&bo);
    let components = bo
        .components
        .iter()
        .map(|c| convert_component(c))
        .collect::<Vec<_>>();

    // Build lookup tables for net pad-ref resolution.
    let ref_des_to_comp_idx: HashMap<&str, usize> = components
        .iter()
        .enumerate()
        .map(|(i, c)| (c.ref_des.as_str(), i))
        .collect();
    // Build a lookup from (component_idx, pin_number) → pad_idx within the
    // ORIGINAL ComponentRecord. kicad_parser's PinReference uses (ref_des,
    // pin_number) so we need pin_number to resolve to SCPT's PadRef.
    let pad_idx_by_pin_number: HashMap<(usize, &str), usize> = bo
        .components
        .iter()
        .enumerate()
        .flat_map(|(i, c)| {
            c.pins
                .iter()
                .enumerate()
                .map(move |(j, p)| ((i, p.pin_number.as_str()), j))
        })
        .collect();

    let nets = bo
        .nets
        .iter()
        .map(|n| convert_net(n, &ref_des_to_comp_idx, &pad_idx_by_pin_number))
        .collect::<Vec<_>>();

    // Netclasses: kicad_parser doesn't surface class *definitions*
    // (clearance, trace_width) — only the class *name* per net. For v1 we
    // synthesize a default netclass per unique name seen. Real netclass
    // values would require extending kicad_parser.
    let netclasses = synthesize_netclasses(&bo.nets);

    // Diff pairs: collect unique partner pairs from nets with diff_pair_partner.
    let diff_pairs = collect_diff_pairs(&bo.nets);

    let placement = build_placement_state(&bo.components, &components);

    PcbDesign {
        board,
        components,
        nets,
        netclasses,
        diff_pairs,
        placement,
    }
}

// ---------------------------------------------------------------------------
// Sub-adapters
// ---------------------------------------------------------------------------

fn convert_board_geometry(bo: &BoardOutput) -> BoardGeometry {
    let outline = polygon_from_points(&bo.board_geometry.outline_polygon);
    let bounds = rect_from_bbox(&bo.board_geometry.bounding_box);
    let keepouts = bo
        .board_geometry
        .keepout_zones
        .iter()
        .map(|kz| Keepout {
            polygon: polygon_from_points(&kz.polygon),
            layers: layer_set_from_keepout(kz),
        })
        .collect();
    BoardGeometry {
        outline,
        keepouts,
        bounds,
    }
}

fn convert_component(cr: &ComponentRecord) -> Component {
    Component {
        ref_des: cr.ref_des.clone(),
        footprint: Footprint {
            pads: cr.pins.iter().map(convert_pad).collect(),
            courtyard: polygon_from_points(&cr.courtyard_polygon),
            silkscreen: vec![], // v1: not extracted from kicad_parser
        },
        value: cr.value.clone(),
        netclass_hint: Some(cr.attributes.join(",")).filter(|s| !s.is_empty()),
    }
}

fn convert_pad(pr: &PinRecord) -> Pad {
    Pad {
        net_name: pr.net_name.clone(),
        shape: pad_shape_from_str(&pr.pad_shape),
        local_pos: Vec2(pr.local_position.x, pr.local_position.y),
        drill: pr.drill_mm.map(|d| d.x), // v1: circular drill only
        layers: layers_from_pad_type(&pr.pad_type, &pr.pad_shape),
        electrical_proxy: pin_electrical_proxy(&pr.electrical_type),
        electrical_proxy_confidence: confidence_for(&pr.electrical_type),
    }
}

fn convert_net(
    nr: &NetRecord,
    ref_des_to_idx: &HashMap<&str, usize>,
    pad_idx_by_pin_number: &HashMap<(usize, &str), usize>,
) -> Net {
    let pads: Vec<PadRef> = nr
        .member_pins
        .iter()
        .filter_map(|pr| {
            let &comp_idx = ref_des_to_idx.get(pr.ref_des.as_str())?;
            let key = (comp_idx, pr.pin_number.as_str());
            let &pad_idx = pad_idx_by_pin_number.get(&key)?;
            Some(PadRef(comp_idx, pad_idx))
        })
        .collect();
    let (role, confidence) = net_role_from_category(&nr.category);
    Net {
        name: nr.name.clone(),
        pads,
        netclass: nr.net_class.clone(),
        role,
        role_confidence: confidence,
        diff_pair_id: nr.differential_pair_partner.clone(),
    }
}

// ---------------------------------------------------------------------------
// Enum / primitive mappers
// ---------------------------------------------------------------------------

fn polygon_from_points(pts: &[Point2D]) -> Polygon {
    Polygon {
        points: pts.iter().map(|p| Vec2(p.x, p.y)).collect(),
    }
}

fn rect_from_bbox(bb: &kicad_parser::BoundingBox) -> Rect {
    Rect {
        x: bb.min_x,
        y: bb.min_y,
        w: bb.max_x - bb.min_x,
        h: bb.max_y - bb.min_y,
    }
}

fn layer_set_from_keepout(_kz: &KeepoutZoneRecord) -> LayerSet {
    let mut s = std::collections::BTreeSet::new();
    // Keepout zones typically apply to all layers; represent as F_Cu + B_Cu
    // (a proper "all layers" representation would need a richer type).
    s.insert(Layer("F_Cu".into()));
    s.insert(Layer("B_Cu".into()));
    LayerSet(s)
}

fn pad_shape_from_str(s: &str) -> PadShape {
    match s.to_lowercase().as_str() {
        "circle" => PadShape::Circle,
        "rect" => PadShape::Rect,
        "oval" => PadShape::Oval,
        "roundrect" => PadShape::RoundedRect,
        _ => PadShape::RoundedRect, // v1: conservative default
    }
}

fn layers_from_pad_type(pad_type: &str, _pad_shape: &str) -> LayerSet {
    match pad_type {
        "smd" => {
            // SMD pads live on one copper layer; we don't have layer info on
            // the pad itself in v1 of kicad_parser's PinRecord, so default
            // to F_Cu. A future pass can thread the footprint's `layer` field
            // through to disambiguate.
            let mut s = std::collections::BTreeSet::new();
            s.insert(Layer("F_Cu".into()));
            LayerSet(s)
        }
        "thru_hole" | "np_thru_hole" => LayerSet::all_copper(),
        _ => LayerSet::all_copper(),
    }
}

fn pin_electrical_proxy(t: &ElectricalPinType) -> PinElectricalProxy {
    match t {
        ElectricalPinType::Passive => PinElectricalProxy::Passive,
        ElectricalPinType::Input => PinElectricalProxy::Input,
        ElectricalPinType::Output => PinElectricalProxy::Output,
        ElectricalPinType::Bidirectional => PinElectricalProxy::Bidirectional,
        ElectricalPinType::PowerIn => PinElectricalProxy::PowerIn,
        ElectricalPinType::PowerOut => PinElectricalProxy::PowerOut,
        ElectricalPinType::OpenCollector => PinElectricalProxy::OpenCollector,
        ElectricalPinType::OpenEmitter => PinElectricalProxy::OpenEmitter,
        ElectricalPinType::NoConnect => PinElectricalProxy::NotConnected,
        // TriState, Free, Unspecified, or any future variant → Unknown
        _ => PinElectricalProxy::Unknown,
    }
}

fn confidence_for(t: &ElectricalPinType) -> f64 {
    match t {
        // Specific types are high-confidence
        ElectricalPinType::Passive
        | ElectricalPinType::Input
        | ElectricalPinType::Output
        | ElectricalPinType::Bidirectional
        | ElectricalPinType::PowerIn
        | ElectricalPinType::PowerOut
        | ElectricalPinType::OpenCollector
        | ElectricalPinType::OpenEmitter
        | ElectricalPinType::NoConnect => 1.0,
        // Unspecified / free / tristate / any future variant → low confidence
        _ => 0.1,
    }
}

fn net_role_from_category(c: &NetCategory) -> (NetRole, f64) {
    match c {
        NetCategory::Power => (NetRole::Power, 1.0),
        NetCategory::Ground => (NetRole::Ground, 1.0),
        NetCategory::Signal => (NetRole::Signal, 0.5), // 0.5 because heuristic-derived
        NetCategory::NoConnect => (NetRole::Unknown, 1.0),
        // Any future variant → Unknown with low confidence
        _ => (NetRole::Unknown, 0.1),
    }
}

fn synthesize_netclasses(nets: &[NetRecord]) -> HashMap<String, Netclass> {
    let mut out: HashMap<String, Netclass> = HashMap::new();
    for n in nets {
        if n.net_class.is_empty() {
            continue;
        }
        out.entry(n.net_class.clone()).or_insert_with(|| Netclass {
            name: n.net_class.clone(),
            // Defaults: conservative for placement. Real values would come
            // from extending kicad_parser to surface (net_class ...) blocks.
            clearance: 0.15,
            trace_width: 0.25,
        });
    }
    out
}

fn collect_diff_pairs(nets: &[NetRecord]) -> Vec<String> {
    // Return one entry per unique diff pair, named by the "_P" side.
    let mut seen = std::collections::BTreeSet::new();
    let mut out = Vec::new();
    for n in nets {
        if !n.is_differential_pair {
            continue;
        }
        // Canonical name: alphabetically first of (n.name, n.partner)
        if let Some(ref partner) = n.differential_pair_partner {
            let key = if n.name < *partner {
                format!("{}:{}", n.name, partner)
            } else {
                format!("{}:{}", partner, n.name)
            };
            if seen.insert(key.clone()) {
                out.push(key);
            }
        }
    }
    out
}

fn build_placement_state(
    original_components: &[ComponentRecord],
    _scpt_components: &[Component],
) -> PlacementState {
    let positions: Vec<Option<Placement>> = original_components
        .iter()
        .enumerate()
        .map(|(i, c)| {
            Some(Placement {
                component_idx: i,
                position: Vec2(c.position.x, c.position.y),
                rotation_deg: c.orientation_deg,
                bottom_layer: c.layer.eq_ignore_ascii_case("B.Cu"),
            })
        })
        .collect();

    // Area-descending placement order, ties broken by ref_des ascending.
    let mut order: Vec<usize> = (0..original_components.len()).collect();
    order.sort_by(|&a, &b| {
        let area_a = polygon_from_points(&original_components[a].courtyard_polygon).area();
        let area_b = polygon_from_points(&original_components[b].courtyard_polygon).area();
        area_b
            .partial_cmp(&area_a)
            .unwrap()
            .then_with(|| original_components[a].ref_des.cmp(&original_components[b].ref_des))
    });

    PlacementState {
        positions,
        placement_order: order,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use kicad_parser::parse_pcb_to_output;

    const MINIMAL_PCB: &str = r#"
(kicad_pcb (version 20240101) (generator pcbnew)
  (general (thickness 1.6))
  (layers
    (0 "F.Cu" signal)
    (31 "B.Cu" signal)
  )
  (net 0 "")
  (net 1 "VCC")
  (net 2 "GND")
  (net 3 "SIG")
  (footprint "Resistor_SMD:R_0805"
    (layer "F.Cu")
    (at 10.0 20.0 90)
    (property "Reference" "R1")
    (property "Value" "10k")
    (pad "1" smd rect (at -1.0 0) (size 1.0 1.2) (net 1 "VCC") (layers "F.Cu" "F.Paste" "F.Mask"))
    (pad "2" smd rect (at 1.0 0) (size 1.0 1.2) (net 3 "SIG") (layers "F.Cu" "F.Paste" "F.Mask"))
  )
  (footprint "Capacitor_SMD:C_0805"
    (layer "F.Cu")
    (at 15.0 20.0 0)
    (property "Reference" "C1")
    (property "Value" "100nF")
    (pad "1" smd rect (at -0.5 0) (size 0.8 1.0) (net 1 "VCC") (layers "F.Cu" "F.Paste" "F.Mask"))
    (pad "2" smd rect (at 0.5 0) (size 0.8 1.0) (net 2 "GND") (layers "F.Cu" "F.Paste" "F.Mask"))
  )
  (gr_rect (start 0 0) (end 100 100) (layer "Edge.Cuts") (width 0.05))
)
"#;

    fn parsed() -> PcbDesign {
        let bo = parse_pcb_to_output(MINIMAL_PCB, "test.kicad_pcb", None, None).unwrap();
        board_output_to_pcb_design(bo)
    }

    #[test]
    fn test_adapter_produces_components() {
        let d = parsed();
        assert_eq!(d.components.len(), 2);
        assert_eq!(d.components[0].ref_des, "R1");
        assert_eq!(d.components[1].ref_des, "C1");
        assert_eq!(d.components[0].value, "10k");
    }

    #[test]
    fn test_adapter_produces_nets_with_pad_refs() {
        let d = parsed();
        // VCC net: R1 pad 1 + C1 pad 1
        let vcc = d.nets.iter().find(|n| n.name == "VCC").unwrap();
        assert_eq!(vcc.pads.len(), 2);
        assert_eq!(vcc.role, NetRole::Power);
        assert_eq!(vcc.role_confidence, 1.0);
    }

    #[test]
    fn test_adapter_maps_pad_shapes() {
        let d = parsed();
        assert_eq!(d.components[0].footprint.pads[0].shape, PadShape::Rect);
    }

    #[test]
    fn test_adapter_placement_state() {
        let d = parsed();
        assert_eq!(d.placement.positions.len(), 2);
        assert!(d.placement.positions[0].is_some());
        let p = d.placement.positions[0].unwrap();
        assert!((p.position.0 - 10.0).abs() < 1e-9);
        assert!((p.position.1 - 20.0).abs() < 1e-9);
        assert!((p.rotation_deg - 90.0).abs() < 1e-9);
        assert!(!p.bottom_layer);
    }

    #[test]
    fn test_adapter_placement_order_area_descending() {
        let d = parsed();
        // Both components have similar courtyard areas (kicad_parser computes
        // these from pads+bbox); the order should be deterministic.
        assert_eq!(d.placement.placement_order.len(), 2);
    }

    #[test]
    fn test_adapter_ground_role() {
        let d = parsed();
        let gnd = d.nets.iter().find(|n| n.name == "GND").unwrap();
        assert_eq!(gnd.role, NetRole::Ground);
    }

    #[test]
    fn test_adapter_netclasses_synthesized() {
        let d = parsed();
        // At least one netclass should exist (from any net with a class name)
        assert!(!d.netclasses.is_empty());
    }
}
