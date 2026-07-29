use crate::error::{Error, Result};
use crate::models::{HoleRecord, KeepoutZoneRecord, Point2D};
use crate::parser::sexpr::{SexprNode, SexprParser};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct RawPcbData {
    pub version: Option<String>,
    pub generator: Option<String>,
    pub thickness_mm: f64,
    pub nets: HashMap<u32, String>,
    pub footprints: Vec<RawFootprint>,
    pub edge_cuts_segments: Vec<EdgeCutSegment>,
    pub keepout_zones: Vec<KeepoutZoneRecord>,
    pub holes: Vec<HoleRecord>,
}

#[derive(Debug, Clone)]
pub struct RawFootprint {
    pub ref_des: String,
    pub value: String,
    pub footprint_name: String,
    pub position: Point2D,
    pub rotation_deg: f64,
    pub layer: String,
    pub attributes: Vec<String>,
    pub pads: Vec<RawPad>,
    pub courtyard_points: Vec<Point2D>,
}

#[derive(Debug, Clone)]
pub struct RawPad {
    pub number: String,
    pub pad_type: String,
    pub shape: String,
    pub local_pos: Point2D,
    pub local_rotation_deg: f64,
    pub size_mm: Point2D,
    /// Drill size: for circular drills x == y; for oval drills x = minor, y = major axis.
    pub drill_mm: Option<Point2D>,
    pub net_id: u32,
    pub net_name: String,
}

#[derive(Debug, Clone)]
pub enum EdgeCutSegment {
    Line { start: Point2D, end: Point2D },
    Arc { start: Point2D, mid: Point2D, end: Point2D },
    Circle { center: Point2D, radius: f64 },
    Polygon { pts: Vec<Point2D> },
}

pub fn parse_pcb(content: &str) -> Result<RawPcbData> {
    let mut parser = SexprParser::new(content);
    let root = parser.parse_root()?;

    if root.name() != Some("kicad_pcb") {
        return Err(Error::InvalidSexpr(
            "Root node is not 'kicad_pcb'".to_string(),
        ));
    }

    let mut version = None;
    let mut generator = None;
    let mut thickness_mm = 1.6;
    let mut nets = HashMap::new();
    let mut footprints = Vec::new();
    let mut edge_cuts_segments = Vec::new();
    let mut keepout_zones = Vec::new();
    let mut holes = Vec::new();

    if let Some(v_node) = root.get_child_by_name("version") {
        if let Some(vf) = v_node.get_float_arg(0) {
            version = Some((vf as u64).to_string());
        } else if let Some(v) = v_node.get_string_arg(0) {
            if !v.is_empty() {
                version = Some(v.to_string());
            }
        }
    }

    if let Some(g_node) = root.get_child_by_name("generator") {
        if let Some(g) = g_node.get_string_arg(0) {
            generator = Some(g.to_string());
        }
    }

    if let Some(gen_node) = root.get_child_by_name("general") {
        if let Some(th_node) = gen_node.get_child_by_name("thickness") {
            if let Some(th) = th_node.get_float_arg(0) {
                thickness_mm = th;
            }
        }
    }

    // Extract nets
    for child in root.children() {
        if child.name() == Some("net") {
            if let (Some(id_f), Some(name)) = (child.get_float_arg(0), child.get_string_arg(1)) {
                nets.insert(id_f as u32, name.to_string());
            }
        }
    }

    // Extract footprints, graphics (Edge.Cuts), zones, holes
    for child in root.children() {
        match child.name() {
            Some("footprint") | Some("module") => {
                let fp = parse_footprint(child)?;
                // Check if any pads are NPTH holes
                for pad in &fp.pads {
                    if pad.pad_type == "np_thru_hole" {
                        let drill_diameter = pad
                            .drill_mm
                            .map(|d| d.x)
                            .unwrap_or(pad.size_mm.x);
                        holes.push(HoleRecord {
                            center: Point2D {
                                x: fp.position.x + pad.local_pos.x,
                                y: fp.position.y + pad.local_pos.y,
                            },
                            diameter_mm: drill_diameter,
                            hole_type: "mounting_hole".to_string(),
                        });
                    }
                }
                footprints.push(fp);
            }
            Some("gr_line")
                if is_edge_cut_layer(child) => {
                    if let (Some(start), Some(end)) = (
                        parse_xy_tuple(child.get_child_by_name("start")),
                        parse_xy_tuple(child.get_child_by_name("end")),
                    ) {
                        edge_cuts_segments.push(EdgeCutSegment::Line { start, end });
                    }
                }
            Some("gr_arc")
                if is_edge_cut_layer(child) => {
                    let start = parse_xy_tuple(child.get_child_by_name("start"));
                    let mid = parse_xy_tuple(child.get_child_by_name("mid"));
                    let end = parse_xy_tuple(child.get_child_by_name("end"));
                    if let (Some(s), Some(m), Some(e)) = (start, mid, end) {
                        edge_cuts_segments.push(EdgeCutSegment::Arc {
                            start: s,
                            mid: m,
                            end: e,
                        });
                    }
                }
            Some("gr_circle")
                if is_edge_cut_layer(child) => {
                    if let (Some(center), Some(end)) = (
                        parse_xy_tuple(child.get_child_by_name("center")),
                        parse_xy_tuple(child.get_child_by_name("end")),
                    ) {
                        let radius =
                            ((end.x - center.x).powi(2) + (end.y - center.y).powi(2)).sqrt();
                        edge_cuts_segments.push(EdgeCutSegment::Circle { center, radius });
                    }
                }
            Some("gr_poly")
                if is_edge_cut_layer(child) => {
                    if let Some(pts_node) = child.get_child_by_name("pts") {
                        let pts = parse_pts_list(pts_node);
                        if !pts.is_empty() {
                            edge_cuts_segments.push(EdgeCutSegment::Polygon { pts });
                        }
                    }
                }
            Some("via") => {
                if let (Some(at), Some(drill)) = (
                    parse_xy_tuple(child.get_child_by_name("at")),
                    child
                        .get_child_by_name("drill")
                        .and_then(|d| d.get_float_arg(0)),
                ) {
                    holes.push(HoleRecord {
                        center: at,
                        diameter_mm: drill,
                        hole_type: "via".to_string(),
                    });
                }
            }
            Some("zone") => {
                if let Some(keepout_node) = child.get_child_by_name("keepout") {
                    let layer = child
                        .get_child_by_name("layer")
                        .and_then(|l| l.get_string_arg(0))
                        .unwrap_or("all")
                        .to_string();
                    let polygon = child
                        .get_child_by_name("polygon")
                        .and_then(|p| p.get_child_by_name("pts"))
                        .map(parse_pts_list)
                        .unwrap_or_default();

                    let keepout_tracks = keepout_node
                        .get_child_by_name("tracks")
                        .and_then(|t| t.get_string_arg(0))
                        .map(|s| s != "allowed")
                        .unwrap_or(true);

                    let keepout_vias = keepout_node
                        .get_child_by_name("vias")
                        .and_then(|v| v.get_string_arg(0))
                        .map(|s| s != "allowed")
                        .unwrap_or(true);

                    let keepout_copperpour = keepout_node
                        .get_child_by_name("copperpour")
                        .and_then(|c| c.get_string_arg(0))
                        .map(|s| s != "allowed")
                        .unwrap_or(true);

                    keepout_zones.push(KeepoutZoneRecord {
                        layer,
                        polygon,
                        keepout_tracks,
                        keepout_vias,
                        keepout_copperpour,
                    });
                }
            }
            _ => {}
        }
    }

    Ok(RawPcbData {
        version,
        generator,
        thickness_mm,
        nets,
        footprints,
        edge_cuts_segments,
        keepout_zones,
        holes,
    })
}

fn parse_footprint(node: &SexprNode) -> Result<RawFootprint> {
    let footprint_name = node.get_string_arg(0).unwrap_or("Unknown").to_string();

    let mut ref_des = String::new();
    let mut value = String::new();
    let mut layer = "F.Cu".to_string();
    let mut position = Point2D { x: 0.0, y: 0.0 };
    let mut rotation_deg = 0.0;
    let mut attributes = Vec::new();
    let mut pads = Vec::new();
    let mut courtyard_points = Vec::new();

    if let Some(l_node) = node.get_child_by_name("layer") {
        if let Some(l) = l_node.get_string_arg(0) {
            layer = l.to_string();
        }
    }

    if let Some(at_node) = node.get_child_by_name("at") {
        if let Some(x) = at_node.get_float_arg(0) {
            position.x = x;
        }
        if let Some(y) = at_node.get_float_arg(1) {
            position.y = y;
        }
        if let Some(rot) = at_node.get_float_arg(2) {
            rotation_deg = rot;
        }
    }

    if let Some(attr_node) = node.get_child_by_name("attr") {
        for child in attr_node.children() {
            if let SexprNode::Atom(atom) = child {
                attributes.push(atom.as_str().to_string());
            }
        }
    }

    // Extract reference designator & value from properties or fp_text
    for child in node.children() {
        if child.name() == Some("property") {
            if let (Some(prop_name), Some(prop_val)) =
                (child.get_string_arg(0), child.get_string_arg(1))
            {
                if prop_name.eq_ignore_ascii_case("Reference") {
                    ref_des = prop_val.to_string();
                } else if prop_name.eq_ignore_ascii_case("Value") {
                    value = prop_val.to_string();
                }
            }
        } else if child.name() == Some("fp_text") {
            if let (Some(text_type), Some(text_val)) =
                (child.get_string_arg(0), child.get_string_arg(1))
            {
                if text_type == "reference" && ref_des.is_empty() {
                    ref_des = text_val.to_string();
                } else if text_type == "value" && value.is_empty() {
                    value = text_val.to_string();
                }
            }
        }
    }

    // Extract pads
    for child in node.children() {
        if child.name() == Some("pad") {
            pads.push(parse_pad(child));
        }
    }

    // Extract courtyard geometry (from fp_poly or fp_line on *.CrtYd)
    for child in node.children() {
        if child.name() == Some("fp_poly") {
            if is_courtyard_layer(child) {
                if let Some(pts_node) = child.get_child_by_name("pts") {
                    courtyard_points.extend(parse_pts_list(pts_node));
                }
            }
        } else if child.name() == Some("fp_line")
            && is_courtyard_layer(child) {
                if let (Some(start), Some(end)) = (
                    parse_xy_tuple(child.get_child_by_name("start")),
                    parse_xy_tuple(child.get_child_by_name("end")),
                ) {
                    courtyard_points.push(start);
                    courtyard_points.push(end);
                }
            }
    }

    if ref_des.is_empty() {
        ref_des = "U?".to_string();
    }

    Ok(RawFootprint {
        ref_des,
        value,
        footprint_name,
        position,
        rotation_deg,
        layer,
        attributes,
        pads,
        courtyard_points,
    })
}

fn parse_pad(node: &SexprNode) -> RawPad {
    let number = node.get_string_arg(0).unwrap_or("").to_string();
    let pad_type = node.get_string_arg(1).unwrap_or("smd").to_string();
    let shape = node.get_string_arg(2).unwrap_or("rect").to_string();

    let mut local_pos = Point2D { x: 0.0, y: 0.0 };
    let mut local_rotation_deg = 0.0;
    let mut size_mm = Point2D { x: 1.0, y: 1.0 };
    let mut drill_mm = None;
    let mut net_id = 0;
    let mut net_name = String::new();

    if let Some(at_node) = node.get_child_by_name("at") {
        if let Some(x) = at_node.get_float_arg(0) {
            local_pos.x = x;
        }
        if let Some(y) = at_node.get_float_arg(1) {
            local_pos.y = y;
        }
        if let Some(rot) = at_node.get_float_arg(2) {
            local_rotation_deg = rot;
        }
    }

    if let Some(size_node) = node.get_child_by_name("size") {
        if let Some(w) = size_node.get_float_arg(0) {
            size_mm.x = w;
        }
        if let Some(h) = size_node.get_float_arg(1) {
            size_mm.y = h;
        }
    }

    if let Some(drill_node) = node.get_child_by_name("drill") {
        // Check for oval drill: (drill oval 0.8 1.2)
        let first_arg = drill_node.get_string_arg(0);
        if first_arg == Some("oval") {
            // (drill oval minor major)
            let minor = drill_node.get_float_arg(1).unwrap_or(0.0);
            let major = drill_node.get_float_arg(2).unwrap_or(minor);
            drill_mm = Some(Point2D {
                x: minor,
                y: major,
            });
        } else if let Some(d) = drill_node.get_float_arg(0) {
            // Circular drill: (drill 0.8) — single diameter
            drill_mm = Some(Point2D { x: d, y: d });
        }
    }

    if let Some(net_node) = node.get_child_by_name("net") {
        if let Some(id_f) = net_node.get_float_arg(0) {
            net_id = id_f as u32;
        }
        if let Some(name) = net_node.get_string_arg(1) {
            net_name = name.to_string();
        }
    }

    RawPad {
        number,
        pad_type,
        shape,
        local_pos,
        local_rotation_deg,
        size_mm,
        drill_mm,
        net_id,
        net_name,
    }
}

fn is_edge_cut_layer(node: &SexprNode) -> bool {
    if let Some(l_node) = node.get_child_by_name("layer") {
        if let Some(layer) = l_node.get_string_arg(0) {
            return layer.eq_ignore_ascii_case("Edge.Cuts");
        }
    }
    false
}

fn is_courtyard_layer(node: &SexprNode) -> bool {
    if let Some(l_node) = node.get_child_by_name("layer") {
        if let Some(layer) = l_node.get_string_arg(0) {
            return layer.eq_ignore_ascii_case("F.CrtYd")
                || layer.eq_ignore_ascii_case("B.CrtYd");
        }
    }
    false
}

fn parse_xy_tuple(node: Option<&SexprNode>) -> Option<Point2D> {
    let node = node?;
    let x = node.get_float_arg(0)?;
    let y = node.get_float_arg(1)?;
    Some(Point2D { x, y })
}

fn parse_pts_list(node: &SexprNode) -> Vec<Point2D> {
    let mut pts = Vec::new();
    for child in node.children() {
        if child.name() == Some("xy") {
            if let (Some(x), Some(y)) = (child.get_float_arg(0), child.get_float_arg(1)) {
                pts.push(Point2D { x, y });
            }
        }
    }
    pts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_pcb_minimal() {
        let input = r#"
        (kicad_pcb (version 20240101) (generator pcbnew)
          (general (thickness 1.6))
          (net 0 "")
          (net 1 "VCC")
        )
        "#;
        let data = parse_pcb(input).unwrap();
        assert_eq!(data.version, Some("20240101".to_string()));
        assert_eq!(data.generator, Some("pcbnew".to_string()));
        assert!((data.thickness_mm - 1.6).abs() < 1e-9);
        assert_eq!(data.nets.len(), 2);
        assert_eq!(data.nets.get(&1), Some(&"VCC".to_string()));
        assert!(data.footprints.is_empty());
    }

    #[test]
    fn test_parse_pcb_with_footprint() {
        let input = r#"
        (kicad_pcb (version 20240101) (generator pcbnew)
          (net 1 "VCC")
          (footprint "Resistor_SMD:R_0805"
            (layer "F.Cu")
            (at 10.0 20.0 90)
            (property "Reference" "R1")
            (property "Value" "10k")
            (pad "1" smd rect (at -1.0 0) (size 1.0 1.2) (net 1 "VCC"))
            (pad "2" smd rect (at 1.0 0) (size 1.0 1.2) (net 0 ""))
          )
        )
        "#;
        let data = parse_pcb(input).unwrap();
        assert_eq!(data.footprints.len(), 1);

        let fp = &data.footprints[0];
        assert_eq!(fp.ref_des, "R1");
        assert_eq!(fp.value, "10k");
        assert_eq!(fp.footprint_name, "Resistor_SMD:R_0805");
        assert!((fp.position.x - 10.0).abs() < 1e-9);
        assert!((fp.position.y - 20.0).abs() < 1e-9);
        assert!((fp.rotation_deg - 90.0).abs() < 1e-9);
        assert_eq!(fp.layer, "F.Cu");
        assert_eq!(fp.pads.len(), 2);

        let pad1 = &fp.pads[0];
        assert_eq!(pad1.number, "1");
        assert_eq!(pad1.pad_type, "smd");
        assert_eq!(pad1.shape, "rect");
        assert!((pad1.local_pos.x - (-1.0)).abs() < 1e-9);
        assert_eq!(pad1.net_id, 1);
        assert_eq!(pad1.net_name, "VCC");
    }

    #[test]
    fn test_parse_pcb_oval_drill() {
        let input = r#"
        (kicad_pcb (version 20240101) (generator pcbnew)
          (footprint "Test:Test"
            (at 0 0)
            (pad "1" thru_hole rect (at 0 0) (size 2.0 3.0) (drill oval 0.8 1.2))
          )
        )
        "#;
        let data = parse_pcb(input).unwrap();
        let pad = &data.footprints[0].pads[0];
        let drill = pad.drill_mm.expect("oval drill should be captured");
        assert!((drill.x - 0.8).abs() < 1e-9, "minor axis should be 0.8");
        assert!((drill.y - 1.2).abs() < 1e-9, "major axis should be 1.2");
    }

    #[test]
    fn test_parse_pcb_circular_drill() {
        let input = r#"
        (kicad_pcb (version 20240101) (generator pcbnew)
          (footprint "Test:Test"
            (at 0 0)
            (pad "1" thru_hole circle (at 0 0) (size 2.0 2.0) (drill 1.0))
          )
        )
        "#;
        let data = parse_pcb(input).unwrap();
        let pad = &data.footprints[0].pads[0];
        let drill = pad.drill_mm.expect("drill should be captured");
        assert!((drill.x - 1.0).abs() < 1e-9, "x should be 1.0");
        assert!((drill.y - 1.0).abs() < 1e-9, "y should equal x for circular");
    }

    #[test]
    fn test_parse_pcb_edge_cuts() {
        let input = r#"
        (kicad_pcb (version 20240101) (generator pcbnew)
          (gr_line (start 0 0) (end 100 0) (layer "Edge.Cuts") (width 0.1))
          (gr_line (start 100 0) (end 100 80) (layer "Edge.Cuts") (width 0.1))
        )
        "#;
        let data = parse_pcb(input).unwrap();
        assert_eq!(data.edge_cuts_segments.len(), 2);
    }

    #[test]
    fn test_parse_pcb_invalid_root() {
        let input = "(kicad_sch (version 20240101))";
        let result = parse_pcb(input);
        assert!(result.is_err());
    }
}
