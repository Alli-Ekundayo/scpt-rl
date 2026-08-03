//! Parsers for KiCad file formats.
//!
//! Provides S-expression parsing, PCB file parsing, and schematic file parsing.
//! Also provides a conversion pipeline from raw parsed data to the high-level
//! `BoardOutput` model suitable for ML pipelines.

pub mod pcb;
pub mod sch;
pub mod sexpr;

pub use pcb::{parse_pcb, EdgeCutSegment, RawFootprint, RawPad, RawPcbData};
pub use sch::{parse_schematic, PinInfo, RawSchematicData, SymbolInstance};

use crate::geometry::{compute_footprint_bbox, extract_board_outline, transform_local_to_world};
use crate::models::*;
use crate::semantics::SemanticsExtractor;
use std::collections::HashMap;

/// Builds a high-level `BoardOutput` from raw parsed PCB data, optionally enriched
/// with schematic data for pin names and electrical types.
///
/// This is the main entry point for converting parsed KiCad data into the ML-pipeline
/// output format.
pub fn build_board_output(
    raw_pcb: &RawPcbData,
    raw_sch: Option<&RawSchematicData>,
    pcb_filename: &str,
    sch_filename: Option<&str>,
) -> BoardOutput {
    let semantics = SemanticsExtractor::new();

    // Build schematic pin lookup: lib_id -> (pin_number -> PinInfo)
    let sch_pin_lookup: Option<&HashMap<String, HashMap<String, sch::PinInfo>>> =
        raw_sch.map(|s| &s.lib_symbols);

    // Build a map from lib_id to value for schematic instances
    let sch_lib_id_for_ref: Option<HashMap<&str, &str>> = raw_sch.map(|s| {
        s.symbol_instances
            .iter()
            .map(|inst| (inst.ref_des.as_str(), inst.lib_id.as_str()))
            .collect()
    });

    // Build components
    let components: Vec<ComponentRecord> = raw_pcb
        .footprints
        .iter()
        .map(|fp| {
            let bbox = compute_footprint_bbox(fp);

            // Transform courtyard points to world coordinates
            let courtyard_polygon: Vec<Point2D> = fp
                .courtyard_points
                .iter()
                .map(|pt| transform_local_to_world(pt, &fp.position, fp.rotation_deg))
                .collect();

            // Convert pads to pins
            let pins: Vec<PinRecord> = fp
                .pads
                .iter()
                .map(|pad| {
                    let world_position =
                        transform_local_to_world(&pad.local_pos, &fp.position, fp.rotation_deg);

                    // Look up pin name and electrical type from schematic
                    let (pin_name, electrical_type) =
                        lookup_pin_info(&fp.ref_des, &pad.number, sch_pin_lookup, &sch_lib_id_for_ref);

                    PinRecord {
                        pin_number: pad.number.clone(),
                        pin_name,
                        electrical_type,
                        pad_type: pad.pad_type.clone(),
                        pad_shape: pad.shape.clone(),
                        local_position: pad.local_pos,
                        world_position,
                        size_mm: pad.size_mm,
                        drill_mm: pad.drill_mm,
                        net_id: pad.net_id,
                        net_name: pad.net_name.clone(),
                    }
                })
                .collect();

            ComponentRecord {
                ref_des: fp.ref_des.clone(),
                value: fp.value.clone(),
                footprint_name: fp.footprint_name.clone(),
                position: fp.position,
                orientation_deg: fp.rotation_deg,
                layer: fp.layer.clone(),
                bounding_box: bbox,
                courtyard_polygon,
                attributes: fp.attributes.clone(),
                pins,
            }
        })
        .collect();

    // Build nets
    // First, collect all pins by net_id
    let mut net_pins: HashMap<u32, Vec<PinReference>> = HashMap::new();
    for comp in &components {
        for pin in &comp.pins {
            if pin.net_id != 0 {
                net_pins
                    .entry(pin.net_id)
                    .or_default()
                    .push(PinReference {
                        ref_des: comp.ref_des.clone(),
                        pin_number: pin.pin_number.clone(),
                    });
            }
        }
    }

    let mut nets: Vec<NetRecord> = raw_pcb
        .nets
        .iter()
        .map(|(&id, name)| {
            let member_pins = net_pins.get(&id).cloned().unwrap_or_default();
            let fanout = member_pins.len();

            // Collect pin electrical types for classification
            let pin_types: Vec<ElectricalPinType> = components
                .iter()
                .flat_map(|c| &c.pins)
                .filter(|p| p.net_id == id)
                .map(|p| p.electrical_type.clone())
                .collect();

            let category = semantics.classify_net(name, "Default", &pin_types);

            NetRecord {
                id,
                name: name.clone(),
                net_class: "Default".to_string(),
                category,
                is_differential_pair: false,
                differential_pair_partner: None,
                fanout,
                member_pins,
            }
        })
        .collect();

    // Identify differential pairs
    semantics.identify_differential_pairs(&mut nets);

    // Sort nets by ID for deterministic output
    nets.sort_by_key(|n| n.id);

    // Build board geometry
    let (outline_polygon, outline_bbox) = extract_board_outline(&raw_pcb.edge_cuts_segments);

    // Compute overall bounding box from outline or from components
    let bounding_box = if outline_polygon.is_empty() {
        // Fall back to component bounding boxes
        let all_pts: Vec<Point2D> = components
            .iter()
            .flat_map(|c| {
                vec![
                    Point2D { x: c.bounding_box.min_x, y: c.bounding_box.min_y },
                    Point2D { x: c.bounding_box.max_x, y: c.bounding_box.max_y },
                ]
            })
            .collect();
        crate::geometry::compute_bounding_box(&all_pts)
    } else {
        outline_bbox
    };

    BoardOutput {
        metadata: BoardMetadata {
            pcb_file: pcb_filename.to_string(),
            sch_file: sch_filename.map(|s| s.to_string()),
            kicad_version: raw_pcb.version.clone(),
            generator: raw_pcb.generator.clone(),
            thickness_mm: raw_pcb.thickness_mm,
        },
        board_geometry: BoardGeometry {
            bounding_box,
            outline_polygon,
            holes: raw_pcb.holes.clone(),
            keepout_zones: raw_pcb.keepout_zones.clone(),
        },
        components,
        nets,
    }
}

/// Looks up pin name and electrical type from schematic library data.
fn lookup_pin_info(
    ref_des: &str,
    pin_number: &str,
    sch_pin_lookup: Option<&HashMap<String, HashMap<String, sch::PinInfo>>>,
    sch_lib_id_for_ref: &Option<HashMap<&str, &str>>,
) -> (Option<String>, ElectricalPinType) {
    let sch_pin_lookup = match sch_pin_lookup {
        Some(lookup) => lookup,
        None => return (None, ElectricalPinType::default()),
    };

    let lib_id_for_ref = match sch_lib_id_for_ref {
        Some(map) => map,
        None => return (None, ElectricalPinType::default()),
    };

    let lib_id = match lib_id_for_ref.get(ref_des) {
        Some(id) => *id,
        None => return (None, ElectricalPinType::default()),
    };

    let pin_map = match sch_pin_lookup.get(lib_id) {
        Some(map) => map,
        None => return (None, ElectricalPinType::default()),
    };

    match pin_map.get(pin_number) {
        Some(pin_info) => (
            Some(pin_info.name.clone()),
            pin_info.electrical_type.clone(),
        ),
        None => (None, ElectricalPinType::default()),
    }
}
