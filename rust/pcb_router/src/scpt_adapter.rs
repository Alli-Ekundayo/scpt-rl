//! Adapter from SCPT's `PcbDesign` IR to the router's `KicadPcbDatabase`.
//!
//! The router's internal type (`KicadPcbDatabase`) is intentionally separate
//! from SCPT's IR: it carries a `kiutils_rs::PcbDocument` for round-trip
//! writes, plus a few router-specific fields (e.g. `new_segments`, `new_vias`).
//! This adapter bridges the two at the PyO3 boundary, so Python callers can
//! pass SCPT's JSON-encoded design and get a routed design back.
//!
//! v1 limitation: since the adapter doesn't have the original KiCad document,
//! the `doc` field is `None` and `write_to_file` will not work. A future pass
//! can thread the original document through the env if round-trip writes are
//! needed.

use std::collections::HashMap;

use crate::kicad_parser::types::{
    Instance, Layer, LayerType, Net, Netclass, Pad,
};
use crate::kicad_parser::KicadPcbDatabase;

/// A JSON-serializable representation of a SCPT `PcbDesign`, matching the
/// shape produced by `pcb_parser::ir::PcbDesign::to_json()`. We deserialize
/// into this rather than importing `pcb_parser::ir::PcbDesign` directly to
/// keep the two crates decoupled — pcb_router doesn't need to depend on
/// pcb_parser, just consume its JSON output.
#[derive(serde::Deserialize)]
struct ScptDesign {
    board: ScptBoard,
    components: Vec<ScptComponent>,
    nets: Vec<ScptNet>,
    netclasses: HashMap<String, ScptNetclass>,
    placement: ScptPlacement,
}

#[derive(serde::Deserialize)]
struct ScptBoard {
    #[allow(dead_code)]
    outline: serde_json::Value,
    #[allow(dead_code)]
    keepouts: Vec<serde_json::Value>,
    bounds: ScptRect,
}

#[derive(serde::Deserialize)]
struct ScptRect {
    x: f64,
    y: f64,
    #[allow(dead_code)]
    w: f64,
    #[allow(dead_code)]
    h: f64,
}

#[derive(serde::Deserialize)]
struct ScptComponent {
    ref_des: String,
    footprint: ScptFootprint,
    #[allow(dead_code)]
    value: String,
    #[allow(dead_code)]
    netclass_hint: Option<String>,
}

#[derive(serde::Deserialize)]
struct ScptFootprint {
    pads: Vec<ScptPad>,
    #[allow(dead_code)]
    courtyard: serde_json::Value,
    #[allow(dead_code)]
    silkscreen: Vec<serde_json::Value>,
}

#[derive(serde::Deserialize)]
struct ScptPad {
    #[allow(dead_code)]
    net_name: String,
    shape: String,
    local_pos: (f64, f64),
    #[allow(dead_code)]
    drill: Option<f64>,
    layers: ScptLayerSet,
    #[allow(dead_code)]
    electrical_proxy: String,
    #[allow(dead_code)]
    electrical_proxy_confidence: f64,
}

#[derive(serde::Deserialize)]
struct ScptLayerSet(Vec<String>);

#[derive(serde::Deserialize)]
struct ScptNet {
    name: String,
    pads: Vec<(usize, usize)>, // (component_idx, pad_idx)
    netclass: String,
    #[allow(dead_code)]
    role: String,
    #[allow(dead_code)]
    role_confidence: f64,
    #[allow(dead_code)]
    diff_pair_id: Option<String>,
}

#[derive(serde::Deserialize)]
struct ScptNetclass {
    name: String,
    clearance: f64,
    trace_width: f64,
}

#[derive(serde::Deserialize)]
struct ScptPlacement {
    positions: Vec<Option<ScptPlacementEntry>>,
    #[allow(dead_code)]
    placement_order: Vec<usize>,
}

#[derive(serde::Deserialize)]
struct ScptPlacementEntry {
    #[allow(dead_code)]
    component_idx: usize,
    position: (f64, f64),
    rotation_deg: f64,
    bottom_layer: bool,
}

// ---------------------------------------------------------------------------
// Coordinate helpers
// ---------------------------------------------------------------------------

fn rotate(v: (f64, f64), rotation_deg: f64) -> (f64, f64) {
    let rad = rotation_deg.to_radians();
    let c = rad.cos();
    let s = rad.sin();
    (v.0 * c - v.1 * s, v.0 * s + v.1 * c)
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Convert a SCPT `PcbDesign` (as JSON string) into a `KicadPcbDatabase` the
/// router can consume.
pub fn scpt_json_to_kicad_db(design_json: &str) -> Result<KicadPcbDatabase, String> {
    let d: ScptDesign = serde_json::from_str(design_json)
        .map_err(|e| format!("deserialize SCPT design: {e}"))?;

    // Build layers: standard 2-layer copper stack. The router's layer mapping
    // is set up from these.
    let layers = vec![
        Layer { id: 0, name: "F.Cu".into(), layer_type: LayerType::Copper },
        Layer { id: 31, name: "B.Cu".into(), layer_type: LayerType::Copper },
    ];
    let mut layer_id_to_name = HashMap::new();
    for l in &layers {
        layer_id_to_name.insert(l.id, l.name.clone());
    }

    // Build netclasses.
    let netclasses: Vec<Netclass> = d.netclasses.values().map(|nc| Netclass {
        name: nc.name.clone(),
        description: String::new(),
        clearance: nc.clearance,
        trace_width: nc.trace_width,
        via_dia: 0.8,
        via_drill: 0.4,
        uvia_dia: 0.3,
        uvia_drill: 0.1,
    }).collect();

    // Build instances + pads from components + placement.
    let mut instances = Vec::with_capacity(d.components.len());
    let mut pads = Vec::new();
    for (comp_idx, comp) in d.components.iter().enumerate() {
        let placement = d.placement.positions.get(comp_idx).and_then(|p| p.as_ref());
        let (pos_x, pos_y, rot, layer) = match placement {
            Some(p) => (
                p.position.0,
                p.position.1,
                p.rotation_deg,
                if p.bottom_layer { "B.Cu".to_string() } else { "F.Cu".to_string() },
            ),
            None => (0.0, 0.0, 0.0, "F.Cu".to_string()),
        };
        instances.push(Instance {
            reference: comp.ref_des.clone(),
            position: (pos_x, pos_y),
            rotation: rot,
            layer: layer.clone(),
        });
        for (pad_idx, pad) in comp.footprint.pads.iter().enumerate() {
            let local = pad.local_pos;
            let rotated = rotate(local, rot);
            let world = (rotated.0 + pos_x, rotated.1 + pos_y);
            // Find net id by searching SCPT nets for one that contains this pad.
            let net_id = d.nets.iter().position(|n| {
                n.pads.iter().any(|pr| pr.0 == comp_idx && pr.1 == pad_idx)
            }).map(|i| i as i32 + 1).unwrap_or(-1); // router uses net_id > 0 filter
            // Pad number: just the index as a string (SCPT doesn't carry pad
            // numbers directly; the router only uses them as unique labels).
            let pad_number = format!("{}", pad_idx + 1);
            pads.push(Pad {
                reference: comp.ref_des.clone(),
                number: pad_number,
                pad_type: "smd".into(), // v1: treat all pads as SMD
                shape: pad.shape.clone(),
                position: world,
                size: (0.5, 0.5), // v1: default pad size; SCPT doesn't carry this
                net_id,
                layers: pad.layers.0.clone().into_iter().collect::<Vec<_>>(),
                instance_position: (pos_x, pos_y),
            });
        }
    }

    // Build nets.
    let nets: Vec<Net> = d.nets.iter().enumerate().map(|(i, n)| {
        // Collect pad refs as "<ref_des>.<pad_number>" strings, matching the
        // router's expected format.
        let pin_labels: Vec<String> = n.pads.iter().filter_map(|(comp_idx, pad_idx)| {
            let comp = d.components.get(*comp_idx)?;
            Some(format!("{}.{}", comp.ref_des, pad_idx + 1))
        }).collect();
        Net {
            id: (i + 1) as i32,
            name: n.name.clone(),
            netclass_name: n.netclass.clone(),
            pins: pin_labels,
        }
    }).collect();

    let mut db = KicadPcbDatabase::new();
    db.layers = layers;
    db.layer_id_to_name = layer_id_to_name;
    db.netclasses = netclasses;
    db.nets = nets;
    db.instances = instances;
    db.pads = pads;
    Ok(db)
}
