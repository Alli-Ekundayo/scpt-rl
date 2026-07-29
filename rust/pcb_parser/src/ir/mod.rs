//! Canonical IR types for PCB designs.
//!
//! This module defines the intermediate representation that the parser produces
//! and that every downstream consumer (geometry, Python env, GNN) reads.
//!
//! All types derive `Clone`, `Debug`, `serde::Serialize`, and `serde::Deserialize`
//! so that designs can be persisted, sent across FFI, and diffed in tests.

use std::collections::{BTreeSet, HashMap};

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Geometric primitives
// ---------------------------------------------------------------------------

/// 2D point / vector in millimetres.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Vec2(pub f64, pub f64);

/// Axis-aligned rectangle. `x`, `y` is the lower-left corner; `w`, `h` are extents.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Rect {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

impl Rect {
    /// Area of the rectangle (`w * h`).
    pub fn area(&self) -> f64 {
        self.w * self.h
    }
}

/// Arbitrary polygon as a list of vertices (in mm, ordered).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Polygon {
    pub points: Vec<Vec2>,
}

impl Polygon {
    /// Build a rectangular polygon from a `Rect` (four corners, CCW).
    pub fn rect(r: Rect) -> Polygon {
        Polygon {
            points: vec![
                Vec2(r.x, r.y),
                Vec2(r.x + r.w, r.y),
                Vec2(r.x + r.w, r.y + r.h),
                Vec2(r.x, r.y + r.h),
            ],
        }
    }

    /// Unsigned area of the polygon via the shoelace formula.
    pub fn area(&self) -> f64 {
        let n = self.points.len();
        if n < 3 {
            return 0.0;
        }
        let mut s = 0.0;
        for i in 0..n {
            let j = (i + 1) % n;
            s += self.points[i].0 * self.points[j].1;
            s -= self.points[j].0 * self.points[i].1;
        }
        (s * 0.5).abs()
    }
}

// ---------------------------------------------------------------------------
// Layers
// ---------------------------------------------------------------------------

/// A named copper / silkscreen / mask layer (e.g. `"F_Cu"`, `"B_SilkS"`).
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Layer(pub String);

/// Set of layers a pad / graphic appears on.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayerSet(pub BTreeSet<Layer>);

impl LayerSet {
    /// The canonical "all copper" set used for through-hole / PTH pads
    /// that connect on both sides: `{F_Cu, B_Cu}`.
    pub fn all_copper() -> LayerSet {
        let mut s = BTreeSet::new();
        s.insert(Layer("F_Cu".into()));
        s.insert(Layer("B_Cu".into()));
        LayerSet(s)
    }
}

// ---------------------------------------------------------------------------
// Pads & footprints
// ---------------------------------------------------------------------------

/// Shape of a pad's copper pour.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PadShape {
    Circle,
    Rect,
    Oval,
    RoundedRect,
}

/// Inferred electrical role of a pin, used as a prior before netlist
/// resolution. `Passive` covers resistors / capacitors / inductors.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PinElectricalProxy {
    Passive,
    Input,
    Output,
    Bidirectional,
    PowerIn,
    PowerOut,
    OpenCollector,
    OpenEmitter,
    NotConnected,
    Unknown,
}

/// A single pad on a footprint.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Pad {
    pub net_name: String,
    pub shape: PadShape,
    pub local_pos: Vec2,
    pub drill: Option<f64>,
    pub layers: LayerSet,
    pub electrical_proxy: PinElectricalProxy,
    pub electrical_proxy_confidence: f64,
}

/// A component footprint: its pads plus courtyard / silkscreen graphics.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Footprint {
    pub pads: Vec<Pad>,
    pub courtyard: Polygon,
    pub silkscreen: Vec<Polygon>,
}

// ---------------------------------------------------------------------------
// Components
// ---------------------------------------------------------------------------

/// A component on the board.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Component {
    pub ref_des: String,
    pub footprint: Footprint,
    pub value: String,
    pub netclass_hint: Option<String>,
}

// ---------------------------------------------------------------------------
// Nets
// ---------------------------------------------------------------------------

/// (component_idx, pad_idx) — a reference to a specific pad in the design.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PadRef(pub usize, pub usize);

/// High-level role a net plays, with an attached confidence (0..=1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NetRole {
    Signal,
    Clock,
    Power,
    Ground,
    Reference,
    Differential,
    Analog,
    Unknown,
}

/// A net — a named equivalence class of pads that must be connected.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Net {
    pub name: String,
    pub pads: Vec<PadRef>,
    pub netclass: String,
    pub role: NetRole,
    pub role_confidence: f64,
    pub diff_pair_id: Option<String>,
}

/// A named netclass with design-rule defaults.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Netclass {
    pub name: String,
    pub clearance: f64,
    pub trace_width: f64,
}

// ---------------------------------------------------------------------------
// Keepouts & board geometry
// ---------------------------------------------------------------------------

/// A region where copper / placement / routing is forbidden.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Keepout {
    pub polygon: Polygon,
    pub layers: LayerSet,
}

/// Board outline, keepouts, and overall bounds.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BoardGeometry {
    pub outline: Polygon,
    pub keepouts: Vec<Keepout>,
    pub bounds: Rect,
}

// ---------------------------------------------------------------------------
// Placement
// ---------------------------------------------------------------------------

/// Placement state of a single component (position + rotation + side).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Placement {
    pub component_idx: usize,
    pub position: Vec2,
    pub rotation_deg: f64,
    pub bottom_layer: bool,
}

/// Current state of component placement:
/// - `positions[i]` is `Some(Placement)` once component `i` has been placed,
/// - `placement_order` lists component indices in the order they should be
///   placed (typically area-descending so large parts anchor the layout first).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlacementState {
    pub positions: Vec<Option<Placement>>,
    pub placement_order: Vec<usize>,
}

/// A partition of components into named groups (e.g. analog / digital / power).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PartitionSpec {
    pub name: String,
    pub components: Vec<usize>,
}

// ---------------------------------------------------------------------------
// Top-level design
// ---------------------------------------------------------------------------

/// The canonical top-level IR: everything the parser knows about one board.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PcbDesign {
    pub board: BoardGeometry,
    pub components: Vec<Component>,
    pub nets: Vec<Net>,
    pub netclasses: HashMap<String, Netclass>,
    pub diff_pairs: Vec<String>,
    pub placement: PlacementState,
}

impl PcbDesign {
    /// Serialize this design to JSON. Useful as an escape hatch between Rust
    /// modules and for debugging / snapshot testing.
    pub fn to_json(&self) -> serde_json::Result<String> {
        serde_json::to_string(self)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_design() -> PcbDesign {
        // 2 components, 1 net with 2 pads, 10x10 board
        PcbDesign {
            board: BoardGeometry {
                outline: Polygon::rect(Rect { x: 0.0, y: 0.0, w: 10.0, h: 10.0 }),
                keepouts: vec![],
                bounds: Rect { x: 0.0, y: 0.0, w: 10.0, h: 10.0 },
            },
            components: vec![
                Component {
                    ref_des: "R1".into(),
                    footprint: Footprint {
                        pads: vec![
                            Pad { net_name: "N1".into(), shape: PadShape::Circle,
                                  local_pos: Vec2(-0.5, 0.0), drill: None,
                                  layers: LayerSet::all_copper(),
                                  electrical_proxy: PinElectricalProxy::Passive,
                                  electrical_proxy_confidence: 1.0 },
                            Pad { net_name: "N1".into(), shape: PadShape::Circle,
                                  local_pos: Vec2(0.5, 0.0), drill: None,
                                  layers: LayerSet::all_copper(),
                                  electrical_proxy: PinElectricalProxy::Passive,
                                  electrical_proxy_confidence: 1.0 },
                        ],
                        // R1 courtyard: area 1.28
                        courtyard: Polygon::rect(Rect { x: -0.8, y: -0.4, w: 1.6, h: 0.8 }),
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
                                  local_pos: Vec2(-0.3, 0.0), drill: None,
                                  layers: LayerSet::all_copper(),
                                  electrical_proxy: PinElectricalProxy::Passive,
                                  electrical_proxy_confidence: 1.0 },
                            Pad { net_name: "GND".into(), shape: PadShape::Rect,
                                  local_pos: Vec2(0.3, 0.0), drill: None,
                                  layers: LayerSet::all_copper(),
                                  electrical_proxy: PinElectricalProxy::Passive,
                                  electrical_proxy_confidence: 1.0 },
                        ],
                        // C1 courtyard: area 2.0 (larger, so placed first)
                        courtyard: Polygon::rect(Rect { x: -1.0, y: -0.5, w: 2.0, h: 1.0 }),
                        silkscreen: vec![],
                    },
                    value: "100nF".into(),
                    netclass_hint: None,
                },
            ],
            nets: vec![
                Net { name: "N1".into(),
                      pads: vec![PadRef(0,0), PadRef(0,1), PadRef(1,0)],
                      netclass: "Default".into(),
                      role: NetRole::Signal, role_confidence: 0.5,
                      diff_pair_id: None },
                Net { name: "GND".into(),
                      pads: vec![PadRef(1,1)],
                      netclass: "Power".into(),
                      role: NetRole::Ground, role_confidence: 1.0,
                      diff_pair_id: None },
            ],
            netclasses: {
                let mut m = std::collections::HashMap::new();
                m.insert("Default".into(), Netclass { name: "Default".into(), clearance: 0.15, trace_width: 0.25 });
                m.insert("Power".into(), Netclass { name: "Power".into(), clearance: 0.2, trace_width: 0.5 });
                m
            },
            diff_pairs: vec![],
            placement: PlacementState {
                positions: vec![None, None],
                placement_order: vec![1, 0],  // C1 (larger courtyard area) before R1
            },
        }
    }

    #[test]
    fn test_ir_construction() {
        let d = tiny_design();
        assert_eq!(d.components.len(), 2);
        assert_eq!(d.nets.len(), 2);
        assert_eq!(d.nets[0].pads.len(), 3);
    }

    #[test]
    fn test_to_json_roundtrip() {
        let d = tiny_design();
        let s = d.to_json().expect("json");
        assert!(s.contains("R1"));
        assert!(s.contains("N1"));
        // Re-parse via serde_json to validate well-formedness
        let _: serde_json::Value = serde_json::from_str(&s).expect("valid json");
    }
}
