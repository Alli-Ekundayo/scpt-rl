//! Data models for the KiCad parser output.
//!
//! These types represent the structured output of parsing KiCad PCB and schematic files,
//! designed for consumption by ML/RL pipelines.

use serde::{Deserialize, Serialize};

/// High-level parsed output representation of a KiCad PCB + Schematic design,
/// specifically structured for machine learning / RL policy models.
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BoardOutput {
    pub metadata: BoardMetadata,
    pub board_geometry: BoardGeometry,
    pub components: Vec<ComponentRecord>,
    pub nets: Vec<NetRecord>,
}

/// File-level metadata extracted from the KiCad PCB header.
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BoardMetadata {
    /// Path or name of the source `.kicad_pcb` file.
    pub pcb_file: String,
    /// Optional path to an associated `.kicad_sch` file used for pin enrichment.
    pub sch_file: Option<String>,
    /// KiCad file format version string.
    pub kicad_version: Option<String>,
    /// Generator tool (e.g. "pcbnew").
    pub generator: Option<String>,
    /// Board thickness in millimeters.
    pub thickness_mm: f64,
}

/// Physical board boundaries, keepout areas, and mechanical features.
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BoardGeometry {
    /// Overall axis-aligned bounding box of the board.
    pub bounding_box: BoundingBox,
    /// Ordered polygon representing the board outline (Edge.Cuts layer).
    pub outline_polygon: Vec<Point2D>,
    /// Drilled holes (mounting holes, vias).
    pub holes: Vec<HoleRecord>,
    /// Zones with keepout restrictions.
    pub keepout_zones: Vec<KeepoutZoneRecord>,
}

/// Axis-aligned bounding box in millimeters.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Default)]
pub struct BoundingBox {
    pub min_x: f64,
    pub min_y: f64,
    pub max_x: f64,
    pub max_y: f64,
}

impl BoundingBox {
    /// Width of the bounding box (clamped to >= 0).
    pub fn width(&self) -> f64 {
        (self.max_x - self.min_x).max(0.0)
    }

    /// Height of the bounding box (clamped to >= 0).
    pub fn height(&self) -> f64 {
        (self.max_y - self.min_y).max(0.0)
    }

    /// Area of the bounding box.
    pub fn area(&self) -> f64 {
        self.width() * self.height()
    }

    /// Returns true if the given point lies within this bounding box.
    pub fn contains(&self, pt: &Point2D) -> bool {
        pt.x >= self.min_x && pt.x <= self.max_x && pt.y >= self.min_y && pt.y <= self.max_y
    }
}

/// 2D point in millimeters (KiCad coordinate space).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Default)]
pub struct Point2D {
    pub x: f64,
    pub y: f64,
}

impl Point2D {
    /// Create a new point.
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// Euclidean distance to another point.
    pub fn distance_to(&self, other: &Point2D) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }
}

impl From<(f64, f64)> for Point2D {
    fn from((x, y): (f64, f64)) -> Self {
        Self { x, y }
    }
}

impl From<Point2D> for (f64, f64) {
    fn from(p: Point2D) -> Self {
        (p.x, p.y)
    }
}

/// A drilled hole on the board.
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HoleRecord {
    /// Center position of the hole.
    pub center: Point2D,
    /// Hole diameter in mm. For oval holes, the minor axis diameter.
    pub diameter_mm: f64,
    /// Hole type: "mounting_hole", "via", "pad".
    pub hole_type: String,
}

/// A copper zone with keepout restrictions.
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct KeepoutZoneRecord {
    /// Layer the zone is on (or "all").
    pub layer: String,
    /// Polygon vertices defining the zone boundary.
    pub polygon: Vec<Point2D>,
    /// Whether tracks are forbidden in this zone.
    pub keepout_tracks: bool,
    /// Whether vias are forbidden in this zone.
    pub keepout_vias: bool,
    /// Whether copper pour is forbidden in this zone.
    pub keepout_copperpour: bool,
}

/// Represents a single physical component footprint on the PCB.
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ComponentRecord {
    /// Reference designator (e.g. "R1", "U3").
    pub ref_des: String,
    /// Component value (e.g. "10k", "STM32F401").
    pub value: String,
    /// Footprint library identifier (e.g. "Resistor_SMD:R_0805").
    pub footprint_name: String,
    /// Position on the PCB in mm.
    pub position: Point2D,
    /// Rotation in degrees.
    pub orientation_deg: f64,
    /// Copper layer: "F.Cu" or "B.Cu".
    pub layer: String,
    /// Axis-aligned bounding box encompassing pads and courtyard.
    pub bounding_box: BoundingBox,
    /// Courtyard polygon vertices in world coordinates.
    pub courtyard_polygon: Vec<Point2D>,
    /// Footprint attributes: "smd", "through_hole", "board_only", etc.
    pub attributes: Vec<String>,
    /// Pins/pads belonging to this component.
    pub pins: Vec<PinRecord>,
}

/// Represents a single pad/pin belonging to a component.
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PinRecord {
    /// Pad number (e.g. "1", "2", "A3").
    pub pin_number: String,
    /// Schematic pin name (if enriched from schematic data).
    pub pin_name: Option<String>,
    /// Electrical type of the pin.
    pub electrical_type: ElectricalPinType,
    /// Pad type: "smd", "thru_hole", "np_thru_hole", "connect".
    pub pad_type: String,
    /// Pad shape: "circle", "rect", "oval", "roundrect", "trapezoid".
    pub pad_shape: String,
    /// Position relative to footprint origin.
    pub local_position: Point2D,
    /// Position in world PCB coordinates.
    pub world_position: Point2D,
    /// Pad size in mm (width, height).
    pub size_mm: Point2D,
    /// Drill size in mm. For circular drills, x == y. For oval drills, x = minor, y = major axis.
    pub drill_mm: Option<Point2D>,
    /// Net ID this pad is connected to (0 = unconnected).
    pub net_id: u32,
    /// Net name this pad is connected to.
    pub net_name: String,
}

/// Standard KiCad symbol electrical pin types.
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[derive(Default)]
pub enum ElectricalPinType {
    Input,
    Output,
    Bidirectional,
    TriState,
    Passive,
    Free,
    #[default]
    Unspecified,
    PowerIn,
    PowerOut,
    OpenCollector,
    OpenEmitter,
    NoConnect,
}


/// Categorized electrical semantics for nets.
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum NetCategory {
    Power,
    Ground,
    Signal,
    NoConnect,
}

/// Represents a net connecting multiple component pins.
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NetRecord {
    /// Net ID from the KiCad file.
    pub id: u32,
    /// Net name (e.g. "VCC", "SPI_MOSI").
    pub name: String,
    /// Net class name from the KiCad file.
    pub net_class: String,
    /// Categorized type of this net.
    pub category: NetCategory,
    /// Whether this net is part of a differential pair.
    pub is_differential_pair: bool,
    /// Name of the differential pair partner net, if any.
    pub differential_pair_partner: Option<String>,
    /// Number of pins connected to this net.
    pub fanout: usize,
    /// List of pins belonging to this net.
    pub member_pins: Vec<PinReference>,
}

/// Reference to a specific pin on a specific component.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct PinReference {
    /// Reference designator of the component (e.g. "R1").
    pub ref_des: String,
    /// Pin number on the component.
    pub pin_number: String,
}
