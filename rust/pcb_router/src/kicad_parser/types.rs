//! Data types shared between the KiCad parser and the rest of the router.

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayerType {
    Copper,
    NonCopper,
}

#[derive(Debug, Clone)]
pub struct Layer {
    pub id: i32,
    pub name: String,
    pub layer_type: LayerType,
}

#[derive(Debug, Clone, Default)]
pub struct Netclass {
    pub name: String,
    pub description: String,
    pub clearance: f64,
    pub trace_width: f64,
    pub via_dia: f64,
    pub via_drill: f64,
    pub uvia_dia: f64,
    pub uvia_drill: f64,
}

#[derive(Debug, Clone, Default)]
pub struct Net {
    pub id: i32,
    pub name: String,
    pub netclass_name: String,
    /// Pin references connected to this net (populated during instance parsing).
    pub pins: Vec<String>,
}

#[derive(Debug, Clone, Default)]
pub struct Instance {
    /// Footprint reference (e.g. "R1").
    pub reference: String,
    /// Absolute board position (x, y) in mm.
    pub position: (f64, f64),
    /// Rotation in degrees.
    pub rotation: f64,
    /// Layer this instance is placed on.
    pub layer: String,
}

#[derive(Debug, Clone)]
pub struct Pad {
    /// Parent instance reference.
    pub reference: String,
    /// Pad number string (e.g. "1", "A1").
    pub number: String,
    /// "smd", "thru_hole", "np_thru_hole", etc.
    pub pad_type: String,
    /// "rect", "circle", "oval", "roundrect", "trapezoid".
    pub shape: String,
    /// Absolute position (x, y) in mm.
    pub position: (f64, f64),
    /// Width and height of the pad in mm.
    pub size: (f64, f64),
    /// Net this pad belongs to (-1 = unconnected).
    pub net_id: i32,
    /// Copper layer names this pad covers.
    pub layers: Vec<String>,
    /// Parent instance absolute position – used to compute absolute pad coords.
    pub instance_position: (f64, f64),
}

impl Default for Pad {
    fn default() -> Self {
        Pad {
            reference: String::new(),
            number: String::new(),
            pad_type: String::new(),
            shape: String::new(),
            position: (0.0, 0.0),
            size: (0.0, 0.0),
            net_id: -1, // -1 = unconnected (consistent with router's `> 0` filter)
            layers: Vec::new(),
            instance_position: (0.0, 0.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_type() {
        let l = Layer { id: 0, name: "F.Cu".into(), layer_type: LayerType::Copper };
        assert_eq!(l.layer_type, LayerType::Copper);
    }

    #[test]
    fn test_net_default() {
        let n = Net::default();
        assert_eq!(n.id, 0);
        assert!(n.name.is_empty());
    }

    #[test]
    fn test_pad_default() {
        let p = Pad::default();
        assert_eq!(p.net_id, -1);
        assert!(p.layers.is_empty());
    }
}
