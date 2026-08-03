use crate::location::Location;
use crate::util::{Point2D, Polygon};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[derive(Default)]
pub enum PinShape {
    #[default]
    Rect,
    RoundRect,
    Circle,
    Oval,
    Trapezoid,
}


/// A single component pin mapped onto the routing grid.
///
/// Mirrors `GridPin` from the C++ codebase.
#[derive(Debug, Clone, Default)]
pub struct GridPin {
    /// Grid-space 3-D locations this pin occupies (one per copper layer for
    /// through-hole pins, one for SMD pins).
    pub pin_with_layers: Vec<Location>,
    pub pin_layers: Vec<i32>,
    pub pin_center: Point2D,

    /// Shape grid-points relative to pin center.
    pub pin_shape_to_grids: Vec<Point2D>,

    pub pin_ll: Point2D,
    pub pin_ur: Point2D,
    pub pin_shape: PinShape,

    /// Boost-polygon equivalent – expanded/contracted bounding polygons.
    pub pin_polygon: Polygon,
    pub expanded_pin_polygon: Polygon,

    pub expanded_pin_ll: Point2D,
    pub expanded_pin_ur: Point2D,
    pub contracted_pin_ll: Point2D,
    pub contracted_pin_ur: Point2D,
}

impl GridPin {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_pin_with_layer(&mut self, loc: Location) {
        self.pin_with_layers.push(loc);
    }

    pub fn is_pin_layer(&self, layer_id: i32) -> bool {
        self.pin_with_layers.iter().any(|l| l.z == layer_id)
    }

    pub fn is_connected_to_pin(&self, l: &Location) -> bool {
        self.pin_with_layers.iter().any(|loc| loc == l)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_pin_default() {
        let gp = GridPin::new();
        assert!(gp.pin_with_layers.is_empty());
        assert!(matches!(gp.pin_shape, PinShape::Rect));
    }

    #[test]
    fn test_is_pin_layer() {
        let mut gp = GridPin::new();
        gp.add_pin_with_layer(Location::new(5, 5, 0));
        gp.add_pin_with_layer(Location::new(5, 5, 2));
        assert!(gp.is_pin_layer(0));
        assert!(gp.is_pin_layer(2));
        assert!(!gp.is_pin_layer(1));
    }

    #[test]
    fn test_is_connected_to_pin() {
        let mut gp = GridPin::new();
        gp.add_pin_with_layer(Location::new(3, 4, 1));
        assert!(gp.is_connected_to_pin(&Location::new(3, 4, 1)));
        assert!(!gp.is_connected_to_pin(&Location::new(3, 4, 0)));
    }
}
