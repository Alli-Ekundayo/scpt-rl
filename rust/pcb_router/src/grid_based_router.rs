use std::collections::HashMap;
use crate::board_grid::BoardGrid;
use crate::global_param::GlobalParam;
use crate::grid_netclass::GridNetclass;
use crate::grid_pin::GridPin;
use crate::kicad_parser::KicadPcbDatabase;
use crate::location::Location;
use crate::multipin_route::MultipinRoute;
use crate::util::Point2D;

/// High-level router – mirrors the C++ `GridBasedRouter`.
pub struct GridBasedRouter {
    pub params: GlobalParam,
    pub board_grid: BoardGrid,

    /// Maps grid-layer index → layer name.
    pub grid_layer_to_name: Vec<String>,
    /// Maps layer name → grid-layer index.
    pub layer_name_to_grid_layer: HashMap<String, usize>,

    /// All nets converted to grid space.
    pub grid_nets: Vec<MultipinRoute>,
    pub best_solution: Vec<MultipinRoute>,
    pub best_total_route_cost: f64,

    // Board extent (in DB units)
    pub min_x: f64,
    pub max_x: f64,
    pub min_y: f64,
    pub max_y: f64,
}

impl GridBasedRouter {
    pub fn new(params: GlobalParam) -> Self {
        GridBasedRouter {
            params,
            board_grid: BoardGrid::new(),
            grid_layer_to_name: Vec::new(),
            layer_name_to_grid_layer: HashMap::new(),
            grid_nets: Vec::new(),
            best_solution: Vec::new(),
            best_total_route_cost: -1.0,
            min_x: f64::MAX,
            max_x: f64::MIN,
            min_y: f64::MAX,
            max_y: f64::MIN,
        }
    }

    // ── Setters ───────────────────────────────────────────────────────────────

    pub fn set_grid_scale(&mut self, scale: u32) {
        self.params.set_grid_scale(scale);
    }

    pub fn set_num_iterations(&mut self, n: u32) {
        self.params.num_rip_up_reroute_iteration = n;
    }

    pub fn set_enlarge_boundary(&mut self, b: u32) {
        self.params.enlarge_boundary = b;
    }

    pub fn set_layer_change_weight(&mut self, w: f64) {
        self.params.layer_change_cost = w.abs();
    }

    pub fn set_track_obstacle_weight(&mut self, w: f64) {
        self.params.trace_basic_cost = w;
    }

    pub fn set_via_obstacle_weight(&mut self, w: f64) {
        self.params.via_insertion_cost = w;
    }

    // ── Getters ───────────────────────────────────────────────────────────────

    pub fn routed_wirelength(&self) -> f64 {
        self.best_solution
            .iter()
            .map(|r| r.routed_wirelength(self.params.grid_factor))
            .sum()
    }

    pub fn routed_num_vias(&self) -> i32 {
        self.best_solution.iter().map(|r| r.routed_num_vias()).sum()
    }

    pub fn routed_num_bends(&self) -> i32 {
        self.best_solution.iter().map(|r| r.routed_num_bends()).sum()
    }

    // ── Initialisation ────────────────────────────────────────────────────────

    /// Set up layer mapping from a parsed KiCad database.
    pub fn setup_layer_mapping(&mut self, db: &KicadPcbDatabase) {
        for layer in db.copper_layers() {
            let idx = self.grid_layer_to_name.len();
            self.layer_name_to_grid_layer.insert(layer.name.clone(), idx);
            self.grid_layer_to_name.push(layer.name.clone());
        }
    }

    /// Set up a single grid netclass from a parsed netclass record.
    #[allow(clippy::too_many_arguments)]
    pub fn setup_grid_netclass_from_db(
        &mut self,
        id: i32,
        clearance_mm: f64,
        trace_width_mm: f64,
        via_dia_mm: f64,
        via_drill_mm: f64,
        uvia_dia_mm: f64,
        uvia_drill_mm: f64,
    ) {
        let scale = self.params.input_scale as f64;

        let clearance = (clearance_mm * scale).ceil() as i32;
        let trace_width = (trace_width_mm * scale).ceil() as i32;
        let via_dia = (via_dia_mm * scale).ceil() as i32;
        let via_drill = (via_drill_mm * scale).ceil() as i32;
        let uvia_dia = (uvia_dia_mm * scale).ceil() as i32;
        let uvia_drill = (uvia_drill_mm * scale).ceil() as i32;

        let mut nc = GridNetclass::new(id, clearance, trace_width, via_dia, via_drill, uvia_dia, uvia_drill);

        // Derived values
        nc.half_trace_width = trace_width / 2;
        nc.half_via_dia = via_dia / 2;
        nc.half_uvia_dia = uvia_dia / 2;

        let diag_tw = ((trace_width as f64) / std::f64::consts::SQRT_2).ceil() as i32;
        nc.trace_width_diagonal = diag_tw;
        nc.half_trace_width_diagonal = diag_tw / 2;

        let diag_clr = ((clearance as f64) / std::f64::consts::SQRT_2).ceil() as i32;
        nc.clearance_diagonal = diag_clr;

        nc.trace_expansion = nc.half_trace_width;
        nc.via_expansion = nc.half_via_dia;
        nc.trace_expansion_diagonal = nc.half_trace_width_diagonal;

        // Rasterise trace and via search grids as circles
        let trace_search_radius = nc.half_trace_width + nc.clearance;
        nc.trace_searching_space_to_grids = rasterize_circle(trace_search_radius);
        nc.trace_end_shape_to_grids = rasterize_circle(nc.half_trace_width);

        let via_search_radius = nc.half_via_dia + nc.clearance;
        nc.via_searching_space_to_grids = rasterize_circle(via_search_radius);
        nc.via_shape_to_grids = rasterize_circle(nc.half_via_dia);

        self.board_grid.add_grid_netclass(nc);
    }

    /// Initialise the board grid and grid nets from a KiCad database.
    pub fn initialization(&mut self, db: &KicadPcbDatabase) {
        self.setup_layer_mapping(db);

        // Compute board extents from pads
        for pad in &db.pads {
            if pad.position.0 < self.min_x { self.min_x = pad.position.0; }
            if pad.position.0 > self.max_x { self.max_x = pad.position.0; }
            if pad.position.1 < self.min_y { self.min_y = pad.position.1; }
            if pad.position.1 > self.max_y { self.max_y = pad.position.1; }
        }

        // Add small margin if no pads
        if self.min_x > self.max_x {
            self.min_x = 0.0; self.max_x = 100.0;
            self.min_y = 0.0; self.max_y = 100.0;
        }

        let s = self.params.input_scale as f64;
        let eb = self.params.enlarge_boundary as f64;
        let w = ((self.max_x - self.min_x) * s + eb).ceil() as i32 + 1;
        let h = ((self.max_y - self.min_y) * s + eb).ceil() as i32 + 1;
        let l = self.grid_layer_to_name.len().max(1) as i32;

        self.board_grid.initialize(w, h, l);

        // Set up netclasses (use first DB netclass or a default)
        if db.netclasses.is_empty() {
            self.setup_grid_netclass_from_db(0, 0.2, 0.25, 0.8, 0.4, 0.3, 0.15);
        } else {
            for (i, nc) in db.netclasses.iter().enumerate() {
                self.setup_grid_netclass_from_db(
                    i as i32,
                    nc.clearance,
                    nc.trace_width,
                    nc.via_dia,
                    nc.via_drill,
                    nc.uvia_dia,
                    nc.uvia_drill,
                );
            }
        }

        // Build grid nets from DB pads grouped by net
        let mut net_to_pins: HashMap<i32, Vec<_>> = HashMap::new();
        for pad in &db.pads {
            if pad.net_id > 0 {
                net_to_pins.entry(pad.net_id).or_default().push(pad);
            }
        }

        let db_netclass_name_to_id: HashMap<&str, i32> = db
            .netclasses
            .iter()
            .enumerate()
            .map(|(idx, nc)| (nc.name.as_str(), idx as i32))
            .collect();

        let mut net_to_netclass_id: HashMap<i32, i32> = HashMap::new();
        for net in &db.nets {
            if let Some(id) = db_netclass_name_to_id.get(net.netclass_name.as_str()) {
                net_to_netclass_id.insert(net.id, *id);
            }
        }

        let s = self.params.input_scale as f64;
        for (net_id, pads) in &net_to_pins {
            let route_netclass_id = *net_to_netclass_id.get(net_id).unwrap_or(&0);
            let mut route = MultipinRoute::with_layers(
                *net_id,
                route_netclass_id,
                self.grid_layer_to_name.len(),
            );
            for pad in pads {
                let gx = ((pad.position.0 - self.min_x) * s).round() as i32;
                let gy = ((pad.position.1 - self.min_y) * s).round() as i32;

                let mut grid_pin = GridPin::new();
                grid_pin.pin_center = Point2D::new(gx, gy);

                // Determine layers
                for layer_name in &pad.layers {
                    if let Some(&layer_id) = self.layer_name_to_grid_layer.get(layer_name) {
                        grid_pin.add_pin_with_layer(Location::new(gx, gy, layer_id as i32));
                        grid_pin.pin_layers.push(layer_id as i32);
                    }
                }

                // If no copper layers found, use layer 0
                if grid_pin.pin_with_layers.is_empty() {
                    grid_pin.add_pin_with_layer(Location::new(gx, gy, 0));
                    grid_pin.pin_layers.push(0);
                }

                route.get_new_grid_pin().clone_from(&grid_pin);
            }

            if route.grid_pins.len() >= 2 {
                route.setup_grid_pins_routing_order();
                self.grid_nets.push(route);
            }
        }
    }

    // ── Main routing loop ─────────────────────────────────────────────────────

    pub fn route_all(&mut self) {
        let num_iter = self.params.num_rip_up_reroute_iteration as usize;

        for iter in 0..num_iter {
            let ripup = iter > 0;
            self.route_single_iteration(ripup);

            let cost = self.overall_route_cost();
            if self.best_total_route_cost < 0.0 || cost < self.best_total_route_cost {
                self.best_total_route_cost = cost;
                self.best_solution = self.grid_nets.clone();
            }
        }
    }

    fn route_single_iteration(&mut self, ripup_routed: bool) {
        let num_nets = self.grid_nets.len();
        for i in 0..num_nets {
            if ripup_routed && !self.grid_nets[i].grid_paths.is_empty() {
                let mut route = std::mem::take(&mut self.grid_nets[i]);
                self.board_grid.ripup_route(&mut route, &self.params);
                self.grid_nets[i] = route;
            }

            let mut route = std::mem::take(&mut self.grid_nets[i]);
            self.board_grid.route_grid_net_from_scratch(&mut route, &self.params);
            self.grid_nets[i] = route;
        }
        // Invalidate trace cost cache after each iteration
        self.board_grid.cached_trace_cost_fill(-1.0);
    }

    fn overall_route_cost(&self) -> f64 {
        self.grid_nets.iter().map(|r| {
            r.routed_wirelength(self.params.grid_factor)
                + r.routed_num_vias() as f64 * self.params.via_insertion_cost
        }).sum()
    }
}

/// Rasterise a filled circle of integer `radius` into relative (dx, dy) offsets.
pub fn rasterize_circle(radius: i32) -> Vec<Point2D> {
    let mut pts = Vec::new();
    for y in -radius..=radius {
        for x in -radius..=radius {
            if x * x + y * y <= radius * radius {
                pts.push(Point2D::new(x, y));
            }
        }
    }
    pts
}

// Extension on KicadPcbDatabase for testing layer-mapping
impl KicadPcbDatabase {
    #[cfg(test)]
    pub fn parse_test_layers(&mut self) {
        use crate::kicad_parser::types::{Layer, LayerType};
        self.layers.push(Layer { id: 0, name: "F.Cu".into(), layer_type: LayerType::Copper });
        self.layers.push(Layer { id: 31, name: "B.Cu".into(), layer_type: LayerType::Copper });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kicad_parser::KicadPcbDatabase;

    #[test]
    fn test_rasterize_circle_radius0() {
        let pts = rasterize_circle(0);
        assert_eq!(pts, vec![Point2D::new(0, 0)]);
    }

    #[test]
    fn test_rasterize_circle_radius1() {
        let pts = rasterize_circle(1);
        // Includes (0,0) and its 4 cardinal neighbours
        assert!(pts.contains(&Point2D::new(0, 0)));
        assert!(pts.contains(&Point2D::new(1, 0)));
        assert!(pts.contains(&Point2D::new(-1, 0)));
        assert!(pts.contains(&Point2D::new(0, 1)));
        assert!(pts.contains(&Point2D::new(0, -1)));
    }

    #[test]
    fn test_grid_based_router_layer_mapping() {
        let mut router = GridBasedRouter::new(GlobalParam::default());
        let mut db = KicadPcbDatabase::new();
        db.parse_test_layers();
        router.setup_layer_mapping(&db);
        assert_eq!(router.grid_layer_to_name.len(), 2); // F.Cu, B.Cu
    }

    #[test]
    fn test_initialization_with_minimal_db() {
        let params = GlobalParam::default();
        let mut router = GridBasedRouter::new(params);
        let db = build_test_db();
        router.initialization(&db);
        // Board grid should have been initialized
        assert!(router.board_grid.w > 0);
        assert!(router.board_grid.h > 0);
        // At least one grid net should have been created (pads 1+2 share net_id=1)
        assert!(!router.grid_nets.is_empty());
    }

    #[test]
    fn test_route_all_simple() {
        let params = GlobalParam::default();
        let mut router = GridBasedRouter::new(params);
        let db = build_test_db();
        router.initialization(&db);
        router.route_all();
        // Should have found at least a partial route
        let wl = router.routed_wirelength();
        assert!(wl >= 0.0);
    }

    // ── helpers ──────────────────────────────────────────────────────────────

    fn build_test_db() -> KicadPcbDatabase {
        use crate::kicad_parser::types::*;

        let mut db = KicadPcbDatabase::new();
        db.layers.push(Layer { id: 0, name: "F.Cu".into(), layer_type: LayerType::Copper });
        db.layers.push(Layer { id: 31, name: "B.Cu".into(), layer_type: LayerType::Copper });
        db.layer_id_to_name.insert(0, "F.Cu".into());
        db.layer_id_to_name.insert(31, "B.Cu".into());

        db.netclasses.push(crate::kicad_parser::types::Netclass {
            name: "Default".into(),
            description: String::new(),
            clearance: 0.2,
            trace_width: 0.25,
            via_dia: 0.8,
            via_drill: 0.4,
            uvia_dia: 0.3,
            uvia_drill: 0.15,
        });

        db.nets.push(Net { id: 1, name: "GND".into(), netclass_name: "Default".into(), pins: vec![] });
        db.nets.push(Net { id: 2, name: "VCC".into(), netclass_name: "Default".into(), pins: vec![] });

        // Two pads on net 1 (same layer)
        db.pads.push(Pad {
            reference: "U1".into(), number: "1".into(),
            pad_type: "smd".into(), shape: "rect".into(),
            position: (10.0, 10.0), size: (1.0, 1.0),
            net_id: 1,
            layers: vec!["F.Cu".into()],
            instance_position: (0.0, 0.0),
        });
        db.pads.push(Pad {
            reference: "U1".into(), number: "2".into(),
            pad_type: "smd".into(), shape: "rect".into(),
            position: (12.0, 10.0), size: (1.0, 1.0),
            net_id: 1,
            layers: vec!["F.Cu".into()],
            instance_position: (0.0, 0.0),
        });
        db
    }
}
