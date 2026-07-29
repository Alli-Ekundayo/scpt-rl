use std::f32;
use crate::grid_cell::{GridCell, GridCellType};
use crate::grid_netclass::GridNetclass;
use crate::grid_pin::GridPin;
use crate::grid_path::GridPath;
use crate::location::{Location, LocationQueue};
use crate::multipin_route::MultipinRoute;
use crate::global_param::GlobalParam;
use crate::util::Point2D;

/// 3-D routing grid used by the A* router.
///
/// Mirrors the C++ `BoardGrid` class.
pub struct BoardGrid {
    pub w: i32,
    pub h: i32,
    pub l: i32,
    cells: Vec<GridCell>,
    pub grid_netclasses: Vec<GridNetclass>,
    current_grid_netclass_id: usize,
    current_targeted_pin_with_layers: Vec<Location>,
}

impl BoardGrid {
    pub fn new() -> Self {
        BoardGrid {
            w: 0,
            h: 0,
            l: 0,
            cells: Vec::new(),
            grid_netclasses: Vec::new(),
            current_grid_netclass_id: 0,
            current_targeted_pin_with_layers: Vec::new(),
        }
    }

    pub fn initialize(&mut self, w: i32, h: i32, l: i32) {
        self.w = w;
        self.h = h;
        self.l = l;
        let size = (w as i64 * h as i64 * l as i64) as usize;
        self.cells = vec![GridCell::new(); size];
        self.base_cost_fill(0.0);
    }

    pub fn add_grid_netclass(&mut self, nc: GridNetclass) {
        self.grid_netclasses.push(nc);
    }

    pub fn grid_netclass(&self, id: usize) -> &GridNetclass {
        &self.grid_netclasses[id]
    }

    // ── indexing ─────────────────────────────────────────────────────────────

    #[inline]
    pub fn location_to_id(&self, l: &Location) -> usize {
        (l.x as i64 + l.y as i64 * self.w as i64 + l.z as i64 * self.w as i64 * self.h as i64) as usize
    }

    pub fn id_to_location(&self, id: usize) -> Location {
        let id = id as i64;
        let wh = self.w as i64 * self.h as i64;
        let z = (id / wh) as i32;
        let rem = id % wh;
        let y = (rem / self.w as i64) as i32;
        let x = (rem % self.w as i64) as i32;
        Location::new(x, y, z)
    }

    #[inline]
    pub fn validate_location(&self, l: &Location) -> bool {
        l.x >= 0 && l.x < self.w
            && l.y >= 0 && l.y < self.h
            && l.z >= 0 && l.z < self.l
    }

    // ── base cost ────────────────────────────────────────────────────────────

    pub fn base_cost_fill(&mut self, value: f32) {
        for c in &mut self.cells { c.base_cost = value; }
    }

    pub fn base_cost_at(&self, l: &Location) -> f32 {
        self.cells[self.location_to_id(l)].base_cost
    }

    pub fn base_cost_set(&mut self, value: f32, l: &Location) {
        let id = self.location_to_id(l);
        self.cells[id].base_cost = value;
    }

    pub fn base_cost_add(&mut self, value: f32, l: &Location) {
        let id = self.location_to_id(l);
        self.cells[id].base_cost += value;
    }

    pub fn base_cost_add_shape(&mut self, value: f32, l: &Location, shape: &[Point2D]) {
        for rel in shape {
            let loc = Location::new(l.x + rel.x, l.y + rel.y, l.z);
            if self.validate_location(&loc) {
                let id = self.location_to_id(&loc);
                self.cells[id].base_cost += value;
            }
        }
    }

    // ── working cost ─────────────────────────────────────────────────────────

    pub fn working_cost_fill(&mut self, value: f32) {
        for c in &mut self.cells { c.working_cost = value; }
    }

    pub fn working_cost_at(&self, l: &Location) -> f32 {
        self.cells[self.location_to_id(l)].working_cost
    }

    pub fn working_cost_set(&mut self, value: f32, l: &Location) {
        let id = self.location_to_id(l);
        self.cells[id].working_cost = value;
    }

    // ── bending cost ─────────────────────────────────────────────────────────

    pub fn bending_cost_fill(&mut self, value: i32) {
        for c in &mut self.cells { c.bending_cost = value; }
    }

    pub fn bending_cost_at(&self, l: &Location) -> i32 {
        self.cells[self.location_to_id(l)].bending_cost
    }

    pub fn bending_cost_set(&mut self, value: i32, l: &Location) {
        let id = self.location_to_id(l);
        self.cells[id].bending_cost = value;
    }

    // ── cached trace cost ────────────────────────────────────────────────────

    pub fn cached_trace_cost_fill(&mut self, value: f32) {
        for c in &mut self.cells { c.cached_trace_cost = value; }
    }

    pub fn cached_trace_cost_at(&self, l: &Location) -> f32 {
        self.cells[self.location_to_id(l)].cached_trace_cost
    }

    pub fn cached_trace_cost_set(&mut self, value: f32, l: &Location) {
        let id = self.location_to_id(l);
        self.cells[id].cached_trace_cost = value;
    }

    // ── came-from id ─────────────────────────────────────────────────────────

    pub fn came_from_id_at(&self, l: &Location) -> i32 {
        self.cells[self.location_to_id(l)].came_from_id
    }

    pub fn came_from_id_set(&mut self, l: &Location, id: i32) {
        let idx = self.location_to_id(l);
        self.cells[idx].came_from_id = id;
    }

    pub fn clear_all_came_from_id(&mut self) {
        for c in &mut self.cells { c.came_from_id = -1; }
    }

    // ── cell type ────────────────────────────────────────────────────────────

    pub fn set_cell_type(&mut self, l: &Location, t: GridCellType) {
        let id = self.location_to_id(l);
        self.cells[id].cell_type = t;
    }

    pub fn cell_type_at(&self, l: &Location) -> &GridCellType {
        &self.cells[self.location_to_id(l)].cell_type
    }

    pub fn is_targeted_pin(&self, l: &Location) -> bool {
        matches!(self.cell_type_at(l), GridCellType::TargetPin)
    }

    pub fn set_targeted_pin(&mut self, l: &Location) {
        self.set_cell_type(l, GridCellType::TargetPin);
    }

    pub fn clear_targeted_pin(&mut self, l: &Location) {
        self.set_cell_type(l, GridCellType::Vacant);
    }

    pub fn set_targeted_pins(&mut self, pins: &[Location]) {
        for p in pins { self.set_targeted_pin(p); }
    }

    pub fn clear_targeted_pins(&mut self, pins: &[Location]) {
        for p in pins { self.clear_targeted_pin(p); }
    }

    pub fn is_via_forbidden(&self, l: &Location) -> bool {
        matches!(self.cell_type_at(l), GridCellType::ViaForbidden)
    }

    pub fn set_via_forbidden(&mut self, l: &Location) {
        self.set_cell_type(l, GridCellType::ViaForbidden);
    }

    // ── obstacle cost (trace / via) ───────────────────────────────────────────

    /// Sum of base costs for all cells in a shape relative to `l`.
    fn sized_trace_cost_at(&self, l: &Location, search_grids: &[Point2D]) -> f32 {
        let mut cost = 0.0_f32;
        for rel in search_grids {
            let loc = Location::new(l.x + rel.x, l.y + rel.y, l.z);
            if self.validate_location(&loc) {
                cost += self.base_cost_at(&loc);
            }
        }
        cost
    }

    pub fn add_pin_shape_obstacle_cost(
        &mut self,
        grid_pin: &GridPin,
        value: f32,
        to_base_cost: bool,
    ) {
        if to_base_cost {
            for loc in &grid_pin.pin_with_layers {
                for rel in &grid_pin.pin_shape_to_grids {
                    let target = Location::new(loc.x + rel.x, loc.y + rel.y, loc.z);
                    if self.validate_location(&target) {
                        self.base_cost_add(value, &target);
                    }
                }
            }
        }
    }

    // ── A* routing ───────────────────────────────────────────────────────────

    /// Route a net from scratch using A*.
    pub fn route_grid_net_from_scratch(
        &mut self,
        route: &mut MultipinRoute,
        params: &GlobalParam,
    ) {
        if route.grid_pins.len() < 2 {
            return;
        }

        let num_pins = route.grid_pins.len();
        self.current_grid_netclass_id = route.grid_netclass_id as usize;

        for pin_idx in 1..num_pins {
            // Set the target
            let target_locations: Vec<Location> = route.grid_pins[pin_idx].pin_with_layers.clone();
            self.current_targeted_pin_with_layers = target_locations.clone();
            self.set_targeted_pins(&target_locations);

            let mut final_end = Location::default();
            let mut final_cost = 0.0_f32;
            self.a_star_searching(route, pin_idx, &mut final_end, &mut final_cost, params);

            self.clear_targeted_pins(&target_locations);

            if final_cost < f32::MAX {
                let new_path = self.backtrack_to_grid_path(&final_end);
                route.grid_paths.push(new_path);
                // Add the route to base cost so subsequent sub-routes see it
                self.add_route_to_base_cost_simple(route, params);
            }
        }
    }

    fn a_star_searching(
        &mut self,
        route: &MultipinRoute,
        target_pin_idx: usize,
        final_end: &mut Location,
        final_cost: &mut f32,
        params: &GlobalParam,
    ) {
        self.working_cost_fill(f32::INFINITY);
        self.bending_cost_fill(0);

        let mut frontier: LocationQueue<Location> = LocationQueue::new();

        // Initialise from previously-routed paths (or from first pin)
        if route.grid_paths.is_empty() {
            // Start from pin 0
            for start in &route.grid_pins[0].pin_with_layers {
                self.initialize_location_to_frontier(*start, &mut frontier, params);
            }
        } else {
            // Start from all routed path points
            for gp in &route.grid_paths {
                for loc in &gp.segments {
                    self.initialize_location_to_frontier(*loc, &mut frontier, params);
                }
            }
            // Also add all connected pins
            for i in 0..target_pin_idx {
                for loc in &route.grid_pins[i].pin_with_layers {
                    self.initialize_location_to_frontier(*loc, &mut frontier, params);
                }
            }
        }

        *final_cost = f32::MAX;

        while let Some(current) = frontier.front().cloned() {
            if self.is_targeted_pin(&current) {
                *final_cost = frontier.front_key().unwrap_or(f32::MAX);
                *final_end = current;
                return;
            }
            frontier.pop();

            let neighbors = self.get_neighbors(&current, params);
            let current_cost = self.working_cost_at(&current);

            for (edge_cost, next) in neighbors {
                let new_cost = current_cost + edge_cost;
                let est_cost = self.get_a_star_estimated_cost_with_bending(&current, &next, params);
                let bend_cost = self.get_bending_cost_of_next(&current, &next);

                if new_cost + (bend_cost as f32)
                    < self.working_cost_at(&next) + (self.bending_cost_at(&next) as f32)
                {
                    self.working_cost_set(new_cost, &next);
                    self.bending_cost_set(bend_cost, &next);
                    self.came_from_id_set(&next, self.location_to_id(&current) as i32);
                    frontier.push(next, new_cost + est_cost + bend_cost as f32);
                }
            }
        }
    }

    fn initialize_location_to_frontier(
        &mut self,
        start: Location,
        frontier: &mut LocationQueue<Location>,
        params: &GlobalParam,
    ) {
        let est = self.get_a_star_estimated_cost(&start, params);
        self.working_cost_set(0.0, &start);
        self.came_from_id_set(&start, self.location_to_id(&start) as i32);
        frontier.push(start, est);
    }

    /// Octile-distance heuristic to the nearest targeted pin.
    fn get_a_star_estimated_cost(&self, l: &Location, params: &GlobalParam) -> f32 {
        let mut cost = f32::INFINITY;
        for target in &self.current_targeted_pin_with_layers {
            let dx = (l.x - target.x).abs() as f32;
            let dy = (l.y - target.y).abs() as f32;
            let min_d = dx.min(dy);
            let max_d = dx.max(dy);
            let c = params.diagonal_cost as f32 * min_d + (max_d - min_d);
            if c < cost { cost = c; }
        }
        cost
    }

    fn get_a_star_estimated_cost_with_bending(
        &self,
        current: &Location,
        next: &Location,
        params: &GlobalParam,
    ) -> f32 {
        let cur_id = self.location_to_id(current) as i32;
        let prev_id = self.came_from_id_at(current);
        let mut bending_penalty = 0.0_f32;

        if prev_id != cur_id {
            let prev = self.id_to_location(prev_id as usize);
            if prev.z == current.z
                && current.z == next.z
                && prev.x - current.x == current.x - next.x
                && prev.y - current.y == current.y - next.y
            {
                bending_penalty += 0.5;
            }
        } else {
            bending_penalty += 0.5;
        }

        let base = self.get_a_star_estimated_cost(next, params);
        (base - bending_penalty).max(0.0)
    }

    fn get_bending_cost_of_next(&self, current: &Location, next: &Location) -> i32 {
        let cur_cost = self.bending_cost_at(current);
        let cur_id = self.location_to_id(current) as i32;
        let prev_id = self.came_from_id_at(current);

        if prev_id != cur_id {
            let prev = self.id_to_location(prev_id as usize);
            if prev.z == current.z
                && current.z == next.z
                && prev.x - current.x == current.x - next.x
                && prev.y - current.y == current.y - next.y
            {
                return cur_cost; // same direction, no new bend
            }
            cur_cost + 1
        } else {
            cur_cost
        }
    }

    /// Enumerate the up-to-8 in-plane neighbours plus up/down-layer vias.
    fn get_neighbors(
        &self,
        l: &Location,
        params: &GlobalParam,
    ) -> Vec<(f32, Location)> {
        let nc = &self.grid_netclasses[self.current_grid_netclass_id];
        let trace_grids = &nc.trace_searching_space_to_grids;

        let mut ns = Vec::with_capacity(10);

        // In-plane 8-connected moves
        let deltas: &[(i32, i32, f32)] = &[
            (-1,  0, 1.0),
            ( 1,  0, 1.0),
            ( 0, -1, 1.0),
            ( 0,  1, 1.0),
            (-1, -1, params.diagonal_cost as f32),
            (-1,  1, params.diagonal_cost as f32),
            ( 1, -1, params.diagonal_cost as f32),
            ( 1,  1, params.diagonal_cost as f32),
        ];

        for &(dx, dy, base_cost) in deltas {
            let next = Location::new(l.x + dx, l.y + dy, l.z);
            if !self.validate_location(&next) { continue; }

            let cached = self.cached_trace_cost_at(&next);
            let obstacle_cost = if cached < 0.0 {
                let c = self.sized_trace_cost_at(&next, trace_grids);
                // We don't mutate here to keep &self; callers should pre-cache
                c
            } else {
                cached
            };
            ns.push((base_cost + obstacle_cost, next));
        }

        // Via moves (layer changes at same xy)
        if params.allow_via_for_routing {
            let via_grids = &nc.via_searching_space_to_grids;
            for dz in [-1_i32, 1_i32] {
                let next = Location::new(l.x, l.y, l.z + dz);
                if !self.validate_location(&next) { continue; }
                if self.is_via_forbidden(&next) { continue; }
                let via_cost = params.via_insertion_cost as f32
                    + self.sized_trace_cost_at(&next, via_grids);
                ns.push((via_cost, next));
            }
        }

        ns
    }

    /// Backtrack from `end` through `came_from_id` to build a `GridPath`.
    /// Safety-bounded to prevent infinite loops on cyclic came_from_id chains.
    fn backtrack_to_grid_path(&self, end: &Location) -> GridPath {
        let mut path = GridPath::new();
        let mut current = *end;
        let max_steps = self.cells.len(); // absolute upper bound

        for _ in 0..max_steps {
            path.segments.push(current);
            let id = self.came_from_id_at(&current) as usize;
            let prev = self.id_to_location(id);
            if prev == current { break; }
            current = prev;
        }
        path.segments.reverse();
        path
    }

    fn add_route_to_base_cost_simple(
        &mut self,
        route: &MultipinRoute,
        params: &GlobalParam,
    ) {
        let nc = &self.grid_netclasses[route.grid_netclass_id as usize];
        let trace_shape = nc.trace_end_shape_to_grids.clone();
        let trace_cost = params.trace_basic_cost as f32;

        for gp in &route.grid_paths {
            for loc in &gp.segments {
                self.base_cost_add_shape(trace_cost, loc, &trace_shape);
            }
        }
        self.cached_trace_cost_fill(-1.0);
    }

    /// Rip-up a route (remove its base-cost contribution).
    pub fn ripup_route(&mut self, route: &mut MultipinRoute, params: &GlobalParam) {
        let nc = &self.grid_netclasses[route.grid_netclass_id as usize];
        let trace_shape = nc.trace_end_shape_to_grids.clone();
        let trace_cost = params.trace_basic_cost as f32;

        for gp in &route.grid_paths {
            for loc in &gp.segments {
                self.base_cost_add_shape(-trace_cost, loc, &trace_shape);
            }
        }
        route.clear_grid_paths();
        // Re-invalidate trace cost cache
        self.cached_trace_cost_fill(-1.0);
    }
}

impl Default for BoardGrid {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid_netclass::GridNetclass;
    use crate::grid_pin::GridPin;
    use crate::util::Point2D;

    fn make_board(w: i32, h: i32, l: i32) -> BoardGrid {
        let mut bg = BoardGrid::new();
        bg.initialize(w, h, l);

        // Add a basic netclass with empty search grids
        let nc = GridNetclass::new(0, 2, 4, 8, 4, 6, 3);
        bg.add_grid_netclass(nc);
        bg
    }

    #[test]
    fn test_initialize() {
        let bg = make_board(10, 10, 2);
        assert_eq!(bg.w, 10);
        assert_eq!(bg.h, 10);
        assert_eq!(bg.l, 2);
        assert_eq!(bg.cells.len(), 200);
    }

    #[test]
    fn test_location_roundtrip() {
        let bg = make_board(5, 5, 3);
        let orig = Location::new(2, 3, 1);
        let id = bg.location_to_id(&orig);
        let back = bg.id_to_location(id);
        assert_eq!(orig, back);
    }

    #[test]
    fn test_base_cost_fill_and_set() {
        let mut bg = make_board(5, 5, 2);
        bg.base_cost_fill(1.0);
        let l = Location::new(2, 2, 0);
        assert!((bg.base_cost_at(&l) - 1.0).abs() < 1e-6);

        bg.base_cost_set(5.5, &l);
        assert!((bg.base_cost_at(&l) - 5.5).abs() < 1e-6);
    }

    #[test]
    fn test_working_cost_fill() {
        let mut bg = make_board(4, 4, 1);
        bg.working_cost_fill(f32::INFINITY);
        let l = Location::new(1, 1, 0);
        assert_eq!(bg.working_cost_at(&l), f32::INFINITY);
    }

    #[test]
    fn test_targeted_pin() {
        let mut bg = make_board(4, 4, 1);
        let loc = Location::new(2, 2, 0);
        assert!(!bg.is_targeted_pin(&loc));
        bg.set_targeted_pin(&loc);
        assert!(bg.is_targeted_pin(&loc));
        bg.clear_targeted_pin(&loc);
        assert!(!bg.is_targeted_pin(&loc));
    }

    #[test]
    fn test_validate_location() {
        let bg = make_board(5, 5, 2);
        assert!(bg.validate_location(&Location::new(0, 0, 0)));
        assert!(bg.validate_location(&Location::new(4, 4, 1)));
        assert!(!bg.validate_location(&Location::new(5, 0, 0)));
        assert!(!bg.validate_location(&Location::new(0, 0, 2)));
        assert!(!bg.validate_location(&Location::new(-1, 0, 0)));
    }

    #[test]
    fn test_simple_routing() {
        let params = GlobalParam::default();
        let mut bg = make_board(20, 20, 2);

        let mut route = MultipinRoute::with_netclass(1, 0);

        let mut pin_a = GridPin::new();
        pin_a.add_pin_with_layer(Location::new(2, 2, 0));
        pin_a.pin_center = Point2D::new(2, 2);

        let mut pin_b = GridPin::new();
        pin_b.add_pin_with_layer(Location::new(8, 2, 0));
        pin_b.pin_center = Point2D::new(8, 2);

        route.grid_pins = vec![pin_a, pin_b];

        bg.current_targeted_pin_with_layers = vec![Location::new(8, 2, 0)];
        bg.route_grid_net_from_scratch(&mut route, &params);

        // At least one path should have been found
        assert!(!route.grid_paths.is_empty(), "Should find a path");
        let vias = route.routed_num_vias();
        assert_eq!(vias, 0, "Same-layer route should have no vias");
    }

    #[test]
    fn test_a_star_estimated_cost() {
        let params = GlobalParam::default();
        let mut bg = make_board(10, 10, 1);
        bg.current_targeted_pin_with_layers = vec![Location::new(5, 0, 0)];

        let l = Location::new(0, 0, 0);
        let cost = bg.get_a_star_estimated_cost(&l, &params);
        // Octile distance from (0,0) to (5,0) = 5.0
        assert!((cost - 5.0).abs() < 1e-3, "cost = {}", cost);
    }
}
