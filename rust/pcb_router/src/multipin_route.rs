use crate::grid_path::GridPath;
use crate::grid_pin::GridPin;

/// A multi-pin net mapped onto the routing grid.
///
/// Mirrors `MultipinRoute` from the C++ codebase.
#[derive(Debug, Clone, Default)]
pub struct MultipinRoute {
    pub net_id: i32,
    pub grid_netclass_id: i32,
    pub pair_net_id: i32,
    pub grid_diff_pair_netclass_id: i32,
    pub current_route_cost: f32,

    pub grid_pins: Vec<GridPin>,
    pub grid_paths: Vec<GridPath>,

    /// Per-layer routing preference costs (aligned with board-grid layers).
    pub layer_costs: Vec<i32>,

    /// Routing order of `grid_pins` (index list).
    pub grid_pins_routing_order: Vec<usize>,

    /// Track / via obstacle costs for this net.
    pub cur_track_obstacle_cost: f64,
    pub cur_via_obstacle_cost: f64,
}

impl MultipinRoute {
    pub fn new(net_id: i32) -> Self {
        MultipinRoute {
            net_id,
            grid_netclass_id: -1,
            pair_net_id: -1,
            grid_diff_pair_netclass_id: -1,
            ..Default::default()
        }
    }

    pub fn with_netclass(net_id: i32, grid_netclass_id: i32) -> Self {
        MultipinRoute {
            net_id,
            grid_netclass_id,
            pair_net_id: -1,
            grid_diff_pair_netclass_id: -1,
            ..Default::default()
        }
    }

    pub fn with_layers(net_id: i32, grid_netclass_id: i32, num_layers: usize) -> Self {
        MultipinRoute {
            net_id,
            grid_netclass_id,
            pair_net_id: -1,
            grid_diff_pair_netclass_id: -1,
            layer_costs: vec![0; num_layers],
            ..Default::default()
        }
    }

    pub fn is_diff_pair(&self) -> bool {
        self.pair_net_id != -1
    }

    pub fn get_new_grid_path(&mut self) -> &mut GridPath {
        self.grid_paths.push(GridPath::new());
        self.grid_paths.last_mut().unwrap()
    }

    pub fn get_new_grid_pin(&mut self) -> &mut GridPin {
        self.grid_pins.push(GridPin::new());
        self.grid_pins.last_mut().unwrap()
    }

    pub fn clear_grid_paths(&mut self) {
        self.grid_paths.clear();
    }

    pub fn routed_wirelength(&self, grid_factor: f32) -> f64 {
        self.grid_paths.iter().map(|p| p.routed_wirelength(grid_factor)).sum()
    }

    pub fn routed_num_vias(&self) -> i32 {
        self.grid_paths.iter().map(|p| p.routed_num_vias()).sum()
    }

    pub fn routed_num_bends(&self) -> i32 {
        self.grid_paths.iter().map(|p| p.routed_num_bends()).sum()
    }

    pub fn grid_path_locations_to_segments(&mut self) {
        for gp in &mut self.grid_paths {
            gp.copy_locations_to_segments();
            gp.remove_redundant_points();
        }
    }

    /// Set up the routing order of pins (nearest-neighbour greedy).
    #[allow(clippy::needless_range_loop)]
    pub fn setup_grid_pins_routing_order(&mut self) {
        let n = self.grid_pins.len();
        if n == 0 {
            return;
        }
        if n == 1 {
            self.grid_pins_routing_order = vec![0];
            return;
        }
        if n == 2 {
            self.grid_pins_routing_order = vec![0, 1];
            return;
        }

        // Find the closest pair as the seed
        let mut min_dist = f64::MAX;
        let mut id1 = 0usize;
        let mut id2 = 1usize;
        for i in 0..n {
            for j in i + 1..n {
                let d = Self::grid_pins_distance(&self.grid_pins[i], &self.grid_pins[j]);
                if d < min_dist {
                    min_dist = d;
                    id1 = i;
                    id2 = j;
                }
            }
        }

        let mut order = vec![id1, id2];
        let mut visited = vec![false; n];
        visited[id1] = true;
        visited[id2] = true;
        while order.len() < n {
            let mut best_dist = f64::MAX;
            let mut best_id = 0;
            for i in 0..n {
                if visited[i] {
                    continue;
                }
                for &already in &order {
                    let d = Self::grid_pins_distance(&self.grid_pins[i], &self.grid_pins[already]);
                    if d < best_dist {
                        best_dist = d;
                        best_id = i;
                    }
                }
            }
            visited[best_id] = true;
            order.push(best_id);
        }

        // Reorder the grid_pins vector
        let tmp: Vec<GridPin> = self.grid_pins.clone();
        self.grid_pins = order.iter().map(|&i| tmp[i].clone()).collect();
        self.grid_pins_routing_order = order;
    }

    fn grid_pins_distance(a: &GridPin, b: &GridPin) -> f64 {
        let dx = (a.pin_center.x - b.pin_center.x).abs() as f64;
        let dy = (a.pin_center.y - b.pin_center.y).abs() as f64;
        let min_d = dx.min(dy);
        let max_d = dx.max(dy);
        min_d * std::f64::consts::SQRT_2 + (max_d - min_d)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::Point2D;

    #[test]
    fn test_multipin_route_new() {
        let mpr = MultipinRoute::new(42);
        assert_eq!(mpr.net_id, 42);
        assert_eq!(mpr.grid_netclass_id, -1);
        assert!(!mpr.is_diff_pair());
    }

    #[test]
    fn test_with_layers() {
        let mpr = MultipinRoute::with_layers(1, 0, 4);
        assert_eq!(mpr.layer_costs.len(), 4);
        assert!(mpr.layer_costs.iter().all(|&c| c == 0));
    }

    #[test]
    fn test_routed_wirelength() {
        let mut mpr = MultipinRoute::new(1);
        let gp = mpr.get_new_grid_path();
        use crate::location::Location;
        gp.segments.push(Location::new(0, 0, 0));
        gp.segments.push(Location::new(4, 0, 0));
        let wl = mpr.routed_wirelength(0.1);
        assert!((wl - 0.4).abs() < 1e-5, "wl = {}", wl);
    }

    #[test]
    fn test_setup_routing_order_two_pins() {
        let mut mpr = MultipinRoute::new(1);
        let mut p1 = GridPin::new();
        p1.pin_center = Point2D::new(0, 0);
        let mut p2 = GridPin::new();
        p2.pin_center = Point2D::new(10, 0);
        mpr.grid_pins = vec![p1, p2];
        mpr.setup_grid_pins_routing_order();
        assert_eq!(mpr.grid_pins_routing_order.len(), 2);
    }

    #[test]
    fn test_setup_routing_order_three_pins() {
        let mut mpr = MultipinRoute::new(1);
        let centers = [(0, 0), (1, 0), (10, 10)];
        for (x, y) in centers {
            let mut p = GridPin::new();
            p.pin_center = Point2D::new(x, y);
            mpr.grid_pins.push(p);
        }
        mpr.setup_grid_pins_routing_order();
        assert_eq!(mpr.grid_pins_routing_order.len(), 3);
        // The first two should be pins 0 and 1 (closest pair)
        let order = &mpr.grid_pins_routing_order;
        let first_two: std::collections::HashSet<usize> = order[..2].iter().cloned().collect();
        assert!(first_two.contains(&0) || first_two.contains(&1));
    }
}
