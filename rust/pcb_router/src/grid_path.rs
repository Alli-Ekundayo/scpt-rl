use crate::location::Location;

/// A single routed path segment between two grid pins.
///
/// Internally stores both a compact *segments* list (waypoints only) and a
/// dense *locations* list (every individual grid step).
#[derive(Debug, Clone, Default)]
pub struct GridPath {
    /// Dense point list (every grid step along the route).
    pub locations: Vec<Location>,
    /// Compact waypoint list (start/end of each straight run plus vias).
    pub segments: Vec<Location>,
}

impl GridPath {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_location(&mut self, l: Location) {
        self.locations.push(l);
    }

    pub fn copy_locations_to_segments(&mut self) {
        self.segments = self.locations.clone();
    }

    /// Remove collinear intermediate points from `segments`.
    pub fn remove_redundant_points(&mut self) {
        if self.segments.len() <= 2 {
            return;
        }

        let mut keep = vec![true; self.segments.len()];

        for (i, window) in self.segments.windows(3).enumerate() {
            let prev = &window[0];
            let cur = &window[1];
            let next = &window[2];
            // Mark collinear (same direction) middle point for removal
            if cur.x - prev.x == next.x - cur.x
                && cur.y - prev.y == next.y - cur.y
            {
                keep[i + 1] = false;
            }
        }

        let mut pruned = Vec::with_capacity(self.segments.len());
        for (loc, &k) in self.segments.drain(..).zip(keep.iter()) {
            if k {
                pruned.push(loc);
            }
        }
        self.segments = pruned;
    }

    /// Expand compact segments back into a dense location list.
    pub fn transform_segments_to_locations(&mut self) {
        self.locations.clear();
        if self.segments.is_empty() {
            return;
        }
        self.locations.push(self.segments[0]);

        for i in 1..self.segments.len() {
            let prev = &self.segments[i - 1];
            let cur = &self.segments[i];

            if cur.z != prev.z {
                // Via
                self.locations.push(*cur);
            } else {
                // Step each axis independently toward cur — guarantees convergence
                // even for non-45-degree segments.
                let mut step = *prev;
                while step != *cur {
                    if step.x < cur.x { step.x += 1; } else if step.x > cur.x { step.x -= 1; }
                    if step.y < cur.y { step.y += 1; } else if step.y > cur.y { step.y -= 1; }
                    self.locations.push(step);
                }
            }
        }
    }

    /// Routed wire-length in database units.
    pub fn routed_wirelength(&self, grid_factor: f32) -> f64 {
        if self.segments.is_empty() {
            return 0.0;
        }
        let mut wl = 0.0_f64;
        for i in 1..self.segments.len() {
            let a = &self.segments[i - 1];
            let b = &self.segments[i];
            if b.x != a.x || b.y != a.y {
                wl += grid_factor as f64 * Location::distance_2d(a, b);
            }
        }
        wl
    }

    /// Number of vias in this path.
    pub fn routed_num_vias(&self) -> i32 {
        let mut count = 0;
        for i in 1..self.segments.len() {
            if self.segments[i].z != self.segments[i - 1].z {
                count += 1;
            }
        }
        count
    }

    /// Number of direction changes (bends) in this path.
    pub fn routed_num_bends(&self) -> i32 {
        if self.segments.len() < 3 {
            return 0;
        }
        let mut bends = 0;
        for i in 1..self.segments.len() - 1 {
            let prev = &self.segments[i - 1];
            let cur = &self.segments[i];
            let next = &self.segments[i + 1];
            if cur.z == prev.z && cur.z == next.z {
                let same_dir =
                    cur.x - prev.x == next.x - cur.x
                    && cur.y - prev.y == next.y - cur.y;
                if !same_dir {
                    bends += 1;
                }
            }
        }
        bends
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_path(locs: &[(i32, i32, i32)]) -> GridPath {
        let mut gp = GridPath::new();
        for &(x, y, z) in locs {
            gp.segments.push(Location::new(x, y, z));
        }
        gp
    }

    #[test]
    fn test_remove_redundant_collinear() {
        // Horizontal run: (0,0,0) -> (1,0,0) -> (2,0,0) -> (3,0,0)
        // Middle points should be removed.
        let mut gp = make_path(&[(0,0,0),(1,0,0),(2,0,0),(3,0,0)]);
        gp.remove_redundant_points();
        assert_eq!(gp.segments, vec![Location::new(0,0,0), Location::new(3,0,0)]);
    }

    #[test]
    fn test_remove_redundant_preserves_bend() {
        // L-shaped: (0,0,0) -> (2,0,0) -> (2,2,0)
        let mut gp = make_path(&[(0,0,0),(2,0,0),(2,2,0)]);
        gp.remove_redundant_points();
        assert_eq!(gp.segments.len(), 3);
    }

    #[test]
    fn test_routed_num_vias() {
        let gp = make_path(&[(0,0,0),(0,0,1)]);
        assert_eq!(gp.routed_num_vias(), 1);
    }

    #[test]
    fn test_routed_num_bends() {
        // L: (0,0,0) -> (3,0,0) -> (3,3,0) — one bend
        let gp = make_path(&[(0,0,0),(3,0,0),(3,3,0)]);
        assert_eq!(gp.routed_num_bends(), 1);
    }

    #[test]
    fn test_routed_num_bends_straight() {
        let gp = make_path(&[(0,0,0),(5,0,0)]);
        assert_eq!(gp.routed_num_bends(), 0);
    }

    #[test]
    fn test_wirelength() {
        // Horizontal 4 units, grid_factor = 0.1 => wirelength ≈ 0.4
        let gp = make_path(&[(0,0,0),(4,0,0)]);
        let wl = gp.routed_wirelength(0.1);
        assert!((wl - 0.4).abs() < 1e-5, "wl = {}", wl);
    }

    #[test]
    fn test_transform_segments_to_locations() {
        // Two-point horizontal segment
        let mut gp = make_path(&[(0,0,0),(3,0,0)]);
        gp.transform_segments_to_locations();
        assert_eq!(gp.locations.len(), 4); // (0,0,0),(1,0,0),(2,0,0),(3,0,0)
        assert_eq!(gp.locations[0], Location::new(0,0,0));
        assert_eq!(gp.locations[3], Location::new(3,0,0));
    }
}
