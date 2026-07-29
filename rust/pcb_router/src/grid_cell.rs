#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[derive(Default)]
pub enum GridCellType {
    #[default]
    Vacant,
    Pad,
    Trace,
    Via,
    KeepOut,
    ViaForbidden,
    TargetPin,
}


/// Single cell in the 3-D routing grid.
#[derive(Debug, Clone, Default)]
pub struct GridCell {
    /// Cost accumulated by already-routed nets' traces / vias.
    pub base_cost: f32,
    /// Cost accumulated during the current A* wave.
    pub working_cost: f32,
    /// Number of direction changes (bends) to reach this cell.
    pub bending_cost: i32,
    /// Cached obstacle cost for traces (−1 = not cached).
    pub cached_trace_cost: f32,
    /// Cached obstacle cost for vias (−1 = not cached).
    pub cached_via_cost: f32,
    /// Index of the cell we came from during search (−1 = unset).
    pub came_from_id: i32,
    pub cell_type: GridCellType,
    pub num_traces: i32,
}

impl GridCell {
    pub fn new() -> Self {
        GridCell {
            base_cost: 0.0,
            working_cost: 0.0,
            bending_cost: 0,
            cached_trace_cost: -1.0,
            cached_via_cost: -1.0,
            came_from_id: -1,
            cell_type: GridCellType::Vacant,
            num_traces: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_cell_defaults() {
        let cell = GridCell::new();
        assert_eq!(cell.base_cost, 0.0);
        assert_eq!(cell.working_cost, 0.0);
        assert_eq!(cell.cached_trace_cost, -1.0);
        assert_eq!(cell.cached_via_cost, -1.0);
        assert_eq!(cell.came_from_id, -1);
        assert!(matches!(cell.cell_type, GridCellType::Vacant));
    }

    #[test]
    fn test_grid_cell_type() {
        let mut cell = GridCell::new();
        cell.cell_type = GridCellType::TargetPin;
        assert!(matches!(cell.cell_type, GridCellType::TargetPin));
        cell.cell_type = GridCellType::Vacant;
        assert!(matches!(cell.cell_type, GridCellType::Vacant));
    }
}
