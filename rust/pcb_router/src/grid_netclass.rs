use crate::util::Point2D;
use crate::incremental_search_grids::IncrementalSearchGrids;

/// Per-netclass routing constraints converted to grid units.
///
/// Mirrors `GridNetclass` from the C++ codebase.
#[derive(Debug, Clone)]
pub struct GridNetclass {
    pub id: i32,
    pub clearance: i32,
    pub trace_width: i32,
    pub via_dia: i32,
    pub via_drill: i32,
    pub uvia_dia: i32,
    pub uvia_drill: i32,

    // Derived
    pub half_trace_width: i32,
    pub trace_width_diagonal: i32,
    pub half_trace_width_diagonal: i32,
    pub half_via_dia: i32,
    pub half_uvia_dia: i32,
    pub clearance_diagonal: i32,

    pub trace_expansion: i32,
    pub trace_expansion_diagonal: i32,
    pub via_expansion: i32,

    /// Shared obstacle expansion (equivalent to the C++ static member).
    pub obstacle_expansion: i32,

    // Shape grids (relative offsets from center)
    pub via_shape_to_grids: Vec<Point2D>,
    pub trace_end_shape_to_grids: Vec<Point2D>,
    pub trace_searching_space_to_grids: Vec<Point2D>,
    pub via_searching_space_to_grids: Vec<Point2D>,

    // Incremental search grids
    pub trace_incremental_search_grids: IncrementalSearchGrids,
    pub via_incremental_search_grids: IncrementalSearchGrids,
}

impl GridNetclass {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: i32,
        clearance: i32,
        trace_width: i32,
        via_dia: i32,
        via_drill: i32,
        uvia_dia: i32,
        uvia_drill: i32,
    ) -> Self {
        GridNetclass {
            id,
            clearance,
            trace_width,
            via_dia,
            via_drill,
            uvia_dia,
            uvia_drill,
            half_trace_width: 0,
            trace_width_diagonal: 0,
            half_trace_width_diagonal: 0,
            half_via_dia: 0,
            half_uvia_dia: 0,
            clearance_diagonal: 0,
            trace_expansion: 0,
            trace_expansion_diagonal: 0,
            via_expansion: 0,
            obstacle_expansion: 0,
            via_shape_to_grids: Vec::new(),
            trace_end_shape_to_grids: Vec::new(),
            trace_searching_space_to_grids: Vec::new(),
            via_searching_space_to_grids: Vec::new(),
            trace_incremental_search_grids: IncrementalSearchGrids::new(),
            via_incremental_search_grids: IncrementalSearchGrids::new(),
        }
    }
}

/// Differential-pair netclass extends `GridNetclass` with IDs of the two
/// base netclasses. (C++ used inheritance; Rust uses composition.)
#[derive(Debug, Clone)]
pub struct GridDiffPairNetclass {
    pub base: GridNetclass,
    pub first_base_gnc_id: i32,
    pub second_base_gnc_id: i32,
}

impl GridDiffPairNetclass {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: i32,
        clearance: i32,
        trace_width: i32,
        via_dia: i32,
        via_drill: i32,
        uvia_dia: i32,
        uvia_drill: i32,
        first_base_gnc: i32,
        second_base_gnc: i32,
    ) -> Self {
        GridDiffPairNetclass {
            base: GridNetclass::new(id, clearance, trace_width, via_dia, via_drill, uvia_dia, uvia_drill),
            first_base_gnc_id: first_base_gnc,
            second_base_gnc_id: second_base_gnc,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_netclass_new() {
        let nc = GridNetclass::new(0, 5, 8, 16, 10, 12, 8);
        assert_eq!(nc.id, 0);
        assert_eq!(nc.clearance, 5);
        assert_eq!(nc.trace_width, 8);
        assert_eq!(nc.via_dia, 16);
        assert_eq!(nc.uvia_dia, 12);
        assert!(nc.via_shape_to_grids.is_empty());
    }

    #[test]
    fn test_grid_diff_pair_netclass() {
        let dp = GridDiffPairNetclass::new(2, 5, 20, 30, 18, 14, 10, 0, 1);
        assert_eq!(dp.base.id, 2);
        assert_eq!(dp.first_base_gnc_id, 0);
        assert_eq!(dp.second_base_gnc_id, 1);
    }
}
