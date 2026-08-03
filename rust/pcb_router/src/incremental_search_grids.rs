use crate::util::Point2D;

/// Incremental search grid offsets for one netclass.
///
/// When routing traces or checking via clearance, the router keeps track of
/// which grid cells to *add* or *remove* from the "active" search window as
/// the trace advances in each of the 8 cardinal/diagonal directions.
#[derive(Debug, Clone, Default)]
pub struct IncrementalSearchGrids {
    // Horizontal / vertical directions
    pub add_l: Vec<Point2D>,
    pub ded_l: Vec<Point2D>,
    pub add_r: Vec<Point2D>,
    pub ded_r: Vec<Point2D>,
    pub add_f: Vec<Point2D>,
    pub ded_f: Vec<Point2D>,
    pub add_b: Vec<Point2D>,
    pub ded_b: Vec<Point2D>,

    // Diagonal directions (LB = left-back, RF = right-forward, …)
    pub add_lb: Vec<Point2D>,
    pub ded_lb: Vec<Point2D>,
    pub add_rb: Vec<Point2D>,
    pub ded_rb: Vec<Point2D>,
    pub add_lf: Vec<Point2D>,
    pub ded_lf: Vec<Point2D>,
    pub add_rf: Vec<Point2D>,
    pub ded_rf: Vec<Point2D>,
}

impl IncrementalSearchGrids {
    pub fn new() -> Self {
        Self::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_incremental_search_grids_default() {
        let isg = IncrementalSearchGrids::new();
        assert!(isg.add_l.is_empty());
        assert!(isg.ded_r.is_empty());
        assert!(isg.add_rf.is_empty());
    }
}
