use crate::multipin_route::MultipinRoute;

/// A differential-pair net – wraps two `MultipinRoute` instances.
///
/// In the C++ codebase `GridDiffPairNet` inherited from `MultipinRoute`.
/// In Rust we use composition: the "combined" route is in `combined` and the
/// two individual nets are referenced by index in the caller's net vector.
#[derive(Debug, Clone)]
pub struct GridDiffPairNet {
    /// The combined (centre-line) route used during A* search.
    pub combined: MultipinRoute,
    /// Index into the caller's net vector for net-1.
    pub net1_idx: usize,
    /// Index into the caller's net vector for net-2.
    pub net2_idx: usize,
    /// Pairs of GridPin indices: `(pin_id_in_net1, pin_id_in_net2)`.
    pub grid_pin_pairs: Vec<(usize, usize)>,
}

impl GridDiffPairNet {
    pub fn new(
        net_id: i32,
        grid_dp_netclass_id: i32,
        num_layers: usize,
        net1_idx: usize,
        net2_idx: usize,
    ) -> Self {
        GridDiffPairNet {
            combined: MultipinRoute::with_layers(net_id, grid_dp_netclass_id, num_layers),
            net1_idx,
            net2_idx,
            grid_pin_pairs: Vec::new(),
        }
    }

    pub fn get_grid_diff_pair_netclass_id(&self) -> i32 {
        self.combined.grid_netclass_id
    }

    pub fn is_diff_pair(&self) -> bool { true }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_diff_pair_net_new() {
        let dpn = GridDiffPairNet::new(10, 2, 4, 0, 1);
        assert_eq!(dpn.combined.net_id, 10);
        assert_eq!(dpn.combined.grid_netclass_id, 2);
        assert_eq!(dpn.net1_idx, 0);
        assert_eq!(dpn.net2_idx, 1);
        assert!(dpn.grid_pin_pairs.is_empty());
        assert_eq!(dpn.combined.layer_costs.len(), 4);
    }

    #[test]
    fn test_is_diff_pair() {
        let dpn = GridDiffPairNet::new(5, 0, 2, 0, 1);
        assert!(dpn.is_diff_pair());
    }
}
