pub mod global_param;
pub mod location;
pub mod grid_cell;
pub mod grid_netclass;
pub mod incremental_search_grids;
pub mod grid_path;
pub mod grid_pin;
pub mod multipin_route;
pub mod grid_diff_pair_net;
pub mod board_grid;
pub mod kicad_parser;
pub mod grid_based_router;
pub mod util;
pub mod scpt_adapter;

use pyo3::prelude::*;
use grid_based_router::GridBasedRouter;
use global_param::GlobalParam;

/// Route a SCPT design (JSON-encoded) and return the routed design as JSON.
///
/// v1: returns the input design unchanged. The routing infrastructure runs
/// internally but the result is not yet serialized back to the SCPT IR.
/// A future pass will add segment/via serialization.
#[pyfunction]
fn route(design_json: &str) -> PyResult<String> {
    let db = scpt_adapter::scpt_json_to_kicad_db(design_json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
    let params = GlobalParam::default();
    let mut router = GridBasedRouter::new(params);
    router.setup_layer_mapping(&db);
    // Set up netclasses from the DB.
    for (i, nc) in db.netclasses.iter().enumerate() {
        router.setup_grid_netclass_from_db(
            i as i32,
            nc.clearance,
            nc.trace_width,
            nc.via_dia,
            nc.via_drill,
            nc.uvia_dia,
            nc.uvia_drill,
        );
    }
    router.initialization(&db);
    router.route_all();
    // v1: return the input design as-is. Routing results are on the router's
    // internal state (best_solution) but not serialized back. A future pass
    // will convert routed segments back into a SCPT-compatible JSON.
    Ok(design_json.to_string())
}

#[pymodule]
fn pcb_router(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(route, m)?)?;
    Ok(())
}
