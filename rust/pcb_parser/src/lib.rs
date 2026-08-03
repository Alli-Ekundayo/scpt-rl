use pyo3::prelude::*;

pub mod geometry;
pub mod ir;

/// Load a KiCad `.kicad_pcb` file and return SCPT's canonical `PcbDesign` as
/// a JSON string. Python callers should `json.loads` the result to get a
/// nested dict.
#[pyfunction]
fn load_kicad_pcb(path: &str) -> PyResult<String> {
    let src = std::fs::read_to_string(path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{path}: {e}")))?;
    let bo = kicad_parser::parse_pcb_to_output(&src, path, None, None)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("parse error: {e}")))?;
    let design = ir::adapter::board_output_to_pcb_design(bo);
    design
        .to_json()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("json: {e}")))
}

/// HPWL over all nets of a design. `design_json` is the JSON output of
/// `load_kicad_pcb`.
#[pyfunction]
fn hpwl(design_json: &str) -> PyResult<f64> {
    let d = deserialize_design(design_json)?;
    Ok(geometry::hpwl(&d))
}

/// HPWL recomputed only for nets touching the named component.
#[pyfunction]
fn hpwl_incremental(design_json: &str, moved_ref_des: &str) -> PyResult<f64> {
    let d = deserialize_design(design_json)?;
    Ok(geometry::hpwl_incremental(&d, moved_ref_des))
}

/// Clearance cost — sum of pairwise courtyard overlaps after expansion.
#[pyfunction]
fn clearance_cost(design_json: &str, min_spacing: f64) -> PyResult<f64> {
    let d = deserialize_design(design_json)?;
    Ok(geometry::clearance_cost(&d, min_spacing))
}

fn deserialize_design(json: &str) -> PyResult<ir::PcbDesign> {
    serde_json::from_str(json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("deserialize: {e}")))
}

#[pymodule]
fn pcb_parser(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load_kicad_pcb, m)?)?;
    m.add_function(wrap_pyfunction!(hpwl, m)?)?;
    m.add_function(wrap_pyfunction!(hpwl_incremental, m)?)?;
    m.add_function(wrap_pyfunction!(clearance_cost, m)?)?;
    Ok(())
}
