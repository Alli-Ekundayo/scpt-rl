use pyo3::prelude::*;

pub mod ir;

/// Load a KiCad `.kicad_pcb` file and return SCPT's canonical `PcbDesign` as
/// a JSON string. Python callers should `json.loads` the result to get a
/// nested dict.
///
/// This v1 boundary keeps the PyO3 surface minimal — a single function
/// returning a string. The Python env layer will deserialize into dataclasses
/// / dicts. A future pass can add `#[pyclass]` annotations to the IR types
/// for direct attribute access if profiling shows the JSON round-trip matters.
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

#[pymodule]
fn pcb_parser(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load_kicad_pcb, m)?)?;
    Ok(())
}
