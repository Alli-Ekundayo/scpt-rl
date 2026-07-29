use pyo3::prelude::*;

pub mod ir;
pub mod kicad;

#[pymodule]
fn pcb_parser(_m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
