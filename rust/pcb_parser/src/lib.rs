use pyo3::prelude::*;

pub mod ir;

#[pymodule]
fn pcb_parser(m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
