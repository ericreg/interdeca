//! Native mesh I/O, color operations, and surface search for InterDeCA.
//!
//! NumPy owns returned buffers through rust-numpy capsules. The Python layer
//! handles file-format selection, schemas, image codecs, and VTK integration.

use numpy::ndarray::Dimension;
use numpy::{Element, PyReadonlyArray, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

mod color;
mod mesh;
mod spatial;
mod texture;

/// Check the native reference invariant even when callers bypass Python schemas.
fn require_aligned<T: Element, D: Dimension>(array: &PyReadonlyArray<'_, T, D>) -> PyResult<()> {
    if !array.is_aligned() {
        return Err(PyValueError::new_err(
            "Native arrays must have aligned storage.",
        ));
    }
    Ok(())
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    color::register(module)?;
    mesh::register(module)?;
    spatial::register(module)?;
    texture::register(module)?;
    Ok(())
}
