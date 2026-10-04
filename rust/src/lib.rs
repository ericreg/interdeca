//! Native mesh I/O and texture arithmetic for the InterDeCA Python package.
//!
//! NumPy owns returned buffers through rust-numpy capsules. The Python layer
//! handles file-format selection, schemas, image codecs, and VTK integration.

use pyo3::prelude::*;

mod mesh;
mod texture;

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    mesh::register(module)?;
    texture::register(module)?;
    Ok(())
}
