// SPDX-FileCopyrightText: 2026 InterDeCA contributors
// SPDX-License-Identifier: GPL-3.0-or-later
//! Python boundary for the pinned Blender Smart UV Project implementation.

use crate::surface::Mesh;
use numpy::ndarray::Array3;
use numpy::{IntoPyArray, PyArray1, PyArray3, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub(crate) type P2 = [f64; 2];
pub(crate) type Tri2 = [P2; 3];

type UVArrays<'py> = (Bound<'py, PyArray3<f64>>, Bound<'py, PyArray1<i64>>);

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn surface_uv<'py>(
    py: Python<'py>,
    vertices: PyReadonlyArray2<'py, f64>,
    faces: PyReadonlyArray2<'py, i64>,
    angle: f64,
    margin: f64,
    width: usize,
    height: usize,
    padding: usize,
) -> PyResult<UVArrays<'py>> {
    let mesh = Mesh::read(&vertices, &faces)?;
    if !angle.is_finite()
        || angle <= 0.0
        || angle > 89.0
        || !margin.is_finite()
        || !(0.0..1.0).contains(&margin)
        || width == 0
        || height == 0
        || width > 8192
        || height > 8192
        || padding > 64
    {
        return Err(PyValueError::new_err(
            "Invalid UV angle, margin, resolution, or padding.",
        ));
    }
    let (uv, ids) = py
        .detach(move || crate::blender::uv::unwrap(mesh, angle, margin))
        .map_err(PyValueError::new_err)?;
    let uv = Array3::from_shape_vec((ids.len(), 3, 2), uv)
        .unwrap()
        .into_pyarray(py);
    let ids = ids.into_pyarray(py);
    uv.call_method1("setflags", (false,))?;
    ids.call_method1("setflags", (false,))?;
    Ok((uv, ids))
}
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(surface_uv, module)?)?;
    Ok(())
}
