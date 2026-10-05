// SPDX-FileCopyrightText: 2026 InterDeCA contributors
// SPDX-License-Identifier: GPL-3.0-or-later
//! Atlas preparation; merge representatives follow the pinned Blender source.

use std::collections::HashSet;

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use parry3d_f64::math::Vector;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub(crate) struct Mesh {
    pub vertices: Vec<Vector>,
    pub faces: Vec<[u32; 3]>,
}

impl Mesh {
    // Snapshot input buffers while Python holds the GIL. Detached algorithms
    // must never observe a concurrently mutated NumPy alias.
    pub fn read(
        vertices: &PyReadonlyArray2<'_, f64>,
        faces: &PyReadonlyArray2<'_, i64>,
    ) -> PyResult<Self> {
        crate::require_aligned(vertices)?;
        crate::require_aligned(faces)?;
        if vertices.shape()[1] != 3
            || faces.shape()[1] != 3
            || vertices.shape()[0] == 0
            || faces.shape()[0] == 0
            || vertices.shape()[0] > u32::MAX as usize
            || faces.shape()[0] > u32::MAX as usize
        {
            return Err(PyValueError::new_err(
                "Expected nonempty (N,3) vertices and (M,3) faces within u32 capacity.",
            ));
        }
        let vertices: Vec<Vector> = vertices
            .as_array()
            .rows()
            .into_iter()
            .map(|p| Vector::new(p[0], p[1], p[2]))
            .collect();
        if vertices.iter().any(|p| !p.is_finite()) {
            return Err(PyValueError::new_err("Coordinates must be finite."));
        }
        let mut triangles = Vec::with_capacity(faces.shape()[0]);
        for f in faces.as_array().rows() {
            if f.iter().any(|&i| i < 0 || i as usize >= vertices.len())
                || f[0] == f[1]
                || f[1] == f[2]
                || f[2] == f[0]
            {
                return Err(PyValueError::new_err(
                    "Faces need three distinct valid vertex indices.",
                ));
            }
            triangles.push([f[0] as u32, f[1] as u32, f[2] as u32]);
        }
        Ok(Self {
            vertices,
            faces: triangles,
        })
    }

    pub fn triangle(&self, f: usize) -> [Vector; 3] {
        self.faces[f].map(|i| self.vertices[i as usize])
    }
}

pub(crate) fn normal(t: [Vector; 3]) -> Vector {
    let a = t[1] - t[0];
    let b = t[2] - t[0];
    let cross = a.cross(b);
    let length = cross.length();
    if !length.is_finite() || length <= 32.0 * f64::EPSILON * a.length() * b.length() {
        Vector::ZERO
    } else {
        cross / length
    }
}

type CleanupArrays<'py> = (
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
);

#[pyfunction]
fn surface_clean<'py>(
    py: Python<'py>,
    vertices: PyReadonlyArray2<'py, f64>,
    faces: PyReadonlyArray2<'py, i64>,
    distance: f64,
) -> PyResult<CleanupArrays<'py>> {
    let mesh = Mesh::read(&vertices, &faces)?;
    if !distance.is_finite() || distance < 0.0 {
        return Err(PyValueError::new_err(
            "Merge distance must be finite and nonnegative.",
        ));
    }
    let (vertices, faces, vertex_map, face_map) = py
        .detach(move || clean(mesh, distance))
        .map_err(PyValueError::new_err)?;
    let vertices = Array2::from_shape_vec((vertices.len() / 3, 3), vertices)
        .unwrap()
        .into_pyarray(py);
    let faces = Array2::from_shape_vec((faces.len() / 3, 3), faces)
        .unwrap()
        .into_pyarray(py);
    let vertex_map = vertex_map.into_pyarray(py);
    let face_map = face_map.into_pyarray(py);
    vertices.call_method1("setflags", (false,))?;
    faces.call_method1("setflags", (false,))?;
    vertex_map.call_method1("setflags", (false,))?;
    face_map.call_method1("setflags", (false,))?;
    Ok((vertices, faces, vertex_map, face_map))
}

type CleanOutput = (Vec<f64>, Vec<i64>, Vec<i64>, Vec<i64>);
fn clean(mesh: Mesh, distance: f64) -> Result<CleanOutput, String> {
    if distance == 0.0 {
        return Ok((
            mesh.vertices.iter().flat_map(|p| p.to_array()).collect(),
            mesh.faces.iter().flatten().map(|&i| i64::from(i)).collect(),
            (0..mesh.vertices.len() as i64).collect(),
            (0..mesh.faces.len() as i64).collect(),
        ));
    }
    // Blender's float32 KD-tree traversal and centroid representative choice
    // are part of the topology contract. The GPL-derived implementation is
    // isolated in blender/cleanup.rs with its upstream source attribution.
    let representative = crate::blender::cleanup::representatives(&mesh.vertices, distance)?;
    let (surviving, face_map) =
        crate::blender::cleanup::weld_triangles(&mesh.faces, &representative);
    let used: HashSet<usize> = surviving.iter().flatten().copied().collect();
    if surviving.is_empty() {
        return Err("Cleanup leaves no valid surface.".into());
    }
    // Match the strict atlas boundary after Blender exports float32 positions.
    // Keep welding faithful: do not silently delete a geometrically flat face.
    if surviving.iter().any(|f| {
        normal(
            f.map(|i| Vector::from_array(mesh.vertices[i].to_array().map(|x| f64::from(x as f32)))),
        ) == Vector::ZERO
    }) {
        return Err("Blender-compatible cleanup leaves degenerate triangles.".into());
    }
    let mut compact = vec![-1; mesh.vertices.len()];
    let mut vertices = Vec::new();
    for (i, p) in mesh.vertices.iter().enumerate() {
        if used.contains(&i) {
            compact[i] = (vertices.len() / 3) as i64;
            // Blender imports and exports these coordinates through float32.
            vertices.extend(p.to_array().map(|x| f64::from(x as f32)));
        }
    }
    let vertex_map = representative.iter().map(|&j| compact[j]).collect();
    let faces = surviving.iter().flatten().map(|&j| compact[j]).collect();
    Ok((vertices, faces, vertex_map, face_map))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(surface_clean, module)?)?;
    Ok(())
}
