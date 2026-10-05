//! Experimental closest-surface search; the production VTK path is unchanged.
//!
//! Inputs use float64 positions and int64 triangle indices. Mesh order is kept:
//! no welding, sorting of input triangles, topology repair, or UV processing.

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use parry3d_f64::math::Vector;
use parry3d_f64::query::PointQueryWithLocation;
use parry3d_f64::shape::TriMesh;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

type ProjectionArrays<'py> = (
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
);

struct Hit {
    point: Vector,
    triangle: i64,
    squared_distance: f64,
}

/// An owned mesh and BVH that can be reused for many batches of query points.
#[pyclass(frozen)]
struct ParryIndex {
    mesh: TriMesh,
    pool: Option<rayon::ThreadPool>,
}

/// Snapshot inputs before releasing Python's GIL: a read-only NumPy view alone
/// cannot prevent a different alias from modifying its underlying storage.
fn vectors(array: &PyReadonlyArray2<'_, f64>) -> PyResult<Vec<Vector>> {
    if !array.is_aligned() || !array.is_c_contiguous() || array.shape()[1] != 3 {
        return Err(PyValueError::new_err("Expected an aligned N by 3 array."));
    }
    let values = array.as_slice()?;
    if !values.iter().all(|value| value.is_finite()) {
        return Err(PyValueError::new_err("Coordinates must be finite."));
    }
    Ok(values
        .chunks_exact(3)
        .map(|row| Vector::new(row[0], row[1], row[2]))
        .collect())
}

#[pymethods]
impl ParryIndex {
    #[new]
    #[pyo3(signature = (vertices, faces, threads=1))]
    fn new(
        py: Python<'_>,
        vertices: PyReadonlyArray2<'_, f64>,
        faces: PyReadonlyArray2<'_, i64>,
        threads: usize,
    ) -> PyResult<Self> {
        let positions = vectors(&vertices)?;
        if positions.is_empty() || positions.len() > u32::MAX as usize || threads == 0 {
            return Err(PyValueError::new_err("Invalid vertex or thread count."));
        }
        if !faces.is_aligned()
            || !faces.is_c_contiguous()
            || faces.shape()[1] != 3
            || faces.shape()[0] == 0
        {
            return Err(PyValueError::new_err(
                "Expected a nonempty aligned M by 3 index array.",
            ));
        }
        let values = faces.as_slice()?;
        if values
            .iter()
            .any(|&i| i < 0 || i as usize >= positions.len())
        {
            return Err(PyValueError::new_err(
                "Triangle index outside vertex array.",
            ));
        }
        let triangles = values
            .chunks_exact(3)
            .map(|row| [row[0] as u32, row[1] as u32, row[2] as u32])
            .collect();
        // Default TriMesh flags leave input vertices and triangle IDs unchanged.
        // BVH construction remains serial in both benchmark configurations.
        py.detach(move || {
            let mesh = TriMesh::new(positions, triangles)
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
            let pool = if threads == 1 {
                None
            } else {
                Some(
                    rayon::ThreadPoolBuilder::new()
                        .num_threads(threads)
                        .build()
                        .map_err(|error| PyValueError::new_err(error.to_string()))?,
                )
            };
            Ok(Self { mesh, pool })
        })
    }

    /// Return closest points, original triangle IDs, and squared distances.
    /// Projection is onto the surface even when a query is inside a closed mesh.
    fn project<'py>(
        &self,
        py: Python<'py>,
        queries: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<ProjectionArrays<'py>> {
        let queries = vectors(&queries)?;
        let hits: Vec<Hit> = py.detach(|| {
            let project = |query: &Vector| {
                let (projection, (triangle, _location)) = self
                    .mesh
                    .project_local_point_and_get_location(*query, false);
                Hit {
                    point: projection.point,
                    triangle: i64::from(triangle),
                    squared_distance: (query - projection.point).length_squared(),
                }
            };
            match &self.pool {
                Some(pool) => pool.install(|| queries.par_iter().map(project).collect()),
                None => queries.iter().map(project).collect(),
            }
        });
        // Include output materialization in measured Python-call latency; callers
        // receive ordinary owned arrays rather than Rust-only result objects.
        let mut points = Vec::with_capacity(hits.len() * 3);
        let mut triangles = Vec::with_capacity(hits.len());
        let mut distances = Vec::with_capacity(hits.len());
        for hit in hits {
            if !hit.point.is_finite() || !hit.squared_distance.is_finite() {
                return Err(PyValueError::new_err(
                    "Projection overflowed finite coordinates.",
                ));
            }
            points.extend_from_slice(&hit.point.to_array());
            triangles.push(hit.triangle);
            distances.push(hit.squared_distance);
        }
        let points = Array2::from_shape_vec((triangles.len(), 3), points)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok((
            points.into_pyarray(py),
            triangles.into_pyarray(py),
            distances.into_pyarray(py),
        ))
    }
}

#[pymodule]
fn interdeca_parry_spatial(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<ParryIndex>()?;
    module.add("PARRY_VERSION", "0.31.1")?;
    Ok(())
}
