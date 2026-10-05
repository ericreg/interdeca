//! Owned float64 triangle BVHs and bounded, ordered parallel surface projection.

use std::sync::OnceLock;

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use parry3d_f64::math::Vector;
use parry3d_f64::query::PointQueryWithLocation;
use parry3d_f64::shape::TriMesh;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

// One process-wide pool bounds concurrency across simultaneously used indices.
// Small landmark batches never start the pool; large surface batches amortize it.
const QUERIES_PER_WORKER: usize = 8192;
static WORKER_LIMIT: OnceLock<Result<usize, String>> = OnceLock::new();
static POOL: OnceLock<Result<rayon::ThreadPool, String>> = OnceLock::new();

fn worker_limit() -> Result<usize, String> {
    WORKER_LIMIT
        .get_or_init(|| {
            let available = std::thread::available_parallelism().map_or(1, usize::from);
            let requested = match std::env::var("INTERDECA_SPATIAL_THREADS") {
                Ok(value) => value
                    .parse::<usize>()
                    .ok()
                    .filter(|&n| n > 0)
                    .ok_or_else(|| {
                        "INTERDECA_SPATIAL_THREADS must be a positive integer.".to_owned()
                    })?,
                Err(std::env::VarError::NotPresent) => 8,
                Err(error) => return Err(error.to_string()),
            };
            Ok(requested.min(available))
        })
        .clone()
}

fn worker_count(count: usize) -> Result<usize, String> {
    let limit = worker_limit()?;
    Ok(if count == 0 {
        0
    } else {
        limit.min((count / QUERIES_PER_WORKER).max(1))
    })
}

fn pool() -> Result<&'static rayon::ThreadPool, String> {
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(worker_limit()?)
            .thread_name(|index| format!("interdeca-spatial-{index}"))
            .build()
            .map_err(|error| error.to_string())
    })
    .as_ref()
    .map_err(Clone::clone)
}

/// Snapshot before releasing the GIL, even when the caller's view is read-only:
/// another Python alias may still be able to mutate its underlying allocation.
fn vectors(array: &PyReadonlyArray2<'_, f64>) -> PyResult<Vec<Vector>> {
    crate::require_aligned(array)?;
    if !array.is_c_contiguous() || array.shape()[1] != 3 {
        return Err(PyValueError::new_err(
            "Spatial coordinates must be C-contiguous (N, 3) arrays.",
        ));
    }
    let values = array.as_slice()?;
    if !values.iter().all(|value| value.is_finite()) {
        return Err(PyValueError::new_err("Spatial coordinates must be finite."));
    }
    Ok(values
        .chunks_exact(3)
        .map(|p| Vector::new(p[0], p[1], p[2]))
        .collect())
}

type ProjectionArrays<'py> = (
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
);

#[derive(Clone, Copy, Default)]
struct Hit {
    point: Vector,
    triangle: i64,
    squared_distance: f64,
}

/// A reusable owned index: neither VTK nor borrowed NumPy storage is retained.
#[pyclass(frozen, module = "interdeca._native")]
struct SurfaceIndex {
    mesh: TriMesh,
}

#[pymethods]
impl SurfaceIndex {
    #[new]
    fn new(
        py: Python<'_>,
        vertices: PyReadonlyArray2<'_, f64>,
        faces: PyReadonlyArray2<'_, i64>,
    ) -> PyResult<Self> {
        let vertices = vectors(&vertices)?;
        crate::require_aligned(&faces)?;
        if vertices.is_empty()
            || vertices.len() > u32::MAX as usize
            || !faces.is_c_contiguous()
            || faces.shape()[1] != 3
            || faces.shape()[0] == 0
            || faces.shape()[0] > u32::MAX as usize
        {
            return Err(PyValueError::new_err(
                "A spatial index needs nonempty (N, 3) vertices and (M, 3) faces within u32 capacity.",
            ));
        }
        let faces = faces.as_slice()?;
        if faces.iter().any(|&i| i < 0 || i as usize >= vertices.len()) {
            return Err(PyValueError::new_err(
                "Spatial face index is outside the vertex array.",
            ));
        }
        let faces = faces
            .chunks_exact(3)
            .map(|f| [f[0] as u32, f[1] as u32, f[2] as u32])
            .collect();
        // Default flags build a BVH without welding, removing, or reindexing
        // input geometry. Returned face IDs therefore still address UV corners.
        py.detach(move || {
            let mesh = TriMesh::new(vertices, faces)
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
            Ok(Self { mesh })
        })
    }

    fn project<'py>(
        &self,
        py: Python<'py>,
        queries: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<ProjectionArrays<'py>> {
        let queries = vectors(&queries)?;
        let workers = worker_count(queries.len()).map_err(PyValueError::new_err)?;
        let hits = py
            .detach(|| -> Result<Vec<Hit>, String> {
                let project = |query: &Vector| -> Result<Hit, String> {
                    // The checked query avoids Parry's unchecked fallback of returning
                    // the query itself when extreme coordinates prevent a valid hit.
                    let (projection, (triangle, _)) = self
                        .mesh
                        .project_local_point_and_get_location_with_max_dist(*query, false, f64::MAX)
                        .ok_or_else(|| "No finite closest surface point was found.".to_owned())?;
                    let squared_distance = (*query - projection.point).length_squared();
                    if !projection.point.is_finite() || !squared_distance.is_finite() {
                        return Err(
                            "Surface projection exceeded finite floating-point range.".to_owned()
                        );
                    }
                    Ok(Hit {
                        point: projection.point,
                        triangle: i64::from(triangle),
                        squared_distance,
                    })
                };
                let mut hits = vec![Hit::default(); queries.len()];
                if workers > 1 {
                    // At most `workers` ordered chunks are active. Each point query
                    // is independent; there are no parallel floating-point reductions.
                    let chunk = queries.len().div_ceil(workers);
                    pool()?.install(|| {
                        hits.par_chunks_mut(chunk)
                            .zip(queries.par_chunks(chunk))
                            .try_for_each(|(output, input)| -> Result<(), String> {
                                for (hit, query) in output.iter_mut().zip(input) {
                                    *hit = project(query)?;
                                }
                                Ok(())
                            })
                    })?;
                } else {
                    for (hit, query) in hits.iter_mut().zip(&queries) {
                        *hit = project(query)?;
                    }
                }
                Ok(hits)
            })
            .map_err(PyValueError::new_err)?;
        // Transfer owned buffers into NumPy; freeze the core's numerical records.
        let mut points = Vec::with_capacity(hits.len() * 3);
        let mut triangles = Vec::with_capacity(hits.len());
        let mut distances = Vec::with_capacity(hits.len());
        for hit in hits {
            points.extend_from_slice(&hit.point.to_array());
            triangles.push(hit.triangle);
            distances.push(hit.squared_distance);
        }
        let points = Array2::from_shape_vec((triangles.len(), 3), points)
            .expect("one three-coordinate point per hit")
            .into_pyarray(py);
        let triangles = triangles.into_pyarray(py);
        let distances = distances.into_pyarray(py);
        points.call_method1("setflags", (false,))?;
        triangles.call_method1("setflags", (false,))?;
        distances.call_method1("setflags", (false,))?;
        Ok((points, triangles, distances))
    }
}

#[pyfunction]
fn spatial_worker_count(query_count: usize) -> PyResult<usize> {
    worker_count(query_count).map_err(PyValueError::new_err)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<SurfaceIndex>()?;
    module.add_function(wrap_pyfunction!(spatial_worker_count, module)?)?;
    Ok(())
}
