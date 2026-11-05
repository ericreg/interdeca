//! Narrow implementation of InterDeCA's triangular OBJ reader and mesh checks.
//!
//! Only v, vt and triangular f records are consumed.
//! Independent geometry/UV indices preserve seams and
//! correspondence. Materials, normals, and other records are ignored as in the
//! Python reader. No processing, welding, triangulation, or parallelism is used.

use std::borrow::Cow;
use std::fs::{self, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use numpy::ndarray::{Array2, Array3};
use numpy::{IntoPyArray, PyArray2, PyArray3, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;

/// Flat buffers have fixed element types and no caller-owned writable aliases.
struct MeshBuffers {
    vertices: Vec<f64>,
    faces: Vec<i64>,
    uv: Option<Vec<f64>>,
}

enum ReadError {
    Io(std::io::Error),
    Invalid(String),
}

impl From<std::io::Error> for ReadError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<ReadError> for PyErr {
    fn from(error: ReadError) -> Self {
        match error {
            ReadError::Io(error) => PyOSError::new_err(error.to_string()),
            ReadError::Invalid(message) => PyValueError::new_err(message),
        }
    }
}

fn invalid(path: &Path, line: usize, message: &str) -> ReadError {
    ReadError::Invalid(format!("{}:{line}: {message}", path.display()))
}

/// Python's numeric converters allow underscores between decimal digits.
/// The usual OBJ tokens take the borrowed path without an allocation.
fn numeric_token(token: &str) -> Result<Cow<'_, str>, &'static str> {
    if !token.contains('_') {
        return Ok(Cow::Borrowed(token));
    }
    let bytes = token.as_bytes();
    for (index, value) in bytes.iter().enumerate() {
        if *value == b'_'
            && (index == 0
                || index + 1 == bytes.len()
                || !bytes[index - 1].is_ascii_digit()
                || !bytes[index + 1].is_ascii_digit())
        {
            return Err("Invalid underscore in numeric token.");
        }
    }
    Ok(Cow::Owned(token.replace('_', "")))
}

fn real(token: Option<&str>, path: &Path, line: usize) -> Result<f64, ReadError> {
    let token = token.ok_or_else(|| invalid(path, line, "Incomplete coordinate."))?;
    let token = numeric_token(token).map_err(|message| invalid(path, line, message))?;
    // Rust's standard parser rounds decimal text directly to binary64. No
    // intermediate float32 or home-grown decimal accumulation loses precision.
    token
        .parse::<f64>()
        .map_err(|_| invalid(path, line, "Invalid floating-point coordinate."))
}

fn obj_index(token: &str, count: usize, path: &Path, line: usize) -> Result<usize, ReadError> {
    let token = numeric_token(token).map_err(|message| invalid(path, line, message))?;
    let index = token
        .parse::<i128>()
        .map_err(|_| invalid(path, line, "Invalid OBJ index."))?;
    // Resolve negative indices against the records available at this face,
    // not against the final file length. Zero is never a valid OBJ index.
    let resolved = if index > 0 {
        index - 1
    } else {
        count as i128 + index
    };
    if index == 0 || resolved < 0 || resolved >= count as i128 {
        return Err(invalid(path, line, "OBJ index is out of bounds."));
    }
    Ok(resolved as usize)
}

fn parse(path: &Path, flip_xy: bool) -> Result<MeshBuffers, ReadError> {
    // Read one file at a time. Borrowed line/field slices avoid a Python object
    // per token; only the final numeric buffers outlive this parsing call.
    let source = fs::read_to_string(path)?;
    let source = source.strip_prefix('\u{feff}').unwrap_or(&source);
    // Python opens files with universal newline handling, including CR-only
    // files. Normal LF files avoid this extra allocation entirely.
    let normalized = if source.contains('\r') {
        Cow::Owned(source.replace("\r\n", "\n").replace('\r', "\n"))
    } else {
        Cow::Borrowed(source)
    };
    let mut vertices = Vec::new();
    let mut texture_vertices = Vec::new();
    let mut faces = Vec::new();
    let mut corner_uv = Vec::new();
    let mut uv_present = false;
    let mut uv_absent = false;

    for (line_index, text) in normalized.lines().enumerate() {
        let line = line_index + 1;
        // Comments must be removed before splitting, including comments after
        // the last face corner. Unknown records do not affect mesh indexing.
        let text = text.split('#').next().unwrap_or("");
        let mut fields = text.split_whitespace();
        match fields.next() {
            Some("v") => {
                for component in 0..3 {
                    let value = real(fields.next(), path, line)?;
                    // Normalize LPS to RAS while filling the final buffer. Two
                    // sign flips preserve winding and require no extra copy.
                    vertices.push(if flip_xy && component < 2 {
                        -value
                    } else {
                        value
                    });
                }
            }
            Some("vt") => {
                for _ in 0..2 {
                    texture_vertices.push(real(fields.next(), path, line)?);
                }
            }
            Some("f") => {
                // Each triangle contributes exactly three geometry indices and,
                // when textured, six UV values. Never invent a triangulation.
                for _ in 0..3 {
                    let corner = fields
                        .next()
                        .ok_or_else(|| invalid(path, line, "OBJ faces must be triangles."))?;
                    let mut indices = corner.split('/');
                    let vertex =
                        obj_index(indices.next().unwrap_or(""), vertices.len() / 3, path, line)?;
                    faces.push(vertex as i64);
                    if let Some(texture) = indices.next().filter(|value| !value.is_empty()) {
                        let index = obj_index(texture, texture_vertices.len() / 2, path, line)?;
                        corner_uv.extend_from_slice(&texture_vertices[2 * index..2 * index + 2]);
                        uv_present = true;
                    } else {
                        uv_absent = true;
                    }
                }
                if fields.next().is_some() {
                    return Err(invalid(path, line, "OBJ faces must be triangles."));
                }
            }
            _ => {}
        }
    }
    // A partially textured mesh cannot be represented as a complete (M,3,2)
    // array. Reject it before constructing the public mesh buffers.
    if uv_present && uv_absent {
        return Err(ReadError::Invalid(format!(
            "{}: texture coordinates are missing from some face corners.",
            path.display()
        )));
    }
    Ok(MeshBuffers {
        vertices,
        faces,
        uv: uv_present.then_some(corner_uv),
    })
}

fn norm(vector: [f64; 3]) -> f64 {
    // Preserve the Python implementation's direct squared norm, rather than
    // introducing rescaling, fused arithmetic, or a different area tolerance.
    (vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]).sqrt()
}

fn validate(mesh: &MeshBuffers, allow_degenerate: bool) -> Result<(), ReadError> {
    if mesh.vertices.is_empty() || mesh.faces.is_empty() {
        return Err(ReadError::Invalid(
            "Vertices and faces must be nonempty.".into(),
        ));
    }
    if mesh.vertices.iter().any(|value| !value.is_finite())
        || mesh.uv.iter().flatten().any(|value| !value.is_finite())
    {
        return Err(ReadError::Invalid(
            "Arrays must not contain NaN or infinity.".into(),
        ));
    }
    // Parsing establishes shapes, integer types, UV coverage, and index bounds.
    // Numerical validation adds the checks that require interpreting geometry.
    for face in mesh.faces.chunks_exact(3) {
        if face
            .iter()
            .any(|&index| index < 0 || index as usize >= mesh.vertices.len() / 3)
        {
            return Err(ReadError::Invalid(
                "A face references a nonexistent vertex.".into(),
            ));
        }
        if face[0] == face[1] || face[1] == face[2] || face[2] == face[0] {
            return Err(ReadError::Invalid(
                "A face must reference three distinct vertex indices.".into(),
            ));
        }
        if !allow_degenerate {
            let a = face[0] as usize * 3;
            let b = face[1] as usize * 3;
            let c = face[2] as usize * 3;
            let edge1 = std::array::from_fn(|i| mesh.vertices[b + i] - mesh.vertices[a + i]);
            let edge2 = std::array::from_fn(|i| mesh.vertices[c + i] - mesh.vertices[a + i]);
            let cross = [
                edge1[1] * edge2[2] - edge1[2] * edge2[1],
                edge1[2] * edge2[0] - edge1[0] * edge2[2],
                edge1[0] * edge2[1] - edge1[1] * edge2[0],
            ];
            // Twice the area divided by the edge-length product is sin(angle).
            // The relative threshold is invariant to a uniform change of units.
            let edge_product = norm(edge1) * norm(edge2);
            let threshold = f64::EPSILON * 32.0 * edge_product;
            if norm(cross) <= threshold {
                return Err(ReadError::Invalid(
                    "Mesh contains zero-area or numerically degenerate triangles.".into(),
                ));
            }
        }
    }
    Ok(())
}

type Output<'py> = (
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<i64>>,
    Option<Bound<'py, PyArray3<f64>>>,
);

/// Read one triangular OBJ into owned, read-only NumPy arrays.
///
/// The reader always performs the same geometric checks as Mesh.validated(). allow_degenerate
/// skips only the area test, preserving the distinct-index requirement.
/// coordinate_system describes source axes; output vertices always use RAS.
#[pyfunction(name = "read_obj", signature = (path, *, coordinate_system="RAS", allow_degenerate=false))]
fn read_obj<'py>(
    py: Python<'py>,
    path: PathBuf,
    coordinate_system: &str,
    allow_degenerate: bool,
) -> PyResult<Output<'py>> {
    let flip_xy = match coordinate_system {
        "RAS" => false,
        "LPS" => true,
        _ => {
            return Err(PyValueError::new_err(
                "coordinate_system must be RAS or LPS.",
            ));
        }
    };
    // Release the GIL during single-threaded Rust work. Ownership then moves
    // directly into NumPy, avoiding Python scalar lists and redundant copies.
    let mesh = py.detach(|| {
        let mesh = parse(&path, flip_xy)?;
        validate(&mesh, allow_degenerate)?;
        Ok::<_, ReadError>(mesh)
    })?;
    let vertex_count = mesh.vertices.len() / 3;
    let face_count = mesh.faces.len() / 3;
    let vertices = Array2::from_shape_vec((vertex_count, 3), mesh.vertices)
        .expect("parser establishes vertex shape")
        .into_pyarray(py);
    let faces = Array2::from_shape_vec((face_count, 3), mesh.faces)
        .expect("parser establishes face shape")
        .into_pyarray(py);
    let uv = mesh.uv.map(|values| {
        Array3::from_shape_vec((face_count, 3, 2), values)
            .expect("parser establishes corner UV shape")
            .into_pyarray(py)
    });
    // Freeze the result before exposing it to the Python numerical core.
    // Rust buffers are owned by NumPy's base capsule after this function returns.
    vertices.call_method1("setflags", (false,))?;
    faces.call_method1("setflags", (false,))?;
    if let Some(array) = &uv {
        array.call_method1("setflags", (false,))?;
    }
    Ok((vertices, faces, uv))
}

/// Snapshot borrowed arrays once, preserving logical order for strided views.
/// Validation and file writing then operate entirely on private Rust buffers.
fn snapshot(
    vertices: PyReadonlyArrayDyn<'_, f64>,
    faces: PyReadonlyArrayDyn<'_, i64>,
    uv: Option<PyReadonlyArrayDyn<'_, f64>>,
) -> PyResult<MeshBuffers> {
    // Inspect metadata before constructing Rust views: empty negative strides
    // and unsupported ranks must never reach rust-numpy's view conversion.
    if vertices.ndim() != 2 || vertices.shape()[1] != 3 || vertices.shape()[0] == 0 {
        return Err(PyValueError::new_err(
            "vertices must have shape (N, 3), with N > 0.",
        ));
    }
    if faces.ndim() != 2 || faces.shape()[1] != 3 || faces.shape()[0] == 0 {
        return Err(PyValueError::new_err(
            "faces must have shape (M, 3), with M > 0.",
        ));
    }
    crate::require_aligned(&vertices)?;
    crate::require_aligned(&faces)?;
    let vertices = vertices.as_array();
    let faces = faces.as_array();
    let uv = match uv {
        None => None,
        Some(uv) => {
            if uv.shape() != [faces.shape()[0], 3, 2] {
                return Err(PyValueError::new_err(
                    "uv must have shape (number_of_faces, 3, 2).",
                ));
            }
            crate::require_aligned(&uv)?;
            let uv = uv.as_array();
            Some(uv.iter().copied().collect())
        }
    };
    Ok(MeshBuffers {
        vertices: vertices.iter().copied().collect(),
        faces: faces.iter().copied().collect(),
        uv,
    })
}

/// Validate before opening the destination, then stream an exact-round-trip OBJ.
///
/// Floating-point Display emits enough decimal digits to recover each f64.
/// Every UV corner gets its own vt record. File text can differ from Python's
/// old .17g formatting, but reading recovers identical numeric buffers.
#[pyfunction(signature = (path, vertices, faces, uv=None, *, overwrite=false, allow_degenerate=false))]
fn write_obj(
    py: Python<'_>,
    path: PathBuf,
    vertices: PyReadonlyArrayDyn<'_, f64>,
    faces: PyReadonlyArrayDyn<'_, i64>,
    uv: Option<PyReadonlyArrayDyn<'_, f64>>,
    overwrite: bool,
    allow_degenerate: bool,
) -> PyResult<()> {
    let mesh = snapshot(vertices, faces, uv)?;
    py.detach(|| -> PyResult<()> {
        validate(&mesh, allow_degenerate)?;
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent)?;
        }
        // Exclusive creation preserves the overwrite contract without a racy
        // exists-then-open check. No giant Unicode/encoded whole-file buffers.
        let file = OpenOptions::new()
            .write(true)
            .create(overwrite)
            .truncate(overwrite)
            .create_new(!overwrite)
            .open(&path)?;
        let mut writer = BufWriter::with_capacity(256 * 1024, file);
        writeln!(
            writer,
            "# InterDeCA; RAS coordinates; triangle and vertex order preserved"
        )?;
        for point in mesh.vertices.chunks_exact(3) {
            writeln!(writer, "v {} {} {}", point[0], point[1], point[2])?;
        }
        if let Some(uv) = &mesh.uv {
            for corner in uv.chunks_exact(2) {
                writeln!(writer, "vt {} {}", corner[0], corner[1])?;
            }
        }
        for (index, face) in mesh.faces.chunks_exact(3).enumerate() {
            if mesh.uv.is_some() {
                writeln!(
                    writer,
                    "f {}/{} {}/{} {}/{}",
                    face[0] + 1,
                    index * 3 + 1,
                    face[1] + 1,
                    index * 3 + 2,
                    face[2] + 1,
                    index * 3 + 3
                )?;
            } else {
                writeln!(writer, "f {} {} {}", face[0] + 1, face[1] + 1, face[2] + 1)?;
            }
        }
        // Propagate final flush errors; BufWriter's destructor cannot report them.
        writer.flush()?;
        Ok(())
    })
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(read_obj, module)?)?;
    module.add_function(wrap_pyfunction!(write_obj, module)?)?;
    Ok(())
}
