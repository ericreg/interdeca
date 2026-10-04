//! Area-weighted diffusion on face adjacency, preserving sample and neighbor order.

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Average each valid face with its valid edge neighbors, then select samples.
///
/// Offsets delimit rows in the flat neighbor array. Each step reads the previous
/// step's colors, so face traversal order cannot influence the diffusion. The
/// face itself is accumulated first, followed by the supplied neighbor order.
#[pyfunction]
fn smooth_face_colors<'py>(
    py: Python<'py>,
    rgb: PyReadonlyArray2<'py, f64>,
    valid: PyReadonlyArray1<'py, bool>,
    areas: PyReadonlyArray1<'py, f64>,
    offsets: PyReadonlyArray1<'py, i64>,
    neighbors: PyReadonlyArray1<'py, i64>,
    samples: PyReadonlyArray1<'py, i64>,
    steps: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    // Contiguous slices also check alignment before creating Rust references.
    // Validate every index once, even when the final pass visits only samples.
    let count = areas.len();
    if count == 0 || rgb.shape() != [count, 3] || valid.len() != count {
        return Err(PyValueError::new_err(
            "Colors, coverage and areas must describe the same nonempty face set.",
        ));
    }
    if !rgb.is_c_contiguous() {
        return Err(PyValueError::new_err(
            "Colors require C-contiguous (face, RGB) storage.",
        ));
    }
    let rgb = rgb.as_slice()?;
    let valid = valid.as_slice()?;
    let areas = areas.as_slice()?;
    let offsets = offsets.as_slice()?;
    let neighbors = neighbors.as_slice()?;
    let samples = samples.as_slice()?;
    if offsets.len() != count + 1
        || offsets[0] != 0
        || offsets[count] != neighbors.len() as i64
        || offsets.windows(2).any(|pair| pair[0] > pair[1])
        || neighbors
            .iter()
            .chain(samples)
            .any(|&face| face < 0 || face as usize >= count)
        || samples.is_empty()
    {
        return Err(PyValueError::new_err(
            "Invalid face adjacency offsets, neighbor indices or sample indices.",
        ));
    }
    if areas.iter().any(|&area| !area.is_finite() || area <= 0.0)
        || rgb.iter().any(|value| !value.is_finite())
    {
        return Err(PyValueError::new_err(
            "Smoothing requires finite colors and positive finite face areas.",
        ));
    }

    // Borrow NumPy buffers with the GIL held. Allocate full-face scratch only
    // for intermediate passes; the final pass produces the sampled output.
    let mut previous = if steps > 1 { rgb.to_vec() } else { Vec::new() };
    let mut scratch = vec![0.0; previous.len()];
    for _step in 0..steps.saturating_sub(1) {
        for face in 0..count {
            let value = average(face, &previous, valid, areas, offsets, neighbors);
            scratch[3 * face..3 * face + 3].copy_from_slice(&value);
        }
        std::mem::swap(&mut previous, &mut scratch);
    }
    let current = if steps > 1 { &previous } else { rgb };
    let mut output = Vec::with_capacity(samples.len() * 3);
    for &face in samples {
        let face = face as usize;
        let value = if steps == 0 {
            if valid[face] {
                [rgb[3 * face], rgb[3 * face + 1], rgb[3 * face + 2]]
            } else {
                [0.0; 3]
            }
        } else {
            average(face, current, valid, areas, offsets, neighbors)
        };
        output.extend_from_slice(&value);
    }
    let result = Array2::from_shape_vec((samples.len(), 3), output)
        .expect("one RGB triple per selected face")
        .into_pyarray(py);
    result.call_method1("setflags", (false,))?;
    Ok(result)
}

fn average(
    face: usize,
    rgb: &[f64],
    valid: &[bool],
    areas: &[f64],
    offsets: &[i64],
    neighbors: &[i64],
) -> [f64; 3] {
    if !valid[face] {
        return [0.0; 3];
    }
    // The positive self weight guarantees a nonempty valid neighborhood.
    // Explicit multiply then add keeps ordinary float64 reduction semantics.
    let mut weight = areas[face];
    let mut total = [
        rgb[3 * face] * weight,
        rgb[3 * face + 1] * weight,
        rgb[3 * face + 2] * weight,
    ];
    for &neighbor in &neighbors[offsets[face] as usize..offsets[face + 1] as usize] {
        let neighbor = neighbor as usize;
        if valid[neighbor] {
            let area = areas[neighbor];
            weight += area;
            for channel in 0..3 {
                total[channel] += rgb[3 * neighbor + channel] * area;
            }
        }
    }
    total.map(|value| value / weight)
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(smooth_face_colors, module)?)?;
    Ok(())
}
