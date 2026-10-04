//! Fused texture arithmetic over NumPy buffers, independent of image codecs.
//!
//! Borrowed inputs may be strided and are never modified. We keep the GIL while
//! using Python-owned buffers; only the file parser works on detached Rust data.
//! Each operation allocates only its final outputs, not full-image temporaries.

use numpy::ndarray::{Array2, Array3, ArrayD};
use numpy::{
    IntoPyArray, PyArray2, PyArray3, PyArrayDyn, PyReadonlyArray2, PyReadonlyArray3,
    PyReadonlyArrayDyn,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

type Channels<'py> = (Bound<'py, PyArray3<f64>>, Bound<'py, PyArray2<f64>>);

/// Normalize every byte as float64(byte)/255, directly into separate channels.
#[pyfunction]
fn texture_from_rgba8<'py>(
    py: Python<'py>,
    pixels: PyReadonlyArray3<'py, u8>,
) -> PyResult<Channels<'py>> {
    let pixels = pixels.as_array();
    let (height, width, channels) = pixels.dim();
    if height == 0 || width == 0 || channels != 4 {
        return Err(PyValueError::new_err(
            "Encoded textures need nonempty shape (height, width, 4).",
        ));
    }
    let mut rgb = Vec::with_capacity(height * width * 3);
    let mut alpha = Vec::with_capacity(height * width);
    // Lanes preserve logical row/column order even for cropped or reversed
    // NumPy views. Division matches the existing float64 normalization exactly.
    for pixel in pixels.lanes(numpy::ndarray::Axis(2)) {
        for channel in 0..3 {
            rgb.push(f64::from(pixel[channel]) / 255.0);
        }
        alpha.push(f64::from(pixel[3]) / 255.0);
    }
    let rgb = Array3::from_shape_vec((height, width, 3), rgb)
        .expect("one RGB triple per pixel")
        .into_pyarray(py);
    let alpha = Array2::from_shape_vec((height, width), alpha)
        .expect("one alpha per pixel")
        .into_pyarray(py);
    // Returned channels own their buffers independently of the source image.
    rgb.call_method1("setflags", (false,))?;
    alpha.call_method1("setflags", (false,))?;
    Ok((rgb, alpha))
}

fn quantize(value: f64) -> PyResult<u8> {
    // Check before casting: Rust's saturating float-to-byte cast must never
    // hide NaN, infinity, or an out-of-range color supplied by the caller.
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(PyValueError::new_err(
            "Texture RGB and alpha must be finite and lie in [0, 1].",
        ));
    }
    // NumPy rint rounds halfway values to the nearest even integer. Ordinary
    // Rust round() uses a different tie rule and would change output pixels.
    Ok((value * 255.0).round_ties_even() as u8)
}

/// Validate unit-range RGB/alpha and produce the final interleaved RGBA bytes.
#[pyfunction]
fn texture_to_rgba8<'py>(
    py: Python<'py>,
    rgb: PyReadonlyArray3<'py, f64>,
    alpha: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let rgb = rgb.as_array();
    let alpha = alpha.as_array();
    let (height, width, channels) = rgb.dim();
    if height == 0 || width == 0 || channels != 3 || alpha.dim() != (height, width) {
        return Err(PyValueError::new_err(
            "Textures need nonempty RGB (height, width, 3) and matching alpha.",
        ));
    }
    let mut pixels = Vec::with_capacity(height * width * 4);
    // Validate and quantize in one traversal. There are no copied float
    // channels, stacked float RGBA image, or intermediate scaled/rounded image.
    for (pixel, coverage) in rgb
        .lanes(numpy::ndarray::Axis(2))
        .into_iter()
        .zip(alpha.iter())
    {
        for channel in 0..3 {
            pixels.push(quantize(pixel[channel])?);
        }
        pixels.push(quantize(*coverage)?);
    }
    let pixels = Array3::from_shape_vec((height, width, 4), pixels)
        .expect("one RGBA tuple per pixel")
        .into_pyarray(py);
    pixels.call_method1("setflags", (false,))?;
    Ok(pixels)
}

/// Decode sRGB's piecewise transfer function with one final output buffer.
#[pyfunction]
fn texture_to_linear<'py>(
    py: Python<'py>,
    rgb: PyReadonlyArrayDyn<'py, f64>,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let rgb = rgb.as_array();
    // The existing arithmetic helper does not enforce [0,1] or finiteness.
    // Preserve that behavior; finite/unit checks belong at the image boundary.
    let values = rgb
        .iter()
        .map(|&value| {
            if value <= 0.04045 {
                value / 12.92
            } else {
                ((value + 0.055) / 1.055).powf(2.4)
            }
        })
        .collect();
    let result = ArrayD::from_shape_vec(rgb.raw_dim(), values)
        .expect("one output per input value")
        .into_pyarray(py);
    result.call_method1("setflags", (false,))?;
    Ok(result)
}

/// Clip linear-light values to [0,1] and encode them as sRGB.
#[pyfunction]
fn texture_from_linear<'py>(
    py: Python<'py>,
    rgb: PyReadonlyArrayDyn<'py, f64>,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let rgb = rgb.as_array();
    let values = rgb
        .iter()
        .map(|&value| {
            // Clamp retains NaN, maps infinities to endpoints, and preserves signed
            // zero as NumPy clip does. Evaluate only the applicable transfer branch.
            let bounded = value.clamp(0.0, 1.0);
            if bounded <= 0.0031308 {
                12.92 * bounded
            } else {
                1.055 * bounded.powf(1.0 / 2.4) - 0.055
            }
        })
        .collect();
    let result = ArrayD::from_shape_vec(rgb.raw_dim(), values)
        .expect("one output per input value")
        .into_pyarray(py);
    result.call_method1("setflags", (false,))?;
    Ok(result)
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(texture_from_rgba8, module)?)?;
    module.add_function(wrap_pyfunction!(texture_to_rgba8, module)?)?;
    module.add_function(wrap_pyfunction!(texture_to_linear, module)?)?;
    module.add_function(wrap_pyfunction!(texture_from_linear, module)?)?;
    Ok(())
}
