// SPDX-FileCopyrightText: 2026 InterDeCA contributors
// SPDX-License-Identifier: GPL-3.0-or-later
//! CPU color transfer; rasterization follows the pinned GPL Blender source.

use crate::surface::Mesh;
use crate::uv::Tri2;
use numpy::ndarray::Array3;
use numpy::{IntoPyArray, PyArray3, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods};
use parry3d_f64::math::Vector;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

struct Texture {
    bytes: Vec<u8>,
    width: usize,
    height: usize,
}
impl Texture {
    fn sample(&self, uv: [f32; 2]) -> [u8; 4] {
        crate::blender::image::sample(&self.bytes, self.width, self.height, uv)
    }
}
fn uv_input(uv: &PyReadonlyArray3<'_, f64>, faces: usize, atlas: bool) -> PyResult<Vec<Tri2>> {
    crate::require_aligned(uv)?;
    if uv.shape() != [faces, 3, 2] {
        return Err(PyValueError::new_err("UVs must have shape (faces,3,2)."));
    }
    let a = uv.as_array();
    let mut out = Vec::with_capacity(faces);
    for f in 0..faces {
        let t = std::array::from_fn(|c| [a[[f, c, 0]], a[[f, c, 1]]]);
        if t.iter()
            .flatten()
            .any(|x| !(*x as f32).is_finite() || (atlas && !(0.0..=1.0).contains(x)))
        {
            return Err(PyValueError::new_err(
                "UVs must be finite; target UVs must lie in [0,1].",
            ));
        }
        out.push(t);
    }
    Ok(out)
}
struct Baker {
    source: Vec<[crate::blender::math::V3; 3]>,
    source_bvh: crate::blender::bvh::Tree,
    source_uv: Vec<Tri2>,
    target: Mesh,
    target_uv: Vec<Tri2>,
    normals: Vec<crate::blender::math::V3>,
    texture: Texture,
    width: usize,
    height: usize,
    extrusion: f64,
    max_distance: f64,
}
impl Baker {
    fn cast(&self, face: usize, b: [f64; 3]) -> Option<[u8; 4]> {
        let (origin, direction) = crate::blender::math::extruded_point(
            self.target.triangle(face),
            self.normals[face],
            b[0] as f32,
            b[1] as f32,
            self.extrusion as f32,
        );
        let test = crate::blender::ray::RayTest::new(
            origin.to_array().map(|x| x as f32),
            direction.to_array().map(|x| x as f32),
        );
        let (triangle, distance) = self.source_bvh.cast(&test, &self.source)?;
        let point = test.point(distance);
        let delta = crate::blender::math::sub(point, test.origin);
        if crate::blender::math::dot(delta, delta) >= self.max_distance as f32 {
            return None;
        }
        let t = self.source[triangle];
        let normal =
            crate::blender::math::face_normal(t.map(|p| Vector::from_array(p.map(f64::from))));
        let mut barycentric = crate::blender::math::barycentric(t, point, normal);
        if (barycentric[0] == 0.0 || barycentric[0] == 1.0)
            && (barycentric[1] == 0.0 || barycentric[1] == 1.0)
        {
            barycentric = crate::blender::bake_sampling::nudge(t, barycentric);
        }
        if barycentric[0] <= 0.0 || barycentric[1] <= 0.0 || barycentric[0] + barycentric[1] >= 1.0
        {
            // Sample zero keeps interior points at their center. Boundary points
            // need Blender's position derivatives and hashed retry sequence.
            let low = self
                .target
                .triangle(face)
                .map(|p| p.to_array().map(|x| x as f32));
            let pixels = crate::blender::raster::pixel_triangle(
                self.target_uv[face],
                self.width,
                self.height,
            );
            let derivatives =
                crate::blender::math::bake_derivatives(low, pixels, t, normal, test.direction);
            barycentric = crate::blender::bake_sampling::jitter(barycentric, derivatives);
        }
        let uv = crate::blender::image::uv(self.source_uv[triangle], barycentric);
        Some(self.texture.sample(uv))
    }
    fn bake(&self, margin: usize) -> Result<Vec<u8>, String> {
        // The source-derived scan converter supplies Blender's edge ownership
        // and float32 barycentric coordinates. Preserve face order: later faces
        // overwrite earlier ones at overlaps, exactly as store_bake_pixel does.
        let mut owners = vec![-1i64; self.width * self.height];
        let mut weights = vec![[0.0f32; 2]; owners.len()];
        let mut raster = crate::blender::raster::Rasterizer::new(self.width, self.height);
        for (f, &uv) in self.target_uv.iter().enumerate() {
            raster.scan(
                crate::blender::raster::pixel_triangle(uv, self.width, self.height),
                |x, y, u, v| {
                    let i = (self.height - 1 - y) * self.width + x;
                    owners[i] = f as i64;
                    weights[i] = [u, v];
                },
            );
        }
        let mut colors = vec![0u8; self.width * self.height * 4];
        crate::spatial::pool()?.install(|| {
            colors.par_chunks_mut(4).enumerate().for_each(|(i, pixel)| {
                if owners[i] < 0 {
                    return;
                }
                let [u, v] = weights[i];
                if let Some(color) = self.cast(
                    owners[i] as usize,
                    [f64::from(u), f64::from(v), f64::from(1.0f32 - u - v)],
                ) {
                    pixel.copy_from_slice(&color);
                }
            });
        });
        if margin == 0 {
            return Ok(colors);
        }
        // RE_bake_pixels_populate_from_objects invalidates the low-poly pixel
        // when its projection ray misses. The margin mask therefore follows
        // successful projections, not merely the target UV rasterization.
        for (owner, pixel) in owners.iter_mut().zip(colors.chunks_exact(4)) {
            if pixel[3] == 0 {
                *owner = -1;
            }
        }
        // The source-derived adjacent-face margin performs two preliminary
        // averaging passes, Dijkstra propagation, UV-neighbor lookup, and final
        // corner extension. It uses coverage independently of image alpha.
        crate::blender::margin::apply(
            &mut colors,
            &owners,
            &self.target,
            &self.target_uv,
            self.width,
            self.height,
            margin,
        );
        Ok(colors)
    }
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn surface_bake<'py>(
    py: Python<'py>,
    source_vertices: PyReadonlyArray2<'py, f64>,
    source_faces: PyReadonlyArray2<'py, i64>,
    source_uv: PyReadonlyArray3<'py, f64>,
    target_vertices: PyReadonlyArray2<'py, f64>,
    target_faces: PyReadonlyArray2<'py, i64>,
    target_uv: PyReadonlyArray3<'py, f64>,
    pixels: PyReadonlyArray3<'py, u8>,
    width: usize,
    height: usize,
    extrusion: f64,
    max_distance: f64,
    margin: usize,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let mut source = Mesh::read(&source_vertices, &source_faces)?;
    let mut target = Mesh::read(&target_vertices, &target_faces)?;
    // Blender's Mesh stores coordinates in float32 before normals or ray setup.
    // Preserve those values when crossing the Python float64 boundary.
    for mesh in [&mut source, &mut target] {
        for p in &mut mesh.vertices {
            *p = Vector::from_array(p.to_array().map(|x| f64::from(x as f32)));
            if !p.is_finite() {
                return Err(PyValueError::new_err(
                    "Blender-compatible coordinates exceed float32 range.",
                ));
            }
        }
    }
    let source_uv = uv_input(&source_uv, source.faces.len(), false)?;
    let target_uv = uv_input(&target_uv, target.faces.len(), true)?;
    if width == 0
        || height == 0
        || width > 8192
        || height > 8192
        || margin > 64
        || !extrusion.is_finite()
        || !(extrusion as f32).is_finite()
        || extrusion < 0.0
        || !max_distance.is_finite()
        || !(max_distance as f32).is_finite()
        || max_distance < 0.0
        || pixels.shape()[0] == 0
        || pixels.shape()[1] == 0
        || pixels.shape()[2] != 4
    {
        return Err(PyValueError::new_err(
            "Invalid bake settings or RGBA image shape.",
        ));
    }
    let mut texture = Texture {
        bytes: pixels.as_array().iter().copied().collect(),
        width: pixels.shape()[1],
        height: pixels.shape()[0],
    };
    let colors = py
        .detach(move || -> Result<Vec<u8>, String> {
            crate::blender::image::associate_alpha(&mut texture.bytes);
            let normals = (0..target.faces.len())
                .map(|f| crate::blender::math::face_normal(target.triangle(f)))
                .collect();
            let source: Vec<_> = (0..source.faces.len())
                .map(|f| source.triangle(f).map(|p| p.to_array().map(|x| x as f32)))
                .collect();
            // Retain Blender's BVH shape and visitation order: a ray on a
            // shared vertex may hit several faces at the same distance.
            let source_bvh = crate::blender::bvh::Tree::new(&source);
            let baker = Baker {
                source,
                source_bvh,
                source_uv,
                target,
                target_uv,
                normals,
                texture,
                width,
                height,
                extrusion,
                max_distance: {
                    let squared = (max_distance as f32) * (max_distance as f32);
                    f64::from(if squared == 0.0 { f32::MAX } else { squared })
                },
            };
            baker.bake(margin)
        })
        .map_err(PyValueError::new_err)?;
    let colors = Array3::from_shape_vec((height, width, 4), colors)
        .unwrap()
        .into_pyarray(py);
    colors.call_method1("setflags", (false,))?;
    Ok(colors)
}
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(surface_bake, module)?)?;
    Ok(())
}
