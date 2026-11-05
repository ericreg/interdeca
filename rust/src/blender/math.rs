// SPDX-FileCopyrightText: 2001-2002 NaN Holding BV. All rights reserved.
// SPDX-FileCopyrightText: 2023 Blender Authors
// SPDX-FileCopyrightText: 2026 InterDeCA contributors
// SPDX-License-Identifier: GPL-2.0-or-later
//
// Blender 5.2.1 BLI_math_vector.hh, mesh_normals.cc::face_normal_calc,
// math_geom.cc::interp_barycentric_tri_v3, and
// bake.cc::calc_point_from_barycentric_extrusion, identity-transform flat-face path.

use parry3d_f64::math::Vector;

pub(crate) type V3 = [f32; 3];
pub(crate) fn sub(a: V3, b: V3) -> V3 {
    std::array::from_fn(|i| a[i] - b[i])
}
pub(crate) fn dot(a: V3, b: V3) -> f32 {
    (a[0] * b[0] + a[1] * b[1]) + a[2] * b[2]
}
pub(crate) fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
pub(crate) fn normalize(v: V3) -> V3 {
    let length = dot(v, v);
    if length > 1e-35 {
        let length = length.sqrt();
        v.map(|x| x / length)
    } else {
        [0.0; 3]
    }
}
pub(crate) fn normalize_c(v: V3) -> V3 {
    let length = dot(v, v);
    if length > 1e-35 {
        let reciprocal = 1.0 / length.sqrt();
        v.map(|x| x * reciprocal)
    } else {
        [0.0; 3]
    }
}
pub(crate) fn face_normal(t: [Vector; 3]) -> V3 {
    let t = t.map(|v| v.to_array().map(|x| x as f32));
    let n = normalize(cross(sub(t[0], t[1]), sub(t[1], t[2])));
    if n == [0.0; 3] { [0.0, 0.0, 1.0] } else { n }
}
pub(crate) fn extruded_point(
    t: [Vector; 3],
    normal: V3,
    u: f32,
    v: f32,
    extrusion: f32,
) -> (Vector, Vector) {
    let t = t.map(|v| v.to_array().map(|x| x as f32));
    let p = std::array::from_fn(|i| {
        let mut p = t[0][i] * u;
        p += t[1][i] * v;
        p += t[2][i] * (1.0f32 - u - v);
        p + normal[i] * extrusion
    });
    // Identity transforms still normalize at the same points in the Blender
    // bake path. Preserve these float32 rounding steps before the BVH query.
    let direction = normalize_c(normal).map(|v| -v);
    let direction = normalize_c(normalize_c(direction));
    (
        Vector::from_array(p.map(f64::from)),
        Vector::from_array(direction.map(f64::from)),
    )
}

/// Position-to-barycentric part of bake.cc's differential helper. Image linear
/// filtering does not use position derivatives, so only u and v are returned.
pub(crate) fn barycentric(t: [V3; 3], point: V3, normal: V3) -> [f32; 2] {
    let n = normal.map(f32::abs);
    let (a, b) = if n[2] >= n[0] && n[2] >= n[1] {
        (0, 1)
    } else if n[1] >= n[0] && n[1] >= n[2] {
        (0, 2)
    } else {
        (1, 2)
    };
    let mut t00 = t[2][a] - t[0][a];
    let mut t01 = t[2][b] - t[0][b];
    let mut t10 = t[2][a] - t[1][a];
    let mut t11 = t[2][b] - t[1][b];
    let determinant = t00 * t11 - t10 * t01;
    let inverse = if determinant != 0.0 {
        1.0 / determinant
    } else {
        0.0
    };
    t00 *= inverse;
    t01 *= inverse;
    t10 *= inverse;
    t11 *= inverse;
    [
        (t[2][a] - point[a]) * t11 - (t[2][b] - point[b]) * t10,
        (t[2][b] - point[b]) * t00 - (t[2][a] - point[a]) * t01,
    ]
}

/// Transfer low-poly pixel position derivatives along the projection ray, then
/// express them in the high-poly triangle's barycentric basis (bake.cc).
pub(crate) fn bake_derivatives(
    low: [V3; 3],
    pixels: [[f32; 2]; 3],
    high: [V3; 3],
    normal: V3,
    direction: V3,
) -> [[f32; 2]; 2] {
    let [p1, p2, p3] = pixels;
    let area = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]);
    let low_uv = if area.abs() > f32::EPSILON {
        let inverse = 1.0 / area;
        [
            [(p2[1] - p3[1]) * inverse, (p3[0] - p2[0]) * inverse],
            [(p3[1] - p1[1]) * inverse, (p1[0] - p3[0]) * inverse],
        ]
    } else {
        [[0.0; 2]; 2]
    };
    let du = sub(low[0], low[2]);
    let dv = sub(low[1], low[2]);
    let mut positions: [V3; 2] = std::array::from_fn(|d| {
        std::array::from_fn(|a| du[a] * low_uv[0][d] + dv[a] * low_uv[1][d])
    });
    let multiplier = 1.0 / dot(direction, normal);
    let tmp = direction.map(|x| x * multiplier);
    for p in &mut positions {
        let factor = -dot(*p, normal);
        for a in 0..3 {
            p[a] += tmp[a] * factor;
        }
    }
    let n = normal.map(f32::abs);
    let (a, b) = if n[2] >= n[0] && n[2] >= n[1] {
        (0, 1)
    } else if n[1] >= n[0] && n[1] >= n[2] {
        (0, 2)
    } else {
        (1, 2)
    };
    let (mut t00, mut t01) = (high[2][a] - high[0][a], high[2][b] - high[0][b]);
    let (mut t10, mut t11) = (high[2][a] - high[1][a], high[2][b] - high[1][b]);
    let determinant = t00 * t11 - t10 * t01;
    let inverse = if determinant != 0.0 {
        1.0 / determinant
    } else {
        0.0
    };
    t00 *= inverse;
    t01 *= inverse;
    t10 *= inverse;
    t11 *= inverse;
    [
        positions.map(|p| p[a] * t11 - p[b] * t10),
        positions.map(|p| p[b] * t00 - p[a] * t01),
    ]
}
