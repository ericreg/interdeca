// SPDX-FileCopyrightText: 2001-2002 NaN Holding BV. All rights reserved.
// SPDX-FileCopyrightText: 2024 Blender Authors
// SPDX-FileCopyrightText: 2026 InterDeCA contributors
// SPDX-License-Identifier: GPL-2.0-or-later
//
// Blender 5.2.1 math_geom.cc watertight triangle test and bvhutils.cc triangle
// callback. The source-derived BVH retains the corresponding traversal order.

use super::math::{V3, sub};

pub(crate) struct RayTest {
    pub origin: V3,
    pub direction: V3,
    axes: [usize; 3],
    shear: V3,
}
impl RayTest {
    pub(crate) fn new(origin: V3, direction: V3) -> Self {
        let d = direction.map(f32::abs);
        let z = if d[0] > d[1] {
            if d[0] > d[2] { 0 } else { 2 }
        } else if d[1] > d[2] {
            1
        } else {
            2
        };
        let mut x = (z + 1) % 3;
        let mut y = (x + 1) % 3;
        if direction[z] < 0.0 {
            std::mem::swap(&mut x, &mut y);
        }
        let inverse = 1.0 / direction[z];
        Self {
            origin,
            direction,
            axes: [x, y, z],
            shear: [direction[x] * inverse, direction[y] * inverse, inverse],
        }
    }
    pub(crate) fn intersect(&self, triangle: [V3; 3], maximum: f64) -> Option<f64> {
        let [x, y, z] = self.axes;
        let [sx, sy, sz] = self.shear;
        let [a, b, c] = triangle.map(|p| sub(p, self.origin));
        let (ax, ay) = (a[x] - sx * a[z], a[y] - sy * a[z]);
        let (bx, by) = (b[x] - sx * b[z], b[y] - sy * b[z]);
        let (cx, cy) = (c[x] - sx * c[z], c[y] - sy * c[z]);
        let (u, v, w) = (cx * by - cy * bx, ax * cy - ay * cx, bx * ay - by * ax);
        if (u < 0.0 || v < 0.0 || w < 0.0) && (u > 0.0 || v > 0.0 || w > 0.0) {
            return None;
        }
        let determinant = u + v + w;
        if determinant == 0.0 || !determinant.is_finite() {
            return None;
        }
        let t = (u * a[z] + v * b[z] + w * c[z]) * sz;
        let signed = f32::from_bits(t.to_bits() ^ (determinant.to_bits() & 0x80000000));
        if signed < 0.0 {
            return None;
        }
        let distance = t * (1.0 / determinant);
        (distance >= 0.0 && (distance as f64) < maximum).then_some(distance as f64)
    }
    pub(crate) fn point(&self, distance: f32) -> V3 {
        // The mesh callback stores the float32 hit position before applying the
        // strict squared world-distance cutoff in cast_ray_highpoly.
        std::array::from_fn(|i| self.origin[i] + self.direction[i] * distance)
    }
}
