// SPDX-FileCopyrightText: 2011-2022 Blender Foundation
// SPDX-FileCopyrightText: 2026 InterDeCA contributors
// SPDX-License-Identifier: Apache-2.0
//
// Blender 5.2.1 Cycles init_from_bake.h's vertex nudge and barycentric jitter,
// and util/hash.h's float2 hash. Only sample zero is used by color-only baking.

use super::math::{V3, dot, normalize, sub};

fn hash(values: &[u32]) -> f32 {
    let mut a = 0xdeadbeefu32
        .wrapping_add((values.len() as u32) << 2)
        .wrapping_add(13);
    let mut b = a;
    let mut c = a;
    a = a.wrapping_add(values[0]);
    b = b.wrapping_add(values[1]);
    if values.len() == 3 {
        c = c.wrapping_add(values[2]);
    }
    c ^= b;
    c = c.wrapping_sub(b.rotate_left(14));
    a ^= c;
    a = a.wrapping_sub(c.rotate_left(11));
    b ^= a;
    b = b.wrapping_sub(a.rotate_left(25));
    c ^= b;
    c = c.wrapping_sub(b.rotate_left(16));
    a ^= c;
    a = a.wrapping_sub(c.rotate_left(4));
    b ^= a;
    b = b.wrapping_sub(a.rotate_left(14));
    c ^= b;
    c = c.wrapping_sub(b.rotate_left(24));
    c as f32 * (1.0f32 / (u32::MAX as f32))
}
fn hash2(p: [f32; 2]) -> [f32; 2] {
    [
        hash(&[p[0].to_bits(), p[1].to_bits()]),
        hash(&[p[0].to_bits(), p[1].to_bits(), 1.0f32.to_bits()]),
    ]
}
pub(crate) fn nudge(triangle: [V3; 3], uv: [f32; 2]) -> [f32; 2] {
    let [u, v] = uv;
    let p: V3 = std::array::from_fn(|a| {
        (u * triangle[0][a] + v * triangle[1][a]) + (1.0 - u - v) * triangle[2][a]
    });
    let center: V3 =
        std::array::from_fn(|a| ((triangle[0][a] + triangle[1][a]) + triangle[2][a]) / 3.0);
    let to_center = sub(center, p);
    let distance = dot(to_center, to_center).sqrt();
    let scale = distance.min(p.map(f32::abs).into_iter().fold(1.0, f32::max) * 1e-4);
    let direction = normalize(to_center);
    let offset: V3 = std::array::from_fn(|a| p[a] + direction[a] * scale);
    let v1 = sub(triangle[1], triangle[0]);
    let v2 = sub(triangle[2], triangle[0]);
    let vp = sub(offset, triangle[0]);
    let (d11, d12, d22, dp1, dp2) = (
        dot(v1, v1),
        dot(v1, v2),
        dot(v2, v2),
        dot(vp, v1),
        dot(vp, v2),
    );
    let denom = d11 * d22 - d12 * d12;
    if denom == 0.0 {
        return [0.0; 2];
    }
    let offset_v = ((d22 * dp1 - d12 * dp2) / denom).clamp(1e-5, 1.0 - 1e-5);
    let offset_w = ((d11 * dp2 - d12 * dp1) / denom).clamp(1e-5, 1.0 - 1e-5);
    [
        (1.0 - offset_v - offset_w).clamp(1e-5, 1.0 - 1e-5),
        offset_v,
    ]
}
pub(crate) fn jitter(uv: [f32; 2], derivatives: [[f32; 2]; 2]) -> [f32; 2] {
    let [u, v] = uv;
    let [[dudx, dudy], [dvdx, dvdy]] = derivatives;
    let mut filter = [0.5; 2];
    // The first sample uses the pixel center. Only an edge/outside point retries
    // with hashes; wholly valid interior samples incur no random displacement.
    for _ in 0..10 {
        let jitter_u = (u + (filter[0] - 0.5) * dudx) + (filter[1] - 0.5) * dudy;
        let jitter_v = (v + (filter[0] - 0.5) * dvdx) + (filter[1] - 0.5) * dvdy;
        if jitter_u > 0.0 && jitter_v > 0.0 && jitter_u + jitter_v < 1.0 {
            return [jitter_u, jitter_v];
        }
        filter = hash2(filter);
    }
    uv
}
