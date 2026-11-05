// SPDX-FileCopyrightText: 2001-2002 NaN Holding BV. All rights reserved.
// SPDX-FileCopyrightText: 2023 Blender Authors
// SPDX-FileCopyrightText: 2026 InterDeCA contributors
// SPDX-License-Identifier: GPL-2.0-or-later
//
// Blender 5.2.1 Smart UV Project, UV island traversal and default AABB packing.
// Specialization: one selected triangle mesh, fresh UV map, no seams/pins,
// image aspect 1, area_weight=0, SCALED margin, AXIS_ALIGNED_Y pre-rotation,
// scale_to_bounds=true. Source provenance is in BLENDER_SOURCE.json.

use super::{boxpack, hull, math, order};
use crate::surface::Mesh;
use std::collections::HashMap;

type P2 = [f32; 2];
type Tri = [P2; 3];
struct Island {
    faces: Vec<usize>,
    angle: f32,
    pivot: P2,
    half: P2,
}

// BMesh's C normal helper multiplies by the reciprocal length. The evaluated
// mesh normal used by baking instead uses BLI_math_vector.hh's division path.
fn normalize(v: math::V3) -> math::V3 {
    let d = math::dot(v, v);
    if d > 1e-35 {
        let reciprocal = 1.0 / d.sqrt();
        v.map(|x| x * reciprocal)
    } else {
        [0.0; 3]
    }
}
fn basis(n: math::V3) -> [math::V3; 2] {
    let f = n[0] * n[0] + n[1] * n[1];
    if f > f32::EPSILON {
        let d = 1.0 / f.sqrt();
        let u = [n[1] * d, -n[0] * d, 0.0];
        [u, [-n[2] * u[1], n[2] * u[0], n[0] * u[1] - n[1] * u[0]]]
    } else {
        [
            [if n[2] < 0.0 { -1.0 } else { 1.0 }, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    }
}
fn project(mesh: &Mesh, angle: f32) -> Vec<Tri> {
    let mut areas = Vec::with_capacity(mesh.faces.len());
    let mut normals = Vec::with_capacity(mesh.faces.len());
    for f in 0..mesh.faces.len() {
        let t = mesh.triangle(f).map(|v| v.to_array().map(|x| x as f32));
        normals.push(normalize(math::cross(
            math::sub(t[0], t[1]),
            math::sub(t[1], t[2]),
        )));
        let mut area = [0.0; 3];
        for c in 0..3 {
            let (a, b) = (t[c], t[(c + 1) % 3]);
            for k in 0..3 {
                area[k] += (a[(k + 1) % 3] - b[(k + 1) % 3]) * (a[(k + 2) % 3] + b[(k + 2) % 3]);
            }
        }
        areas.push(math::dot(area, area).sqrt() * 0.5);
    }
    let mut faces: Vec<_> = (0..mesh.faces.len()).collect();
    order::sort(
        &mut faces,
        |i| if areas[i] > 1e-12 { -areas[i] } else { 0.0 },
    );
    faces.retain(|&i| areas[i] > 1e-12);
    let mut uv = vec![[[0.0; 2]; 3]; mesh.faces.len()];
    if faces.is_empty() {
        return uv;
    }
    let mut seed = normals[faces[0]];
    let mut tagged = vec![false; mesh.faces.len()];
    let mut group = Vec::new();
    let mut projections = Vec::new();
    let half_cos = (angle / 2.0).cos();
    let cosine = angle.cos();
    loop {
        for &f in faces.iter().rev() {
            if !tagged[f] && math::dot(normals[f], seed) > half_cos {
                group.push(f);
                tagged[f] = true;
            }
        }
        let mut average = [0.0; 3];
        for &f in &group {
            for (a, n) in average.iter_mut().zip(normals[f]) {
                *a += n;
            }
        }
        let average = normalize(average);
        if average != [0.0; 3] {
            projections.push(average);
        }
        let mut best = 1.0;
        let mut next = faces[0];
        for &f in faces.iter().rev() {
            if tagged[f] {
                continue;
            }
            let similarity = projections
                .iter()
                .map(|&n| math::dot(n, normals[f]))
                .fold(-1.0, f32::max);
            if similarity < best {
                best = similarity;
                next = f;
            }
        }
        if best < cosine {
            seed = normals[next];
            group.clear();
            group.push(next);
            tagged[next] = true;
        } else {
            break;
        }
    }
    // A face chooses the most aligned projection even if another seed originally
    // collected it. Equal projections keep the earliest normal.
    let axes: Vec<_> = projections.iter().map(|&n| basis(n)).collect();
    for &f in &faces {
        let mut chosen = 0;
        let mut best = math::dot(normals[f], projections[0]);
        for (i, &n) in projections.iter().enumerate().skip(1) {
            let score = math::dot(normals[f], n);
            if score > best {
                chosen = i;
                best = score;
            }
        }
        for c in 0..3 {
            let v = mesh.vertices[mesh.faces[f][c] as usize]
                .to_array()
                .map(|x| x as f32);
            uv[f][c] = axes[chosen].map(|a| math::dot(a, v));
        }
    }
    uv
}
fn islands(mesh: &Mesh, uv: &[Tri]) -> Vec<Island> {
    let mut edges: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    for (f, t) in mesh.faces.iter().enumerate() {
        for c in 0..3 {
            let (a, b) = (t[c], t[(c + 1) % 3]);
            edges
                .entry((a.min(b), a.max(b)))
                .or_default()
                .push(f * 3 + c);
        }
    }
    let mut radial = vec![0; mesh.faces.len() * 3];
    for loops in edges.values() {
        for (i, &l) in loops.iter().enumerate() {
            radial[l] = loops[(i + 1) % loops.len()];
        }
    }
    let mut visited = vec![false; mesh.faces.len()];
    let mut result = Vec::new();
    for seed in 0..mesh.faces.len() {
        if visited[seed] {
            continue;
        }
        visited[seed] = true;
        let mut stack = vec![seed];
        let mut faces = Vec::new();
        // This stack order follows BM_mesh_calc_face_groups, including the
        // radial cycle when more than two faces meet at an edge.
        while let Some(f) = stack.pop() {
            faces.push(f);
            for c in 0..3 {
                let l = f * 3 + c;
                let mut other = radial[l];
                while other != l {
                    let (g, d) = (other / 3, other % 3);
                    let (mut da, mut db) = (d, (d + 1) % 3);
                    if mesh.faces[f][c] != mesh.faces[g][d] {
                        std::mem::swap(&mut da, &mut db);
                    }
                    if !visited[g] && uv[f][c] == uv[g][da] && uv[f][(c + 1) % 3] == uv[g][db] {
                        visited[g] = true;
                        stack.push(g);
                    }
                    other = radial[other];
                }
            }
        }
        result.push(Island {
            faces,
            angle: 0.0,
            pivot: [0.0; 2],
            half: [0.0; 2],
        });
    }
    result
}
fn bounds(points: impl Iterator<Item = P2>) -> (P2, P2) {
    let mut lo = [f32::MAX; 2];
    let mut hi = [-f32::MAX; 2];
    for p in points {
        for a in 0..2 {
            lo[a] = lo[a].min(p[a]);
            hi[a] = hi[a].max(p[a]);
        }
    }
    (lo, hi)
}
fn rotate(p: P2, angle: f32, double_trig: bool) -> P2 {
    let (s, c) = if double_trig {
        ((angle as f64).sin() as f32, (angle as f64).cos() as f32)
    } else {
        (angle.sin(), angle.cos())
    };
    [c * p[0] + s * p[1], -s * p[0] + c * p[1]]
}
fn finalize(island: &mut Island, uv: &[Tri]) {
    let mut points: Vec<_> = island.faces.iter().flat_map(|&f| uv[f]).collect();
    let mut angle = -hull::fit_angle(&points);
    let (lo, hi) = bounds(points.iter().map(|&p| rotate(p, angle, false)));
    if hi[0] - lo[0] > hi[1] - lo[1] {
        angle += std::f32::consts::FRAC_PI_2;
    }
    // M_PI constants promote these expressions to double, while floorf stores
    // its input and result as float. Preserve that boundary explicitly.
    let turns =
        (((angle as f64 + std::f64::consts::FRAC_PI_2) / std::f64::consts::PI) as f32).floor();
    angle = (angle as f64 - turns as f64 * std::f64::consts::PI) as f32;
    island.angle = angle;
    if angle != 0.0 {
        for p in &mut points {
            *p = rotate(*p, angle, true);
        }
    }
    let (lo, hi) = bounds(points.into_iter());
    island.pivot = std::array::from_fn(|a| (lo[a] + hi[a]) * 0.5);
    island.half = std::array::from_fn(|a| (hi[a] - lo[a]) * 0.5);
}
fn alpaca(order: &[usize], sizes: &[P2], start: usize, mut extent: P2, positions: &mut [P2]) -> P2 {
    let mut vertical = extent[0] < extent[1];
    let mut cursor = if vertical {
        [extent[0], 0.0]
    } else {
        [0.0, extent[1]]
    };
    for &i in &order[start..] {
        let d = sizes[i];
        if if vertical {
            extent[1] < cursor[1] + d[1]
        } else {
            extent[0] < cursor[0] + d[0]
        } {
            vertical = extent[0] < extent[1];
            cursor = if vertical {
                [extent[0], 0.0]
            } else {
                [0.0, extent[1]]
            };
        }
        positions[i] = [cursor[0] + d[0] * 0.5, cursor[1] + d[1] * 0.5];
        if vertical {
            cursor[1] += d[1];
            extent[0] = extent[0].max(cursor[0] + d[0]);
            extent[1] = extent[1].max(cursor[1]);
        } else {
            cursor[0] += d[0];
            extent[1] = extent[1].max(cursor[1] + d[1]);
            extent[0] = extent[0].max(cursor[0]);
        }
    }
    extent
}
pub(crate) fn unwrap(
    mesh: Mesh,
    degrees: f64,
    margin: f64,
) -> Result<(Vec<f64>, Vec<i64>), String> {
    if mesh
        .vertices
        .iter()
        .any(|v| v.to_array().iter().any(|&x| !(x as f32).is_finite()))
    {
        return Err("UV coordinates must be representable as float32.".into());
    }
    let mut uv = project(&mesh, degrees.to_radians() as f32);
    let mut islands = islands(&mesh, &uv);
    for island in &mut islands {
        finalize(island, &uv);
    }
    let length_sum: f64 = islands
        .iter()
        .map(|i| ((i.half[0] * 2.0) * (i.half[1] * 2.0)).sqrt() as f64)
        .sum();
    let margin = ((margin as f32 as f64) * length_sum * (0.1f32 as f64)) as f32;
    let sizes: Vec<P2> = islands
        .iter()
        .map(|i| i.half.map(|d| d * 2.0 + 2.0 * margin))
        .collect();
    let mut order: Vec<_> = (0..islands.len()).collect();
    order.sort_by(|&a, &b| (sizes[b][0] * sizes[b][1]).total_cmp(&(sizes[a][0] * sizes[a][1])));
    let cutoff = order.len().min(1024);
    let mut positions = vec![[0.0; 2]; order.len()];
    let fast = alpaca(&order[..cutoff], &sizes, 0, [0.0; 2], &mut positions);
    let (boxes, boxed) = boxpack::pack(
        &order[..cutoff]
            .iter()
            .map(|&i| sizes[i])
            .collect::<Vec<_>>(),
    );
    let (a, b) = (fast[0].max(fast[1]), boxed[0].max(boxed[1]));
    let equal = (a - b).abs() <= f32::EPSILON || a.to_bits().abs_diff(b.to_bits()) <= 64;
    let better = if equal {
        boxed[0] * boxed[1] < fast[0] * fast[1]
    } else {
        b < a
    };
    let extent = if better {
        for (j, &i) in order[..cutoff].iter().enumerate() {
            positions[i] = std::array::from_fn(|a| boxes[j][a] + sizes[i][a] * 0.5);
        }
        alpaca(&order, &sizes, cutoff, boxed, &mut positions)
    } else {
        alpaca(&order, &sizes, 0, [0.0; 2], &mut positions)
    };
    let largest = extent[0].max(extent[1]);
    let scale = if largest > 1e-14 { 1.0 / largest } else { 1.0 };
    let mut ids = vec![0; mesh.faces.len()];
    for (i, island) in islands.iter().enumerate() {
        let mut translation = std::array::from_fn(|a| positions[i][a] - island.pivot[a]);
        if island.angle != 0.0 {
            translation = rotate(translation, -island.angle, true);
        }
        let c = ((island.angle as f64).cos() * scale as f64) as f32;
        let s = ((island.angle as f64).sin() * scale as f64) as f32;
        for &f in &island.faces {
            ids[f] = i as i64;
            for p in &mut uv[f] {
                let x = p[0] as f64 + translation[0] as f64;
                let y = p[1] as f64 + translation[1] as f64;
                *p = [
                    (c as f64 * x + s as f64 * y) as f32,
                    (-s as f64 * x + c as f64 * y) as f32,
                ];
            }
        }
    }
    // Smart Project stretches the entire selected map back to the unit square,
    // including degenerate faces; no extra pixel padding is reserved here.
    let (lo, hi) = bounds(uv.iter().flatten().copied());
    let span: P2 = std::array::from_fn(|a| {
        if hi[a] > lo[a] {
            1.0 / (hi[a] - lo[a])
        } else {
            0.0
        }
    });
    for p in uv.iter_mut().flatten() {
        for a in 0..2 {
            p[a] = (p[a] - lo[a]) * span[a];
        }
    }
    Ok((
        uv.into_iter().flatten().flatten().map(f64::from).collect(),
        ids,
    ))
}
