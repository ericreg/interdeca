// SPDX-FileCopyrightText: 2023 Blender Authors
// SPDX-FileCopyrightText: 2026 InterDeCA contributors
// SPDX-License-Identifier: GPL-2.0-or-later
//
// Rust adaptation of Blender 5.2.1 (9e2066aef7ef7e20c142ad7bd3303138a4304c93):
// - BLI_kdtree.hh: kdtree_balance, kdtree_range_search_cb, kdtree_calc_duplicates_cb
// - bmo_removedoubles.cc: bmesh_find_doubles_by_distance_impl,
//   bmo_weld_verts_exec, remdoubles_createface
// Scope: all input vertices selected, no keep-vertex set, Euclidean radius.
// See BLENDER_SOURCE.json and THIRD_PARTY_NOTICES.md for provenance and changes.

use parry3d_f64::math::Vector;

#[derive(Clone, Copy)]
struct Node {
    point: [f32; 3],
    index: usize,
    left: Option<usize>,
    right: Option<usize>,
    axis: usize,
}

// Blender's right-pivot partition must be retained: equal coordinates change
// storage order, which subsequently changes the order in which clusters form.
fn balance(nodes: &mut [Node], axis: usize, offset: usize) -> Option<usize> {
    if nodes.is_empty() {
        return None;
    }
    if nodes.len() == 1 {
        return Some(offset);
    }
    let mut left = 0isize;
    let mut right = nodes.len() as isize - 1;
    let median = nodes.len() / 2;
    while right > left {
        let pivot = nodes[right as usize].point[axis];
        let mut i = left - 1;
        let mut j = right;
        loop {
            loop {
                i += 1;
                if nodes[i as usize].point[axis] >= pivot {
                    break;
                }
            }
            loop {
                j -= 1;
                if nodes[j as usize].point[axis] <= pivot || j <= left {
                    break;
                }
            }
            if i >= j {
                break;
            }
            nodes.swap(i as usize, j as usize);
        }
        nodes.swap(i as usize, right as usize);
        if i >= median as isize {
            right = i - 1;
        }
        if i <= median as isize {
            left = i + 1;
        }
    }
    let (before, rest) = nodes.split_at_mut(median);
    let (middle, after) = rest.split_first_mut().unwrap();
    middle.axis = axis;
    middle.left = balance(before, (axis + 1) % 3, offset);
    middle.right = balance(after, (axis + 1) % 3, offset + median + 1);
    Some(offset + median)
}

// Explicit component order reproduces float3 dot products without f64 promotion
// or fused operations. This is part of the merge decision, not a tolerance tweak.
fn distance_squared(a: [f32; 3], b: [f32; 3]) -> f32 {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    (d[0] * d[0] + d[1] * d[1]) + d[2] * d[2]
}

fn range_search(nodes: &[Node], root: usize, point: [f32; 3], radius: f32, found: &mut Vec<usize>) {
    let radius_squared = radius * radius;
    let mut stack = vec![root];
    while let Some(i) = stack.pop() {
        let n = nodes[i];
        if point[n.axis] + radius < n.point[n.axis] {
            if let Some(left) = n.left {
                stack.push(left);
            }
        } else if point[n.axis] - radius > n.point[n.axis] {
            if let Some(right) = n.right {
                stack.push(right);
            }
        } else {
            if distance_squared(n.point, point) <= radius_squared {
                found.push(n.index);
            }
            // This push order visits the right subtree first, as in Blender.
            if let Some(left) = n.left {
                stack.push(left);
            }
            if let Some(right) = n.right {
                stack.push(right);
            }
        }
    }
}

pub(crate) fn representatives(vertices: &[Vector], distance: f64) -> Result<Vec<usize>, String> {
    let radius = distance as f32;
    let points: Vec<[f32; 3]> = vertices
        .iter()
        .map(|v| v.to_array().map(|x| x as f32))
        .collect();
    if !radius.is_finite() || points.iter().flatten().any(|x| !x.is_finite()) {
        return Err(
            "Blender-compatible cleanup requires coordinates and radius within float32 range."
                .into(),
        );
    }
    let mut nodes: Vec<Node> = points
        .iter()
        .enumerate()
        .map(|(i, &p)| Node {
            point: p,
            index: i,
            left: None,
            right: None,
            axis: 0,
        })
        .collect();
    let Some(root) = balance(&mut nodes, 0, 0) else {
        return Ok(Vec::new());
    };
    let mut duplicates = vec![None; vertices.len()];
    let mut cluster = Vec::new();
    for node in &nodes {
        let index = node.index;
        if duplicates[index].is_some() {
            continue;
        }
        cluster.clear();
        range_search(&nodes, root, points[index], radius, &mut cluster);
        cluster.retain(|&i| duplicates[i].is_none());
        if cluster.is_empty() {
            continue;
        }
        // Upstream appends the origin after the range query, even though it is
        // also returned by that query. Preserve this extra centroid weight.
        cluster.push(index);
        let best = if cluster.len() == 2 {
            if cluster[0] < cluster[1] { 0 } else { 1 }
        } else {
            let mut centroid = [0.0f32; 3];
            for &i in &cluster {
                for a in 0..3 {
                    centroid[a] += points[i][a];
                }
            }
            for c in &mut centroid {
                *c /= cluster.len() as f32;
            }
            let mut best = cluster.len() - 1;
            let mut best_distance = distance_squared(centroid, points[cluster[best]]);
            for i in 0..cluster.len() - 1 {
                let d = distance_squared(centroid, points[cluster[i]]);
                if d > best_distance || (d == best_distance && cluster[i] > cluster[best]) {
                    continue;
                }
                best = i;
                best_distance = d;
            }
            best
        };
        let representative = cluster[best];
        for &i in &cluster {
            duplicates[i] = Some(representative);
        }
    }
    Ok(duplicates
        .into_iter()
        .enumerate()
        .map(|(i, d)| d.unwrap_or(i))
        .collect())
}

// Adapt bmo_weld_verts_exec/remdoubles_createface to triangle-only inputs.
// Faces unaffected by the vertex map remain in place, even when duplicates
// already existed. A changed face reuses an existing mapped face, or replaces
// its own geometry in place. Three-corner faces cannot require polygon splitting.
pub(crate) fn weld_triangles(
    faces: &[[u32; 3]],
    representatives: &[usize],
) -> (Vec<[usize; 3]>, Vec<i64>) {
    use std::collections::HashMap;
    let key = |mut f: [usize; 3]| {
        f.sort_unstable();
        f
    };
    let mut current: Vec<[usize; 3]> = faces.iter().map(|f| f.map(|i| i as usize)).collect();
    let mut lookup: HashMap<[usize; 3], Vec<usize>> = HashMap::new();
    for (i, &f) in current.iter().enumerate() {
        lookup.entry(key(f)).or_default().push(i);
    }
    let mut target: Vec<Option<usize>> = (0..faces.len()).map(Some).collect();
    for (i, original) in faces.iter().enumerate() {
        let mapped = original.map(|v| representatives[v as usize]);
        if mapped == current[i] {
            continue;
        }
        if let Some(entries) = lookup.get_mut(&key(current[i])) {
            entries.retain(|&f| f != i);
        }
        if mapped[0] == mapped[1] || mapped[1] == mapped[2] || mapped[2] == mapped[0] {
            target[i] = None;
            continue;
        }
        if let Some(&existing) = lookup.get(&key(mapped)).and_then(|entries| entries.first()) {
            target[i] = Some(existing);
        } else {
            current[i] = mapped;
            lookup.entry(key(mapped)).or_default().push(i);
        }
    }
    let mut compact = vec![-1; faces.len()];
    let mut output = Vec::new();
    for (i, &f) in current.iter().enumerate() {
        if target[i] == Some(i) {
            compact[i] = output.len() as i64;
            output.push(f);
        }
    }
    let mapping = target
        .into_iter()
        .map(|f| f.map_or(-1, |i| compact[i]))
        .collect();
    (output, mapping)
}
