// SPDX-FileCopyrightText: 2006 NaN Holding BV. All rights reserved.
// SPDX-FileCopyrightText: 2026 InterDeCA contributors
// SPDX-License-Identifier: GPL-2.0-or-later
//
// Blender 5.2.1 BLI_kdopbvh.cc: binary, six-axis, zero-radius triangle BVH.
// Array indices replace pointers; partitioning and traversal order are retained
// because equal-distance hits select the first visited source triangle.

use super::math::V3;
use super::ray::RayTest;

#[derive(Clone, Copy)]
struct Node {
    bounds: [f32; 6],
    children: [usize; 2],
    count: usize,
    axis: usize,
}
const EMPTY: Node = Node {
    bounds: [
        f32::MAX,
        -f32::MAX,
        f32::MAX,
        -f32::MAX,
        f32::MAX,
        -f32::MAX,
    ],
    children: [0; 2],
    count: 0,
    axis: 0,
};

pub(crate) struct Tree {
    nodes: Vec<Node>,
    leaves: usize,
}

fn longest(b: [f32; 6]) -> usize {
    let [x, y, z] = [b[1] - b[0], b[3] - b[2], b[5] - b[4]];
    if x > y {
        if x > z { 0 } else { 2 }
    } else if y > z {
        1
    } else {
        2
    }
}

fn partition(
    ids: &mut [usize],
    nodes: &[Node],
    mut begin: usize,
    mut end: usize,
    n: usize,
    axis: usize,
) {
    let key = |id: usize| nodes[id].bounds[2 * axis + 1];
    // Match the source median-of-three quickselect, including equal-key swaps.
    // A stable sort would change triangle ownership at edges and vertices.
    while end - begin > 3 {
        let (lo, mid, hi) = (ids[begin], ids[(begin + end) / 2], ids[end - 1]);
        let pivot = if key(mid) < key(lo) {
            if key(hi) < key(mid) {
                mid
            } else if key(hi) < key(lo) {
                hi
            } else {
                lo
            }
        } else if key(hi) < key(mid) {
            if key(hi) < key(lo) { lo } else { hi }
        } else {
            mid
        };
        let (mut i, mut j) = (begin, end);
        let cut = loop {
            while key(ids[i]) < key(pivot) {
                i += 1;
            }
            j -= 1;
            while key(pivot) < key(ids[j]) {
                j -= 1;
            }
            if i >= j {
                break i;
            }
            ids.swap(i, j);
            i += 1;
        };
        if cut <= n {
            begin = cut;
        } else {
            end = cut;
        }
    }
    for i in begin..end {
        let id = ids[i];
        let mut j = i;
        while j != begin && key(id) < key(ids[j - 1]) {
            ids[j] = ids[j - 1];
            j -= 1;
        }
        ids[j] = id;
    }
}

impl Tree {
    pub(crate) fn new(triangles: &[[V3; 3]]) -> Self {
        let leaves = triangles.len();
        let branches = leaves.saturating_sub(1).max(1);
        let mut nodes = vec![EMPTY; leaves + branches];
        // Blender expands each leaf by FLT_EPSILON before refitting parents.
        for (node, triangle) in nodes.iter_mut().zip(triangles) {
            for a in 0..3 {
                node.bounds[2 * a] =
                    triangle.iter().map(|p| p[a]).fold(f32::MAX, f32::min) - f32::EPSILON;
                node.bounds[2 * a + 1] =
                    triangle.iter().map(|p| p[a]).fold(-f32::MAX, f32::max) + f32::EPSILON;
            }
        }
        if leaves == 0 {
            return Self { nodes, leaves };
        }
        if leaves == 1 {
            nodes[1] = Node {
                bounds: nodes[0].bounds,
                children: [0, 0],
                count: 1,
                axis: longest(nodes[0].bounds),
            };
            return Self { nodes, leaves };
        }
        let power = leaves.next_power_of_two();
        let remaining = 2 * (leaves - power / 2);
        // The implicit shape places leaves on two adjacent depths. These index
        // formulas retain Blender's balanced shape even for non-power-of-two N.
        let leaf_index = |depth: usize, child: usize| {
            let previous = power >> (depth - 1);
            let current = power >> depth;
            let minimum = child * previous;
            if minimum <= remaining {
                minimum
            } else if current != 0 {
                leaves - ((1usize << (depth - 1)) - child) * current
            } else {
                remaining
            }
        };
        let mut ids: Vec<_> = (0..leaves).collect();
        let (mut level, mut depth) = (1, 1);
        while level <= branches {
            let next = level * 2;
            for j in level..next.min(branches + 1) {
                let lo = leaf_index(depth, j - level);
                let hi = leaf_index(depth, j - level + 1);
                let mut parent = EMPTY;
                for &id in &ids[lo..hi] {
                    for a in 0..3 {
                        parent.bounds[2 * a] = parent.bounds[2 * a].min(nodes[id].bounds[2 * a]);
                        parent.bounds[2 * a + 1] =
                            parent.bounds[2 * a + 1].max(nodes[id].bounds[2 * a + 1]);
                    }
                }
                parent.axis = longest(parent.bounds);
                let mid = leaf_index(depth + 1, j * 2 + 1 - next);
                partition(&mut ids, &nodes, lo, hi, mid, parent.axis);
                for k in 0..2 {
                    let child = j * 2 + k;
                    let begin = leaf_index(depth + 1, child - next);
                    let end = leaf_index(depth + 1, child - next + 1);
                    if end == begin {
                        break;
                    }
                    parent.children[k] = if end - begin > 1 {
                        leaves + child - 1
                    } else {
                        ids[begin]
                    };
                    parent.count += 1;
                }
                nodes[leaves + j - 1] = parent;
            }
            level = next;
            depth += 1;
        }
        Self { nodes, leaves }
    }

    pub(crate) fn cast(&self, ray: &RayTest, triangles: &[[V3; 3]]) -> Option<(usize, f32)> {
        if self.leaves == 0 {
            return None;
        }
        let mut dots = ray.direction;
        let mut inverse = [0.0; 3];
        let mut near = [0; 3];
        for a in 0..3 {
            if dots[a].abs() < f32::EPSILON {
                dots[a] = 0.0;
                inverse[a] = f32::MAX;
            } else {
                inverse[a] = 1.0 / dots[a];
            }
            near[a] = 2 * a + usize::from(inverse[a] < 0.0);
        }
        let mut best = f32::MAX / 2.0;
        let mut hit = None;
        // Binary depth is bounded by the number of bits in usize. A stack on
        // the caller's frame avoids allocating for every texture pixel.
        let mut stack = [0usize; usize::BITS as usize + 1];
        stack[0] = self.leaves;
        let mut count = 1;
        while count != 0 {
            count -= 1;
            let id = stack[count];
            let node = self.nodes[id];
            let mut t1 = [0.0; 3];
            let mut t2 = [0.0; 3];
            for a in 0..3 {
                t1[a] = (node.bounds[near[a]] - ray.origin[a]) * inverse[a];
                t2[a] = (node.bounds[2 * a + 1 - (near[a] % 2)] - ray.origin[a]) * inverse[a];
            }
            if t1[0] > t2[1]
                || t2[0] < t1[1]
                || t1[0] > t2[2]
                || t2[0] < t1[2]
                || t1[1] > t2[2]
                || t2[1] < t1[2]
                || t2.iter().any(|&x| x < 0.0)
                || t1.iter().any(|&x| x > best)
                || t1[0].max(t1[1]).max(t1[2]) >= best
            {
                continue;
            }
            if node.count == 0 {
                if let Some(distance) = ray.intersect(triangles[id], f64::from(best)) {
                    best = distance as f32;
                    hit = Some((id, best));
                }
            } else {
                // Zero split-axis direction uses reverse child order in the
                // source. Strict leaf comparison then preserves the first tie.
                for k in 0..node.count {
                    let child = if dots[node.axis] > 0.0 {
                        node.count - 1 - k
                    } else {
                        k
                    };
                    stack[count] = node.children[child];
                    count += 1;
                }
            }
        }
        hit
    }
}
