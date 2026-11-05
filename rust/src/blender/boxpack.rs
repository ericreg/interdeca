// SPDX-FileCopyrightText: 2023 Blender Authors
// SPDX-FileCopyrightText: 2026 InterDeCA contributors
// SPDX-License-Identifier: GPL-2.0-or-later
//
// Rust translation of Blender 5.2.1 BLI_box_pack_2d (sort_boxes=false),
// with array indices replacing pointers. Fixed-area packing is not used.

use super::order;

const BL: usize = 0;
const TR: usize = 1;
const TL: usize = 2;
const BR: usize = 3;
const EPSILON: f32 = 1e-7;
const MERGE: f32 = 1e-5;

#[derive(Clone)]
struct Vertex {
    p: [f32; 2],
    free: u8,
    used: bool,
    bias: f32,
    boxes: [Option<usize>; 4],
    cache: [Option<usize>; 4],
}
struct Box2 {
    size: [f32; 2],
    v: [usize; 4],
}

// The two diagonal corners define all four positions until vertices are shared.
fn place(b: &Box2, vertices: &mut [Vertex], p: [f32; 2], quadrant: usize) {
    let lo = [
        if quadrant == BL || quadrant == TL {
            p[0] - b.size[0]
        } else {
            p[0]
        },
        if quadrant == BL || quadrant == BR {
            p[1] - b.size[1]
        } else {
            p[1]
        },
    ];
    let hi = [
        if quadrant == BL || quadrant == TL {
            p[0]
        } else {
            p[0] + b.size[0]
        },
        if quadrant == BL || quadrant == BR {
            p[1]
        } else {
            p[1] + b.size[1]
        },
    ];
    for (i, point) in [lo, hi, [lo[0], hi[1]], [hi[0], lo[1]]]
        .into_iter()
        .enumerate()
    {
        vertices[b.v[i]].p = point;
    }
}
fn intersects(a: &Box2, b: &Box2, vertices: &[Vertex]) -> bool {
    let (al, ah, bl, bh) = (
        vertices[a.v[BL]].p,
        vertices[a.v[TR]].p,
        vertices[b.v[BL]].p,
        vertices[b.v[TR]].p,
    );
    !(al[0] + EPSILON >= bh[0]
        || al[1] + EPSILON >= bh[1]
        || ah[0] - EPSILON <= bl[0]
        || ah[1] - EPSILON <= bl[1])
}
fn mark(vertices: &mut [Vertex], index: usize) {
    vertices[index].used = true;
    vertices[index].bias = (vertices[index].p[0] * vertices[index].p[1]) * 1e-6;
}
fn merge(
    boxes: &mut [Box2],
    vertices: &mut [Vertex],
    a: (usize, usize),
    b: (usize, usize),
    mask: u8,
) {
    let (ai, bi) = (boxes[a.0].v[a.1], boxes[b.0].v[b.1]);
    if vertices[ai].used {
        vertices[ai].free &= vertices[bi].free & !mask;
        boxes[b.0].v[b.1] = ai;
    } else {
        vertices[bi].free &= vertices[ai].free & !mask;
        boxes[a.0].v[a.1] = bi;
    }
}

pub(crate) fn pack(sizes: &[[f32; 2]]) -> (Vec<[f32; 2]>, [f32; 2]) {
    if sizes.is_empty() {
        return (Vec::new(), [0.0; 2]);
    }
    let mut vertices = Vec::with_capacity(sizes.len() * 4);
    let mut boxes = Vec::with_capacity(sizes.len());
    for (i, &size) in sizes.iter().enumerate() {
        boxes.push(Box2 {
            size,
            v: std::array::from_fn(|q| i * 4 + q),
        });
        for opposite in [TR, BL, BR, TL] {
            let mut adjacent = [None; 4];
            adjacent[opposite] = Some(i);
            vertices.push(Vertex {
                p: [0.0; 2],
                free: 15 & !(1 << opposite),
                used: false,
                bias: 0.0,
                boxes: adjacent,
                cache: [None; 4],
            });
        }
    }
    // The first rectangle anchors both axes; subsequent corners may be merged.
    vertices[BL].free = 0;
    vertices[BR].free &= !9;
    vertices[TL].free &= !5;
    place(&boxes[0], &mut vertices, [0.0; 2], TR);
    for i in 0..4 {
        mark(&mut vertices, i);
    }
    let mut candidates = vec![TR, TL, BR];
    let mut extent = sizes[0];
    let mut positions = vec![[0.0; 2]; sizes.len()];
    for current in 1..boxes.len() {
        let size = boxes[current].size;
        order::reentrant_sort(&mut candidates, |v| {
            if vertices[v].free == 0 {
                f32::INFINITY
            } else {
                (vertices[v].p[0] + size[0]).max(vertices[v].p[1] + size[1]) + vertices[v].bias
            }
        });
        while candidates.len() > 1 && vertices[*candidates.last().unwrap()].free == 0 {
            candidates.pop();
        }
        let mut selected = None;
        'search: for &v in &candidates {
            for q in 0..4 {
                if vertices[v].free & (1 << q) == 0 {
                    continue;
                }
                let p = vertices[v].p;
                place(&boxes[current], &mut vertices, p, q);
                let lo = vertices[boxes[current].v[BL]].p;
                if lo[0] < 0.0 || lo[1] < 0.0 {
                    continue;
                }
                if vertices[v].cache[q]
                    .is_some_and(|i| intersects(&boxes[current], &boxes[i], &vertices))
                {
                    continue;
                }
                if let Some(i) =
                    (0..current).find(|&i| intersects(&boxes[current], &boxes[i], &vertices))
                {
                    vertices[v].cache[q] = Some(i);
                    continue;
                }
                selected = Some((v, q));
                break 'search;
            }
        }
        let (v, q) = selected.expect("An outer rectangle corner always permits placement");
        let hi = vertices[boxes[current].v[TR]].p;
        extent = [extent[0].max(hi[0]), extent[1].max(hi[1])];
        vertices[v].free &= !(1 << q);
        boxes[current].v[[TR, BL, BR, TL][q]] = v;
        vertices[v].boxes[q] = Some(current);
        let lo = vertices[boxes[current].v[BL]].p;
        if lo[0] <= 0.0 {
            for corner in [TL, BL] {
                vertices[boxes[current].v[corner]].free &= !5;
            }
        } else if lo[1] <= 0.0 {
            for corner in [BL, BR] {
                vertices[boxes[current].v[corner]].free &= !9;
            }
        }

        // Equal adjacent sides share their far corner. Unequal sides invalidate
        // quadrants that are already covered by the taller or wider rectangle.
        let adj = vertices[v].boxes;
        if let (Some(l), Some(r)) = (adj[TL], adj[TR])
            && (current == l || current == r)
        {
            if (boxes[l].size[1] - boxes[r].size[1]).abs() < MERGE {
                merge(&mut boxes, &mut vertices, (r, TL), (l, TR), 9);
            } else if boxes[l].size[1] > boxes[r].size[1] {
                vertices[boxes[r].v[TL]].free &= !5;
            } else {
                vertices[boxes[l].v[TR]].free &= !10;
            }
        } else if let (Some(l), Some(r)) = (adj[BL], adj[BR])
            && (current == l || current == r)
        {
            if (boxes[l].size[1] - boxes[r].size[1]).abs() < MERGE {
                merge(&mut boxes, &mut vertices, (l, BR), (r, BL), 6);
            } else if boxes[l].size[1] > boxes[r].size[1] {
                vertices[boxes[r].v[BL]].free &= !5;
            } else {
                vertices[boxes[l].v[BR]].free &= !10;
            }
        }
        if let (Some(t), Some(b)) = (adj[TL], adj[BL])
            && (current == t || current == b)
        {
            if (boxes[t].size[0] - boxes[b].size[0]).abs() < MERGE {
                merge(&mut boxes, &mut vertices, (b, TL), (t, BL), 10);
            } else if boxes[t].size[0] > boxes[b].size[0] {
                vertices[boxes[b].v[TL]].free &= !6;
            } else {
                vertices[boxes[t].v[BL]].free &= !9;
            }
        } else if let (Some(t), Some(b)) = (adj[TR], adj[BR])
            && (current == t || current == b)
        {
            if (boxes[t].size[0] - boxes[b].size[0]).abs() < MERGE {
                merge(&mut boxes, &mut vertices, (b, TR), (t, BR), 5);
            } else if boxes[t].size[0] > boxes[b].size[0] {
                vertices[boxes[b].v[TR]].free &= !6;
            } else {
                vertices[boxes[t].v[BR]].free &= !9;
            }
        }
        for corner in boxes[current].v {
            if !vertices[corner].used {
                mark(&mut vertices, corner);
                candidates.push(corner);
            }
        }
        positions[current] = vertices[boxes[current].v[BL]].p;
    }
    (positions, extent)
}
