// SPDX-FileCopyrightText: 2001-2002 NaN Holding BV. All rights reserved.
// SPDX-FileCopyrightText: 2024 Blender Authors
// SPDX-FileCopyrightText: 2026 InterDeCA contributors
// SPDX-License-Identifier: GPL-2.0-or-later
//
// Blender 5.2.1 translation: render/intern/texture_margin.cc,
// imbuf/intern/filter.cc (IMB_filter_extend, byte/mask branch), and
// blenlib/intern/math_interp.cc (bilinear_byte_impl scalar bordered branch).
// Only triangle faces, RGBA8 buffers, and a single UV tile are required here.

use crate::surface::Mesh;
use crate::uv::Tri2;
use std::collections::HashMap;

const UNSET: u32 = u32::MAX;
const DIJKSTRA: u32 = 0x80000000;
const DIRECTIONS: [[i32; 2]; 8] = [
    [-1, 0],
    [-1, -1],
    [0, -1],
    [1, -1],
    [1, 0],
    [1, 1],
    [0, 1],
    [-1, 1],
];
const DISTANCES: [u32; 8] = [2, 3, 2, 3, 2, 3, 2, 3];
fn packed(distance: u32, direction: usize) -> u32 {
    DIJKSTRA + (distance << 4) + direction as u32
}
fn distance(pixel: u32) -> u32 {
    (pixel ^ DIJKSTRA) >> 4
}
fn index(x: i32, y: i32, w: usize, h: usize) -> Option<usize> {
    if x >= 0 && y >= 0 && x < w as i32 && y < h as i32 {
        Some(y as usize * w + x as usize)
    } else {
        None
    }
}

// Translate the synchronous, cross-neighbor-triggered extension passes. A mask
// controls assignment independently of alpha, so transparent covered pixels are
// not overwritten and can contribute fractional alpha at the growing boundary.
fn extend(image: &mut [u8], mask: &mut [u8], w: usize, h: usize, passes: usize) {
    let weights = [1.0f32, 2.0, 1.0, 2.0, 0.0, 2.0, 1.0, 2.0, 1.0];
    let mut next_image = image.to_vec();
    let mut next_mask = mask.to_vec();
    for _ in 0..passes {
        let mut changed = false;
        for y in 0..h as i32 {
            for x in 0..w as i32 {
                let i = y as usize * w + x as usize;
                if mask[i] != 0 {
                    continue;
                }
                let assigned = |x, y| index(x, y, w, h).is_some_and(|j| mask[j] != 0);
                if !assigned(x - 1, y)
                    && !assigned(x + 1, y)
                    && !assigned(x, y - 1)
                    && !assigned(x, y + 1)
                {
                    continue;
                }
                let mut sum = [0.0f32; 4];
                let mut total = 0.0;
                let mut k = 0;
                for dx in -1..=1 {
                    for dy in -1..=1 {
                        if (dx != 0 || dy != 0)
                            && let Some(j) = index(x + dx, y + dy, w, h)
                            && mask[j] != 0
                        {
                            total += weights[k];
                            for c in 0..4 {
                                sum[c] += weights[k] * f32::from(image[4 * j + c]);
                            }
                        }
                        k += 1;
                    }
                }
                if total != 0.0 {
                    for c in 0..4 {
                        next_image[4 * i + c] = (sum[c] / total).round().clamp(0.0, 255.0) as u8;
                    }
                    next_mask[i] = 2;
                    changed = true;
                }
            }
        }
        image.copy_from_slice(&next_image);
        mask.copy_from_slice(&next_mask);
        if !changed {
            break;
        }
    }
}

#[derive(Clone, Copy)]
struct Active {
    distance: u32,
    x: i32,
    y: i32,
}
// Use the source's distance-only binary-heap operations, including equal-cost
// child selection and its prepopulated initial queue. Ties affect seam ownership.
fn pop_heap(heap: &mut Vec<Active>) -> Active {
    let top = heap[0];
    let last = heap.pop().unwrap();
    if heap.is_empty() {
        return top;
    }
    let mut hole = 0;
    while 2 * hole + 1 < heap.len() {
        let mut child = 2 * hole + 1;
        if child + 1 < heap.len() && heap[child].distance > heap[child + 1].distance {
            child += 1;
        }
        heap[hole] = heap[child];
        hole = child;
    }
    while hole > 0 {
        let parent = (hole - 1) / 2;
        if heap[parent].distance <= last.distance {
            break;
        }
        heap[hole] = heap[parent];
        hole = parent;
    }
    heap[hole] = last;
    top
}
fn push_heap(heap: &mut Vec<Active>, value: Active) {
    let mut hole = heap.len();
    heap.push(value);
    while hole > 0 {
        let parent = (hole - 1) / 2;
        if heap[parent].distance <= value.distance {
            break;
        }
        heap[hole] = heap[parent];
        hole = parent;
    }
    heap[hole] = value;
}

struct Lookup {
    point: [f32; 2],
    face: usize,
    distance: f32,
}
struct MarginMap {
    w: usize,
    h: usize,
    pixels: Vec<u32>,
    adjacency: Vec<Option<usize>>,
    uv: Vec<[f32; 2]>,
}
impl MarginMap {
    fn new(mesh: &Mesh, uv: &[Tri2], owners: &[i64], w: usize, h: usize) -> Self {
        let mut adjacency = vec![None; mesh.faces.len() * 3];
        let mut first = HashMap::new();
        for (f, vertices) in mesh.faces.iter().enumerate() {
            for c in 0..3 {
                let mut edge = [vertices[c], vertices[(c + 1) % 3]];
                edge.sort_unstable();
                let i = f * 3 + c;
                if let Some(&other) = first.get(&edge) {
                    adjacency[i] = Some(other);
                    adjacency[other] = Some(i);
                } else {
                    first.insert(edge, i);
                }
            }
        }
        let uv = uv
            .iter()
            .flat_map(|t| {
                t.map(|p| {
                    [
                        p[0] as f32 * w as f32 - (0.5f32 + 0.001f32),
                        p[1] as f32 * h as f32 - (0.5f32 + 0.001f32),
                    ]
                })
            })
            .collect();
        let pixels = (0..w * h)
            .map(|i| {
                let f = owners[(h - 1 - i / w) * w + i % w];
                if f < 0 { UNSET } else { f as u32 }
            })
            .collect();
        Self {
            w,
            h,
            pixels,
            adjacency,
            uv,
        }
    }
    fn get(&self, x: i32, y: i32) -> u32 {
        index(x, y, self.w, self.h).map_or(UNSET, |i| self.pixels[i])
    }
    fn set(&mut self, x: i32, y: i32, p: u32) {
        self.pixels[y as usize * self.w + x as usize] = p;
    }
    fn grow(&mut self, margin: usize) {
        let mut heap = Vec::new();
        for y in 0..self.h as i32 {
            for x in 0..self.w as i32 {
                if self.get(x, y) != UNSET {
                    continue;
                }
                for (i, [dx, dy]) in DIRECTIONS.into_iter().enumerate() {
                    let xx = x - dx;
                    let yy = y - dy;
                    // Retain upstream's width-bound check on yy; get() still guards
                    // the true height before accessing the buffer on non-square maps.
                    if xx >= 0
                        && xx < self.w as i32
                        && yy >= 0
                        && yy < self.w as i32
                        && self.get(xx, yy) & DIJKSTRA == 0
                    {
                        self.set(x, y, packed(DISTANCES[i], i));
                        heap.push(Active {
                            distance: DISTANCES[i],
                            x,
                            y,
                        });
                        break;
                    }
                }
            }
        }
        while !heap.is_empty() {
            let p = pop_heap(&mut heap);
            if p.distance >= 2 * (margin as u32 + 1) {
                continue;
            }
            for (i, [dx, dy]) in DIRECTIONS.into_iter().enumerate() {
                let x = p.x + dx;
                let y = p.y + dy;
                if let Some(j) = index(x, y, self.w, self.h) {
                    let old = self.pixels[j];
                    let d = p.distance + DISTANCES[i];
                    if old & DIJKSTRA != 0 && distance(old) > d {
                        self.pixels[j] = packed(d, i);
                        push_heap(&mut heap, Active { distance: d, x, y });
                    }
                }
            }
        }
    }
    fn lookup(&self, p: [f32; 2], face: usize) -> Option<Lookup> {
        let mut found: Option<(usize, f32, f32)> = None;
        for c in 0..3 {
            let a = self.uv[face * 3 + c];
            let b = self.uv[face * 3 + (c + 1) % 3];
            let ab = [b[0] - a[0], b[1] - a[1]];
            let ap = [p[0] - a[0], p[1] - a[1]];
            let t = (ab[0] * ap[0] + ab[1] * ap[1]) / (ab[0] * ab[0] + ab[1] * ab[1]);
            if !(0.0..=1.0).contains(&t) {
                continue;
            }
            let reflect = [a[0] + t * ab[0] - p[0], a[1] + t * ab[1] - p[1]];
            let length = (reflect[0] * reflect[0] + reflect[1] * reflect[1]).sqrt();
            let cross = ab[0] * reflect[1] - ab[1] * reflect[0];
            if cross > 0.0 && found.is_none_or(|(_, d, _)| length < d) {
                found = Some((face * 3 + c, length, t));
            }
        }
        let (edge, d, t) = found?;
        let other = self.adjacency[edge]?;
        let dst = other / 3;
        let a = self.uv[other];
        let b = self.uv[dst * 3 + (other % 3 + 1) % 3];
        let ab = [a[0] - b[0], a[1] - b[1]];
        let reflect = [b[0] + t * ab[0], b[1] + t * ab[1]];
        let perpendicular = [ab[1], -ab[0]];
        let factor =
            d / (perpendicular[0] * perpendicular[0] + perpendicular[1] * perpendicular[1]).sqrt();
        let point = [
            reflect[0] + factor * perpendicular[0],
            reflect[1] + factor * perpendicular[1],
        ];
        point.iter().all(|v| v.is_finite()).then_some(Lookup {
            point,
            face: dst,
            distance: d,
        })
    }
    fn neighborhood(&self, p: [f32; 2], face: usize) -> Option<Lookup> {
        if let Some(hit) = self.lookup(p, face) {
            return Some(hit);
        }
        let mut best: Option<Lookup> = None;
        for c in 0..3 {
            if let Some(other) = self.adjacency[face * 3 + c]
                && let Some(hit) = self.lookup(p, other / 3)
                && best.as_ref().is_none_or(|b| hit.distance < b.distance)
            {
                best = Some(hit);
            }
        }
        best
    }
    fn fill(&self, image: &mut [u8], mask: &mut [u8]) {
        for y in 0..self.h {
            for x in 0..self.w {
                let i = y * self.w + x;
                let dp = self.pixels[i];
                if dp == UNSET || dp & DIJKSTRA == 0 {
                    mask[i] = 1;
                    continue;
                }
                let mut d = distance(dp);
                let mut direction = (dp & 15) as usize;
                let mut xx = x as i32;
                let mut yy = y as i32;
                while d > 0 {
                    xx -= DIRECTIONS[direction][0];
                    yy -= DIRECTIONS[direction][1];
                    d = d.saturating_sub(DISTANCES[direction]);
                    direction = (self.get(xx, yy) & 15) as usize;
                }
                let face = self.get(xx, yy);
                if face & DIJKSTRA != 0 {
                    continue;
                }
                let Some(mut hit) = self.neighborhood([x as f32, y as f32], face as usize) else {
                    continue;
                };
                for _ in 0..3 {
                    let nx = hit.point[0].round() as i32;
                    let ny = hit.point[1].round() as i32;
                    if self.get(nx, ny) == hit.face as u32 {
                        let color = bilinear(image, self.w, self.h, hit.point);
                        image[4 * i..4 * i + 4].copy_from_slice(&color);
                        mask[i] = 1;
                        break;
                    }
                    match self.lookup([nx as f32, ny as f32], hit.face) {
                        Some(next) => hit = next,
                        None => break,
                    }
                }
            }
        }
    }
}
fn bilinear(image: &[u8], w: usize, h: usize, p: [f32; 2]) -> [u8; 4] {
    let x = p[0].floor() as i32;
    let y = p[1].floor() as i32;
    let a = p[0] - p[0].floor();
    let b = p[1] - p[1].floor();
    let sample = |x, y, c| index(x, y, w, h).map_or(0.0, |i| f32::from(image[4 * i + c]));
    std::array::from_fn(|c| {
        ((1.0 - a) * (1.0 - b) * sample(x, y, c)
            + a * (1.0 - b) * sample(x + 1, y, c)
            + (1.0 - a) * b * sample(x, y + 1, c)
            + a * b * sample(x + 1, y + 1, c)
            + 0.5) as u8
    })
}

pub(crate) fn apply(
    image: &mut [u8],
    owners: &[i64],
    mesh: &Mesh,
    uv: &[Tri2],
    w: usize,
    h: usize,
    margin: usize,
) {
    // Work in Blender's bottom-up pixel order. This preserves traversal order
    // in the in-place adjacency lookup; Pillow-facing output stays top-down.
    let mut buffer = vec![0; image.len()];
    for y in 0..h {
        buffer[y * w * 4..(y + 1) * w * 4]
            .copy_from_slice(&image[(h - 1 - y) * w * 4..(h - y) * w * 4]);
    }
    let mut map = MarginMap::new(mesh, uv, owners, w, h);
    let mut mask: Vec<u8> = map.pixels.iter().map(|&p| u8::from(p != UNSET)).collect();
    let mut temporary_mask = mask.clone();
    extend(&mut buffer, &mut temporary_mask, w, h, 2);
    map.grow(margin);
    map.fill(&mut buffer, &mut mask);
    extend(&mut buffer, &mut mask, w, h, margin);
    for y in 0..h {
        image[(h - 1 - y) * w * 4..(h - y) * w * 4]
            .copy_from_slice(&buffer[y * w * 4..(y + 1) * w * 4]);
    }
}
