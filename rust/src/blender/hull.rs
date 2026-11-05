// SPDX-FileCopyrightText: 2023 Blender Authors
// SPDX-FileCopyrightText: 2001 softSurfer (http://www.softsurfer.com)
// SPDX-FileCopyrightText: 2026 InterDeCA contributors
// SPDX-License-Identifier: GPL-2.0-or-later
//
// Blender 5.2.1 convexhull_2d.cc: hull construction and rotating calipers.
// Copyright 2001, softSurfer (http://www.softsurfer.com)
// This code may be freely used and modified for any purpose providing that this
// copyright notice is included with it. SoftSurfer makes no warranty for this
// code, and cannot be held liable for any real or imagined damage resulting from
// its use. Users of this code must verify correctness for their application.

pub(crate) type P2 = [f32; 2];
fn left(a: P2, b: P2, c: P2) -> f32 {
    (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])
}
fn push(points: &[P2], stack: &mut Vec<usize>, i: usize) {
    // Preserve the source's second convexity check and argument order: nearly
    // collinear edges can otherwise reverse the caliper's angle progression.
    while stack.len() > 1
        && left(
            points[stack[stack.len() - 2]],
            points[i],
            points[*stack.last().unwrap()],
        ) >= 0.0
    {
        stack.pop();
    }
    stack.push(i);
}
fn hull(points: &[P2]) -> Vec<P2> {
    let mut sorted = points.to_vec();
    sorted.sort_by(|a, b| a[1].total_cmp(&b[1]).then(a[0].total_cmp(&b[0])));
    if sorted.len() < 2 {
        return sorted;
    }
    let maxmax = sorted.len() - 1;
    let minmax = sorted
        .iter()
        .position(|p| p[0] != sorted[0][0])
        .unwrap_or(sorted.len())
        - 1;
    let mut stack = Vec::new();
    push(&sorted, &mut stack, 0);
    if minmax == maxmax {
        if sorted[minmax][1] != sorted[0][1] {
            push(&sorted, &mut stack, minmax);
        }
    } else {
        let maxmin = sorted
            .iter()
            .rposition(|p| p[0] != sorted[maxmax][0])
            .map_or(0, |i| i + 1);
        for i in minmax + 1..=maxmin {
            if i < maxmin && left(sorted[0], sorted[maxmin], sorted[i]) >= 0.0 {
                continue;
            }
            while stack.len() > 1
                && left(
                    sorted[stack[stack.len() - 2]],
                    sorted[*stack.last().unwrap()],
                    sorted[i],
                ) <= 0.0
            {
                stack.pop();
            }
            push(&sorted, &mut stack, i);
        }
        if maxmax != maxmin {
            push(&sorted, &mut stack, maxmax);
        }
        let bottom = stack.len() - 1;
        let mut closed = false;
        for i in (minmax..maxmin).rev() {
            if i > minmax && left(sorted[maxmax], sorted[minmax], sorted[i]) >= 0.0 {
                continue;
            }
            while stack.len() > bottom + 1
                && left(
                    sorted[stack[stack.len() - 2]],
                    sorted[*stack.last().unwrap()],
                    sorted[i],
                ) <= 0.0
            {
                stack.pop();
            }
            if sorted[i] == sorted[stack[0]] {
                closed = true;
                break;
            }
            push(&sorted, &mut stack, i);
        }
        if !closed && minmax != 0 && stack[0] != 0 {
            push(&sorted, &mut stack, 0);
        }
    }
    let mut bottom = 0;
    while stack.len() - bottom >= 3 {
        let top = stack.len() - 1;
        if left(
            sorted[stack[top - 1]],
            sorted[stack[bottom]],
            sorted[stack[top]],
        ) >= 0.0
        {
            stack.pop();
            continue;
        }
        if left(
            sorted[stack[top]],
            sorted[stack[bottom + 1]],
            sorted[stack[bottom]],
        ) >= 0.0
        {
            bottom += 1;
            continue;
        }
        break;
    }
    // Unlike upstream's index remapping storage, these entries already contain
    // the sorted coordinates, so discarding the unused prefix is explicit.
    stack[bottom..].iter().map(|&i| sorted[i]).collect()
}
fn normalize(p: P2) -> P2 {
    let d = p[0] * p[0] + p[1] * p[1];
    if d > 1e-35 {
        let d = d.sqrt();
        [p[0] / d, p[1] / d]
    } else {
        [0.0; 2]
    }
}
fn canonical(p: P2) -> P2 {
    if p[0] < 0.0 {
        if p[1] < 0.0 {
            [-p[0], -p[1]]
        } else if p[0] == -1.0 && p[1] == 0.0 {
            [-p[0], p[1]]
        } else {
            [p[1], -p[0]]
        }
    } else if p[1] < 0.0 {
        [-p[1], p[0]]
    } else if p[0] == 0.0 && p[1] == 1.0 {
        [p[1], p[0]]
    } else {
        p
    }
}
#[derive(Clone, Copy, Default)]
struct Step {
    index: usize,
    limit: usize,
    edge: usize,
    direction: P2,
    canonical: P2,
}
fn advance(points: &[P2], step: &mut Step) -> bool {
    while step.index != step.limit {
        let i = step.index;
        step.index = (i + 1) % points.len();
        let d = normalize([
            points[step.index][0] - points[i][0],
            points[step.index][1] - points[i][1],
        ]);
        if d != [0.0; 2] {
            step.edge = i;
            step.direction = d;
            step.canonical = canonical(d);
            return true;
        }
    }
    false
}
fn insert(order: &mut Vec<usize>, steps: &[Step; 4], index: usize) {
    let b = steps[index];
    let pos = order
        .iter()
        .position(|&j| {
            let a = steps[j];
            a.canonical[0]
                .total_cmp(&b.canonical[0])
                .then(b.canonical[1].total_cmp(&a.canonical[1]))
                .then(b.edge.cmp(&a.edge))
                != std::cmp::Ordering::Greater
        })
        .unwrap_or(order.len());
    order.insert(pos, index);
}
fn extent(points: &[P2], angle: P2, index: &mut usize, axis: usize, minimum: bool) -> f32 {
    let project = |p: P2| {
        if axis == 0 {
            angle[0] * p[0] + angle[1] * p[1]
        } else {
            angle[1] * p[0] - angle[0] * p[1]
        }
    };
    let start = *index;
    let mut best = project(points[start]);
    for n in 1..points.len() {
        let i = (start + n) % points.len();
        let value = project(points[i]);
        if if minimum { value > best } else { value < best } {
            break;
        }
        best = value;
        *index = i;
    }
    best
}
pub(crate) fn fit_angle(points: &[P2]) -> f32 {
    let points = hull(points);
    if points.len() <= 1 {
        return 0.0;
    }
    let mut steps = [Step::default(); 4];
    // Each side of the axis-aligned bounds starts a quarter-turn iterator.
    for axis in 0..2 {
        let mut range = [points[0][axis]; 2];
        for (i, p) in points.iter().enumerate().skip(1) {
            if range[0] < p[axis] {
                range[0] = p[axis];
                steps[axis * 2].index = i;
            }
            if range[1] > p[axis] {
                range[1] = p[axis];
                steps[axis * 2 + 1].index = i;
            }
        }
        for side in 0..2 {
            let step = &mut steps[axis * 2 + side];
            let original = step.index;
            loop {
                let prev = (step.index + points.len() - 1) % points.len();
                if prev == original {
                    break;
                }
                let d = normalize([
                    points[step.index][0] - points[prev][0],
                    points[step.index][1] - points[prev][1],
                ]);
                if d != [0.0; 2] && (d[axis].abs() > 0.5 || canonical(d)[0] != 1.0) {
                    break;
                }
                step.index = prev;
            }
        }
    }
    for (i, next) in [2, 3, 1, 0].into_iter().enumerate() {
        steps[i].limit = steps[next].index;
    }
    let mut order = Vec::new();
    for i in 0..4 {
        steps[i].edge = steps[i].index;
        if advance(&points, &mut steps[i]) {
            insert(&mut order, &steps, i);
        }
    }
    let mut bounds = [
        [steps[1].edge, steps[0].edge],
        [steps[2].edge, steps[3].edge],
    ];
    let mut best_area = f32::MAX;
    let mut best_index = usize::MAX;
    let mut best = [0.0, 1.0];
    while !order.is_empty() {
        let i = order.remove(0);
        let step = steps[i];
        let spans: [f32; 2] = std::array::from_fn(|a| {
            let lo = extent(&points, step.canonical, &mut bounds[a][0], a, true);
            let hi = extent(&points, step.canonical, &mut bounds[a][1], a, false);
            hi - lo
        });
        let area = spans[0] * spans[1];
        if area < best_area || (area == best_area && step.edge < best_index) {
            best_area = area;
            best_index = step.edge;
            best = step.direction;
        }
        if advance(&points, &mut steps[i]) {
            insert(&mut order, &steps, i);
        }
    }
    if best_area != f32::MAX {
        best[0].atan2(best[1])
    } else {
        0.0
    }
}
