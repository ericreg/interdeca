// SPDX-FileCopyrightText: 2001-2002 NaN Holding BV. All rights reserved.
// SPDX-FileCopyrightText: 2026 InterDeCA contributors
// SPDX-License-Identifier: GPL-2.0-or-later
//
// Adapted from Blender 5.2.1 render/intern/zbuf.cc: zbuf_add_to_span,
// zspan_scanconvert; bake.cc: RE_bake_pixels_populate. Revision in BLENDER_SOURCE.json.

pub(crate) type Triangle = [[f32; 2]; 3];

struct Span {
    values: Vec<f32>,
    min_y: i32,
    max_y: i32,
    min_point: Option<usize>,
    max_point: Option<usize>,
}
impl Span {
    fn new(height: usize) -> Self {
        Self {
            values: vec![0.0; height],
            min_y: height as i32 + 1,
            max_y: -1,
            min_point: None,
            max_point: None,
        }
    }
    fn reset(&mut self) {
        self.min_y = self.values.len() as i32 + 1;
        self.max_y = -1;
        self.min_point = None;
        self.max_point = None;
    }
}

pub(crate) struct Rasterizer {
    width: usize,
    height: usize,
    spans: [Span; 2],
}
impl Rasterizer {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            spans: [Span::new(height), Span::new(height)],
        }
    }
    fn add_edge(&mut self, t: Triangle, a: usize, b: usize) {
        let (min, max) = if t[a][1] < t[b][1] { (a, b) } else { (b, a) };
        let y0 = t[min][1].ceil() as i32;
        let y2 = t[max][1].floor() as i32;
        if y2 < 0 || y0 >= self.height as i32 {
            return;
        }
        let y0 = y0.max(0);
        let y2 = y2.min(self.height as i32 - 1);
        if y0 > y2 {
            return;
        }
        let length = t[max][1] - t[min][1];
        let (dx, mut x) = if length > f32::EPSILON {
            let dx = (t[min][0] - t[max][0]) / length;
            (dx, dx * (t[min][1] - y2 as f32) + t[min][0])
        } else {
            (0.0, t[min][0].min(t[max][0]))
        };
        // Endpoint identity decides which edge continues the first span. Using
        // coordinate equality here would change rasterization of collapsed UVs.
        let first = self.spans[0].max_point.is_none()
            || self.spans[0].min_point == Some(max)
            || self.spans[0].max_point == Some(min);
        let span = &mut self.spans[usize::from(!first)];
        if span.min_point.is_none_or(|i| t[i][1] > t[min][1]) {
            span.min_point = Some(min);
        }
        if span.max_point.is_none_or(|i| t[i][1] < t[max][1]) {
            span.max_point = Some(max);
        }
        span.min_y = span.min_y.min(y0);
        span.max_y = span.max_y.max(y2);
        for y in (y0..=y2).rev() {
            span.values[y as usize] = x;
            x += dx;
        }
    }
    pub fn scan(&mut self, t: Triangle, mut pixel: impl FnMut(usize, usize, f32, f32)) {
        self.spans.iter_mut().for_each(Span::reset);
        self.add_edge(t, 0, 1);
        self.add_edge(t, 1, 2);
        self.add_edge(t, 2, 0);
        if self.spans[1].min_point.is_none() || self.spans[1].max_point.is_none() {
            return;
        }
        let min_y = self.spans[0].min_y.max(self.spans[1].min_y);
        let max_y = self.spans[0].max_y.min(self.spans[1].max_y);
        if max_y < min_y {
            return;
        }
        let x1 = t[0][0] - t[1][0];
        let x2 = t[1][0] - t[2][0];
        let y1 = t[0][1] - t[1][1];
        let y2 = t[1][1] - t[2][1];
        let mut x0 = y1 * 0.0 - y2;
        let mut y0 = x2 - x1 * 0.0;
        let z0 = x1 * y2 - y1 * x2;
        if z0 == 0.0 {
            return;
        }
        let xx1 = (x0 * t[0][0] + y0 * t[0][1]) / z0 + 1.0;
        let ux = (-f64::from(x0) / f64::from(z0)) as f32;
        let uy = (-f64::from(y0) / f64::from(z0)) as f32;
        let u0 = (f64::from(max_y) * f64::from(uy) + f64::from(xx1)) as f32;
        x0 = y1 + y2;
        y0 = -x2 - x1;
        let xx1 = (x0 * t[0][0] + y0 * t[0][1]) / z0;
        let vx = (-f64::from(x0) / f64::from(z0)) as f32;
        let vy = (-f64::from(y0) / f64::from(z0)) as f32;
        let v0 = (f64::from(max_y) * f64::from(vy) + f64::from(xx1)) as f32;
        // The left endpoint is excluded, the right endpoint is included. The
        // explicit f64 promotions and f32 stores follow the original scan loop.
        for (i, y) in (min_y..=max_y).rev().enumerate() {
            let a = self.spans[0].values[y as usize];
            let b = self.spans[1].values[y as usize];
            let left = (a.min(b).floor() as i32 + 1).max(0);
            let right = (a.max(b).floor() as i32).min(self.width as i32 - 1);
            let u =
                (f64::from(left) * f64::from(ux) + f64::from(u0) - f64::from(i as f32 * uy)) as f32;
            let v =
                (f64::from(left) * f64::from(vx) + f64::from(v0) - f64::from(i as f32 * vy)) as f32;
            for (j, x) in (left..=right).enumerate() {
                pixel(x as usize, y as usize, u + j as f32 * ux, v + j as f32 * vx);
            }
        }
    }
}

pub(crate) fn pixel_triangle(uv: [[f64; 2]; 3], width: usize, height: usize) -> Triangle {
    // Blender's pixel-alignment offsets are intentionally different on X and Y.
    uv.map(|p| {
        [
            (p[0] as f32) * width as f32 - (0.5f32 + 0.001f32),
            (p[1] as f32) * height as f32 - (0.5f32 + 0.002f32),
        ]
    })
}
