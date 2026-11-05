// SPDX-FileCopyrightText: 2011-2022 Blender Foundation
// SPDX-FileCopyrightText: 2011-2026 Blender Authors
// SPDX-FileCopyrightText: 2026 InterDeCA contributors
// SPDX-License-Identifier: Apache-2.0
//
// Blender 5.2.1 Cycles image.h ImageInterpolator<uchar4>::interp_linear and
// triangle.h triangle_interpolate, single image with REPEAT extension.

use super::math::V3;

fn decode(encoded: f32) -> f32 {
    // Cycles util/color.h's scalar color_srgb_to_linear. Decode follows byte
    // filtering, preserving the image node's COMPRESS_AS_SRGB behavior.
    if encoded < 0.04045 {
        if encoded < 0.0 {
            0.0
        } else {
            encoded * (1.0 / 12.92)
        }
    } else {
        ((encoded + 0.055) * (1.0 / 1.055)).powf(2.4)
    }
}

pub(crate) fn associate_alpha(bytes: &mut [u8]) {
    // ImageMetaData::conform_pixels associates straight RGBA8 source alpha in
    // its storage format before filtering. The diffuse color output does not
    // connect the image node's Alpha output, so it is not unassociated later.
    for pixel in bytes.chunks_exact_mut(4) {
        for c in 0..3 {
            pixel[c] = ((pixel[c] as u32 * pixel[3] as u32) / 255) as u8;
        }
    }
}

pub(crate) fn uv(corners: [[f64; 2]; 3], barycentric: [f32; 2]) -> [f32; 2] {
    let [a, b] = barycentric;
    // Blender and Cycles assign barycentric u/v to different triangle corners.
    let (u, v) = (b, 1.0 - a - b);
    std::array::from_fn(|axis| {
        ((1.0 - u - v) * (corners[0][axis] as f32) + u * (corners[1][axis] as f32))
            + v * (corners[2][axis] as f32)
    })
}
fn fraction(x: f32) -> (i64, f32) {
    // Source frac truncates then subtracts one for negative values, including
    // negative integers. Wrapping makes its endpoint convention equivalent.
    let i = x as i64 - i64::from(x < 0.0);
    (i, x - i as f32)
}
pub(crate) fn sample(bytes: &[u8], width: usize, height: usize, uv: [f32; 2]) -> [u8; 4] {
    let (x, tx) = fraction(uv[0] * width as f32 - 0.5);
    let (y, ty) = fraction(uv[1] * height as f32 - 0.5);
    let read = |x: i64, y: i64| -> V3 {
        let x = x.rem_euclid(width as i64) as usize;
        let y = y.rem_euclid(height as i64) as usize;
        let i = ((height - 1 - y) * width + x) * 4;
        std::array::from_fn(|c| bytes[i + c] as f32 * (1.0f32 / 255.0))
    };
    let taps = [
        read(x, y),
        read(x + 1, y),
        read(x, y + 1),
        read(x + 1, y + 1),
    ];
    let weights = [
        (1.0 - ty) * (1.0 - tx),
        (1.0 - ty) * tx,
        ty * (1.0 - tx),
        ty * tx,
    ];
    let mut result = [0, 0, 0, 255];
    // Diffuse color decodes the filtered sample into scene-linear RGB. Blender
    // then applies its configured OCIO output curve before half-up byte storage.
    // Keep both stages: they do not numerically cancel at quantization boundaries.
    for c in 0..3 {
        let value = ((weights[0] * taps[0][c] + weights[1] * taps[1][c]) + weights[2] * taps[2][c])
            + weights[3] * taps[3][c];
        result[c] = super::color::to_byte(super::ocio::encode(decode(value)));
    }
    result
}
