// SPDX-FileCopyrightText: Contributors to the OpenColorIO Project
// SPDX-FileCopyrightText: 2026 InterDeCA contributors
// SPDX-License-Identifier: BSD-3-Clause
//
// OpenColorIO 2.5.0 GammaOpUtils.cpp::ComputeParamsRev,
// GammaOpCPU.cpp::GammaMoncurveOpCPURevSSE, and SSE.h::ssePower.
// Blender's pinned config uses inverse ExponentWithLinear(gamma=2.4,offset=.055).
// Scalar lanes retain the source SIMD arithmetic without requiring a C++ library.

use std::sync::OnceLock;

struct Parameters {
    gamma: f32,
    scale: f32,
    offset: f32,
    breakpoint: f32,
    slope: f32,
}

fn power(value: f32, exponent: f32) -> f32 {
    // Split the IEEE float into exponent and mantissa, then evaluate the source
    // minimax log2 polynomial in the same multiply/add order (no fused steps).
    let mantissa = f32::from_bits((value.to_bits() & !0x7f800000) | 0x3f800000);
    let coefficients: [f32; 6] = [
        4.487361286440374006195e-2,
        -4.165637071209677112635e-1,
        1.631148826119436277100,
        -3.550793018041176193407,
        5.091710879305474367557,
        -2.800364054395965731506,
    ];
    let mut logarithm = coefficients[0];
    for coefficient in &coefficients[1..] {
        logarithm = logarithm * mantissa + coefficient;
    }
    logarithm += (((value.to_bits() & 0x7f800000) >> 23) as i32 - 127) as f32;
    let x = exponent * logarithm;
    // The source subtracts one after truncating every negative value, including
    // negative integers. Preserve that convention rather than substituting floor.
    let integer = (x as i32).wrapping_add(if 0.0 <= x { 0 } else { -1 });
    let scale = f32::from_bits((integer.wrapping_add(127) as u32) << 23);
    let fraction = x - integer as f32;
    let coefficients: [f32; 5] = [
        1.353416792833547468620e-2,
        5.201146058412685018921e-2,
        2.414427569091865207710e-1,
        6.930038344665415134202e-1,
        1.000002593370603213644,
    ];
    let mut exponential = coefficients[0];
    for coefficient in &coefficients[1..] {
        exponential = exponential * fraction + coefficient;
    }
    if value <= 0.0 || x < -126.0 {
        0.0
    } else if x >= 128.0 {
        f32::INFINITY
    } else {
        scale * exponential
    }
}

pub(crate) fn encode(linear: f32) -> f32 {
    static PARAMETERS: OnceLock<Parameters> = OnceLock::new();
    let p = PARAMETERS.get_or_init(|| {
        // OCIO computes its smooth-join parameters in double precision before
        // storing float lanes. This toe differs slightly from the IEC threshold.
        let gamma: f64 = 2.4;
        let offset: f64 = 0.055;
        Parameters {
            gamma: (1.0 / gamma) as f32,
            scale: (1.0 + offset) as f32,
            offset: offset as f32,
            breakpoint: ((offset * gamma) / ((gamma - 1.0) * (1.0 + offset))).powf(gamma) as f32,
            slope: (((gamma - 1.0) / offset).powf(gamma - 1.0)
                * ((1.0 + offset) / gamma).powf(gamma)) as f32,
        }
    });
    if linear > p.breakpoint {
        power(linear, p.gamma) * p.scale - p.offset
    } else {
        linear * p.slope
    }
}
