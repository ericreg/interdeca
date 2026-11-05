// SPDX-FileCopyrightText: 2001-2002 NaN Holding BV. All rights reserved.
// SPDX-FileCopyrightText: 2026 InterDeCA contributors
// SPDX-License-Identifier: GPL-2.0-or-later
//
// Blender 5.2.1 math_base_inline.cc::unit_float_to_uchar_clamp, called by
// IMB_buffer_byte_from_float after the bake output's color-space conversion.

pub(crate) fn to_byte(value: f32) -> u8 {
    // Bake conversion rounds exact positive half bytes upward. This differs
    // from the NumPy-compatible ties-to-even contract of ordinary TextureIO.
    if value <= 0.0 {
        0
    } else if value > 1.0 - 0.5 / 255.0 {
        255
    } else {
        (255.0 * value + 0.5) as u8
    }
}
