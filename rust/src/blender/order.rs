// SPDX-FileCopyrightText: 2026 InterDeCA contributors
// SPDX-FileCopyrightText: 2013 Blender Authors
// SPDX-License-Identifier: GPL-2.0-or-later
//! Blender uses host qsort for face areas and its bundled BLI_qsort_r for boxes.

use std::ffi::{c_int, c_void};

#[repr(C)]
#[derive(Clone, Copy)]
struct Ranked {
    index: usize,
    key: f32,
}

unsafe extern "C" {
    fn qsort(
        base: *mut c_void,
        count: usize,
        size: usize,
        compare: unsafe extern "C" fn(*const c_void, *const c_void) -> c_int,
    );
}

unsafe extern "C" fn compare(a: *const c_void, b: *const c_void) -> c_int {
    // qsort passes aligned, initialized Ranked entries from our live slice.
    let (a, b) = unsafe { (&*a.cast::<Ranked>(), &*b.cast::<Ranked>()) };
    if a.key < b.key {
        -1
    } else if a.key > b.key {
        1
    } else {
        0
    }
}

pub(crate) fn sort(indices: &mut [usize], key: impl Fn(usize) -> f32) {
    let mut ranked: Vec<_> = indices
        .iter()
        .map(|&index| Ranked {
            index,
            key: key(index),
        })
        .collect();
    // No Rust references escape through the callback and comparison cannot panic.
    // Ties deliberately remain unspecified, as in the source qsort call.
    unsafe {
        qsort(
            ranked.as_mut_ptr().cast(),
            ranked.len(),
            size_of::<Ranked>(),
            compare,
        );
    }
    for (index, rank) in indices.iter_mut().zip(ranked) {
        *index = rank.index;
    }
}

// Translate sort.cc's element swaps directly to slice swaps. The comparator
// reads immutable geometry, so there is no callback context or raw pointer here.
pub(crate) fn reentrant_sort(indices: &mut [usize], key: impl Fn(usize) -> f32) {
    fn cmp(a: usize, b: usize, key: &impl Fn(usize) -> f32) -> i32 {
        let (a, b) = (key(a), key(b));
        if a < b {
            -1
        } else if a > b {
            1
        } else {
            0
        }
    }
    fn median(a: &[usize], i: usize, j: usize, k: usize, key: &impl Fn(usize) -> f32) -> usize {
        if cmp(a[i], a[j], key) < 0 {
            if cmp(a[j], a[k], key) < 0 {
                j
            } else if cmp(a[i], a[k], key) < 0 {
                k
            } else {
                i
            }
        } else if cmp(a[j], a[k], key) > 0 {
            j
        } else if cmp(a[i], a[k], key) < 0 {
            i
        } else {
            k
        }
    }
    fn insertion(a: &mut [usize], key: &impl Fn(usize) -> f32) {
        for i in 1..a.len() {
            let mut j = i;
            while j > 0 && cmp(a[j - 1], a[j], key) > 0 {
                a.swap(j, j - 1);
                j -= 1;
            }
        }
    }
    fn run(mut a: &mut [usize], key: &impl Fn(usize) -> f32) {
        loop {
            let n = a.len();
            if n < 7 {
                insertion(a, key);
                return;
            }
            let mut middle = n / 2;
            if n > 7 {
                let (mut first, mut last) = (0, n - 1);
                if n > 40 {
                    let d = n / 8;
                    first = median(a, first, first + d, first + 2 * d, key);
                    middle = median(a, middle - d, middle, middle + d, key);
                    last = median(a, last - 2 * d, last - d, last, key);
                }
                middle = median(a, first, middle, last, key);
            }
            a.swap(0, middle);
            let (mut pa, mut pb) = (1isize, 1isize);
            let (mut pc, mut pd) = (n as isize - 1, n as isize - 1);
            let mut swapped = false;
            loop {
                while pb <= pc {
                    let comparison = cmp(a[pb as usize], a[0], key);
                    if comparison > 0 {
                        break;
                    }
                    if comparison == 0 {
                        swapped = true;
                        a.swap(pa as usize, pb as usize);
                        pa += 1;
                    }
                    pb += 1;
                }
                while pb <= pc {
                    let comparison = cmp(a[pc as usize], a[0], key);
                    if comparison < 0 {
                        break;
                    }
                    if comparison == 0 {
                        swapped = true;
                        a.swap(pc as usize, pd as usize);
                        pd -= 1;
                    }
                    pc -= 1;
                }
                if pb > pc {
                    break;
                }
                a.swap(pb as usize, pc as usize);
                swapped = true;
                pb += 1;
                pc -= 1;
            }
            if !swapped {
                insertion(a, key);
                return;
            }
            let r = pa.min(pb - pa) as usize;
            for i in 0..r {
                a.swap(i, pb as usize - r + i);
            }
            let r = (pd - pc).min(n as isize - pd - 1) as usize;
            for i in 0..r {
                a.swap(pb as usize + i, n - r + i);
            }
            let left = (pb - pa) as usize;
            if left > 1 {
                run(&mut a[..left], key);
            }
            let right = (pd - pc) as usize;
            if right <= 1 {
                return;
            }
            a = &mut a[n - right..];
        }
    }
    run(indices, &key);
}
