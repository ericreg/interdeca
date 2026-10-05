# InterDeCA distribution licenses

The InterDeCA package and linked native extension are distributed under
**GPL-3.0-or-later**. See `LICENSE`. Existing third-party components retain their
own copyright notices and licenses; this notice does not relicense those works.

## Blender-derived functionality

Rust translations under `rust/src/blender/` are derived from Blender **5.2.1 LTS**,
commit `9e2066aef7ef7e20c142ad7bd3303138a4304c93`:
https://github.com/blender/blender/tree/9e2066aef7ef7e20c142ad7bd3303138a4304c93

The original source files identify their copyright holders in SPDX headers,
including Blender Authors, NaN Holding BV, softSurfer, and Blender Foundation.
The translated Blender core files use `GPL-2.0-or-later`; translated Cycles image
and triangle helpers use `Apache-2.0`. Those notices are retained in the
adaptations; the combined InterDeCA distribution uses GPL version 3 or later.
The GPL and Apache license texts are supplied in `LICENSES/`. The softSurfer
notice is also preserved in the hull translation.

`rust/BLENDER_SOURCE.json` records original file paths, SHA-256 checksums,
functions translated, and the scope of each port. The Rust adaptations are
modified works: Blender's scene, UI, allocator, and generic mesh APIs are replaced
by InterDeCA's owned triangle buffers and Python boundary. The implementation
retains numerical decisions and traversal order where they affect results.
The source ledger distinguishes implemented paths from pending work.

Corresponding InterDeCA source consists of the Python and Rust sources, Cargo
manifests/lockfile, Python build metadata/lockfile, and build instructions in this
repository. Distribute those sources with the binaries or provide them through a
GPL-compliant source distribution. Do not describe a binary wheel alone as a
complete source distribution. Modified source files retain their license headers.

## Existing dependencies

The small sRGB output conversion in `rust/src/blender/ocio.rs` is translated
from OpenColorIO 2.5.0, the version pinned by Blender 5.2.1, commit
`592a122b37467c3376f2387fbbdbe7b571fad39f`. Copyright Contributors to the
OpenColorIO Project; BSD-3-Clause. Its license is preserved in
`LICENSES/BSD-3-Clause-OpenColorIO.txt` and its source checksums and translated
functions are recorded in `rust/OCIO_SOURCE.json`. This translation does not link
to an OpenColorIO or Blender library.

The vendored MorphoWeave tree retains its `LICENSE` and
`THIRD_PARTY_NOTICES.md` under `src/interdeca/_vendor/morphoweave/`.
Other Python and Rust dependencies retain the licenses supplied by their authors
in their distributions. Their declarations remain in `pyproject.toml`, `uv.lock`,
`rust/Cargo.toml`, and `rust/Cargo.lock`.
