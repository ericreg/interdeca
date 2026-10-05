"""Record and verify the exact Blender source used for the GPL translation."""

import argparse
import hashlib
import subprocess
from pathlib import Path
from typing import NamedTuple

from serde import BlenderSourceFile, BlenderSourceManifest


class Source(NamedTuple):
    path: str
    symbols: tuple[str, ...]
    status: str
    adaptation: str


REVISION = "9e2066aef7ef7e20c142ad7bd3303138a4304c93"
OCIO_REVISION = "592a122b37467c3376f2387fbbdbe7b571fad39f"
SOURCES = (
    Source("source/blender/blenlib/BLI_kdtree.hh", ("kdtree_balance", "kdtree_range_search_cb", "kdtree_calc_duplicates_cb"),
           "ported", "rust/src/blender/cleanup.rs; float3, no preselected keep vertices; owned Rust nodes and stack"),
    Source("source/blender/bmesh/operators/bmo_removedoubles.cc", ("bmesh_find_doubles_by_distance_impl", "bmo_weld_verts_exec", "remdoubles_createface"),
           "ported", "Triangle-only representative selection and in-place weld; exact chain and captured atlas checks pass; both actual pre-cleanup workflow atlases match exactly"),
    Source("source/blender/editors/uvedit/uvedit_unwrap_ops.cc", ("smart_uv_project_calculate_project_normals", "smart_project_exec", "uvedit_pack_islands_multi", "uv_map_clip_correct"),
           "ported", "rust/src/blender/uv.rs; single selected triangle mesh, fresh UVs, area_weight=0, aspect=1; both full captured atlases and eight analytical fixtures match all corners exactly"),
    Source("source/blender/geometry/intern/uv_pack.cc", ("pack_islands", "calculate_pre_rotation_", "calculate_pivot_", "place_", "pack_islands_alpaca_turbo", "pack_island_box_pack_2d", "calc_margin_from_aabb_length_sum", "mul_v2_m2_add_v2v2"),
           "ported", "rust/src/blender/uv.rs; Smart UV defaults: AABB shape, AXIS_ALIGNED_Y, SCALED margin; no xatlas/occupancy path is executed for these settings"),
    Source("source/blender/editors/uvedit/uvedit_islands.cc", ("bm_mesh_calc_uv_islands", "bm_loop_uv_shared_edge_check"),
           "ported", "rust/src/blender/uv.rs; fresh mesh has no seams, pins, hidden or unselected faces"),
    Source("source/blender/bmesh/intern/bmesh_query.cc", ("BM_mesh_calc_face_groups",),
           "ported", "rust/src/blender/uv.rs; explicit stack and radial-cycle traversal"),
    Source("source/blender/bmesh/intern/bmesh_structure.cc", ("bmesh_radial_loop_append",),
           "ported", "rust/src/blender/uv.rs; array indices preserve radial insertion order"),
    Source("source/blender/bmesh/intern/bmesh_polygon.cc", ("BM_face_calc_area", "BM_face_calc_normal"),
           "ported", "rust/src/blender/uv.rs; triangle-only normal and float32 Newell area"),
    Source("source/blender/blenlib/intern/convexhull_2d.cc", ("BLI_convexhull_2d", "BLI_convexhull_aabb_fit_points_2d", "convexhull_aabb_fit_hull_2d"),
           "ported", "rust/src/blender/hull.rs; guarded hull construction and four-axis rotating calipers"),
    Source("source/blender/blenlib/intern/boxpack_2d.cc", ("BLI_box_pack_2d",),
           "ported", "rust/src/blender/boxpack.rs; sort_boxes=false; owned array indices replace box/vertex pointers"),
    Source("source/blender/blenlib/intern/sort.cc", ("BLI_qsort_r",),
           "ported", "rust/src/blender/order.rs; slice swaps preserve the bundled sorting algorithm; face-area sorting still calls host qsort as upstream does"),
    Source("source/blender/render/intern/bake.cc", ("RE_bake_pixels_populate", "RE_bake_pixels_populate_from_objects", "RE_bake_margin"),
           "ported", "Pixel population, float32 ray setup, strict squared-distance cutoff, derivatives, position-to-barycentric conversion and ray-miss mask invalidation; population qualification recorded under docs/native-surface"),
    Source("source/blender/render/intern/texture_margin.cc", ("RE_generate_texturemargin_adjacentfaces",),
           "ported", "rust/src/blender/margin.rs; triangle/RGBA8/single-tile specialization; analytical padding fixture matches byte-for-byte"),
    Source("source/blender/render/intern/zbuf.cc", ("zbuf_add_to_span", "zspan_scanconvert"),
           "ported", "rust/src/blender/raster.rs; owned span storage and callback; arithmetic and endpoint rules retained"),
    Source("source/blender/imbuf/intern/filter.cc", ("IMB_filter_extend", "filter_make_index", "check_pixel_assigned"),
           "ported", "rust/src/blender/margin.rs; RGBA8 and explicit mask branch"),
    Source("source/blender/blenlib/intern/math_interp.cc", ("bilinear_byte_impl",),
           "ported", "rust/src/blender/margin.rs; scalar bordered RGBA8 branch"),

    Source("source/blender/blenlib/BLI_math_vector.hh", ("normalize_and_get_length", "normal_tri", "cross_tri", "dot"),
           "ported", "rust/src/blender/math.rs; float3 specialization used by face normals"),
    Source("source/blender/blenkernel/intern/mesh_normals.cc", ("face_normal_calc",),
           "ported", "rust/src/blender/math.rs; triangle-only branch and collapsed-face +Z fallback"),
    Source("source/blender/blenlib/intern/math_geom.cc", ("interp_barycentric_tri_v3", "isect_ray_tri_watertight_v3_precalc", "isect_ray_tri_watertight_v3"),
           "ported", "rust/src/blender/math.rs and ray.rs; preserved float32 interpolation and triangle intersection"),
    Source("source/blender/blenlib/intern/math_geom_inline.cc", ("axis_dominant_v3", "axis_dominant_v3_single"),
           "ported", "rust/src/blender/math.rs and ray.rs; dominant-axis tie conventions retained"),
    Source("source/blender/blenlib/intern/math_vector_inline.cc", ("normalize_v3", "add_newell_cross_v3_v3v3"),
           "ported", "rust/src/blender/math.rs and uv.rs; distinguish reciprocal-based C normalization from vector-API division"),
    Source("source/blender/blenlib/intern/math_vector.cc", ("ortho_basis_v3v3_v3",),
           "ported", "rust/src/blender/uv.rs; dominant projection basis and degenerate-axis fallback"),
    Source("source/blender/blenkernel/intern/bvhutils.cc", ("bvhtree_ray_tri_intersection", "mesh_corner_tris_spherecast", "bvhtree_new_common"),
           "ported", "rust/src/blender/ray.rs and bvh.rs; zero-radius triangle callback, float32 hit position, binary six-axis tree"),
    Source("source/blender/blenlib/intern/BLI_kdopbvh.cc", ("build_implicit_tree_helper", "implicit_leafs_index", "non_recursive_bvh_div_nodes", "partition_nth_element", "fast_ray_nearest_hit", "dfs_raycast", "bvhtree_ray_cast_data_precalc"),
           "ported", "rust/src/blender/bvh.rs; tree_type=2, axis=6, zero-radius triangles; indices replace pointers, source partition and tie visitation retained"),
    Source("intern/cycles/kernel/geom/triangle.h", ("triangle_interpolate", "triangle_attribute"),
           "ported", "rust/src/blender/image.rs; float2 corner attribute interpolation"),
    Source("intern/cycles/kernel/device/cpu/image.h", ("frac", "ImageInterpolator::read", "ImageInterpolator::interp_linear", "wrap_periodic"),
           "ported", "rust/src/blender/image.rs; uchar4 single image, linear REPEAT sampling and alpha association; decoded samples pass through the configured OCIO output curve"),
    Source("intern/cycles/kernel/integrator/init_from_bake.h", ("integrator_init_from_bake", "bake_jitter_barycentric", "bake_offset_towards_center"),
           "ported", "rust/src/blender/bake_sampling.rs; sample-zero center, vertex nudge, ten-retry hashed boundary jitter; shared-vertex fixture matches exactly"),
    Source("intern/cycles/util/hash.h", ("hash_uint2", "hash_uint3", "hash_float2_to_float2"),
           "ported", "rust/src/blender/bake_sampling.rs; wrapping uint32 arithmetic and float-bit input"),
    Source("intern/cycles/util/image.h", ("util_image_multiply_native", "util_image_cast_from_float"),
           "ported", "rust/src/blender/image.rs; uchar4 RGB times alpha divided by 255, truncating byte storage"),
    Source("intern/cycles/util/image_metadata.cpp", ("ImageMetaData::finalize", "conform_pixels_to_metadata_type", "ImageMetaData::conform_pixels"),
           "ported", "rust/src/blender/image.rs; supported RGB/RGBA8 sRGB input is associated once before linear filtering; source alpha is not connected to a shader"),
    Source("source/blender/blenlib/intern/math_base_inline.cc", ("unit_float_to_uchar_clamp",),
           "ported", "rust/src/blender/color.rs; bake-specific half-up byte conversion; ordinary TextureIO retains NumPy-compatible ties-to-even"),
    Source("source/blender/editors/object/object_bake_api.cc", ("write_internal_bake_pixels",),
           "ported", "RGBA8 image output and byte conversion translated; source-derived OCIO sRGB output curve; numerical equivalence rather than byte equality is the public contract"),
    Source("intern/cycles/util/color.h", ("color_srgb_to_linear",),
           "ported", "rust/src/blender/image.rs; scalar CPU decode after encoded-byte filtering; cross-platform SIMD qualification pending"),

)

OCIO_SOURCES = (
    Source("src/OpenColorIO/SSE.h", ("sseLog2", "sseExp2", "ssePower"),
           "ported", "rust/src/blender/ocio.rs; independent scalar float lanes retain SIMD arithmetic, bit extraction and polynomial order"),
    Source("src/OpenColorIO/ops/gamma/GammaOpUtils.cpp", ("ComputeParamsRev", "monCurveBreakRev", "monCurveSlopeRev"),
           "ported", "rust/src/blender/ocio.rs; gamma=2.4, offset=.055, double parameters converted to float32"),
    Source("src/OpenColorIO/ops/gamma/GammaOpCPU.cpp", ("GammaMoncurveOpCPURevSSE::apply",),
           "ported", "rust/src/blender/ocio.rs; configured inverse sRGB ExponentWithLinear output, no C++ runtime dependency"),
)


def source_files(root: Path, sources: tuple[Source,...]) -> list[BlenderSourceFile]:
    files=[]
    for source in sources:
        content=(root/source.path).read_bytes()
        header=content[:1000].decode()
        license=next(line.split("SPDX-License-Identifier:")[1].strip(" */\n")
                     for line in header.splitlines() if "SPDX-License-Identifier:" in line)
        files.append(BlenderSourceFile(path=source.path,sha256=hashlib.sha256(content).hexdigest(),
            license=license,symbols=list(source.symbols),status=source.status,adaptation=source.adaptation))
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("build/blender-source"))
    parser.add_argument("--output", type=Path, default=Path("rust/BLENDER_SOURCE.json"))
    parser.add_argument("--ocio-source", type=Path, default=Path("build/opencolorio-source"))
    parser.add_argument("--ocio-output", type=Path, default=Path("rust/OCIO_SOURCE.json"))
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise ValueError(f"Expected pinned Blender revision {REVISION}, got {revision}.")
    files = source_files(args.source,SOURCES)
    manifest = BlenderSourceManifest(version="5.2.1 LTS", revision=revision,
        repository="https://github.com/blender/blender", distribution_license="GPL-3.0-or-later", files=files)
    args.output.write_text(manifest.model_dump_json(indent=2) + "\n")
    ocio_revision=subprocess.check_output(["git","-C",str(args.ocio_source),"rev-parse","HEAD"],text=True).strip()
    if ocio_revision != OCIO_REVISION:
        raise ValueError(f"Expected pinned OCIO revision {OCIO_REVISION}, got {ocio_revision}.")
    ocio=BlenderSourceManifest(version="2.5.0",revision=ocio_revision,
        repository="https://github.com/AcademySoftwareFoundation/OpenColorIO",distribution_license="GPL-3.0-or-later",
        files=source_files(args.ocio_source,OCIO_SOURCES))
    args.ocio_output.write_text(ocio.model_dump_json(indent=2)+"\n")


if __name__ == "__main__":
    main()
