"""Compare retained full-workflow boundaries, preserving every measured difference.

Cleanup is checked modulo a bijective vertex map with oriented face winding.
Downstream arrays are reported directly. PCA signs are aligned before comparing
scores/components; arbitrary rotations or reordered feature columns are refused.
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from meshcheck import oriented_faces
from serde import ArrayDifference, WorkflowComparison
from tools.profiling.serde import Run


def difference(name: str, observed: np.ndarray, expected: np.ndarray) -> ArrayDifference:
    """Record exact equality and absolute errors without rounding them away."""
    if observed.shape != expected.shape:
        return ArrayDifference(name=name,same_shape=False,exact=False)
    delta = np.abs(observed.astype(float)-expected.astype(float))
    return ArrayDifference(name=name,same_shape=True,exact=bool(np.array_equal(observed,expected)),
        maximum=float(delta.max()) if delta.size else 0,
        mean=float(delta.mean()) if delta.size else 0,
        rms=float(np.sqrt(np.mean(delta*delta))) if delta.size else 0)


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference",type=Path,required=True)
    parser.add_argument("--candidate",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    reference=Run.model_validate_json((args.reference/'run.json').read_text())
    candidate=Run.model_validate_json((args.candidate/'run.json').read_text())
    if reference.exit_code or candidate.exit_code or reference.config.backend != candidate.config.backend:
        raise ValueError("Both runs must succeed using the same atlas backend.")
    ref=args.reference/'numerical'
    new=args.candidate/'numerical'
    arrays=[]
    with np.load(ref/'pre-cleanup.npz') as expected, np.load(new/'pre-cleanup.npz') as observed:
        same_input=all(np.array_equal(observed[key],expected[key]) for key in ('vertices','faces'))
        tolerance=float(np.linalg.norm(np.ptp(expected['vertices'],axis=0)))*1e-6
    if not same_input:
        raise ValueError("The paired conditions did not receive identical pre-cleanup geometry.")
    with np.load(ref/'post-cleanup.npz') as expected, np.load(new/'post-cleanup.npz') as observed:
        distance,mapping=cKDTree(expected['vertices']).query(observed['vertices'])
        bijective=(len(observed['vertices'])==len(expected['vertices']) and len(np.unique(mapping))==len(mapping))
        connectivity=bool(np.array_equal(oriented_faces(mapping[observed['faces']]),oriented_faces(expected['faces'])))
        cleanup_maximum=float(distance.max())
        for key in ('vertices','faces'):
            arrays.append(difference('cleanup/'+key,observed[key],expected[key]))
    with np.load(ref/'post-uv.npz') as expected,np.load(new/'post-uv.npz') as observed:
        arrays.append(difference('uv',observed['uv'],expected['uv']))
    # Compare keyed specimen arrays and identities before numerical model output.
    # Missing/reordered samples must not be hidden by a statistical summary.
    reference_files={p.name for p in ref.glob('*.npz')}
    candidate_files={p.name for p in new.glob('*.npz')}
    if reference_files != candidate_files:
        raise ValueError("Paired conditions retain different numerical outputs.")
    for name in sorted(reference_files-{'pre-cleanup.npz','post-cleanup.npz','post-uv.npz','pca.npz'}):
        with np.load(ref/name) as expected,np.load(new/name) as observed:
            for key in expected.files:
                if expected[key].dtype.kind in 'US':
                    if not np.array_equal(observed[key],expected[key]):
                        raise ValueError(f"Changed array identities: {name}/{key}.")
                    continue
                arrays.append(difference(name+'/'+key,observed[key],expected[key]))
    with np.load(ref/'pca.npz') as expected,np.load(new/'pca.npz') as observed:
        # Score error is dimensionless relative to the reference score energy.
        score_rms = float(np.sqrt(np.mean(expected['scores'] ** 2)))
        signs=np.where(np.sum(expected['components']*observed['components'],axis=1)<0,-1,1)
        for key in expected.files:
            value=observed[key]
            if key=='scores':
                value=value*signs
            elif key=='components':
                value=value*signs[:,None]
            arrays.append(difference('pca/'+key,value,expected[key]))
    # Byte equality is diagnostic. Small RGB rounding differences may propagate
    # into Lab and PCA without materially changing the population model.
    limits = {"rgb_mean": 1/255, "rgb_maximum": 5/255,
              "lab_rms": .05, "lab_maximum": 1,
              "pca_score_relative_rms": 1e-4, "pca_components_maximum": 1e-3,
              "pca_variance_maximum": 1e-4}
    downstream_passed = True
    score_relative_rms = None
    for row in arrays:
        if row.name in ('cleanup/vertices', 'cleanup/faces', 'uv'):
            continue
        passed = row.same_shape
        if row.name.endswith('/rgb'):
            passed = passed and row.mean <= limits['rgb_mean'] and row.maximum <= limits['rgb_maximum']
        elif row.name.endswith('/lab') or row.name == 'features.npz/values':
            passed = passed and row.rms <= limits['lab_rms'] and row.maximum <= limits['lab_maximum']
        elif row.name == 'pca/scores':
            score_relative_rms = row.rms / max(score_rms, np.finfo(float).eps) if row.same_shape else None
            passed = passed and score_relative_rms <= limits['pca_score_relative_rms']
        elif row.name in ('pca/components', 'pca/variance'):
            passed = passed and row.maximum <= limits[row.name.replace('/', '_') + '_maximum']
        else:
            # Masks, geometry, model scaling and identities remain exact.
            passed = passed and row.exact
        downstream_passed = downstream_passed and passed
    # Runtime and simultaneous tree RSS are compared only across these complete,
    # identically instrumented fresh children, not reused image-reference runs.
    report=WorkflowComparison(backend=reference.config.backend,reference=args.reference.resolve(),candidate=args.candidate.resolve(),
        cleanup_bijective=bijective,cleanup_same_connectivity=connectivity,
        cleanup_max_distance=cleanup_maximum,cleanup_coordinate_tolerance=tolerance,
        cleanup_passed=bijective and connectivity and cleanup_maximum<=tolerance,
        reference_wall_seconds=reference.wall_seconds,candidate_wall_seconds=candidate.wall_seconds,
        runtime_passed=candidate.wall_seconds<=reference.wall_seconds,
        reference_peak_rss_bytes=reference.tree_peak_rss_bytes,candidate_peak_rss_bytes=candidate.tree_peak_rss_bytes,
        memory_passed=candidate.tree_peak_rss_bytes<=reference.tree_peak_rss_bytes*1.1,
        downstream_arrays_exact=all(row.exact for row in arrays if row.name not in ('cleanup/vertices','cleanup/faces','uv')),
        downstream_passed=downstream_passed, pca_score_relative_rms=score_relative_rms,
        numerical_limits=limits, arrays=arrays)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(report.model_dump_json(indent=2)+'\n')
    print(report.model_dump_json(indent=2))
    return 0 if report.cleanup_passed and report.runtime_passed and report.memory_passed and report.downstream_passed else 2


if __name__=='__main__':
    raise SystemExit(main())
