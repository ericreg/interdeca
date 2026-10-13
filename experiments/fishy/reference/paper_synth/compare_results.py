"""Compare a fresh rerun against a reference snapshot of results/ and figures/.

    cp -r results results_ref && cp -r figures figures_ref     # before rerunning
    ... rerun (README section 4) ...
    python compare_results.py [--ref results_ref] [--new results] \
                              [--ref-figs figures_ref] [--new-figs figures] [-v]

Every numeric leaf of every JSON and every array of every .npz is compared. Each file is
reported as IDENTICAL, CLOSE (all differences within --rtol) or DIFFERS (with the largest
relative difference and, with -v, the individual leaves). Wall-clock fields are ignored.
Figures are compared pixel-wise.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

HERE = Path(__file__).resolve().parent
SKIP = {"seconds", "elapsed", "time", "runtime_s", "wall_s"}


def leaves(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items():
            if k not in SKIP:
                yield from leaves(v, f"{p}.{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from leaves(v, f"{p}[{i}]")
    else:
        yield p, o


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def rel(a, b):
    """Relative difference, or None for exact equality; inf for non-numeric mismatches."""
    if _num(a) and _num(b):
        if a == b or (a != a and b != b):
            return None
        return abs(a - b) / max(abs(a), abs(b), 1e-12)
    return None if a == b else float("inf")


def compare_json(ref: Path, new: Path):
    A, B = dict(leaves(json.loads(new.read_text()))), dict(leaves(json.loads(ref.read_text())))
    out = []
    for k in sorted(set(A) | set(B)):
        if k not in A or k not in B:
            out.append((k, float("inf"), A.get(k, "<absent>"), B.get(k, "<absent>")))
        else:
            r = rel(A[k], B[k])
            if r is not None:
                out.append((k, r, A[k], B[k]))
    return out, len(set(A) | set(B))


def compare_npz(ref: Path, new: Path):
    A, B = np.load(new, allow_pickle=True), np.load(ref, allow_pickle=True)
    out = []
    for k in sorted(set(A.files) | set(B.files)):
        if k not in A.files or k not in B.files:
            out.append((k, float("inf"), "present" if k in A.files else "<absent>",
                        "present" if k in B.files else "<absent>"))
            continue
        a, b = A[k], B[k]
        if a.shape != b.shape:
            out.append((k, float("inf"), a.shape, b.shape))
        elif a.dtype.kind in "fc" and b.dtype.kind in "fc":
            d = np.abs(a - b)
            fin = np.isfinite(d)
            if np.any(np.isnan(a) != np.isnan(b)):
                out.append((k, float("inf"), "NaN pattern", "differs"))
            elif fin.any() and d[fin].max() > 0:
                # relative to the array's magnitude, so near-zero entries don't dominate
                scale = np.abs(b)[np.isfinite(b)].max() if np.isfinite(b).any() else 1.0
                r = float(d[fin].max() / max(scale, 1e-12))
                out.append((k, r, f"max|d|={d[fin].max():.3g}", f"{(d[fin] > 0).mean():.1%} of entries"))
        elif not np.array_equal(a, b):
            out.append((k, float("inf"), "differs", f"{(a != b).mean():.1%} of entries"))
    return out, len(set(A.files) | set(B.files))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", default=HERE / "results_ref", type=Path)
    ap.add_argument("--new", default=HERE / "results", type=Path)
    ap.add_argument("--ref-figs", default=HERE / "figures_ref", type=Path)
    ap.add_argument("--new-figs", default=HERE / "figures", type=Path)
    ap.add_argument("--rtol", default=1e-6, type=float, help="relative difference treated as CLOSE")
    ap.add_argument("-v", "--verbose", action="store_true", help="list every differing leaf")
    args = ap.parse_args()

    n_bad = 0
    print(f"results: {args.new}  vs reference {args.ref}")
    for ref in sorted(list(args.ref.glob("*.json")) + list(args.ref.glob("*.npz"))):
        new = args.new / ref.name
        if not new.exists():
            print(f"  {ref.name:40s} MISSING from rerun"); n_bad += 1
            continue
        diffs, n = (compare_json if ref.suffix == ".json" else compare_npz)(ref, new)
        worst = max((d[1] for d in diffs), default=0.0)
        if not diffs:
            status = "IDENTICAL"
        elif worst <= args.rtol:
            status = f"CLOSE      ({len(diffs)}/{n} differ, max rel {worst:.1e})"
        else:
            status = f"DIFFERS    ({len(diffs)}/{n} differ, max rel {worst:.2g})"
            n_bad += 1
        print(f"  {ref.name:40s} {status}")
        if args.verbose or status.startswith("DIFFERS"):
            shown = sorted(diffs, key=lambda d: -d[1])
            for k, r, a, b in shown[: None if args.verbose else 8]:
                print(f"      {k}: new={a!r} ref={b!r} (rel {r:.2g})")
            if not args.verbose and len(shown) > 8:
                print(f"      ... {len(shown) - 8} more (-v to list all)")
    extra = {p.name for p in args.new.glob("*.json")} | {p.name for p in args.new.glob("*.npz")}
    extra -= {p.name for p in args.ref.iterdir()} if args.ref.exists() else set()
    for name in sorted(extra):
        print(f"  {name:40s} new file (no reference)")

    if args.ref_figs.exists():
        print(f"figures: {args.new_figs}  vs reference {args.ref_figs}")
        for ref in sorted(args.ref_figs.glob("*.png")):
            new = args.new_figs / ref.name
            if not new.exists():
                print(f"  {ref.name:40s} MISSING from rerun"); n_bad += 1
                continue
            a, b = imageio.imread(new), imageio.imread(ref)
            if a.shape != b.shape:
                print(f"  {ref.name:40s} DIFFERS    (size {a.shape} vs {b.shape})"); n_bad += 1
            elif np.array_equal(a, b):
                print(f"  {ref.name:40s} IDENTICAL")
            else:
                frac = np.any(a != b, axis=-1).mean() if a.ndim == 3 else (a != b).mean()
                print(f"  {ref.name:40s} DIFFERS    ({frac:.2%} of pixels)"); n_bad += 1

    print(f"\n{n_bad} file(s) outside tolerance")
    raise SystemExit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
