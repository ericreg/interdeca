"""Run baker qualification in fresh interpreters, retaining failed gate reports.

This measures the native kernel and process RSS. Reused Blender references are
quality oracles, not comparable measurements of Blender's runtime or memory.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("build/native-surface/repeated"))
    parser.add_argument("--reference-root", type=Path, default=Path("build/native-surface"),
                        help="Contains native-2048 and morphoweave-2048 reference runs.")
    parser.add_argument("--captures", type=Path, default=Path("build/profiling/parry-47-2026-10-04/current"))
    parser.add_argument("--inputs", type=Path, default=Path.home() / "Library/Application Support/InterDeCA/full-population-2026-10-03/inputs")
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("At least one repetition is required.")
    failed = False
    script = Path(__file__).with_name("benchmark.py")
    # Sequential runs avoid measuring competition for the same CPU or disks.
    for backend in ("native", "morphoweave"):
        for resolution in (512, 2048):
            for repetition in range(1, args.repetitions + 1):
                output = args.output / f"{backend}-{resolution}-{repetition}"
                output.mkdir(parents=True, exist_ok=True)
                command = [sys.executable, str(script), "--backend", backend,
                           "--resolution", str(resolution), "--output", str(output),
                           "--captures", str(args.captures), "--inputs", str(args.inputs)]
                if resolution == 2048:
                    command += ["--reference-dir", str(args.reference_root / f"{backend}-2048")]
                with (output / "run.log").open("w") as log:
                    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
                print(f"{output}: exit {result.returncode}", flush=True)
                if result.returncode not in (0, 2):
                    return result.returncode
                failed |= result.returncode == 2
    return 2 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
