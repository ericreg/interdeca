"""Record whole-host memory/swap context while a particular benchmark parent lives."""

import argparse
import shutil
import time
from pathlib import Path

import psutil

from tools.profiling.serde import HostMemorySample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pid", type=int)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    parent = psutil.Process(args.pid)
    created = parent.create_time()
    # PID identity prevents an unrelated later process from extending this capture.
    # Append permits a resumed benchmark to retain its earlier host context.
    with args.output.open("a") as stream:
        while True:
            # The recorder can exit between the liveness and identity checks.
            # A normal benchmark completion must also end monitoring cleanly.
            try:
                if not parent.is_running() or parent.create_time() != created:
                    break
            except psutil.NoSuchProcess:
                break
            memory, swap = psutil.virtual_memory(), psutil.swap_memory()
            sample = HostMemorySample(
                time=time.perf_counter(),
                unix_time=time.time(),
                available_bytes=memory.available,
                swap_used_bytes=swap.used,
                swap_in_bytes=swap.sin,
                swap_out_bytes=swap.sout,
                free_disk_bytes=shutil.disk_usage(args.output.parent).free,
            )
            stream.write(sample.model_dump_json() + "\n")
            stream.flush()
            time.sleep(2)


if __name__ == "__main__":
    main()
