"""Run paired full workflows sequentially, keeping every fresh child and report."""

import argparse
import subprocess
import sys
from pathlib import Path

from tools.profiling.serde import Run


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("build/native-surface/workflow-repeated"))
    parser.add_argument("--repetitions",type=int,default=3)
    parser.add_argument("--backend",choices=("native","morphoweave"))
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    if args.repetitions<3:
        parser.error("The full-workflow release gate requires three fresh runs.")
    script=Path(__file__).with_name("workflow.py")
    compare=Path(__file__).with_name("workflow_compare.py")
    args.output.mkdir(parents=True,exist_ok=True)
    failed=False
    for backend in ((args.backend,) if args.backend else ("native","morphoweave")):
        for repetition in range(1,args.repetitions+1):
            paths=[]
            for engine in ("blender","rust"):
                output=args.output/f"{backend}-{engine}-{repetition}"
                paths.append(output)
                record=output/'run.json'
                # Resume only complete successful children. A partial directory
                # remains evidence and is never overwritten by a retry.
                if args.resume and record.is_file():
                    run=Run.model_validate_json(record.read_text())
                    if run.exit_code==0:
                        print(f"Reusing completed {output}",flush=True)
                        continue
                if output.exists():
                    raise FileExistsError(f"Incomplete/existing run: {output}. Choose a new destination.")
                command=[sys.executable,str(script),'--backend',backend,'--output',str(output)]
                if engine=='rust':
                    command.append('--native')
                result=subprocess.run(command,check=False)
                if result.returncode:
                    return result.returncode
            command=[sys.executable,str(compare),'--reference',str(paths[0]),'--candidate',str(paths[1]),
                     '--output',str(args.output/f'{backend}-comparison-{repetition}.json')]
            with (args.output/f'{backend}-comparison-{repetition}.log').open('w') as log:
                result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=False)
            if result.returncode not in (0,2):
                return result.returncode
            failed |= result.returncode==2
            print(f"{backend} comparison {repetition}: exit {result.returncode}",flush=True)
    return 2 if failed else 0


if __name__=='__main__':
    sys.exit(main())
