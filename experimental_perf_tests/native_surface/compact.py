"""Retain verified OBJ bytes once while compacting generated workflow copies."""

import argparse
import gzip
import hashlib
import json
import os
import shutil
from pathlib import Path


def digest(path: Path, compressed: bool = False) -> str:
    with (gzip.open(path,'rb') if compressed else path.open('rb')) as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def compact(directory: Path, archive_roots: tuple[Path,...]) -> None:
    """Hash originals and verify archives before removing any generated copy.

    Existing archives are immutable and can be hard-linked when their decoded
    bytes match. New files receive a verified gzip archive. Numerical boundaries,
    presentations, textures, logs and input datasets remain untouched.
    """
    known={}
    for root in archive_roots:
        for manifest in root.rglob('omitted-artifacts.json'):
            for relative,sha in json.loads(manifest.read_text()).items():
                archive=manifest.parent/'mesh-archive'/(relative+'.gz')
                if archive.is_file():
                    known.setdefault(sha,archive)
    manifest=directory/'omitted-artifacts.json'
    omitted=json.loads(manifest.read_text()) if manifest.exists() else {}
    linked=0
    archived=0
    for original in sorted((directory/'atlas').rglob('*.obj')):
        sha=digest(original)
        relative=str(original.relative_to(directory))
        destination=directory/'mesh-archive'/(relative+'.gz')
        destination.parent.mkdir(parents=True,exist_ok=True)
        if not destination.exists():
            if sha in known:
                os.link(known[sha],destination)
                linked+=1
            else:
                temporary=destination.with_suffix('.partial')
                temporary.unlink(missing_ok=True)
                with original.open('rb') as source, temporary.open('xb') as output:
                    with gzip.GzipFile(fileobj=output,mode='wb',compresslevel=3,mtime=0) as zipped:
                        shutil.copyfileobj(source,zipped,1024*1024)
                temporary.rename(destination)
                archived+=1
        if digest(destination,compressed=True)!=sha:
            raise ValueError(f'Archive differs from original: {original}')
        omitted[relative]=sha
        temporary=manifest.with_suffix('.partial')
        temporary.write_text(json.dumps(omitted,indent=2)+'\n')
        temporary.replace(manifest)
        known[sha]=destination
        original.unlink()
    print(f'{directory}: {linked} shared archives, {archived} new archives',flush=True)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    compact(args.directory,(Path('build/profiling'),Path('build/native-surface')))


if __name__=='__main__':
    main()
