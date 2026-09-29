"""Safely extract the separately downloaded dataset release asset."""
import argparse
import zipfile
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive',type=Path)
    p.add_argument('--output',type=Path,default=Path('data/full'))
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    base=a.output.resolve()
    with zipfile.ZipFile(a.archive) as z:
        for item in z.infolist():
            target=(base/item.filename).resolve()
            if base not in target.parents and target!=base:
                raise ValueError(f'Unsafe archive member: {item.filename}')
        z.extractall(base)
    print(f'Extracted to {base}')

if __name__=='__main__': main()
