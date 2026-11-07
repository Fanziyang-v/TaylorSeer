"""Normalize local official prompt files without rewriting text."""
import argparse
import json
from pathlib import Path
from .common import save


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['geneval','drawbench','t2i_compbench'],required=True)
    p.add_argument('--source',required=True,help='Official metadata JSONL, prompt TXT, or CompBench dataset directory')
    p.add_argument('--revision',required=True,help='Source commit or release identifier')
    p.add_argument('--output',required=True)
    a=p.parse_args(); source=Path(a.source)
    if a.dataset=='geneval':
        rows=[json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    elif a.dataset=='drawbench':
        rows=[{'prompt':line} for line in source.read_text().splitlines() if line.strip()]
    else:
        rows=[]
        for category in ('color','shape','texture','spatial','non_spatial','complex'):
            rows.extend({'prompt':line,'category':category} for line in
                        (source/f'{category}_val.txt').read_text().splitlines() if line.strip())
    if not rows or any(not r.get('prompt') for r in rows): raise ValueError('empty prompt data')
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows))
    save(out.with_suffix('.source.json'),dict(dataset=a.dataset,source=str(source),revision=a.revision,count=len(rows)))

if __name__=='__main__': main()
