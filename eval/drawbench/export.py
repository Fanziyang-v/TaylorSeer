import argparse
import csv
import random
import shutil
from pathlib import Path
from ..common import read_run as read
from ..common import save


def main():
    p=argparse.ArgumentParser(); p.add_argument('--baseline',required=True)
    p.add_argument('--candidate',required=True); p.add_argument('--output',required=True)
    a=p.parse_args(); bc,base=read(a.baseline); tc,target=read(a.candidate)
    if bc['data_sha256']!=tc['data_sha256']: raise ValueError('different datasets')
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    mapping={}; rows=[]
    for x,y in zip(base,target,strict=True):
        if (x['sample_id'],x['seed'],x['metadata'])!=(y['sample_id'],y['seed'],y['metadata']):
            raise ValueError('unpaired samples')
        i=x['sample_id']; swap=random.Random(i).getrandbits(1)
        pair=[x,y] if not swap else [y,x]
        for label,record in zip('AB',pair): shutil.copyfile(record['image'],out/f'{i}_{label}.png')
        mapping[i]='B' if swap else 'A'
        rows.append([i,x['metadata']['prompt'],f'{i}_A.png',f'{i}_B.png','',''])
    with (out/'votes.csv').open('w') as f:
        writer=csv.writer(f); writer.writerow(['id','prompt','A','B','quality','alignment']); writer.writerows(rows)
    save(out.parent/(out.name+'_private_mapping.json'),mapping)

if __name__=='__main__': main()
