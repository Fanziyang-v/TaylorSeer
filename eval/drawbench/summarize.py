"""Summarize votes: use A, B, or tie in quality/alignment columns."""
import argparse
import csv
import json
from ..common import save


def main():
    p=argparse.ArgumentParser(); p.add_argument('--votes',required=True)
    p.add_argument('--mapping',required=True); p.add_argument('--output',required=True)
    a=p.parse_args(); mapping=json.load(open(a.mapping)); result={}
    rows=list(csv.DictReader(open(a.votes)))
    for metric in ('quality','alignment'):
        counts=dict(baseline=0,taylorseer=0,tie=0)
        for row in rows:
            vote=row[metric].strip()
            if not vote: continue
            if vote not in ('A','B','tie'): raise ValueError('expected A, B, or tie')
            label='tie' if vote=='tie' else 'baseline' if vote==mapping[row['id']] else 'taylorseer'
            counts[label]+=1
        result[metric]=dict(counts,valid_votes=sum(counts.values()))
    save(a.output,result)

if __name__=='__main__': main()
