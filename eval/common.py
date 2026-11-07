"""Small shared primitives for deterministic evaluation."""
import hashlib
import json
from pathlib import Path

COUNTS = dict(drawbench=1)

def load(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)

def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def tasks(prompts, count, seed, rank=0, world=1):
    for i in range(rank, len(prompts), world):
        for j in range(count):
            yield dict(prompt_id=i, sample_id=i*count+j, seed=seed+i*count+j,
                       sample_index=j, metadata=prompts[i])

def records(root):
    return [json.loads(p.read_text()) for p in sorted(Path(root).glob('records/*.json'))]


def read_run(root):
    config = json.loads((Path(root) / 'config.json').read_text())
    rows = records(root)
    ids = [row['sample_id'] for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(range(config['expected'])):
        raise ValueError('incomplete or duplicate sample set')
    if not rows or any(not Path(row['image']).is_file() for row in rows):
        raise ValueError('missing images')
    return config, sorted(rows, key=lambda row: row['sample_id'])


def paired_runs(baseline, candidate):
    bc, base = read_run(baseline)
    tc, target = read_run(candidate)
    if bc['method'] != 'baseline' or tc['method'] != 'taylorseer':
        raise ValueError('expected baseline and taylorseer runs')
    ignore = {'method', 'order', 'interval', 'warmup', 'data'}
    if {k:v for k,v in bc.items() if k not in ignore} != {k:v for k,v in tc.items() if k not in ignore}:
        raise ValueError('incompatible generation configurations')
    pairs = list(zip(base, target, strict=True))
    for a, b in pairs:
        if (a['sample_id'], a['seed'], a['metadata']) != (b['sample_id'], b['seed'], b['metadata']):
            raise ValueError('unpaired samples')
    return bc, tc, pairs


def ensure_config(path, config):
    """Serialize config creation across local/shared-filesystem workers."""
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            if json.loads(path.read_text()) != config:
                raise ValueError('configuration mismatch; use a new output directory')
        else:
            save(path, config)
