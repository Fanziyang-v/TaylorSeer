from eval.common import tasks, save, records


def test_shards_and_seeds():
    prompts=[{'prompt':str(i)} for i in range(7)]
    expected=list(tasks(prompts,4,42))
    for world in (1,2,10):
        actual=[r for rank in range(world) for r in tasks(prompts,4,42,rank,world)]
        assert sorted(actual,key=lambda r:r['sample_id'])==expected
        assert len({r['seed'] for r in actual})==28


def test_atomic_records(tmp_path):
    save(tmp_path/'records/0.json',{'sample_id':0})
    assert records(tmp_path)==[{'sample_id':0}]


def test_drawbench_source():
    from pathlib import Path
    from eval.common import load
    path=Path('eval/drawbench/drawbench.jsonl')
    import hashlib
    assert hashlib.sha256(path.read_bytes()).hexdigest()=='b923632e6c8de120f3f3f4f2298f3b9d2d0bb4bebe74771557b13583f1a620f4'
    data=load(path)
    assert len(data)==200
    assert all(row['prompt'] and row['category'] for row in data)


def make_runs(tmp_path, n=3):
    from PIL import Image
    roots=[tmp_path/'base',tmp_path/'ts']
    for root,method in zip(roots,['baseline','taylorseer']):
        save(root/'config.json',dict(method=method,expected=n,data_sha256='abc',model='flux'))
        for i in range(n):
            image=root/f'{i}.png'
            Image.new('RGB',(16,16)).save(image)
            save(root/f'records/{i:08d}.json',dict(sample_id=i,seed=42+i,
                 metadata=dict(prompt=str(i),category='color'),image=str(image)))
    return roots


def test_pair_validation(tmp_path):
    import pytest
    from eval.common import paired_runs
    a,b=make_runs(tmp_path)
    assert len(paired_runs(a,b)[2])==3
    save(b/'records/00000003.json',dict(sample_id=0))
    with pytest.raises(ValueError): paired_runs(a,b)


def test_parallel_scoring_resume_and_report(tmp_path):
    from eval.common import paired_runs
    from eval.score import score_pairs
    from eval.report import build_report
    import pytest
    a,b=make_runs(tmp_path)
    pairs=paired_runs(a,b)[2]
    class FakeMetrics:
        def __init__(self): self.prompts=[]
        def compute(self,prompt,baseline,candidate):
            self.prompts.append(prompt)
            assert baseline.mode==candidate.mode=='RGB'
            return dict(lpips=0,ssim=1,psnr='inf',baseline_clipscore=2,
                        taylorseer_clipscore=3,baseline_imagereward=.1,taylorseer_imagereward=.2)
    single,parallel=tmp_path/'single',tmp_path/'parallel'
    metric=FakeMetrics()
    score_pairs(pairs,single,metric)
    for rank in range(5): score_pairs(pairs,parallel,metric,rank,5)
    assert records(single)==records(parallel)
    assert sorted(metric.prompts)==['0','0','1','1','2','2']
    score_pairs(pairs,single,metric)
    assert len(metric.prompts)==6
    save(single/'config.rank0.json',dict(expected=3))
    report=build_report(single)
    assert report['overall']['psnr']=='inf'
    assert report['overall']['clipscore_delta']==1
    assert report['categories']['color']['samples']==3
    (single/'records/00000002.json').unlink()
    with pytest.raises(ValueError): build_report(single)


def test_report_weighting():
    from eval.report import summarize
    rows=[dict(lpips=i,ssim=1,psnr=10,baseline_clipscore=2,taylorseer_clipscore=3,
               baseline_imagereward=0,taylorseer_imagereward=1) for i in range(3)]
    assert summarize(rows)['lpips']==1


def test_config_guard(tmp_path):
    import pytest
    from eval.common import ensure_config
    path=tmp_path/'config.json'
    ensure_config(path,{'seed':42})
    ensure_config(path,{'seed':42})
    with pytest.raises(ValueError): ensure_config(path,{'seed':0})


def test_real_image_pair_metrics():
    import pytest
    pytest.importorskip('torchmetrics')
    import torch
    from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure
    x=torch.rand(1,3,32,32)
    assert torch.isinf(peak_signal_noise_ratio(x,x,data_range=1.0))
    torch.testing.assert_close(structural_similarity_index_measure(x,x,data_range=1.0),torch.tensor(1.))
