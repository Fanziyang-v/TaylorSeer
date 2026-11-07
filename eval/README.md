# DrawBench evaluation

FLUX.1-dev only. The bundled `drawbench/drawbench.jsonl` is copied unchanged
from `TaylorSeer-v1/TaylorSeer-v0/data/drawbench.jsonl`: 200 categorized prompts.
Default: one image per prompt, 200 images per method. Generation and scoring
use independent replicas and deterministic data shards, without DDP wrapping.

Install the optional `evaluation` dependencies (`pip install -e '.[evaluation]'`,
or `uv sync --extra evaluation --locked`). ImageReward's legacy dependencies require
Transformers 5 with the local `reward_compat` bridge and setuptools below 81.
The bridge is limited to ImageReward's vendored BERT; FLUX is not patched. Metric weights download on first
use; pre-cache them before a multi-process run. In an offline environment,
`--clip-model` and `--reward-model` can specify compatible local weights.

Run from the repository root:

```bash
python -m eval.generate --model checkpoints/FLUX.1-dev --method baseline --output outputs/drawbench/base
python -m eval.generate --model checkpoints/FLUX.1-dev --method taylorseer --output outputs/drawbench/ts
python -m eval.score --baseline outputs/drawbench/base --candidate outputs/drawbench/ts --output outputs/drawbench/scores
python -m eval.report --scores outputs/drawbench/scores --output outputs/drawbench/report.json
```

For parallel generation **or** scoring, replace `python -m` with
`torchrun --standalone --nproc_per_node=4 -m`. One GPU holds a complete model
replica. The score process uses LOCAL_RANK; `--device cpu` is available for
single-process scoring. All workers need a shared output filesystem.

Use `--limit 2` on both generation commands for smoke testing. Generation
supports `--seed`, `--steps`, `--height`, `--width`, `--guidance`, `--order`,
`--interval`, `--warmup`, and optional `--samples`. Defaults: seed 42, 50 steps,
1024x1024, guidance 3.5, order 1, interval 6, warmup 3. Seed is base seed plus
global sample ID. It does not depend on GPU count. Each pipeline call has a
fresh TaylorSeer cache. Matching image and record files are skipped on restart.
Changed configurations require a new directory. Scoring records include image
hashes and refuse reuse when their input images change.

## Five metrics

- CLIPScore: `openai/clip-vit-base-patch16`, separately for each method.
- ImageReward: `ImageReward-v1.0`, separately for each method.
- LPIPS: AlexNet, candidate versus baseline, RGB [0,1] with normalize=True.
- SSIM and PSNR: candidate versus baseline, RGB [0,1], data_range=1.

All scores are calculated per image before averaging. The report contains
sample counts, overall and per-category means, plus candidate-minus-baseline
CLIPScore/ImageReward deltas. LPIPS is lower-is-better; the other metrics are
higher-is-better. Pair metrics measure deviation from baseline, not independent
image quality. Infinite PSNR is encoded as the string `"inf"`, with its sample
count. Raw records remain available in `scores/records/`.

Incomplete or duplicate sample sets and incompatible generation configurations
are rejected. Final reporting checks complete scoring coverage and worker
configuration agreement. Reports contain only the five quality metrics;
generation timing remains in raw generation records.

```bash
PYTHONPATH=.:src python -m pytest tests/test_eval.py -q
```

Heavy metric weights are not needed by unit tests. The optional real SSIM/PSNR
test runs when torchmetrics is installed. Real FLUX generation and downloaded
CLIP/LPIPS/ImageReward smoke tests require a configured inference environment.
GenEval, CompBench and human-evaluation tools are outside this workflow.


## CPU-only preparation

`uv sync --extra evaluation --locked` works without a GPU or NVIDIA driver.
Linux keeps the project's CUDA-enabled PyTorch wheels so the same lockfile
can be reused on GPU nodes. Their CPU operations are available on CPU nodes;
no CUDA toolkit compilation is required. This command installs dependencies,
not model weights.

```bash
uv sync --extra evaluation --locked
uv pip check
uv run --extra evaluation --locked python -m pytest tests -q
```

Generation still requires a GPU. Scoring can run on CPU using `--device cpu`.
The offline tests exercise ImageReward's small BERT cross-attention forward,
SSIM/PSNR and FLUX import/adapter behavior without downloading model weights.
