# From Reusing to Forecasting: Accelerating Diffusion Models with TaylorSeers

Here, We provide a highly readable and easy-to-use implementation of TaylorSeer, an awesome, efficient generation paradigm with *cache-then-forecast* mechanism.

## ✨Introduction

> **Abstract**: Diffusion Transformers (DiT) have revolutionized high-fidelity image and video synthesis, yet their computational demands remain prohibitive for real-time applications. To solve this problem, feature caching has been proposed to accelerate diffusion models by caching the features in the previous timesteps and then reusing them in the following timesteps. However, at timesteps with significant intervals, the feature similarity in diffusion models decreases substantially, leading to a pronounced increase in errors introduced by feature caching, significantly harming the generation quality. To solve this problem, we propose TaylorSeer, which firstly shows that features of diffusion models at future timesteps can be predicted based on their values at previous timesteps. Based on the fact that features change slowly and continuously across timesteps, TaylorSeer employs a differential method to approximate the higher-order derivatives of features and predict features in future timesteps with Taylor series expansion. Extensive experiments demonstrate its significant effectiveness in both image and video synthesis, especially in high acceleration ratios. For instance, it achieves an almost lossless acceleration of 4.99 $\times$ on FLUX and 5.00 $\times$ on HunyuanVideo without additional training. On DiT, it achieves 3.41 lower FID compared with previous SOTA at 4.53 $\times$ acceleration. Our code is provided in the supplementary materials and will be made publicly available on GitHub. Our codes have been released in Github:https://github.com/Shenyi-Z/TaylorSeer

![Illustration of TaylorSeer](assets/taylorseer.png)

## 📋TODO List

- [x] Initialize this project.
- [x] Implement core algorithm of TaylorSeer.
- [x] Support DiT and FLUX.1-dev models.
- [ ] Support evaluation.


## 📦Installation

In this project, we use [uv](https://github.com/astral-sh/uv) for package management.

1. **Clone this repository and navigate to the TaylorSeer folder:**

```
git clone https://github.com/Fanziyang-v/TaylorSeer.git
cd TaylorSeer
```

2. **Install the inference package:**

```
uv sync

## 🔌Adapters

```

Currently, we support 2 adapters: *DiT* and *FLUX*, with each adapter targets one model family. Inference demos can be found in `playground/`. Here is an running example:

```
python playground/flux.py \
    --prompt "A cat holding a sign that says hello world" \
    --num-inference-steps 50 \
    --guidance-scale 3.5 \
    --enable-taylorseer \
    --order 2 \
    --interval 4 \
    --warmup_steps 3 \
    --height 1024 \
    --width 1024 \
    --output-path flux_ts.png
```


## 👏Acknowledgement

This project is built upon the official implementation of [TaylorSeer](https://github.com/Shenyi-Z/TaylorSeer) and [diffusers](https://github.com/huggingface/diffusers). Thanks for their excellent work!
