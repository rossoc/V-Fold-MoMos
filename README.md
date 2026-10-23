# MoMos — Neural Network Simplification through Kolmogorov Complexity Bounding

## Abstract

Modern deep neural networks achieve remarkable empirical success through
over-parameterization, but at the expense of high computational, memory, and
energy costs. Trained DNNs exhibit substantial structural redundancy, which we
investigate through Algorithmic Information Theory (AIT) and Kolmogorov
complexity. Building on the 
[Mosaic-of-Motifs (MoMos) framework](https://arxiv.org/abs/2602.14896), this 
project provides a comprehensive characterization of algorithmic compression in
deep learning.

1. [**Quantized Block Decomposition (QuBD)**](https://arxiv.org/pdf/2605.15551):
   to estimate the algorithmic complexity of DNNs.
2. **Implementing MoMos2D**: aligned with internal network structure.
3. **V-Fold MoMos**: the first recursive step of MoMos.

Together these results give theoretical and empirical foundations for designing
neural networks with strictly bounded algorithmic complexity.

## Training entry point — `src/train.py`

`train.py` is the single PyTorch Lightning + Hydra command that runs every
experiment in this repo. Key behaviour:

- **Config-driven (Hydra).** Reads `src/configs/config.yaml` plus overrides on
  the CLI (`dataset.name`, `quantization`, `model.*`, `accelerator`, …). No
  hard-coded experiment settings.
- **`run_training(cfg, optuna_trial, datamodule)`** is the core function:
  sets the seed, resolves the runtime (CPU/CUDA/MPS), auto-selects mixed
  precision (bf16 on Ampere+, else 32-bit), builds the data module and the
  backbone, assembles callbacks, and launches a `L.Trainer`.
- **Logging**: optional Weights & Biases (`cfg.wandb.enabled`); otherwise
  logging is disabled so no `lightning_logs/` is created.
- **Outputs**: trains, validates on the best checkpoint, evaluates on the test
  split, and returns a dict of metrics plus the best model path.

Run it with:

```bash
uv run python src/train.py dataset.name=cifar10 quantization=momos accelerator=cuda
```

## `notebook/` — analysis and figures

A set of scripts and notebooks that turn training artifacts into the results in
the thesis. Each `*.py` is the source-of-truth (the `*.ipynb` are companion
notebooks):

- **2D MoMos**: `momos2d_analysis.py` (main figure driver), `momos2d_wa.py`,
  `v-fold_wa.py`, `vfold_vit_analysis.py`.
- **Hierarchical MoMos**: `momos_hierarchical_analysis.py` (+ notebook).

Figures render through the `src/view` matplotlib wrapper (no raw pyplot).

## `scripts/` — ready-to-run training commands

Bash launchers that wrap `uv run python src/train.py` with the exact overrides
used for each experiment family. Use these as the canonical examples of how the
models were trained.

Example invocation:

```bash
bash scripts/momos-sweep.sh
```

## Layout

```
src/            source: train.py, model/ (backbones + LitClassifier),
                data/, quantizers/, configs/ (Hydra), utils/, view/
notebook/       analysis + figure scripts
scripts/        example training commands
tests/          test suite (uv run pytest)
```

## Setup

```bash
uv sync          # install dependencies from pyproject.toml
uv run pytest    # run tests
```
