#!/bin/bash
set -e

# (rows, cols) block shapes
SHAPES=("1 2" "2 1")

for shape in "${SHAPES[@]}"; do
    read -r row col <<< "$shape"

    echo "------------------------------------------------"
    echo "Running soft MoMos (MLP, CIFAR-10): rows=$row, cols=$col"
    echo "------------------------------------------------"

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python src/train.py \
        epochs=400 \
        accelerator=cuda \
        periodic_checkpoint=true \
        dataset.name=cifar10 \
        model=mlp \
        seed=42 \
        wandb.enabled=true \
        wandb.project=soft-momos \
        wandb.name="soft_momos_mlp_c${col}_r${row}_s42" \
        "metrics=[sparsity,l2,gzip,bz2,lzma,bdm]" \
        quantization=soft_momos \
        quantization.rows=$row \
        quantization.cols=$col \
        quantization.k=256 \
        quantization.final_k=256 \
        quantization.force_zero=true
done