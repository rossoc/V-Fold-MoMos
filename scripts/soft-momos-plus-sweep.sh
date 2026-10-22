#!/bin/bash
set -e

# (rows, cols) block shapes
SHAPES=("1 2" "2 1" "2 2")
# Training-mosaic size: absolute k=256, then capacities (fractions of blocks)
KS=("k=256 capacity=null" "k=null capacity=0.001" "k=null capacity=0.01")
LAMS=(0.01 0.1 1 10)
SEEDS=(42 777 50 78 291)

for shape in "${SHAPES[@]}"; do
    read -r row col <<< "$shape"
    for kspec in "${KS[@]}"; do
        read -r kval capval <<< "$kspec"
        for lam in "${LAMS[@]}"; do
            for i in "${!SEEDS[@]}"; do
                seed="${SEEDS[$i]}"
                fold=$i  # the fold is linked to the seed index (0..4)

                echo "------------------------------------------------"
                echo "Running: cols=$col, rows=$row, $kspec, lam=$lam, seed=$seed, fold=$fold"
                echo "------------------------------------------------"

                uv run python src/train.py \
                    epochs=100 \
                    dataset.name=cifar10 \
                    wandb.enabled=true \
                    wandb.project=soft-momos-plus \
                    wandb.name="soft_momos_plus_c${col}_r${row}_${kval/=/}_${capval/=/}_lam${lam}_s${seed}_f${fold}" \
                    "metrics=[sparsity,l2,gzip,bz2,lzma,bdm]" \
                    quantization=soft_momos_plus \
                    quantization.cols=$col \
                    quantization.rows=$row \
                    quantization.$kval \
                    quantization.$capval \
                    quantization.lam=$lam \
                    quantization.final_k=256 \
                    quantization.force_zero=true \
                    seed=$seed \
                    fold=$fold
            done
        done
    done
done
