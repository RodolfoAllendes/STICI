#!/bin/bash
#SBATCH --job-name=stici_train
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=500G
#SBATCH --cpus-per-task=15
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

mkdir -p logs

# has to be the same file used for training
REF=./data/beadchip_reference_all_minaf_05_snps_hwe_1e-2_filtered_train.vcf.gz
# STICI saves training and impute results to the same directory, it will read
# the model from this location
SAVE_DIR=./training_results/snp/bench_ep200
# the file to be imputed
TARGET=./data/test_data_beadchip_hwe_filtered.vcf.gz

export SINGULARITYENV_LD_LIBRARY_PATH=/usr/local/cuda-12.8/targets/x86_64-linux/lib

singularity exec --nv \
    -B /usr/local/cuda-12.8:/usr/local/cuda-12.8 \
    stici.sif \
    python STICI_V1.1.py \
    --mode impute \
    --ref "$REF" \
    --save-dir "$SAVE_DIR" \
    --target "$TARGET" \
    --tihp true