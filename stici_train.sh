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

export SINGULARITYENV_LD_LIBRARY_PATH=/usr/local/cuda-12.8/targets/x86_64-linux/lib

# singularity exec --nv \
#     -B /usr/local/cuda-12.8:/usr/local/cuda-12.8 \
#     stici.sif \
#     python - <<'EOF'
# import ctypes
# import tensorflow as tf

# ctypes.CDLL("libnvrtc.so")
# print("NVRTC OK")
# print("TF:", tf.__version__)
# print("GPUs:", tf.config.list_physical_devices('GPU'))
# EOF

singularity exec --nv \
    -B /usr/local/cuda-12.8:/usr/local/cuda-12.8 \
    stici.sif \
    python STICI_V1.1.py \
    --mode train \
    --ref ./data/beadchip_reference_all_minaf_05_snps_hwe_1e-2_filtered_train.vcf.gz \
    --save-dir ./training_results/snp/bench_ep200 \
    --epochs 200 \
    --tihp true \
    --batch-size-per-gpu 4 \
    --sites-per-model 10240 \
    --co 64 \
    --restart-training true