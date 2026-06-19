"""
Training script for SNP genotype imputation on Chr22 1000 Genomes Phase 3.
Imports the STICI model and training pipeline from STICI_V1.1.py.

Usage:
    python train_snp.py
    python train_snp.py --save-dir ./my_results --epochs 500 --which-chunk 1
"""

import argparse
import importlib.util
import os
import sys

# STICI_V1.1.py has dots in the filename so standard import does not work
_spec = importlib.util.spec_from_file_location('STICI', 'STICI_V1.1.py')
_mod  = importlib.util.module_from_spec(_spec)
sys.modules['STICI'] = _mod
_spec.loader.exec_module(_mod)

from STICI import train_the_model

ROOT_DIR = '/mnt/storage4/rallendes/STICI'
DATA_DIR = f'{ROOT_DIR}/data'
SNP_VCF  = f'{DATA_DIR}/ALL.chr22.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz'


def parse_args():
    parser = argparse.ArgumentParser(description='STICI SNP training — Chr22 1000GP3')
    parser.add_argument('--ref',               default=SNP_VCF,                        help='Reference VCF path')
    parser.add_argument('--save-dir',          required=True,                              help='Output directory (e.g. training_results/snp/run_name)')
    parser.add_argument('--epochs',            type=int,   default=1000)
    parser.add_argument('--which-chunk',       type=int,   default=-1,     help='Train a single chunk (1-indexed); -1 = all')
    parser.add_argument('--batch-size-per-gpu',type=int,   default=4)
    parser.add_argument('--lr',                type=float, default=0.002)
    parser.add_argument('--embed-dim',         type=int,   default=128)
    parser.add_argument('--na-heads',          type=int,   default=16)
    parser.add_argument('--cs',                type=int,   default=2048,   help='Chunk size (SNPs)')
    parser.add_argument('--co',                type=int,   default=64,     help='Chunk overlap (SNPs)')
    parser.add_argument('--sites-per-model',   type=int,   default=10240)
    parser.add_argument('--min-mr',            type=float, default=0.85,   help='Min masking rate')
    parser.add_argument('--max-mr',            type=float, default=0.95,   help='Max masking rate')
    parser.add_argument('--restart-training',  action='store_true',        help='Clear save-dir and restart')
    parser.add_argument('--mixed-precision',   action='store_true',        help='Enable bfloat16 mixed precision')
    parser.add_argument('--verbose',           type=int,   default=1)
    return parser.parse_args()


def main():
    args = parse_args()

    # Map to the attribute names expected by train_the_model
    args.tihp            = True    # phased SNP data
    args.use_r2          = True
    args.val_n_batches   = 8
    args.random_seed     = 2022
    args.ref_vac         = False
    args.ref_sep         = None
    args.ref_file_format = 'infer'
    args.ref_fcai        = False
    args.ref_comment     = '##'

    # argparse uses hyphens, but train_the_model accesses underscored attrs
    args.save_dir          = args.save_dir
    args.batch_size_per_gpu = args.batch_size_per_gpu
    args.which_chunk       = args.which_chunk
    args.sites_per_model   = args.sites_per_model
    args.embed_dim         = args.embed_dim
    args.na_heads          = args.na_heads
    args.min_mr            = args.min_mr
    args.max_mr            = args.max_mr
    args.restart_training      = args.restart_training
    args.use_mixed_precision   = args.mixed_precision

    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, 'command'), 'w') as f:
        f.write(' '.join(sys.argv) + '\n')

    train_the_model(args)


if __name__ == '__main__':
    main()
