# STICI — Working Notes

---

## 1. Chunks, Index Positions, and Inference

### Two levels of chunking

STICI applies chunking at two distinct levels, which are easy to conflate.

**Level 1 — within a single model (inside `STICI.call()`)**

Each model covers `sites_per_model` SNPs (default 10,240). Internally that window is
divided into sub-chunks of `chunk_size` SNPs (default 2,048):

```
chunk_starts = [0, 2048, 4096, 6144, 8192]   # 5 sub-chunks for a 10,240-SNP model
```

Each sub-chunk has its own independent transformer block (`chunk_module`). They run in
parallel, their outputs are concatenated, then passed through two Conv1D layers to
produce per-SNP predictions. `attention_range=64` allows each sub-chunk to attend 64
SNPs across the boundary to its neighbour, reducing edge artefacts.

**Level 2 — between models, at inference (`impute_the_target()`)**

The genome (or reference panel) is divided into non-overlapping windows of
`sites_per_model` SNPs, each with a separately trained model (`w_0.keras`,
`w_1.keras`, …). At inference each model is loaded sequentially and predicts its
window. Results are concatenated with a hard join:

```python
all_preds = []
for w in range(n_chunks):
    model = tf.keras.models.load_model(f"models/w_{w}.keras")
    all_preds.append(model.predict(test_dataset))
all_preds = np.hstack(all_preds)   # hard horizontal concatenation
```

The `co` overlap parameter (default 64 SNPs) gives each model a context buffer beyond
its core window. The model trims those buffer positions from its output:

```python
x = x[:, self.offset_before : self.seq_len - self.offset_after]
```

so `np.hstack` joins clean, non-overlapping slices. There is no blending or learned
junction at boundaries.

---

### Position embeddings are sequential indices, not genomic coordinates

Inside `CatEmbeddings.call()`:

```python
positions = tf.range(start=0, limit=self.n_snps, delta=1)
# → [0, 1, 2, …, 10239]
```

These are **sequential integers**. The model has no knowledge of:
- Physical distance in base pairs between adjacent SNPs
- Whether two adjacent indices span 100 bp or 5 Mb
- Recombination hotspots or LD block boundaries
- Which chromosome the data came from

The position embedding at index `k` learns *"I am the k-th SNP in this reference
panel chunk"* — nothing more. LD correlations are learned implicitly from the data
because the VCF is sorted by genomic coordinate, so index proximity coincidentally
tracks genomic proximity. But the model is entirely blind to the *magnitude* of
physical distances.

---

### What this means for imputation of test samples

The model always receives a tensor of shape `(batch, n_snps, n_alleles)` — the full
chunk length regardless of how many SNPs are actually observed in the test sample.
Unobserved positions are filled with the missing-value code by `DataReader`.

The constraint is not on *how many* SNPs a test sample has, but on *which* SNPs:

- Each test variant is matched to the reference panel by VCF identity
  (chromosome + position + REF + ALT).
- Matched variants are placed in their reference-panel index slot; unmatched positions
  are marked missing.
- SNPs in the test sample that do not appear in the reference panel are discarded —
  there is no index slot for them.

There is also a soft constraint on the observation rate. Training uses random masking
at 85–95% (5–15% of positions visible). At inference with a typical beadchip panel
(~23% of reference positions observed), the model receives more context than it was
trained with, which is generally beneficial. Sparse panels with <5% overlap with the
reference may produce lower quality imputations.

---

### Hard-join boundary weakness

Because adjacent chunk models are trained and applied independently, a haplotype that
spans the boundary between two windows is never seen in full by either model. This can
reduce imputation quality at chunk boundaries. Addressing this would require either
a single model with sparse global attention, or a learned blending layer at the join.

---

## 2. Training and Evaluation Scripts

### Environment

The `stici` conda environment stores GPU-related variables in activate/deactivate
hooks so they persist across sessions:

```
/mnt/storage2/install/miniconda3/envs/stici/etc/conda/activate.d/tf_env.sh
/mnt/storage2/install/miniconda3/envs/stici/etc/conda/deactivate.d/tf_env.sh
```

Key variables set on activation:

```bash
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export LD_LIBRARY_PATH=/mnt/storage2/install/miniconda3/envs/stici/lib:$LD_LIBRARY_PATH
```

Hardware: 2× NVIDIA RTX PRO 6000 Blackwell, ~96 GB VRAM each. Both are used via
`tf.distribute.MirroredStrategy` during training.

---

### `train_snp.py` — CLI training script

Imports STICI via `importlib` (the filename `STICI_V1.1.py` contains dots, which
prevents standard Python import):

```python
_spec = importlib.util.spec_from_file_location('STICI', 'STICI_V1.1.py')
_mod  = importlib.util.module_from_spec(_spec)
sys.modules['STICI'] = _mod
_spec.loader.exec_module(_mod)
from STICI import train_the_model
```

**Key flags:**

| Flag | Default | Notes |
|---|---|---|
| `--ref` | benchmark VCF | Reference panel path |
| `--which-chunk` | `-1` (all) | Train a single chunk (1-indexed) |
| `--epochs` | 1000 | Max training epochs |
| `--batch-size-per-gpu` | 4 | Effective batch = × num GPUs |
| `--mixed-precision` | off | Enables bfloat16 |
| `--restart-training` | off | Wipes save-dir before training |
| `--embed-dim` | 128 | Transformer embedding dimension |
| `--na-heads` | 16 | Number of attention heads |
| `--cs` | 2048 | Sub-chunk size (SNPs) |
| `--co` | 64 | Chunk overlap (SNPs) |
| `--sites-per-model` | 10240 | SNPs per macro-level chunk model |

**Typical quick-test run (single chunk, ~2 hours):**

```bash
python train_snp.py \
  --ref data/STI_benchmark_datasets/beadchip_reference_all_minaf_05_snps_hwe_1e-2_filtered_train.vcf.gz \
  --mixed-precision \
  --which-chunk 1 \
  --epochs 25
```

**Weekend run (all 4 chunks on benchmark, ~2 weeks sequential):**

```bash
python train_snp.py \
  --ref data/STI_benchmark_datasets/beadchip_reference_all_minaf_05_snps_hwe_1e-2_filtered_train.vcf.gz \
  --mixed-precision \
  --epochs 1000
```

---

### Resumption and checkpointing behaviour

`--restart-training` deletes the entire save directory and starts from scratch.

Without it, STICI resumes at the **chunk level** only: it reads `models/chunks_info.json`
and skips any chunk already marked complete. If a chunk was interrupted mid-training,
it restarts that chunk from epoch 1 — there is no within-chunk resume because
`ModelCheckpoint` is currently commented out in `create_callbacks()`.

---

### Callbacks

Three callbacks are configured in `create_callbacks()`:

- **`ReduceLROnPlateau`** — halves LR after 3 epochs without improvement; floor `1e-7`
- **`EarlyStopping`** — stops training if `val_loss` does not improve for 35 epochs;
  restores best weights automatically
- **`ModelCheckpoint`** — saves best checkpoint; currently **commented out** (line 569)

EarlyStopping with `patience=35` means a 1,000-epoch run can terminate early. With
only 25 epochs requested this will not trigger.

---

### Mixed precision

Enabled in `train_the_model()` with `--mixed-precision`. Uses `bfloat16` (not
`float16`) because BF16 has the same dynamic range as FP32 and is safer for training.

In practice VRAM usage stays near 95 GB even with BF16 because Keras maintains FP32
master weights and FP32 optimizer states (LAMB momentum + variance = 2× model params).
The BF16 benefit is faster Tensor Core computation, not reduced memory.

---

### Observed training performance (benchmark, 2× RTX PRO 6000 Blackwell)

| Metric | Value |
|---|---|
| Steps per epoch | 593 |
| Step time | ~467–500 ms |
| Epoch time | ~5 min |
| VRAM per GPU | ~95.5 GB / 97.9 GB |
| GPU utilisation | 50–76% (bursty) |
| Power draw | ~170–200 W / 300 W max |

---

### `eval_snp.py` — evaluation script

Loads each trained chunk model, runs imputation on the test samples, and computes
per-variant Pearson r² against the ground-truth VCF.

```bash
python eval_snp.py --which-chunk 1
```

**Test dataset (benchmark):**

| File | Samples | Variants | Role |
|---|---|---|---|
| `beadchip_reference_all_minaf_05_snps_hwe_1e-2_filtered_train.vcf.gz` | 2,404 | 31,143 | Reference panel (training) |
| `test_data_beadchip_hwe_filtered.vcf.gz` | 100 | 7,336 | Sparse target (observed SNPs only) |
| `test_true_data_beadchip_hwe_filtered.vcf.gz` | 100 | 31,143 | Ground truth (all positions) |

The model imputes the 23,807 positions that are missing in the test data. Quality is
reported as mean r², median r², and fraction of variants with r² ≥ 0.8 (the standard
well-imputed threshold used by Minimac4 and BEAGLE).

---

### Full Chr22 VCF — memory issue

The full Chr22 VCF (`ALL.chr22.phase3_shapeit2_mvncall_integrated_v5b.20130502...`)
has 1,103,547 variants × 2,504 samples. `DataReader.__read_csv()` uses
`datatable.fread()` followed by `.to_pandas()`. The pandas conversion materialises
every genotype string (`"0|1"`, `"1|0"`) as a Python object (~70 bytes each):

```
1.1M rows × 2,513 columns × 70 bytes ≈ 196 GB peak
```

This exceeds the 188 GB system RAM. The fix is to pre-convert the VCF to integer
genotypes in a binary format (zarr, HDF5, or numpy) before feeding it to the
DataReader — a necessary step before training on the full chromosome.

---

## 3. Expanding STICI Toward a Foundation Model

### What STICI currently is

STICI is a **collection of position-specific models**. Each chunk model trains a
separate set of weights for a fixed genomic region of a fixed reference panel.
Concretely:

- The `CatEmbeddings` position embedding has one learned 128-dimensional vector per
  SNP index (0 to 10,239). That vector encodes the local LD context of that specific
  variant in that specific reference panel.
- There is no weight sharing across chunks, chromosomes, or populations.
- A model trained on Chr22 positions 0–10,239 cannot be applied to any other region.
- Whole-genome coverage would require ~2,200 separate models (22 autosomes × ~100
  chunks each).

This is the correct design for the intended use case — production genotype imputation
from a beadchip panel — because you *want* the model to memorise the specific
haplotype structure of the reference at nucleotide resolution.

---

### The architectural gap

The single change that prevents transfer is the positional encoding:

```python
# Current: sequential index — non-transferable
positions = tf.range(start=0, limit=self.n_snps, delta=1)
embedding = self.position_embedding(positions)   # lookup table, 1 row per index
```

Index 5 means *"the 6th SNP in this reference panel chunk"*. It has no meaning
elsewhere.

A transferable encoding would represent physical or genetic distance instead:

```python
# Foundation model direction: distance-based — transfers anywhere
bp_distances = genomic_positions[1:] - genomic_positions[:-1]
# or genetic distances in cM, which better reflect recombination probability
```

With distance-based encoding the model learns *"SNPs 1.2 kb apart with r²=0.9 form a
haplotype block"* — a biological rule that holds across all chromosomes and populations.

---

### Required architectural changes (in order of difficulty)

1. **Relative/physical positional encoding**
   Replace the absolute index lookup table with an encoding of base-pair (or cM)
   distance between adjacent SNPs. This is the foundational change that unlocks
   everything below.

2. **Multi-chromosome pre-training**
   Once positions are relative, a single model can train on all autosomes jointly
   instead of one model per region. The DataReader would need to stream chromosomes
   rather than loading one at a time.

3. **SNP tokenisation by genomic context**
   Represent each SNP by learnable features of its local context — flanking k-mer
   sequence, population MAF, local recombination rate — rather than by index. This
   enables transfer across panels and species that share no variant positions.

4. **Pre-train + fine-tune paradigm**
   Pre-train on a large, diverse reference (1000G + HGDP + gnomAD across all
   chromosomes), then fine-tune a lightweight head for a specific target panel.
   The transformer backbone weights would be frozen or lightly adapted.

---

### Hard-join boundary as a secondary target

At inference, adjacent chunk models are concatenated with `np.hstack` — a hard join
with no blending. A haplotype spanning the boundary between two models is never seen
in full by either. A natural improvement (independent of the foundation model
direction) is a learned boundary blending layer, or replacing the per-region models
with a single sparse-attention model that maintains global context.

---

### Compute plan

**Track 1 — Benchmark current STICI at scale**

1. Pre-convert full Chr22 VCF to zarr/HDF5 to fix the 196 GB parsing OOM.
2. Write SLURM job array for the H200 cluster (4 nodes × 4 H200, 141 GB each):
   108 chunks submitted as parallel jobs → full Chr22 in ~1 week.
3. Run evaluation against ground truth; compare r² to Minimac4/BEAGLE baseline.

**Track 2 — Foundation model expansion**

1. Replace index-based position embedding with bp/cM distance encoding.
2. Validate that a single model trained on multiple Chr22 chunks achieves parity
   with per-chunk models (proves transfer within chromosome).
3. Extend to multi-chromosome training.
4. Add SNP context tokenisation for cross-panel and cross-species transfer.

Track 1 can proceed immediately. Track 2 begins after the distance encoding change is
validated on the benchmark.
