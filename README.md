# Haploid vs diploid maize seed identification: ResNet50 + quantum head

Classifies maize seed images as **haploid** or **diploid** using frozen ResNet50
features and three classifier heads:

| Head | What it is |
|---|---|
| `linear` | 2048 → 2, a standard linear probe |
| `classical` | 2048 → 64 → 4 (tanh) → 2 |
| `hybrid` | 2048 → 64 → 4 (tanh × π) → 4-qubit PennyLane circuit → 2 |

`classical` and `hybrid` share the same bottleneck, so any difference between them
comes from the quantum circuit alone.

## Dataset

Altuntaş, Cömert & Kocamaz (2019), *Computers and Electronics in Agriculture* 163, 104874,
doi:10.1016/j.compag.2019.104874. It has 1,230 haploid and 1,770 diploid embryo-up images
on a white background, labelled by the **R1-nj anthocyanin marker**. The labels are
therefore *putative* ploidy and were not confirmed in the field (see Sabadin et al., 2021).

Expected location: `C:\Users\arpit\Downloads\haploid_and_diploid_maize_seeds_dataset\{haploid,diploid}\*.jpg`
(change `DATA_DIR` in `common.py` to use another path).

## How to run

```powershell
.\.venv\Scripts\python.exe 01_make_splits.py        # stratified 70/15/15 split, seed 42
.\.venv\Scripts\python.exe 02_extract_features.py   # ResNet50 features (clean + 4 augmented train views)
.\.venv\Scripts\python.exe 03_train_evaluate.py     # 3 heads x 5 seeds, metrics + figures
.\.venv\Scripts\python.exe check_quantum_circuit.py # shows the original circuit's weights had no effect
```

Everything runs on a CPU. Feature extraction takes about 25 minutes on an i5-1155G7 and
saves after each stage, so a rerun resumes where it stopped. Training all 15 models takes
about 20 minutes.

## Method details

- **Preprocessing:** each image is padded to a square with white, resized to 224×224 and
  normalised with ImageNet statistics. Padding is used instead of centre-cropping, which
  could cut off the embryo, where the R1-nj marker is.
- **Augmentation (training only):** rotation up to ±180°, horizontal and vertical flips, a
  mild resized crop, and ±15% brightness and contrast. Hue is left unchanged because the
  marker is a colour.
- **Training:** AdamW (lr 1e-3, weight decay 1e-4), batch 64, class-weighted cross-entropy,
  up to 60 epochs with early stopping on validation loss (patience 10).
- **Evaluation:** the test set (450 images) is used once, at the end. Haploid is the positive
  class, and results are reported as mean ± std over 5 seeds.
- **Quantum circuit:** `AngleEmbedding` (RY) followed by 2 `StronglyEntanglingLayers`
  (24 trainable angles), with PauliZ expectation readout, simulated with `default.qubit`.

## Outputs (`results/`)

- `table_results.csv`: main results table (mean ± std)
- `metrics_per_seed.csv`: every run
- `figures/`: confusion matrices, ROC curves, training curves
- `models/`: trained head weights
- `splits.csv`: which image went to train, validation or test
