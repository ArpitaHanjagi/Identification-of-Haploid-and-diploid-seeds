# Haploid vs diploid maize seed identification: ResNet18 + quantum CNN

Classifies maize seed images as **haploid** or **diploid** with a ResNet18 fine-tuned end to end
together with a small **quantum convolutional neural network (QCNN)** head: 8 simulated qubits and
24 trainable circuit parameters. A classical head of the same size is trained under identical
settings, so the two heads can be compared fairly.

## Results

The test set has 450 images (184 haploid, 266 diploid) and is used once, at the end. Haploid is the
positive class, and the threshold is 0.5.

| Model | Accuracy | Sensitivity | Specificity | F1 | AUC | MCC |
|---|---|---|---|---|---|---|
| ResNet18 + QCNN, final (seed 0, TTA) | 93.33% | 94.57% | 92.48% | 92.06% | 98.84% | 0.864 |
| ResNet18 + classical head, final (ensemble of 3 seeds, TTA) | 94.22% | 94.57% | 93.98% | 93.05% | 98.90% | 0.881 |
| Altuntaş et al. 2019, VGG-19 (reported, different split) | 94.22% | 94.58% | 93.97% | 93.07% | — | — |

Test accuracy averaged over 3 seeds (mean ± sample std):

| Head | No TTA | TTA |
|---|---|---|
| QCNN | 93.63 ± 0.34% | 93.48 ± 0.13% |
| Classical | 93.48 ± 0.51% | 94.07 ± 0.13% |

**What the results show**

- **QCNN vs classical head:** no significant difference. The two final models disagree on only 6 of the
  450 test images (QCNN right on 1, classical right on 5; exact McNemar p = 0.22). Seed-for-seed
  comparisons are also not significant (p = 0.45 to 1.0).
- **Comparison with Altuntaş et al.:** comparable. The bootstrap 95% interval for the QCNN's accuracy
  is 90.9–95.6%, which includes their 94.22%. Their test split is also different, so a sub-1-point
  difference either way should not be read as better or worse.
- **Parameter count:** no advantage. The QCNN head has 26 trainable parameters (24 in the circuit, 2 in
  the readout), and the classical head has 18. Both sit on the same ResNet18 (11.2 M parameters)
  and the same 512 → 8 layer (4,104 parameters), which do almost all of the work.

In short, a compact 8-qubit QCNN can replace a small classical classifier head with no measurable loss
of accuracy, but it brings no accuracy or efficiency gain on this task. A later 5-fold
cross-validation on the non-test images agrees (see [Cross-validation](#cross-validation-follow-up-experiment)).
Ideas for further experiments are in [FUTURE_SCOPE.md](FUTURE_SCOPE.md).

## Dataset

Altuntaş, Cömert & Kocamaz (2019), *Computers and Electronics in Agriculture* 163, 104874,
doi:10.1016/j.compag.2019.104874. It has 1,230 haploid and 1,770 diploid embryo-up images on a white
background, labelled by the **R1-nj anthocyanin marker**. The labels are therefore *putative* ploidy
and were not confirmed in the field (see Sabadin et al., 2021).

The dataset is not included in this repository. Put it in a folder with `haploid/` and `diploid/`
subfolders of `.jpg` files, and either set the `HAPLOID_DATA_DIR` environment variable to that
folder or change the default in [common.py](common.py).

## Method

- **Split:** stratified 70/15/15 train/validation/test with seed 42: 2,100 / 450 / 450 images.
  [01_make_splits.py](01_make_splits.py) saves the assignment to `results/splits.csv`.
- **Preprocessing:** each image is padded to a square with white and resized to 224×224. Padding is
  used instead of centre-cropping, which could cut off the embryo, where the R1-nj marker is.
- **Augmentation (training only):** rotation up to ±180°, horizontal and vertical flips, a mild
  resized crop (scale 0.85–1) and ±15% brightness and contrast. Hue is left unchanged because the
  marker is a colour.
- **Model ([04_finetune.py](04_finetune.py)):** ImageNet-pretrained ResNet18 → dropout 0.2 →
  linear 512 → 8, then one of two heads:
  - **QCNN:** sigmoid × π gives 8 rotation angles → `AngleEmbedding` (RY) on 8 qubits → three
    convolution + pooling stages (8 → 4 → 2 → 1 qubits, weights shared within a stage) → ⟨Z⟩ of the
    last qubit → 2 logits. It is simulated with PennyLane `default.qubit`, following Cong et al.
    (2019) and Hur et al. (2022). The circuit is defined in [common.py](common.py).
  - **Classical:** sigmoid → linear 8 → 2.
- **Training:** AdamW (weight decay 1e-4) for 12 epochs, batch 32, one-cycle learning-rate schedule.
  Peak learning rates are 3e-4 for the backbone, 1e-3 for the 512 → 8 layer, and 5e-2 for the QCNN
  head (1e-3 for the classical head). The loss is class-weighted cross-entropy with label smoothing
  0.05. The checkpoint with the best validation accuracy is kept. Each head is trained with seeds
  0, 1 and 2.
- **Validation-only decisions ([06_report.py](06_report.py)):** for each head, the following are
  chosen on validation data only, then applied to the test set once:
  - whether to use test-time augmentation (TTA): the mean over the 8 rotations and flips of the
    square image ([tta.py](tta.py));
  - which seed to use (ties go to the lower seed), or whether to average all 3 seeds instead;
  - the decision threshold. It came out at 0.5 for both heads.
- **Statistics:** test accuracy averaged over seeds, a bootstrap 95% interval (10,000 resamples) for
  each final model, and exact McNemar tests on paired QCNN vs classical predictions.

## How to run

Requires Python 3.12. The reported results used an NVIDIA GeForce GTX 1650. Without a GPU, the same
code runs on the CPU, more slowly.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

.\.venv\Scripts\python.exe 01_make_splits.py                       # split; test set = 184 haploid / 266 diploid
.\.venv\Scripts\python.exe check_quantum_circuit.py                # sanity check of the circuit design
.\.venv\Scripts\python.exe 04_finetune.py --head qcnn --seed 0 --epochs 1 --smoke   # quick end-to-end test on a tiny subset

foreach ($h in "qcnn", "classical") { foreach ($s in 0, 1, 2) {
    .\.venv\Scripts\python.exe -u 04_finetune.py --head $h --seed $s --epochs 12
} }
.\.venv\Scripts\python.exe 05_evaluate.py                          # plain + TTA probabilities for val and test
.\.venv\Scripts\python.exe 06_report.py                            # final table, decisions, statistics, figures
```

To classify photos with the trained models (each head's final configuration, with TTA):

```powershell
.\.venv\Scripts\python.exe predict.py photo1.jpg photo2.jpg           # or a folder of photos
.\.venv\Scripts\python.exe predict.py --from-test 10                  # random test photos, with true labels
.\.venv\Scripts\python.exe predict.py --from-test 450                 # whole test set: 420 of 450 (93.3%)
.\.venv\Scripts\python.exe predict.py --from-test 10 --head classical
```

- **Run time:** on the GTX 1650, the 6 training runs take about 25 minutes in total (13 s per epoch
  for the classical head, 22 s for the QCNN).
- **Interruptions:** each run saves a checkpoint every epoch and resumes if interrupted.
- **Devices:** `04_finetune.py` picks CUDA, Apple MPS or CPU automatically (`--device` overrides it).
  The backbone runs on the GPU. The head always runs on the CPU, because the quantum simulator works
  in float64/complex128. Other GPUs and the workstation: see
  [Training on another PC or a workstation](#training-on-another-pc-or-a-workstation).
- **Windows Smart App Control:** if it is on, it blocks PyTorch's DLLs (`WinError 4551 ... c10.dll`).
  Turn it off, or run the project inside WSL2.
- **No local GPU:** [kaggle_train.ipynb](kaggle_train.ipynb) runs the whole pipeline on a free Kaggle
  GPU.

## Using the trained models on another PC

No GPU is needed: `predict.py --device cpu` classifies 10 photos (with TTA) in about 12 seconds.

1. **Get the code:** `git clone https://github.com/ArpitaHanjagi/Identification-of-Haploid-and-diploid-seeds.git`.
   The repository is private, so you need access. Alternatively, copy the project folder without
   `.venv` (a virtual environment doesn't work after being moved).
2. **Get the trained models** from release `v1.0` and put each one where the scripts expect it:

   ```powershell
   gh release download v1.0 --repo ArpitaHanjagi/Identification-of-Haploid-and-diploid-seeds --pattern "*.pt"
   foreach ($r in "qcnn_seed0","qcnn_seed1","qcnn_seed2","classical_seed0","classical_seed1","classical_seed2") {
       New-Item -ItemType Directory -Force "results\finetune\$r" | Out-Null
       Move-Item "$r.pt" "results\finetune\$r\best_model.pt"
   }
   ```

   The final QCNN model needs only `qcnn_seed0.pt`; the final classical model is the average of all
   three classical seeds.
3. **Install Python 3.12 and the packages:**

   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt          # NVIDIA GPU on Windows
   ```

   Without an NVIDIA GPU, install the smaller CPU build of PyTorch instead:

   ```powershell
   .\.venv\Scripts\python.exe -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
   .\.venv\Scripts\python.exe -m pip install pennylane==0.45.1 scikit-learn==1.9.1 scipy==1.18.1 pandas==3.0.6 numpy==2.5.2 matplotlib==3.11.2 pillow==12.3.0
   ```

4. **No ImageNet download is needed** to classify photos: `predict.py` builds the model from the
   trained weights alone. (Training does need the ImageNet weights; see `check_setup.py --download`.)
5. **The dataset** is only needed for `--from-test` and for retraining. Copy it, then point the code
   at it: `$env:HAPLOID_DATA_DIR = "D:\path\to\haploid_and_diploid_maize_seeds_dataset"`.
   `results/splits.csv` stores the original PC's paths; `predict.py` finds each file by name under
   that folder instead.
6. **Run it:** `.\.venv\Scripts\python.exe predict.py --from-test 10` or `predict.py photo.jpg`.

## Training on another PC or a workstation

The main target is a workstation with an NVIDIA RTX Blackwell GPU, but the same steps work on any
PC with an NVIDIA GPU, on an Apple Silicon Mac, and (slowly) on a CPU. The code picks the device
and its speed settings itself.

**1. Code and Python 3.12 packages.** Get the code as in step 1 above, then install the packages
that fit the GPU:

| GPU | Install |
|---|---|
| NVIDIA RTX Blackwell (RTX PRO 6000/5000/4000 Blackwell, RTX 50 series) | `requirements-cu130.txt` (CUDA 13.0 build; NVIDIA driver 580 or newer) |
| Other NVIDIA GPUs from the GTX 16 / RTX 20 series on | either file; `requirements-cu130.txt` needs driver 580 or newer, `requirements.txt` works with older drivers |
| NVIDIA GTX 10 series, Titan V and older (Pascal, Volta) | `requirements.txt` (CUDA 12.6 build; the CUDA 13.0 build has no code for them) |
| Apple Silicon Mac | `pip install torch==2.14.1 torchvision==0.29.1`, then the other packages as in step 3 above |
| No GPU | the CPU build, as in step 3 above |

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-cu130.txt     # Blackwell workstation
```

The CUDA 12.6 build in `requirements.txt` has no code for Blackwell GPUs, so it cannot use them.
On Linux, use `python3.12 -m venv .venv` and `.venv/bin/python` instead.

**2. Dataset and split.** Copy the dataset, point the code at it and re-make the split, because
`results/splits.csv` holds this laptop's paths. The split comes out the same (same file names,
same seed). Don't commit the re-made `splits.csv`; only its paths differ.

```powershell
$env:HAPLOID_DATA_DIR = "D:\path\to\haploid_and_diploid_maize_seeds_dataset"
.\.venv\Scripts\python.exe 01_make_splits.py
```

**3. Check the setup.** [check_setup.py](check_setup.py) checks the GPU and PyTorch build, the
dataset, the split (test set 184 haploid + 266 diploid), the cross-validation folds against
`results/cv/folds.csv`, and the QCNN simulator. `--download` also fetches the ImageNet weights of
every backbone (about 350 MB; an offline PC can copy `%USERPROFILE%\.cache\torch\hub\checkpoints`
from another PC instead). It must end with `Ready to train.`

```powershell
.\.venv\Scripts\python.exe check_setup.py --download
```

**4. Measure the speed** of each backbone and image size you plan to use, before the long runs.
A smoke run with 1,024 images per split prints the seconds per epoch and the GPU memory used. Smoke
runs write only to `results/cv_smoke/` and `results/finetune_smoke/`:

```powershell
$env:PYTHONWARNINGS = "ignore"
.\.venv\Scripts\python.exe -u 04_finetune.py --head qcnn --fold 0 --epochs 2 --smoke --smoke-images 1024 --backbone resnet50 --img-size 384
```

**5. Train.** Cross-validation for one backbone and image size, both heads; then the report:

```powershell
foreach ($f in 0..4) { foreach ($h in "qcnn", "classical") {
    .\.venv\Scripts\python.exe -u 04_finetune.py --head $h --fold $f --epochs 12 --backbone resnet50 --img-size 384
} }
.\.venv\Scripts\python.exe 07_cv_report.py
```

Runs of another backbone or size go to their own folder (`results/cv/resnet50_384/`), so they never
mix with the ResNet18 results. To repeat runs that already exist from another PC, such as the
ResNet18 baseline, add a tag: `--tag ws` writes to `results/cv/resnet18_224_ws/`. The report then
compares the two and marks the comparison `[different GPUs]`.

**Options of [04_finetune.py](04_finetune.py)**

| Option | Default | What it does |
|---|---|---|
| `--backbone` | `resnet18` | `resnet18`, `resnet50`, `convnext_tiny`, `efficientnet_b0` or `efficientnet_v2_s` (ImageNet-pretrained) |
| `--img-size` | `224` | input size in pixels. The photos' longest side is 300–718 px (median 534), so 320–448 adds real detail; above about 448, most photos are only upscaled |
| `--epochs`, `--batch-size`, `--lr-backbone` | `12`, `32`, `3e-4` | training settings; change them for both heads alike |
| `--fold`, `--seed`, `--tag` | none, `0`, none | cross-validation fold; random seed; label for the results folder |
| `--amp` | `auto` | bf16 mixed precision for the backbone: on for NVIDIA RTX 30 series and newer (including Blackwell), off on older GPUs, Apple and CPU |
| `--workers` | `auto` | processes that augment training images in parallel: as many as `--threads` on a GPU PC, none on a CPU |
| `--threads` | `auto` | CPU threads of the main process, which also runs the QCNN simulator: half the logical CPUs, at most 8 |
| `--device` | `auto` | `cuda`, `mps` or `cpu` |

`--amp`, `--workers` and `--threads` change only the speed. Each run records what it used in
`run_config.json`. A run resumes from its checkpoint only with the same training settings.

**Time**

| Hardware | ResNet18, 224 px: 6 fixed-split runs | ResNet18, 224 px: 5-fold CV, both heads, 12 epochs |
|---|---|---|
| NVIDIA GTX 1650 laptop (measured) | about 25 min | about 35 min |
| RTX Blackwell workstation | measure with step 4 | measure with step 4 |
| 4-core laptop CPU, no GPU (estimated) | about 4 hours | about 6 hours |

On the GTX 1650, half precision is not usable (fp16 gave NaN at batch size 64, and ran 4× slower),
and the 4 GB of memory limits bigger backbones and images to batch size 16 at most.

- **Keep the PC plugged in.** Windows is kept awake automatically; on a Mac, prefix commands with
  `caffeinate -i`. Interrupted runs resume from their last epoch.
- **Different results:** a retrain on different hardware gives results within about a point of the
  original ones. Compare only runs from the same PC, or tag them and compare them as separate variants.
- **Windows Smart App Control** blocks PyTorch's DLLs (`WinError 4551`); turn it off, or use WSL2.

**Final model of a new variant.** Once cross-validation has picked the backbone and size, train it on
the fixed split (3 seeds per head), then evaluate and report it. Its tables and figures go to
`results/finetune/<variant>/`, so the reported results in `results/` are not touched. This scores
the test set, so do it once, at the end:

```powershell
foreach ($h in "qcnn", "classical") { foreach ($s in 0, 1, 2) {
    .\.venv\Scripts\python.exe -u 04_finetune.py --head $h --seed $s --epochs 12 --backbone resnet50 --img-size 384
} }
.\.venv\Scripts\python.exe 05_evaluate.py --variant resnet50_384
.\.venv\Scripts\python.exe 06_report.py --variant resnet50_384
.\.venv\Scripts\python.exe predict.py --variant resnet50_384 --from-test 10
```

## Cross-validation (follow-up experiment)

This tests whether longer training helps and gives a more reliable QCNN vs classical comparison than
one split. It uses only the 2,550 train + validation images, in 5 stratified folds
(`common.cv_folds`, seed 42); the 450 test images are never loaded. Each fold trains on 2,040 images
and holds out 510. Both heads are trained the same way at 30 epochs and at the original 12.

```powershell
$env:PYTHONWARNINGS = "ignore"
foreach ($f in 0..4) { foreach ($e in 30, 12) { foreach ($h in "qcnn", "classical") {
    .\.venv\Scripts\python.exe -u 04_finetune.py --head $h --fold $f --epochs $e
} } }
.\.venv\Scripts\python.exe 07_cv_report.py
```

- **Main number:** the final-epoch model, scored on its held-out fold, with and without TTA. The epoch
  count is fixed in advance, so this estimate is unbiased. The best-epoch model is also reported,
  but it was chosen on the same fold, so its accuracy is optimistic.
- **Comparisons:** the held-out predictions of the 5 folds cover all 2,550 images once, so they are
  pooled for exact McNemar tests (QCNN vs classical, 30 vs 12 epochs, and one backbone or image size
  vs another for the same head), next to fold-by-fold differences.
- **Outputs (`results/cv/`):** `<head>_e<epochs>_fold<k>_seed0/` (`cv_result.json`, `history.csv`,
  `oof.npz`, `run_config.json`), `cv_summary.csv`, `cv_stats.json`, `folds.csv`, and
  `results/figures/cv_curves.png`. Other backbones and image sizes are in subfolders such as
  `results/cv/resnet50_384/`. Finished runs are skipped, so the loop can be restarted.
- **Same folds everywhere:** `04_finetune.py` and `07_cv_report.py` check the folds against
  `results/cv/folds.csv` and stop if they differ, so runs from different PCs stay comparable.

**Results** (GTX 1650, seed 0, final-epoch model, threshold 0.5; mean ± sample std over the 5
folds of 510 images; the pooled accuracy over all 2,550 images is the same number):

| Head | 12 epochs | 12 epochs + TTA | 30 epochs | 30 epochs + TTA |
|---|---|---|---|---|
| QCNN | 95.33 ± 1.07% | 95.80 ± 0.63% | 95.80 ± 1.21% | 96.16 ± 0.80% |
| Classical | 95.29 ± 0.89% | 95.88 ± 1.03% | 95.33 ± 0.70% | 95.73 ± 1.26% |

- **Training longer does not help significantly.** Held-out accuracy levels off at about 95.5% by
  epoch 10, while training accuracy keeps rising to about 98% (`cv_curves.png`). 30 vs 12 epochs:
  classical +0.04 ± 0.51 points (McNemar p = 1.0); QCNN +0.47 ± 0.75 points (p = 0.16).
- **QCNN vs classical: still no significant difference.** At 12 epochs they are equal (+0.04 points,
  p = 1.0). At 30 epochs the QCNN is ahead by 0.47 ± 0.73 points (right on 27 images where the
  classical head is wrong, the reverse on 15; p = 0.09). That is not significant, it is one of 8
  comparisons, and it rests on one seed per fold, so it is at most a hint worth re-checking with
  more seeds.
- **TTA** adds about 0.4–0.6 points to the final-epoch model in every setting.
- **These numbers are not comparable with the test-set results above.** They come from different
  images and from models trained on 2,040 instead of 2,100 images. The test set was not used, so
  the reported test results stand unchanged.

## Outputs (`results/`)

- `final_table.csv`: every run (with and without TTA), the final models and the Altuntaş reference.
- `final_decisions.json`: the choices made on validation for each head.
- `final_stats.json`: seed averages, bootstrap intervals and McNemar tests.
- `figures/final_confusion_qcnn.png`, `final_roc.png`, `final_training_curves.png`.
- `finetune/<head>_seed<n>/`: `best_model.pt` (weights), `history.csv`, `test_result.json`, `probs.npz`,
  and for runs made after November 2026 `run_config.json` (backbone, image size, settings, GPU).
- `finetune/<variant>/`: the same for another backbone or image size, plus its own `final_*` files
  and `figures/` from `06_report.py --variant <variant>`.
- `cv/`: cross-validation runs and summary (see above).
- `splits.csv`: which image is in the train, validation or test set.
- `image_cache_<size>.pt`: the resized images, rebuilt automatically if the split changes.

## Earlier experiment: frozen ResNet50 features

[02_extract_features.py](02_extract_features.py) and [03_train_evaluate.py](03_train_evaluate.py) form
an earlier, faster experiment. They extract frozen ResNet50 features once (clean images plus 4
augmented views of each training image), then train three small heads on them with 5 seeds:

- **linear:** 2048 → 2;
- **classical:** 2048 → 64 → 4 (tanh) → 2;
- **hybrid:** the same bottleneck → a 4-qubit `StronglyEntanglingLayers` circuit → 2.

[check_quantum_circuit.py](check_quantum_circuit.py) shows why the circuit was redesigned. The
original notebook's circuit applied trainable RZ gates right before a PauliZ measurement, so its
weights could not affect the output.
