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
of accuracy, but it brings no accuracy or efficiency gain on this task. Ideas for further experiments
are in [FUTURE_SCOPE.md](FUTURE_SCOPE.md).

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

- **Run time:** on the GTX 1650, the 6 training runs take about 25 minutes in total (13 s per epoch
  for the classical head, 22 s for the QCNN).
- **Interruptions:** each run saves a checkpoint every epoch and resumes if interrupted.
- **Devices:** `04_finetune.py` picks CUDA, Apple MPS or CPU automatically (`--device` overrides it).
  The ResNet18 runs on the GPU. The head always runs on the CPU, because the quantum simulator works
  in float64/complex128.
- **Windows Smart App Control:** if it is on, it blocks PyTorch's DLLs (`WinError 4551 ... c10.dll`).
  Turn it off, or run the project inside WSL2.
- **No local GPU:** [kaggle_train.ipynb](kaggle_train.ipynb) runs the whole pipeline on a free Kaggle
  GPU.

## Outputs (`results/`)

- `final_table.csv`: every run (with and without TTA), the final models and the Altuntaş reference.
- `final_decisions.json`: the choices made on validation for each head.
- `final_stats.json`: seed averages, bootstrap intervals and McNemar tests.
- `figures/final_confusion_qcnn.png`, `final_roc.png`, `final_training_curves.png`.
- `finetune/<head>_seed<n>/`: `best_model.pt` (weights), `history.csv`, `test_result.json`, `probs.npz`.
- `splits.csv`: which image is in the train, validation or test set.

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
