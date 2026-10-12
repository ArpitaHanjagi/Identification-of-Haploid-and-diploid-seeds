# Future scope

Experiments that could improve or extend the current results. None of these are part of the
reported results (see [README.md](README.md)). The time estimates assume the same GTX 1650, which
measured 13 s per epoch for the classical head and 22 s for the QCNN at 2,100 training images.

## Starting point

- **Final models:** the QCNN reaches 93.33% test accuracy (93.63 ± 0.34% over 3 seeds without TTA),
  and the classical head 94.22% (3-seed ensemble with TTA). Altuntaş et al. (2019) report 94.22% on a
  different split.
- **QCNN vs classical:** no significant difference (exact McNemar p = 0.22 for the final models).
- **No parameter advantage:** the QCNN head has 26 parameters and the classical head 18.

## Ground rules for any new experiment

1. **Treat both heads the same.** Any change (epochs, resolution, backbone, tuning budget) is applied
   to the QCNN and the classical head alike. Tuning one head harder than the other makes the
   comparison unfair.
2. **Protect the test set.** It has already been used twice (an earlier run on another PC, and the
   reported run). Make every choice on validation data or with cross-validation, and score the
   test set once at the end. Do not repeat runs until a test number looks better.
3. **Report averages and intervals.** Give the mean ± std over seeds or folds and a confidence
   interval, not only the best run. On 450 test images the 95% interval is about ±2 percentage
   points, so a difference under about 2 points is not evidence that one model is better.

## A. Improvements that should help both heads

| Idea | Why | Extra GPU time (est.) |
|---|---|---|
| **Train longer (25–30 epochs instead of 12)** | The models look under-trained. The best epoch was 10–12 of 12 in 5 of the 6 runs, and the final training accuracy was only 96–97%. | about 1.5 h for 2 heads × 5 folds at 30 epochs |
| **5-fold cross-validation** | More reliable estimates and comparisons than a single split; every image is used for validation once. Keep the 450 test images fixed and fold the remaining 2,550. | included above |
| **Higher input resolution (320 or 384)** | The R1-nj marker is a small coloured area on the embryo, and detail may be lost at 224. | about 2× the time at 320. May need mixed precision to fit in 4 GB of GPU memory. |
| **Stronger backbone (ResNet50, EfficientNet, ConvNeXt-Tiny)** | The backbone does almost all of the work, so it is the most likely way to raise accuracy. | about 2–4× the time. May need mixed precision or a smaller batch. |
| **Equal-budget hyperparameter search** | Search learning rates and weight decay on validation with the same number of trials for each head. | depends on the budget |

Even with these, beating Altuntaş et al. could only be claimed as "higher accuracy", not
"significantly better", unless the gain is well above the roughly ±2-point interval. Their split is
also different.

## B. Experiments where the quantum head could show something new

| Idea | What it tests | Extra GPU time (est.) |
|---|---|---|
| **Low-data regime:** train on 10%, 25% and 50% of the training images, with 5 repeats each, for both heads | Whether the QCNN degrades more gracefully than the classical head with few labels. Some studies suggest quantum heads do; this would be a genuine finding either way. | about 1.5–2 h |
| **Noise robustness:** evaluate the trained QCNN under simulated hardware noise (depolarising and readout errors) | Whether the accuracy holds on realistic near-term quantum hardware. | under 1 h; inference only |
| **Real quantum hardware:** run test-set inference on an 8-qubit device (e.g. the IBM Quantum free tier) | Whether the model works outside simulation. Would be a novel result for this application. | limited by free QPU minutes; check the quota first |
| **Larger or deeper circuits:** more qubits (via 16 angles) or repeated convolution layers | Whether circuit capacity limits accuracy. Compare against a classical head of matched size. | about 1–2 h per variant |

A careful negative result ("no advantage under a fair comparison") is also publishable, particularly
at quantum machine learning venues.

## C. Literature and writing

- **Check later papers on the same dataset.** Several papers after Altuntaş et al. (2019) used these
  3,000 images, for example work based on pre-trained CNNs, intuitionistic fuzzy sets and GLCM
  features. Some may report more than 94.22%, so compare against the best published result, not only
  Altuntaş et al.
- **State the comparison caveats in the paper.** The test splits differ (450 images here, 900
  implied by their reported numbers), and the labels are putative ploidy from the R1-nj marker.

## Suggested order

1. **A: 30 epochs plus 5-fold CV for both heads** (about 1.5 h of GPU time).
2. **B: the low-data experiment on the same folds** (about 1.5–2 h).
3. **B: noise robustness**, then real hardware if the simulated-noise result holds up.

Together, steps 1 and 2 take about half a day: roughly 1 hour of code changes and 3–3.5 hours of
unattended training.
