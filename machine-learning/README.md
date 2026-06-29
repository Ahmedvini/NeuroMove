# EEG Motor Imagery Classification — BCI Graduation Project

> **Egypt-Japan University of Science and Technology (E-JUST)**
> Brain-Computer Interface System for Assistive Motion Control & Rehabilitation

---

## Overview

This project implements a complete EEG signal processing and machine learning pipeline for **motor imagery (MI) classification**, designed for use in a brain-computer interface system. The pipeline covers subject selection, preprocessing, feature extraction, data augmentation, model training, and evaluation across two benchmark datasets.

---

## Datasets

### PhysioNet EEG Motor Movement/Imagery Dataset (Primary)

| Property | Details |
|---|---|
| Source | PhysioNet (Version 1.0.0) |
| Format | EDF |
| Subjects | 109 total; **19 selected** via 95% confidence interval filtering |
| Channels | 64 EEG electrodes (BCI2000 system) |
| Sampling Rate | 160 Hz |
| Files | 1,526 EDF files; 14 runs per subject |
| Epochs per run | ~24 motor imagery epochs, avg. length 4.15 s |
| Tasks | Left fist (T1), Right fist (T2), Rest (T0) |
| Total Epochs | 3,240 (after subject selection and preprocessing) |

### BCI Competition IV Dataset 2a (Secondary / Benchmarking)

| Property | Details |
|---|---|
| Subjects | 2 (intensive, subject-specific evaluation) |
| Channels | 22 EEG channels |
| Sampling Rate | 250 Hz |
| Task | Binary motor imagery (left hand vs. right hand) |

---

## Subject Selection

Rather than using all 109 subjects, a principled statistical criterion was applied. A preliminary LDA+CSP classification accuracy was computed per subject, and the mean and standard deviation across all subjects were used to define a **95% confidence interval via the t-distribution**. Only subjects whose accuracy fell within this interval were retained, resulting in **19 representative subjects**. This removes outliers (both unusually easy and unusually difficult subjects) to ensure fair and generalizable evaluation.

---

## Preprocessing Pipeline

### 1. Initial Data Quality Assessment
Channels were flagged if they exceeded three standard deviations from the mean channel variance, showed signal clipping (> ±150 µV), or had near-zero variance (dead channels). Only physiologically plausible signals were retained.

### 2. Signal Filtering
- **Notch Filter** at 50 Hz and 60 Hz to suppress power line interference.
- **Bandpass Filter** at 8–30 Hz to isolate the **alpha (8–12 Hz)** and **beta (13–30 Hz)** frequency bands, which correspond to motor imagery-related ERD/ERS phenomena.

### 3. Motor Cortex Channel Selection
21 channels over the motor cortex were selected according to the **10–20 international system**:
`FC5, FC3, FC1, FCz, FC2, FC4, FC6, C5, C3, C1, Cz, C2, C4, C6, CP5, CP3, CP1, CPz, CP2, CP4, CP6`

This targets primary motor cortex, premotor, and supplementary motor areas while discarding unrelated cortical activity.

### 4. Artifact Removal — ICA
**Independent Component Analysis (ICA)** was applied with 15 components, separately to each data split (train / validation / test), to prevent leakage of artifact signatures between sets.

### 5. Spatial Filtering — Common Average Reference (CAR)
CAR was applied by subtracting the mean signal across all selected channels from each individual channel, reducing spatially correlated noise and sharpening localized cortical activity.

### 6. Epoch Extraction & Baseline Correction
- Window: **−0.5 s to +3.0 s** relative to motor imagery cue onset.
- Baseline correction using the **−0.5 to 0.0 s pre-stimulus interval** to remove DC offsets and slow drifts.

---

## Feature Extraction

### Temporal Focus
A fixed analysis window of **1.0–3.0 seconds post-cue** was used, based on neurophysiological evidence that motor imagery effects emerge ~1 second after cue onset. This reduced per-epoch dimensionality from 481 to 231 samples.

### Power Spectral Density (PSD) — Spectral Features
Welch's method with a 1-second Hamming window and 50% overlap was used to compute spectral power in 1 Hz bins across:
- **Alpha/Mu band (8–12 Hz):** sensorimotor idling rhythm.
- **Beta band (13–30 Hz):** movement-related oscillations.

### Common Spatial Patterns (CSP) — Spatial Features
CSP was applied to bandpass-filtered epochs to learn spatial filters that maximize variance ratio between classes.

| CSP Setting | Value |
|---|---|
| Components | **6** (selected by grid search over [4, 6, 8, 10, 12]) |
| Regularization | Shrinkage estimator on covariance matrices |
| Output features | Log-variance of CSP-filtered signals |

### Feature Integration
CSP and PSD features were concatenated into a **28-dimensional feature vector** per epoch, combining spatial discriminability and spectral characteristics.

---

## Data Augmentation

Six augmentation techniques were applied to the training set only to mitigate overfitting and class imbalance. One augmented copy was generated per original trial by randomly applying one transformation:

| # | Technique | Details |
|---|---|---|
| 1 | **Gaussian Noise Injection** | σ = 5% of signal standard deviation |
| 2 | **Amplitude Scaling** | Random scale factor α ∈ [0.9, 1.1] |
| 3 | **Time Shifting** | Zero-padded shift, max 50 samples |
| 4 | **Time Masking** | 10% of time points zeroed |
| 5 | **Channel Dropout** | 10% of channels zeroed randomly |
| 6 | **Mixup** | Weighted average of two trials |

Augmentation doubled the training set from 1,800 to 3,600 trials (in the binary classification pipeline).

---

## Machine Learning Models

All models were implemented in **scikit-learn (v1.3.0)**. A **5-fold nested cross-validation** strategy was used throughout — inner loop for hyperparameter optimization, outer loop for unbiased performance estimation.

### Linear Discriminant Analysis (LDA)
Projects data onto lower-dimensional spaces while maximizing class separability. **Shrinkage regularization (Ledoit-Wolf lemma)** was applied to stabilize covariance estimation under the small-sample conditions typical of EEG studies. Selected for computational efficiency and proven effectiveness with CSP features.

### Support Vector Machine (SVM)
Two kernel configurations were evaluated:
- **RBF Kernel:** C=1, gamma=0.001 — non-linear mapping for complex EEG relationships.
- **Linear Kernel:** C values searched over [0.01, 0.1, 1, 10, 100] — efficient and strongly generalizable for CSP-transformed features.

### k-Nearest Neighbors (KNN)
Non-parametric instance-based classifier. k optimized via grid search over [3, 5, 7, 9, 11] using Euclidean distance. Optimal: k=7, uniform weights.

### Random Forest
Bootstrap aggregation of decision trees with majority voting. Optimized configuration: **50 trees, max depth 20, min_samples_split=10, min_samples_leaf=4.** Robust to noise with built-in feature importance.

### XGBoost
Sequential gradient boosting ensemble. Optimized configuration: **200 estimators, max depth 3, learning rate 0.01, subsample=1.0, colsample_bytree=0.8.** Early stopping used for regularization.

---

## Hyperparameter Optimization

Grid search identified the following optimal settings:

| Parameter | Search Space | Optimal Value |
|---|---|---|
| CSP components | [4, 6, 8, 10, 12] | **6** |
| Frequency band | mu / mu+low-beta / beta / broadband | **13–30 Hz (beta)** |
| SVM kernel | Linear / RBF | **Linear** for CSP features |
| SVM C | [0.01, 0.1, 1, 10, 100] | **1.0** |
| XGBoost estimators | — | 200 |
| XGBoost max depth | — | 3 |

---

## Evaluation Metrics

- **Accuracy** (primary metric)
- **F1-Score, Precision, Recall**
- **ROC-AUC**
- **Confusion Matrix** (per-class analysis)

---

## Results

### Optimized CSP + SVM Pipeline (BCI Competition IV — Subject-Specific)

| Configuration | Accuracy |
|---|---|
| Baseline (1–40 Hz, default CSP) | 74.5% |
| + Beta band optimization (13–30 Hz) | 85.5% |
| + 6 CSP components | **89.1%** |

Class-level performance at 89.1% accuracy:

| Class | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Left Hand | 91.9% | 0.92 | 0.92 | 0.92 |
| Right Hand | 83.3% | 0.83 | 0.83 | 0.83 |

---

### Cross-Subject Classification — PhysioNet (19 Subjects)

| Model | Test Accuracy | Rank |
|---|---|---|
| LDA (Ledoit-Wolf) | **54.03%** | 1st |
| SVM (RBF, C=1, γ=0.001) | 53.19% | 2nd |
| XGBoost | 52.92% | 3rd |
| Random Forest | 52.08% | 4th |
| KNN (k=7) | 49.86% | 5th |

---

### BCI Competition IV Dataset 2a

| Model | Train Acc. | Test Acc. | F1 | ROC-AUC |
|---|---|---|---|---|
| Random Forest | 100.00% | 61.67% | 0.623 | 0.537 |
| KNN | 66.19% | 46.67% | 0.500 | 0.458 |
| LDA | 63.31% | 45.00% | 0.459 | 0.509 |
| SVM | 59.71% | 40.00% | 0.471 | 0.493 |

> Note: Random Forest's 100% training accuracy indicates significant overfitting.

---

### Subject-Dependent Classification (3 Strategies, 19 Subjects)

| Strategy | Mean Accuracy | Best | Worst |
|---|---|---|---|
| Ensemble (Multi-window voting) | **93.8% ± 7.3%** | 100% | 78.6% |
| Spatial — CSP + LDA | 92.1% | 100% | 75.4% |
| Time-Frequency — PSD + SVC | 90.3% | 100% | 72.8% |

9 out of 19 subjects achieved **100% accuracy** under the ensemble strategy. Median accuracy: 97.3%.

| Performance Tier | Subjects | Accuracy Range |
|---|---|---|
| Elite | 9 | 100% |
| High | 6 | 90–99.9% |
| Moderate | 3 | 85–89.9% |
| Challenging | 1 | 78.6% (Subject 011) |

---

## Key Findings

- **Subject-specific vs. cross-subject gap:** Peak subject-specific accuracy (89.1%) far exceeds the best cross-subject accuracy (54.03%), confirming inter-subject variability as the primary bottleneck.
- **Frequency band is the most impactful hyperparameter:** The 13–30 Hz beta band improved accuracy by 11 percentage points over a broadband configuration.
- **Linear models outperform non-linear ones cross-subject:** LDA and Linear SVM generalize better with limited cross-subject data; tree-based models require careful regularization.
- **Ensemble strategy dominates subject-dependent classification:** Multi-window temporal voting achieves the best balance of accuracy and robustness.

---

## Limitations

- **Binary classification only:** The rest condition (T0) was excluded; real-world BCI systems must handle non-imagery periods.
- **Inter-subject variability:** Cross-subject accuracy is substantially below subject-specific performance.
- **Computational constraints:** Deep learning models showed limited accuracy gains over well-tuned classical pipelines, with significantly higher resource demands.

---

## Appendix — ECoG Preliminary Analysis

An exploratory ECoG analysis was conducted as a proof-of-concept prior to the main EEG pipeline. The dataset comprised **22 cortical electrodes at 200 Hz**, with a single subject performing **6 imagined movement conditions** (left hand, right hand, left leg, right leg, tongue, passive). A total of 952 epochs (3-second windows) were analyzed.

PSD features across 6 frequency bands (delta, theta, alpha, beta, gamma, high-gamma) per channel yielded a 132-dimensional feature vector. Multiclass logistic regression with 5-fold cross-validation achieved **40.2% accuracy** against a 16.7% chance baseline, confirming statistically significant class-discriminative content in spectral features.

Class-specific neurophysiological signatures:
- **Right-leg imagery:** theta-band activity in channels 9 and 10.
- **Tongue imagery:** gamma-band activity in channels 13 and 14.
- **Passive/neutral:** elevated delta and alpha power.

This analysis directly informed the frequency band selection strategy and feature engineering design of the primary EEG pipeline.

---

## References

- Goldberger et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. *Circulation*, 101(23).
- Blankertz et al. (2008). Optimizing spatial filters for robust EEG single-trial analysis. *IEEE Signal Processing Magazine*.
- Lotte et al. (2018). A review of classification algorithms for EEG-based BCI. *Journal of Neural Engineering*.
- BCI Competition IV Dataset 2a — Graz University of Technology.
- BNCI Horizon 2020 Repository — Zenodo.

---
