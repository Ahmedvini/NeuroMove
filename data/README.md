# Data

Datasets are **not tracked by Git** due to file size. This file defines the expected layout and provides exact download instructions to reproduce the full dataset pipeline.

---

## Expected Layout

```
data/
├── raw/
│   ├── bciciv2a/       # BCI Competition IV Dataset 2a (.gdf)
│   ├── bciciv2b/       # BCI Competition IV Dataset 2b (.gdf)
│   ├── eegphysionet/   # PhysioNet EEGMMIDB (.edf)
│   ├── halt/           # HALT dataset — primary dataset (.mat)
│   └── shin2016/       # Simultaneous EEG–fNIRS (Shin et al., 2016)
├── interim/            # Windowed, filtered, partially processed signals
├── processed/          # Final model-ready .npy arrays per subject
└── external/           # Reference files, electrode maps, montages
```

---

## Dataset 1 — EEGMMIDB (PhysioNet EEG Motor/Mental Imagery Database)

| Field | Details |
|---|---|
| **Full name** | EEG Motor Movement/Imagery Dataset |
| **Source** | https://physionet.org/content/eegmmidb/1.0.0/ |
| **DOI** | 10.13026/C28G6P |
| **Format** | `.edf` (European Data Format) |
| **Subjects** | 109 |
| **Channels** | 64 EEG electrodes |
| **Sampling rate** | 160 Hz |
| **Tasks** | Motor execution and motor imagery: left fist, right fist, both fists, both feet |
| **Trials per subject** | 14 runs (~20 min per subject) |
| **Usage in project** | Architecture comparison and cross-subject generalization benchmarking |
| **Place in** | `data/raw/eegphysionet/` |
| **License** | PhysioNet Credentialed Health Data License |

### Download

```bash
bash scripts/download_data.sh eegphysionet
```

Or manually:
```bash
wget -r -N -c -np -P data/raw/eegphysionet/ \
  https://physionet.org/files/eegmmidb/1.0.0/
```

### Citation
> Goldberger, A. L., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. *Circulation*, 101(23), e215–e220. https://doi.org/10.13026/C28G6P

---

## Dataset 2 — BCI Competition IV Dataset 2a

| Field | Details |
|---|---|
| **Full name** | BCI Competition IV Dataset 2a |
| **Source** | https://www.bbci.de/competition/iv/ |
| **Format** | `.gdf` (GDF EEG format) |
| **Subjects** | 9 (A01–A09) |
| **Channels** | 22 EEG + 3 EOG |
| **Sampling rate** | 250 Hz |
| **Tasks** | 4-class motor imagery: left hand, right hand, feet, tongue |
| **Trials per subject** | 288 training + 288 evaluation |
| **Usage in project** | Standard 4-class benchmark for architecture comparison across 9 deep learning models |
| **Place in** | `data/raw/bciciv2a/` |
| **License** | Free for research use; citation required |

### Files Needed

| Split | Files |
|---|---|
| Training | `A01T.gdf` → `A09T.gdf` |
| Evaluation | `A01E.gdf` → `A09E.gdf` |
| Labels | `A01T.mat` → `A09T.mat` (true labels for evaluation set) |

### Download

Manual download — registration required:
1. Go to https://www.bbci.de/competition/iv/
2. Navigate to **Dataset 2a**
3. Download all training and evaluation `.gdf` files + label `.mat` files
4. Place in `data/raw/bciciv2a/`

### Citation
> Brunner, C., Leeb, R., Müller-Putz, G., Schlögl, A., & Pfurtscheller, G. (2008). BCI Competition 2008 – Graz data sets A and B. *Technical Report*, Institute for Knowledge Discovery, TU Graz.

---

## Dataset 3 — BCI Competition IV Dataset 2b

| Field | Details |
|---|---|
| **Full name** | BCI Competition IV Dataset 2b |
| **Source** | https://www.bbci.de/competition/iv/ |
| **Format** | `.gdf` (GDF EEG format) |
| **Subjects** | 9 (B01–B09) |
| **Channels** | 3 EEG (C3, Cz, C4) + 3 EOG |
| **Sampling rate** | 250 Hz |
| **Tasks** | 2-class motor imagery: left hand vs. right hand |
| **Sessions per subject** | 5 sessions (3 training + 2 evaluation) |
| **Usage in project** | Sparse-channel evaluation; tests electrode reduction methods under minimal-electrode constraints |
| **Place in** | `data/raw/bciciv2b/` |
| **License** | Free for research use; citation required |

### Files Needed

| Subject | Training | Evaluation |
|---|---|---|
| B01 | `B0101T.gdf`, `B0102T.gdf`, `B0103T.gdf` | `B0104E.gdf`, `B0105E.gdf` |
| B02–B09 | Same pattern | Same pattern |

### Download

Manual download — registration required:
1. Go to https://www.bbci.de/competition/iv/
2. Navigate to **Dataset 2b**
3. Download all session `.gdf` files for all 9 subjects
4. Place in `data/raw/bciciv2b/`

### Citation
> Leeb, R., Brunner, C., Müller-Putz, G., Schlögl, A., & Pfurtscheller, G. (2008). BCI Competition 2008 – Graz data set B. *Technical Report*, Institute for Knowledge Discovery, TU Graz.

---

## Dataset 4 — HALT Dataset ⭐ Primary Dataset

| Field | Details |
|---|---|
| **Full name** | A Large Electroencephalographic Motor Imagery Dataset for EEG Brain-Computer Interfaces |
| **Source** | https://www.nature.com/articles/sdata2018211 |
| **DOI** | 10.1038/sdata.2018.211 |
| **Data repository** | Figshare (linked from paper above) |
| **Format** | `.mat` (MATLAB struct variable `o`) |
| **Subjects** | 13 (Subject A – Subject M; 8 male, 5 female, ages 20–35) |
| **Channels** | 19 EEG (10–20 system) + 2 ground (A1, A2) + 1 sync (X3) = 22 total recorded |
| **Sampling rate** | 200 Hz standard; 1000 Hz for select 5F sessions |
| **Total recording** | 60 hours, 75 sessions, >60,000 motor imagery trials |
| **Place in** | `data/raw/halt/` |
| **License** | Open access — Creative Commons |

### Electrode Positions (19 EEG channels)

```
Fp1  Fp2
F7   F3   Fz   F4   F8
T3   C3   Cz   C4   T4
T5   P3   Pz   P4   T6
     O1        O2
```

### Paradigms Available

| Paradigm | Classes | Description |
|---|---|---|
| **CLA** | 3 | Left hand / Right hand / Passive |
| **HaLT** ← *used* | 6 | Left hand / Right hand / Left leg / Right leg / Tongue / Passive |
| **5F** | 5 | Individual finger movements (thumb → pinkie) |
| **FreeForm** | 2 | Self-paced voluntary left/right key press |

### Marker Codes (HaLT paradigm)

| Code | Mental Imagery |
|---|---|
| 1 | Left hand |
| 2 | Right hand |
| 3 | Passive / Neutral |
| 4 | Left leg |
| 5 | Tongue |
| 6 | Right leg |
| 91 | Inter-session break |
| 99 | Initial relaxation period |

### MATLAB Data Structure

Each `.mat` file contains struct `o` with:

```matlab
o.data      % [nS × 22] — EEG voltage in µV; columns = channels; col 22 = sync only
o.marker    % [nS × 1]  — integer codes (see table above)
o.sampFreq  % scalar    — sampling frequency in Hz
o.nS        % scalar    — total number of EEG samples
```

### File Naming Convention

```
{Paradigm}-Subject{X}-{YYMMDD}-{N}St-{Mnemonic}.mat
```

Example:
```
HaLT-SubjectJ-161121-6St-LRHandLegTongue.mat
```

### Usage in This Project

- **Task:** Binary classification — **right hand (class 2) vs. left leg (class 6)**
- **Evaluation:** Leave-One-Subject-Out (LOSO) cross-validation across all 13 subjects
- **Electrode reduction experiments:** All three methods (ablation-based, weight-magnitude ranking, Gumbel-softmax learned selection) run on this dataset
- **Key result:** Gumbel-softmax selection identifies **Cz, P3, T5, F7, F8** (5 channels) achieving **~89.94% accuracy** vs. **~91.92%** with all 19 channels — only 1.98% drop with 73.7% channel reduction

### Download

1. Go to https://www.nature.com/articles/sdata2018211
2. Click **Data Citation 1** → redirects to Figshare repository
3. Download all `HaLT-Subject*.mat` files
4. Place in `data/raw/halt/`

### Citation
> Kaya, M., Binli, M. K., Ozbay, E., Yanar, H., & Mishchenko, Y. (2018). A large electroencephalographic motor imagery dataset for electroencephalographic brain computer interfaces. *Scientific Data*, 5, 180211. https://doi.org/10.1038/sdata.2018.211

---

## Dataset 5 — Simultaneous EEG–fNIRS Motor Imagery (Shin et al., 2016)

| Field | Details |
|---|---|
| **Full name** | Open Access Dataset for EEG+NIRS Single-Trial Classification |
| **Source** | https://doi.org/10.1109/TNSRE.2016.2628057 |
| **Data repository** | http://doc.ml.tu-berlin.de/simultaneous_EEG_NIRS/ |
| **Format** | `.mat` (EEG) + `.oxy3` / `.nirs` (fNIRS) |
| **Subjects** | 26 |
| **EEG channels** | 30 (motor cortex coverage) |
| **fNIRS channels** | 30 (same motor cortex regions) |
| **Sampling rate** | EEG: 200 Hz; fNIRS: 10.4 Hz |
| **Tasks** | 2-class motor imagery: left hand vs. right hand |
| **Trials per subject** | 60 per class (120 total) |
| **Usage in project** | Multimodal neurovascular coupling analysis; validates motor cortex electrode relevance for cross-modal signal generation |
| **Place in** | `data/raw/shin2016/` |
| **License** | Creative Commons Attribution 4.0 |

### Download

Manually from: http://doc.ml.tu-berlin.de/simultaneous_EEG_NIRS/

### Citation
> Shin, J., von Lühmann, A., Blankertz, B., Kim, D. W., Jeong, J., Hwang, H. J., & Müller, K. R. (2017). Open access dataset for EEG+NIRS single-trial classification. *IEEE Transactions on Neural Systems and Rehabilitation Engineering*, 25(10), 1735–1745. https://doi.org/10.1109/TNSRE.2016.2628057

---

## Preprocessing Pipeline

Raw → Interim → Processed is fully reproducible. **Never edit files in `raw/` directly.**

```bash
python scripts/preprocess.py --dataset bciciv2a
python scripts/preprocess.py --dataset bciciv2b
python scripts/preprocess.py --dataset eegphysionet
python scripts/preprocess.py --dataset halt
python scripts/preprocess.py --dataset shin2016
```

### Pipeline Stages

| Stage | Folder | Description |
|---|---|---|
| Raw | `data/raw/` | Original files, never modified |
| Interim | `data/interim/` | Bandpass filtered (0.5–40 Hz), epoched (−0.5 to +4 s), artifact-rejected |
| Processed | `data/processed/` | Z-score normalized `.npy` arrays, model-ready |

### Preprocessing Parameters (HALT — primary dataset)

| Parameter | Value |
|---|---|
| Bandpass filter | 0.5–40 Hz (4th-order Butterworth) |
| Epoch window | 0 to +4 s post-stimulus onset |
| Baseline correction | −0.5 to 0 s pre-stimulus |
| Artifact rejection | Amplitude threshold ±100 µV |
| Classes used | Right hand (marker=2) vs. Left leg (marker=6) |
| Evaluation | LOSO cross-validation (13 folds) |

---

## Processed File Naming Convention

All processed `.npy` arrays in `data/processed/` follow:

```
S{subject_id}_{dataset}_{split}.npy
```

### Examples

```
S01_halt_train.npy         # Subject 1, HALT dataset, training split
S01_halt_test.npy          # Subject 1, HALT dataset, test split (LOSO held-out)
S03_bciciv2a_train.npy     # Subject 3, BCI IV 2a, training split
S05_eegphysionet_train.npy
```

### Array Shape

```python
X.npy  # shape: (n_trials, n_channels, n_timepoints)
y.npy  # shape: (n_trials,) — integer class labels
```

---

## EEG Signal Frequency Bands Reference

| Band | Frequency | Relevance to Motor Imagery |
|---|---|---|
| Delta | 0.5–4 Hz | Deep sleep baseline; not used |
| Theta | 4–8 Hz | Drowsiness, working memory |
| **Alpha / Mu** | **8–13 Hz** | **ERD during motor imagery — primary MI signal** |
| **Beta** | **13–30 Hz** | **ERS post-movement; MRCP detection** |
| Gamma | >30 Hz | High-level processing; artifact-prone |

> Motor imagery produces **Event-Related Desynchronization (ERD)** in the mu (8–12 Hz) and beta (13–30 Hz) bands, contralateral to the imagined movement. This is the neurophysiological basis for all classification in this project.

---

## Electrode Coverage — 10–20 System

| Region | Electrodes | Motor Cortex Relevance |
|---|---|---|
| Frontal | Fp1, Fp2, F3, F4, F7, F8, Fz | Motor planning, premotor cortex (PMC), supplementary motor area (SMA) |
| **Central** | **C3, C4, Cz** | **Primary motor cortex (M1) — strongest MI signal; contralateral ERD** |
| Parietal | P3, P4, Pz | Somatosensory cortex; spatial processing |
| Temporal | T3, T4, T5, T6 | Supplementary motor coordination; secondary somatosensory |
| Occipital | O1, O2 | Visual imagery; not primary for MI |

### Project Finding — Optimal Electrode Cluster

> Gumbel-softmax learned selection on the HALT dataset identified **Cz, P3, T5, F7, F8** as the optimal 5-electrode subset for right-hand vs. left-leg binary classification — outperforming the conventional C3/C4 pair used in standard motor imagery BCI. This suggests that parietal (somatosensory integration) and temporal (supplementary motor) regions carry discriminative information beyond the primary motor cortex alone for lower-limb vs. upper-limb imagery.

---

## Dataset Summary Table

| Dataset | Subjects | Channels | Classes | Sampling Rate | Format | Usage |
|---|---|---|---|---|---|---|
| EEGMMIDB | 109 | 64 EEG | 4 | 160 Hz | `.edf` | Architecture benchmarking |
| BCI IV 2a | 9 | 22 EEG + 3 EOG | 4 | 250 Hz | `.gdf` | 4-class benchmark |
| BCI IV 2b | 9 | 3 EEG + 3 EOG | 2 | 250 Hz | `.gdf` | Sparse-channel evaluation |
| **HALT** | **13** | **19 EEG** | **6 (2 used)** | **200 Hz** | **`.mat`** | **Primary — all experiments** |
| Shin et al. | 26 | 30 EEG + 30 fNIRS | 2 | 200 / 10.4 Hz | `.mat` + `.nirs` | Multimodal coupling analysis |

