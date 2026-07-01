# Secure EEG-Based Person Identification and Authentication Systems

A deep learning framework for biometric person identification and authentication using EEG (Electroencephalogram) signals. This project implements two complementary systems using the [PhysioNet EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/).

## Overview

| System | Approach | Accuracy | Description |
|--------|----------|----------|-------------|
| **Identification** | 1D CNN (109-class classifier) | 96.7% | Classifies which person an EEG signal belongs to |
| **Authentication** | CNN + Cosine Similarity | EER ~4.26% | Verifies whether an EEG signal matches a claimed identity |
| **Authentication** | Siamese Network (Triplet Loss) | EER ~3.8% | Learns an embedding space for open-set verification |

## Project Structure

```
.
├── identification/                  # Person Identification (closed-set)
│   ├── CNN_local.py                 # Training script (1D CNN, 64-channel)
│   ├── CNN_Person_Identification_local_notebook.ipynb
│   └── gui_pro.py                   # Real-time identification GUI
│
├── authentication/                  # Person Authentication (open-set)
│   ├── eeg_auth_gui.py              # Authentication GUI application
│   ├── eeg_authentication_cnn.ipynb           # CNN-based authentication
│   ├── eeg_authentication_siamese_network.ipynb  # Siamese network approach
│   └── GUI_README.md                # GUI usage guide
│
├── requirements.txt
└── README.md
```

## Dataset

This project uses the **PhysioNet EEG Motor Movement/Imagery Dataset**:
- **109 subjects**, each with 14 experimental runs
- **64 EEG channels** at 160 Hz sampling rate
- Download from: https://physionet.org/content/eegmmidb/1.0.0/

Place the dataset in a `data/files/` directory:
```
data/files/
├── S001/
│   ├── S001R01.edf
│   ├── S001R02.edf
│   └── ...
├── S002/
│   └── ...
└── S109/
    └── ...
```

Or set the `EEG_DATASET_PATH` environment variable to your dataset location.

## Installation

```bash
git clone https://github.com/safii-74/-Secure-EEG-Based-Person-Identification-and-Authentication-Systems.git
cd -Secure-EEG-Based-Person-Identification-and-Authentication-Systems
pip install -r requirements.txt
```

### Requirements
- Python 3.9+
- TensorFlow 2.15+
- MNE-Python
- scikit-learn
- NumPy, SciPy, Matplotlib

## Usage

### 1. Person Identification (Training)

```bash
# Set dataset path
export EEG_DATASET_PATH="/path/to/dataset/files"

# Run training script
python identification/CNN_local.py
```

Or use the Jupyter notebook `identification/CNN_Person_Identification_local_notebook.ipynb` for step-by-step training.

### 2. Person Authentication

Open and run either notebook:
- `authentication/eeg_authentication_cnn.ipynb` - CNN with fingerprint layer
- `authentication/eeg_authentication_siamese_network.ipynb` - Siamese network with triplet loss

### 3. GUI Applications

**Identification GUI** (real-time monitoring):
```bash
python identification/gui_pro.py
```

**Authentication GUI** (enroll & verify):
```bash
python authentication/eeg_auth_gui.py
```

See [authentication/GUI_README.md](authentication/GUI_README.md) for detailed GUI usage instructions.

## Methodology

### Identification Pipeline
1. Load 64-channel EEG from EDF files
2. Segment into 1-second windows (160 samples)
3. Train a 4-layer 1D CNN classifier (109 output classes)
4. Evaluate on held-out recordings (R13/R14)

### Authentication Pipeline
1. Select 3 target channels (Oz, T7, Cz)
2. Apply **Gram-Schmidt orthogonalization** to decorrelate channels
3. Normalize with MinMaxScaler
4. Segment with sliding window (160 samples, stride 4)
5. Train CNN with **fingerprint layer** (128-dim embedding)
6. Authentication via **cosine similarity** against enrolled templates
7. Evaluate with ROC curve and Equal Error Rate (EER)

### Siamese Network Variant
- Uses the same preprocessing pipeline
- Trains with **triplet loss** (anchor, positive, negative)
- Learns a metric space where same-person embeddings cluster together

## Results

### Identification
- **Test Accuracy**: 96.7%
- **FAR**: 3.2% | **FRR**: 3.4% | **EER**: 3.3%

### Authentication (CNN)
- **AUC**: 0.96+ | **EER**: ~4.26%
- **Optimal Threshold**: 0.8148

### Authentication (Siamese)
- **AUC**: 0.98+ | **EER**: ~3.8%

## License

This project is for academic and research purposes.

## Acknowledgments

- [PhysioNet](https://physionet.org/) for the EEG Motor Movement/Imagery Dataset
- Goldberger, A., et al. "PhysioBank, PhysioToolkit, and PhysioNet." *Circulation*, 2000.
